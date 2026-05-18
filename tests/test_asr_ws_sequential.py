# coding=utf-8
"""
Sequential WebSocket streaming test for Qwen3-ASR.

wav/sequential/001 ~ 010 까지의 WAV 파일을 한 개씩 순차적으로 WebSocket 으로
스트리밍하여 실시간 ASR 성능을 측정한다.

각 파일별로:
  1. 16kHz로 리샘플
  2. STEP_MS(=200ms) 단위 청크를 실제 오디오 시간 흐름에 맞춰 페이싱 송신
  3. 서버가 보내는 `ready` / `result(partial)` / `result(final)` / `closed` 수집

수집 지표:
  - 첫 partial 도착 시각 (TTFT, time to first text)
  - end 송신 → FINAL 도착 지연 (end→final)
  - partial 청크별 lag = wall_clock - audio_pos  (음수일수록 실시간 대비 빠름)
  - 평균 RTF = Σ wall_clock / Σ audio_duration
  - 최종 인식 텍스트 / 언어

실행:
  python tests/test_asr_ws_sequential.py
"""

import asyncio
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import soundfile as sf
import websockets


SERVER = "ws://172.31.79.202:30000/api/ws"
WAV_DIR = "wav/sequential"
STEP_MS = 200
LANGUAGE = "Korean"
INDEX_RANGE = range(1, 11)  # 001 ~ 010


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _load_16k(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if sr == 16000:
        return wav
    dur = wav.shape[0] / float(sr)
    n = int(round(dur * 16000))
    x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, dur, num=n, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


@dataclass
class FileResult:
    name: str
    duration_s: float
    wall_clock_s: float = 0.0
    first_partial_s: Optional[float] = None
    final_s: Optional[float] = None
    partial_count: int = 0
    final_text: str = ""
    final_language: str = ""
    chunk_lags: List[float] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def end_to_final_ms(self) -> float:
        if self.final_s is None:
            return float("nan")
        return (self.final_s - self.duration_s) * 1000.0


# --------------------------------------------------------------------------- #
# Per-file run                                                                 #
# --------------------------------------------------------------------------- #

async def run_one(path: str) -> FileResult:
    wav = _load_16k(path)
    dur = wav.shape[0] / 16000.0
    name = os.path.basename(path)
    result = FileResult(name=name, duration_s=dur)

    try:
        async with websockets.connect(SERVER, max_size=None) as ws:
            await ws.send(json.dumps({"type": "config", "language": LANGUAGE}))
            ready_raw = await asyncio.wait_for(ws.recv(), timeout=10)
            ready = json.loads(ready_raw)
            if ready.get("type") != "ready":
                result.error = f"unexpected first event: {ready}"
                return result
            chunk_sec = float(ready["chunk_size_sec"])

            t0 = time.perf_counter()
            step_samples = int(STEP_MS / 1000.0 * 16000)

            async def sender():
                pos = 0
                while pos < wav.shape[0]:
                    target_t = pos / 16000.0
                    wait = target_t - (time.perf_counter() - t0)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    seg = wav[pos : pos + step_samples]
                    await ws.send(seg.tobytes())
                    pos += seg.shape[0]
                wait_end = dur - (time.perf_counter() - t0)
                if wait_end > 0:
                    await asyncio.sleep(wait_end)
                await ws.send(json.dumps({"type": "end"}))

            async def receiver():
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        result.error = "recv timeout"
                        return
                    except websockets.ConnectionClosed:
                        return
                    evt = json.loads(raw)
                    et = evt.get("type")
                    t = time.perf_counter() - t0
                    if et == "result":
                        cid = int(evt.get("chunk_id", 0))
                        if evt.get("is_final"):
                            result.final_s = t
                            result.final_text = evt.get("text", "")
                            result.final_language = evt.get("language", "")
                        else:
                            result.partial_count += 1
                            audio_pos = cid * chunk_sec
                            result.chunk_lags.append(t - audio_pos)
                            if result.first_partial_s is None and evt.get("text"):
                                result.first_partial_s = t
                    elif et == "error":
                        result.error = f"{evt.get('code')}: {evt.get('message')}"
                        return
                    elif et == "closed":
                        return

            await asyncio.gather(sender(), receiver())
            result.wall_clock_s = time.perf_counter() - t0
            return result
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        return result


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def _collect_files() -> List[str]:
    files: List[str] = []
    for idx in INDEX_RANGE:
        hits = sorted(glob.glob(os.path.join(WAV_DIR, f"{idx:03d}_*.wav")))
        if hits:
            files.append(hits[0])
        else:
            print(f"⚠ {idx:03d}_*.wav 파일 없음", file=sys.stderr)
    return files


async def main() -> None:
    files = _collect_files()
    if not files:
        print(f"❌ {WAV_DIR}/ 에서 대상 파일을 찾지 못함", file=sys.stderr)
        sys.exit(1)

    print(f"=== Sequential WS test: {len(files)} files via {SERVER} ===\n")

    results: List[FileResult] = []
    for i, path in enumerate(files, 1):
        name = os.path.basename(path)
        print(f"[{i:>2}/{len(files)}] {name}", flush=True)
        r = await run_one(path)
        results.append(r)
        if r.error:
            print(f"        ✗ ERROR: {r.error}")
        else:
            tttp = r.first_partial_s if r.first_partial_s is not None else float("nan")
            print(f"        dur={r.duration_s:.2f}s  first_partial={tttp:.3f}s  "
                  f"final={r.final_s:.3f}s  end→final={r.end_to_final_ms:+.0f}ms  "
                  f"partials={r.partial_count}")
            print(f"        text: '{r.final_text}'")
        print()

    # ----- Summary table -----
    print("=" * 110)
    header = (f"{'#':<3}{'file':<32}{'dur(s)':>8}{'1st_p(s)':>10}"
              f"{'final(s)':>10}{'end→final(ms)':>15}{'partials':>10}{'rtf':>8}")
    print(header)
    print("-" * 110)
    for i, r in enumerate(results, 1):
        if r.error:
            print(f"{i:<3}{r.name[:30]:<32}  ERROR: {r.error}")
            continue
        rtf = (r.wall_clock_s / r.duration_s) if r.duration_s > 0 else float("nan")
        print(f"{i:<3}{r.name[:30]:<32}"
              f"{r.duration_s:>8.2f}"
              f"{(r.first_partial_s or 0):>10.3f}"
              f"{(r.final_s or 0):>10.3f}"
              f"{r.end_to_final_ms:>+14.0f}"
              f"{r.partial_count:>10}"
              f"{rtf:>8.3f}")
    print("-" * 110)

    ok = [r for r in results if not r.error]
    if ok:
        firsts = [r.first_partial_s for r in ok if r.first_partial_s is not None]
        end_finals = [r.end_to_final_ms for r in ok if r.final_s is not None]
        all_lags_ms = [lag * 1000.0 for r in ok for lag in r.chunk_lags]
        total_dur = sum(r.duration_s for r in ok)
        total_wall = sum(r.wall_clock_s for r in ok)

        print(f"\n=== 종합 ({len(ok)}/{len(results)} 성공) ===")
        if firsts:
            print(f"  첫 partial 도착 (s)       "
                  f"avg={np.mean(firsts):.3f}  min={min(firsts):.3f}  max={max(firsts):.3f}")
        if end_finals:
            print(f"  end→FINAL 지연 (ms)       "
                  f"avg={np.mean(end_finals):+.0f}  min={min(end_finals):+.0f}  max={max(end_finals):+.0f}")
        if all_lags_ms:
            print(f"  partial 청크 lag (ms)     "
                  f"avg={np.mean(all_lags_ms):+.0f}  min={min(all_lags_ms):+.0f}  max={max(all_lags_ms):+.0f}  "
                  f"(음수 = 실시간 대비 앞섬)")
        print(f"  총 오디오 길이             {total_dur:.2f}s")
        print(f"  총 실측 wall-clock         {total_wall:.2f}s")
        print(f"  평균 RTF                   {total_wall / total_dur:.3f}  (1.0 = 실시간)")
    failed = [r for r in results if r.error]
    if failed:
        print(f"\n=== 실패 {len(failed)}건 ===")
        for r in failed:
            print(f"  {r.name}: {r.error}")


if __name__ == "__main__":
    asyncio.run(main())
