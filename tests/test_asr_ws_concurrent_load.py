# coding=utf-8
"""
Concurrent WebSocket load test for Qwen3-ASR.

`wav/sequential/` 의 정렬된 첫 N개 음원을 동시에 실시간 페이싱으로 WS 스트리밍하고,
N ∈ {1, 10, 30, 50, 80, 100} 단계로 부하를 올려가며 세 가지 핵심 지표를 측정한다.

측정 지표:
  - TTFT      : 세션 시작(t0) → 첫 non-empty partial 도착까지 (ms)
  - end→final : `{"type":"end"}` 송신 → final result 수신까지 (ms)
  - RTF       :
      * per-session  : wall_clock / audio_duration (1.0 = 실시간 유지)
      * throughput   : Σ audio_duration / level_wall_clock (×배율)
  - 보조: partial chunk lag (실시간 페이싱 유지 여부 검증용)

실행:
  python tests/test_asr_ws_concurrent_load.py
"""

import asyncio
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import websockets


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

SERVER = "ws://172.31.79.202:30000/api/ws"
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
WAV_DIR = os.path.join(_PROJECT_ROOT, "wav", "sequential")

STEP_MS = 200
LANGUAGE = "Korean"
CONCURRENCY_LEVELS: List[int] = [1, 10, 30, 50, 80, 100]
LEVEL_SETTLE_SEC = 4.0
WARMUP_ROUNDS = 3  # cold-start 회피용. 서버 idle 직후 첫 부하 배치가 5초+ 걸리는 케이스 방지.


# --------------------------------------------------------------------------- #
# Audio helpers                                                                #
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


def _collect_files(max_n: int) -> List[str]:
    paths = sorted(glob.glob(os.path.join(WAV_DIR, "*.wav")))
    if not paths:
        raise FileNotFoundError(f"no wav files in {WAV_DIR}")
    return paths[:max_n]


# --------------------------------------------------------------------------- #
# Session metric container                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class SessionResult:
    name: str
    duration_s: float
    t0: float = 0.0
    first_partial_s: Optional[float] = None
    end_sent_at: Optional[float] = None
    final_recv_at: Optional[float] = None
    partial_count: int = 0
    final_text: str = ""
    final_language: str = ""
    chunk_lags: List[float] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def wall_clock_s(self) -> float:
        if self.final_recv_at is None:
            return float("nan")
        return self.final_recv_at - self.t0

    @property
    def end_to_final_ms(self) -> float:
        if self.final_recv_at is None or self.end_sent_at is None:
            return float("nan")
        return (self.final_recv_at - self.end_sent_at) * 1000.0

    @property
    def rtf_session(self) -> float:
        if self.duration_s <= 0:
            return float("nan")
        return self.wall_clock_s / self.duration_s


# --------------------------------------------------------------------------- #
# Per-session WS worker (real-time paced)                                      #
# --------------------------------------------------------------------------- #

async def run_session(name: str, wav: np.ndarray) -> SessionResult:
    dur = wav.shape[0] / 16000.0
    result = SessionResult(name=name, duration_s=dur)

    try:
        async with websockets.connect(
            SERVER, max_size=None, ping_interval=None, open_timeout=15
        ) as ws:
            await ws.send(json.dumps({"type": "config", "language": LANGUAGE}))
            ready_raw = await asyncio.wait_for(ws.recv(), timeout=10)
            ready = json.loads(ready_raw)
            if ready.get("type") != "ready":
                result.error = f"unexpected first event: {ready}"
                return result
            chunk_sec = float(ready["chunk_size_sec"])

            t0 = time.perf_counter()
            result.t0 = t0
            step_samples = int(STEP_MS / 1000.0 * 16000)

            async def sender() -> None:
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
                result.end_sent_at = time.perf_counter()
                await ws.send(json.dumps({"type": "end"}))

            async def receiver() -> None:
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=60)
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
                        text = (evt.get("text") or "").replace("�", "").strip()
                        if evt.get("is_final"):
                            result.final_recv_at = time.perf_counter()
                            result.final_text = text
                            result.final_language = evt.get("language", "")
                        else:
                            result.partial_count += 1
                            audio_pos = cid * chunk_sec
                            result.chunk_lags.append(t - audio_pos)
                            if result.first_partial_s is None and text:
                                result.first_partial_s = t
                    elif et == "error":
                        result.error = f"{evt.get('code')}: {evt.get('message')}"
                        return
                    elif et == "closed":
                        return

            await asyncio.gather(sender(), receiver())
            return result
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        return result


# --------------------------------------------------------------------------- #
# Stats helpers                                                                #
# --------------------------------------------------------------------------- #

def _pct(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, q))


def _fmt_ms(ms: float) -> str:
    return "  n/a" if math.isnan(ms) else f"{ms:>5.0f}"


def _fmt_f3(x: float) -> str:
    return "  n/a" if math.isnan(x) else f"{x:>5.3f}"


@dataclass
class LevelSummary:
    n: int
    ok: int
    fail: int
    total_audio_s: float
    level_wall_s: float
    ttft_ms: List[float]
    e2f_ms: List[float]
    rtf_session: List[float]
    mean_chunk_lag_ms: float


def summarize(level: int, results: List[SessionResult], level_wall_s: float) -> LevelSummary:
    ok = [r for r in results if r.error is None and r.final_recv_at is not None]
    fail = [r for r in results if not (r.error is None and r.final_recv_at is not None)]

    ttft = [r.first_partial_s * 1000.0 for r in ok if r.first_partial_s is not None]
    e2f = [r.end_to_final_ms for r in ok if not math.isnan(r.end_to_final_ms)]
    rtf_s = [r.rtf_session for r in ok if not math.isnan(r.rtf_session)]
    lag_ms = [lag * 1000.0 for r in ok for lag in r.chunk_lags]
    total_audio = sum(r.duration_s for r in ok)

    return LevelSummary(
        n=level,
        ok=len(ok),
        fail=len(fail),
        total_audio_s=total_audio,
        level_wall_s=level_wall_s,
        ttft_ms=ttft,
        e2f_ms=e2f,
        rtf_session=rtf_s,
        mean_chunk_lag_ms=float(np.mean(lag_ms)) if lag_ms else float("nan"),
    )


def print_level_block(s: LevelSummary, failed: List[SessionResult]) -> None:
    print(f"\n--- level {s.n} sess: ok={s.ok}/{s.n}  wall={s.level_wall_s:.2f}s  "
          f"audio_sum={s.total_audio_s:.1f}s  "
          f"thru_RTF={s.total_audio_s / s.level_wall_s:.1f}×  "
          f"mean_lag={s.mean_chunk_lag_ms:+.0f}ms ---")

    def line(label: str, vals: List[float], unit: str) -> None:
        if not vals:
            print(f"  {label:<14} (no data)")
            return
        print(f"  {label:<14} min={min(vals):>5.0f}  avg={np.mean(vals):>5.0f}  "
              f"p50={_pct(vals, 50):>5.0f}  p95={_pct(vals, 95):>5.0f}  "
              f"max={max(vals):>5.0f}  {unit}")

    line("TTFT",        s.ttft_ms, "ms")
    line("end→final",   s.e2f_ms,  "ms")
    if s.rtf_session:
        vals = s.rtf_session
        print(f"  {'RTF/session':<14} min={min(vals):>5.3f}  avg={np.mean(vals):>5.3f}  "
              f"p50={_pct(vals, 50):>5.3f}  p95={_pct(vals, 95):>5.3f}  "
              f"max={max(vals):>5.3f}  (1.0=실시간)")

    if failed:
        print(f"  failed {len(failed)}:")
        for r in failed[:5]:
            print(f"    {r.name}: {r.error}")
        if len(failed) > 5:
            print(f"    ... (+{len(failed) - 5} more)")


def print_compare_table(summaries: List[LevelSummary]) -> None:
    print("\n" + "=" * 116)
    print("=== 레벨별 비교 ===")
    print("-" * 116)
    header = (f"{'sess':>4} {'audio_s':>8} {'wall_s':>8} {'thru_RTF':>10} "
              f"{'TTFT_avg':>9} {'TTFT_p95':>9} {'e→f_avg':>9} {'e→f_p95':>9} "
              f"{'RTFs_p95':>9} {'lag_avg':>9} {'ok/fail':>9}")
    print(header)
    print("-" * 116)
    for s in summaries:
        thru = s.total_audio_s / s.level_wall_s if s.level_wall_s > 0 else float("nan")
        ttft_avg = float(np.mean(s.ttft_ms)) if s.ttft_ms else float("nan")
        ttft_p95 = _pct(s.ttft_ms, 95)
        e2f_avg = float(np.mean(s.e2f_ms)) if s.e2f_ms else float("nan")
        e2f_p95 = _pct(s.e2f_ms, 95)
        rtf_p95 = _pct(s.rtf_session, 95)
        print(f"{s.n:>4} {s.total_audio_s:>8.1f} {s.level_wall_s:>8.2f} "
              f"{thru:>9.1f}× "
              f"{_fmt_ms(ttft_avg)}ms {_fmt_ms(ttft_p95)}ms "
              f"{_fmt_ms(e2f_avg)}ms {_fmt_ms(e2f_p95)}ms "
              f"{_fmt_f3(rtf_p95)}   "
              f"{s.mean_chunk_lag_ms:>+6.0f}ms "
              f"{s.ok:>4}/{s.fail:<4}")
    print("=" * 116)


# --------------------------------------------------------------------------- #
# Level runner                                                                 #
# --------------------------------------------------------------------------- #

async def run_level(level: int, files: List[Tuple[str, np.ndarray]]) -> LevelSummary:
    subset = files[:level]
    print(f"\n=========== concurrency = {level} sess "
          f"({len(subset)} files) ===========", flush=True)

    t_start = time.perf_counter()
    tasks = [run_session(name, wav) for name, wav in subset]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    level_wall = time.perf_counter() - t_start

    s = summarize(level, results, level_wall)
    failed = [r for r in results if not (r.error is None and r.final_recv_at is not None)]
    print_level_block(s, failed)
    return s


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

async def main() -> None:
    max_n = max(CONCURRENCY_LEVELS)
    paths = _collect_files(max_n)
    if len(paths) < max_n:
        print(f"⚠ {max_n}개 필요한데 {len(paths)}개만 발견됨, 일부 레벨 부족 가능", file=sys.stderr)

    print(f"=== Concurrent WS load test ===")
    print(f"  server  : {SERVER}")
    print(f"  wav_dir : {WAV_DIR}")
    print(f"  levels  : {CONCURRENCY_LEVELS}")
    print(f"  step_ms : {STEP_MS}  language={LANGUAGE}  (real-time paced)")

    print(f"\n[load] {len(paths)} files → memory ...", end=" ", flush=True)
    t_l = time.perf_counter()
    files: List[Tuple[str, np.ndarray]] = []
    for p in paths:
        files.append((os.path.basename(p), _load_16k(p)))
    print(f"done ({time.perf_counter() - t_l:.2f}s)")

    # ---- warm-up (results excluded from aggregates) --------------------
    # 첫 cold 부하가 5s+ 걸리는 케이스가 관찰돼 sequential 3회 워밍업으로 보강.
    print(f"\n[warmup] {WARMUP_ROUNDS} sequential sessions", flush=True)
    for i in range(WARMUP_ROUNDS):
        w = await run_session(files[0][0], files[0][1])
        if w.error:
            print(f"  [{i+1}] ✗ error: {w.error}")
        else:
            print(f"  [{i+1}] ok  TTFT={1000 * (w.first_partial_s or 0):.0f}ms  "
                  f"end→final={w.end_to_final_ms:+.0f}ms  RTF_sess={w.rtf_session:.3f}")
    await asyncio.sleep(LEVEL_SETTLE_SEC)

    # ---- main levels ---------------------------------------------------
    summaries: List[LevelSummary] = []
    for i, level in enumerate(CONCURRENCY_LEVELS):
        if level > len(files):
            print(f"\n⚠ skip level {level}: only {len(files)} files available", file=sys.stderr)
            continue
        s = await run_level(level, files)
        summaries.append(s)
        if i < len(CONCURRENCY_LEVELS) - 1:
            await asyncio.sleep(LEVEL_SETTLE_SEC)

    print_compare_table(summaries)


if __name__ == "__main__":
    asyncio.run(main())
