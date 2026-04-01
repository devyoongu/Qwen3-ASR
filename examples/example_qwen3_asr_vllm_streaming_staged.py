# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
wav/sequential/ 하위 001~150번 음원을 사용해 50 → 100 → 150 세션을 순차적으로
동시 요청하는 단계별 부하 테스트.

- 각 단계는 번호 순으로 정렬된 파일에서 앞 N개를 사용 (001~050 / 001~100 / 001~150)
- 단계 사이에 STAGE_SLEEP_SEC 초 대기
- 단계마다 성능 통계(RTF, latency 분포)를 출력

사용법:
    python examples/example_qwen3_asr_vllm_streaming_staged.py
"""

import asyncio
import io
import time
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp
import numpy as np
import soundfile as sf


# ------------------------------------------------------------------ #
# 설정                                                                 #
# ------------------------------------------------------------------ #

SERVER_URL = "http://172.31.79.202:30000"
WAV_DIR = Path(__file__).parent.parent / "wav" / "sequential"

STEP_MS = 1000           # 스트리밍 청크 크기 (ms)
STAGE_SLEEP_SEC = 5      # 단계 사이 대기 시간 (초)
STAGES = [50, 100, 150]  # 순차적으로 실행할 동시 세션 수


# ------------------------------------------------------------------ #
# 오디오 유틸                                                           #
# ------------------------------------------------------------------ #

def _load_wav(path: Path) -> Tuple[np.ndarray, int]:
    with open(path, "rb") as f:
        data = f.read()
    with io.BytesIO(data) as buf:
        wav, sr = sf.read(buf, dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr)


def _resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return wav.astype(np.float32, copy=False)
    wav = wav.astype(np.float32, copy=False)
    dur = wav.shape[0] / float(sr)
    n16 = int(round(dur * 16000))
    if n16 <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, dur, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, dur, num=n16, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def load_audio_files(wav_dir: Path, n: int) -> List[Tuple[str, np.ndarray]]:
    """번호 순으로 정렬된 wav 파일에서 앞 n개를 로드해 (filename, wav16k) 리스트로 반환."""
    paths = sorted(wav_dir.glob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"wav 파일이 없습니다: {wav_dir}")
    if len(paths) < n:
        raise ValueError(
            f"요청 수({n})가 음원 파일 수({len(paths)})보다 많습니다. "
            f"STAGES 값을 {len(paths)} 이하로 설정하세요."
        )
    result = []
    for p in paths[:n]:
        wav, sr = _load_wav(p)
        result.append((p.name, _resample_to_16k(wav, sr)))
    return result


# ------------------------------------------------------------------ #
# 스트리밍 세션                                                         #
# ------------------------------------------------------------------ #

async def run_streaming_session(
    session: aiohttp.ClientSession,
    wav16k: np.ndarray,
    filename: str,
    worker_id: int,
    step_ms: int,
) -> dict:
    sr = 16000
    step = int(round(step_ms / 1000.0 * sr))
    duration_sec = wav16k.shape[0] / sr

    # 1. 세션 시작
    async with session.post(f"{SERVER_URL}/api/start") as resp:
        resp.raise_for_status()
        session_id = (await resp.json())["session_id"]

    # 2. 청크 전송
    pos = 0
    call_id = 0
    first_text_latency: Optional[float] = None
    stream_start = time.perf_counter()

    while pos < wav16k.shape[0]:
        seg = wav16k[pos: pos + step]
        pos += seg.shape[0]
        call_id += 1

        async with session.post(
            f"{SERVER_URL}/api/chunk",
            params={"session_id": session_id},
            headers={"Content-Type": "application/octet-stream"},
            data=seg.tobytes(),
        ) as resp:
            resp.raise_for_status()
            res_json = await resp.json()

        text = res_json.get("text", "").replace("\ufffd", "").strip()
        if text and first_text_latency is None:
            first_text_latency = time.perf_counter() - stream_start

    # 3. 최종 결과
    async with session.post(
        f"{SERVER_URL}/api/finish", params={"session_id": session_id}
    ) as resp:
        resp.raise_for_status()
        final = await resp.json()

    elapsed = time.perf_counter() - stream_start
    return {
        "worker_id": worker_id,
        "filename": filename,
        "duration_sec": duration_sec,
        "chunk_count": call_id,
        "first_text_latency": first_text_latency,
        "total_elapsed": elapsed,
        "final_language": final.get("language", ""),
        "final_text": final.get("text", "").replace("\ufffd", "").strip(),
    }


# ------------------------------------------------------------------ #
# 단계 실행 및 출력                                                     #
# ------------------------------------------------------------------ #

async def run_stage(audio_files: List[Tuple[str, np.ndarray]], n: int) -> None:
    print(f"\n{'=' * 70}")
    print(f"  [단계 시작]  동시 세션 수: {n}  /  chunk: {STEP_MS}ms")
    print(f"{'=' * 70}")

    global_start = time.perf_counter()

    connector = aiohttp.TCPConnector(limit=n)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            run_streaming_session(
                session,
                wav16k,
                filename=fname,
                worker_id=i + 1,
                step_ms=STEP_MS,
            )
            for i, (fname, wav16k) in enumerate(audio_files[:n])
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    total_elapsed = time.perf_counter() - global_start

    # ── 개별 결과 출력 ────────────────────────────────────────────────
    print("\n[개별 결과]")
    print(f"{'#':>3}  {'파일명':<40}  {'길이':>5}  {'청크':>4}  {'첫응답':>7}  {'결과'}")
    print("-" * 100)

    latencies: List[float] = []
    total_audio_sec = 0.0
    errors = 0

    for result in results:
        if isinstance(result, Exception):
            errors += 1
            print(f"{'':>3}  [ERROR] {result}")
            continue

        wid = result["worker_id"]
        fname = result["filename"][:38]
        dur = result["duration_sec"]
        chunks = result["chunk_count"]
        lat = result["first_text_latency"]
        lat_str = f"{lat:.3f}s" if lat is not None else "  N/A "
        text_preview = result["final_text"][:40] + ("…" if len(result["final_text"]) > 40 else "")

        print(f"{wid:>3}  {fname:<40}  {dur:>4.1f}s  {chunks:>4}  {lat_str}  {text_preview}")

        if lat is not None:
            latencies.append(lat)
        total_audio_sec += dur

    # ── 종합 통계 ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"[종합 성능 통계]  세션 수: {n}")
    print(f"  전체 소요 시간     : {total_elapsed:.3f}s")
    print(f"  총 오디오 분량     : {total_audio_sec:.1f}s")
    print(f"  실시간 배율 (RTF)  : {total_audio_sec / total_elapsed:.1f}×")
    print(f"  성공 / 실패        : {len(results) - errors} / {errors}")

    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        print(f"\n[최초 텍스트 응답 Latency]")
        print(f"  min : {min(latencies):.3f}s")
        print(f"  avg : {avg_lat:.3f}s")
        print(f"  max : {max(latencies):.3f}s")

        buckets = [
            ("<0.3s",    sum(1 for l in latencies if l < 0.3)),
            ("0.3-0.5s", sum(1 for l in latencies if 0.3 <= l < 0.5)),
            ("0.5-0.8s", sum(1 for l in latencies if 0.5 <= l < 0.8)),
            ("0.8-1.5s", sum(1 for l in latencies if 0.8 <= l < 1.5)),
            (">=1.5s",   sum(1 for l in latencies if l >= 1.5)),
        ]
        print(f"\n[Latency 분포]")
        for label, cnt in buckets:
            bar = "█" * cnt
            print(f"  {label:>9}  {bar:<55}  {cnt:>3}건 ({cnt / len(latencies) * 100:.0f}%)")

    # ── 전체 인식 결과 ────────────────────────────────────────────────
    print(f"\n[전체 인식 결과]")
    for result in results:
        if isinstance(result, Exception):
            continue
        wid = result["worker_id"]
        fname = result["filename"]
        lang = result["final_language"]
        text = result["final_text"]
        print(f"  [{wid:>3}] {fname}")
        print(f"        [{lang}] {text}")


# ------------------------------------------------------------------ #
# 메인                                                                 #
# ------------------------------------------------------------------ #

async def main_async() -> None:
    max_n = max(STAGES)
    print(f"음원 파일 로딩 중... ({WAV_DIR})")
    all_audio = load_audio_files(WAV_DIR, n=max_n)
    print(f"{len(all_audio)}개 파일 로드 완료")
    print(f"단계: {STAGES}  /  단계 간 대기: {STAGE_SLEEP_SEC}초")

    for stage_idx, n in enumerate(STAGES):
        await run_stage(all_audio, n)

        if stage_idx < len(STAGES) - 1:
            print(f"\n  ▶ 다음 단계({STAGES[stage_idx + 1]}세션) 까지 {STAGE_SLEEP_SEC}초 대기 중...")
            await asyncio.sleep(STAGE_SLEEP_SEC)

    print(f"\n{'=' * 70}")
    print("  모든 단계 완료")
    print(f"{'=' * 70}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
