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
Examples for Qwen3ASRModel Concurrent Streaming Inference (REST API).

10개의 요청을 동시에 1000ms 청크 단위로 스트리밍 전송.
"""

import asyncio
import io
import os
import time
from typing import Tuple

import aiohttp
import numpy as np
import soundfile as sf


SERVER_URL = "http://172.31.79.202:30000"
LOCAL_WAV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "wav", "google_tts.wav")

STEP_MS = 1000
CONCURRENT_REQUESTS = 10


def _load_audio_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _read_wav_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    with io.BytesIO(audio_bytes) as f:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
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


async def run_streaming_session(
    session: aiohttp.ClientSession,
    wav16k: np.ndarray,
    worker_id: int,
    step_ms: int,
) -> dict:
    sr = 16000
    step = int(round(step_ms / 1000.0 * sr))
    logs = []

    # 1. 세션 시작
    async with session.post(f"{SERVER_URL}/api/start") as resp:
        resp.raise_for_status()
        session_id = (await resp.json())["session_id"]

    # 2. 청크 전송
    pos = 0
    call_id = 0
    first_text_latency: float | None = None
    stream_start = time.perf_counter()

    while pos < wav16k.shape[0]:
        seg = wav16k[pos : pos + step]
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

        language = res_json.get("language")
        text = res_json.get("text", "").replace("\ufffd", "").strip()

        if text and first_text_latency is None:
            first_text_latency = time.perf_counter() - stream_start

        logs.append(f"  [call {call_id:03d}] language={language!r} text={text!r}")

    # 3. 최종 결과
    async with session.post(
        f"{SERVER_URL}/api/finish", params={"session_id": session_id}
    ) as resp:
        resp.raise_for_status()
        final = await resp.json()

    final_language = final.get("language")
    final_text = final.get("text", "").replace("\ufffd", "").strip()

    return {
        "worker_id": worker_id,
        "session_id": session_id,
        "first_text_latency": first_text_latency,
        "final_language": final_language,
        "final_text": final_text,
        "logs": logs,
    }


async def main_async(wav16k: np.ndarray) -> None:
    print(f"동시 요청 수: {CONCURRENT_REQUESTS}, chunk: {STEP_MS}ms")
    print("=" * 60)

    global_start = time.perf_counter()

    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            run_streaming_session(session, wav16k, worker_id=i + 1, step_ms=STEP_MS)
            for i in range(CONCURRENT_REQUESTS)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    total_elapsed = time.perf_counter() - global_start

    # 결과 출력
    for result in results:
        if isinstance(result, Exception):
            print(f"\n[ERROR] {result}")
            continue

        wid = result["worker_id"]
        print(f"\n----- worker {wid:02d} | session={result['session_id']} -----")
        for log in result["logs"]:
            print(log)
        latency = result["first_text_latency"]
        latency_str = f"{latency:.3f}s" if latency is not None else "N/A"
        print(f"  [latency] 최초 텍스트 응답: {latency_str}")
        print(f"  [final] language={result['final_language']!r} text={result['final_text']!r}")

    # 요약
    print("\n" + "=" * 60)
    print(f"전체 소요 시간: {total_elapsed:.3f}s")

    latencies = [
        r["first_text_latency"]
        for r in results
        if not isinstance(r, Exception) and r["first_text_latency"] is not None
    ]
    if latencies:
        print(f"최초 응답 latency - min: {min(latencies):.3f}s  max: {max(latencies):.3f}s  avg: {sum(latencies)/len(latencies):.3f}s")

    errors = sum(1 for r in results if isinstance(r, Exception))
    print(f"성공: {len(results) - errors}/{len(results)}  실패: {errors}/{len(results)}")


def main() -> None:
    audio_bytes = _load_audio_bytes(LOCAL_WAV_PATH)
    wav, sr = _read_wav_from_bytes(audio_bytes)
    wav16k = _resample_to_16k(wav, sr)
    asyncio.run(main_async(wav16k))


if __name__ == "__main__":
    main()
