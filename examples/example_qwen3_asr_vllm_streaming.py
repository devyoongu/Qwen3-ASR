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
Examples for Qwen3ASRModel Streaming Inference (REST API).

Note:
  Requires a running ASR server at SERVER_URL.
"""

import io
import os
import time
from typing import Tuple

import numpy as np
import requests
import soundfile as sf


SERVER_URL = "http://172.31.79.202:30000"
LOCAL_WAV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "wav", "google_tts.wav")


def _load_audio_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _read_wav_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    with io.BytesIO(audio_bytes) as f:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr)


def _resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    """Simple resample to 16k if needed (uses linear interpolation; good enough for a test)."""
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


def run_streaming_case(wav16k: np.ndarray, step_ms: int) -> None:
    sr = 16000
    step = int(round(step_ms / 1000.0 * sr))

    print(f"\n===== streaming step = {step_ms} ms =====")

    # 1. 세션 시작
    start_res = requests.post(f"{SERVER_URL}/api/start")
    start_res.raise_for_status()
    session_id = start_res.json()["session_id"]
    print(f"session_id={session_id!r}")

    # 2. 청크 전송
    pos = 0
    call_id = 0
    first_text_latency: float | None = None
    stream_start = time.perf_counter()

    while pos < wav16k.shape[0]:
        seg = wav16k[pos : pos + step]
        pos += seg.shape[0]
        call_id += 1

        response = requests.post(
            f"{SERVER_URL}/api/chunk",
            params={"session_id": session_id},
            headers={"Content-Type": "application/octet-stream"},
            data=seg.tobytes(),
        )
        response.raise_for_status()
        res_json = response.json()
        language = res_json.get('language')
        text = res_json.get('text', '').replace('\ufffd', '').strip()

        if text and first_text_latency is None:
            first_text_latency = time.perf_counter() - stream_start

        print(f"[call {call_id:03d}] language={language!r} text={text!r}")

    if first_text_latency is not None:
        print(f"[latency] 최초 텍스트 응답: {first_text_latency:.3f}s")

    # 3. 최종 결과
    finish_res = requests.post(f"{SERVER_URL}/api/finish", params={"session_id": session_id})
    finish_res.raise_for_status()
    final = finish_res.json()
    language = final.get('language')
    text = final.get('text', '').replace('\ufffd', '').strip()
    print(f"[final] language={language!r} text={text!r}")


def main() -> None:
    audio_bytes = _load_audio_bytes(LOCAL_WAV_PATH)
    wav, sr = _read_wav_from_bytes(audio_bytes)
    wav16k = _resample_to_16k(wav, sr)

    for step_ms in [500, 1000, 2000, 4000]:
        run_streaming_case(wav16k, step_ms)


if __name__ == "__main__":
    main()
