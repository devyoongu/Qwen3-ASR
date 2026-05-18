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
Example: Qwen3-ASR streaming via WebSocket.

Wire protocol (see qwen_asr/cli/serve_async.py docstring):
  C → S [text]    {"type":"config", "language":"ko", ...}     # first message
  C → S [binary]  float32 LE PCM (16kHz, mono)                # audio chunks
  C → S [text]    {"type":"end"}                              # flush & close
  S → C [text]    {"type":"ready", "session_id":...}
  S → C [text]    {"type":"result", "is_final":false|true, "chunk_id":N,
                    "language":..., "text":...}
  S → C [text]    {"type":"error", "code":..., "message":...}
  S → C [text]    {"type":"closed", "reason":"end"|"error"}

Note:
  Requires a running ASR server at SERVER_URL and the `websockets` package.
"""

import asyncio
import io
import json
import os
import time
from typing import Tuple

import numpy as np
import soundfile as sf
import websockets


SERVER_URL = "ws://172.31.79.202:30000/api/ws"
LOCAL_WAV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "wav", "google_tts.wav")


def _read_wav(path: str) -> Tuple[np.ndarray, int]:
    with open(path, "rb") as f:
        audio_bytes = f.read()
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


async def _recv_loop(ws, state: dict) -> None:
    """Print every server event with a timestamp until the connection closes."""
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                # 서버는 텍스트만 보내므로 도착하면 안 되지만, 방어적으로 무시.
                continue
            event = json.loads(msg)
            etype = event.get("type")
            t = time.perf_counter() - state["t0"]

            if etype == "ready":
                state["session_id"] = event.get("session_id")
                print(f"[{t:6.3f}s] <ready> session_id={state['session_id']} "
                      f"sample_rate={event.get('sample_rate')} "
                      f"chunk_size_sec={event.get('chunk_size_sec')} "
                      f"unfixed_chunk_num={event.get('unfixed_chunk_num')} "
                      f"unfixed_token_num={event.get('unfixed_token_num')}")
                state["ready"].set()
            elif etype == "result":
                tag = "final" if event.get("is_final") else "partial"
                if state["first_text_latency"] is None and event.get("text"):
                    state["first_text_latency"] = t
                print(f"[{t:6.3f}s] <result {tag}> chunk_id={event.get('chunk_id')} "
                      f"language={event.get('language')!r} text={event.get('text')!r}")
            elif etype == "error":
                print(f"[{t:6.3f}s] <error> code={event.get('code')} "
                      f"message={event.get('message')}")
            elif etype == "closed":
                print(f"[{t:6.3f}s] <closed> reason={event.get('reason')}")
            else:
                print(f"[{t:6.3f}s] <{etype}> {event}")
    except websockets.ConnectionClosed:
        pass


async def run_streaming_case(wav16k: np.ndarray, step_ms: int, language: str = "Korean") -> None:
    sr = 16000
    step = int(round(step_ms / 1000.0 * sr))

    print(f"\n===== streaming step = {step_ms} ms =====")

    async with websockets.connect(SERVER_URL, max_size=None) as ws:
        state = {
            "t0": time.perf_counter(),
            "session_id": None,
            "ready": asyncio.Event(),
            "first_text_latency": None,
        }
        recv_task = asyncio.create_task(_recv_loop(ws, state))

        # 1. config 전송 + ready 대기
        await ws.send(json.dumps({"type": "config", "language": language}))
        await asyncio.wait_for(state["ready"].wait(), timeout=10.0)

        # 2. 오디오 청크 전송
        pos = 0
        call_id = 0
        while pos < wav16k.shape[0]:
            seg = wav16k[pos : pos + step]
            pos += seg.shape[0]
            call_id += 1
            await ws.send(seg.tobytes())
            # 실시간 페이싱을 흉내내고 싶다면 다음 줄 활성화:
            # await asyncio.sleep(step_ms / 1000.0)

        # 3. end 전송 → 서버가 final 결과 후 close
        await ws.send(json.dumps({"type": "end"}))

        # 4. recv 루프가 ConnectionClosed 로 끝날 때까지 대기
        await recv_task

        if state["first_text_latency"] is not None:
            print(f"[latency] 최초 텍스트 응답: {state['first_text_latency']:.3f}s")


async def main() -> None:
    wav, sr = _read_wav(LOCAL_WAV_PATH)
    wav16k = _resample_to_16k(wav, sr)

    for step_ms in [500, 1000, 2000, 4000]:
        await run_streaming_case(wav16k, step_ms)


if __name__ == "__main__":
    asyncio.run(main())
