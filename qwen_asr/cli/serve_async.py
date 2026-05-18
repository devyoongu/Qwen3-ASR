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
FastAPI + AsyncLLMEngine based concurrent ASR server.

Compared to demo_streaming.py (Flask + sync LLM), this server uses
AsyncLLMEngine so that multiple sessions can submit inference requests
simultaneously and vLLM's continuous-batching engine groups them into
a single GPU batch — reducing idle time and improving throughput.

Install:
  pip install qwen-asr[vllm]

Run:
  qwen-asr-serve-async --asr-model-path <path> --host 0.0.0.0 --port 30000
Open:
  http://0.0.0.0:30000

Interfaces:

  1) HTTP REST API (polling-style, session_id 기반)
     - POST /api/start                             → {"session_id": "..."}
     - POST /api/chunk?session_id=...  (octet-stream float32 PCM)
                                                   → {"language": "...", "text": "..."}
     - POST /api/finish?session_id=...             → final {"language", "text"}

  2) WebSocket streaming API (server-pushed partial results)
     - Endpoint:  ws://<host>:<port>/api/ws
     - Frames:
         Client → Server
           [text JSON]  {"type":"config", "context":"...", "language":"...",
                          "chunk_size_sec":1.0, "unfixed_chunk_num":4,
                          "unfixed_token_num":5}                    # 첫 메시지 필수
           [binary]     float32 LE PCM (16kHz, mono), len % 4 == 0  # 오디오 청크
           [text JSON]  {"type":"end"}                              # flush + final + close
         Server → Client
           [text JSON]  {"type":"ready", "session_id":"...", "sample_rate":16000,
                         "chunk_size_sec":..., "unfixed_chunk_num":...,
                         "unfixed_token_num":...}
           [text JSON]  {"type":"result", "is_final":false, "chunk_id":N,
                         "language":"...", "text":"..."}            # 청크 처리될 때마다
           [text JSON]  {"type":"result", "is_final":true, "chunk_id":N,
                         "language":"...", "text":"..."}            # end 후
           [text JSON]  {"type":"error", "code":"<string>",
                         "message":"..."}                            # 잘못된 입력/내부 오류
           [text JSON]  {"type":"closed", "reason":"end"|"error"}    # close 직전 마지막
"""

import argparse
import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

# Register custom transformers config / model so AutoModel works.
from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration as _TransformersASRModel,
    Qwen3ASRProcessor,
)
from transformers import AutoConfig, AutoModel, AutoProcessor

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, _TransformersASRModel)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

# Register vLLM backend model and import async engine.
from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration as _VllmASRModel
from vllm import AsyncEngineArgs, AsyncLLMEngine, ModelRegistry, SamplingParams

ModelRegistry.register_model("Qwen3ASRForConditionalGeneration", _VllmASRModel)

from qwen_asr.cli.demo_streaming import INDEX_HTML
from qwen_asr.inference.qwen3_asr import ASRStreamingState
from qwen_asr.inference.utils import (
    SAMPLE_RATE,
    normalize_language_name,
    parse_asr_output,
    validate_language,
)


# --------------------------------------------------------------------------- #
# Globals (populated in main())                                                #
# --------------------------------------------------------------------------- #

engine: AsyncLLMEngine = None          # type: ignore[assignment]
processor: Qwen3ASRProcessor = None    # type: ignore[assignment]
sampling_params: SamplingParams = None # type: ignore[assignment]

SESSIONS: Dict[str, "AsyncSession"] = {}
SESSION_TTL_SEC = 10 * 60

UNFIXED_CHUNK_NUM: int = 4
UNFIXED_TOKEN_NUM: int = 5
CHUNK_SIZE_SEC: float = 1.0


# --------------------------------------------------------------------------- #
# Session                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class AsyncSession:
    state: ASRStreamingState
    lock: asyncio.Lock
    created_at: float
    last_seen: float


def _gc_sessions() -> None:
    now = time.time()
    dead = [sid for sid, s in list(SESSIONS.items()) if now - s.last_seen > SESSION_TTL_SEC]
    for sid in dead:
        SESSIONS.pop(sid, None)


def _get_session(session_id: str) -> Optional[AsyncSession]:
    _gc_sessions()
    s = SESSIONS.get(session_id)
    if s:
        s.last_seen = time.time()
    return s


# --------------------------------------------------------------------------- #
# CPU helpers (no GPU call)                                                    #
# --------------------------------------------------------------------------- #

def _build_text_prompt(context: str, force_language: Optional[str]) -> str:
    msgs = [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    base = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if force_language:
        base = base + f"language {force_language}<asr_text>"
    return base


def _init_streaming_state(
    context: str = "",
    language: Optional[str] = None,
    unfixed_chunk_num: int = 4,
    unfixed_token_num: int = 5,
    chunk_size_sec: float = 1.0,
) -> ASRStreamingState:
    force_language = None
    if language is not None and str(language).strip():
        ln = normalize_language_name(str(language))
        validate_language(ln)
        force_language = ln

    chunk_size_samples = max(1, int(round(float(chunk_size_sec) * SAMPLE_RATE)))
    prompt_raw = _build_text_prompt(context=context, force_language=force_language)

    return ASRStreamingState(
        unfixed_chunk_num=int(unfixed_chunk_num),
        unfixed_token_num=int(unfixed_token_num),
        chunk_size_sec=float(chunk_size_sec),
        chunk_size_samples=chunk_size_samples,
        chunk_id=0,
        buffer=np.zeros((0,), dtype=np.float32),
        audio_accum=np.zeros((0,), dtype=np.float32),
        prompt_raw=prompt_raw,
        context=context or "",
        force_language=force_language,
        language="",
        text="",
        _raw_decoded="",
    )


def _build_chunk_prefix(state: ASRStreamingState) -> str:
    """Build rollback prefix for an in-progress chunk (with unicode guard)."""
    if state.chunk_id < state.unfixed_chunk_num:
        return ""
    cur_ids = processor.tokenizer.encode(state._raw_decoded)
    k = int(state.unfixed_token_num)
    while True:
        end_idx = max(0, len(cur_ids) - k)
        prefix = processor.tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
        if "\ufffd" not in prefix:
            return prefix
        if end_idx == 0:
            return ""
        k += 1


def _build_finish_prefix(state: ASRStreamingState) -> str:
    """Build rollback prefix for the final flush (no unicode guard — mirrors finish_streaming_transcribe)."""
    if state.chunk_id < state.unfixed_chunk_num:
        return ""
    cur_ids = processor.tokenizer.encode(state._raw_decoded)
    end_idx = max(1, len(cur_ids) - int(state.unfixed_token_num))
    return processor.tokenizer.decode(cur_ids[:end_idx])


# --------------------------------------------------------------------------- #
# Async GPU helper                                                             #
# --------------------------------------------------------------------------- #

async def _async_generate(inp: dict) -> str:
    """Submit one inference request to AsyncLLMEngine and return generated text."""
    request_id = uuid.uuid4().hex
    final = None
    async for output in engine.generate(inp, sampling_params, request_id):
        final = output
    return final.outputs[0].text if final else ""


# --------------------------------------------------------------------------- #
# FastAPI app                                                                  #
# --------------------------------------------------------------------------- #

app = FastAPI()


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


@app.post("/api/start")
async def api_start() -> JSONResponse:
    session_id = uuid.uuid4().hex
    state = _init_streaming_state(
        unfixed_chunk_num=UNFIXED_CHUNK_NUM,
        unfixed_token_num=UNFIXED_TOKEN_NUM,
        chunk_size_sec=CHUNK_SIZE_SEC,
    )
    now = time.time()
    SESSIONS[session_id] = AsyncSession(
        state=state,
        lock=asyncio.Lock(),
        created_at=now,
        last_seen=now,
    )
    return JSONResponse({"session_id": session_id})


@app.post("/api/chunk")
async def api_chunk(request: Request) -> JSONResponse:
    session_id = request.query_params.get("session_id", "")
    s = _get_session(session_id)
    if not s:
        return JSONResponse({"error": "invalid session_id"}, status_code=400)

    if request.headers.get("content-type", "") != "application/octet-stream":
        return JSONResponse({"error": "expect application/octet-stream"}, status_code=400)

    raw = await request.body()
    if len(raw) % 4 != 0:
        return JSONResponse({"error": "float32 bytes length not multiple of 4"}, status_code=400)

    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1)

    # session.lock serialises chunks within one session while letting other
    # sessions' awaits run concurrently → AsyncLLMEngine can batch them.
    async with s.lock:
        state = s.state

        x = wav.astype(np.float32, copy=False)
        if x.shape[0] > 0:
            state.buffer = np.concatenate([state.buffer, x], axis=0)

        # Consume all full chunks available in the buffer.
        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[: state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples :]

            if state.audio_accum.shape[0] == 0:
                state.audio_accum = chunk
            else:
                state.audio_accum = np.concatenate([state.audio_accum, chunk], axis=0)

            prefix = _build_chunk_prefix(state)
            inp = {
                "prompt": state.prompt_raw + prefix,
                "multi_modal_data": {"audio": [state.audio_accum]},
            }

            gen_text = await _async_generate(inp)  # yields to event loop → other sessions run

            state._raw_decoded = (prefix + gen_text) if prefix else gen_text
            lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
            state.language = lang
            state.text = txt
            state.chunk_id += 1

    return JSONResponse({
        "language": s.state.language or "",
        "text": s.state.text or "",
    })


@app.post("/api/finish")
async def api_finish(request: Request) -> JSONResponse:
    session_id = request.query_params.get("session_id", "")
    s = _get_session(session_id)
    if not s:
        return JSONResponse({"error": "invalid session_id"}, status_code=400)

    async with s.lock:
        state = s.state

        if state.buffer is not None and state.buffer.shape[0] > 0:
            tail = state.buffer
            state.buffer = np.zeros((0,), dtype=np.float32)

            if state.audio_accum.shape[0] == 0:
                state.audio_accum = tail
            else:
                state.audio_accum = np.concatenate([state.audio_accum, tail], axis=0)

            prefix = _build_finish_prefix(state)
            inp = {
                "prompt": state.prompt_raw + prefix,
                "multi_modal_data": {"audio": [state.audio_accum]},
            }

            gen_text = await _async_generate(inp)

            state._raw_decoded = (prefix + gen_text) if prefix else gen_text
            lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
            state.language = lang
            state.text = txt
            state.chunk_id += 1

    out = {
        "language": s.state.language or "",
        "text": s.state.text or "",
    }
    SESSIONS.pop(session_id, None)
    return JSONResponse(out)


# --------------------------------------------------------------------------- #
# WebSocket streaming endpoint                                                 #
# --------------------------------------------------------------------------- #

async def _ws_send_json(ws: WebSocket, payload: Dict[str, Any]) -> None:
    await ws.send_text(json.dumps(payload, ensure_ascii=False))


async def _ws_send_error(ws: WebSocket, code: str, message: str) -> None:
    await _ws_send_json(ws, {"type": "error", "code": code, "message": message})


async def _ws_consume_buffered_chunks(ws: WebSocket, state: ASRStreamingState) -> None:
    """Drain full chunks out of state.buffer, sending one interim result per chunk."""
    while state.buffer.shape[0] >= state.chunk_size_samples:
        chunk = state.buffer[: state.chunk_size_samples]
        state.buffer = state.buffer[state.chunk_size_samples :]

        if state.audio_accum.shape[0] == 0:
            state.audio_accum = chunk
        else:
            state.audio_accum = np.concatenate([state.audio_accum, chunk], axis=0)

        prefix = _build_chunk_prefix(state)
        inp = {
            "prompt": state.prompt_raw + prefix,
            "multi_modal_data": {"audio": [state.audio_accum]},
        }
        gen_text = await _async_generate(inp)

        state._raw_decoded = (prefix + gen_text) if prefix else gen_text
        lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
        state.language = lang
        state.text = txt
        state.chunk_id += 1

        await _ws_send_json(ws, {
            "type": "result",
            "is_final": False,
            "chunk_id": state.chunk_id,
            "language": state.language or "",
            "text": state.text or "",
        })


async def _ws_flush_final(ws: WebSocket, state: ASRStreamingState) -> None:
    """Mirror of /api/finish: flush remaining tail audio and send the final result."""
    if state.buffer is not None and state.buffer.shape[0] > 0:
        tail = state.buffer
        state.buffer = np.zeros((0,), dtype=np.float32)

        if state.audio_accum.shape[0] == 0:
            state.audio_accum = tail
        else:
            state.audio_accum = np.concatenate([state.audio_accum, tail], axis=0)

        prefix = _build_finish_prefix(state)
        inp = {
            "prompt": state.prompt_raw + prefix,
            "multi_modal_data": {"audio": [state.audio_accum]},
        }
        gen_text = await _async_generate(inp)

        state._raw_decoded = (prefix + gen_text) if prefix else gen_text
        lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
        state.language = lang
        state.text = txt
        state.chunk_id += 1

    await _ws_send_json(ws, {
        "type": "result",
        "is_final": True,
        "chunk_id": state.chunk_id,
        "language": state.language or "",
        "text": state.text or "",
    })


@app.websocket("/api/ws")
async def api_ws(ws: WebSocket) -> None:
    """Streaming ASR over WebSocket. See module docstring for the wire protocol."""
    await ws.accept()
    session_id = uuid.uuid4().hex
    state: Optional[ASRStreamingState] = None

    try:
        # ---- 1. config (first text frame) ------------------------------------
        first = await ws.receive()
        if first.get("type") == "websocket.disconnect":
            return

        first_text = first.get("text")
        if first_text is None:
            await _ws_send_error(ws, "invalid_config",
                                 "first message must be a JSON text frame with type='config'")
            await _ws_send_json(ws, {"type": "closed", "reason": "error"})
            await ws.close()
            return

        try:
            cfg = json.loads(first_text)
            if not isinstance(cfg, dict) or cfg.get("type") != "config":
                raise ValueError("first message must be JSON object with type='config'")
            state = _init_streaming_state(
                context=cfg.get("context") or "",
                language=cfg.get("language"),
                unfixed_chunk_num=int(cfg.get("unfixed_chunk_num", UNFIXED_CHUNK_NUM)),
                unfixed_token_num=int(cfg.get("unfixed_token_num", UNFIXED_TOKEN_NUM)),
                chunk_size_sec=float(cfg.get("chunk_size_sec", CHUNK_SIZE_SEC)),
            )
        except Exception as e:
            await _ws_send_error(ws, "invalid_config", str(e))
            await _ws_send_json(ws, {"type": "closed", "reason": "error"})
            await ws.close()
            return

        await _ws_send_json(ws, {
            "type": "ready",
            "session_id": session_id,
            "sample_rate": SAMPLE_RATE,
            "chunk_size_sec": state.chunk_size_sec,
            "unfixed_chunk_num": state.unfixed_chunk_num,
            "unfixed_token_num": state.unfixed_token_num,
        })

        # ---- 2. audio (binary) / control (text) loop -------------------------
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                return

            raw_bytes = msg.get("bytes")
            raw_text = msg.get("text")

            if raw_bytes is not None:
                if len(raw_bytes) % 4 != 0:
                    await _ws_send_error(ws, "invalid_audio",
                                         "float32 bytes length not multiple of 4")
                    continue
                if len(raw_bytes) == 0:
                    continue
                wav = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1)
                if wav.shape[0] > 0:
                    state.buffer = np.concatenate([state.buffer, wav], axis=0)
                await _ws_consume_buffered_chunks(ws, state)

            elif raw_text is not None:
                try:
                    payload = json.loads(raw_text)
                except Exception as e:
                    await _ws_send_error(ws, "invalid_message", f"not JSON: {e}")
                    continue
                if not isinstance(payload, dict):
                    await _ws_send_error(ws, "invalid_message", "JSON must be an object")
                    continue

                ptype = payload.get("type")
                if ptype == "end":
                    await _ws_flush_final(ws, state)
                    await _ws_send_json(ws, {"type": "closed", "reason": "end"})
                    await ws.close()
                    return
                else:
                    await _ws_send_error(ws, "invalid_message",
                                         f"unknown type: {ptype!r}")
            else:
                # Neither bytes nor text — ignore.
                continue

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await _ws_send_error(ws, "internal_error", str(e))
            await _ws_send_json(ws, {"type": "closed", "reason": "error"})
            await ws.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# CLI entry point                                                              #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3-ASR Async Streaming Server (FastAPI + AsyncLLMEngine)"
    )
    p.add_argument("--asr-model-path", default="Qwen/Qwen3-ASR-1.7B", help="Model name or local path")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=8000, help="Bind port")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="vLLM GPU memory utilization")
    p.add_argument("--unfixed-chunk-num", type=int, default=4)
    p.add_argument("--unfixed-token-num", type=int, default=5)
    p.add_argument("--chunk-size-sec", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global engine, processor, sampling_params
    global UNFIXED_CHUNK_NUM, UNFIXED_TOKEN_NUM, CHUNK_SIZE_SEC

    UNFIXED_CHUNK_NUM = args.unfixed_chunk_num
    UNFIXED_TOKEN_NUM = args.unfixed_token_num
    CHUNK_SIZE_SEC = args.chunk_size_sec

    engine_args = AsyncEngineArgs(
        model=args.asr_model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    processor = Qwen3ASRProcessor.from_pretrained(args.asr_model_path, fix_mistral_regex=True)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

    print("Model loaded.")
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
