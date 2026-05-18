# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`qwen-asr` is a Python package wrapping Qwen3-ASR speech recognition models with two interchangeable backends (Transformers and vLLM), plus CLIs for batch demo, streaming demo, and a concurrent streaming server. This fork's active work is the Korean-language streaming server (`qwen-asr-serve-async`) deployed via Docker on a remote GPU host — see `SERVER_GUIDE.md` (Korean) for the operational playbook and `docs/*.md` for load-test results.

## Install / run

```bash
pip install -e ".[vllm]"      # vLLM extras required for serve / streaming
```

CLI entrypoints (declared in `pyproject.toml`):

| Command | Purpose |
|---|---|
| `qwen-asr-demo` | Gradio batch-inference UI (Transformers backend) |
| `qwen-asr-demo-streaming` | Flask single-process streaming demo (vLLM, sync `LLM`) |
| `qwen-asr-serve` | Thin wrapper that forwards to `vllm serve` with custom model registered |
| `qwen-asr-serve-async` | **Primary production server** — FastAPI + `AsyncLLMEngine` for concurrent sessions |

Production runs the async server via Docker because vLLM 0.14.0 requires glibc ≥ 2.28 (host is Ubuntu 18.04). `docker-compose.yml` mounts the working tree into the container and `pip install -e /app/src --no-deps` re-registers entry points inside the container at startup — so a `git pull` + `docker-compose up -d` is the deploy cycle (no image rebuild). See `SERVER_GUIDE.md` §2 and §5.

Run a streaming load test against the deployed server:

```bash
python examples/example_qwen3_asr_vllm_streaming.py                # single session
python examples/example_qwen3_asr_vllm_streaming_concurrent.py     # N copies of one file
python examples/example_qwen3_asr_vllm_streaming_multi_audio.py    # N distinct files from wav/sequential/
```

`SERVER_URL` is hard-coded near the top of each example (currently `http://172.31.79.202:30000`); edit it for local runs.

There is no automated test suite — `tests/test_asr_ws.py` is a single ad-hoc websocket script, not pytest.

## Architecture

```
qwen_asr/
├── inference/        # Backend-agnostic Python API (Qwen3ASRModel, Qwen3ForcedAligner, ASRStreamingState)
├── core/
│   ├── transformers_backend/   # Custom HF model: Qwen3ASRConfig + Qwen3ASRForConditionalGeneration + Qwen3ASRProcessor
│   └── vllm_backend/           # vLLM-compatible model wrapper
└── cli/              # demo.py, demo_streaming.py, serve.py, serve_async.py
```

**Backend registration is load-bearing.** Every entrypoint (and `qwen_asr/inference/qwen3_asr.py` itself) executes the same boilerplate to register the custom model with both `transformers.AutoConfig`/`AutoModel`/`AutoProcessor` and vLLM's `ModelRegistry`. If you add a new entrypoint, copy this block — without it, `AutoModel.from_pretrained` and `vllm.LLM` won't recognise `Qwen3ASRForConditionalGeneration`. Pattern lives at the top of `qwen_asr/cli/serve_async.py:46-61`.

**Two API surfaces:**

- *Library*: `Qwen3ASRModel.from_pretrained(...)` (Transformers) or `Qwen3ASRModel.LLM(...)` (vLLM). Single class, two factories, `backend` attribute switches behaviour internally (`qwen_asr/inference/qwen3_asr.py`).
- *HTTP*: REST session API on the async server — `POST /api/start` → `POST /api/chunk?session_id=...` (raw float32 PCM bytes, 16kHz, `Content-Type: application/octet-stream`) → `POST /api/finish?session_id=...`. Sessions GC at `SESSION_TTL_SEC = 10 * 60` (`serve_async.py:82`).

**Streaming algorithm** (`ASRStreamingState` in `qwen_asr/inference/qwen3_asr.py:77` + helpers in `serve_async.py:120-186`):

1. Audio is buffered until `chunk_size_samples` (= `chunk_size_sec * 16000`) accumulate, then consumed one chunk at a time. `audio_accum` keeps the running prefix audio fed to the model on every call — the model re-sees all audio so far, not just the new chunk.
2. For the first `unfixed_chunk_num` chunks (default 4), no text prefix is appended to the prompt — the model decodes freely.
3. From chunk N onward, the previously decoded text is tokenized, the last `unfixed_token_num` tokens are dropped (rollback to reduce boundary jitter), and the remainder is appended after the chat-template prompt as a forced decoding prefix. `_build_chunk_prefix` additionally retries with a larger rollback if the truncation lands inside a multi-byte unicode character (yields `�`); `_build_finish_prefix` skips that guard on the final flush.
4. After each generation, raw decoded text passes through `parse_asr_output` to split into `language` and `text` fields.

**Async concurrency model** (`serve_async.py`): each session holds an `asyncio.Lock`, so chunks within one session serialise but multiple sessions await `_async_generate` concurrently. The `engine.generate(...)` calls yield to the event loop, which is what lets vLLM's continuous batcher group requests from different sessions into a single GPU batch. This is the whole point of `serve_async` vs `demo_streaming` (sync Flask + sync `vllm.LLM` → no batching across sessions).

**Server tuning knobs** (CLI flags on `qwen-asr-serve-async`):

| Flag | Default | Notes |
|---|---|---|
| `--asr-model-path` | `Qwen/Qwen3-ASR-1.7B` | HF repo or local dir |
| `--gpu-memory-utilization` | 0.8 | vLLM GPU memory fraction |
| `--chunk-size-sec` | 1.0 | Streaming chunk size |
| `--unfixed-chunk-num` | 4 | Chunks before prefix kicks in |
| `--unfixed-token-num` | 5 | Tokens to roll back per chunk |
| `--port` | 8000 | Docker maps this to 30000 |

`max_model_len=4096` and `SamplingParams(temperature=0.0, max_tokens=32)` are hard-coded in `serve_async.py:359/365` — change there, not via CLI.

## Korean docs

`docs/` contains load-test reports and design notes in Korean (`stt_accuracy_analysis_0401.md`, `multi_audio_load_test_0401.md`, `serve_async_improvement.md`, etc.). Read these before changing streaming parameters or session lifecycle — they document tested defaults.

## Things to know before editing

- `qwen_asr/inference/qwen3_asr.py` registers `qwen3_asr` with `AutoConfig` at import time (line 28). Importing this module has the side effect of mutating global transformers state — don't be surprised when the registration is implicit.
- The vLLM backend import in the same file is wrapped in `try/except: pass` (line 49). Failures are silent; if `from qwen_asr.core.vllm_backend import ...` breaks, the package still imports but `Qwen3ASRModel.LLM(...)` will fail later with a less obvious error.
- `Qwen3ASRProcessor.from_pretrained(..., fix_mistral_regex=True)` is required everywhere the processor is loaded — there is a tokenizer regex bug it works around.
- `wav/sequential/*.wav` is in `.gitignore` (`*.wav`) but tracked anyway because the rule was added after the files were committed. Don't try to "clean up" untracked-looking wav files — they're load-test fixtures.
