# FastAPI + AsyncLLMEngine 기반 동시 ASR 서버 개선

- **날짜**: 2026-03-30
- **대상 파일**: `qwen_asr/cli/serve_async.py` (신규), `pyproject.toml`, `docker-compose.yml`

---

## 1. 개선 배경

### 기존 구조의 문제

기존 `demo_streaming.py`는 **Flask + vLLM `LLM` (동기)** 조합으로 동작한다.

```
[Flask threaded=True]
  Thread A → streaming_transcribe() → model.generate()  ─┐ 직렬
  Thread B → streaming_transcribe() → model.generate()  ─┤ (엔진 점유)
  Thread C → streaming_transcribe() → model.generate()  ─┘
```

- Flask의 `threaded=True`로 HTTP 레이어는 병렬이지만
- `self.model.generate()` 가 vLLM 엔진을 **블로킹** 으로 점유
- 동시 요청이 들어와도 GPU 추론이 **직렬 실행** → 대기 시간 누적
- 세션 수가 늘수록 tail latency가 선형 증가

### 해결 방향

**FastAPI + uvicorn (async) + vLLM `AsyncLLMEngine`** 으로 교체.

- 다수의 세션이 `await engine.generate()` 를 동시에 호출
- asyncio 이벤트 루프가 각 await 지점에서 다른 코루틴으로 전환
- `AsyncLLMEngine` 이 여러 요청을 **continuous batching** 으로 묶어 한 번에 처리
- GPU 유휴 시간 감소, 동시 처리량 향상

---

## 2. 아키텍처 비교

### Before: Flask + sync LLM

```
Client A ──▶ Flask Thread A ──▶ LLM.generate() ────────────────▶ GPU
Client B ──▶ Flask Thread B ──▶ [대기 중...] ──▶ LLM.generate() ▶ GPU
Client C ──▶ Flask Thread C ──▶              [대기 중...] ───────▶ GPU
                                              (순차 처리)
```

### After: FastAPI + AsyncLLMEngine

```
Client A ──▶ FastAPI coroutine A ──▶ await engine.generate() ─┐
Client B ──▶ FastAPI coroutine B ──▶ await engine.generate() ─┤─▶ AsyncLLMEngine
Client C ──▶ FastAPI coroutine C ──▶ await engine.generate() ─┘   (continuous batch)
                                      (동시 처리)
```

### 동시성 모델 상세

```
세션 A: api_chunk → 오디오 버퍼링 → await _async_generate()  ─┐
세션 B: api_chunk → 오디오 버퍼링 → await _async_generate()  ─┤→ AsyncLLMEngine (1 batch)
세션 C: api_chunk → 오디오 버퍼링 → await _async_generate()  ─┘
```

- **세션 내** 청크 순서: `asyncio.Lock()` 으로 보장 (같은 세션의 청크 A, B가 뒤바뀌지 않음)
- **세션 간** GPU 추론: 동시 실행 → 엔진이 하나의 배치로 처리

---

## 3. 변경 파일 요약

| 파일 | 변경 유형 | 내용 |
|------|----------|------|
| `qwen_asr/cli/serve_async.py` | **신규** | FastAPI 비동기 서버 |
| `pyproject.toml` | 수정 | `fastapi`, `uvicorn[standard]` 의존성 추가; `qwen-asr-serve-async` 스크립트 등록 |
| `docker-compose.yml` | 수정 | command를 `qwen-asr-serve-async` 로 변경 |

> `qwen_asr/inference/qwen3_asr.py` 는 **수정하지 않음** — 기존 코드 보존

---

## 4. `serve_async.py` 구현 상세

### 4-1. 엔진 초기화

```python
engine_args = AsyncEngineArgs(
    model=args.asr_model_path,
    gpu_memory_utilization=args.gpu_memory_utilization,
    max_model_len=4096,
)
engine = AsyncLLMEngine.from_engine_args(engine_args)
processor = Qwen3ASRProcessor.from_pretrained(args.asr_model_path, fix_mistral_regex=True)
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
```

`processor` 는 GPU 없이 토크나이저 전용으로 로드 — prefix rollback 계산에 사용.

### 4-2. 세션 구조

```python
@dataclass
class AsyncSession:
    state: ASRStreamingState   # 기존 상태 객체 재사용
    lock: asyncio.Lock         # 세션 내 청크 순서 보장
    created_at: float
    last_seen: float
```

### 4-3. 비동기 추론 헬퍼

```python
async def _async_generate(inp: dict) -> str:
    request_id = uuid.uuid4().hex
    final = None
    async for output in engine.generate(inp, sampling_params, request_id):
        final = output
    return final.outputs[0].text if final else ""
```

`inp` 형식: `{"prompt": str, "multi_modal_data": {"audio": [np.ndarray]}}`
→ 기존 `streaming_transcribe()` 내 vLLM 입력 구조와 동일.

### 4-4. 청크 처리 핵심 로직 (`/api/chunk`)

```python
async with s.lock:                          # 세션 내 순서 직렬화
    _append_to_buffer(wav, state)           # CPU: 오디오 누적

    while buffer_has_full_chunk(state):     # 준비된 청크 모두 처리
        inp, prefix = _pop_and_build(state) # CPU: 프롬프트 빌드
        gen_text = await _async_generate(inp)  # GPU: 비동기 추론
        _update_state(state, gen_text, prefix)  # CPU: 결과 반영
```

`lock` 내부에서 `await` 를 사용하지만, asyncio.Lock 은 `await` 중에 다른 코루틴이
실행되도록 허용한다. 따라서 세션 A 가 GPU 추론을 기다리는 동안 세션 B 도 동시에
GPU 추론을 요청할 수 있다.

### 4-5. prefix rollback 로직 분리

`streaming_transcribe()` 에서 CPU 전용 로직을 두 함수로 분리하여 재구현:

| 함수 | 사용 시점 | 특이사항 |
|------|----------|---------|
| `_build_chunk_prefix(state)` | `/api/chunk` | `\ufffd` 감지 시 k 증가 (unicode guard) |
| `_build_finish_prefix(state)` | `/api/finish` | unicode guard 없음 (`finish_streaming_transcribe` 동일) |

### 4-6. 엔드포인트

| 경로 | 메서드 | 설명 |
|------|-------|------|
| `GET /` | — | 기존 INDEX_HTML 그대로 (demo_streaming에서 import) |
| `POST /api/start` | — | 세션 생성, GPU 호출 없음 |
| `POST /api/chunk?session_id=...` | octet-stream body | 청크 처리 (위 로직) |
| `POST /api/finish?session_id=...` | — | tail flush + 세션 제거 |

---

## 5. 의존성 변경 (`pyproject.toml`)

```toml
[project.optional-dependencies]
vllm = [
  "vllm==0.14.0",
  "fastapi",           # 추가
  "uvicorn[standard]", # 추가
]

[project.scripts]
...
qwen-asr-serve-async = "qwen_asr.cli.serve_async:main"  # 추가
```

---

## 6. docker-compose.yml 변경

```yaml
# 변경 전
command: >
  qwen-asr-demo-streaming
  --asr-model-path /app/models
  --host 0.0.0.0
  --port 30000

# 변경 후
command: >
  qwen-asr-serve-async
  --asr-model-path /app/models
  --host 0.0.0.0
  --port 30000
```

---

## 7. 배포 절차

```bash
# 1. docker compose 중단
docker compose down

# 2. 최신 코드 반영
git pull

# 3. 의존성 설치 (venv 내)
pip install -e ".[vllm]"

# 4. 직접 실행 (기동 테스트)
qwen-asr-serve-async \
  --asr-model-path /home/posicube/.cache/huggingface/hub \
  --host 0.0.0.0 \
  --port 30000

# 5. 안정 확인 후 docker compose로 전환
docker compose up -d
```

---

## 8. 검증 방법

### 단일 세션 정상 동작 확인

```bash
python examples/example_qwen3_asr_vllm_streaming_concurrent.py
# CONCURRENT_REQUESTS = 1 로 설정
```

### 동시 처리 확인

`examples/example_qwen3_asr_vllm_streaming_concurrent.py` 의 `CONCURRENT_REQUESTS` 값을 늘려서 테스트:

```python
CONCURRENT_REQUESTS = 2   # → 10 까지 증가
```

**기대 동작**: 전체 소요 시간이 단일 요청(1개)과 비슷하면 concurrent batching 이 동작 중.

```
단일 요청:  elapsed ≈ T
2개 동시:   elapsed ≈ T      ← 이상적
2개 직렬:   elapsed ≈ 2×T    ← 기존 Flask 동작
```

### GPU 사용률 모니터링

```bash
watch -n 1 nvidia-smi
```

동시 요청 시 GPU Util 이 단일 요청 대비 유사하거나 높아야 한다.
단일 요청에서 GPU Util이 낮고 동시 요청에서 높아진다면 batching이 효과를 내는 것.

---

## 9. CLI 옵션 (serve_async)

| 옵션 | 기본값 | 설명 |
|------|-------|------|
| `--asr-model-path` | `Qwen/Qwen3-ASR-1.7B` | 모델 경로 또는 HF repo |
| `--host` | `0.0.0.0` | 바인드 host |
| `--port` | `8000` | 바인드 port |
| `--gpu-memory-utilization` | `0.8` | vLLM GPU 메모리 비율 |
| `--unfixed-chunk-num` | `4` | 초반 N 청크는 prefix 미사용 |
| `--unfixed-token-num` | `5` | prefix 롤백 토큰 수 |
| `--chunk-size-sec` | `1.0` | 모델 추론 단위 청크 크기(초) |
