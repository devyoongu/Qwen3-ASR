# Qwen3-ASR STT 서버 운용 가이드

원격 GPU 서버(`172.31.79.202`)에서 STT 서버를 실행하는 방법을 설명합니다.

---

## 배경: Ubuntu 18.04 호환성 문제와 Docker 해결

서버 OS가 **Ubuntu 18.04 (glibc 2.27)** 이라 `vllm==0.14.0` 을 호스트에 직접 설치할 수 없습니다.
vLLM 0.14.0은 **glibc 2.28 이상**을 요구하기 때문입니다.

```
# 호스트에서 직접 설치 시 발생하는 오류
ERROR: vllm requires glibc >= 2.28
```

**해결책: Docker 사용**

`qwenllm/qwen3-asr:latest` 이미지는 Ubuntu 22.04+ 기반으로 glibc 2.35를 포함하므로
컨테이너 내부에서는 vLLM이 정상 설치·실행됩니다.

이를 위해 `docker-compose.yml` 에 두 가지를 추가했습니다:

```yaml
volumes:
  - /home/posicube/stt/Qwen3-ASR:/app/src   # 소스코드를 컨테이너에 마운트

command:
  - bash
  - -c
  - |
    pip install -e /app/src --no-deps --quiet &&   # 컨테이너(Ubuntu 22.04) 안에서 패키지 재설치
    qwen-asr-serve-async ...
```

| 역할 | 설명 |
|------|------|
| 볼륨 마운트 (`/app/src`) | 이미지 재빌드 없이 `git pull` 만으로 코드 변경 즉시 반영 |
| 시작 시 `pip install -e` | 컨테이너 환경(glibc 2.35)에서 entry point 등록, vLLM 정상 동작 |
| `--no-deps` 플래그 | 이미지에 이미 설치된 의존성을 재설치하지 않아 기동 시간 단축 |

---

## 사전 요구 사항

- NVIDIA GPU (VRAM 24GB 이상 권장)
- Docker + NVIDIA Container Toolkit (`nvidia-container-runtime`)
- `huggingface-cli` 설치 (`pip install huggingface_hub`)

---

## 1. 최초 설치

### 1-1. 모델 다운로드

```bash
huggingface-cli download Qwen/Qwen3-ASR-1.7B \
  --local-dir ~/.cache/huggingface/hub/
```

> 모델 크기: ~3.5GB. 다운로드에 수 분 소요됩니다.

### 1-2. 소스 코드 Clone

```bash
cd /home/posicube/stt
git clone <repo-url> Qwen3-ASR
cd Qwen3-ASR
```

---

## 2. 서버 실행

### 방법 A — docker-compose (권장, 운영 환경)

컨테이너 내부에서 패키지를 자동 설치한 뒤 서버를 기동합니다.

```bash
# 서버 시작 (백그라운드)
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 서버 중지
docker-compose down
```

컨테이너 기동 시 다음 순서로 자동 실행됩니다:
1. `pip install -e /app/src --no-deps` — 소스코드 마운트 경로에서 패키지 재설치
2. `qwen-asr-serve-async` 서버 기동 (포트 30000)

> **모델 로드 완료 메시지**: `Model loaded.` 출력 후 요청 수신 가능.

---

### 방법 B — 직접 실행 (테스트·디버깅용)

Python 3.10+ 가상환경이 준비된 경우:

```bash
# 가상환경 활성화
source .venv/bin/activate

# 의존성 설치 (최초 1회)
pip install -e ".[vllm]"

# 서버 실행
qwen-asr-serve-async \
  --asr-model-path /home/posicube/.cache/huggingface/hub \
  --host 0.0.0.0 \
  --port 30000
```

---

## 3. 서버 옵션

| 옵션 | 기본값 | 설명 |
|------|-------|------|
| `--asr-model-path` | `Qwen/Qwen3-ASR-1.7B` | 모델 경로 또는 HuggingFace repo ID |
| `--host` | `0.0.0.0` | 바인드 호스트 |
| `--port` | `8000` | 바인드 포트 |
| `--gpu-memory-utilization` | `0.8` | vLLM GPU 메모리 사용 비율 |
| `--chunk-size-sec` | `1.0` | 스트리밍 추론 청크 크기(초) |
| `--unfixed-chunk-num` | `4` | 초반 N 청크는 prefix 미사용 |
| `--unfixed-token-num` | `5` | Prefix rollback 토큰 수 |

---

## 4. API 엔드포인트

두 가지 인터페이스를 제공합니다.

| 경로 | 프로토콜 | 설명 |
|------|---------|------|
| `GET /` | HTTP | 스트리밍 데모 UI |
| `POST /api/start` | HTTP | 세션 생성, `{"session_id": "..."}` 반환 |
| `POST /api/chunk?session_id=<id>` | HTTP | 오디오 청크 전송 (body: float32 PCM, 16kHz) |
| `POST /api/finish?session_id=<id>` | HTTP | 스트리밍 종료 및 최종 결과 반환 |
| `/api/ws` | **WebSocket** | 양방향 스트리밍 — 청크별 partial 결과를 서버가 푸시 |

`/api/chunk` 요청 헤더: `Content-Type: application/octet-stream`

---

### 4-1. WebSocket 프로토콜 (`/api/ws`)

OpenAI Realtime / Google StreamingRecognize 스타일의 양방향 스트리밍 인터페이스입니다.
HTTP API와 달리 **연결 수명 = 세션 수명**이며 청크별 결과를 서버가 즉시 푸시하므로
폴링이 필요 없습니다.

#### 프레임 종류

- **오디오** = WebSocket 바이너리 프레임 (float32 LE PCM, 16kHz, mono, 길이 % 4 == 0)
- **제어/결과** = WebSocket 텍스트 프레임 (JSON)

#### Client → Server

| 순서 | 프레임 | 페이로드 | 비고 |
|------|--------|---------|------|
| 1 (필수) | text JSON | `{"type":"config", "context":"...", "language":"...", "chunk_size_sec":1.0, "unfixed_chunk_num":4, "unfixed_token_num":5}` | 첫 메시지. 모든 필드 optional — 누락 시 서버 CLI 기본값 사용 |
| 2..N | binary | float32 PCM 임의 크기 | 서버가 `chunk_size_sec` 단위로 자동 분할 |
| 마지막 | text JSON | `{"type":"end"}` | 잔여 버퍼 flush + final 결과 후 서버가 close |

#### Server → Client

| 이벤트 | 페이로드 | 시점 |
|--------|---------|------|
| `ready` | `{"type":"ready", "session_id":"...", "sample_rate":16000, "chunk_size_sec":..., "unfixed_chunk_num":..., "unfixed_token_num":...}` | config 검증 성공 직후 |
| `result` (interim) | `{"type":"result", "is_final":false, "chunk_id":N, "language":"...", "text":"..."}` | 청크 1개 처리될 때마다 |
| `result` (final) | `{"type":"result", "is_final":true, "chunk_id":N, "language":"...", "text":"..."}` | `end` 처리 후 |
| `error` | `{"type":"error", "code":"...", "message":"..."}` | 잘못된 입력 또는 내부 오류. `code` ∈ {`invalid_config`, `invalid_audio`, `invalid_message`, `internal_error`} |
| `closed` | `{"type":"closed", "reason":"end"\|"error"}` | WebSocket close 직전 마지막 메시지 |

#### 흐름 예시

```
C → S [text]   {"type":"config","language":"ko"}
S → C [text]   {"type":"ready","session_id":"...","sample_rate":16000,...}
C → S [bin]    <16000×4B float32 PCM>
S → C [text]   {"type":"result","is_final":false,"chunk_id":1,"language":"한국어","text":"안녕"}
C → S [bin]    <16000×4B float32 PCM>
S → C [text]   {"type":"result","is_final":false,"chunk_id":2,"language":"한국어","text":"안녕하세요"}
C → S [text]   {"type":"end"}
S → C [text]   {"type":"result","is_final":true,"chunk_id":N,"language":"한국어","text":"안녕하세요 반갑습니다"}
S → C [text]   {"type":"closed","reason":"end"}
[WebSocket close]
```

#### 동작 특성

- VAD/자동 종료 없음 — 클라이언트가 `end`를 명시적으로 보낼 때만 종료
- 락 없음 — 한 연결의 메시지는 단일 코루틴에서 순차 처리. 여러 WebSocket 연결의 `engine.generate()`는 vLLM continuous batching이 동시 묶음 (HTTP `/api/chunk` 다중 세션과 동일한 동시성 모델)
- 세션 TTL/GC 없음 — 연결 종료 시 자동으로 정리됨 (HTTP API의 10분 TTL과 무관)
- `invalid_audio` / `invalid_message`는 연결을 끊지 않고 에러 이벤트만 보냄. `invalid_config` / `internal_error`는 close

---

## 5. 코드 업데이트 반영

```bash
# 1. 서버 중지
docker-compose down

# 2. 최신 코드 받기
git pull

# 3. 서버 재시작
docker-compose up -d
```

> docker-compose는 시작 시 `pip install -e /app/src` 를 자동 실행하므로
> 컨테이너 이미지 재빌드 없이 코드 변경이 즉시 반영됩니다.

---

## 6. 동작 확인

### 서버 기동 확인

```bash
curl http://localhost:30000/api/start -X POST
# 응답 예시: {"session_id": "a1b2c3..."}
```

### 클라이언트 예제 실행 (로컬 PC에서)

```bash
# 단일 세션 (HTTP API)
python examples/example_qwen3_asr_vllm_streaming.py

# 동시 N 세션 (스크립트 내 CONCURRENT_REQUESTS 변수 조정)
python examples/example_qwen3_asr_vllm_streaming_concurrent.py

# WebSocket 스트리밍 (양방향, partial 결과를 서버가 푸시)
pip install websockets
python examples/example_qwen3_asr_vllm_streaming_ws.py
```

### WebSocket 빠른 테스트 (`websocat`)

```bash
# 1. websocat 설치 (macOS)
brew install websocat

# 2. 연결 후 config 전송 → ready 수신
websocat ws://172.31.79.202:30000/api/ws
> {"type":"config","language":"ko"}
< {"type":"ready","session_id":"...","sample_rate":16000,...}

# 3. 종료 (오디오는 binary frame이라 websocat 대화모드로는 보내기 어려우니
#    실제 오디오 전송은 example_qwen3_asr_vllm_streaming_ws.py 사용)
> {"type":"end"}
< {"type":"result","is_final":true,...}
< {"type":"closed","reason":"end"}
```

---

## 7. GPU 모니터링

```bash
# 실시간 GPU 상태 확인
watch -n 1 nvidia-smi

# CSV 형식으로 로그 기록 (부하 테스트 시)
nvidia-smi \
  --query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory,memory.used,pcie.link.gen.current,pcie.link.width.current \
  --format=csv -lms 100 -f gpu_log.csv
```

정상 동작 시 기대 수치:
- 유휴: GPU Util 0%, Power ~22W
- 추론 중: GPU Util **100%**, Power ~320W, VRAM ~19,689 MiB

---

## 8. 트러블슈팅

| 증상 | 원인 | 조치 |
|------|------|------|
| `Model loaded.` 가 출력되지 않음 | 모델 경로 오류 또는 VRAM 부족 | `--asr-model-path` 경로 확인, `nvidia-smi` 로 VRAM 여유 확인 |
| `connection refused` | 서버 미기동 또는 포트 불일치 | `docker-compose logs` 로 기동 오류 확인 |
| `invalid session_id` 오류 | 세션 TTL(10분) 초과 후 재사용 | `/api/start` 로 새 세션 생성 |
| GPU Util이 낮고 응답이 느림 | 단일 세션만 요청 중 | 다중 세션 동시 요청 시 continuous batching 활성화됨 |
| 컨테이너가 즉시 종료됨 | `pip install` 오류 | `docker-compose logs` 확인, `--no-deps` 플래그 확인 |
| WS 연결 직후 `invalid_config` 후 close | 첫 메시지가 바이너리이거나 `type != "config"` | 첫 메시지로 반드시 `{"type":"config", ...}` 텍스트 JSON 전송 |
| WS `invalid_audio` 이벤트 반복 | 바이너리 길이가 4의 배수가 아님 (float32가 아님) | 클라이언트에서 `np.float32` 로 캐스팅 후 `tobytes()` 사용 |
| WS `recv()` 가 멈춤 | `end` 미전송 또는 서버측 처리 지연 | 마지막에 반드시 `{"type":"end"}` 송신, 또는 `closed` 이벤트 수신까지 대기 |
| `ModuleNotFoundError: websockets` | 예시 클라이언트 의존성 누락 | 클라이언트 PC에서 `pip install websockets` (서버는 `uvicorn[standard]`로 이미 포함) |
