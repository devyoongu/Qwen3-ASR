# Qwen3-ASR STT 서버 운용 가이드

원격 GPU 서버(`172.31.79.202`)에서 STT 서버를 실행하는 방법을 설명합니다.

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

### 방법 A — Docker Compose (권장, 운영 환경)

컨테이너 내부에서 패키지를 자동 설치한 뒤 서버를 기동합니다.

```bash
# 서버 시작 (백그라운드)
docker compose up -d

# 로그 확인
docker compose logs -f

# 서버 중지
docker compose down
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

| 경로 | 메서드 | 설명 |
|------|-------|------|
| `GET /` | GET | 스트리밍 데모 UI |
| `POST /api/start` | POST | 세션 생성, `{"session_id": "..."}` 반환 |
| `POST /api/chunk?session_id=<id>` | POST | 오디오 청크 전송 (body: float32 PCM, 16kHz) |
| `POST /api/finish?session_id=<id>` | POST | 스트리밍 종료 및 최종 결과 반환 |

`/api/chunk` 요청 헤더: `Content-Type: application/octet-stream`

---

## 5. 코드 업데이트 반영

```bash
# 1. 서버 중지
docker compose down

# 2. 최신 코드 받기
git pull

# 3. 서버 재시작
docker compose up -d
```

> Docker Compose는 시작 시 `pip install -e /app/src` 를 자동 실행하므로
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
# 단일 세션
python examples/example_qwen3_asr_vllm_streaming.py

# 동시 N 세션 (스크립트 내 CONCURRENT_REQUESTS 변수 조정)
python examples/example_qwen3_asr_vllm_streaming_concurrent.py
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
| `connection refused` | 서버 미기동 또는 포트 불일치 | `docker compose logs` 로 기동 오류 확인 |
| `invalid session_id` 오류 | 세션 TTL(10분) 초과 후 재사용 | `/api/start` 로 새 세션 생성 |
| GPU Util이 낮고 응답이 느림 | 단일 세션만 요청 중 | 다중 세션 동시 요청 시 continuous batching 활성화됨 |
| 컨테이너가 즉시 종료됨 | `pip install` 오류 | `docker compose logs` 확인, `--no-deps` 플래그 확인 |
