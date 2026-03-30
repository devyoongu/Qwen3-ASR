import asyncio
import json
import base64
import time
import argparse
import os
import requests
import numpy as np
from pydub import AudioSegment
import pandas as pd
from datetime import datetime

# 서버 설정 (Flask 서버 주소)
SERVER_URL = "http://172.31.79.202:30000"

def load_audio(file_path):
    """MP3/WAV를 로드하여 서버가 원하는 16kHz, Mono, Float32 데이터로 변환"""
    # pydub을 사용하여 mp3, wav 등 다양한 포맷 지원
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    # numpy 배열로 변환
    samples = np.array(audio.get_array_of_samples())
    
    # 정규화: int16 -> float32 (-1.0 ~ 1.0)
    # 서버의 np.frombuffer(raw, dtype=np.float32)에 대응
    return samples.astype(np.float32) / 32768.0

def test_intermediate_results(audio_file):
    if not os.path.exists(audio_file):
        print(f"❌ 파일을 찾을 수 없습니다: {audio_file}")
        return

    timeline_events = []
    global_start_time = time.time()

    def record_event(event, details=""):
        now = datetime.now()
        elapsed = time.time() - global_start_time
        timeline_events.append({
            "Time (Absolute)": now.strftime("%H:%M:%S.%f")[:-3],
            "Elapsed (s)": round(elapsed, 3),
            "Event": event,
            "Details": details
        })

    record_event("Session Start", f"Loading file: {audio_file}")

    # 1. 오디오 로드 (MP3 복호화 포함)
    print(f"📂 파일 로드 중: {audio_file}")
    audio_float32 = load_audio(audio_file)
    record_event("Audio Loaded", f"Length: {len(audio_float32)} samples")

    # 2. 세션 시작 (/api/start)
    try:
        start_res = requests.post(f"{SERVER_URL}/api/start")
        start_res.raise_for_status()
        session_id = start_res.json()["session_id"]
        print(f"🔗 세션 생성됨: {session_id}")
        record_event("Session Created", f"ID: {session_id}")
    except Exception as e:
        print(f"❌ 세션 생성 실패: {e}")
        record_event("Error", f"Session Creation Failed: {e}")
        return

    # 3. 오디오를 청크 단위로 전송 (서버의 CHUNK_SIZE_SEC=1.0에 맞춰 16000 샘플씩)
    chunk_size = 16000 
    print("🎙️ 오디오 스트리밍 전송 시작 (Intermediate 결과 수신)...")
    print("-" * 30)

    try:
        start_time = None
        first_response_time = None
        final_response_time = None
        final_chunk_sent_time = None
        total_chunks = (len(audio_float32) + chunk_size - 1) // chunk_size

        for i in range(0, len(audio_float32), chunk_size):
            if start_time is None:
                start_time = time.time()
                record_event("First Chunk Processing Started")

            chunk = audio_float32[i : i + chunk_size]
            chunk_num = i // chunk_size + 1
            
            # float32 바이너리 전송
            raw_bytes = chunk.tobytes()
            
            params = {"session_id": session_id}
            headers = {"Content-Type": "application/octet-stream"}
            
            record_event(f"Chunk {chunk_num}/{total_chunks} Sent")
            if chunk_num == total_chunks:
                final_chunk_sent_time = time.time()
            
            response = requests.post(
                f"{SERVER_URL}/api/chunk", 
                params=params, 
                headers=headers, 
                data=raw_bytes
            )
            
            if response.ok:
                res_json = response.json()
                text = res_json.get('text', '')
                record_event(f"Chunk {chunk_num}/{total_chunks} Response", text)
                if text and first_response_time is None:
                    first_response_time = time.time()
                # 서버 state.text에 담긴 중간 인식 결과 출력
                print(f"\r[중간 결과]: {text}", end="", flush=True)
            else:
                record_event(f"Chunk {chunk_num}/{total_chunks} Error", f"Status: {response.status_code}")
            
            # 실제 스트리밍 속도와 비슷하게 조절
            time.sleep(0.1)

        # 4. 전송 완료 및 최종 결과 수신 (/api/finish)
        print("\n" + "-" * 30)
        record_event("Finish Request Sent")
        finish_res = requests.post(f"{SERVER_URL}/api/finish", params={"session_id": session_id})
        if finish_res.ok:
            final_data = finish_res.json()
            final_response_time = time.time()
            text = final_data.get('text', '')
            record_event("Finish Response Received", f"Text: {text}, Lang: {final_data.get('language', '')}")
            print(f"🏁 최종 인식 문장: {text}")
            print(f"🌐 인식 언어: {final_data.get('language', '')}")
            
            if start_time:
                ttft = (first_response_time - start_time) if first_response_time else 0
                tt_final = (final_response_time - start_time)
                tt_from_last = (final_response_time - final_chunk_sent_time) if final_chunk_sent_time else 0
                print("-" * 30)
                print(f"⏱️ 최초 전달 후 첫 응답까지 걸린 시간: {ttft:.3f}초")
                print(f"⏱️ 첫 전달부터 최종 텍스트 수신까지 걸린 시간: {tt_final:.3f}초")
                print(f"⏱️ 마지막 전달 후 최종 응답까지 걸린 시간: {tt_from_last:.3f}초")

    except Exception as e:
        print(f"\n❌ 전송 중 에러 발생: {e}")
        record_event("Exception Occurred", str(e))
    finally:
        try:
            df = pd.DataFrame(timeline_events)
            df.to_excel("timeline.xlsx", index=False)
            print("📊 타임라인 데이터가 timeline.xlsx 파일로 저장되었습니다.")
        except Exception as e:
            print(f"⚠️ 엑셀 파일 저장 중 오류 발생: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ASR Realtime HTTP Streaming Test")
    # 사용자님의 요구사항대로 기본값은 google_tts.wav로 설정
    parser.add_argument("--audio", type=str, default="google_tts.wav", help="Audio file path (default: google_tts.wav)")
    args = parser.parse_args()

    test_intermediate_results(args.audio)