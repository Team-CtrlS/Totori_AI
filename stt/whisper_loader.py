import os
import whisper
from typing import Dict, Any

# 앱 시작 시 1번만 로드
MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")
_model = whisper.load_model(MODEL_NAME)

# 프리셋 정의
PRESETS = {
    # 언어모델 보정 약화
    "raw": dict(
        language="ko",
        task="transcribe",
        temperature=0.7,    # 높을수록 여러 가능성. (발음의 흔들림이 남음)
        beam_size=1,        # 보정 최소
        best_of=5,
        condition_on_previous_text=False,   # 이전 segment 결과 다음에도 반영 x
        word_timestamps=True,   # 시작/끝 시간 함께 반환
    ),
    "balanced": dict(
        language="ko",
        task="transcribe",
        temperature=0.2,
        beam_size=5,
        best_of=1,
        condition_on_previous_text=True,
        word_timestamps=True,
    ),
    # 가장 자연스러운 전사
    "clean": dict(
        language="ko",
        task="transcribe",
        temperature=0.0,    # 강한 보정
        beam_size=5,        # 문맥 고려
        best_of=1,
        condition_on_previous_text=True,
        word_timestamps=True,
    ),
}

def transcribe_with_timestamps(
    audio_path: str,
    preset: str = "raw",
) -> Dict[str, Any]:
    if preset not in PRESETS:
        raise ValueError(f"Invalid preset: {preset}")

    result = _model.transcribe(
        audio_path,
        **PRESETS[preset]
    )

    # 결과 받아오기
    segments = []
    for s in result.get("segments", []):
        seg = {
            "id": s.get("id"),
            "start": s.get("start"),
            "end": s.get("end"),
            "text": (s.get("text") or "").strip(),
        }

        if s.get("words"):
            seg["words"] = [
                {
                    "word": (w.get("word") or "").strip(),
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "probability": w.get("probability"),
                }
                for w in s["words"]
            ]

        segments.append(seg)

    return {
        "model": MODEL_NAME,
        "language": result.get("language"),
        "text": (result.get("text") or "").strip(),
        "segments": segments,
    }
