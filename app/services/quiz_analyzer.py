from fastapi import HTTPException
from app.services.text_cleaner import normalize_for_quiz

class QuizAnalyzerService:
    def __init__(self):
        pass

    def analyze(self, stt_result: dict, original_quiz: str) -> bool:
        stt_text = stt_result.get("text", "").strip()

        if not stt_text:
            raise HTTPException(status_code=422, detail="STT 변환 결과가 비어있습니다.")
        
        normalized_stt = normalize_for_quiz(stt_text)
        normalized_quiz = normalize_for_quiz(original_quiz)

        return normalized_stt == normalized_quiz