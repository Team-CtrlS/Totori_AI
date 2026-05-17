from pydantic import BaseModel, Field
from typing import List

# 음운 퀴즈
class QuizRequest(BaseModel):
    child_id: int = Field(..., description="아동 ID")
    book_id: int = Field(..., description="동화 ID")
    level: str = Field(..., description="아동의 현재 읽기 레벨(L1~L6)")

class QuizResponse(BaseModel):
    quiz_items: List[str] = Field(..., description="단어 4개(L1~L3) 또는 문장 4개(L4~L6)")
    audio_data: List[str] = Field(..., description="quiz_items 순서에 대응하는 base64 인코딩된 MP3 오디오 (TTS 실패 시 빈 문자열)")

class AnalyzeQuizResponse(BaseModel):
    is_correct: bool = Field(..., description="퀴즈 정답 여부")