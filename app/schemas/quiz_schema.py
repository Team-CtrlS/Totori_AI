from pydantic import BaseModel, Field
from typing import List, Optional

# 음운 퀴즈
class QuizRequest(BaseModel):
    original_text: str = Field(..., description="동화 원문 텍스트")
    stt_text: str = Field(..., description="아이가 낭독한 STT 결과 텍스트")
    level: str = Field(..., description="아동의 현재 읽기 레벨(L1-L6)")

class PhonemeErrorDetail(BaseModel):
    error_pattern: str = Field(..., description="가장 빈번한 오류 패턴")
    target_word: str = Field(..., description="오류가 발생한 원래 단어")
    error_count: int = Field(..., description="해당 오류 패턴의 발생 횟수")

class QuizResponse(BaseModel):
    quiz_words: List[str] = Field(..., description="퀴즈 단어 4개")

# 조사 퀴즈
class JosaQuizRequest(BaseModel):
    original_text: str = Field(..., description="동화 원문 텍스트")
    stt_text: str = Field(..., description="아이가 낭독한 STT 결과 텍스트")
    level: str = Field(..., description="아동의 현재 읽기 레벨(L4-L6)")

class JosaErrorDetail(BaseModel):
    kind: str = Field(..., description="조사 오류 종류: DELETE, SUBSTITUTION | INSERTION")
    stem: str = Field(..., description="조사 오류가 발생한 체언")
    target_josa: Optional[str] = Field(None, description="원문 조사")
    stt_josa: Optional[str] = Field(None, description="아이가 읽은 조사")

class JosaQuizResponse(BaseModel):
    quiz_sentences: List[str] = Field(..., description="퀴즈 문장 4개")