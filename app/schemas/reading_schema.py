from pydantic import BaseModel, Field
from typing import Optional

class PhonemeError(BaseModel):
    type: str = "phoneme"
    pattern: str = Field(..., description="음소 오류 패턴")
    word: str = Field(..., description="오류가 발생한 원문 단어")

class JosaError(BaseModel):
    type: str = "josa"
    kind: str = Field(..., description="조사 오류 종류")
    stem: str = Field(..., description="오류가 발생한 체언")
    target_josa: Optional[str] = Field(None, description="원문 조사")
    stt_josa: Optional[str] = Field(None, description="아이가 읽은 조사")

class CompleteResponse(BaseModel):
    child_id: int = Field(..., description="아동 ID")
    book_id: str = Field(..., description="동화 ID")
    total_errors: int = Field(..., description="동화 전체에서 누적된 오류 수")
    errors: list[dict] = Field(..., description="누적된 읽기 오류 패턴 목록")
    avg_wcpm: Optional[float] = Field(..., description="동화 평균 WCPM")