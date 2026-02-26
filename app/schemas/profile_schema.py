from pydantic import BaseModel, Field
from typing import List, Optional

class ProfileRequest(BaseModel):
    level: str = Field(
        ...,
        description="아동의 현재 읽기 레벨 (L1 ~ L6)"
    )
    interests: List[str] = Field(
        ..., 
        description="STT 모듈을 거쳐 정제된 아동의 관심사 키워드 리스트"
    )
    recent_wcpm: Optional[float] = Field(
        None, 
        description="이전 학습 세션에서 측정한 아동의 WCPM"
    )
    weak_phonemes: Optional[List[str]] = Field(
        None, 
        description="아동이 자주 틀리는 취약 발음 리스트"
    )

class ConstraintResponse(BaseModel):
    level_name: str = Field(..., description="레벨 이름")
    age_group: str = Field(..., description="대상 연령대")
    max_sentence_len: int = Field(..., description="문장당 최대 어절 수")
    vocab_level: str = Field(..., description="어휘 난이도")
    total_pages: int = Field(..., description="총 동화 페이지 수")
    theme_keywords: List[str] = Field(..., description="동화 관심사")
    focus_phonemes: List[str] = Field(..., description="동화에 반복 노출할 취약 음소")
    adjusted_target_wcpm: float = Field(..., description="아동의 최근 성적을 반영해 조정한 목표 WCPM")