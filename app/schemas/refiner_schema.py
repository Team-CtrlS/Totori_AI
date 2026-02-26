from pydantic import BaseModel, Field
from typing import List

class InterestRequest(BaseModel):
    text: str = Field(
        ...,
        description="STT 모듈에서 추출된 아동의 원본 음성 텍스트"
    )

class InterestResponse(BaseModel):
    original_text: str = Field(
        ...,
        description="입력되었던 원본 텍스트"
    )
    keywords: List[str] = Field(
        ...,
        description="정제된 관심사 키워드 리스트"
    )
    is_empty: bool = Field(
        ...,
        description="유효한 키워드가 있는지 여부 (True면 프론트에서 재입력 유도)"
    )