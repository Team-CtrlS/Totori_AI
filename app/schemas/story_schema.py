from pydantic import BaseModel, Field
from typing import List, Optional

class GenerateStoryRequest(BaseModel):
    stt_text: str = Field(..., description="아이가 말한 음성의 원본 STT 텍스트")
    level: str = Field(..., description="아동의 현재 읽기 레벨")
    recent_wcpm: Optional[float] = Field(None, description="최근 WCPM")
    weak_phonemes:  Optional[List[str]] = Field(None, description="취약 발음 리스트")

class Pages(BaseModel):
    page_order: int = Field(..., description="장면 순서")
    image_prompt: str = Field(..., description="장면 이미지 생성을 위한 영어 프롬프트")
    sentences: List[str] = Field(..., description="장면에 속하는 문장 리스트")

class StoryResponse(BaseModel):
    title: str = Field(..., description="동화 제목")
    cover_image_prompt: str = Field(..., description="표지 이미지 생성을 위한 영어 프롬프트")
    pages: List[Pages] = Field(..., description="동화 페이지 리스트")