from pydantic import BaseModel, Field
from typing import List, Optional

class GenerateStoryRequest(BaseModel):
    stt_text: str = Field(..., description="아이가 말한 음성의 원본 STT 텍스트")
    level: str = Field(..., description="아동의 현재 읽기 레벨")
    recent_wcpm: Optional[float] = Field(None, description="최근 WCPM")
    weak_phonemes:  Optional[List[str]] = Field(None, description="취약 발음 리스트")

class Scene(BaseModel):
    pageOrder: int = Field(..., description="페이지 순서")
    image_prompt: str = Field(..., description="장면 이미지 생성을 위한 영어 프롬프트")
    sentences: List[str] = Field(..., description="장면에 속하는 문장 리스트")

class StoryResponse(BaseModel):
    title: str = Field(..., description="동화 제목")
    pages: List[Scene] = Field(..., description="동화를 구성하는 장면 리스트")