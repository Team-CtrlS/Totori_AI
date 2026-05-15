from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

class GenerateStoryRequest(BaseModel):
    stt_text: str = Field(..., description="아이가 말한 음성의 원본 STT 텍스트")
    level: str = Field(..., description="아동의 현재 읽기 레벨")
    recent_wcpm: Optional[float] = Field(None, description="최근 WCPM")
    weak_phonemes:  Optional[List[str]] = Field(None, description="취약 발음 리스트")

class Sentence(BaseModel):
    text: str = Field(..., description="장면에 속하는 문장")
    audio_S3_key: Optional[str] = Field(None, description="TTS 음성 S3 Key")
    duration_ms: Optional[int] = Field(None, description="음성 기간 ms")

class Pages(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    page_order: int = Field(..., description="장면 순서")
    image_prompt: str = Field(..., description="장면 이미지 생성을 위한 영어 프롬프트")
    sentences: list[Sentence] = Field(..., description="장면에 속하는 문장들")

class StoryResponse(BaseModel):
    title: str = Field(..., description="동화 제목")
    cover_image_prompt: str = Field(..., description="표지 이미지 생성을 위한 영어 프롬프트")
    pages: List[Pages] = Field(..., description="동화 페이지 리스트")