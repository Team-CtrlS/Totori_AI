import json
import os
from contextlib import asynccontextmanager
from tempfile import NamedTemporaryFile
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from app.schemas.story_schema import GenerateStoryRequest, StoryResponse
from app.services.story_orchestrator import StoryOrchestratorService
from app.services.whisper_loader import transcribe_with_timestamps

router = APIRouter(
    prefix="/ai/story",
    tags=["Story Generator"]
)

def get_orchestrator():
    return StoryOrchestratorService()

# 동화 생성 파이프라인 + 에러 처리
async def _run_pipeline(orchestrator: StoryOrchestratorService,
                        request: GenerateStoryRequest) -> StoryResponse:
    try:
        return await orchestrator.run_pipeline(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"동화 생성 중 알 수 없는 오류 발생: {str(e)}")
    
# 음성 임시 파일 생성
@asynccontextmanager
async def _save_audio_to_tempfile(file: UploadFile):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())
    try:
        yield tmp_path
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@router.post("/generate", response_model=StoryResponse)
async def generate_story(
    request: GenerateStoryRequest,
    orchestrator: StoryOrchestratorService = Depends(get_orchestrator)
):
    return await _run_pipeline(orchestrator, request)

@router.post("/make", response_model=StoryResponse)
async def generate_story_from_audio(
    file: UploadFile = File(...),
    level: str = Form(...),
    recent_wcpm: Optional[float] = Form(None),
    weak_phonemes: Optional[str] = Form(None),
    orchestrator: StoryOrchestratorService = Depends(get_orchestrator)
):
    async with _save_audio_to_tempfile(file) as tmp_path:
        try:
            stt_result = transcribe_with_timestamps(audio_path=tmp_path, preset="balanced")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
    stt_text = stt_result.get("text", "").strip()
    if not stt_result:
        raise HTTPException(status_code=422, detail="STT 변환 결과가 비어있습니다.")
    
    parsed_weak_phonemes = None
    if weak_phonemes:
        try:
            parsed_weak_phonemes = json.loads(weak_phonemes)
        except json.JSONDecodeError:
            parsed_weak_phonemes = None
    
    story_request = GenerateStoryRequest(
        stt_text=stt_text,
        level=level,
        recent_wcpm=recent_wcpm,
        weak_phonemes=parsed_weak_phonemes
    )
    return await _run_pipeline(orchestrator, story_request)