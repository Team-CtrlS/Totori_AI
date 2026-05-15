import os
from contextlib import asynccontextmanager
from tempfile import NamedTemporaryFile
from fastapi import APIRouter, File, Form, HTTPException, Path, UploadFile, Response
from app.schemas.reading_schema import CompleteResponse
from app.services.reading_service import get_reading_service

router = APIRouter(
    prefix="/ai/reading", 
    tags=["Reading Analyzer"])

_service = get_reading_service()

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

# 낭독 음성 분석
@router.post("/analyze", status_code=204)
async def analyze_reading(
    file: UploadFile = File(..., description="낭독 음성 파일 (wav/m4a/mp3)"),
    original_text: str = Form(..., description="현재 페이지 원문 텍스트"),
    child_id: int = Form(..., description="아동 ID"),
    book_id: int = Form(..., description="동화 ID"),
    level: str = Form(..., description="아동 읽기 레벨 (L1~L6)"),
):
    async with _save_audio_to_tempfile(file) as tmp_path:
        try:
            await _service.analyze_and_store(
                audio_path=tmp_path,
                original_text=original_text,
                child_id=child_id,
                book_id=book_id,
                level=level,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"낭독 분석 중 오류 발생: {str(e)}")

# 동화 완료 시 누적 오류 및 wcpm 전체 반환 및 redis 삭제
@router.post("/complete/{book_id}", response_model=CompleteResponse)
async def complete_reading(
    child_id: int = Path(..., description="아동 ID"),
    book_id: int = Path(..., description="동화 ID"),
):
    try:
        result = await _service.get_all_and_delete(child_id, book_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 패턴 조회 중 오류 발생: {str(e)}")
    
    return CompleteResponse(
        child_id=child_id,
        book_id=book_id,
        total_errors=len(result["errors"]),
        errors=result["errors"],
        avg_wcpm=result["avg_wcpm"],
    )