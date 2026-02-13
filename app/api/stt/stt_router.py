from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from tempfile import NamedTemporaryFile
import os

from stt.whisper_loader import transcribe_with_timestamps

router = APIRouter(
    prefix="/api/stt",
    tags=["STT"]
)

@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    preset: str = Query("raw", description="raw | balanced | clean"),
):
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"

    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        result = transcribe_with_timestamps(
            audio_path=tmp_path,
            preset=preset,
        )
        result["preset"] = preset
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
