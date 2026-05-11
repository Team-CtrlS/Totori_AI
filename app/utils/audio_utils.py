import os
from contextlib import asynccontextmanager
from tempfile import NamedTemporaryFile

from fastapi import UploadFile

@asynccontextmanager
async def save_audio_to_tempfile(file: UploadFile):
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