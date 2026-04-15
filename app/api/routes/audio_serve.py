import mimetypes
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings

router = APIRouter()

_FAVICON_PATH = os.path.join("app", "static", "assets", "favicon.ico")


@router.get("/audio/{filename}")
def get_audio(filename: str):
    """Serve a generated audio file (mp3 or wav)."""
    file_path = os.path.join(settings.OUTPUT_AUDIO_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")

    media_type, _ = mimetypes.guess_type(file_path)
    return FileResponse(file_path, media_type=media_type or "audio/mpeg")
