from fastapi import APIRouter
from fastapi.responses import FileResponse
from app.core.config import settings
import os

router = APIRouter()

@router.get("/audio/{filename}")
def get_audio(filename: str):
    file_path = os.path.join(settings.OUTPUT_AUDIO_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg")
    return {"success": False, "error": "File not found"}, 404
