from fastapi import HTTPException
from fastapi.responses import JSONResponse
from app.core.logging_config import logger

class EmpathyEngineException(Exception):
    """Base exception for Empathy Engine"""
    pass

class ModelInferenceError(EmpathyEngineException):
    pass

class TTSGenerationError(EmpathyEngineException):
    pass

def custom_exception_handler(request, exc: Exception):
    error_msg = str(exc)
    logger.error(f"Application error: {error_msg}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "An internal server error occurred", "details": error_msg}
    )

def http_exception_handler(request, exc: HTTPException):
    logger.error(f"HTTP warning/error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail}
    )
