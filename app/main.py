import uuid
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import health, synthesize, audio_serve, stream
from app.core.exceptions import EmpathyEngineException, custom_exception_handler
from app.core.logging_config import logger, request_id_var

def create_app() -> FastAPI:
    app = FastAPI(title="Empathy Engine API", version="1.0.0")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Context variable middleware for request ids
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        req_id = str(uuid.uuid4())
        token = request_id_var.set(req_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        request_id_var.reset(token)
        return response

    # Mount API routers
    app.include_router(health.router, prefix="/api/v1", tags=["system"])
    app.include_router(synthesize.router, prefix="/api/v1", tags=["speech"])
    app.include_router(audio_serve.router, prefix="/api/v1", tags=["audio"])
    app.include_router(stream.router, prefix="/api/v1", tags=["streaming"])

    # Exception Handling
    app.add_exception_handler(EmpathyEngineException, custom_exception_handler)
    app.add_exception_handler(Exception, custom_exception_handler)

    # Static and Templates
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
    templates = Jinja2Templates(directory="app/templates")

    @app.get("/")
    async def root(request: Request):
        return templates.TemplateResponse(request=request, name="index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse("app/static/assets/favicon.ico", media_type="image/x-icon")

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    from app.core.config import settings
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
