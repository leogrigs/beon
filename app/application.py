from fastapi import FastAPI

from app.config import Settings
from app.controllers import health, qa


def create_app(settings: Settings | None = None) -> FastAPI:
    application = FastAPI(title="BEON.tech Q&A")
    application.state.settings = settings if settings is not None else Settings.from_environment()
    application.include_router(qa.router, prefix="/api/v1")
    application.include_router(health.router, prefix="/api/v1")
    return application
