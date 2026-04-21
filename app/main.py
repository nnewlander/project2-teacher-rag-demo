from __future__ import annotations

from fastapi import FastAPI

from app.api.docs_zh import router as docs_router
from app.api.routes import router
from app.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="教师智能知识库问答系统（RAG）最小可运行版本（MVP）。",
        docs_url=None,  # 用中文引导版 /docs 替换默认英文 UI
        redoc_url=None,
    )
    app.include_router(router)
    app.include_router(docs_router)
    return app


app = create_app()

