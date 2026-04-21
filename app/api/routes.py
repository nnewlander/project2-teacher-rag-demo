from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.schemas.answer import AnswerResponse
from app.schemas.query import QueryRequest
from app.services.qa_service import QAService


router = APIRouter()
_qa: QAService | None = None


def get_qa() -> QAService:
    global _qa
    if _qa is None:
        _qa = QAService(get_settings())
    # 最小稳妥兜底：在 reload/热更新或其他路径下，_qa 可能已存在但尚未完成 init_kb，
    # 这会导致 /health 里读取到的 artifacts 为空，从而 faq_count/chunks_count 为 None。
    if not getattr(_qa, "_ready", False) or getattr(_qa, "_artifacts", None) is None:
        _qa.init_kb()
    return _qa


@router.get("/health")
def health() -> dict:
    global _qa
    if _qa is None:
        _qa = QAService(get_settings())
    if not getattr(_qa, "_ready", False) or getattr(_qa, "_artifacts", None) is None:
        limit = int(getattr(_qa.settings, "health_init_limit", 0) or 0)
        _qa.init_kb(limit=limit if limit > 0 else None)
    artifacts = _qa._artifacts  # MVP：健康检查允许读取内部状态
    return {
        "status": "ok",
        "app": _qa.settings.app_name,
        "faq_count": getattr(artifacts, "faq_count", None),
        "chunks_count": getattr(artifacts, "chunks_count", None),
        "health_init_limit": int(getattr(_qa, "_init_limit_used", 0) or 0),
    }


@router.post("/ask", response_model=AnswerResponse)
def ask(req: QueryRequest) -> AnswerResponse:
    qa = get_qa()
    return qa.ask(req.query, top_k=req.top_k)

