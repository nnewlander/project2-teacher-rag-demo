from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.schemas.answer import AnswerResponse
from app.schemas.query import QueryRequest
from app.schemas.search import SearchRequest, SearchResponse
from app.services.qa_service import QAService


router = APIRouter()
_qa: QAService | None = None


def get_qa() -> QAService:
    global _qa
    if _qa is None:
        _qa = QAService(get_settings())
    return _qa


@router.get("/health")
def health() -> dict:
    # 轻量健康检查：不触发任何重资源初始化
    return {"status": "ok", "service": "project2_rag"}


@router.get("/ready")
def ready() -> dict:
    global _qa
    if _qa is None:
        return {
            "status": "not_ready",
            "faq_ready": False,
            "bm25_ready": False,
            "vector_ready": False,
            "model_loaded": False,
            "lightweight_ready": True,
            "fallback_ready": True,
        }
    return _qa.ready_status()


@router.post("/ask", response_model=AnswerResponse)
def ask(req: QueryRequest) -> AnswerResponse:
    qa = get_qa()
    return qa.ask(req.query, top_k=req.top_k)


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    qa = get_qa()
    return qa.search(
        query=req.query,
        top_k=req.top_k,
        filters=req.filters,
        request_id=req.request_id,
    )

