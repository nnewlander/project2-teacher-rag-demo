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
        fallback_ready = True
        lightweight_search_ready = False
        full_rag_ready = False
        serving_mode = "fallback_only"
        return {
            "status": "fallback_only",
            "faq_ready": False,
            "bm25_ready": False,
            "vector_ready": False,
            "model_loaded": False,
            "lightweight_ready": False,
            "fallback_ready": fallback_ready,
            "lightweight_search_ready": lightweight_search_ready,
            "full_rag_ready": full_rag_ready,
            "serving_mode": serving_mode,
            "search_mode": str(get_settings().search_mode or "lightweight"),
            "vector_enabled_for_search": str(get_settings().search_mode or "lightweight").lower() == "hybrid",
            "embedding_model_cached": False,
            "faq_doc_count": 0,
            "bm25_doc_count": 0,
            "last_warmup_error": None,
            "last_warmup_cost_ms": None,
        }
    return _qa.ready_status()


@router.get("/warmup")
def warmup() -> dict:
    qa = get_qa()
    out = qa.warmup_lightweight()
    # 兼容固定返回形状：status + ready 字段 + cost_ms
    if out.get("ok"):
        return {
            "status": "ok",
            "faq_ready": bool(out.get("faq_ready")),
            "bm25_ready": bool(out.get("bm25_ready")),
            "vector_ready": bool(out.get("vector_ready")),
            "model_loaded": bool(out.get("model_loaded")),
            "lightweight_ready": bool(out.get("lightweight_ready")),
            "fallback_ready": bool(out.get("fallback_ready")),
            "lightweight_search_ready": bool(out.get("lightweight_search_ready")),
            "full_rag_ready": bool(out.get("full_rag_ready")),
            "serving_mode": out.get("serving_mode"),
            "search_mode": out.get("search_mode"),
            "vector_enabled_for_search": bool(out.get("vector_enabled_for_search")),
            "embedding_model_cached": bool(out.get("embedding_model_cached")),
            "faq_doc_count": int(out.get("faq_doc_count") or 0),
            "bm25_doc_count": int(out.get("bm25_doc_count") or 0),
            "last_warmup_error": out.get("last_warmup_error"),
            "last_warmup_cost_ms": out.get("last_warmup_cost_ms"),
            "cost_ms": out.get("cost_ms"),
        }
    return {
        "status": "error",
        "faq_ready": bool(out.get("faq_ready")),
        "bm25_ready": bool(out.get("bm25_ready")),
        "vector_ready": bool(out.get("vector_ready")),
        "model_loaded": bool(out.get("model_loaded")),
        "lightweight_ready": bool(out.get("lightweight_ready")),
        "fallback_ready": bool(out.get("fallback_ready")),
        "lightweight_search_ready": bool(out.get("lightweight_search_ready")),
        "full_rag_ready": bool(out.get("full_rag_ready")),
        "serving_mode": out.get("serving_mode"),
        "search_mode": out.get("search_mode"),
        "vector_enabled_for_search": bool(out.get("vector_enabled_for_search")),
        "embedding_model_cached": bool(out.get("embedding_model_cached")),
        "faq_doc_count": int(out.get("faq_doc_count") or 0),
        "bm25_doc_count": int(out.get("bm25_doc_count") or 0),
        "last_warmup_error": out.get("last_warmup_error"),
        "last_warmup_cost_ms": out.get("last_warmup_cost_ms"),
        "cost_ms": out.get("cost_ms"),
        "error": out.get("error"),
    }

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

