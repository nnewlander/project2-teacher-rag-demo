from __future__ import annotations

from app.config import get_settings
from app.schemas.query import RetrievedContext
from app.services.hybrid_retriever import HybridResult
from app.services.qa_service import QAService
from app.services.router import RouteDecision
from app.services.subquery_builder import SubqueryBuilder


def test_subquery_builder_split() -> None:
    b = SubqueryBuilder()
    q = "登录一直失败，同时作业发布入口在哪里？另外学生端显示权限不足怎么办？"
    r = b.build(q)
    assert len(r.subqueries) >= 2


class _DummyRouter:
    def decide(self, query: str, *, query_type: str | None = None) -> RouteDecision:  # noqa: ARG002
        return RouteDecision(route="subquery", reason="test_force", eval_label="subquery")


class _DummyHybrid:
    def __init__(self) -> None:
        self.queries = []

    def retrieve(self, query: str, faq_top_k: int, vec_top_k: int, hybrid_top_k: int) -> HybridResult:  # noqa: ARG002
        self.queries.append(query)
        # 用 query 作为 source_id，便于验证去重/合并
        ctx = RetrievedContext(source="chunk", source_id=f"c::{query}", score=1.0, text=query, metadata={"source": "document"})
        return HybridResult(contexts=[ctx], debug={"timing_ms": {"faq": 0.0, "vector": 0.0, "rerank": 0.0}, "rerank": {"enabled": False}})


def test_subquery_route_trace_smoke() -> None:
    settings = get_settings()
    qa = QAService(settings)
    qa._ready = True  # 跳过 init_kb
    qa.router = _DummyRouter()  # type: ignore[assignment]
    qa.hybrid = _DummyHybrid()  # type: ignore[assignment]
    qa.settings.backtrack_enabled = False

    q = "登录一直失败，同时作业发布入口在哪里？"
    resp = qa.ask(q, top_k=3)

    steps = [x.step for x in resp.route_trace]
    assert "subquery.split" in steps
    assert "subquery.retrieve" in steps
    assert resp.route == "subquery"

