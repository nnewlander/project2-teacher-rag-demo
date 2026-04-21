from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.query import RetrievedContext


AnswerMode = Literal["faq", "rag_mock"]


class Citation(BaseModel):
    """
    可解释性证据条目（尽量稳定、便于前端展示）。
    """

    source_id: str
    source_type: str
    title: str = ""
    score: float = 0.0
    parent_id: Optional[str] = None
    snippet: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RouteTraceItem(BaseModel):
    """
    路由/检索链路追踪（不改变业务逻辑，只用于解释性输出）。
    """

    step: str
    detail: Dict[str, Any] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    query: str
    mode: AnswerMode = Field(..., description="命中 FAQ 或走 RAG(当前为 mock 生成)")
    route: Optional[str] = Field(None, description="路由结果：bm25_faq / rag_standard / need_clarify 等")
    answer: str
    contexts: List[RetrievedContext] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list, description="结构化证据链（推荐前端展示用）")
    matched_sources: List[Citation] = Field(default_factory=list, description="citations 的别名，保持兼容/便于理解")
    route_trace: List[RouteTraceItem] = Field(default_factory=list, description="路由与检索过程的可解释追踪")
    clarifications: List[str] = Field(default_factory=list, description="当需要澄清时，返回建议补充的问题清单")
    faq_id: Optional[str] = None
    filtered_out_count: Optional[int] = Field(
        None, description="证据过滤阶段丢弃的候选条数（检索+rerkank 之后）"
    )
    kept_context_count: Optional[int] = Field(None, description="过滤后保留的检索证据条数")
    debug: dict = Field(default_factory=dict)

