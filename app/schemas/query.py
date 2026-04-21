from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="用户问题（中文教育教学场景）")
    top_k: Optional[int] = Field(None, description="可选：覆盖默认召回数量")
    metadata: Dict[str, str] = Field(default_factory=dict, description="可选：请求侧元信息（如 teacher_role/grade_band）")


class RetrievedContext(BaseModel):
    """
    检索结果统一结构，便于后续加 rerank、引用片段、证据链等。
    """

    source: str
    source_id: str
    score: float
    text: str
    metadata: Dict[str, str] = Field(default_factory=dict)

