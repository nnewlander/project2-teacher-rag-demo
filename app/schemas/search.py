from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description="检索问题文本")
    top_k: int = Field(3, ge=1, le=20, description="返回证据条数")
    filters: Dict[str, Any] = Field(default_factory=dict, description="预留过滤条件")
    request_id: Optional[str] = Field(None, description="调用方请求 ID（可选）")


class SearchHit(BaseModel):
    source_id: str
    title: str
    snippet: str
    score: float
    source_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    hits: List[SearchHit] = Field(default_factory=list)
    query: str
    route_trace: List[str] = Field(default_factory=list)
    debug: Dict[str, Any] = Field(default_factory=dict)
