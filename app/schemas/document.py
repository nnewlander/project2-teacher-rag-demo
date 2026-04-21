from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


SourceType = Literal["document", "support_ticket", "faq", "code_example"]


class RawRecord(BaseModel):
    """
    原始输入层记录（jsonl 一行），以统一结构承载。
    """

    source: SourceType
    source_id: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class InternalDocument(BaseModel):
    """
    系统内部统一文档结构：后续 OCR/解析/结构化都在这里扩展。
    """

    doc_id: str
    source: SourceType
    title: str = ""
    text: str
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """
    切分后的 chunk。
    预留 parent_id / hierarchy 以支持父子块、段落级回溯。
    """

    chunk_id: str
    doc_id: str
    text: str
    start: int = 0
    end: int = 0
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

