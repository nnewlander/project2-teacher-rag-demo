"""
检索结果进入生成前的证据过滤（与 qa_service 调用约定一致）。

当前为最小可运行实现：默认原样保留 contexts；关闭开关时返回 disabled 标记。
后续可在本文件扩展规则，无需改 qa_service 主结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

from app.schemas.query import RetrievedContext

if TYPE_CHECKING:
    from app.config import Settings


@dataclass
class EvidenceFilterResult:
    kept: List[RetrievedContext]
    filtered_out_count: int
    kept_context_count: int
    by_reason: Dict[str, int] = field(default_factory=dict)
    relaxed_kept_top1: bool = False


def filter_evidence(*, query: str, contexts: List[RetrievedContext], settings: "Settings") -> EvidenceFilterResult:
    """
    输入：
    - query：当前用户问题（保留参数供后续规则使用）
    - contexts：待过滤的 RetrievedContext 列表
    - settings：含 evidence_filter_enabled 等

    输出：
    - EvidenceFilterResult：kept、filtered_out_count、kept_context_count、by_reason、relaxed_kept_top1
    """
    _ = query
    if not bool(getattr(settings, "evidence_filter_enabled", True)):
        return EvidenceFilterResult(
            kept=list(contexts),
            filtered_out_count=0,
            kept_context_count=len(contexts),
            by_reason={"disabled": 1},
            relaxed_kept_top1=False,
        )

    # 最小行为：不丢条，仅保证与调用方字段兼容
    return EvidenceFilterResult(
        kept=list(contexts),
        filtered_out_count=0,
        kept_context_count=len(contexts),
        by_reason={},
        relaxed_kept_top1=False,
    )
