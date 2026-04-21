from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Tuple

from app.schemas.query import RetrievedContext
from app.services.faq_retriever import FaqRetriever
from app.services.reranker import KeywordOverlapReranker, Reranker
from app.services.vector_retriever import VectorRetriever


@dataclass(frozen=True)
class HybridResult:
    contexts: List[RetrievedContext]
    debug: dict


class HybridRetriever:
    """
    组合 FAQ 与向量召回（MVP）。

    约定：
    - 先取 FAQ top_k 候选
    - 再取向量 top_k 候选
    - 合并去重并截断为 hybrid_top_k

    TODO:
    - 替换 rerank 为 cross-encoder（如 bge-reranker）
    - 加权融合（BM25/embedding score normalization）
    - 支持多路召回与策略路由（HyDE/子查询）
    """

    def __init__(self, faq: FaqRetriever, vec: VectorRetriever, reranker: Reranker | None = None) -> None:
        self.faq = faq
        self.vec = vec
        self.reranker = reranker or KeywordOverlapReranker()

    def retrieve(self, query: str, faq_top_k: int, vec_top_k: int, hybrid_top_k: int) -> HybridResult:
        t0 = time.perf_counter()
        faq_hits = self.faq.search(query, top_k=faq_top_k)
        t_faq = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        vec_hits = self.vec.search(query, top_k=vec_top_k)
        t_vec = (time.perf_counter() - t1) * 1000

        contexts: List[RetrievedContext] = []
        contexts.extend(self.faq.as_contexts(faq_hits))
        contexts.extend(self.vec.as_contexts(vec_hits))

        # 轻量去重：按 (source, source_id)
        uniq = {}
        for c in contexts:
            uniq[(c.source, c.source_id)] = c

        # 简单排序：分数降序
        merged = list(uniq.values())
        merged.sort(key=lambda x: x.score, reverse=True)
        pre_rerank = merged[: max(1, hybrid_top_k)]

        # rerank：对“合并后的候选”重排（轻量版本可运行，后续可替换为 cross-encoder）
        t2 = time.perf_counter()
        reranked = self.reranker.rerank(query, pre_rerank) if self.reranker else pre_rerank
        t_rerank = (time.perf_counter() - t2) * 1000
        merged = reranked[: max(1, hybrid_top_k)]

        debug = {
            "faq_candidates": [{"faq_id": h.faq_id, "score": h.score} for h in faq_hits[:5]],
            "vector_candidates": [{"chunk_id": h.chunk_id, "score": h.score} for h in vec_hits[:5]],
            "rerank": {"enabled": bool(self.reranker), "type": type(self.reranker).__name__ if self.reranker else None},
            "timing_ms": {"faq": round(t_faq, 2), "vector": round(t_vec, 2), "rerank": round(t_rerank, 2)},
        }
        return HybridResult(contexts=merged, debug=debug)

