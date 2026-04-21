from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi

from app.schemas.document import InternalDocument
from app.schemas.query import RetrievedContext
from app.utils.text_utils import normalize_text, simple_tokenize_zh


@dataclass(frozen=True)
class FaqHit:
    faq_id: str
    score: float
    question: str
    answer: str
    category: str = ""


class FaqRetriever:
    """
    FAQ 直达检索（BM25）。

    MVP 行为：
    - 只索引 standard_question（或 question）
    - 返回 top_k 候选 + 分数

    TODO:
    - 增加别名/同义词（alias_question）
    - 多字段索引（question + category + tags）
    - 支持热度/人工审核状态加权（hit_count_30d, status）
    """

    def __init__(self) -> None:
        self._faq_docs: List[InternalDocument] = []
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_tokens: List[List[str]] = []

    def build(self, faq_docs: List[InternalDocument]) -> None:
        self._faq_docs = [d for d in faq_docs if d.source == "faq"]
        questions = [self._extract_question(d) for d in self._faq_docs]
        self._corpus_tokens = [simple_tokenize_zh(q) for q in questions]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._faq_docs else None

    def search(self, query: str, top_k: int = 5) -> List[FaqHit]:
        query = normalize_text(query)
        if not query or self._bm25 is None:
            return []

        q_tokens = simple_tokenize_zh(query)
        scores = self._bm25.get_scores(q_tokens)

        scored: List[Tuple[int, float]] = [(i, float(s)) for i, s in enumerate(scores)]
        scored.sort(key=lambda x: x[1], reverse=True)

        hits: List[FaqHit] = []
        for i, s in scored[: max(1, top_k)]:
            d = self._faq_docs[i]
            q = self._extract_question(d)
            a = self._extract_answer(d)
            hits.append(
                FaqHit(
                    faq_id=d.doc_id,
                    score=s,
                    question=q,
                    answer=a,
                    category=str(d.metadata.get("category") or ""),
                )
            )
        return hits

    def as_contexts(self, hits: List[FaqHit]) -> List[RetrievedContext]:
        out: List[RetrievedContext] = []
        for h in hits:
            text = f"FAQ 问：{h.question}\nFAQ 答：{h.answer}"
            out.append(
                RetrievedContext(
                    source="faq",
                    source_id=h.faq_id,
                    score=h.score,
                    text=text,
                    metadata={"category": h.category},
                )
            )
        return out

    def _extract_question(self, d: InternalDocument) -> str:
        q = d.metadata.get("question") if isinstance(d.metadata, dict) else None
        if q:
            return normalize_text(str(q))
        return normalize_text(d.title or "")

    def _extract_answer(self, d: InternalDocument) -> str:
        a = d.metadata.get("answer") if isinstance(d.metadata, dict) else None
        if a:
            return normalize_text(str(a))

        # 兜底：尝试从 text 中拆分（注意 cleaner 可能会压平换行）
        t = d.text or ""
        parts = t.split("\n\n", 1)
        if len(parts) == 2:
            return normalize_text(parts[1])
        return ""

