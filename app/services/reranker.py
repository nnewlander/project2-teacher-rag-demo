from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Protocol, Tuple

import numpy as np

from app.schemas.query import RetrievedContext
from app.utils.text_utils import simple_tokenize_zh

if TYPE_CHECKING:
    from app.config import Settings

_log = logging.getLogger("kbqa.rerank")

def _normalize_device(raw: str) -> str:
    x = (raw or "").strip().lower()
    if x in ("auto", ""):
        return "auto"
    if x in ("cuda", "gpu"):
        return "cuda"
    if x in ("cpu",):
        return "cpu"
    return x


def _normalize_bool_auto(raw: str) -> str:
    x = (raw or "").strip().lower()
    if x in ("", "auto"):
        return "auto"
    if x in ("1", "true", "yes", "y", "on"):
        return "true"
    if x in ("0", "false", "no", "n", "off"):
        return "false"
    return x


def _pick_device_and_fp16(settings: "Settings") -> tuple[str, bool]:
    req_device = _normalize_device(str(getattr(settings, "device", "auto") or "auto"))
    req_fp16 = _normalize_bool_auto(str(getattr(settings, "use_fp16", "auto") or "auto"))

    try:
        import torch  # type: ignore
    except Exception:
        dev = "cpu" if req_device == "auto" else req_device
        fp16 = False if req_fp16 in ("auto", "false") else True
        return dev, fp16

    cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    if req_device == "auto":
        dev = "cuda" if cuda_ok else "cpu"
    else:
        dev = req_device

    if req_fp16 == "auto":
        fp16 = dev == "cuda"
    else:
        fp16 = req_fp16 == "true"
    return dev, fp16


class Reranker(Protocol):
    """
    Rerank 后端：keyword_overlap_reranker | bge_reranker（FlagReranker）| cross_encoder（CrossEncoder）。
    """

    def rerank(self, query: str, contexts: List[RetrievedContext]) -> List[RetrievedContext]: ...


def normalize_reranker_backend(raw: str) -> str:
    x = (raw or "").strip().lower().replace("-", "_")
    if x in ("keyword_overlap", "keyword_overlap_reranker", "overlap", "kw_overlap"):
        return "keyword_overlap_reranker"
    if x in ("bge_reranker", "bge_rerank", "flag_reranker"):
        return "bge_reranker"
    if x in ("cross_encoder", "crossencoder", "ce_reranker"):
        return "cross_encoder"
    return x


def create_reranker_from_settings(settings: "Settings") -> Reranker:
    backend = normalize_reranker_backend(str(getattr(settings, "reranker_backend", "keyword_overlap_reranker") or ""))
    if backend == "keyword_overlap_reranker":
        return KeywordOverlapReranker()

    bs = int(getattr(settings, "reranker_batch_size", 16) or 16)
    fallback = bool(getattr(settings, "reranker_fallback_on_error", True))

    if backend == "bge_reranker":
        model_name = (getattr(settings, "reranker_model_name", "") or "").strip() or "BAAI/bge-reranker-v2-m3"
        try:
            from FlagEmbedding import FlagReranker  # type: ignore
            device, fp16 = _pick_device_and_fp16(settings)
            try:
                rr = FlagReranker(model_name, use_fp16=fp16, device=device)
            except TypeError:
                rr = FlagReranker(model_name, use_fp16=fp16)
            return FlagBgeReranker(rr, batch_size=max(1, bs))
        except Exception as e:
            if fallback:
                _log.warning(
                    "bge_reranker 初始化失败，回退到 keyword_overlap_reranker（原因: %s）。"
                    "安装 FlagEmbedding、配置有效 reranker_model_name，或设置 KBQA_RERANKER_FALLBACK_ON_ERROR=false 以强制失败。",
                    e,
                )
                return KeywordOverlapReranker()
            raise RuntimeError(
                f"reranker_backend=bge_reranker 初始化失败（model={model_name!r}）。"
                "请安装 FlagEmbedding 并检查模型下载与显存/内存。"
            ) from e

    if backend == "cross_encoder":
        model_name = (getattr(settings, "reranker_model_name", "") or "").strip() or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            return CrossEncoderReranker(CrossEncoder(model_name), batch_size=max(1, bs))
        except Exception as e:
            if fallback:
                _log.warning(
                    "cross_encoder 初始化失败，回退到 keyword_overlap_reranker（原因: %s）。",
                    e,
                )
                return KeywordOverlapReranker()
            raise RuntimeError(
                f"reranker_backend=cross_encoder 初始化失败（model={model_name!r}）。请检查 sentence-transformers 与模型名。"
            ) from e

    raise RuntimeError(
        f"不支持的 reranker_backend={getattr(settings, 'reranker_backend', None)!r}（规范化后={backend!r}）。"
    )


@dataclass(frozen=True)
class KeywordOverlapReranker:
    overlap_weight: float = 1.0
    base_score_weight: float = 0.05

    def rerank(self, query: str, contexts: List[RetrievedContext]) -> List[RetrievedContext]:
        if not query or not contexts:
            return contexts

        q_tokens = set(simple_tokenize_zh(query))
        if not q_tokens:
            return contexts

        scored: List[Tuple[float, RetrievedContext]] = []
        for c in contexts:
            t_tokens = set(simple_tokenize_zh(c.text or ""))
            if not t_tokens:
                overlap = 0.0
            else:
                inter = len(q_tokens & t_tokens)
                overlap = inter / max(1, len(q_tokens))

            s = float(self.overlap_weight * overlap + self.base_score_weight * float(c.score))
            scored.append((s, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]


@dataclass(frozen=True)
class FlagBgeReranker:
    """FlagEmbedding.FlagReranker，贴近项目二 BGE reranker 口径。"""

    model: Any
    batch_size: int = 16

    def rerank(self, query: str, contexts: List[RetrievedContext]) -> List[RetrievedContext]:
        if not query or not contexts:
            return contexts
        pairs: List[List[str]] = [[query, (c.text or "")[:8000]] for c in contexts]
        try:
            scores = self.model.compute_score(pairs, batch_size=self.batch_size)
        except TypeError:
            scores = self.model.compute_score(pairs)
        arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        scores = [float(x) for x in arr.tolist()]
        if len(scores) != len(contexts):
            raise RuntimeError(f"FlagReranker 返回分数条数 {len(scores)} 与候选 {len(contexts)} 不一致。")
        scored = list(zip(scores, contexts))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]


@dataclass(frozen=True)
class CrossEncoderReranker:
    """sentence-transformers CrossEncoder。"""

    model: Any
    batch_size: int = 16

    def rerank(self, query: str, contexts: List[RetrievedContext]) -> List[RetrievedContext]:
        if not query or not contexts:
            return contexts
        pairs = [(query, (c.text or "")[:8000]) for c in contexts]
        raw = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        scores = [float(x) for x in arr.tolist()]
        if len(scores) != len(contexts):
            raise RuntimeError(f"CrossEncoder 返回分数条数 {len(scores)} 与候选 {len(contexts)} 不一致。")
        scored = list(zip(scores, contexts))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]
