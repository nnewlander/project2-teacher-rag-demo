from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from app.schemas.document import DocumentChunk
from app.schemas.query import RetrievedContext
from app.config import get_settings
from app.utils.text_utils import normalize_text

_logger = logging.getLogger("kbqa.vector")

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
    """
    返回：auto / true / false
    """
    x = (raw or "").strip().lower()
    if x in ("", "auto"):
        return "auto"
    if x in ("1", "true", "yes", "y", "on"):
        return "true"
    if x in ("0", "false", "no", "n", "off"):
        return "false"
    return x


def _pick_device_and_fp16(settings: Any) -> tuple[str, bool]:
    """
    - device=auto：torch.cuda.is_available() ? cuda : cpu
    - use_fp16=auto：cuda->True, cpu->False
    """
    req_device = _normalize_device(str(getattr(settings, "device", "auto") or "auto"))
    req_fp16 = _normalize_bool_auto(str(getattr(settings, "use_fp16", "auto") or "auto"))

    try:
        import torch  # type: ignore
    except Exception:
        # 无 torch 时（或导入失败），退回 cpu + fp16=False，确保 CPU 环境仍可运行 ST 路径
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


def _normalize_embedding_backend(raw: str) -> str:
    x = (raw or "").strip().lower().replace("-", "_")
    if x in ("st", "sentence_transformers"):
        return "sentence_transformers"
    if x in ("bge_m3", "bge_m3_flagembedding", "bge_m3_dense"):
        return "bge_m3"
    return x


def _normalize_vector_store_backend(raw: str) -> str:
    x = (raw or "").strip().lower().replace("-", "_")
    if x in ("faiss", "faiss_cpu", "faiss_flat_ip"):
        return "faiss"
    if x in ("milvus", "milvus_lite", "zilliz"):
        return "milvus"
    if x in ("memory", "in_memory", "numpy"):
        return "memory"
    return x


def _l2_normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, str]


class VectorRetriever:
    """
    向量检索：`embedding_backend`（sentence_transformers | bge_m3）+ FAISS。
    bge_m3 使用 FlagEmbedding 的 BGEM3FlagModel（dense 向量）；失败时可按配置回退到 ST。
    """

    def __init__(self) -> None:
        self._chunks: List[DocumentChunk] = []
        self._index = None
        self._dim: Optional[int] = None
        self._model: Any = None
        self._active_embedder: Optional[str] = None

        settings = get_settings()
        self._requested_backend = _normalize_embedding_backend(
            str(getattr(settings, "embedding_backend", "sentence_transformers") or "sentence_transformers")
        )
        self._embedding_backend = self._requested_backend
        self._vector_store_backend = _normalize_vector_store_backend(
            str(getattr(settings, "vector_store_backend", "faiss") or "faiss")
        )
        self._model_name = (settings.embedding_model_name or "").strip()
        self._batch_size = int(settings.embedding_batch_size)
        self._settings = settings

    def build(self, chunks: Iterable[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        if not self._chunks:
            self._index = None
            self._dim = None
            return

        if self._vector_store_backend == "milvus":
            raise RuntimeError(
                "当前 vector_store_backend=milvus 尚未接入（项目二完整版需 Milvus SDK 与集合 schema）。"
                "请保持 vector_store_backend=faiss，或在 VectorRetriever 中实现 Milvus 分支后再切换。"
            )
        if self._vector_store_backend not in ("faiss",):
            raise RuntimeError(
                f"不支持的 vector_store_backend={self._vector_store_backend!r}。"
                "默认可用值：faiss；预留：milvus（未实现）。"
            )

        self._ensure_model()
        embeddings = self._encode_texts([c.text for c in self._chunks])
        self._dim = int(embeddings.shape[1])

        self._index = self._build_faiss_index(embeddings)

    def _build_faiss_index(self, embeddings: np.ndarray):
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(self._dim or int(embeddings.shape[1]))
        index.add(embeddings)
        return index

    def search(self, query: str, top_k: int = 6) -> List[VectorHit]:
        query = normalize_text(query)
        if not query or not self._chunks or self._index is None:
            return []

        self._ensure_model()
        q = self._encode_texts([query])

        k = max(1, int(top_k))
        scores, idx = self._index.search(q, k)  # type: ignore[attr-defined]
        scored: List[Tuple[int, float]] = []
        for j in range(idx.shape[1]):
            i = int(idx[0, j])
            if i < 0 or i >= len(self._chunks):
                continue
            s = float(scores[0, j]) * 10.0
            scored.append((i, s))

        hits: List[VectorHit] = []
        for i, s in scored[:k]:
            c = self._chunks[i]
            hits.append(
                VectorHit(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    score=s,
                    text=c.text,
                    metadata={k: str(v) for k, v in (c.metadata or {}).items()},
                )
            )
        return hits

    def as_contexts(self, hits: List[VectorHit]) -> List[RetrievedContext]:
        out: List[RetrievedContext] = []
        for h in hits:
            out.append(
                RetrievedContext(
                    source="chunk",
                    source_id=h.chunk_id,
                    score=h.score,
                    text=h.text,
                    metadata={"doc_id": h.doc_id, **(h.metadata or {})},
                )
            )
        return out

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        s = self._settings
        if self._embedding_backend == "bge_m3":
            try:
                self._load_bge_m3()
                self._active_embedder = "bge_m3"
                return
            except Exception as e:
                if bool(getattr(s, "embedding_fallback_on_error", True)):
                    _logger.warning(
                        "BGE-M3 embedding 初始化失败，将回退到 sentence_transformers（原因: %s）。"
                        "若需强制使用 BGE-M3，请安装 FlagEmbedding、检查网络/缓存模型，并设置 KBQA_EMBEDDING_FALLBACK_ON_ERROR=false。",
                        e,
                    )
                    self._model_name = str(
                        getattr(s, "embedding_fallback_model_name", None)
                        or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    )
                    self._load_sentence_transformers()
                    self._active_embedder = "sentence_transformers"
                    self._embedding_backend = "sentence_transformers"
                    return
                raise RuntimeError(
                    "embedding_backend=bge_m3 初始化失败，且已关闭 embedding_fallback_on_error。"
                    "请检查是否已 pip install FlagEmbedding、模型名与下载环境。"
                ) from e

        if self._embedding_backend not in ("sentence_transformers",):
            raise RuntimeError(
                f"不支持的 embedding_backend={self._requested_backend!r}（规范化后={self._embedding_backend!r}）。"
            )
        self._load_sentence_transformers()
        self._active_embedder = "sentence_transformers"

    def _load_sentence_transformers(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ModuleNotFoundError as e:  # pragma: no cover
            raise RuntimeError("缺少依赖 sentence-transformers，请先 pip install -r requirements.txt") from e
        name = self._model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self._model = SentenceTransformer(name)

    def _load_bge_m3(self) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "embedding_backend=bge_m3 需要安装 FlagEmbedding（pip install FlagEmbedding）。"
            ) from e
        name = (self._model_name or "").strip() or "BAAI/bge-m3"
        device, fp16 = _pick_device_and_fp16(self._settings)
        # BGEM3FlagModel 的参数随版本可能不同：优先传 device/use_fp16，失败则降级到旧签名
        try:
            self._model = BGEM3FlagModel(name, device=device, use_fp16=fp16)
        except TypeError:
            self._model = BGEM3FlagModel(name, use_fp16=fp16)

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if self._model is None:
            self._ensure_model()
        assert self._active_embedder is not None

        if self._active_embedder == "bge_m3":
            try:
                out = self._model.encode(
                    texts,
                    batch_size=self._batch_size,
                    max_length=8192,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
            except TypeError:
                out = self._model.encode(texts, batch_size=self._batch_size, max_length=8192)
            if isinstance(out, dict):
                dense = out.get("dense_vecs")
                if dense is None:
                    raise RuntimeError("BGE-M3 encode 未返回 dense_vecs，请检查 FlagEmbedding 版本。")
            else:
                dense = out
            emb = _l2_normalize_rows(np.asarray(dense, dtype="float32"))
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            return emb

        emb = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        emb = np.asarray(emb, dtype="float32")
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        return emb
