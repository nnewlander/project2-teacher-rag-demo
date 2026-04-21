"""
GPU 小规模严格验证（只验证 embedding + 小索引构建，不走 FAQ/LLM/RAG 全链路）。

目标：
- 从现有数据源抽样 limit 条 raw_records（document + support_ticket）
- 走 DataLoader -> Cleaner -> Chunker -> VectorRetriever.build
- 使用 Settings / 环境变量中的 device/fp16/embedding_backend/embedding_model/batch_size
- 打印：torch/cuda、picked_device、picked_use_fp16、文档数、chunk 数、各阶段耗时、总耗时

用法（建议在 strict GPU 环境中）：
  python -m scripts.validate_small_kb_gpu --limit 300 --batch-size 4

提示（GTX 1650 Ti 显存紧张）：
- 默认 batch-size=4；可改为 2/4/8 逐步尝试
- limit 建议 200~500 之间先验证跑通
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple


def _pick_device_and_fp16(device: str, use_fp16: str) -> Tuple[str, bool]:
    dev_req = (device or "auto").strip().lower()
    fp16_req = (use_fp16 or "auto").strip().lower()

    import torch  # type: ignore

    cuda_ok = bool(torch.cuda.is_available())
    if dev_req in ("", "auto"):
        dev = "cuda" if cuda_ok else "cpu"
    else:
        dev = dev_req

    if fp16_req in ("", "auto"):
        fp16 = dev == "cuda"
    else:
        fp16 = fp16_req in ("1", "true", "yes", "y", "on")
    return dev, fp16


def main() -> int:
    ap = argparse.ArgumentParser(description="Small KB GPU validation (embedding only)")
    ap.add_argument("--limit", type=int, default=300, help="抽样 raw_records 上限（推荐 200~500）")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="embedding batch size（GTX1650Ti 建议 2/4/8；默认 4）",
    )
    ap.add_argument("--documents-only", action="store_true", help="仅使用 raw_documents，不加载 support_ticket")
    args = ap.parse_args()

    # 允许脚本级覆盖 batch size（不改业务代码）：通过 env 覆盖 Settings.embedding_batch_size
    os.environ["KBQA_EMBEDDING_BATCH_SIZE"] = str(int(args.batch_size))

    # Settings/get_settings 可能被缓存，脚本内清空确保读取到覆盖后的 env
    from app.config import Settings, get_settings

    get_settings.cache_clear()
    s = Settings()

    print("=== torch / cuda ===")
    import torch  # type: ignore

    print("torch:", getattr(torch, "__version__", "?"))
    print("cuda.is_available():", bool(torch.cuda.is_available()))
    if torch.cuda.is_available():
        try:
            print("gpu:", torch.cuda.get_device_name(0))
        except Exception:
            pass

    picked_device, picked_fp16 = _pick_device_and_fp16(getattr(s, "device", "auto"), getattr(s, "use_fp16", "auto"))

    print("\n=== config (effective) ===")
    print("embedding_backend:", s.embedding_backend)
    print("embedding_model_name:", s.embedding_model_name)
    print("embedding_batch_size:", s.embedding_batch_size)
    print("device:", s.device, "-> picked_device:", picked_device)
    print("use_fp16:", s.use_fp16, "-> picked_use_fp16:", picked_fp16)

    # 仅验证 embedding + 索引构建：document + support_ticket（不加载 faq）
    from app.services.data_loader import DataLoader
    from app.services.cleaner import Cleaner
    from app.services.chunker import Chunker, ChunkingConfig
    from app.services.vector_retriever import VectorRetriever

    t0 = time.perf_counter()
    loader = DataLoader()

    t_load0 = time.perf_counter()
    docs_records = loader.load_raw_records("document", s.raw_documents_path, limit=int(args.limit))
    tkt_records = []
    if not args.documents_only:
        tkt_records = loader.load_raw_records("support_ticket", s.raw_support_tickets_path, limit=int(args.limit))
    t_load = (time.perf_counter() - t_load0) * 1000

    t_map0 = time.perf_counter()
    docs = loader.to_internal_documents([*docs_records, *tkt_records])
    t_map = (time.perf_counter() - t_map0) * 1000

    t_clean0 = time.perf_counter()
    docs = Cleaner().clean_documents(docs)
    t_clean = (time.perf_counter() - t_clean0) * 1000

    t_chunk0 = time.perf_counter()
    chunker = Chunker(ChunkingConfig(chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap))
    chunks = chunker.chunk_documents(docs)
    t_chunk = (time.perf_counter() - t_chunk0) * 1000

    t_vec0 = time.perf_counter()
    vec = VectorRetriever()
    vec.build(chunks)
    t_vec = (time.perf_counter() - t_vec0) * 1000

    total = (time.perf_counter() - t0) * 1000

    print("\n=== sizes ===")
    print("raw_records.document:", len(docs_records))
    print("raw_records.support_ticket:", len(tkt_records))
    print("internal_docs(after_clean):", len(docs))
    print("chunks:", len(chunks))

    print("\n=== timing_ms ===")
    print("load_raw_records:", round(t_load, 2))
    print("map_to_internal:", round(t_map, 2))
    print("clean:", round(t_clean, 2))
    print("chunk:", round(t_chunk, 2))
    print("embed+faiss_build:", round(t_vec, 2))
    print("total:", round(total, 2))

    print("\nRESULT: SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

