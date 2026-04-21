"""
严格环境下的 bge_reranker 分阶段验证（不走 FAQ / LLM / QAService / Router / /health）。

目标：
1) bge_reranker 能成功初始化（FlagEmbedding.FlagReranker）
2) 能对少量 query-passage 对 compute_score 打分
3) 小规模 rerank 可视化：给定 query，从小索引召回候选 chunks，输出 rerank 前后顺序与分数

用法：
  python -m scripts.inspect_bge_rerank --limit 300 --batch-size 4 --vec-top-k 8 --rerank-top-k 6
  python -m scripts.inspect_bge_rerank --query "班级开课后学生端一直显示未开始，老师需要在哪里确认？"
  python -m scripts.inspect_bge_rerank --reranker-model "BAAI/bge-reranker-v2-m3" --device cuda --use-fp16 true
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_QUERY = "班级开课后学生端一直显示未开始，老师问是不是还需要在课堂管理后台再点一次确认？"

PAIR_SCORE_QUERIES_AND_PASSAGES: Tuple[Tuple[str, str], ...] = (
    (
        "老师在作业批改台里找不到上节课的作业发布入口，是入口改版了吗？",
        "老师在作业批改台里找不到上节课的作业发布入口，怀疑入口改版。建议先确认教师端/学生端环境与新版入口路径。",
    ),
    (
        "算法入门这节课的导入环节有点生硬，想找一个能自然引到循环嵌套的案例，有讲评模板吗？",
        "算法入门课程的导入可以先讲可见结果，再回到概念定义，并提供适合课堂讲评的模板与示例。",
    ),
    (
        "课堂演示的 Python 代码一运行就提示 IndentationError，最常见的原因和改法是什么？",
        "IndentationError 常见于缩进层级不一致、混用 tab 与空格、或缺少冒号后的缩进。建议用最小复现示例解释并演示修正。",
    ),
)


def _summarize(text: str, max_chars: int = 180) -> str:
    t = " ".join((text or "").split())
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


def _safe_meta(ctx) -> Dict[str, Any]:
    meta = getattr(ctx, "metadata", None)
    return meta if isinstance(meta, dict) else {}


def _print_kv(title: str, kv: Dict[str, Any]) -> None:
    print(title)
    for k, v in kv.items():
        print(f"- {k}: {v}")


@dataclass(frozen=True)
class Candidate:
    source_id: str
    source: str
    score_vec: float
    title: str
    text: str
    metadata: Dict[str, Any]


def _build_small_kb_candidates(
    *,
    limit: int,
    embedding_batch_size: int,
    vec_top_k: int,
    query: str,
    documents_only: bool,
) -> Tuple[List[Any], List[Candidate]]:
    """
    返回：
    - contexts: List[RetrievedContext]（用于 reranker.rerank）
    - candidates: 结构化信息（用于打印）
    """
    os.environ["KBQA_EMBEDDING_BATCH_SIZE"] = str(int(embedding_batch_size))

    from app.config import Settings, get_settings
    from app.services.chunker import Chunker, ChunkingConfig
    from app.services.cleaner import Cleaner
    from app.services.data_loader import DataLoader
    from app.services.vector_retriever import VectorRetriever

    get_settings.cache_clear()
    s = Settings()

    loader = DataLoader()
    docs_records = loader.load_raw_records("document", s.raw_documents_path, limit=int(limit))
    tkt_records: list = []
    if not documents_only:
        tkt_records = loader.load_raw_records("support_ticket", s.raw_support_tickets_path, limit=int(limit))
    docs = loader.to_internal_documents([*docs_records, *tkt_records])
    docs = Cleaner().clean_documents(docs)
    chunker = Chunker(ChunkingConfig(chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap))
    chunks = chunker.chunk_documents(docs)

    vec = VectorRetriever()
    vec.build(chunks)
    hits = vec.search(query, top_k=max(1, int(vec_top_k)))
    contexts = vec.as_contexts(hits)

    out: List[Candidate] = []
    for h in hits:
        meta = h.metadata or {}
        out.append(
            Candidate(
                source_id=h.chunk_id,
                source=str(meta.get("source") or "?"),
                score_vec=float(h.score),
                title=str(meta.get("title") or ""),
                text=str(h.text or ""),
                metadata=dict(meta),
            )
        )
    return contexts, out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Strict validate bge_reranker init/score/rerank (no QAService).")
    ap.add_argument("--limit", type=int, default=300, help="抽样 raw_records 上限（document + support_ticket）")
    ap.add_argument("--batch-size", type=int, default=4, help="embedding batch size（仅用于召回候选 chunks）")
    ap.add_argument("--vec-top-k", type=int, default=8, help="向量召回候选条数（rerank 前的原始顺序）")
    ap.add_argument("--rerank-top-k", type=int, default=6, help="打印 rerank 后的 top-k（<= vec-top-k）")
    ap.add_argument("--documents-only", action="store_true", help="仅 document，不加载 support_ticket")

    ap.add_argument("--query", type=str, default=DEFAULT_QUERY, help="用于小规模 rerank 的 query")
    ap.add_argument("--reranker-model", type=str, default="", help="覆盖 Settings.reranker_model_name")
    ap.add_argument("--device", type=str, default="", help="覆盖 Settings.device（auto/cpu/cuda）")
    ap.add_argument("--use-fp16", type=str, default="", help="覆盖 Settings.use_fp16（auto/true/false）")
    ap.add_argument(
        "--no-fallback",
        action="store_true",
        help="关闭 reranker_fallback_on_error（初始化失败直接 FAIL）",
    )
    args = ap.parse_args()

    # 强制走 bge_reranker（不改业务逻辑，只在脚本层覆盖 env）
    os.environ["KBQA_RERANKER_BACKEND"] = "bge_reranker"
    if args.no_fallback:
        os.environ["KBQA_RERANKER_FALLBACK_ON_ERROR"] = "false"
    if str(args.reranker_model).strip():
        os.environ["KBQA_RERANKER_MODEL_NAME"] = str(args.reranker_model).strip()
    if str(args.device).strip():
        os.environ["KBQA_DEVICE"] = str(args.device).strip()
    if str(args.use_fp16).strip():
        os.environ["KBQA_USE_FP16"] = str(args.use_fp16).strip()

    from app.config import Settings, get_settings
    from app.services.reranker import create_reranker_from_settings
    from app.services.reranker import _pick_device_and_fp16 as pick_device_and_fp16  # noqa: PLC2701

    get_settings.cache_clear()
    s = Settings()

    # -------- stage 0: torch / cuda --------
    print("=== torch / cuda ===")
    try:
        import torch  # type: ignore

        torch_ver = getattr(torch, "__version__", "?")
        cuda_ok = bool(torch.cuda.is_available()) if getattr(torch, "cuda", None) is not None else False
        print("torch:", torch_ver)
        print("cuda.is_available():", cuda_ok)
        if cuda_ok:
            try:
                print("gpu:", torch.cuda.get_device_name(0))
            except Exception:
                pass
    except Exception as e:
        print("torch: (import failed)", repr(e))

    picked_device, picked_fp16 = pick_device_and_fp16(s)
    print("\n=== config (effective) ===")
    print("reranker_backend:", s.reranker_backend)
    print("reranker_model_name:", (s.reranker_model_name or "").strip() or "BAAI/bge-reranker-v2-m3")
    print("device:", s.device, "-> picked_device:", picked_device)
    print("use_fp16:", s.use_fp16, "-> picked_use_fp16:", picked_fp16)
    print("reranker_fallback_on_error:", bool(getattr(s, "reranker_fallback_on_error", True)))

    ok = True
    t0 = time.perf_counter()

    # -------- stage 1: init --------
    print("\n=== [1] init bge_reranker ===")
    try:
        rr = create_reranker_from_settings(s)
        rr_cls = rr.__class__.__name__
        print("reranker_instance:", rr_cls)
        # strict 期望是 FlagBgeReranker；若 fallback 则会是 KeywordOverlapReranker
        if rr_cls != "FlagBgeReranker":
            print("WARN: 当前生效 reranker 不是 FlagBgeReranker（可能发生了 fallback）。")
            if args.no_fallback:
                raise RuntimeError("no-fallback 模式下 reranker 不应回退。")
        print("OK")
    except Exception as e:
        ok = False
        print("FAIL:", repr(e))
        print("\nRESULT: FAIL")
        return 1

    # -------- stage 2: pair scoring --------
    print("\n=== [2] compute_score on query-passage pairs ===")
    try:
        model = getattr(rr, "model", None)  # FlagBgeReranker.model
        core = getattr(model, "model", None)  # FlagBgeReranker.model is FlagReranker, no nested .model expected
        flag_reranker = model if model is not None else core
        if flag_reranker is None or not hasattr(flag_reranker, "compute_score"):
            raise RuntimeError("当前 reranker 无 compute_score（可能不是 FlagEmbedding.FlagReranker 路径）。")

        pairs: List[List[str]] = [[q, p] for (q, p) in PAIR_SCORE_QUERIES_AND_PASSAGES]
        try:
            scores = flag_reranker.compute_score(pairs, batch_size=1)
        except TypeError:
            scores = flag_reranker.compute_score(pairs)
        arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        scores_f = [float(x) for x in arr.tolist()]
        for i, ((q, p), sc) in enumerate(zip(PAIR_SCORE_QUERIES_AND_PASSAGES, scores_f), start=1):
            print(f"- pair {i}: score={sc:.6f}")
            print("  query  :", _summarize(q, 120))
            print("  passage:", _summarize(p, 140))
        print("OK")
    except Exception as e:
        ok = False
        print("FAIL:", repr(e))
        print("\nRESULT: FAIL")
        return 1

    # -------- stage 3: rerank visualize --------
    print("\n=== [3] small rerank visualize ===")
    try:
        query = str(args.query or "").strip()
        if not query:
            raise RuntimeError("empty --query")

        contexts, cand = _build_small_kb_candidates(
            limit=int(args.limit),
            embedding_batch_size=int(args.batch_size),
            vec_top_k=int(args.vec_top_k),
            query=query,
            documents_only=bool(args.documents_only),
        )
        if not contexts:
            raise RuntimeError("no candidates retrieved (vector search returned empty)")

        # 原始候选顺序（向量召回顺序）
        print("\n[query]")
        print(query)
        print("\n[original candidates by vector score]")
        for i, c in enumerate(cand, start=1):
            print(
                f"- #{i:02d} vec_score={c.score_vec:8.4f} source={c.source:14s} "
                f"title={_summarize(c.title, 60)} id={c.source_id}"
            )

        # 计算 reranker 分数（不改 rerank 逻辑，只做打印）
        pairs3: List[List[str]] = [[query, (getattr(c, 'text', '') or '')[:8000]] for c in contexts]
        try:
            rs = flag_reranker.compute_score(pairs3, batch_size=1)
        except TypeError:
            rs = flag_reranker.compute_score(pairs3)
        r_arr = np.asarray(rs, dtype=np.float64).reshape(-1)
        rerank_scores = [float(x) for x in r_arr.tolist()]
        if len(rerank_scores) != len(contexts):
            raise RuntimeError("reranker score length mismatch")

        # rerank 后顺序（调用现有 rerank）
        reranked = rr.rerank(query, list(contexts))
        # 为了展示“前后变化”，把 reranked 映射回原始索引
        id_to_idx = {getattr(c, "source_id", ""): i for i, c in enumerate(contexts)}

        print("\n[reranked candidates (by bge_reranker score)]")
        topn = min(max(1, int(args.rerank_top_k)), len(reranked))
        for rank, ctx in enumerate(reranked[:topn], start=1):
            meta = _safe_meta(ctx)
            sid = str(getattr(ctx, "source_id", "") or "")
            i0 = id_to_idx.get(sid, -1)
            sc = rerank_scores[i0] if 0 <= i0 < len(rerank_scores) else float("nan")
            vec_sc = float(getattr(ctx, "score", 0.0))
            title = str(meta.get("title") or "")
            src = str(meta.get("source") or "?")
            print(
                f"- rank {rank:02d} rerank_score={sc:9.6f} vec_score={vec_sc:8.4f} "
                f"source={src:14s} title={_summarize(title, 60)} id={sid}"
            )
            print(f"  summary: {_summarize(str(getattr(ctx, 'text', '') or ''), 180)}")

        print("OK")
    except Exception as e:
        ok = False
        print("FAIL:", repr(e))
        print("\nRESULT: FAIL")
        return 1

    total_ms = (time.perf_counter() - t0) * 1000
    print(f"\n=== timing_ms ===\n- total: {total_ms:.2f}")
    print("\nRESULT: SUCCESS" if ok else "\nRESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

