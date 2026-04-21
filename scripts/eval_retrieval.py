from __future__ import annotations

"""
检索评测脚本（最小版）

评测对象：
- FAQ 检索（BM25）
- 向量检索（FAISS + embedding）
- Hybrid（FAQ + 向量 + rerank）

评测数据：
- 默认使用 project2_rag_raw_data_10pct/data/route_eval_queries_10pct.jsonl

指标（最小可运行，基于现有评测集字段的“弱标注”定义）：
- FAQ 命中率：是否能在 FAQ top_k 中召回到 FAQ 类型候选（且 top1 分数 >= 配置阈值）
- Recall@k：在 top-k contexts 中是否出现“期望来源类型”候选
- MRR@k：期望来源类型首次出现的位置倒数（无则 0）

弱标注说明：
- 评测集里没有每条 query 的精确相关 doc_id/faq_id，这里用 linked_source_type 做“相关性”近似：
  - linked_source_type == "faq"      -> 期望 contexts source 为 "faq"
  - linked_source_type == "ticket"   -> 期望 chunk metadata.source 为 "support_ticket"
  - linked_source_type == "document" -> 期望 chunk metadata.source 为 "document"
  - linked_source_type == "code_example" -> 当前未接入该数据源，指标会计入 unknown 并跳过 Recall/MRR 分母

TODO:
- 引入更严格的标注（gold doc_id / faq_id / chunk_id）后再做真实 Recall/MRR
- 增加 nDCG、按知识点/模块的细分统计
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from app.config import get_settings
from app.services.qa_service import QAService
from app.utils.jsonl import read_jsonl
from app.utils.text_utils import normalize_text


def _expected_source_from_linked(linked_source_type: str) -> Optional[str]:
    linked = (linked_source_type or "").strip().lower()
    if linked == "faq":
        return "faq"
    if linked == "ticket":
        return "support_ticket"
    if linked == "document":
        return "document"
    # code_example / external / unknown：当前不参与严格指标
    return None


def _is_relevant_context(expected_source: str, ctx) -> bool:
    # ctx 是 RetrievedContext
    if expected_source == "faq":
        return ctx.source == "faq"

    # 对非 FAQ，命中的是 chunk，真实来源在 ctx.metadata["source"]
    meta = ctx.metadata if isinstance(ctx.metadata, dict) else {}
    return ctx.source == "chunk" and str(meta.get("source") or "") == expected_source


def _recall_and_rr_at_k(expected_source: Optional[str], contexts: List, k: int) -> Tuple[Optional[float], Optional[float]]:
    if expected_source is None:
        return None, None
    top = contexts[: max(1, k)]
    for i, ctx in enumerate(top, start=1):
        if _is_relevant_context(expected_source, ctx):
            return 1.0, 1.0 / i
    return 0.0, 0.0


@dataclass
class MetricAgg:
    n_total: int = 0
    n_scored: int = 0  # 能计算 recall/mrr 的样本数（expected_source 非 None）
    faq_hits: int = 0
    recall_sum: float = 0.0
    rr_sum: float = 0.0
    unknown_linked: int = 0

    def add(self, *, faq_hit: bool, recall: Optional[float], rr: Optional[float], unknown: bool) -> None:
        self.n_total += 1
        if unknown:
            self.unknown_linked += 1
        if faq_hit:
            self.faq_hits += 1
        if recall is not None and rr is not None:
            self.n_scored += 1
            self.recall_sum += float(recall)
            self.rr_sum += float(rr)

    def summary(self) -> Dict[str, float]:
        faq_hit_rate = (self.faq_hits / self.n_total) if self.n_total else 0.0
        recall_at_k = (self.recall_sum / self.n_scored) if self.n_scored else 0.0
        mrr_at_k = (self.rr_sum / self.n_scored) if self.n_scored else 0.0
        return {
            "n_total": float(self.n_total),
            "n_scored": float(self.n_scored),
            "unknown_linked": float(self.unknown_linked),
            "faq_hit_rate": faq_hit_rate,
            "recall_at_k": recall_at_k,
            "mrr_at_k": mrr_at_k,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FAQ / vector / hybrid retrieval (minimal).")
    parser.add_argument("--k", type=int, default=5, help="top-k for Recall/MRR")
    parser.add_argument("--limit", type=int, default=None, help="最多评测多少条（调试用）")
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="评测集路径（默认读取 config.data_root 下的 route_eval_queries_10pct.jsonl）",
    )
    args = parser.parse_args()

    settings = get_settings()
    default_path = Path(settings.data_root) / "route_eval_queries_10pct.jsonl"
    path = Path(args.path) if args.path else default_path
    k = max(1, int(args.k))

    # 初始化知识库与检索器（复用现有结构）
    qa = QAService(settings)
    qa.init_kb()

    overall = {
        "faq": MetricAgg(),
        "vector": MetricAgg(),
        "hybrid": MetricAgg(),
    }

    # 分组统计：按评测集 route_label、按 router 预测 eval_label
    by_gold: Dict[str, Dict[str, MetricAgg]] = defaultdict(lambda: {"faq": MetricAgg(), "vector": MetricAgg(), "hybrid": MetricAgg()})
    by_pred: Dict[str, Dict[str, MetricAgg]] = defaultdict(lambda: {"faq": MetricAgg(), "vector": MetricAgg(), "hybrid": MetricAgg()})

    for row in read_jsonl(path, limit=args.limit):
        q = normalize_text(str(row.get("raw_query") or row.get("query") or ""))
        gold = str(row.get("route_label") or row.get("gold_route") or row.get("expected_route") or "").strip()
        linked = str(row.get("linked_source_type") or "").strip()
        if not q:
            continue

        expected_source = _expected_source_from_linked(linked)
        unknown = expected_source is None

        decision = qa.router.decide(q)
        pred_label = getattr(decision, "eval_label", decision.route)

        # 1) FAQ 检索
        faq_hits = qa.faq.search(q, top_k=int(settings.faq_top_k))
        faq_ctxs = qa.faq.as_contexts(faq_hits[:k])
        faq_hit = bool(faq_hits) and float(faq_hits[0].score) >= float(settings.faq_min_score)
        faq_recall, faq_rr = _recall_and_rr_at_k(expected_source, faq_ctxs, k)

        # 2) 向量检索（仅 chunk）
        vec_hits = qa.vec.search(q, top_k=int(settings.vector_top_k))
        vec_ctxs = qa.vec.as_contexts(vec_hits[:k])
        vec_recall, vec_rr = _recall_and_rr_at_k(expected_source, vec_ctxs, k)

        # 3) Hybrid（合并 + rerank）
        hy = qa.hybrid.retrieve(q, faq_top_k=int(settings.faq_top_k), vec_top_k=int(settings.vector_top_k), hybrid_top_k=k)
        hy_ctxs = hy.contexts[:k]
        hy_recall, hy_rr = _recall_and_rr_at_k(expected_source, hy_ctxs, k)

        # 聚合
        overall["faq"].add(faq_hit=faq_hit, recall=faq_recall, rr=faq_rr, unknown=unknown)
        overall["vector"].add(faq_hit=False, recall=vec_recall, rr=vec_rr, unknown=unknown)
        overall["hybrid"].add(faq_hit=faq_hit, recall=hy_recall, rr=hy_rr, unknown=unknown)

        if gold:
            by_gold[gold]["faq"].add(faq_hit=faq_hit, recall=faq_recall, rr=faq_rr, unknown=unknown)
            by_gold[gold]["vector"].add(faq_hit=False, recall=vec_recall, rr=vec_rr, unknown=unknown)
            by_gold[gold]["hybrid"].add(faq_hit=faq_hit, recall=hy_recall, rr=hy_rr, unknown=unknown)

        by_pred[pred_label]["faq"].add(faq_hit=faq_hit, recall=faq_recall, rr=faq_rr, unknown=unknown)
        by_pred[pred_label]["vector"].add(faq_hit=False, recall=vec_recall, rr=vec_rr, unknown=unknown)
        by_pred[pred_label]["hybrid"].add(faq_hit=faq_hit, recall=hy_recall, rr=hy_rr, unknown=unknown)

    def _print_block(title: str, agg: MetricAgg) -> None:
        s = agg.summary()
        print(
            f"{title:12s}  n={int(s['n_total']):4d}  scored={int(s['n_scored']):4d}  "
            f"unknown={int(s['unknown_linked']):4d}  faq_hit={s['faq_hit_rate']:.4f}  "
            f"R@{k}={s['recall_at_k']:.4f}  MRR@{k}={s['mrr_at_k']:.4f}"
        )

    print(f"Eval file: {path}")
    print(f"top_k (k): {k}")
    print()
    print("Overall:")
    _print_block("FAQ", overall["faq"])
    _print_block("Vector", overall["vector"])
    _print_block("Hybrid", overall["hybrid"])
    print()

    print("By gold route_label (Hybrid only):")
    for gold, d in sorted(by_gold.items(), key=lambda x: x[1]["hybrid"].n_total, reverse=True):
        _print_block(gold[:12], d["hybrid"])
    print()

    print("By router predicted eval_label (Hybrid only):")
    for pred, d in sorted(by_pred.items(), key=lambda x: x[1]["hybrid"].n_total, reverse=True):
        _print_block(pred[:12], d["hybrid"])


if __name__ == "__main__":
    main()

