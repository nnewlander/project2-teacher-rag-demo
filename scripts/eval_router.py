from __future__ import annotations

"""
路由评测脚本（最小版）。

读取 route_eval_queries_10pct.jsonl，跑当前 Router，并输出：
- 总体准确率
- 各 route_label 的样本数 / 命中数 / 准确率
- 预测分布（pred label counts）

注意：
- 当前 Router 仅支持两类策略标签：bm25_faq / rag_standard
- 评测集中还有 hyde/subquery/backtrack/need_clarify 等标签，现阶段会自然拉低准确率
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from app.config import get_settings
from app.services.router import Router
from app.utils.jsonl import read_jsonl
from app.utils.text_utils import normalize_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rule-based router on route_eval_queries_10pct.jsonl")
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="评测集路径（默认读取 config.data_root 下的 route_eval_queries_10pct.jsonl）",
    )
    parser.add_argument("--limit", type=int, default=None, help="最多评测多少条（调试用）")
    args = parser.parse_args()

    settings = get_settings()
    default_path = Path(settings.data_root) / "route_eval_queries_10pct.jsonl"
    path = Path(args.path) if args.path else default_path

    router = Router()

    total = 0
    correct = 0

    by_gold_total: Counter[str] = Counter()
    by_gold_correct: Counter[str] = Counter()
    pred_dist: Counter[str] = Counter()

    # confusion[gold][pred] += 1
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)

    for row in read_jsonl(path, limit=args.limit):
        q = normalize_text(str(row.get("raw_query") or row.get("query") or ""))
        gold = str(row.get("route_label") or row.get("gold_route") or row.get("expected_route") or "").strip()
        if not q or not gold:
            continue

        decision = router.decide(q)
        pred = decision.eval_label

        total += 1
        by_gold_total[gold] += 1
        pred_dist[pred] += 1
        confusion[gold][pred] += 1

        if pred == gold:
            correct += 1
            by_gold_correct[gold] += 1

    if total == 0:
        print(f"No valid rows found in: {path}")
        return

    acc = correct / total
    print(f"Eval file: {path}")
    print(f"Total: {total}  Correct: {correct}  Accuracy: {acc:.4f}")
    print()

    print("Per-gold-label stats:")
    for gold, n in by_gold_total.most_common():
        c = by_gold_correct.get(gold, 0)
        a = c / n if n else 0.0
        print(f"- {gold:12s}  n={n:4d}  correct={c:4d}  acc={a:.4f}")
    print()

    print("Predicted label distribution:")
    for pred, n in pred_dist.most_common():
        print(f"- {pred:12s}  n={n:4d}")
    print()

    # 输出一个简易 confusion（每个 gold 只展示 top-3 pred）
    print("Confusion (top-3 predicted per gold):")
    for gold, ctr in sorted(confusion.items(), key=lambda x: by_gold_total[x[0]], reverse=True):
        top3 = ", ".join([f"{p}:{k}" for p, k in ctr.most_common(3)])
        print(f"- gold={gold:12s} -> {top3}")


if __name__ == "__main__":
    main()

