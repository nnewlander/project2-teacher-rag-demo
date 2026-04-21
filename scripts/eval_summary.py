from __future__ import annotations

"""
统一展示用评测汇总脚本（面试展示友好）。

目标：
- 不改 FastAPI / 不重构检索主逻辑
- 复用 QAService.ask 的真实返回（route/mode/debug.timing_ms/debug.cache.hit）
- 汇总：FAQ 命中率、route 分布、平均 total 耗时、FAQ vs RAG 平均耗时、cache hit 统计

说明：
- “cache hit” 统计需要重复调用同一 query 才可能出现命中（取决于 cache_min_hits_to_store）。
  默认做多轮 pass；最终 pass 的 debug.cache.hit==True 计为命中。
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.services.qa_service import QAService
from app.utils.jsonl import read_jsonl
from app.utils.text_utils import normalize_text


def _get_timing_total_ms(debug: Dict[str, Any]) -> Optional[float]:
    if not isinstance(debug, dict):
        return None
    timing = debug.get("timing_ms")
    if not isinstance(timing, dict):
        return None
    v = timing.get("total")
    try:
        return float(v)
    except Exception:
        return None


def _get_cache_hit(debug: Dict[str, Any]) -> bool:
    if not isinstance(debug, dict):
        return False
    cache = debug.get("cache")
    if not isinstance(cache, dict):
        return False
    return bool(cache.get("hit", False))


@dataclass
class Agg:
    n: int = 0
    total_ms_sum: float = 0.0
    total_ms_n: int = 0
    cache_hits: int = 0

    def add(self, *, total_ms: Optional[float], cache_hit: bool) -> None:
        self.n += 1
        if total_ms is not None:
            self.total_ms_sum += float(total_ms)
            self.total_ms_n += 1
        if cache_hit:
            self.cache_hits += 1

    def mean_total_ms(self) -> float:
        return (self.total_ms_sum / self.total_ms_n) if self.total_ms_n else 0.0

    def cache_hit_rate(self) -> float:
        return (self.cache_hits / self.n) if self.n else 0.0


def _safe_json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_csv_dump(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified demo-friendly evaluation summary.")
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="评测集路径（默认读取 config.data_root 下的 route_eval_queries_10pct.jsonl）",
    )
    parser.add_argument("--limit", type=int, default=None, help="最多评测多少条（调试用）")
    parser.add_argument(
        "--passes",
        type=int,
        default=3,
        help="重复跑同一评测集多少轮（用于观察 cache hit；默认 3 轮更容易出现命中）",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/eval",
        help="导出目录（json/csv），默认 outputs/eval",
    )
    parser.add_argument(
        "--export",
        type=str,
        default="both",
        choices=["none", "json", "csv", "both"],
        help="导出格式：none/json/csv/both",
    )
    args = parser.parse_args()

    settings = get_settings()
    default_path = Path(settings.data_root) / "route_eval_queries_10pct.jsonl"
    path = Path(args.path) if args.path else default_path
    outdir = Path(args.outdir)
    passes = max(1, int(args.passes))

    qa = QAService(settings)
    print("Initializing knowledge base (may take a while on first run)...", flush=True)
    qa.init_kb()
    print("Knowledge base ready. Starting evaluation...", flush=True)

    # 读取评测 queries（去空）
    queries: List[Tuple[str, Dict[str, Any]]] = []
    for row in read_jsonl(path, limit=args.limit):
        q = normalize_text(str(row.get("raw_query") or row.get("query") or ""))
        if not q:
            continue
        queries.append((q, row))

    if not queries:
        print(f"No valid queries found in: {path}")
        return

    # 统计容器
    route_dist: Counter[str] = Counter()
    mode_dist: Counter[str] = Counter()
    faq_direct_hits = 0  # 以 mode == "faq" 计（展示口径：FAQ 直答命中）

    overall = Agg()
    by_route: Dict[str, Agg] = defaultdict(Agg)
    by_mode: Dict[str, Agg] = defaultdict(Agg)

    # cache hit 展示：只统计最后一轮（更接近“热缓存后”的体验）
    final_pass_cache_hits = 0

    # 可选明细导出（最后一轮）
    last_rows: List[Dict[str, Any]] = []

    for p in range(1, passes + 1):
        if passes > 1:
            print(f"Pass {p}/{passes} ...", flush=True)
        for q, row in queries:
            resp = qa.ask(q)
            route = str(getattr(resp, "route", "") or "")
            mode = str(getattr(resp, "mode", "") or "")
            total_ms = _get_timing_total_ms(getattr(resp, "debug", {}) or {})
            cache_hit = _get_cache_hit(getattr(resp, "debug", {}) or {})

            if p == passes:
                route_dist[route or "unknown"] += 1
                mode_dist[mode or "unknown"] += 1
                if mode == "faq":
                    faq_direct_hits += 1

                overall.add(total_ms=total_ms, cache_hit=cache_hit)
                by_route[route or "unknown"].add(total_ms=total_ms, cache_hit=cache_hit)
                by_mode[mode or "unknown"].add(total_ms=total_ms, cache_hit=cache_hit)
                if cache_hit:
                    final_pass_cache_hits += 1

                # 导出展示明细：只保留少量字段，避免过长
                last_rows.append(
                    {
                        "query": q,
                        "route": route,
                        "mode": mode,
                        "total_ms": total_ms,
                        "cache_hit": cache_hit,
                        "query_type": (resp.route_trace[0].detail.get("query_type") if getattr(resp, "route_trace", None) else None),
                        "cleaned_query": (resp.route_trace[0].detail.get("cleaned_query") if getattr(resp, "route_trace", None) else None),
                        "gold_route_label": str(row.get("route_label") or ""),
                        "linked_source_type": str(row.get("linked_source_type") or ""),
                    }
                )

    n = overall.n
    faq_hit_rate = (faq_direct_hits / n) if n else 0.0

    # FAQ vs RAG 平均耗时（展示口径：按 mode 分组）
    faq_ms = by_mode.get("faq", Agg()).mean_total_ms()
    rag_ms = by_mode.get("rag_mock", Agg()).mean_total_ms()

    summary: Dict[str, Any] = {
        "eval_file": str(path),
        "n_queries": n,
        "passes": passes,
        "faq_direct_hit_rate": round(faq_hit_rate, 4),
        "route_distribution": dict(route_dist),
        "avg_total_ms": round(overall.mean_total_ms(), 2),
        "avg_total_ms_by_mode": {
            "faq": round(faq_ms, 2),
            "rag_mock": round(rag_ms, 2),
        },
        "avg_total_ms_by_route": {k: round(v.mean_total_ms(), 2) for k, v in sorted(by_route.items(), key=lambda x: -x[1].n)},
        "cache": {
            "hit_count_final_pass": int(final_pass_cache_hits),
            "hit_rate_final_pass": round(overall.cache_hit_rate(), 4),
        },
    }

    # 控制台打印（展示友好）
    print(f"Eval file: {path}")
    print(f"Queries (final pass): {n}  Passes: {passes}")
    print()
    print(f"FAQ direct hit rate (mode==faq): {summary['faq_direct_hit_rate']:.4f}")
    print()
    print("Route distribution (final pass):")
    for k, v in route_dist.most_common():
        print(f"- {k:14s}  n={v:4d}  ratio={(v/n if n else 0.0):.4f}")
    print()
    print(f"Avg total latency (ms): {summary['avg_total_ms']:.2f}")
    print(f"Avg latency FAQ vs RAG (ms): faq={summary['avg_total_ms_by_mode']['faq']:.2f}  rag={summary['avg_total_ms_by_mode']['rag_mock']:.2f}")
    print()
    print(f"Cache hit (final pass): hits={summary['cache']['hit_count_final_pass']}  rate={summary['cache']['hit_rate_final_pass']:.4f}")

    # 导出
    if args.export != "none":
        if args.export in ("json", "both"):
            _safe_json_dump(outdir / "eval_summary.json", summary)
            _safe_json_dump(outdir / "eval_last_pass_rows.json", last_rows[:2000])
        if args.export in ("csv", "both"):
            _safe_csv_dump(
                outdir / "eval_last_pass_rows.csv",
                last_rows,
                fieldnames=[
                    "query",
                    "cleaned_query",
                    "query_type",
                    "route",
                    "mode",
                    "total_ms",
                    "cache_hit",
                    "gold_route_label",
                    "linked_source_type",
                ],
            )


if __name__ == "__main__":
    main()

