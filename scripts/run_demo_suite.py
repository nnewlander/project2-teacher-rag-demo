"""
演示问题集一键验收：读取 data/demo_queries.jsonl，逐条调用 /ask，打印简明报告。

默认使用 FastAPI TestClient（进程内 ASGI，无需单独起服务）。
可选 --http-base URL 对已启动的 uvicorn 发 HTTP 请求。

不修改业务主链路，仅消费现有 API。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 项目根：.../scripts/run_demo_suite.py -> parents[1]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = ROOT / "data" / "demo_queries.jsonl"


def _load_queries(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _trace_steps(data: Dict[str, Any]) -> List[str]:
    rt = data.get("route_trace") or []
    out: List[str] = []
    for item in rt:
        if isinstance(item, dict) and item.get("step"):
            out.append(str(item["step"]))
    return out


def _analyze_response(data: Dict[str, Any]) -> Tuple[bool, bool, bool, bool, bool, Optional[float]]:
    route = str(data.get("route") or "")
    mode = str(data.get("mode") or "")
    steps = set(_trace_steps(data))

    faq_hit = mode == "faq" or route == "bm25_faq"
    hyde = any(s.startswith("hyde.") or s == "hyde.generate" for s in steps)
    subquery = "subquery.split" in steps or route == "subquery"
    backtrack = "backtrack" in steps
    cites = data.get("citations") or data.get("matched_sources") or []
    has_citations = bool(cites)

    dbg = data.get("debug") if isinstance(data.get("debug"), dict) else {}
    timing = dbg.get("timing_ms") if isinstance(dbg.get("timing_ms"), dict) else {}
    total_ms = timing.get("total")
    if total_ms is not None:
        try:
            total_ms = float(total_ms)
        except (TypeError, ValueError):
            total_ms = None
    return faq_hit, hyde, subquery, backtrack, has_citations, total_ms


def _make_test_client():
    from fastapi.testclient import TestClient

    from app.main import create_app

    return TestClient(create_app())


def _ask_test_client(client: Any, query: str, top_k: int) -> Dict[str, Any]:
    r = client.post("/ask", json={"query": query, "top_k": top_k})
    r.raise_for_status()
    return r.json()


def _ask_http(base: str, query: str, top_k: int) -> Dict[str, Any]:
    import httpx

    url = base.rstrip("/") + "/ask"
    with httpx.Client(timeout=300.0, trust_env=False) as client:
        r = client.post(url, json={"query": query, "top_k": top_k})
        r.raise_for_status()
        return r.json()


def main() -> int:
    ap = argparse.ArgumentParser(description="运行演示问题集并输出验收报告")
    ap.add_argument("--file", type=Path, default=DEFAULT_QUERIES, help="demo_queries.jsonl 路径")
    ap.add_argument("--top-k", type=int, default=8, dest="top_k", help="传入 /ask 的 top_k")
    ap.add_argument(
        "--http-base",
        type=str,
        default="",
        help="若指定（如 http://127.0.0.1:8001），则通过 HTTP 调用已启动服务；默认用 TestClient",
    )
    args = ap.parse_args()

    path: Path = args.file
    if not path.is_file():
        print(f"queries file not found: {path}", file=sys.stderr)
        return 2

    rows = _load_queries(path)
    use_http = bool(args.http_base.strip())
    tc = None if use_http else _make_test_client()

    print("=" * 72)
    print("Demo suite report (demo_queries.jsonl)")
    print(f"File: {path}")
    print(f"Transport: {'HTTP ' + args.http_base if use_http else 'FastAPI TestClient (in-process)'}")
    print(f"top_k: {args.top_k}")
    print("=" * 72)

    # 表头
    hdr = f"{'id':<18} {'category':<16} {'route':<18} {'FAQ':<5} {'HyDE':<5} {'Subq':<5} {'Btrk':<5} {'Cit':<5} {'total_ms':>10}"
    print(hdr)
    print("-" * len(hdr))

    wall0 = time.perf_counter()
    for row in rows:
        qid = str(row.get("id", ""))[:16]
        cat = str(row.get("category", ""))[:14]
        query = str(row.get("query", "")).strip()
        if not query:
            print(f"{qid:<18} {cat:<16} {'(empty query)':<18}")
            continue

        t0 = time.perf_counter()
        try:
            if use_http:
                data = _ask_http(args.http_base.strip(), query, args.top_k)
            else:
                data = _ask_test_client(tc, query, args.top_k)
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"{qid:<18} {cat:<16} {'ERROR':<18} {'-':<5} {'-':<5} {'-':<5} {'-':<5} {'-':<5} {elapsed:>10.1f}")
            print(f"  !! {e!r}")
            continue

        elapsed = (time.perf_counter() - t0) * 1000
        route = str(data.get("route") or "")[:16]
        faq_hit, hyde, subq, btrk, cit, total_api = _analyze_response(data)
        total_show = total_api if total_api is not None else elapsed
        try:
            total_show_f = float(total_show)
        except (TypeError, ValueError):
            total_show_f = elapsed

        print(
            f"{qid:<18} {cat:<16} {route:<18} "
            f"{'Y' if faq_hit else 'n':<5} "
            f"{'Y' if hyde else 'n':<5} "
            f"{'Y' if subq else 'n':<5} "
            f"{'Y' if btrk else 'n':<5} "
            f"{'Y' if cit else 'n':<5} "
            f"{total_show_f:>10.1f}"
        )
        note = row.get("note")
        if note:
            print(f"  note: {note}")

    wall = (time.perf_counter() - wall0) * 1000
    print("-" * len(hdr))
    print(f"Wall time (all rows): {wall:.1f} ms (first call may include model/KB cold start)")
    print("=" * 72)
    print("Legend: FAQ=direct FAQ hit; HyDE/Subq/Btrk=step present in route_trace; Cit=has citations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
