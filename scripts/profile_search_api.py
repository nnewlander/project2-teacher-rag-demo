from __future__ import annotations

import argparse
import time
from typing import Any, Dict, Optional

import requests


def call(
    method: str,
    url: str,
    json_body: Optional[Dict[str, Any]] = None,
    *,
    case_name: str = "",
    timeout_s: float = 8.0,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    label = case_name or f"{method} {url}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=timeout_s)
        else:
            resp = requests.post(url, json=json_body, timeout=timeout_s)
        cost_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()
        slow = cost_ms > 5000
        mark = " [SLOW]" if slow else ""
        print(f"{label} cost_ms={cost_ms:.2f}{mark}")
        if isinstance(data, dict):
            timing = (data.get("debug") or {}).get("timing_ms")
            if timing:
                print(f"  debug.timing_ms={timing}")
        return {"ok": True, "timeout": False, "slow": slow, "cost_ms": cost_ms, "data": data}
    except requests.exceptions.Timeout as e:
        cost_ms = (time.perf_counter() - t0) * 1000
        print(f"{label} timeout_s={timeout_s} error={e}")
        return {"ok": False, "timeout": True, "slow": True, "cost_ms": cost_ms, "error": str(e)}
    except Exception as e:
        cost_ms = (time.perf_counter() - t0) * 1000
        print(f"{label} error={e}")
        return {"ok": False, "timeout": False, "slow": cost_ms > 5000, "cost_ms": cost_ms, "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile /health /ready /search latency.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001", help="API base URL")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    cases = [
        ("health", "GET", f"{base}/health", None),
        ("ready", "GET", f"{base}/ready", None),
        (
            "search_nameerror",
            "POST",
            f"{base}/search",
            {"query": "课堂演示遇到 NameError，应该怎么给学生解释？", "top_k": 3, "filters": {}, "request_id": "profile-search-api"},
        ),
        (
            "search_typeerror",
            "POST",
            f"{base}/search",
            {"query": "课堂上 TypeError 常见原因是什么，怎么排查？", "top_k": 3, "filters": {}, "request_id": "profile-search-api"},
        ),
        (
            "search_syntaxerror",
            "POST",
            f"{base}/search",
            {"query": "SyntaxError 报错通常怎么讲解更清楚？", "top_k": 3, "filters": {}, "request_id": "profile-search-api"},
        ),
    ]

    total_cases = len(cases)
    success_cases = 0
    timeout_cases = 0
    slow_cases = 0
    for case_name, method, url, body in cases:
        res = call(method, url, body, case_name=case_name, timeout_s=8.0)
        if res.get("ok"):
            success_cases += 1
        if res.get("timeout"):
            timeout_cases += 1
        if res.get("slow"):
            slow_cases += 1

    print(
        "summary:",
        {
            "total_cases": total_cases,
            "success_cases": success_cases,
            "timeout_cases": timeout_cases,
            "slow_cases": slow_cases,
        },
    )


if __name__ == "__main__":
    main()
