from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import requests


def validate_hit(hit: Dict[str, Any]) -> None:
    required = ("source_id", "title", "snippet", "score", "source_type", "metadata")
    missing = [k for k in required if k not in hit]
    if missing:
        raise AssertionError(f"first hit missing fields: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Call /search and validate evidence-only response schema.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001", help="API base url, default http://127.0.0.1:8001")
    parser.add_argument("--top-k", type=int, default=3, help="top_k for /search")
    args = parser.parse_args()

    payload = {
        "query": "课堂演示遇到 NameError，应该怎么给学生解释？",
        "top_k": int(args.top_k),
        "filters": {},
        "request_id": "script-test-search-api",
    }
    url = f"{args.base_url.rstrip('/')}/search"
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    for key in ("hits", "query", "route_trace", "debug"):
        if key not in data:
            raise AssertionError(f"missing response field: {key}")

    hits = data.get("hits") or []
    print(f"hits_count={len(hits)}")
    if hits:
        validate_hit(hits[0])
        print("first_hit=" + json.dumps(hits[0], ensure_ascii=False, indent=2))
    else:
        print("first_hit=None")


if __name__ == "__main__":
    main()
