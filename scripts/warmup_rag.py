from __future__ import annotations

import argparse
import json

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm up lightweight FAQ/BM25 resources for Project2 RAG.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001", help="API base url, default http://127.0.0.1:8001")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    warmup = requests.get(f"{base}/warmup", timeout=20).json()
    ready = requests.get(f"{base}/ready", timeout=5).json()

    print("warmup=" + json.dumps(warmup, ensure_ascii=False, indent=2))
    print("ready=" + json.dumps(ready, ensure_ascii=False, indent=2))

    if not ready.get("faq_ready") or not ready.get("bm25_ready"):
        print("WARNING: faq_ready/bm25_ready is still false after warmup.")


if __name__ == "__main__":
    main()

