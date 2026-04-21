from __future__ import annotations

"""
离线构建索引（MVP）。

当前：
- 直接加载本地 jsonl
- 构建 FAQ BM25 与 mock 向量索引

TODO:
- 将 artifacts 持久化（本地文件/SQLite/Redis）
- 接入 Milvus 与 embedding 模型
"""

import argparse
import time

from app.config import get_settings
from app.services.qa_service import QAService


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP: build FAQ + mock vector indexes from jsonl.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每个数据源最多读取多少条（用于快速验证；不传则全量）",
    )
    args = parser.parse_args()

    settings = get_settings()
    qa = QAService(settings)
    t0 = time.time()
    artifacts = qa.init_kb(limit=args.limit)
    cost_ms = int((time.time() - t0) * 1000)
    print("Ingest done.")
    print(f"- elapsed_ms: {cost_ms}")
    print(f"- docs: {len(artifacts.docs)}")
    print(f"- faq_count: {artifacts.faq_count}")
    print(f"- chunks_count: {artifacts.chunks_count}")


if __name__ == "__main__":
    main()

