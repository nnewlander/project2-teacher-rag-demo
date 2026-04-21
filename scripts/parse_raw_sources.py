from __future__ import annotations

"""
离线：解析多格式原始资料目录 -> 导出 document 源 jsonl。

用途：
- 将 docs/samples（或任意目录）里的 .txt/.md/.html/.docx/.pdf 解析为统一结构
- 输出可直接接入 DataLoader（source=document）的 jsonl
- 为后续 cleaner/chunker/ingest_data 复用提供“原始输入层”
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

from app.services.document_parser import DocumentParser
from app.utils.jsonl import write_jsonl


SUPPORTED_EXTS = {".txt", ".md", ".html", ".htm", ".docx", ".pdf"}


def iter_files(root: Path, *, recursive: bool = True) -> Iterable[Path]:
    if recursive:
        yield from (p for p in root.rglob("*") if p.is_file())
    else:
        yield from (p for p in root.glob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse multi-format raw sources to document jsonl.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="docs/samples",
        help="原始资料目录（默认 docs/samples）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/parsed_documents.jsonl",
        help="输出 jsonl 路径（默认 outputs/parsed_documents.jsonl）",
    )
    parser.add_argument(
        "--source_type",
        type=str,
        default="course_doc",
        help="写入 document 源的 source_type（默认 course_doc）",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描子目录（默认开启；传了该参数也为 True，保留兼容）",
    )
    parser.add_argument("--limit", type=int, default=None, help="最多解析多少个文件（调试用）")
    parser.add_argument(
        "--pdf_min_text_chars",
        type=int,
        default=120,
        help="PDF 文本太短时触发 OCR fallback 的阈值（当前仅预留，不接真实 OCR）",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"input_dir not found or not a directory: {in_dir}")

    out_path = Path(args.output)
    parser_impl = DocumentParser(default_source_type=str(args.source_type), pdf_min_text_chars_for_non_ocr=int(args.pdf_min_text_chars))

    rows: List[Dict[str, Any]] = []
    n = 0
    skipped = 0
    failed = 0

    # 默认递归扫描（面试/演示更符合“原始资料目录”）
    for p in iter_files(in_dir, recursive=True):
        if args.limit is not None and n >= int(args.limit):
            break
        if p.suffix.lower() not in SUPPORTED_EXTS:
            skipped += 1
            continue
        try:
            doc = parser_impl.parse_file(p, source_type=str(args.source_type))
            rows.append(doc.to_jsonl_row())
            n += 1
        except Exception as e:
            failed += 1
            # 最小可观测：把错误写入一条“失败记录”，便于定位具体文件
            rows.append(
                {
                    "record_id": f"parse_failed::{p.as_posix()}",
                    "source_type": str(args.source_type),
                    "title": p.stem,
                    "raw_text": "",
                    "metadata": {"file_path": str(p.as_posix()), "error": repr(e)},
                    "parse_method": "failed",
                    "ocr_used": False,
                    "file_name": p.name,
                    "file_type": p.suffix.lstrip("."),
                }
            )

    write_jsonl(out_path, rows)
    print("Parse done.")
    print(f"- input_dir: {in_dir}")
    print(f"- output: {out_path}")
    print(f"- parsed: {n}")
    print(f"- skipped_non_supported: {skipped}")
    print(f"- failed: {failed}")


if __name__ == "__main__":
    main()

