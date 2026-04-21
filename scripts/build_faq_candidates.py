from __future__ import annotations

"""
FAQ 候选沉淀脚本（最小版）

目标：
- 从历史工单 raw_support_tickets_10pct.jsonl 中抽取高频 FAQ 候选
- 输出可人工审核的候选文件（jsonl / csv）

约束：
- 不接 LLM
- 不写回正式 FAQ 库
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from app.config import get_settings
from app.utils.jsonl import read_jsonl, write_jsonl
from app.utils.text_utils import dedup_keep_order, normalize_text


_PUNCT_RE = re.compile(r"[，。！？、；：,.!?;:\-—()\[\]{}<>\"'“”‘’·…/\\|]+")
_MULTI_WS_RE = re.compile(r"\s+")


def normalize_question(q: str) -> str:
    """
    面向 FAQ 聚合的更强规范化（尽量把“同问不同写法”聚到一起）。

    TODO:
    - 加入同义词/别名表（入口=位置=在哪里）
    - 对“课程阶段/模块名”做标准化映射
    """
    q = normalize_text(q)
    q = q.lower()
    q = _PUNCT_RE.sub(" ", q)
    q = _MULTI_WS_RE.sub(" ", q).strip()
    return q


def shorten(text: str, n: int = 240) -> str:
    t = normalize_text(text).replace("\n", " ")
    if len(t) <= n:
        return t
    return t[:n] + "…"


def extract_question(row: Dict[str, Any]) -> str:
    return str(row.get("raw_question") or "").strip()


def extract_resolution(row: Dict[str, Any]) -> str:
    return str(row.get("raw_resolution_note") or "").strip()


def build_candidates(
    rows: Iterable[Dict[str, Any]],
    *,
    min_frequency: int,
    max_samples: int,
    max_hint_chars: int,
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[str]] = defaultdict(list)
    hints: Dict[str, Counter[str]] = defaultdict(Counter)

    for r in rows:
        q = extract_question(r)
        if not q:
            continue
        nq = normalize_question(q)
        if not nq:
            continue
        buckets[nq].append(normalize_text(q))

        res = extract_resolution(r)
        if res:
            hint = shorten(res, n=max_hint_chars)
            if hint:
                hints[nq][hint] += 1

    out: List[Dict[str, Any]] = []
    for nq, qs in buckets.items():
        freq = len(qs)
        if freq < min_frequency:
            continue

        # 采样若干原始问法（去重后保序）
        samples = dedup_keep_order(qs)[:max_samples]

        # 候选答案提示：取 resolution_note 最常见的一个短摘要
        hint = ""
        if nq in hints and hints[nq]:
            hint = hints[nq].most_common(1)[0][0]

        out.append(
            {
                "normalized_question": nq,
                "sample_questions": samples,
                "frequency": freq,
                "candidate_answer_hint": hint,
            }
        )

    out.sort(key=lambda x: int(x["frequency"]), reverse=True)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["normalized_question", "frequency", "sample_questions", "candidate_answer_hint"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "normalized_question": r["normalized_question"],
                    "frequency": r["frequency"],
                    "sample_questions": json.dumps(r["sample_questions"], ensure_ascii=False),
                    "candidate_answer_hint": r["candidate_answer_hint"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAQ candidates from support tickets (no LLM).")
    parser.add_argument("--min-frequency", type=int, default=3, help="最小出现次数阈值（默认 3）")
    parser.add_argument("--max-samples", type=int, default=5, help="每个候选保留多少条样例问法（默认 5）")
    parser.add_argument("--max-hint-chars", type=int, default=240, help="答案提示截断长度（默认 240）")
    parser.add_argument("--limit", type=int, default=None, help="最多读取多少条工单（调试用）")
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default="outputs/faq_candidates.jsonl",
        help="输出 jsonl 路径（默认 outputs/faq_candidates.jsonl）",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="outputs/faq_candidates.csv",
        help="输出 csv 路径（默认 outputs/faq_candidates.csv）",
    )
    args = parser.parse_args()

    settings = get_settings()
    src = Path(settings.raw_support_tickets_path)
    rows = list(read_jsonl(src, limit=args.limit))

    candidates = build_candidates(
        rows,
        min_frequency=int(args.min_frequency),
        max_samples=int(args.max_samples),
        max_hint_chars=int(args.max_hint_chars),
    )

    out_jsonl = Path(args.out_jsonl)
    out_csv = Path(args.out_csv)
    write_jsonl(out_jsonl, candidates)
    write_csv(out_csv, candidates)

    print("Done.")
    print(f"- input: {src}")
    print(f"- tickets_read: {len(rows)}")
    print(f"- candidates: {len(candidates)} (min_frequency={args.min_frequency})")
    print(f"- jsonl: {out_jsonl}")
    print(f"- csv: {out_csv}")


if __name__ == "__main__":
    main()

