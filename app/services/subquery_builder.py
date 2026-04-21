from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from app.utils.text_utils import normalize_text


_SPLIT_PUNCT_RE = re.compile(r"[；;\n。]+")
_QUESTION_SPLIT_RE = re.compile(r"[？?]+")

# 连接词/并列提示：用于把一条长 query 拆成多个可检索的子查询
_CONNECTOR_RE = re.compile(r"(同时|并且|另外|以及|还有|再者|顺便|然后|并行|一方面|另一方面)")


@dataclass(frozen=True)
class SubqueryResult:
    subqueries: List[str]
    debug: Dict[str, Any]


class SubqueryBuilder:
    """
    规则版 subquery 拆分（MVP）：
    - 不依赖 agent/多轮推理
    - 仅按标点/连接词/并列结构做最小拆分
    - 控制数量：最多 3 条（2~3 条为主）
    """

    def __init__(self, *, max_subqueries: int = 3, min_len: int = 4) -> None:
        self.max_subqueries = max(1, int(max_subqueries))
        self.min_len = max(1, int(min_len))

    def build(self, query: str) -> SubqueryResult:
        q0 = normalize_text(query)
        if not q0:
            return SubqueryResult(subqueries=[], debug={"reason": "empty"})

        # 1) 标点拆分
        q1 = _SPLIT_PUNCT_RE.sub("\n", q0)
        q1 = _QUESTION_SPLIT_RE.sub("\n", q1)

        # 2) 连接词拆分（用换行作为分隔符）
        q2 = _CONNECTOR_RE.sub("\n", q1)

        parts = [p.strip(" ，,、:：-—\t") for p in q2.split("\n")]
        parts = [p for p in parts if p]

        # 3) 去重（保持顺序）+ 过滤过短
        out: List[str] = []
        seen = set()
        for p in parts:
            p = normalize_text(p)
            if not p or len(p) < self.min_len:
                continue
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
            if len(out) >= self.max_subqueries:
                break

        # 若没拆出 2 条以上，则认为不适合 subquery
        if len(out) < 2:
            return SubqueryResult(subqueries=[q0], debug={"reason": "no_split", "n": 1})

        return SubqueryResult(subqueries=out, debug={"reason": "split", "n": len(out)})

