from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


def read_jsonl(path: Path, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """
    读取 jsonl（每行一个 JSON 对象）。

    TODO:
    - 增加健壮性：跳过坏行并记录
    - 支持 gzip/zip 等压缩格式
    """
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if limit is not None and n >= limit:
                return


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default)

