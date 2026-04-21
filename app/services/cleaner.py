from __future__ import annotations

import hashlib
from typing import Iterable, List, Set, Tuple

from app.schemas.document import InternalDocument
from app.utils.text_utils import normalize_text


class Cleaner:
    """
    MVP 清洗：
    - 去空白/简单规范化
    - 基于 (source, text_hash) 去重

    TODO:
    - 更精细的重复检测（duplicate_group_id、相似度）
    - 噪声过滤（noise_flags）与质量评分
    """

    def clean_documents(self, docs: Iterable[InternalDocument]) -> List[InternalDocument]:
        out: List[InternalDocument] = []
        seen: Set[Tuple[str, str]] = set()

        for d in docs:
            text = normalize_text(d.text)
            if not text:
                continue

            h = hashlib.md5(text.encode("utf-8")).hexdigest()
            key = (d.source, h)
            if key in seen:
                continue
            seen.add(key)

            out.append(
                d.model_copy(
                    update={
                        "title": normalize_text(d.title),
                        "text": text,
                    }
                )
            )
        return out

