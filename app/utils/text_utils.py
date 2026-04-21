from __future__ import annotations

import re
from typing import Iterable, List


_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    最小规范化：去两端空白、合并多空白、统一常见全角空格。

    TODO:
    - 中文标点/全半角进一步规范化
    - 简单分句/标题抽取
    """
    if text is None:
        return ""
    text = text.replace("\u3000", " ")
    text = text.strip()
    text = _WS_RE.sub(" ", text)
    return text


def simple_tokenize_zh(text: str) -> List[str]:
    """
    MVP 的中文/混合文本分词：
    - 英文/数字按连续块切分
    - 中文按单字切分（非常粗糙，但可跑通 BM25）

    TODO:
    - 替换为更好的分词：jieba / pkuseg / tokenizer + ngram
    """
    text = normalize_text(text)
    if not text:
        return []

    tokens: List[str] = []
    buf: List[str] = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            if buf:
                tokens.append("".join(buf).lower())
                buf.clear()
            tokens.append(ch)
        elif ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                tokens.append("".join(buf).lower())
                buf.clear()
    if buf:
        tokens.append("".join(buf).lower())
    return [t for t in tokens if t]


def dedup_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

