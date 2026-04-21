from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from app.utils.text_utils import normalize_text


QueryType = Literal["faq_like", "platform_usage", "code_error", "semantic_retrieval", "need_clarify"]


_FILLER_PREFIX_RE = re.compile(
    r"^(请问|麻烦|帮我|帮忙|老师好|您好|你好|想问下|想问一下|请教一下|求助|咨询下|咨询一下)\s*"
)
_FILLER_SUFFIX_RE = re.compile(r"\s*(谢谢|谢谢你|感谢|多谢|哈|呀|呢|哦|噢|啦|吧|嘛|啊)\s*$")
_MULTI_SPACE_RE = re.compile(r"\s+")

_PRONOUN_AMBIG_RE = re.compile(r"(这个|那个|这样|那样|这种|咋办|怎么办|怎么弄|为啥|为什么不行)")

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_TRACEBACK_RE = re.compile(r"(traceback\s*\(most recent call last\)|stack trace|exception)", re.IGNORECASE)
_ERROR_TYPE_RE = re.compile(
    r"\b([A-Za-z_]+Error|Exception|Traceback|SyntaxError|IndentationError|TypeError|NameError|ValueError|KeyError|"
    r"IndexError|AttributeError|ModuleNotFoundError|ImportError)\b"
)
_LINE_HINT_RE = re.compile(r"\bline\s*\d+\b", re.IGNORECASE)

_PLATFORM_HINT_RE = re.compile(
    r"(登录|入口|哪里|怎么进入|找不到|教师端|学生端|工作台|批改台|作业|发布|班级|课程|课件|投屏|账号|权限|绑定|导入|导出|下载|安装|浏览器|小程序|APP)",
    re.IGNORECASE,
)

_FAQ_LIKE_HINT_RE = re.compile(r"(哪里|怎么|如何|为什么|入口|找不到|步骤|设置|登录|作业|发布|批改)")


@dataclass(frozen=True)
class QueryProcessingResult:
    cleaned_query: str
    query_type: QueryType
    debug: Dict[str, Any]


class QueryProcessor:
    """
    轻量 query 处理（MVP）：
    - 基础清洗：去重复空白、去明显口语噪声
    - 规则分类：faq_like / platform_usage / code_error / semantic_retrieval / need_clarify
    """

    def process(self, query: str) -> QueryProcessingResult:
        raw = query or ""
        q0 = normalize_text(raw)

        q1 = _MULTI_SPACE_RE.sub(" ", q0).strip()
        q2 = _FILLER_PREFIX_RE.sub("", q1).strip()
        q3 = _FILLER_SUFFIX_RE.sub("", q2).strip()

        # 过长的口语前缀可能多次出现，重复清理一次即可（不做循环避免误伤）
        q4 = _FILLER_PREFIX_RE.sub("", q3).strip()

        qt, reason, signals = self._classify(q4)
        return QueryProcessingResult(
            cleaned_query=q4,
            query_type=qt,
            debug={
                "reason": reason,
                "signals": signals,
                "raw_len": len(raw),
                "cleaned_len": len(q4),
            },
        )

    def _classify(self, q: str) -> tuple[QueryType, str, Dict[str, Any]]:
        if not q:
            return "need_clarify", "empty_after_clean", {"empty": True}

        very_short = len(q) <= 10
        short = len(q) <= 30
        looks_like_question = ("？" in q) or ("?" in q) or q.endswith(("吗", "么"))

        has_pronoun = bool(_PRONOUN_AMBIG_RE.search(q))

        has_code_block = bool(_CODE_BLOCK_RE.search(q))
        has_traceback = bool(_TRACEBACK_RE.search(q))
        has_error_type = bool(_ERROR_TYPE_RE.search(q))
        has_line_hint = bool(_LINE_HINT_RE.search(q))
        mentions_error_cn = any(x in q for x in ["报错", "错误", "异常", "失败", "闪退"])
        codeish = has_code_block or has_traceback or has_error_type or has_line_hint

        has_platform_hint = bool(_PLATFORM_HINT_RE.search(q))
        has_faq_hint = bool(_FAQ_LIKE_HINT_RE.search(q))

        # 1) need_clarify：过短 + 指代不明 / 泛问
        if very_short and (has_pronoun or (looks_like_question and not has_platform_hint and not codeish and not has_faq_hint)):
            return "need_clarify", "too_short_or_ambiguous", {"very_short": True, "pronoun": has_pronoun}

        # 2) code_error：明确错误类型/Traceback/代码块（优先级最高）
        if codeish or (mentions_error_cn and (has_error_type or has_line_hint)):
            return "code_error", "error_signals", {
                "code_block": has_code_block,
                "traceback": has_traceback,
                "error_type": has_error_type,
                "line_hint": has_line_hint,
            }

        # 3) platform_usage：平台入口/操作类
        if has_platform_hint:
            return "platform_usage", "platform_keywords", {"platform_hint": True}

        # 4) faq_like：短问句 + 常见提示词
        if (short and has_faq_hint) or (looks_like_question and has_faq_hint):
            return "faq_like", "faq_like_rule", {"short": short, "faq_hint": True}

        # 5) 默认：语义检索（概念/长问/描述性问题）
        return "semantic_retrieval", "default", {"short": short, "looks_like_question": looks_like_question}

