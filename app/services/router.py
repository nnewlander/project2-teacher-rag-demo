from __future__ import annotations

import re
from dataclasses import dataclass

from app.utils.text_utils import normalize_text


_FAQ_HINT_RE = re.compile(r"(哪里|怎么|如何|为什么|入口|找不到|报错|错误|失败|步骤|设置|登录|作业|发布)")
_CLARIFY_PRONOUN_RE = re.compile(r"(这个|那个|这样|那样|这种|怎么弄|怎么办|为啥|咋办)")
_ERROR_WORD_RE = re.compile(r"(报错|错误|异常|失败|闪退)")
_ERROR_TYPE_RE = re.compile(r"([A-Za-z]+Error|Exception|Traceback)", re.IGNORECASE)
_OBJECT_HINT_RE = re.compile(r"(Scratch|Python|作业|班级|登录|教师端|学生端|平台|批改台|工作台|课程|课件|投屏)", re.IGNORECASE)
_HYDE_HINT_RE = re.compile(r"(如何|怎样|我想|希望|需要|场景|背景|方案|策略|设计|讲法|讲解|过渡|串起来|不顺|没有思路|建议)")
_SUBQUERY_HINT_RE = re.compile(r"(同时|并且|另外|以及|还有|顺便|然后|分别|两个问题|三个问题|一方面|另一方面)")


@dataclass(frozen=True)
class RouteDecision:
    route: str  # "faq_first" | "hybrid"
    reason: str
    # 用于离线评测集的标签（route_eval_queries_10pct.jsonl）
    # 当前 Router 只实现最小两类，因此只会输出 bm25_faq / rag_standard
    # TODO: 扩展 hyde/subquery/backtrack/need_clarify 等策略路由标签
    eval_label: str


class Router:
    """
    最小规则路由（MVP）。

    - 如果问题“像标准 FAQ”（短、带典型问法/关键词），先走 FAQ 检索
    - 否则走混合检索（FAQ 候选 + 向量候选）

    TODO:
    - 接入查询分类器（intent: platform_usage/code_error/course_design/py_syntax）
    - 策略路由：HyDE、子查询、回溯检索、对话态
    """

    def decide(self, query: str, *, query_type: str | None = None) -> RouteDecision:
        q = normalize_text(query)
        if not q:
            return RouteDecision(route="hybrid", reason="empty_query", eval_label="rag_standard")

        # 显式 query classification 的最小接入：
        # - need_clarify：直接短路（不进入检索/LLM）
        # - faq_like：偏向走 faq_first
        if query_type == "need_clarify":
            return RouteDecision(route="need_clarify", reason="classifier_need_clarify", eval_label="need_clarify")
        if query_type == "faq_like":
            return RouteDecision(route="faq_first", reason="classifier_faq_like", eval_label="bm25_faq")

        # subquery：多意图/多子问题的最小策略（不与 HyDE 做复杂联动）
        # 触发条件极简：出现并列/连接结构，且文本不算太短
        qmark_cnt = q.count("？") + q.count("?")
        has_connector = bool(_SUBQUERY_HINT_RE.search(q))
        if len(q) >= 25 and (has_connector or qmark_cnt >= 2):
            return RouteDecision(route="subquery", reason="multi_intent_subquery", eval_label="subquery")

        # HyDE：仅对语义型问题启用；FAQ-like 与 need_clarify 不走 HyDE
        # 触发条件保持极简：query_type=semantic_retrieval 且文本较长/表达不稳定
        if query_type == "semantic_retrieval":
            is_longish = len(q) >= 35
            has_semantic_hint = bool(_HYDE_HINT_RE.search(q))
            if is_longish and has_semantic_hint:
                return RouteDecision(route="hyde", reason="semantic_longish_hyde", eval_label="hyde")

        is_short = len(q) <= 30
        has_hint = bool(_FAQ_HINT_RE.search(q))
        looks_like_question = "？" in q or "?" in q or q.endswith(("吗", "么"))

        # need_clarify：信息明显不足时先澄清（不直接检索生成）
        # 规则：指代不明/过短泛问/提到报错但没给错误类型或关键信息
        very_short = len(q) <= 10
        has_pronoun = bool(_CLARIFY_PRONOUN_RE.search(q))
        mentions_error = bool(_ERROR_WORD_RE.search(q))
        has_error_type = bool(_ERROR_TYPE_RE.search(q))
        has_object = bool(_OBJECT_HINT_RE.search(q))

        # 1) 过短且像泛问/指代不明：优先澄清
        # - 有指代词（这个/那个/这样…）时优先 need_clarify
        # - 或者：非常短、并且不包含典型 FAQ 提示词时，且像问题句
        if very_short and not has_object and (has_pronoun or (looks_like_question and not has_hint)):
            return RouteDecision(route="need_clarify", reason="too_short_or_ambiguous", eval_label="need_clarify")

        # 2) 说“报错/异常”但没有错误类型/Traceback/代码信息（粗略判断）
        if mentions_error and not has_error_type and ("代码" not in q) and ("截图" not in q) and len(q) <= 40:
            return RouteDecision(route="need_clarify", reason="missing_error_details", eval_label="need_clarify")

        if (is_short and has_hint) or (looks_like_question and has_hint):
            return RouteDecision(route="faq_first", reason="faq_like_rule", eval_label="bm25_faq")
        return RouteDecision(route="hybrid", reason="default_fallback", eval_label="rag_standard")

