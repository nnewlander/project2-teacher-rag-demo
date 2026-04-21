from __future__ import annotations

from app.services.router import Router


def test_router_faq_like() -> None:
    r = Router()
    d = r.decide("\u5165\u53e3\u5728\u54ea\u91cc\uff1f")  # 入口在哪里？
    assert d.route == "faq_first"


def test_router_default() -> None:
    r = Router()
    d = r.decide("我想把输入输出和while循环串起来讲，但过渡不顺，求课程设计建议。")
    assert d.route in ("faq_first", "hybrid", "hyde")


def test_router_need_clarify() -> None:
    r = Router()
    # 用 unicode escape 避免 Windows 环境编码导致用例乱码
    d = r.decide("\u8fd9\u4e2a\u600e\u4e48\u5f04\uff1f")  # 这个怎么弄？
    assert d.route == "need_clarify"


def test_router_hyde_semantic() -> None:
    r = Router()
    q = "我想把输入输出和while循环串起来讲，但过渡不顺，想要一个更自然的课堂讲解策略和示例。"
    d = r.decide(q, query_type="semantic_retrieval")
    assert d.route == "hyde"

