from __future__ import annotations

from app.config import get_settings
from app.services.hybrid_retriever import HybridResult
from app.services.qa_service import QAService


class _DummyHybrid:
    def retrieve(self, query: str, faq_top_k: int, vec_top_k: int, hybrid_top_k: int) -> HybridResult:  # noqa: ARG002
        # 返回空 contexts，避免触发向量模型/FAISS 等重依赖
        return HybridResult(contexts=[], debug={"timing_ms": {"faq": 0.0, "vector": 0.0, "rerank": 0.0}, "rerank": {"enabled": False}})


def test_hyde_route_trace_smoke_no_llm() -> None:
    settings = get_settings()
    # 确保测试环境不依赖外部 LLM
    settings.llm_provider = "disabled"
    settings.llm_model_name = ""

    qa = QAService(settings)
    qa._ready = True  # 跳过 init_kb（避免向量模型下载/建索引）
    qa.hybrid = _DummyHybrid()  # type: ignore[assignment]
    qa.settings.backtrack_enabled = False

    q = "我想把输入输出和while循环串起来讲，但过渡不顺，想要一个更自然的课堂讲解策略和示例。"
    resp = qa.ask(q, top_k=3)

    steps = [x.step for x in resp.route_trace]
    assert "hyde.generate" in steps
    assert "hyde.retrieve" in steps
    # 未配置 LLM 时应该安全降级：route_trace 中 hyde.generate.enabled=False
    hyde_items = [x for x in resp.route_trace if x.step == "hyde.generate"]
    assert hyde_items and hyde_items[0].detail.get("enabled") is False

