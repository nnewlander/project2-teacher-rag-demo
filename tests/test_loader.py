from __future__ import annotations

from app.config import get_settings
from app.services.qa_service import QAService


def test_kb_init_smoke() -> None:
    qa = QAService(get_settings())
    artifacts = qa.init_kb(limit=50)
    assert len(artifacts.docs) > 0
    assert artifacts.faq_count > 0
    assert artifacts.chunks_count > 0


def test_backtrack_smoke() -> None:
    # 让 backtrack 更容易触发：把阈值设置得很高
    s = get_settings().model_copy(update={"backtrack_min_top_score": 1e9, "backtrack_enabled": True})
    qa = QAService(s)
    qa.init_kb(limit=80)
    resp = qa.ask("Scratch图形化是什么？", top_k=5)
    steps = [t.step for t in resp.route_trace]
    assert "backtrack" in steps

