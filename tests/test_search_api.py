from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.api import routes
from app.config import get_settings
from app.main import create_app
from app.services.qa_service import QAService
from scripts.profile_search_api import call


def _assert_hit_schema(hit: dict) -> None:
    for k in ("source_id", "title", "snippet", "score", "source_type", "metadata"):
        assert k in hit


def test_search_nameerror_top_hit() -> None:
    routes._qa = QAService(get_settings())
    client = TestClient(create_app())
    r = client.post(
        "/search",
        json={
            "query": "课堂演示遇到 NameError，应该怎么给学生解释？ NameError 变量 命名规范 报错排查",
            "top_k": 3,
            "filters": {},
            "request_id": "test-nameerror",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "hits" in data and isinstance(data["hits"], list)
    assert len(data["hits"]) >= 1
    top = data["hits"][0]
    _assert_hit_schema(top)
    text = f"{top.get('title', '')} {top.get('snippet', '')}"
    assert ("NameError" in text) or ("变量未定义" in text)
    assert "debug" in data
    assert data["debug"].get("detected_error_type") == "NameError"
    assert "expanded_terms" in data["debug"]
    assert "top_hit_reason" in data["debug"]


def test_search_typeerror_top_hit() -> None:
    routes._qa = QAService(get_settings())
    client = TestClient(create_app())
    r = client.post(
        "/search",
        json={
            "query": "课堂上 Python 代码报 TypeError 应该怎么解释？请给出类型错误排查步骤",
            "top_k": 3,
            "filters": {},
            "request_id": "test-typeerror",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data.get("hits") or []) >= 1
    top = data["hits"][0]
    _assert_hit_schema(top)
    text = f"{top.get('title', '')} {top.get('snippet', '')}"
    assert ("TypeError" in text) or ("类型错误" in text)


def test_search_response_compatibility_fields() -> None:
    client = TestClient(create_app())
    r = client.post(
        "/search",
        json={"query": "NameError 是什么", "top_k": 2, "filters": {}, "request_id": "test-compat"},
    )
    assert r.status_code == 200
    data = r.json()
    for k in ("hits", "query", "route_trace", "debug"):
        assert k in data
    if data["hits"]:
        _assert_hit_schema(data["hits"][0])


def test_health_fast_response() -> None:
    client = TestClient(create_app())
    t0 = time.perf_counter()
    r = client.get("/health")
    cost = time.perf_counter() - t0
    assert r.status_code == 200
    assert cost < 1.0
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "project2_rag"


def test_ready_fast_response() -> None:
    client = TestClient(create_app())
    t0 = time.perf_counter()
    r = client.get("/ready")
    cost = time.perf_counter() - t0
    assert r.status_code == 200
    assert cost < 1.0
    data = r.json()
    assert data["status"] in ("ready", "partial_ready", "fallback_only", "not_ready")
    for k in ("faq_ready", "bm25_ready", "vector_ready", "model_loaded", "lightweight_ready", "fallback_ready"):
        assert k in data
    for k in ("lightweight_search_ready", "full_rag_ready", "serving_mode"):
        assert k in data
    for k in ("faq_doc_count", "bm25_doc_count", "last_warmup_error", "last_warmup_cost_ms"):
        assert k in data


def test_search_no_llm_and_has_timing() -> None:
    client = TestClient(create_app())
    r = client.post(
        "/search",
        json={"query": "NameError 是什么", "top_k": 2, "filters": {}, "request_id": "test-timing"},
    )
    assert r.status_code == 200
    data = r.json()
    debug = data.get("debug") or {}
    assert debug.get("llm_called") is False
    timing = debug.get("timing_ms") or {}
    for k in (
        "total",
        "detect_error_type",
        "fallback_build",
        "faq_search",
        "phrase_match",
        "rerank_boost",
        "hybrid_retrieve",
        "timeout_guard",
    ):
        assert k in timing


def test_error_type_fast_path_used() -> None:
    client = TestClient(create_app())
    r = client.post(
        "/search",
        json={"query": "NameError 变量未定义如何讲解", "top_k": 3, "filters": {}, "request_id": "test-fast-path"},
    )
    assert r.status_code == 200
    debug = (r.json().get("debug") or {})
    assert debug.get("detected_error_type") == "NameError"
    assert "fast_path_used" in debug
    assert debug.get("vector_optional") is True
    assert "hybrid_skipped" in debug
    assert "fallback_inserted" in debug


def test_search_nameerror_not_ready_returns_fallback_fast() -> None:
    routes._qa = QAService(get_settings())
    client = TestClient(create_app())
    t0 = time.perf_counter()
    r = client.post(
        "/search",
        json={"query": "NameError 变量未定义怎么解释", "top_k": 3, "filters": {}, "request_id": "test-not-ready-fallback"},
    )
    assert r.status_code == 200
    assert (time.perf_counter() - t0) < 2.0
    data = r.json()
    assert len(data.get("hits") or []) >= 1
    first = data["hits"][0]
    assert str(first.get("source_id", "")).startswith("FALLBACK-NameError")
    debug = data.get("debug") or {}
    assert debug.get("fast_path_used") is True
    assert debug.get("fallback_inserted") is True
    assert debug.get("fallback_only") is True
    assert debug.get("fallback_reason") == "faq_bm25_not_ready"


def test_warmup_enables_real_faq_and_search_prefers_real_hit() -> None:
    routes._qa = QAService(get_settings())
    client = TestClient(create_app())
    w = client.get("/warmup")
    assert w.status_code == 200
    wj = w.json()
    assert wj["status"] == "ok"
    assert wj["lightweight_search_ready"] is True
    assert wj["full_rag_ready"] is False
    assert wj["serving_mode"] == "lightweight_search"

    r = client.get("/ready")
    assert r.status_code == 200
    ready = r.json()
    assert ready["faq_ready"] is True
    assert ready["bm25_ready"] is True
    assert ready["lightweight_ready"] is True
    assert ready["lightweight_search_ready"] is True
    assert ready["full_rag_ready"] is False
    assert ready["status"] == "partial_ready"
    assert ready["serving_mode"] == "lightweight_search"

    s = client.post(
        "/search",
        json={"query": "课堂演示 NameError 变量未定义 怎么解释", "top_k": 3, "filters": {}, "request_id": "test-real-faq"},
    )
    assert s.status_code == 200
    data = s.json()
    assert len(data.get("hits") or []) >= 1
    top = data["hits"][0]
    # warmup 会注入 seed FAQ，因此 top hit 应为真实 faq，而不是 fallback_error_guide
    assert top["source_type"] != "fallback_error_guide"
    debug = data.get("debug") or {}
    assert debug.get("real_retrieval_used") is True
    assert debug.get("real_hit_count", 0) >= 1


def test_project1_smoke_query_prefers_real_faq_over_fallback() -> None:
    routes._qa = QAService(get_settings())
    client = TestClient(create_app())
    w = client.get("/warmup")
    assert w.status_code == 200

    s = client.post(
        "/search",
        json={
            "query": "课堂演示遇到 NameError，应该怎么给学生解释？ NameError 变量 命名规范 报错排查",
            "top_k": 3,
            "filters": {},
            "request_id": "test-project1-smoke",
        },
    )
    assert s.status_code == 200
    data = s.json()
    assert len(data.get("hits") or []) >= 1
    top = data["hits"][0]
    text = f"{top.get('title','')} {top.get('snippet','')}"
    assert top.get("source_id") != "FALLBACK-NameError"
    assert ("NameError" in text) or ("变量未定义" in text) or ("函数名未定义" in text)
    debug = data.get("debug") or {}
    assert debug.get("final_top1_is_fallback") is False


def test_lightweight_mode_never_uses_vector_or_loads_model() -> None:
    import os

    os.environ["KBQA_SEARCH_MODE"] = "lightweight"
    get_settings.cache_clear()
    routes._qa = QAService(get_settings())
    client = TestClient(create_app())

    w = client.get("/warmup")
    assert w.status_code == 200

    s = client.post(
        "/search",
        json={"query": "NameError 变量未定义 函数名未定义 课堂解释", "top_k": 3, "filters": {}, "request_id": "test-lightweight"},
    )
    assert s.status_code == 200
    debug = s.json().get("debug") or {}
    assert debug.get("search_mode") == "lightweight"
    assert debug.get("vector_used") is False
    assert debug.get("model_loaded_this_request") is False


def test_profile_call_timeout_not_crash() -> None:
    out = call("GET", "http://127.0.0.1:1/health", case_name="timeout-case", timeout_s=0.001)
    assert out.get("ok") is False
    assert out.get("timeout") is True
