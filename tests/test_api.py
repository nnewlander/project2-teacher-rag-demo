from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_health() -> None:
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ask() -> None:
    client = TestClient(create_app())
    r = client.post("/ask", json={"query": "老师在作业批改台里找不到作业发布入口，是入口改版了吗？"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert data["mode"] in ("faq", "rag_mock")
    assert "route" in data


def test_search() -> None:
    client = TestClient(create_app())
    r = client.post(
        "/search",
        json={
            "query": "课堂演示遇到 NameError，应该怎么给学生解释？",
            "top_k": 3,
            "filters": {},
            "request_id": "test-search-001",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "hits" in data
    assert "query" in data
    assert "route_trace" in data
    assert "debug" in data
    if data["hits"]:
        h = data["hits"][0]
        for k in ("source_id", "title", "snippet", "score", "source_type", "metadata"):
            assert k in h
