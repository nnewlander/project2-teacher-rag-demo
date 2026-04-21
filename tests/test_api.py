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
