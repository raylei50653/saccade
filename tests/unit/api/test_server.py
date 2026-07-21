"""Unit tests for the FastAPI inference/state server (saccade.api.server)."""

# scope: api
# function: behavior
# lifecycle: active

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

import saccade.api.server as server


class StubRedis:
    def __init__(
        self,
        *,
        active_objects: list[int] | None = None,
        history: dict[int, dict[str, Any] | None] | None = None,
        active_error: Exception | None = None,
        history_error: Exception | None = None,
    ) -> None:
        self.active_objects = active_objects or []
        self.history = history or {}
        self.active_error = active_error
        self.history_error = history_error
        self.connected = False
        self.disconnected = False

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnected = True

    async def get_active_objects(self) -> list[int]:
        if self.active_error is not None:
            raise self.active_error
        return self.active_objects

    async def get_object_history(self, obj_id: int) -> dict[str, Any] | None:
        if self.history_error is not None:
            raise self.history_error
        return self.history.get(obj_id)


class StubChroma:
    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.result = result or {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        self.calls: list[dict[str, Any]] = []

    def hybrid_query(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.result


@pytest.fixture
def installed_stubs(monkeypatch: pytest.MonkeyPatch) -> tuple[StubRedis, StubChroma]:
    redis = StubRedis()
    chroma = StubChroma()
    monkeypatch.setattr(server, "_redis_cache", redis)
    monkeypatch.setattr(server, "_chroma_store", chroma)
    return redis, chroma


def test_root_uses_startup_and_shutdown(
    installed_stubs: tuple[StubRedis, StubChroma],
) -> None:
    redis, _ = installed_stubs

    with TestClient(server.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "status": "online",
        "system": "Saccade",
        "api_version": "1.0",
    }
    assert redis.connected is True
    assert redis.disconnected is True


def test_list_active_objects_returns_count(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = StubRedis(active_objects=[3, 7, 9])
    monkeypatch.setattr(server, "_redis_cache", redis)
    monkeypatch.setattr(server, "_chroma_store", StubChroma())

    with TestClient(server.app) as client:
        response = client.get("/objects")

    assert response.status_code == 200
    assert response.json() == {"count": 3, "active_objects": [3, 7, 9]}


def test_list_active_objects_maps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = StubRedis(active_error=RuntimeError("redis down"))
    monkeypatch.setattr(server, "_redis_cache", redis)
    monkeypatch.setattr(server, "_chroma_store", StubChroma())

    with TestClient(server.app) as client:
        response = client.get("/objects")

    assert response.status_code == 500
    assert response.json() == {"detail": "redis down"}


def test_get_object_history_adds_dwell_time(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = StubRedis(
        history={
            42: {
                "id": 42,
                "label": "person",
                "first_seen": 10.0,
                "last_seen": 15.236,
            }
        }
    )
    monkeypatch.setattr(server, "_redis_cache", redis)
    monkeypatch.setattr(server, "_chroma_store", StubChroma())

    with TestClient(server.app) as client:
        response = client.get("/objects/42")

    assert response.status_code == 200
    assert response.json()["dwell_time_seconds"] == 5.24


def test_get_object_history_returns_404_for_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "_redis_cache", StubRedis(history={42: None}))
    monkeypatch.setattr(server, "_chroma_store", StubChroma())

    with TestClient(server.app) as client:
        response = client.get("/objects/42")

    assert response.status_code == 404
    assert response.json() == {"detail": "Object 42 not found or expired."}


def test_get_object_history_maps_redis_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    redis = StubRedis(history_error=RuntimeError("history failed"))
    monkeypatch.setattr(server, "_redis_cache", redis)
    monkeypatch.setattr(server, "_chroma_store", StubChroma())

    with TestClient(server.app) as client:
        response = client.get("/objects/42")

    assert response.status_code == 500
    assert response.json() == {"detail": "history failed"}


def test_semantic_search_formats_chroma_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chroma = StubChroma(
        {
            "ids": [["m1"]],
            "documents": [["Scene contains: 1 person."]],
            "metadatas": [[{"frame_id": 12}]],
            "distances": [[0.123]],
        }
    )
    monkeypatch.setattr(server, "_redis_cache", StubRedis())
    monkeypatch.setattr(server, "_chroma_store", chroma)

    with TestClient(server.app) as client:
        response = client.post(
            "/search",
            json={
                "text": "person",
                "n_results": 3,
                "start_time": 0.0,
                "is_anomaly": False,
            },
        )

    assert response.status_code == 200
    assert chroma.calls == [
        {
            "query_text": "person",
            "n_results": 3,
            "start_time": 0.0,
            "is_anomaly": 0,
        }
    ]
    assert response.json() == {
        "query": "person",
        "results": [
            {
                "id": "m1",
                "content": "Scene contains: 1 person.",
                "metadata": {"frame_id": 12},
                "distance": 0.123,
            }
        ],
    }


def test_semantic_search_defaults_n_results_only_when_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chroma = StubChroma()
    monkeypatch.setattr(server, "_redis_cache", StubRedis())
    monkeypatch.setattr(server, "_chroma_store", chroma)

    with TestClient(server.app) as client:
        response = client.post("/search", json={"text": "person", "n_results": None})

    assert response.status_code == 200
    assert chroma.calls[0]["n_results"] == 5


def test_semantic_search_maps_chroma_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingChroma:
        def hybrid_query(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("chroma failed")

    monkeypatch.setattr(server, "_redis_cache", StubRedis())
    monkeypatch.setattr(server, "_chroma_store", FailingChroma())

    with TestClient(server.app) as client:
        response = client.post("/search", json={"text": "person"})

    assert response.status_code == 500
    assert response.json() == {"detail": "chroma failed"}
