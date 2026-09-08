from __future__ import annotations

from fastapi.testclient import TestClient

from short_video_recsys.serving.app import create_app


class FakePipeline:
    backend_name = "fake"

    def recommend(self, user_id: int, top_k: int, tab: int):
        if user_id == 9999:
            return None
        return {
            "user_id": user_id,
            "as_of_ms": 123,
            "items": [{"video_id": index, "score": 1.0} for index in range(top_k)],
            "feature_store": self.backend_name,
            "latency_ms": {"total": 1.0},
            "tab": tab,
        }


def test_health_and_recommendation_contract() -> None:
    with TestClient(create_app(FakePipeline())) as client:
        assert client.get("/health").json() == {
            "status": "ok",
            "feature_store": "fake",
        }
        response = client.get("/recommend/42", params={"top_k": 3, "tab": 1})

    assert response.status_code == 200
    assert response.json()["user_id"] == 42
    assert len(response.json()["items"]) == 3


def test_recommendation_validates_inputs_and_unknown_users() -> None:
    with TestClient(create_app(FakePipeline())) as client:
        assert client.get("/recommend/42", params={"top_k": 0}).status_code == 422
        assert client.get("/recommend/9999").status_code == 404
