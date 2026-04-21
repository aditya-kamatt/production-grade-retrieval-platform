import pytest
from fastapi.testclient import TestClient

import app.main as main_module


class FakeSearchService:
    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture
def client(monkeypatch):
    # Patch SearchService so app startup doesn't load real models
    monkeypatch.setattr(
        "app.main.SearchService",
        FakeSearchService,
    )

    test_app = main_module.app
    return TestClient(test_app)


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    assert data["message"] == "LEC Search API is running"
    assert data["docs_url"] == "/docs"
    assert data["search_endpoint"] == "/search"