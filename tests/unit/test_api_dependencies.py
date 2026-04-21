import pytest
from fastapi import HTTPException
from types import SimpleNamespace

from app.api.dependencies import get_search_service


def test_get_search_service_returns_instance():
    fake_service = object()

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(search_service=fake_service))
    )

    result = get_search_service(request)

    assert result is fake_service


def test_get_search_service_raises_if_missing():
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(search_service=None))
    )

    with pytest.raises(HTTPException) as exc:
        get_search_service(request)

    assert exc.value.status_code == 500
    assert "Search service not initialized" in exc.value.detail