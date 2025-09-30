"""Integration tests for the public recommendation APIs."""

from __future__ import annotations

import os
from typing import Dict, Any, List

import pytest
from fastapi.testclient import TestClient

# Ensure configuration uses local SQLite + demo API key
os.environ.setdefault("USE_POSTGRES", "false")
os.environ.setdefault("DATABASE_URL_SQLITE", "sqlite:///./ai_recommender.db")
os.environ.setdefault("PLATFORM_API_KEY", "demo-api-key-123")

from app.main import app  # noqa: E402
from scripts.populate_realistic_data import main as populate_realistic_data  # noqa: E402

API_KEY = os.environ["PLATFORM_API_KEY"]
API_HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture(scope="session", autouse=True)
def seed_realistic_demo_data() -> None:
    """Populate the database with the curated realistic dataset once per test session."""
    success = populate_realistic_data()
    assert success, "Failed to populate realistic demo data"


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Provide a FastAPI test client with lifespan support."""
    with TestClient(app) as test_client:
        yield test_client


def _extract_items(results: List[Dict[str, Any]]) -> List[int]:
    return [item["item_id"] for item in results]


def test_recommendations_normalize_student_to_developer(client: TestClient) -> None:
    response = client.post(
        "/api/v1/recommendations",
        json={"user_id": 1, "user_type": "student", "limit": 5},
        headers=API_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["metadata"]["normalized_user_type"] == "developer"
    assert payload["total_results"] > 0

    item_ids = _extract_items(payload["results"])
    assert len(item_ids) == len(set(item_ids)), "Duplicate recommendations returned"
    assert all(item["item_type"] == "position" for item in payload["results"])


def test_founder_receives_developer_profiles(client: TestClient) -> None:
    response = client.post(
        "/api/v1/recommendations",
        json={"user_id": 6, "user_type": "founder", "limit": 5},
        headers=API_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["total_results"] > 0
    assert all(item["item_type"] == "user" for item in payload["results"])
    assert all(
        item["metadata"].get("user_type") == "developer"
        for item in payload["results"]
    )
    assert len(set(_extract_items(payload["results"]))) == len(payload["results"])


def test_investor_receives_startup_matches(client: TestClient) -> None:
    response = client.post(
        "/api/v1/recommendations",
        json={"user_id": 10, "user_type": "investor", "limit": 5},
        headers=API_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["total_results"] > 0
    assert all(item["item_type"] == "startup" for item in payload["results"])
    assert len(set(_extract_items(payload["results"]))) == len(payload["results"])


def test_api_key_is_enforced(client: TestClient) -> None:
    response = client.post(
        "/api/v1/recommendations",
        json={"user_id": 1, "user_type": "developer"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_uc_investor_startups_returns_named_results(client: TestClient) -> None:
    response = client.post(
        "/api/v1/uc/investor/startups",
        json={"investor_id": 10, "limit": 5},
        headers=API_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()

    recs = payload["recommendations"]
    assert recs, "Expected investor use case to return startup matches"
    assert [item["rank"] for item in recs] == list(range(1, len(recs) + 1))
    for item in recs:
        assert item["startup_id"] > 0
        assert item["name"] and not item["name"].startswith("Startup #")
        assert isinstance(item["score"], float)
        assert item.get("industry") is None or isinstance(item["industry"], list)


def test_uc_trending_startups_highlight_popular_companies(client: TestClient) -> None:
    response = client.get(
        "/api/v1/uc/trending",
        params={"item_type": "startup", "limit": 5},
        headers=API_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()

    recs = payload["recommendations"]
    assert recs, "Trending startups should not be empty"
    popularity = [item["popularity"] for item in recs]
    assert popularity == sorted(popularity, reverse=True)
    for item in recs:
        assert item["name"], "Startup name should be populated"
        assert item["popularity"] > 0
        assert item["rank"] >= 1


def test_uc_trending_positions_returns_titles(client: TestClient) -> None:
    response = client.get(
        "/api/v1/uc/trending",
        params={"item_type": "position", "limit": 5},
        headers=API_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()

    recs = payload["recommendations"]
    assert recs, "Trending positions should not be empty"
    popularity = [item["popularity"] for item in recs]
    assert popularity == sorted(popularity, reverse=True)
    for item in recs:
        assert item["title"], "Position title should be populated"
        assert item["popularity"] > 0
        assert item["rank"] >= 1
