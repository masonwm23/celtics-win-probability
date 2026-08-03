"""
Tests for Phase 9c: the API.

The property under test is not "does it return JSON". It is that the interface
can never blur the two kinds of number it serves:

  - timeline probabilities are OUT OF FOLD, from a model that never saw the
    season being replayed;
  - what-if probabilities come from the deployment model, which was fitted on
    all eight seasons and is therefore in-sample for every game here.

Both are legitimate. Confusing them would make the dashboard misleading, so the
caveat is asserted rather than trusted, and partial feature input is required to
fail rather than be filled in.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import api, config


@pytest.fixture
def client(tmp_path, monkeypatch):
    # fastapi and uvicorn are DASHBOARD dependencies, not research ones. Every
    # number in the paper is produced without them, so a machine that has never
    # installed them must still see a green suite. These tests skip rather than
    # fail; the three below that need no server still run.
    pytest.importorskip(
        "fastapi", reason="dashboard only: pip install fastapi uvicorn")
    pytest.importorskip(
        "httpx", reason="fastapi's TestClient needs httpx")
    from fastapi.testclient import TestClient

    serving = tmp_path / "serving"
    (serving / "games").mkdir(parents=True)
    (serving / "index.json").write_text(json.dumps({
        "count": 1,
        "seasons": ["2016-17"],
        "games": [{"game_id": "0021600006", "season": "2016-17"}],
    }))
    (serving / "games" / "0021600006.json").write_text(json.dumps({
        "meta": {"game_id": "0021600006",
                 "probability_source": "out-of-fold tier3_celtics"},
        "events": {"wp": [0.5, 0.6]},
    }))
    (serving / "coverage.json").write_text(json.dumps({"by_season": []}))

    models = tmp_path / "models"
    models.mkdir()
    order = ["celtics_margin", "seconds_remaining_game", "period"]
    (models / "model_metadata.json").write_text(json.dumps({
        "feature_order": order, "tier": "tier3_celtics", "n_features": 3,
    }))

    monkeypatch.setattr(config, "SERVING_DIR", serving)
    monkeypatch.setattr(config, "MODELS_DIR", models)
    api.load_metadata.cache_clear()
    api.load_model.cache_clear()
    api.load_feature_frame.cache_clear()

    class StubModel:
        def predict_proba(self, matrix):
            # Probability rises with margin, so an override is visible.
            p = 1 / (1 + np.exp(-np.asarray(matrix)[:, 0] / 10.0))
            return np.column_stack([1 - p, p])

    monkeypatch.setattr(api, "load_model", lambda: StubModel())

    frame = pd.DataFrame({
        "game_id": ["0021600006"] * 2,
        "event_index": [0, 1],
        "celtics_margin": [-10.0, 4.0],
        "seconds_remaining_game": [2880.0, 1200.0],
        "period": [1.0, 2.0],
    }).set_index(["game_id", "event_index"])
    monkeypatch.setattr(api, "load_feature_frame", lambda: frame)

    return TestClient(api.create_app())


# ---------------------------------------------------------------------------
# Serving the replay
# ---------------------------------------------------------------------------

def test_health_states_the_probabilities_are_out_of_fold(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert "out of fold" in body["note"]


def test_game_index_is_served(client):
    body = client.get("/api/games").json()
    assert body["count"] == 1


def test_a_game_is_served_and_says_where_its_numbers_came_from(client):
    body = client.get("/api/games/0021600006").json()
    assert "out-of-fold" in body["meta"]["probability_source"]


def test_an_unpadded_game_id_still_resolves(client):
    """
    Game ids appear as 21600006 in one file and 0021600006 in another. A URL
    carrying either must find the same game.
    """
    assert client.get("/api/games/21600006").status_code == 200


def test_a_missing_game_returns_404_with_the_script_to_run(client):
    response = client.get("/api/games/0029999999")
    assert response.status_code == 404
    assert "20_build_serving" in response.json()["detail"]


# ---------------------------------------------------------------------------
# The caveat that keeps the two kinds of number apart
# ---------------------------------------------------------------------------

def test_whatif_carries_the_in_sample_caveat(client):
    response = client.post("/api/whatif", json={
        "game_id": "0021600006", "event_index": 0, "overrides": {}})
    body = response.json()
    assert response.status_code == 200
    assert "in-sample" in body["caveat"]
    assert "out of fold" in body["caveat"]


def test_predict_carries_the_same_caveat(client):
    response = client.post("/api/predict", json={"features": {
        "celtics_margin": 5.0, "seconds_remaining_game": 600.0,
        "period": 4.0}})
    assert response.status_code == 200
    assert "in-sample" in response.json()["caveat"]


def test_the_caveat_names_both_kinds_of_number():
    """
    A viewer must be able to tell which number they are looking at without
    reading the source.
    """
    assert "deployment model" in api.IN_SAMPLE_CAVEAT
    assert "out of fold" in api.IN_SAMPLE_CAVEAT
    assert "out of fold" in api.OUT_OF_FOLD_NOTE


# ---------------------------------------------------------------------------
# What-if keeps the real features real
# ---------------------------------------------------------------------------

def test_whatif_uses_the_events_real_features_for_everything_not_overridden(client):
    body = client.post("/api/whatif", json={
        "game_id": "0021600006", "event_index": 1, "overrides": {}}).json()
    used = body["features_used"]
    assert used["celtics_margin"] == 4.0
    assert used["seconds_remaining_game"] == 1200.0
    assert used["period"] == 2.0


def test_an_override_changes_only_the_named_feature(client):
    body = client.post("/api/whatif", json={
        "game_id": "0021600006", "event_index": 1,
        "overrides": {"celtics_margin": -20.0}}).json()
    used = body["features_used"]
    assert used["celtics_margin"] == -20.0
    assert used["seconds_remaining_game"] == 1200.0     # untouched, and real


def test_an_override_actually_moves_the_probability(client):
    down = client.post("/api/whatif", json={
        "game_id": "0021600006", "event_index": 1,
        "overrides": {"celtics_margin": -20.0}}).json()["probability"]
    up = client.post("/api/whatif", json={
        "game_id": "0021600006", "event_index": 1,
        "overrides": {"celtics_margin": 20.0}}).json()["probability"]
    assert up > down


def test_an_unknown_override_is_rejected_rather_than_ignored(client):
    """
    Silently ignoring an unrecognised name would return a number that does not
    answer the question that was asked.
    """
    response = client.post("/api/whatif", json={
        "game_id": "0021600006", "event_index": 0,
        "overrides": {"star_player_injured": 1}})
    assert response.status_code == 422
    assert "not model features" in response.json()["detail"]


def test_a_nonexistent_event_is_a_404_not_a_guess(client):
    response = client.post("/api/whatif", json={
        "game_id": "0021600006", "event_index": 9999, "overrides": {}})
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Explicit predict refuses partial input
# ---------------------------------------------------------------------------

def test_predict_rejects_a_partial_feature_vector(client):
    """
    The important one. Defaulting the missing twelve features would produce a
    confident number about a state nobody described.
    """
    response = client.post("/api/predict",
                           json={"features": {"celtics_margin": 5.0}})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "missing features" in detail
    assert "none are defaulted" in detail


def test_predict_accepts_a_complete_vector(client):
    response = client.post("/api/predict", json={"features": {
        "celtics_margin": 12.0, "seconds_remaining_game": 300.0,
        "period": 4.0}})
    assert response.status_code == 200
    assert 0.0 <= response.json()["probability"] <= 1.0


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

def test_model_endpoint_exposes_the_feature_order(client):
    body = client.get("/api/model").json()
    assert body["tier"] == "tier3_celtics"
    assert body["feature_order"][0] == "celtics_margin"


def test_predict_rows_rejects_a_frame_missing_a_feature(monkeypatch):
    monkeypatch.setattr(api, "feature_order", lambda: ["a", "b"])
    with pytest.raises(KeyError, match="missing features"):
        api.predict_rows(pd.DataFrame({"a": [1.0]}))


def test_cors_is_limited_to_the_local_frontend():
    """An open CORS policy on a local research tool is needless exposure."""
    assert api.ALLOWED_ORIGINS
    assert all(o.startswith("http://localhost")
               or o.startswith("http://127.0.0.1")
               for o in api.ALLOWED_ORIGINS)
