"""
Tests for Phase 9a: player bios.

The important property is the boundary. This data is for display, it must never
reach a model feature, and the tests below enforce that as well as checking the
parsing.
"""

import numpy as np
import pandas as pd
import pytest

from src import features, models, pull_player_bios


# ---------------------------------------------------------------------------
# Height parsing
# ---------------------------------------------------------------------------

def test_height_parses_feet_and_inches():
    assert pull_player_bios.parse_height_inches("6-8") == 80.0
    assert pull_player_bios.parse_height_inches("7-0") == 84.0
    assert pull_player_bios.parse_height_inches("5-9") == 69.0


def test_height_returns_nan_rather_than_guessing():
    """
    A blank or malformed height must not become a plausible-looking number.
    A missing card field is honest; an invented one is not.
    """
    for value in ("", None, "six eight", "6", np.nan, 68):
        assert np.isnan(pull_player_bios.parse_height_inches(value))


def test_height_ordering_survives_parsing():
    heights = ["5-9", "6-0", "6-8", "7-2"]
    parsed = [pull_player_bios.parse_height_inches(h) for h in heights]
    assert parsed == sorted(parsed)


# ---------------------------------------------------------------------------
# The display-only boundary
# ---------------------------------------------------------------------------

def test_no_bio_field_is_a_model_feature():
    """
    The guard that matters. If a bio column ever appeared in a feature list it
    would be a season-summary leak of the kind Phase 5 exists to prevent.
    """
    bio_columns = set(pull_player_bios.WANTED.values()) | {
        "height_inches", "full_name", "headshot_url"}
    model_columns = set(models.GAME_STATE_FEATURES) | set(
        models.GENERIC_FEATURES) | set(models.PREGAME_FEATURES)
    assert not bio_columns & model_columns


def test_bio_columns_do_not_collide_with_feature_names():
    """A collision would let a join silently overwrite a real feature."""
    bio_columns = set(pull_player_bios.WANTED.values())
    assert not bio_columns & set(features.FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Schema discipline
# ---------------------------------------------------------------------------

def test_wanted_columns_are_requested_by_name():
    """
    Positional access to an endpoint's frame breaks silently when the endpoint
    changes. Every column this module uses is named.
    """
    assert "PLAYER_ID" in pull_player_bios.WANTED
    assert "HEIGHT" in pull_player_bios.WANTED
    assert "POSITION" in pull_player_bios.WANTED
    assert all(isinstance(k, str) and k.isupper()
               for k in pull_player_bios.WANTED)
    assert all(isinstance(v, str) and v.islower()
               for v in pull_player_bios.WANTED.values())


def test_roster_bounds_bracket_a_real_nba_roster():
    """
    A season roster carries roughly 14 to 21 players once ten-day and two-way
    contracts are counted. The bounds must admit that and reject a truncated or
    wrong-season response.
    """
    assert pull_player_bios.MIN_ROSTER <= 14
    assert pull_player_bios.MAX_ROSTER >= 21


def test_every_team_is_covered():
    """All 30 franchises, from the static table, so no network is needed."""
    teams = pull_player_bios.all_teams()
    assert len(teams) == 30
    assert len({t["abbreviation"] for t in teams}) == 30
    assert "BOS" in {t["abbreviation"] for t in teams}


def test_team_logo_url_is_built_from_the_team_id():
    url = pull_player_bios.TEAM_LOGO_TEMPLATE.format(team_id=1610612738)
    assert "1610612738" in url and url.startswith("https://")


def test_headshot_url_is_built_from_the_person_id():
    url = pull_player_bios.HEADSHOT_TEMPLATE.format(person_id=1628369)
    assert url.endswith("/1628369.png")
    assert url.startswith("https://")


def test_bio_pull_is_resumable(tmp_path, monkeypatch):
    """
    240 calls is long enough that an interruption must not mean starting over.
    A cached team-season is read from disk and never re-fetched.
    """
    monkeypatch.setattr(pull_player_bios.config, "RAW_DIR", tmp_path)

    def explode(*args, **kwargs):
        raise AssertionError("network was called for a cached team-season")

    cached = pd.DataFrame({"person_id": [1], "full_name": ["A Player"]})
    path = pull_player_bios.roster_dir() / "2016-17_BOS.csv"
    cached.to_csv(path, index=False)

    monkeypatch.setattr(pull_player_bios, "fetch_team_season", explode)
    team = {"id": 1610612738, "abbreviation": "BOS", "full_name": "Boston"}
    frame = pull_player_bios.load_or_fetch(team, "2016-17", resume=True)
    assert len(frame) == 1


def test_a_roster_of_implausible_size_raises(tmp_path, monkeypatch):
    """Two players is not a roster. It must stop, not be written to cache."""
    monkeypatch.setattr(pull_player_bios.config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(
        pull_player_bios, "fetch_team_season",
        lambda *a, **k: pd.DataFrame({"person_id": [1, 2],
                                      "full_name": ["A", "B"]}))
    team = {"id": 1610612738, "abbreviation": "BOS", "full_name": "Boston"}
    with pytest.raises(RuntimeError, match="outside the plausible range"):
        pull_player_bios.load_or_fetch(team, "2016-17", resume=False)
    assert not (pull_player_bios.roster_dir() / "2016-17_BOS.csv").exists()


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------

def test_coverage_counts_players_without_a_bio(monkeypatch, tmp_path):
    rosters = pd.DataFrame({
        "person_id": [1, 2, 3, 4],
        "season": ["2016-17"] * 4,
    })
    path = tmp_path / "rosters.parquet"
    rosters.to_parquet(path, index=False)
    monkeypatch.setattr(pull_player_bios.config, "ROSTERS_PARQUET", path)

    bios = pd.DataFrame({"person_id": [1, 2], "season": ["2016-17"] * 2})
    coverage = pull_player_bios.coverage_against_rosters(bios)
    row = coverage.iloc[0]
    assert row["players_seen"] == 4
    assert row["with_bio"] == 2
    assert row["missing"] == 2


def test_coverage_returns_empty_when_rosters_are_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(pull_player_bios.config, "ROSTERS_PARQUET",
                        tmp_path / "nope.parquet")
    assert pull_player_bios.coverage_against_rosters(pd.DataFrame()).empty


def test_coverage_matches_by_season_not_across_seasons(monkeypatch, tmp_path):
    """
    A player with a bio row in a different season does not count as covered.
    Rosters change, and a 2016-17 card should use 2016-17 information.
    """
    rosters = pd.DataFrame({"person_id": [7], "season": ["2016-17"]})
    path = tmp_path / "rosters.parquet"
    rosters.to_parquet(path, index=False)
    monkeypatch.setattr(pull_player_bios.config, "ROSTERS_PARQUET", path)

    bios = pd.DataFrame({"person_id": [7], "season": ["2023-24"]})
    coverage = pull_player_bios.coverage_against_rosters(bios)
    assert coverage.iloc[0]["missing"] == 1
