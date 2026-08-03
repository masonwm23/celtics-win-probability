"""
Tests for the Phase 1 raw pull logic.

No network. No project data touched. Payloads here are minimal synthetic stubs
that mimic the SHAPE of the API response, not its content. They exist to prove
the inspection and reporting code behaves correctly, especially the parts whose
whole job is to notice something is wrong.

The most important test in this file is
test_summary_warns_when_no_game_reaches_period_four. If the play-by-play period
parameters were wrong, we would download only first quarters for all 636 games,
and every downstream number would be quietly wrong in a way that is hard to
spot. That check has to work.

HOW TO RUN IN SPYDER
  Open scripts/03_run_tests.py and press F5. It runs every test file.
"""

import json

import pandas as pd
import pytest

from src import pull_raw


# ---------------------------------------------------------------------------
# Synthetic payload builders
# ---------------------------------------------------------------------------

def pbp_v3_payload(n_events=10, periods=(1, 2, 3, 4)):
    """Mimic the PlayByPlayV3 game.actions shape."""
    actions = []
    for i in range(n_events):
        actions.append({
            "actionNumber": i + 1,
            "period": periods[i % len(periods)],
            "clock": "PT10M00.00S",
            "description": f"synthetic event {i}",
            "scoreHome": "10",
            "scoreAway": "12",
        })
    return {"meta": {"version": 1}, "game": {"gameId": "0021600001",
                                             "actions": actions}}


def pbp_legacy_payload(n_events=5, period_values=(1, 2)):
    """Mimic the older resultSets shape, in case the endpoint returns it."""
    headers = ["GAME_ID", "EVENTNUM", "PERIOD", "HOMEDESCRIPTION"]
    rows = [["0021600001", i, period_values[i % len(period_values)], "desc"]
            for i in range(n_events)]
    return {"resource": "playbyplay",
            "resultSets": [{"name": "PlayByPlay", "headers": headers,
                            "rowSet": rows}]}


def box_v3_payload(n_home=13, n_away=12):
    """Mimic the BoxScoreTraditionalV3 shape."""
    def player(i, tri):
        return {"personId": 1000 + i, "firstName": "First",
                "familyName": f"Last{i}", "position": "",
                "minutes": "PT24M00.00S", "points": 10, "teamTricode": tri}
    return {"meta": {"version": 1}, "boxScoreTraditional": {
        "gameId": "0021600001",
        "homeTeam": {"teamId": 1610612738, "teamTricode": "BOS",
                     "players": [player(i, "BOS") for i in range(n_home)]},
        "awayTeam": {"teamId": 1610612737, "teamTricode": "ATL",
                     "players": [player(i, "ATL") for i in range(n_away)]},
    }}


# ---------------------------------------------------------------------------
# Play-by-play event counting
# ---------------------------------------------------------------------------

def test_counts_events_in_v3_shape():
    n, periods, shape = pull_raw.count_pbp_events(pbp_v3_payload(20))
    assert n == 20
    assert periods == [1, 2, 3, 4]
    assert shape == "game.actions shape"


def test_counts_events_in_legacy_shape():
    n, periods, shape = pull_raw.count_pbp_events(pbp_legacy_payload(8))
    assert n == 8
    assert periods == [1, 2]
    assert shape == "resultSets shape"


def test_reports_unrecognised_shape_instead_of_guessing():
    n, periods, shape = pull_raw.count_pbp_events({"something": "else"})
    assert n == 0
    assert shape == "UNRECOGNISED SHAPE"


def test_overtime_periods_are_preserved():
    n, periods, _ = pull_raw.count_pbp_events(
        pbp_v3_payload(12, periods=(1, 2, 3, 4, 5, 6))
    )
    assert periods == [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Boxscore player counting
# ---------------------------------------------------------------------------

def test_counts_players_from_both_teams():
    """Opponent players must be counted too. The matchup context needs them."""
    n, shape = pull_raw.count_box_players(box_v3_payload(13, 12))
    assert n == 25
    assert shape == "boxScoreTraditional shape"


def test_box_unrecognised_shape_reported():
    n, shape = pull_raw.count_box_players({"nope": 1})
    assert n == 0
    assert shape == "UNRECOGNISED SHAPE"


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------

def test_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(pull_raw.config, "RAW_PBP_DIR", tmp_path)
    payload = pbp_v3_payload(3)
    pull_raw._save("pbp", "0021600001", payload)
    assert pull_raw._load_cached("pbp", "0021600001") == payload


def test_missing_cache_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(pull_raw.config, "RAW_PBP_DIR", tmp_path)
    assert pull_raw._load_cached("pbp", "9999999999") is None


def test_corrupt_cache_treated_as_missing(tmp_path, monkeypatch):
    """
    A half-written file from an interrupted run must not be trusted. This is the
    exact failure mode that produced corrupted files earlier in this project,
    so it gets a test.
    """
    monkeypatch.setattr(pull_raw.config, "RAW_PBP_DIR", tmp_path)
    (tmp_path / "0021600001.json").write_text('{"game": {"actions": [', "utf-8")
    assert pull_raw._load_cached("pbp", "0021600001") is None


def test_save_is_atomic_no_tmp_left_behind(tmp_path, monkeypatch):
    monkeypatch.setattr(pull_raw.config, "RAW_PBP_DIR", tmp_path)
    pull_raw._save("pbp", "0021600001", pbp_v3_payload(2))
    assert (tmp_path / "0021600001.json").exists()
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# Season sampling for the smoke test
# ---------------------------------------------------------------------------

def make_index(games_per_season=20):
    seasons = ["2016-17", "2017-18", "2018-19", "2019-20",
               "2020-21", "2021-22", "2022-23", "2023-24"]
    rows = []
    gid = 0
    for s in seasons:
        start = pd.Timestamp(year=int(s.split("-")[0]), month=10, day=25)
        for i in range(games_per_season):
            gid += 1
            rows.append({"SEASON": s, "GAME_ID": f"002{gid:07d}",
                         "GAME_DATE": start + pd.Timedelta(days=i)})
    return pd.DataFrame(rows)


def test_sample_spans_every_season():
    """A smoke test drawn from one season would miss format changes."""
    sampled = pull_raw.sample_across_seasons(make_index(), 16)
    assert len(sampled) == 16
    assert sampled["SEASON"].nunique() == 8
    assert (sampled.groupby("SEASON").size() == 2).all()


def test_sample_never_exceeds_limit():
    assert len(pull_raw.sample_across_seasons(make_index(), 5)) == 5


def test_sample_returns_unique_games():
    sampled = pull_raw.sample_across_seasons(make_index(), 16)
    assert sampled["GAME_ID"].is_unique


# ---------------------------------------------------------------------------
# Summary reporting. These are the checks that protect against silent damage.
# ---------------------------------------------------------------------------

def manifest_row(**kw):
    base = {
        "game_id": "0021600001", "season": "2016-17",
        "game_date": "2016-10-26", "matchup": "BOS vs. BKN",
        "opponent": "BKN", "pbp_status": "fetched", "pbp_events": 480,
        "pbp_periods": "1,2,3,4", "pbp_shape": "game.actions shape",
        "box_status": "fetched", "box_players": 26,
        "box_shape": "boxScoreTraditional shape", "error": "",
    }
    base.update(kw)
    return base


def test_summary_warns_when_no_game_reaches_period_four():
    """
    The silent-disaster check. If the period parameters were wrong we would
    download first quarters only, and every downstream number would be wrong
    without anything looking broken.
    """
    m = pd.DataFrame([manifest_row(pbp_periods="1", pbp_events=120)])
    summary = pull_raw.build_summary(m, 1, 10.0, None)
    assert "WARNING" in summary
    assert "period 4" in summary


def test_summary_accepts_full_period_coverage():
    m = pd.DataFrame([manifest_row()])
    summary = pull_raw.build_summary(m, 1, 10.0, None)
    assert "WARNING" not in summary
    assert "no failures and no empty payloads" in summary


def test_summary_lists_failed_games():
    m = pd.DataFrame([
        manifest_row(),
        manifest_row(game_id="0021600002", pbp_status="FAILED",
                     pbp_events=0, pbp_periods="", error="pbp: timeout"),
    ])
    summary = pull_raw.build_summary(m, 2, 10.0, None)
    assert "1 game(s) need attention" in summary
    assert "timeout" in summary


def test_summary_flags_empty_payload_as_a_problem():
    """Zero events is a failure, not a game with nothing in it."""
    m = pd.DataFrame([manifest_row(pbp_status="EMPTY", pbp_events=0,
                                   pbp_periods="")])
    summary = pull_raw.build_summary(m, 1, 10.0, None)
    assert "need attention" in summary


def test_summary_flags_suspiciously_thin_games():
    """A game with 150 events downloaded fine but is probably truncated."""
    m = pd.DataFrame([manifest_row(pbp_events=150, pbp_periods="1,2")])
    summary = pull_raw.build_summary(m, 1, 10.0, None)
    assert "under 300 events" in summary


def test_summary_reports_mixed_shapes():
    """If two response shapes appear, the parser has to handle both."""
    m = pd.DataFrame([
        manifest_row(),
        manifest_row(game_id="0021600002", pbp_shape="resultSets shape"),
    ])
    summary = pull_raw.build_summary(m, 2, 10.0, None)
    assert "game.actions shape" in summary
    assert "resultSets shape" in summary


def test_fetch_one_game_records_failure_without_raising(tmp_path, monkeypatch):
    """
    One dead game must not abort a 636 game run, and must not be recorded as a
    success either.
    """
    monkeypatch.setattr(pull_raw.config, "RAW_PBP_DIR", tmp_path)
    monkeypatch.setattr(pull_raw.config, "RAW_BOX_DIR", tmp_path)

    def boom(*a, **k):
        raise pull_raw.NBARequestError("simulated network failure")

    monkeypatch.setattr(pull_raw, "call_endpoint", boom)

    row = pull_raw.fetch_one_game("0021600001")
    assert row["pbp_status"] == "FAILED"
    assert row["box_status"] == "FAILED"
    assert "simulated network failure" in row["error"]
    assert row["pbp_events"] == 0
