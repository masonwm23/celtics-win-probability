"""
Tests for Phase 12f: the resumable gap fill.

The properties that matter are the ones that make a multi-day job safe:

  1. RESUMABILITY. Attempted games are skipped, so a re-run continues rather
     than repeating, and a quota stop loses nothing.
  2. UNOFFICIAL CHANNELS CANNOT ENTER. Filtered before assessment, not after.
  3. AMBIGUITY IS NOT RESOLVED BY GUESSING. Two confirmed candidates means
     neither is used.
  4. A QUOTA STOP IS NOT AN ABSENCE.
"""

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from src import config, youtube_fill as yf


def game(game_id="0022300906", season="2023-24", day=8, opponent="NYK",
         is_home=True):
    return {"season": season, "game_id": game_id,
            "game_date": datetime(2023, 12, day, tzinfo=timezone.utc),
            "opponent_tricode": opponent, "matchup": f"BOS vs. {opponent}",
            "is_home": is_home}


def games_across_seasons():
    out = []
    for i, season in enumerate(yf.SEASON_PRIORITY):
        out.append(game(game_id=f"002{i}300{i:03d}", season=season))
    return out


# ---------------------------------------------------------------------------
# 1. Queue and resumability
# ---------------------------------------------------------------------------

def test_productive_seasons_are_attempted_first():
    """
    A limited daily quota should be spent where Phase 12d found reels, not on
    the two seasons it found empty.
    """
    queue = yf.build_queue(games_across_seasons(), set(), set())
    seasons = [g["season"] for g in queue]
    assert seasons[0] == "2023-24"
    assert seasons[-2:] == ["2017-18", "2016-17"]


def test_already_mapped_games_are_not_requeued():
    games = games_across_seasons()
    mapped = {games[0]["game_id"]}
    queue = yf.build_queue(games, mapped, set())
    assert games[0]["game_id"] not in {g["game_id"] for g in queue}
    assert len(queue) == len(games) - 1


def test_already_attempted_games_are_not_retried():
    """
    The core of resumability. Without this a re-run burns the whole day's
    quota re-searching games that already came back empty.
    """
    games = games_across_seasons()
    attempted = {games[1]["game_id"], games[2]["game_id"]}
    queue = yf.build_queue(games, set(), attempted)
    assert len(queue) == len(games) - 2
    assert attempted.isdisjoint({g["game_id"] for g in queue})


def test_queue_is_empty_when_everything_is_done():
    games = games_across_seasons()
    assert yf.build_queue(games, {g["game_id"] for g in games}, set()) == []


def test_progress_is_loaded_from_disk(tmp_path):
    path = tmp_path / "progress.csv"
    pd.DataFrame([{"game_id": 22300906, "verdict": "unmatched"}]).to_csv(
        path, index=False)
    # Zero padding matters: the CSV stores an int, the games carry strings.
    assert yf.load_progress(path) == {"0022300906"}


def test_missing_or_corrupt_progress_starts_clean(tmp_path):
    assert yf.load_progress(tmp_path / "nope.csv") == set()
    bad = tmp_path / "bad.csv"
    bad.write_text("not,a,valid\nprogress")
    assert yf.load_progress(bad) == set()


def test_corrupt_mapping_does_not_wipe_the_run(tmp_path):
    bad = tmp_path / "map.json"
    bad.write_text("{ this is not json")
    assert yf.load_json(bad, {}) == {}


# ---------------------------------------------------------------------------
# 2. Unofficial channels cannot enter
# ---------------------------------------------------------------------------

def api_item(video_id="v1", title="KNICKS at CELTICS | FULL GAME HIGHLIGHTS | December 8, 2023",
             channel_id="UC_nba", published="2023-12-09T04:00:00Z",
             embeddable=True, privacy="public", duration="PT9M20S"):
    return {"id": video_id,
            "snippet": {"title": title, "channelId": channel_id,
                        "channelTitle": "NBA", "publishedAt": published},
            "status": {"embeddable": embeddable, "privacyStatus": privacy},
            "contentDetails": {"duration": duration}}


def stub_search(monkeypatch, items):
    """One search variant returning `items`, then nothing."""
    calls = {"n": 0}

    def fake_search(key, query, after, before):
        calls["n"] += 1
        if calls["n"] > 1:
            return {"items": []}, None
        return {"items": [{"id": {"videoId": i["id"]}} for i in items]}, None

    monkeypatch.setattr(yf, "search_variant", fake_search)
    monkeypatch.setattr(yf, "hydrate",
                        lambda key, ids: {i["id"]: i for i in items})
    return calls


def test_an_unofficial_channel_is_filtered_before_assessment(monkeypatch):
    """
    FreeDawkins and similar re-uploads post titles that would otherwise pass
    every content test. They are removed on channel id first.
    """
    stub_search(monkeypatch, [api_item(channel_id="UC_freedawkins")])
    result, dead = yf.attempt_game("k", game(), {"UC_nba"})
    assert not dead
    assert result["verdict"] == "unmatched"
    assert result["row"] is None
    assert result["reviews"] == []


def test_an_official_channel_candidate_is_confirmed(monkeypatch):
    stub_search(monkeypatch, [api_item()])
    result, dead = yf.attempt_game("k", game(), {"UC_nba"})
    assert result["verdict"] == "confirmed"
    assert result["row"]["video_id"] == "v1"


# ---------------------------------------------------------------------------
# 3. Ambiguity is never resolved by guessing
# ---------------------------------------------------------------------------

def test_two_confirmed_candidates_produce_no_match(monkeypatch):
    stub_search(monkeypatch, [
        api_item(video_id="v1"),
        api_item(video_id="v2"),
    ])
    result, _ = yf.attempt_game("k", game(), {"UC_nba"})
    assert result["verdict"] == "review"
    assert result["row"] is None
    assert len(result["reviews"]) == 2
    assert all("more than one confirmed candidate" in r["problems"]
               for r in result["reviews"])


def test_a_wrong_date_in_the_title_goes_to_review_not_the_mapping(monkeypatch):
    stub_search(monkeypatch, [api_item(
        title="KNICKS at CELTICS | FULL GAME HIGHLIGHTS | December 15, 2023")])
    result, _ = yf.attempt_game("k", game(), {"UC_nba"})
    assert result["verdict"] == "review"
    assert result["row"] is None
    assert "title date" in result["reviews"][0]["problems"]


def test_nothing_found_is_unmatched_not_review(monkeypatch):
    monkeypatch.setattr(yf, "search_variant",
                        lambda key, q, a, b: ({"items": []}, None))
    result, _ = yf.attempt_game("k", game(), {"UC_nba"})
    assert result["verdict"] == "unmatched"


# ---------------------------------------------------------------------------
# 4. Quota
# ---------------------------------------------------------------------------

def test_a_quota_error_stops_the_game_immediately(monkeypatch):
    monkeypatch.setattr(
        yf, "search_variant",
        lambda key, q, a, b: (None, "HTTPError: HTTP Error 403: Forbidden"))
    result, dead = yf.attempt_game("k", game(), {"UC_nba"})
    assert dead is True
    assert result is None


def test_run_stops_on_quota_and_writes_what_it_had(tmp_path, monkeypatch):
    """
    The multi-day property. Two games succeed, the third kills the quota, and
    the mapping on disk must hold exactly the two.
    """
    paths = {"map": tmp_path / "map.json",
             "progress": tmp_path / "progress.csv",
             "review": tmp_path / "review.csv"}
    games = [game(game_id=f"00223009{i:02d}", day=i + 1) for i in range(5)]
    calls = {"n": 0}

    def fake_attempt(key, g, official_ids):
        calls["n"] += 1
        if calls["n"] <= 2:
            return {"verdict": "confirmed", "reviews": [], "row": {
                "video_id": f"v{calls['n']}", "title": "t", "handle": "@NBA",
                "published_at": "2023-12-09T04:00:00Z",
                "duration_seconds": 588}}, False
        return None, True

    monkeypatch.setattr(yf, "attempt_game", fake_attempt)
    result = yf.run("k", games, {"UC_nba"}, paths)

    assert result["stopped"] is True
    assert calls["n"] == 3, "must stop, not keep spending"
    on_disk = json.loads(paths["map"].read_text())
    assert len(on_disk) == 2
    assert set(on_disk) == {games[0]["game_id"], games[1]["game_id"]}


def test_a_second_run_continues_rather_than_repeating(tmp_path, monkeypatch):
    paths = {"map": tmp_path / "map.json",
             "progress": tmp_path / "progress.csv",
             "review": tmp_path / "review.csv"}
    games = [game(game_id=f"00223009{i:02d}", day=i + 1) for i in range(4)]

    seen = []

    def fake_attempt(key, g, official_ids):
        seen.append(g["game_id"])
        return {"verdict": "unmatched", "reviews": [], "row": None}, False

    monkeypatch.setattr(yf, "attempt_game", fake_attempt)
    yf.run("k", games, {"UC_nba"}, paths, max_games=2)
    first_pass = list(seen)
    seen.clear()
    yf.run("k", games, {"UC_nba"}, paths)

    assert len(first_pass) == 2
    assert set(seen).isdisjoint(set(first_pass)), (
        "the second run must not re-attempt games from the first")
    assert len(seen) == 2


def test_report_says_a_quota_stop_lost_nothing():
    result = {"mapping": {"g": {}}, "progress": pd.DataFrame(),
              "reviews": pd.DataFrame(), "stopped": True, "queued": 10}
    report = yf.build_report(result, [game()])
    assert "STOPPED ON QUOTA" in report
    assert "Nothing was lost" in report
    assert "again tomorrow" in report


def test_report_states_the_celtics_channel_question_is_closed():
    result = {"mapping": {}, "progress": pd.DataFrame(),
              "reviews": pd.DataFrame(), "stopped": False, "queued": 0}
    report = yf.build_report(result, [game()])
    assert "contains no full-game reels" in report
    assert "every match" in report.lower()


def test_report_does_not_call_the_playlist_cap_an_absence():
    result = {"mapping": {}, "progress": pd.DataFrame(),
              "reviews": pd.DataFrame(), "stopped": False, "queued": 0}
    report = yf.build_report(result, [game()])
    assert "cannot see further back" in report


def test_module_is_metadata_only_and_writes_nothing_to_serving():
    import inspect
    source = inspect.getsource(yf)
    for forbidden in ("yt_dlp", "youtube_dl", "pytube", "ffmpeg",
                      "beautifulsoup", "bs4", "selenium", "scrapy"):
        assert forbidden not in source.lower()
    assert "SERVING_DIR" not in source
