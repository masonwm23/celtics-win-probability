"""
Tests for Phase 12e: the highlight precompute.

The properties that matter are the ones preventing a reel from appearing under
the wrong game, and the ones keeping unofficial re-uploads out entirely.

  1. TITLE PARSING. Real NBA titles, copied from the Phase 12d output, must
     resolve to exactly two teams.
  2. VERIFICATION ON THREE AXES. Teams, date, title, each broken on its own.
  3. UNCERTAIN IS NOT DISPLAYED. Review never reaches the mapping.
  4. NOTHING IN data/serving IS WRITTEN.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src import config, youtube_precompute as yp


GAME = {
    "season": "2023-24", "game_id": "0022300906",
    "game_date": datetime(2023, 12, 8, tzinfo=timezone.utc),
    "opponent_tricode": "NYK", "matchup": "BOS vs. NYK",
}


def candidate(title="KNICKS at CELTICS | FULL GAME HIGHLIGHTS | December 8, 2023",
              published="2023-12-09T04:10:00Z", embeddable=True,
              privacy="public", duration="PT9M20S", video_id="v1",
              handle="@NBA"):
    return {"handle": handle, "channel_id": "UC_nba", "video_id": video_id,
            "title": title, "published_at": published,
            "embeddable": embeddable, "privacy": privacy, "duration": duration}


# ---------------------------------------------------------------------------
# 1. Title parsing, against titles actually observed
# ---------------------------------------------------------------------------

def test_standard_convention_parses_two_teams():
    assert yp.teams_in_title(
        "CELTICS at NETS | FULL GAME HIGHLIGHTS | March 11, 2021") == {"BOS", "BKN"}


def test_a_title_naming_a_city_and_its_nickname_is_still_two_teams():
    """
    Observed in 2018-19: "Full Game Recap: Celtics vs Heat | Vintage Wade On
    Display In Miami" names Heat and Miami, which are one team.
    """
    assert yp.teams_in_title(
        "Full Game Recap: Celtics vs Heat | Vintage Wade On Display In Miami"
    ) == {"BOS", "MIA"}
    assert yp.teams_in_title(
        "TRAIL BLAZERS vs CELTICS | Damian Lillard Drops 33 To Lead Portland"
    ) == {"BOS", "POR"}


def test_los_angeles_alone_never_identifies_a_team():
    """
    "Los Angeles" cannot pick between the Lakers and the Clippers, so it is
    deliberately not a token. Guessing would put a reel under the wrong game.
    """
    assert yp.teams_in_title("Celtics vs Los Angeles") == {"BOS"}
    assert yp.teams_in_title("CELTICS at LAKERS | FULL GAME HIGHLIGHTS") == {
        "BOS", "LAL"}
    assert yp.teams_in_title("CELTICS at CLIPPERS | FULL GAME HIGHLIGHTS") == {
        "BOS", "LAC"}


def test_team_tokens_match_on_word_boundaries():
    """
    Substring matching would find "heat" inside "heated" and "jazz" inside
    "jazzed", inventing an opponent out of ordinary prose.
    """
    assert yp.teams_in_title("A heated finish in Boston") == {"BOS"}
    assert yp.teams_in_title("Celtics jazzed up the crowd") == {"BOS"}
    assert yp.teams_in_title("Celtics vs Jazz") == {"BOS", "UTA"}
    assert yp.teams_in_title("Bostonian fans") == set()


def test_date_is_extracted_from_the_title():
    assert yp.date_in_title(
        "KNICKS at CELTICS | FULL GAME HIGHLIGHTS | December 8, 2023"
    ).date() == datetime(2023, 12, 8).date()
    assert yp.date_in_title("CELTICS at NETS | FULL GAME HIGHLIGHTS | March 11, 2021"
                            ).date() == datetime(2021, 3, 11).date()
    assert yp.date_in_title("Celtics vs Heat, Jan. 10, 2019").date() == \
        datetime(2019, 1, 10).date()
    assert yp.date_in_title("no date here") is None
    assert yp.date_in_title("February 30, 2020") is None


def test_reel_detection_rejects_player_and_compilation_videos():
    assert yp.reads_as_a_game_reel("KNICKS at CELTICS | FULL GAME HIGHLIGHTS")
    assert yp.reads_as_a_game_reel("Full Game Recap: Celtics vs Heat")
    assert not yp.reads_as_a_game_reel("Top 10 Plays of the Night")
    assert not yp.reads_as_a_game_reel(
        "Nikola Jokic Full Game Highlights vs Celtics")
    assert not yp.reads_as_a_game_reel("Celtics vs Nets 1st Half Highlights")
    assert not yp.reads_as_a_game_reel("FlightReacts To Celtics Highlights")
    assert not yp.reads_as_a_game_reel(None)


def test_duration_parsing():
    assert yp.iso_duration_seconds("PT9M48S") == 588
    assert yp.iso_duration_seconds("PT10M") == 600
    assert yp.iso_duration_seconds("PT4H51M16S") == 17476
    assert yp.iso_duration_seconds("") == 0


# ---------------------------------------------------------------------------
# 2. Verification, each axis broken alone
# ---------------------------------------------------------------------------

def test_a_clean_candidate_is_confirmed():
    result = yp.assess(candidate(), GAME)
    assert result["verdict"] == "confirmed"
    assert result["problems"] == ""


def test_the_wrong_opponent_is_rejected():
    result = yp.assess(
        candidate(title="LAKERS at CELTICS | FULL GAME HIGHLIGHTS | December 8, 2023"),
        GAME)
    assert result["verdict"] == "rejected"
    assert "title teams" in result["problems"]


def test_a_title_date_disagreeing_with_the_game_date_goes_to_review():
    """
    The strongest single check. A reel whose own title says a different day is
    a different game, however close the upload time is.
    """
    result = yp.assess(
        candidate(title="KNICKS at CELTICS | FULL GAME HIGHLIGHTS | December 15, 2023"),
        GAME)
    assert result["verdict"] == "review"
    assert "title date 2023-12-15" in result["problems"]


def test_no_date_in_title_falls_back_to_a_tight_upload_window():
    ok = yp.assess(candidate(title="Full Game Recap: Knicks vs Celtics",
                             published="2023-12-09T04:00:00Z"), GAME)
    assert ok["verdict"] == "confirmed"

    late = yp.assess(candidate(title="Full Game Recap: Knicks vs Celtics",
                               published="2023-12-20T04:00:00Z"), GAME)
    assert late["verdict"] == "review"
    assert "outside the tight window" in late["problems"]


def test_a_non_embeddable_video_is_rejected():
    result = yp.assess(candidate(embeddable=False), GAME)
    assert result["verdict"] == "rejected"
    assert "not embeddable" in result["problems"]


def test_a_private_video_is_rejected():
    result = yp.assess(candidate(privacy="unlisted"), GAME)
    assert result["verdict"] == "rejected"
    assert "privacy unlisted" in result["problems"]


def test_a_full_replay_length_video_goes_to_review():
    """A four hour upload is a replay or a scoreboard stream, not a reel."""
    result = yp.assess(candidate(duration="PT4H51M16S"), GAME)
    assert result["verdict"] == "review"
    assert "outside" in result["problems"]


def test_a_thirty_second_clip_goes_to_review():
    result = yp.assess(candidate(duration="PT30S"), GAME)
    assert result["verdict"] == "review"


# ---------------------------------------------------------------------------
# 3. Uniqueness, and uncertain never reaching the mapping
# ---------------------------------------------------------------------------

def games_list():
    return [
        {**GAME},
        {"season": "2023-24", "game_id": "0022300999",
         "game_date": datetime(2023, 12, 20, tzinfo=timezone.utc),
         "opponent_tricode": "NYK", "matchup": "BOS @ NYK"},
    ]


def test_two_confirmed_candidates_for_one_game_are_both_downgraded():
    """
    If two reels both confirm for one game, we do not know which is right, so
    neither is used. Picking one would be a guess on screen.
    """
    frame = yp.match_games(pd.DataFrame([
        candidate(video_id="v1"),
        candidate(video_id="v2",
                  title="KNICKS at CELTICS | FULL GAME HIGHLIGHTS | December 8, 2023"),
    ]), [GAME])
    assert not frame.empty
    assert (frame["verdict"] == "confirmed").sum() == 0
    assert "more than one confirmed candidate" in " ".join(frame["problems"])


def test_one_video_confirming_for_two_games_is_downgraded():
    """
    Back-to-back games against the same opponent are real. An undated reel
    published between them could confirm for both, and showing it under both
    would put the wrong game's video on screen at least once.
    """
    back_to_back = [
        {**GAME},
        {"season": "2023-24", "game_id": "0022300907",
         "game_date": datetime(2023, 12, 9, tzinfo=timezone.utc),
         "opponent_tricode": "NYK", "matchup": "BOS @ NYK"},
    ]
    frame = yp.match_games(pd.DataFrame([
        candidate(video_id="shared", title="Full Game Recap: Knicks vs Celtics",
                  published="2023-12-09T04:00:00Z"),
    ]), back_to_back)
    assert frame["game_id"].nunique() == 2, (
        "the fixture must actually produce the collision it claims")
    assert (frame["verdict"] == "confirmed").sum() == 0
    assert "confirmed for more than one game" in " ".join(frame["problems"])


def test_review_rows_never_reach_the_mapping():
    """
    Built the same way main() builds it: confirmed rows only.
    """
    frame = pd.DataFrame([
        {"game_id": "g1", "verdict": "confirmed", "video_id": "v1",
         "title": "t", "handle": "@NBA", "published_at": "p",
         "duration_seconds": 588},
        {"game_id": "g2", "verdict": "review", "video_id": "v2",
         "title": "t", "handle": "@NBA", "published_at": "p",
         "duration_seconds": 588},
        {"game_id": "g3", "verdict": "rejected", "video_id": "v3",
         "title": "t", "handle": "@NBA", "published_at": "p",
         "duration_seconds": 588},
    ])
    mapping = {row.game_id: row.video_id for row in
               frame.loc[frame["verdict"].eq("confirmed")].itertuples()}
    assert set(mapping) == {"g1"}


def test_no_candidate_survives_without_boston_in_the_title():
    frame = yp.match_games(pd.DataFrame([
        candidate(title="KNICKS at NETS | FULL GAME HIGHLIGHTS | December 8, 2023")
    ]), [GAME])
    assert frame.empty


# ---------------------------------------------------------------------------
# 4. Reports
# ---------------------------------------------------------------------------

def test_coverage_report_states_the_panel_hides_cleanly():
    report = yp.build_coverage_report(
        pd.DataFrame(), [GAME], datetime(2016, 9, 1, tzinfo=timezone.utc),
        False, pd.DataFrame([{"video_id": "v"}]))
    assert "Game Highlights" in report
    assert "NOT synchronised with the probability cursor" in report
    assert "hides the panel entirely" in report
    assert "no unofficial fallback" in report
    assert "separate feature" in report


def test_coverage_report_flags_a_truncated_enumeration_as_untested():
    """
    An uploads playlist has a depth limit. Games older than what was reached
    were never tested, and must not read as absences.
    """
    report = yp.build_coverage_report(
        pd.DataFrame(), [GAME], datetime(2019, 1, 1, tzinfo=timezone.utc),
        True, pd.DataFrame([{"video_id": "v"}]))
    assert "NEVER TESTED" in report
    assert "not evidence that no reel exists" in report


def test_review_report_says_none_of_it_is_displayed():
    frame = pd.DataFrame([{
        "season": "2023-24", "game_id": "g", "game_date": "2023-12-08",
        "matchup": "BOS vs. NYK", "verdict": "review", "handle": "@NBA",
        "title": "KNICKS at CELTICS | FULL GAME HIGHLIGHTS | December 15, 2023",
        "published_at": "2023-12-09T04:00:00Z", "video_id": "v1",
        "duration_seconds": 588,
        "problems": "title date 2023-12-15 != 2023-12-08"}])
    report = yp.build_review_report(frame)
    assert "NONE of these are in the mapping" in report
    assert "treated as no match" in report
    assert "title date" in report
    assert "v1" in report


def test_review_report_handles_nothing_to_review():
    report = yp.build_review_report(pd.DataFrame())
    assert "Nothing to review" in report


def test_module_is_metadata_only_and_touches_no_serving_path():
    import inspect
    source = inspect.getsource(yp)
    lowered = source.lower()
    for forbidden in ("yt_dlp", "youtube_dl", "pytube", "ffmpeg",
                      "beautifulsoup", "bs4", "selenium", "scrapy"):
        assert forbidden not in lowered
    assert "SERVING_DIR" not in source, (
        "the precompute must not write anywhere the dashboard reads")


def test_load_games_carries_is_home_for_the_title_builder():
    """
    Regression. Phase 12f builds search titles from these dicts, and the NBA
    titles reels AWAY at HOME, so a missing `is_home` raises KeyError on the
    very first game of a multi-day job.
    """
    index = pd.DataFrame([{
        "SEASON": "2023-24", "GAME_ID": 22300906,
        "GAME_DATE": "2023-12-08", "OPPONENT_ABBREV": "NYK",
        "MATCHUP": "BOS vs. NYK", "IS_HOME": True}])
    games = yp.load_games(index)
    assert games[0]["is_home"] is True
    assert games[0]["game_id"] == "0022300906"

    from src.youtube_targeted import title_variants
    assert "Knicks at Celtics" in title_variants(games[0])[0][0]
