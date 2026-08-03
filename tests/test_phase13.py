"""
Tests for Phase 13a: play-level video synchronisation feasibility.

The properties that matter:

  1. A TIMESTAMP IS NEVER INVENTED. Only description lines that BEGIN with a
     time count, and a description with no chapters yields nothing.
  2. AMBIGUITY IS NOT RESOLVED. A chapter label matching two plays equally
     well is left unmatched.
  3. NO TIMING ACCURACY IS CLAIMED. The probe can show a label describes a
     play; it cannot show the offset is right, and must say so.
  4. NOTHING IS DOWNLOADED OR SCRAPED.
"""

import pandas as pd
import pytest

from src import video_sync_probe as vs


def events_frame():
    rows = [
        (12, 1, "PT10M31.00S", 5, 7, "Jayson Tatum",
         "Tatum 25' 3PT Jump Shot (3 PTS) (Brown 1 AST)"),
        (40, 1, "PT04M12.00S", 18, 14, "Jaylen Brown",
         "Brown 3' Driving Layup (8 PTS)"),
        (77, 2, "PT08M02.00S", 31, 29, "Derrick White",
         "White 26' 3PT Jump Shot (11 PTS)"),
        (95, 2, "PT02M45.00S", 44, 40, "Jayson Tatum",
         "Tatum 2' Driving Dunk (17 PTS)"),
    ]
    return pd.DataFrame(
        [{"event_index": i, "period": p, "clock_raw": c,
          "celtics_score": cs, "opponent_score": os_, "player_name": n,
          "description": d} for i, p, c, cs, os_, n, d in rows])


# ---------------------------------------------------------------------------
# 1. Timestamps are read, never invented
# ---------------------------------------------------------------------------

def test_chapter_lines_are_parsed():
    description = ("0:00 Intro\n"
                   "1:24 Tatum 25' 3PT Jump Shot\n"
                   "3:05 Brown driving layup\n"
                   "1:02:11 Late game\n")
    stamps = vs.parse_timestamps(description)
    assert stamps[0] == (0, "Intro")
    assert stamps[1] == (84, "Tatum 25' 3PT Jump Shot")
    assert stamps[2] == (185, "Brown driving layup")
    assert stamps[3][0] == 3731


def test_a_time_mentioned_mid_sentence_is_not_a_chapter():
    """
    "check out 2:15 for the dunk" is prose. Treating it as a chapter would
    invent structure that YouTube does not render and the NBA did not mark.
    """
    assert vs.parse_timestamps("Watch the highlights, check out 2:15") == []
    assert vs.parse_timestamps("Subscribe! More at nba.com") == []


def test_no_description_yields_no_timestamps():
    assert vs.parse_timestamps(None) == []
    assert vs.parse_timestamps("") == []


def test_real_chapters_require_three_starting_at_zero():
    """
    YouTube only renders chapters under those conditions. Fewer is a
    description that happens to contain a time.
    """
    assert vs.has_real_chapters([(0, "a"), (60, "b"), (120, "c")])
    assert not vs.has_real_chapters([(60, "b"), (120, "c"), (180, "d")])
    assert not vs.has_real_chapters([(0, "a"), (60, "b")])
    assert not vs.has_real_chapters([])


# ---------------------------------------------------------------------------
# 2. Matching, and refusing to break ties
# ---------------------------------------------------------------------------

def test_a_clear_chapter_matches_its_play():
    result = vs.match_chapter("Tatum 25' 3PT Jump Shot", events_frame())
    assert result["matched"]
    assert result["event_index"] == 12
    assert result["period"] == 1


def test_a_label_naming_nobody_does_not_match():
    result = vs.match_chapter("Intro", events_frame())
    assert not result["matched"]


def test_a_label_with_the_right_player_but_wrong_play_does_not_match():
    """
    Surname alone is not enough. Tatum appears twice in this game and the
    words have to agree too.
    """
    result = vs.match_chapter("Tatum blocks the shot at the rim",
                              events_frame())
    assert not result["matched"]


def test_two_equally_good_matches_are_left_unmatched():
    """
    The rule that keeps a guess off the screen. Two identical plays by the
    same player cannot be told apart from a label, so neither is used.
    """
    duplicated = pd.concat([events_frame(), events_frame()], ignore_index=True)
    duplicated["event_index"] = range(len(duplicated))
    result = vs.match_chapter("Tatum 25' 3PT Jump Shot", duplicated)
    assert not result["matched"]
    assert "tie" in result["reason"]


def test_an_empty_label_is_not_matched():
    assert not vs.match_chapter("", events_frame())["matched"]
    assert not vs.match_chapter("   ", events_frame())["matched"]


def test_surname_extraction():
    assert vs.surname("Jayson Tatum") == "tatum"
    assert vs.surname("Nikola Jokic") == "jokic"
    assert vs.surname("") == ""
    assert vs.surname(None) == ""


# ---------------------------------------------------------------------------
# 3. The report's honesty
# ---------------------------------------------------------------------------

def scan_frame(has_chapters=False, n=3):
    return pd.DataFrame([{
        "game_id": f"g{i}", "video_id": f"v{i}", "title": "t",
        "timestamp_lines": 5 if has_chapters else 0,
        "chapter_count": 5 if has_chapters else 0,
        "has_chapters": has_chapters, "description_chars": 400}
        for i in range(n)])


def test_report_closes_the_question_when_no_video_has_chapters():
    """
    The likely outcome, and the one where the honest answer is "not possible"
    rather than an algorithm.
    """
    report = vs.build_report(scan_frame(False), pd.DataFrame(), None, [])
    assert "NONE. No mapped video carries chapter markers." in report
    assert "is guessing" in report
    assert "NOT achievable from this source" in report
    assert "figure animation" in report


def test_report_never_states_a_timing_accuracy_figure():
    """
    This probe cannot measure whether an offset is correct. Publishing a
    number it cannot measure is the exact error made earlier in this project.
    """
    chapters = pd.DataFrame([{
        "game_id": "g0", "video_id": "v0", "timestamp": 84,
        "label": "Tatum 3PT", "matched": True, "agreement": 1.0,
        "event_index": 12, "period": 1, "clock": "PT10M31.00S",
        "celtics_score": 5, "opponent_score": 7, "player": "Jayson Tatum",
        "event_description": "Tatum 25' 3PT Jump Shot", "reason": ""}])
    report = vs.build_report(scan_frame(True), chapters, "g0", [])
    assert "CANNOT prove the timestamp is right" in report
    assert "No timing accuracy figure is reported" in report
    assert "&t=84s" in report


def test_report_states_the_fallback_for_unverified_plays():
    report = vs.build_report(scan_frame(False), pd.DataFrame(), None, [])
    assert "No verified video for this play." in report
    assert "not replaced by video under any outcome" in report


def test_report_explains_captions_cannot_be_read():
    report = vs.build_report(scan_frame(False), pd.DataFrame(), None,
                             [{"video_id": "v0", "tracks": 2,
                               "kinds": ["asr"], "error": None}])
    assert "OAuth as the video owner" in report
    assert "scraping" in report


# ---------------------------------------------------------------------------
# 4. Nothing is downloaded or scraped
# ---------------------------------------------------------------------------

def test_module_downloads_nothing_and_touches_no_serving_path():
    import inspect
    source = inspect.getsource(vs)
    for forbidden in ("yt_dlp", "youtube_dl", "pytube", "ffmpeg", "cv2",
                      "beautifulsoup", "bs4", "selenium", "scrapy",
                      "timedtext", "pytesseract"):
        assert forbidden not in source.lower(), f"{forbidden} must not appear"
    assert "SERVING_DIR" not in source


def test_caption_content_is_never_requested():
    """
    captions.list returns metadata. captions.download returns content and
    needs owner OAuth. Only the former may appear.
    """
    import ast
    import inspect
    source = inspect.getsource(vs)
    assert '"captions"' in source

    # Check the CODE, not the prose. Every API endpoint this module names must
    # be one of the metadata endpoints; captions/download would be content.
    tree = ast.parse(source)
    endpoints = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "safe_api_get"
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            endpoints.add(node.args[0].value)
    assert endpoints <= {"videos", "captions"}, endpoints
    assert "download" not in endpoints
