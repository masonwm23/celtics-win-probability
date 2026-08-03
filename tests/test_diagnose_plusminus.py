"""
Tests for the PLUS_MINUS diagnostic.

No network. The point of these tests is that the LineScore parser must locate
the right result set BY NAME and refuse anything ambiguous, because this
diagnostic is what decides whether the model's target variable is trustworthy.
A parser that quietly returns wrong scores here would be worse than no
diagnostic at all.
"""

import pandas as pd
import pytest

from src.diagnose_plusminus import final_scores_from_summary, build_report


def summary_payload(sets):
    """Build a BoxScoreSummaryV2-shaped payload from {name: (headers, rows)}."""
    return {
        "resource": "boxscoresummary",
        "resultSets": [
            {"name": name, "headers": h, "rowSet": r}
            for name, (h, r) in sets.items()
        ],
    }


def line_score(bos_pts, opp_abbrev, opp_pts):
    return summary_payload({
        "GameSummary": (["GAME_ID"], [["0021700743"]]),
        "LineScore": (
            ["GAME_ID", "TEAM_ABBREVIATION", "PTS"],
            [["0021700743", "BOS", bos_pts],
             ["0021700743", opp_abbrev, opp_pts]],
        ),
    })


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_reads_both_final_scores():
    scores, note = final_scores_from_summary(line_score(111, "DEN", 110))
    assert scores == {"BOS": 111, "DEN": 110}
    assert note == "LineScore"


def test_locates_line_score_by_name_not_position():
    """
    LineScore is deliberately placed last. Reading result sets positionally is a
    classic way to silently return the wrong numbers.
    """
    payload = summary_payload({
        "GameSummary": (["GAME_ID"], [["1"]]),
        "OtherStats": (["GAME_ID", "PTS"], [["1", 999]]),
        "LineScore": (["TEAM_ABBREVIATION", "PTS"],
                      [["BOS", 97], ["POR", 96]]),
    })
    scores, note = final_scores_from_summary(payload)
    assert scores == {"BOS": 97, "POR": 96}
    assert 999 not in scores.values()


# ---------------------------------------------------------------------------
# Refusals. Each of these must return no scores plus an explanation.
# ---------------------------------------------------------------------------

def test_refuses_when_line_score_absent():
    payload = summary_payload({"GameSummary": (["GAME_ID"], [["1"]])})
    scores, note = final_scores_from_summary(payload)
    assert scores == {}
    assert "no LineScore" in note
    assert "GameSummary" in note


def test_refuses_when_no_result_sets():
    scores, note = final_scores_from_summary({"unexpected": True})
    assert scores == {}
    assert "no resultSets" in note


def test_refuses_when_points_column_missing():
    payload = summary_payload({
        "LineScore": (["TEAM_ABBREVIATION", "FG_PCT"],
                      [["BOS", 0.5], ["DEN", 0.4]]),
    })
    scores, note = final_scores_from_summary(payload)
    assert scores == {}
    assert "missing PTS" in note


def test_refuses_on_null_points():
    payload = summary_payload({
        "LineScore": (["TEAM_ABBREVIATION", "PTS"],
                      [["BOS", None], ["DEN", 110]]),
    })
    scores, note = final_scores_from_summary(payload)
    assert scores == {}
    assert "null" in note


def test_refuses_when_not_exactly_two_teams():
    payload = summary_payload({
        "LineScore": (["TEAM_ABBREVIATION", "PTS"], [["BOS", 111]]),
    })
    scores, note = final_scores_from_summary(payload)
    assert scores == {}
    assert "expected 2 teams" in note


# ---------------------------------------------------------------------------
# Verdict logic in the report
# ---------------------------------------------------------------------------

def row(**kw):
    base = {
        "game_id": "0021700743", "season": "2017-18",
        "game_date": "2018-01-29", "matchup": "BOS @ DEN",
        "index_wl": "W", "index_pts": 111, "index_plus_minus": -2.6,
        "bos_pts": 111, "opp_pts": 110, "true_margin": 1,
        "true_winner": "BOS", "pts_agrees": True, "wl_agrees": True,
        "note": "LineScore",
    }
    base.update(kw)
    return base


def test_report_clears_target_when_everything_agrees():
    suspects = pd.DataFrame([row()])
    controls = pd.DataFrame([row(game_date="2016-11-02", index_plus_minus=5.0)])
    text = build_report(suspects, controls, 636)
    assert "The method is sound" in text
    assert "The target variable is safe to train on." in text
    assert "check is NOT being deleted" in text


def test_report_refuses_to_conclude_when_controls_fail():
    """If the controls disagree, the diagnostic is broken and must say so."""
    suspects = pd.DataFrame([row()])
    controls = pd.DataFrame([row(wl_agrees=False, bos_pts=90, opp_pts=100)])
    text = build_report(suspects, controls, 636)
    assert "cannot be trusted" in text
    assert "Do not proceed" in text


def test_report_escalates_when_target_variable_is_wrong():
    suspects = pd.DataFrame([row(wl_agrees=False, bos_pts=108, opp_pts=110)])
    controls = pd.DataFrame([row(game_date="2016-11-02")])
    text = build_report(suspects, controls, 636)
    assert "must be rebuilt" in text
    assert "Do not proceed" in text


def test_report_notes_when_plus_minus_disagrees_with_true_margin():
    suspects = pd.DataFrame([row()])          # margin 1, plus_minus -2.6
    controls = pd.DataFrame([row(game_date="2016-11-02")])
    text = build_report(suspects, controls, 636)
    assert "PLUS_MINUS equals the true margin in 0 of 1" in text


def test_report_handles_unverified_games():
    suspects = pd.DataFrame([row(bos_pts=None, opp_pts=None, true_margin=None,
                                 pts_agrees=None, wl_agrees=None,
                                 note="FETCH FAILED: Timeout")])
    controls = pd.DataFrame([row(game_date="2016-11-02")])
    text = build_report(suspects, controls, 636)
    assert "could not be verified" in text
