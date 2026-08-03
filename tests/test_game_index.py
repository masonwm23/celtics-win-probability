"""
Tests for the Phase 1 game index logic.

These run with no network access. They exist to prove two things:

  1. parse_matchup correctly extracts opponent and home/away, and REFUSES
     anything it does not recognise instead of guessing.

  2. The validation audit actually catches problems. A validation script that
     has only ever seen clean data is not evidence of anything. So each test
     below deliberately corrupts a synthetic-but-correctly-shaped game index
     and asserts that the specific check fails.

The synthetic index is clearly labelled as synthetic. It is a test fixture, not
research data, and it never touches data/.

HOW TO RUN IN SPYDER
  Open scripts/03_run_tests.py and press F5.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.pull_game_index import parse_matchup
from src.validate_game_index import run_audit

# The 29 non-Boston franchise abbreviations as nba_api reports them for the
# 2016-17 through 2023-24 window.
OPPONENTS = [
    "ATL", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU",
    "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC",
    "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]


# ---------------------------------------------------------------------------
# parse_matchup
# ---------------------------------------------------------------------------

def test_parse_matchup_home():
    assert parse_matchup("BOS vs. ATL") == ("ATL", True)


def test_parse_matchup_away():
    assert parse_matchup("BOS @ ATL") == ("ATL", False)


def test_parse_matchup_tolerates_padding():
    assert parse_matchup("  BOS vs. NYK  ") == ("NYK", True)


@pytest.mark.parametrize("bad", [
    "BOS ATL",          # no separator
    "ATL vs. BOS",      # Boston not on the left
    "BOS vs. ATLANTA",  # not a 3 letter abbreviation
    "",
    None,
    123,
])
def test_parse_matchup_rejects_bad_input(bad):
    """The parser must raise, never silently return a wrong answer."""
    with pytest.raises(ValueError):
        parse_matchup(bad)


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

def make_clean_index(seed=0) -> pd.DataFrame:
    """
    Build a SYNTHETIC game index that satisfies every audit check.

    This is a test fixture only. It fabricates dates, scores and results in the
    correct shape so the audit can be exercised. It is never written to data/
    and is never used for any research result.
    """
    rng = np.random.default_rng(seed)
    rows = []
    game_counter = 0

    for season, n_games in config.EXPECTED_GAME_COUNTS.items():
        start_year = int(season.split("-")[0])
        season_start = pd.Timestamp(year=start_year, month=10, day=20)

        # Exactly half home for 82-game seasons, and an even split for the
        # shortened seasons too, which keeps the balance check satisfied.
        home_flags = [True] * (n_games // 2) + [False] * (n_games - n_games // 2)
        rng.shuffle(home_flags)

        # One game every two days guarantees unique dates inside the window.
        for i in range(n_games):
            game_counter += 1
            is_home = bool(home_flags[i])
            bos_pts = int(rng.integers(90, 130))
            opp_pts = int(rng.integers(90, 130))
            while opp_pts == bos_pts:          # NBA games cannot end tied
                opp_pts = int(rng.integers(90, 130))
            won = bos_pts > opp_pts

            rows.append({
                "SEASON_ID": f"2{start_year}",
                "TEAM_ID": config.CELTICS_TEAM_ID,
                "TEAM_ABBREVIATION": "BOS",
                "GAME_ID": f"002{game_counter:07d}",
                "GAME_DATE": season_start + pd.Timedelta(days=2 * i),
                "MATCHUP": f"BOS {'vs.' if is_home else '@'} "
                           f"{OPPONENTS[i % len(OPPONENTS)]}",
                "WL": "W" if won else "L",
                "PTS": bos_pts,
                # Float, because that is what LeagueGameFinder actually returns
                # (the real file contains values like 5.0 and -4.0). Keeping the
                # fixture's dtype faithful to the API matters: an int column
                # would hide the fractional-value problem this project found.
                "PLUS_MINUS": float(bos_pts - opp_pts),
                "SEASON": season,
                "OPPONENT_ABBREV": OPPONENTS[i % len(OPPONENTS)],
                "IS_HOME": is_home,
                "CELTICS_WON": int(won),
            })

    return pd.DataFrame(rows)


def failed_names(df):
    return [name for name, _, _ in run_audit(df).failed]


# ---------------------------------------------------------------------------
# The audit passes on clean data
# ---------------------------------------------------------------------------

def test_clean_index_passes_every_check():
    df = make_clean_index()
    assert len(df) == config.EXPECTED_TOTAL_GAMES == 636
    assert failed_names(df) == [], f"unexpected failures: {failed_names(df)}"


# ---------------------------------------------------------------------------
# The audit catches each specific corruption
# ---------------------------------------------------------------------------

def test_audit_catches_missing_game():
    df = make_clean_index().iloc[:-1]          # drop one game
    fails = failed_names(df)
    assert any("Season game counts" in f for f in fails)


def test_audit_catches_duplicate_game_id():
    df = make_clean_index()
    df.loc[df.index[5], "GAME_ID"] = df.loc[df.index[4], "GAME_ID"]
    fails = failed_names(df)
    assert any("unique" in f for f in fails)


def test_audit_catches_playoff_game_id():
    df = make_clean_index()
    df.loc[df.index[10], "GAME_ID"] = "0042100101"   # playoff prefix
    fails = failed_names(df)
    assert any("format" in f for f in fails)


def test_audit_catches_boston_as_opponent():
    df = make_clean_index()
    df.loc[df.index[3], "OPPONENT_ABBREV"] = "BOS"
    fails = failed_names(df)
    assert any("never BOS" in f for f in fails)


def test_audit_catches_wrong_opponent_count():
    df = make_clean_index()
    df["OPPONENT_ABBREV"] = "ATL"
    fails = failed_names(df)
    assert any("Distinct opponents" in f for f in fails)


def test_audit_catches_home_away_imbalance():
    df = make_clean_index()
    mask = df["SEASON"].eq("2016-17")
    df.loc[mask, "IS_HOME"] = True
    fails = failed_names(df)
    assert any("balanced" in f for f in fails)


def test_audit_catches_date_in_wrong_season():
    df = make_clean_index()
    df.loc[df.index[0], "GAME_DATE"] = pd.Timestamp("2005-01-01")
    fails = failed_names(df)
    assert any("window" in f for f in fails)


def test_audit_catches_two_games_same_day():
    df = make_clean_index().reset_index(drop=True)
    df.loc[1, "GAME_DATE"] = df.loc[0, "GAME_DATE"]
    fails = failed_names(df)
    assert any("same calendar date" in f for f in fails)


def test_audit_catches_impossible_score():
    df = make_clean_index()
    df.loc[df.index[7], "PTS"] = 4
    fails = failed_names(df)
    assert any("point totals" in f for f in fails)


def test_audit_catches_win_loss_contradiction():
    """A recorded win with a negative margin must be caught."""
    df = make_clean_index().reset_index(drop=True)
    idx = df.index[df["CELTICS_WON"].eq(1)][0]
    df.loc[idx, "PLUS_MINUS"] = -9
    fails = failed_names(df)
    assert any("PLUS_MINUS" in f for f in fails)


def test_audit_catches_missing_plus_minus():
    df = make_clean_index().reset_index(drop=True)
    df.loc[0, "PLUS_MINUS"] = np.nan
    fails = failed_names(df)
    assert any("PLUS_MINUS" in f for f in fails)


# ---------------------------------------------------------------------------
# Config sanity
# ---------------------------------------------------------------------------

def test_expected_totals_are_internally_consistent():
    assert len(config.SEASONS) == 8
    assert set(config.SEASONS) == set(config.EXPECTED_GAME_COUNTS)
    assert config.EXPECTED_TOTAL_GAMES == 636
