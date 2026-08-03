"""
Tests for the documented anomaly registry and its interaction with the audit.

The registry is the one mechanism in this project that can make a failing check
pass. That makes it the most dangerous file in the codebase, so it gets the most
adversarial tests. Specifically:

  - An anomaly NOT in the registry must still fail the audit.
  - A registry row without evidence must be rejected on load.
  - Excusing a game for one column must not excuse it for another.
  - The registry must appear in the report, so excused games are visible.
  - The diagnostic must REFUSE to write a registry when its controls fail, or
    when a suspect game's recorded result is actually wrong.
"""

import pandas as pd
import pytest

from src import config, known_anomalies
from src.diagnose_plusminus import write_anomaly_registry
from src.validate_game_index import run_audit
from tests.test_game_index import make_clean_index


@pytest.fixture
def registry_path(tmp_path, monkeypatch):
    """Point the registry at a temp file so no test touches real project data."""
    path = tmp_path / "known_data_anomalies.csv"
    monkeypatch.setattr(config, "KNOWN_ANOMALIES_CSV", path)
    return path


def write_registry(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def good_row(game_id="0021700743", column="PLUS_MINUS"):
    return {
        "game_id": game_id, "game_date": "2018-01-29", "matchup": "BOS @ DEN",
        "column": column, "issue": "Non-integer team plus/minus.",
        "reported_value": -2.6, "verified_bos_pts": 111,
        "verified_opp_pts": 110, "verified_margin": 1,
        "implied_player_sum": -13, "expected_player_sum": 5,
        "verification_source": "nba_api BoxScoreSummaryV2 LineScore",
        "verified_on": "2026-07-30", "resolution": "WL and PTS confirmed.",
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_missing_registry_is_empty_not_an_error(registry_path):
    assert known_anomalies.load_anomalies().empty
    assert known_anomalies.excused_game_ids("PLUS_MINUS") == set()


def test_loads_valid_registry(registry_path):
    write_registry(registry_path, [good_row()])
    assert known_anomalies.excused_game_ids("PLUS_MINUS") == {"0021700743"}


def test_game_ids_keep_leading_zeros(registry_path):
    """
    NBA game IDs start with 002. If pandas reads the column as an integer the
    leading zeros vanish and the ID would never match, silently un-excusing the
    game or, worse, matching a different one.
    """
    write_registry(registry_path, [good_row(game_id="0021700743")])
    ids = known_anomalies.excused_game_ids("PLUS_MINUS")
    assert ids == {"0021700743"}
    assert all(len(i) == 10 for i in ids)


def test_rejects_registry_missing_evidence_columns(registry_path):
    row = good_row()
    del row["verification_source"]
    write_registry(registry_path, [row])
    with pytest.raises(known_anomalies.AnomalyRegistryError, match="missing"):
        known_anomalies.load_anomalies()


def test_rejects_row_with_blank_verification_source(registry_path):
    write_registry(registry_path, [good_row() | {"verification_source": "  "}])
    with pytest.raises(known_anomalies.AnomalyRegistryError,
                       match="no verification_source"):
        known_anomalies.load_anomalies()


def test_rejects_row_with_blank_verified_on(registry_path):
    write_registry(registry_path, [good_row() | {"verified_on": None}])
    with pytest.raises(known_anomalies.AnomalyRegistryError,
                       match="no verified_on"):
        known_anomalies.load_anomalies()


def test_excuse_is_scoped_to_one_column(registry_path):
    """A bad PLUS_MINUS does not buy forgiveness for a bad GAME_ID."""
    write_registry(registry_path, [good_row(column="PLUS_MINUS")])
    assert known_anomalies.excused_game_ids("PLUS_MINUS") == {"0021700743"}
    assert known_anomalies.excused_game_ids("GAME_ID") == set()
    assert known_anomalies.excused_game_ids("WL") == set()


# ---------------------------------------------------------------------------
# Audit interaction. This is the important part.
# ---------------------------------------------------------------------------

def index_with_fractional_pm(game_id, value=-2.6):
    """Clean synthetic index, with one game given a fractional plus/minus."""
    df = make_clean_index().reset_index(drop=True)
    df.loc[0, "GAME_ID"] = game_id
    df.loc[0, "PLUS_MINUS"] = value
    # Keep the sign consistent with the recorded result so the ONLY thing wrong
    # is that the value is not a whole number. That isolates check 10.
    df.loc[0, "WL"] = "L"
    df.loc[0, "CELTICS_WON"] = 0
    return df


def failed_names(df):
    return [name for name, _, _ in run_audit(df).failed]


def test_undocumented_fractional_plus_minus_still_fails(registry_path):
    """The whole point. An anomaly nobody documented is a hard failure."""
    df = index_with_fractional_pm("0021700743")
    assert known_anomalies.load_anomalies().empty
    fails = failed_names(df)
    assert any("whole number" in f for f in fails)


def test_documented_fractional_plus_minus_is_excused(registry_path):
    write_registry(registry_path, [good_row(game_id="0021700743")])
    df = index_with_fractional_pm("0021700743")
    assert failed_names(df) == []


def test_documenting_one_game_does_not_excuse_another(registry_path):
    """A registry entry must excuse exactly the game it names."""
    write_registry(registry_path, [good_row(game_id="0029999999")])
    df = index_with_fractional_pm("0021700743")
    fails = failed_names(df)
    assert any("whole number" in f for f in fails)


def test_sign_disagreement_also_excused_only_when_documented(registry_path):
    df = make_clean_index().reset_index(drop=True)
    df.loc[0, "GAME_ID"] = "0021700743"
    df.loc[0, "WL"] = "W"
    df.loc[0, "CELTICS_WON"] = 1
    df.loc[0, "PLUS_MINUS"] = -2.6          # win with a negative margin

    assert any("whole number" in f or "sign" in f for f in failed_names(df))

    write_registry(registry_path, [good_row(game_id="0021700743")])
    assert failed_names(df) == []


def test_audit_records_deferred_checks(registry_path):
    """Deferred checks must be visible, not silently absent."""
    auditor = run_audit(make_clean_index())
    names = [n for n, _ in auditor.deferred]
    assert any("final scores" in n for n in names)
    assert any("5 x margin" in n for n in names)


def test_registry_renders_evidence_into_the_report(registry_path):
    write_registry(registry_path, [good_row()])
    text = "\n".join(known_anomalies.render_registry())
    assert "0021700743" in text
    assert "BoxScoreSummaryV2" in text
    assert "2026-07-30" in text
    assert "verified_margin" in text


def test_empty_registry_says_so_explicitly(registry_path):
    text = "\n".join(known_anomalies.render_registry())
    assert "Registry is empty" in text


# ---------------------------------------------------------------------------
# The diagnostic's refusal to write a registry it cannot justify
# ---------------------------------------------------------------------------

def diag_row(**kw):
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


def test_registry_written_when_everything_checks_out(registry_path):
    written = write_anomaly_registry(
        pd.DataFrame([diag_row()]), pd.DataFrame([diag_row()])
    )
    assert written is not None
    assert len(written) == 1
    assert registry_path.exists()
    # Round trip through the loader, which enforces the evidence requirements.
    assert known_anomalies.excused_game_ids("PLUS_MINUS") == {"0021700743"}


def test_registry_refused_when_controls_disagree(registry_path):
    """If the verification method is broken it must not excuse anything."""
    written = write_anomaly_registry(
        pd.DataFrame([diag_row()]),
        pd.DataFrame([diag_row(wl_agrees=False)]),
    )
    assert written is None
    assert not registry_path.exists()


def test_registry_refused_when_a_suspect_result_is_actually_wrong(registry_path):
    """A wrong WL is a corrupted target variable, not a benign anomaly."""
    written = write_anomaly_registry(
        pd.DataFrame([diag_row(wl_agrees=False, bos_pts=108, opp_pts=110)]),
        pd.DataFrame([diag_row()]),
    )
    assert written is None
    assert not registry_path.exists()


def test_registry_refused_when_a_suspect_could_not_be_verified(registry_path):
    """Partial evidence is not evidence. No half registries."""
    written = write_anomaly_registry(
        pd.DataFrame([diag_row(), diag_row(game_id="0021700784", bos_pts=None,
                                          wl_agrees=None, pts_agrees=None)]),
        pd.DataFrame([diag_row()]),
    )
    assert written is None
    assert not registry_path.exists()


def test_registry_derives_player_sums_rather_than_transcribing(registry_path):
    """
    The implied and expected player sums must be computed from the verified
    numbers. Hand transcription is what produced four wrong game IDs earlier in
    this project, which is why this file is generated instead of typed.
    """
    written = write_anomaly_registry(
        pd.DataFrame([diag_row(index_plus_minus=10.4, bos_pts=133,
                               opp_pts=118, true_margin=15)]),
        pd.DataFrame([diag_row()]),
    )
    assert written is not None
    row = written.iloc[0]
    assert row["implied_player_sum"] == 52      # 10.4 * 5
    assert row["expected_player_sum"] == 75     # 15 * 5
    assert row["verified_margin"] == 15
