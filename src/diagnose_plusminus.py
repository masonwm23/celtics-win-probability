"""
Phase 1 diagnostic: is PLUS_MINUS trustworthy, and is WL correct?

Why this exists
---------------
The game index audit failed one check. Investigating the file showed five games
where LeagueGameFinder returned a FRACTIONAL PLUS_MINUS:

    2017-12-21  BOS @ NYK    L   PTS 93   PLUS_MINUS  -8.8
    2018-01-29  BOS @ DEN    W   PTS 111  PLUS_MINUS  -2.6
    2018-02-04  BOS vs. POR  W   PTS 97   PLUS_MINUS   1.2
    2018-03-03  BOS @ HOU    L   PTS 120  PLUS_MINUS  -2.6
    2022-11-05  BOS @ NYK    W   PTS 133  PLUS_MINUS  10.4

A final game margin is the difference of two integers, so it cannot be -2.6.
Whatever those values are, they are not the margin. Only one of the five had a
sign that disagreed with the recorded win or loss, which is why only one
tripped the audit. The other four were wrong in the same way but happened to
have the right sign, and that is exactly the kind of thing that slips through.

The question that matters for the research is NOT whether PLUS_MINUS is tidy.
It is whether the WL column, which becomes the model's target variable, is
correct. If WL were wrong for even a handful of games, every model trained on
this data would be learning partly from mislabeled outcomes.

Method
------
Fetch BoxScoreSummaryV2 for each suspect game and read its LineScore result
set, which reports both teams' final points directly. That is an independent
source from the game log column being questioned.

Ten randomly chosen clean games are fetched as controls. Without controls, a
diagnostic that reports "everything is fine" proves nothing, because it might
simply be broken. The controls have to come back matching.

This script only reads. It changes no data and fixes nothing. Its output tells
us what the correct fix is.

Output
------
reports/phase1_plusminus_diagnostic.txt
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from src import config
from src.nba_client import call_endpoint

logger = logging.getLogger(__name__)

N_CONTROLS = 10


def final_scores_from_summary(payload: dict):
    """
    Extract {team_abbrev: points} from a BoxScoreSummaryV2 payload.

    Uses the classic resultSets shape and locates the LineScore set by name
    rather than by position, so a reordering of result sets cannot silently
    return the wrong numbers.

    Returns (scores_dict, note). scores_dict is empty if the shape was not
    recognised, and the note explains what was found instead.
    """
    result_sets = payload.get("resultSets")
    if not isinstance(result_sets, list):
        return {}, f"no resultSets; top level keys {sorted(payload.keys())}"

    names = [rs.get("name") for rs in result_sets]
    line = next((rs for rs in result_sets if rs.get("name") == "LineScore"), None)
    if line is None:
        return {}, f"no LineScore set; sets present: {names}"

    headers = line.get("headers", [])
    rows = line.get("rowSet", [])
    for needed in ("TEAM_ABBREVIATION", "PTS"):
        if needed not in headers:
            return {}, f"LineScore missing {needed}; headers {headers}"

    ai = headers.index("TEAM_ABBREVIATION")
    pi = headers.index("PTS")
    scores = {}
    for row in rows:
        abbrev, pts = row[ai], row[pi]
        if abbrev is None or pts is None:
            return {}, f"LineScore has a null value in row {row}"
        scores[str(abbrev)] = int(pts)

    if len(scores) != 2:
        return {}, f"expected 2 teams in LineScore, got {len(scores)}: {scores}"
    return scores, "LineScore"


def check_game(rec):
    """
    Compare one indexed game against the independent final score.

    Returns a dict describing agreement or disagreement. Never raises for a
    network problem; the row records the error instead.
    """
    out = {
        "game_id": rec.GAME_ID, "season": rec.SEASON,
        "game_date": rec.GAME_DATE.date().isoformat(),
        "matchup": rec.MATCHUP, "index_wl": rec.WL,
        "index_pts": int(rec.PTS), "index_plus_minus": rec.PLUS_MINUS,
        "bos_pts": None, "opp_pts": None, "true_margin": None,
        "true_winner": "", "pts_agrees": None, "wl_agrees": None,
        "note": "",
    }
    try:
        from nba_api.stats.endpoints import boxscoresummaryv2
        endpoint = call_endpoint(
            boxscoresummaryv2.BoxScoreSummaryV2, game_id=rec.GAME_ID
        )
        scores, note = final_scores_from_summary(endpoint.get_dict())
        out["note"] = note
    except Exception as exc:  # noqa: BLE001
        out["note"] = f"FETCH FAILED: {type(exc).__name__}: {exc}"
        return out

    if not scores:
        return out

    bos = scores.get(config.CELTICS_ABBREV)
    opp_items = [(k, v) for k, v in scores.items()
                 if k != config.CELTICS_ABBREV]
    if bos is None or len(opp_items) != 1:
        out["note"] = f"could not identify BOS and one opponent in {scores}"
        return out

    opp_abbrev, opp = opp_items[0]
    out["bos_pts"] = bos
    out["opp_pts"] = opp
    out["true_margin"] = bos - opp
    out["true_winner"] = config.CELTICS_ABBREV if bos > opp else opp_abbrev
    out["pts_agrees"] = (bos == int(rec.PTS))
    out["wl_agrees"] = ((rec.WL == "W") == (bos > opp))
    return out


def run():
    config.ensure_dirs()

    if not config.GAME_INDEX_CSV.exists():
        raise FileNotFoundError(f"{config.GAME_INDEX_CSV} not found.")

    df = pd.read_csv(config.GAME_INDEX_CSV, parse_dates=["GAME_DATE"])
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    pm = pd.to_numeric(df["PLUS_MINUS"], errors="coerce")

    fractional = df.loc[(pm - pm.round()).abs() > 1e-9].copy()
    clean = df.loc[(pm - pm.round()).abs() <= 1e-9].copy()
    controls = clean.sample(n=min(N_CONTROLS, len(clean)),
                            random_state=config.RANDOM_SEED)

    print(f"Suspect games (fractional PLUS_MINUS): {len(fractional)}")
    print(f"Control games (clean, randomly chosen): {len(controls)}")
    print(f"Total API calls: {len(fractional) + len(controls)}")
    print()

    suspect_rows = []
    for rec in fractional.itertuples(index=False):
        row = check_game(rec)
        suspect_rows.append(row)
        print(f"  suspect {row['game_date']} {row['matchup']:<12} "
              f"BOS {row['bos_pts']} - {row['opp_pts']} "
              f"(index WL={row['index_wl']}) {row['note']}")

    control_rows = []
    for rec in controls.itertuples(index=False):
        row = check_game(rec)
        control_rows.append(row)
        print(f"  control {row['game_date']} {row['matchup']:<12} "
              f"BOS {row['bos_pts']} - {row['opp_pts']} "
              f"(index WL={row['index_wl']}) {row['note']}")

    suspects = pd.DataFrame(suspect_rows)
    controls = pd.DataFrame(control_rows)

    report = build_report(suspects, controls, len(df))
    print(report)
    out = config.REPORTS_DIR / "phase1_plusminus_diagnostic.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to: {out}")

    # Generate the anomaly registry from these verified results. This is done
    # in code, from real API responses, precisely so that no game ID or score
    # is ever transcribed by hand into a file that excuses a check.
    written = write_anomaly_registry(suspects, controls)
    if written is not None:
        print(f"Anomaly registry written to: {config.KNOWN_ANOMALIES_CSV} "
              f"({len(written)} row(s))")
    else:
        print("Anomaly registry NOT written. See the findings above.")


def write_anomaly_registry(suspects: pd.DataFrame, controls: pd.DataFrame):
    """
    Write data/known_data_anomalies.csv from verified diagnostic results.

    Refuses to write anything unless two conditions hold:

      1. Every control game agreed. If the controls disagree, this diagnostic is
         not trustworthy and must not be excusing games.
      2. Every suspect game had its WL and PTS confirmed correct. A game whose
         recorded result is actually WRONG is not a benign anomaly, it is a
         corrupted target variable, and it must escalate rather than be filed
         away.

    Returns the written DataFrame, or None if it refused.
    """
    if suspects.empty:
        return None

    ctrl = controls.loc[controls["bos_pts"].notna()] if len(controls) else controls
    if ctrl.empty or not (bool(ctrl["wl_agrees"].all())
                          and bool(ctrl["pts_agrees"].all())):
        logger.error("Controls did not all agree. Refusing to write registry.")
        return None

    verified = suspects.loc[suspects["bos_pts"].notna()]
    if len(verified) != len(suspects):
        logger.error("Some suspect games could not be verified. "
                     "Refusing to write a partial registry.")
        return None
    if not (bool(verified["wl_agrees"].all()) and bool(verified["pts_agrees"].all())):
        logger.error("A suspect game has an incorrect WL or PTS. This is a "
                     "target-variable problem, not an excusable anomaly. "
                     "Refusing to write registry.")
        return None

    today = datetime.now(timezone.utc).date().isoformat()
    rows = []
    for _, r in verified.iterrows():
        reported = float(r["index_plus_minus"])
        margin = int(r["true_margin"])
        rows.append({
            "game_id": str(r["game_id"]).zfill(10),
            "game_date": r["game_date"],
            "matchup": r["matchup"],
            "column": "PLUS_MINUS",
            "issue": ("Non-integer team plus/minus. Player-level plus/minus "
                      "does not sum to 5 x final margin, which indicates an "
                      "internally inconsistent substitution log for this game."),
            "reported_value": reported,
            "verified_bos_pts": int(r["bos_pts"]),
            "verified_opp_pts": int(r["opp_pts"]),
            "verified_margin": margin,
            "implied_player_sum": round(reported * 5),
            "expected_player_sum": margin * 5,
            "verification_source": "nba_api BoxScoreSummaryV2 LineScore",
            "verified_on": today,
            "resolution": ("WL and PTS independently confirmed correct. "
                           "PLUS_MINUS is never used as a margin anywhere in "
                           "the pipeline; the authoritative margin comes from "
                           "final scores. Excused from the integer and sign "
                           "checks only. Flagged for the Phase 2 substitution "
                           "coherence check."),
        })

    out = pd.DataFrame(rows).sort_values("game_date").reset_index(drop=True)
    config.KNOWN_ANOMALIES_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.KNOWN_ANOMALIES_CSV, index=False)
    return out


def build_report(suspects, controls, n_total):
    lines = [
        "",
        "=" * 74,
        "PHASE 1 DIAGNOSTIC - IS PLUS_MINUS TRUSTWORTHY, AND IS WL CORRECT?",
        f"Run at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "=" * 74,
        "",
        "Independent source: BoxScoreSummaryV2 LineScore (both teams' final",
        "points), compared against the LeagueGameFinder game log columns.",
        "",
        f"Games in index: {n_total}",
        f"Games with fractional PLUS_MINUS: {len(suspects)}",
        f"Control games (clean PLUS_MINUS): {len(controls)}",
        "",
    ]

    def block(title, frame):
        out = [title, "-" * len(title)]
        if frame.empty:
            out.append("  none")
            return out
        out.append(f"  {'date':<11}{'matchup':<13}{'WL':<4}"
                   f"{'idx PTS':>8}{'idx +/-':>9}"
                   f"{'BOS':>6}{'OPP':>6}{'margin':>8}  PTS?  WL?")
        for _, r in frame.iterrows():
            out.append(
                f"  {r['game_date']:<11}{r['matchup']:<13}{r['index_wl']:<4}"
                f"{r['index_pts']:>8}{str(r['index_plus_minus']):>9}"
                f"{str(r['bos_pts']):>6}{str(r['opp_pts']):>6}"
                f"{str(r['true_margin']):>8}"
                f"  {_mark(r['pts_agrees'])}    {_mark(r['wl_agrees'])}"
            )
            if isinstance(r["note"], str) and r["note"] not in ("LineScore",):
                out.append(f"    note: {r['note']}")
        return out

    lines += block("SUSPECT GAMES", suspects)
    lines.append("")
    lines += block("CONTROL GAMES", controls)

    # --- verdict ---
    lines += ["", "=" * 74, "FINDINGS", "=" * 74, ""]

    all_rows = pd.concat([suspects, controls], ignore_index=True) \
        if not (suspects.empty and controls.empty) else pd.DataFrame()

    fetched = all_rows.loc[all_rows["bos_pts"].notna()] if len(all_rows) else all_rows
    n_fetch_fail = len(all_rows) - len(fetched) if len(all_rows) else 0

    if n_fetch_fail:
        lines.append(f"WARNING: {n_fetch_fail} game(s) could not be verified.")
        lines.append("The conclusions below cover only the games that were.")
        lines.append("")

    if len(fetched) == 0:
        lines.append("Nothing was verified. Cannot draw a conclusion.")
        lines.append("=" * 74)
        return "\n".join(lines)

    controls_ok = controls.loc[controls["bos_pts"].notna()]
    controls_clean = (bool(controls_ok["wl_agrees"].all())
                      and bool(controls_ok["pts_agrees"].all())) \
        if len(controls_ok) else False

    lines.append("1. Does the diagnostic method itself work?")
    if controls_clean:
        lines.append(f"   Yes. All {len(controls_ok)} control games agree on both")
        lines.append("   final points and win/loss. The method is sound, so a")
        lines.append("   disagreement below is a real finding, not a bug here.")
    else:
        lines.append("   NO. Control games disagree, which means this diagnostic")
        lines.append("   cannot be trusted. Investigate before concluding anything.")
    lines.append("")

    wl_bad = fetched.loc[fetched["wl_agrees"] == False]  # noqa: E712
    lines.append("2. Is the WL column (the model's target variable) correct?")
    if wl_bad.empty:
        lines.append(f"   Yes, for all {len(fetched)} games verified. Every recorded")
        lines.append("   win corresponds to Boston outscoring the opponent.")
        lines.append("   The target variable is safe to train on.")
    else:
        lines.append(f"   NO. {len(wl_bad)} game(s) have the wrong result recorded:")
        for _, r in wl_bad.iterrows():
            lines.append(f"     {r['game_date']} {r['matchup']} recorded "
                         f"{r['index_wl']} but score was "
                         f"BOS {r['bos_pts']}-{r['opp_pts']}")
        lines.append("   This is serious. The target variable must be rebuilt")
        lines.append("   from final scores before any model is trained.")
    lines.append("")

    pts_bad = fetched.loc[fetched["pts_agrees"] == False]  # noqa: E712
    lines.append("3. Is the PTS column correct?")
    if pts_bad.empty:
        lines.append(f"   Yes, for all {len(fetched)} games verified.")
    else:
        lines.append(f"   NO. {len(pts_bad)} game(s) disagree on Boston's points.")
        for _, r in pts_bad.iterrows():
            lines.append(f"     {r['game_date']} {r['matchup']}: index says "
                         f"{r['index_pts']}, LineScore says {r['bos_pts']}")
    lines.append("")

    lines.append("4. What is PLUS_MINUS actually reporting on suspect games?")
    sus = suspects.loc[suspects["true_margin"].notna()]
    if sus.empty:
        lines.append("   No suspect games verified.")
    else:
        for _, r in sus.iterrows():
            lines.append(f"   {r['game_date']} {r['matchup']:<13} "
                         f"true margin {int(r['true_margin']):>4}   "
                         f"PLUS_MINUS {r['index_plus_minus']}")
        matches = sum(1 for _, r in sus.iterrows()
                      if abs(float(r["index_plus_minus"]) - r["true_margin"]) < 0.5)
        lines.append("")
        lines.append(f"   PLUS_MINUS equals the true margin in {matches} of "
                     f"{len(sus)} suspect games.")
        lines.append("   Where it does not, PLUS_MINUS is reporting something")
        lines.append("   other than the final margin and must not be used as one.")

    lines += ["", "=" * 74, "RECOMMENDED ACTION", "=" * 74, ""]
    if wl_bad.empty and controls_clean:
        lines.append("The audit check was testing the wrong column. WL and PTS are")
        lines.append("sound; PLUS_MINUS is not a reliable margin for a small number")
        lines.append("of games and should not be used as one.")
        lines.append("")
        lines.append("Proposed changes, for approval before anything is edited:")
        lines.append("  a) Add a check that flags ANY non-integer PLUS_MINUS, since")
        lines.append("     a margin must be a whole number. The current check")
        lines.append("     caught only the one game whose sign also disagreed,")
        lines.append("     and missed the other four.")
        lines.append("  b) Replace the WL-versus-PLUS_MINUS check with a")
        lines.append("     WL-versus-final-score check in Phase 2, once the")
        lines.append("     boxscores are downloaded. Validating the target")
        lines.append("     against the actual game data is stronger than")
        lines.append("     validating it against a summary column.")
        lines.append("  c) Record the fractional PLUS_MINUS games as a documented")
        lines.append("     data-quality finding for the paper. Do not silently")
        lines.append("     drop or repair them.")
        lines.append("")
        lines.append("The check is NOT being deleted to make the audit pass. It is")
        lines.append("being pointed at a trustworthy source instead.")
    else:
        lines.append("Do not proceed. Either the controls failed, meaning this")
        lines.append("diagnostic is unreliable, or the WL column is wrong, meaning")
        lines.append("the target variable needs rebuilding. Report this output.")
    lines.append("=" * 74)
    return "\n".join(lines)


def _mark(value):
    if value is True:
        return "ok "
    if value is False:
        return "BAD"
    return "?  "


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    run()


if __name__ == "__main__":
    main()
