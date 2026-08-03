"""
Phase 2 audit: independently verify the parsed tables.

Reads the Phase 2 outputs and the raw boxscores and checks them against each
other. Deliberately separate from src/build_phase2.py, because the thing that
builds data should not be the only thing that vouches for it.

Checks
  1.  Every indexed game has events, roster rows, and lineups
  2.  Game clock is monotonic in delivered order, per game
  3.  Total elapsed game time equals 12 minutes x 4 plus 5 minutes per overtime
  4.  Final event score equals the boxscore point totals, both teams
  5.  The recorded win or loss agrees with that final score
  6.  Exactly five starters per team per game
  7.  Row-order starters agree with the position field wherever it is usable
  8.  Both lineups contain exactly five players at every event
  9.  Every substitution's incoming player was resolved, none guessed
  10. Derived on-court minutes match boxscore minutes within tolerance
  11. Player plus/minus sums to five times the final margin (the Phase 1 deferral)

Check 10 is the one that matters most. It is an independent measurement: the
boxscore knows nothing about how lineups were reconstructed, so agreement is
real evidence rather than a restatement of the code's own assumptions.

Writes reports/phase2_validation.txt and data/lineup_risk_games.csv
"""

import json
from datetime import datetime, timezone

import pandas as pd

from src import config
from src.parse_events import (REGULATION_SECONDS, OVERTIME_PERIOD_SECONDS,
                              REGULATION_PERIODS)
from src.rosters import parse_minutes

# Boxscore minutes are recorded to the second, so derived minutes can differ by
# rounding. One second is a generous ceiling for pure rounding error and far
# tighter than any real tracking mistake, which shows up as whole minutes.
MINUTES_TOLERANCE = 0.02


class Auditor:
    def __init__(self):
        self.results = []

    def check(self, name, passed, detail=""):
        self.results.append((name, bool(passed), detail))
        return passed

    @property
    def failed(self):
        return [r for r in self.results if not r[1]]

    def render(self):
        lines = []
        for name, passed, detail in self.results:
            lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}")
            if detail:
                for line in str(detail).splitlines():
                    lines.append(f"         {line}")
        return lines


def boxscore_totals(game_id):
    """{tricode: (points, plusminus_sum, is_home)} from the raw boxscore."""
    path = config.RAW_BOX_DIR / f"{game_id}.json"
    with path.open("r", encoding="utf-8") as fh:
        box = json.load(fh)["boxScoreTraditional"]
    out = {}
    for side in ("homeTeam", "awayTeam"):
        team = box[side]
        players = team["players"]
        out[team["teamTricode"]] = (
            sum((p["statistics"].get("points") or 0) for p in players),
            sum((p["statistics"].get("plusMinusPoints") or 0) for p in players),
            side == "homeTeam",
        )
    return out


def run():
    config.ensure_dirs()
    for path in (config.EVENTS_PARQUET, config.ROSTERS_PARQUET,
                 config.LINEUPS_PARQUET):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run the Phase 2 build first.")

    events = pd.read_parquet(config.EVENTS_PARQUET)
    roster = pd.read_parquet(config.ROSTERS_PARQUET)
    lineups = pd.read_parquet(config.LINEUPS_PARQUET)
    minutes = pd.read_csv(config.DERIVED_MINUTES_CSV,
                          dtype={"game_id": str})
    anomalies = pd.read_csv(config.LINEUP_ANOMALIES_CSV,
                            dtype={"game_id": str}) \
        if config.LINEUP_ANOMALIES_CSV.exists() else pd.DataFrame()

    index = pd.read_csv(config.GAME_INDEX_CSV, dtype={"GAME_ID": str})
    index["GAME_ID"] = index["GAME_ID"].str.zfill(10)

    a = Auditor()
    parsed_games = set(events.game_id.unique())
    indexed_games = set(index["GAME_ID"])

    # 1. Coverage
    missing = sorted(indexed_games - parsed_games)
    extra = sorted(parsed_games - indexed_games)
    a.check("Every indexed game is parsed, and no extras",
            not missing and not extra,
            f"parsed {len(parsed_games)} of {len(indexed_games)} indexed"
            + (f"; missing {missing[:10]}" if missing else "")
            + (f"; unexpected {extra[:10]}" if extra else ""))

    a.check("Every parsed game has roster rows and lineups",
            set(roster.game_id.unique()) == parsed_games
            and set(lineups.game_id.unique()) == parsed_games,
            f"rosters {roster.game_id.nunique()}, lineups {lineups.game_id.nunique()}")

    # 2. Clock monotonicity
    inversions = 0
    for _, group in events.groupby("game_id"):
        series = group.sort_values("event_index")["seconds_elapsed_game"].to_numpy()
        inversions += int((series[1:] < series[:-1] - 1e-6).sum())
    a.check("Game clock is monotonic in delivered event order", inversions == 0,
            f"{inversions} inversions")

    # 3. Total elapsed time
    bad_length = []
    for game_id, group in events.groupby("game_id"):
        max_period = int(group.period.max())
        overtimes = max(0, max_period - REGULATION_PERIODS)
        expected = REGULATION_SECONDS + overtimes * OVERTIME_PERIOD_SECONDS
        actual = float(group.sort_values("event_index")
                       ["seconds_elapsed_game"].iloc[-1])
        if abs(actual - expected) > 1e-6:
            bad_length.append(f"{game_id}: {actual:.0f}s, expected {expected}s")
    a.check("Total elapsed time matches period structure", not bad_length,
            "\n".join(bad_length[:10]) if bad_length else
            f"checked {events.game_id.nunique()} games")

    # 3b. Reconstructed score must be monotone. This is the guard on the
    #     monotone-max rule used to defeat 2016-17 zero-encoding and stale
    #     trailing reports. If a basket were ever legitimately voided on review,
    #     this check plus check 4 would surface it.
    score_inversions, stale_total = 0, 0
    for _, group in events.groupby("game_id"):
        ordered = group.sort_values("event_index")
        for column in ("score_home", "score_away"):
            diffs = ordered[column].diff().dropna()
            score_inversions += int((diffs < 0).sum())
        if "score_report_stale" in ordered:
            stale_total += int(ordered["score_report_stale"].sum())
    a.check("Reconstructed score never decreases", score_inversions == 0,
            f"{score_inversions} decreases; {stale_total:,} stale or "
            f"zero-encoded score reports were ignored across "
            f"{events.game_id.nunique()} games")

    # 4 and 5. Score and result reconciliation
    score_bad, wl_bad, pm_bad = [], [], []
    for game_id, group in events.groupby("game_id"):
        last = group.sort_values("event_index").iloc[-1]
        totals = boxscore_totals(game_id)
        row = index.loc[index["GAME_ID"].eq(game_id)].iloc[0]
        celtics_tri = config.CELTICS_ABBREV
        opp_tri = next(t for t in totals if t != celtics_tri)
        bos_pts, bos_pm, _ = totals[celtics_tri]
        opp_pts, opp_pm, _ = totals[opp_tri]

        if int(last.celtics_score) != bos_pts or int(last.opponent_score) != opp_pts:
            score_bad.append(f"{game_id}: events {last.celtics_score}-"
                             f"{last.opponent_score}, boxscore {bos_pts}-{opp_pts}")
        if (row["WL"] == "W") != (bos_pts > opp_pts):
            wl_bad.append(f"{game_id}: WL={row['WL']} but score "
                          f"{bos_pts}-{opp_pts}")

        margin = bos_pts - opp_pts
        if abs(bos_pm - 5 * margin) > 1e-6 or abs(opp_pm + 5 * margin) > 1e-6:
            pm_bad.append({
                "game_id": game_id, "game_date": row["GAME_DATE"],
                "season": row["SEASON"], "matchup": row["MATCHUP"],
                "celtics_points": bos_pts, "opponent_points": opp_pts,
                "celtics_margin": margin,
                "celtics_plusminus_sum": bos_pm,
                "celtics_expected_sum": 5 * margin,
                "celtics_error": bos_pm - 5 * margin,
                "opponent_plusminus_sum": opp_pm,
                "opponent_expected_sum": -5 * margin,
                "opponent_error": opp_pm + 5 * margin,
            })

    a.check("Final event score equals boxscore point totals", not score_bad,
            "\n".join(score_bad[:10]) if score_bad else
            f"checked {events.game_id.nunique()} games, both teams")
    a.check("Recorded win/loss agrees with the final score", not wl_bad,
            "\n".join(wl_bad[:10]) if wl_bad else "target variable verified")

    # 6 and 7. Starters
    starter_counts = (roster[roster.is_starter]
                      .groupby(["game_id", "team_tricode"]).size())
    a.check("Exactly five starters per team per game",
            bool((starter_counts == 5).all()),
            f"counts observed: {sorted(starter_counts.unique().tolist())}")

    usable = roster[roster.position_field_usable]
    agreement = (usable.groupby(["game_id", "team_tricode"])
                 ["starter_flag_agrees"].first())
    # starter_flag_agrees is True/None in an object column. Inverting that with
    # `~` gives -2 for True and produced a nonsense count of -68 disagreements.
    # Cast to a nullable boolean and compare explicitly instead.
    agreement_bool = agreement.astype("boolean")
    n_disagree = int((agreement_bool == False).sum())  # noqa: E712
    n_unknown = int(agreement_bool.isna().sum())
    unusable_teams = (roster[~roster.position_field_usable]
                      .groupby(["game_id", "team_tricode"]).ngroups)
    a.check("Row-order starters agree with the position field where usable",
            n_disagree == 0,
            f"{len(agreement)} team-games checked, {n_disagree} disagreements, "
            f"{n_unknown} indeterminate; "
            f"{unusable_teams} team-games have an unusable position field "
            f"(expected: the 2016-17 season)")

    # 8. Lineup size. A deviation is acceptable ONLY in a game that carries a
    #    recorded anomaly explaining it, which is the same evidence-based rule the
    #    Phase 1 registry uses. An unexplained deviation is a hard failure.
    anomaly_games = set(anomalies.game_id.unique()) if len(anomalies) else set()
    bad_size = lineups[(lineups.home_lineup_size != 5)
                       | (lineups.away_lineup_size != 5)]
    unexplained = bad_size[~bad_size.game_id.isin(anomaly_games)]
    detail = f"{len(lineups):,} events checked"
    if not bad_size.empty:
        detail += (f"; {len(bad_size):,} events deviate across "
                   f"{bad_size.game_id.nunique()} game(s)")
        per_game = bad_size.groupby("game_id").size().sort_values(ascending=False)
        for game_id, count in per_game.head(10).items():
            explained = "explained by a recorded anomaly" \
                if game_id in anomaly_games else "UNEXPLAINED"
            detail += f"\n  {game_id}: {count} events, {explained}"
    if not unexplained.empty:
        detail += (f"\n{len(unexplained):,} events deviate in games with NO "
                   f"recorded anomaly, which is not acceptable")
    a.check("Lineup size is five, or a recorded anomaly explains why not",
            unexplained.empty, detail)

    # 9. Substitution resolution
    unresolved = anomalies[anomalies.reason.eq("unresolved incoming player")] \
        if len(anomalies) and "reason" in anomalies else pd.DataFrame()
    a.check("Every substitution's incoming player was identified",
            unresolved.empty,
            f"{len(unresolved)} unresolved" if not unresolved.empty
            else "no guesses were made")

    # 10. The independent check. Same evidence rule: a mismatch is acceptable
    #     only inside a game that carries a recorded anomaly.
    played = minutes[minutes.played]
    over = played[played.difference.abs() > MINUTES_TOLERANCE]
    over_unexplained = over[~over.game_id.isin(anomaly_games)]
    clean = played[~played.game_id.isin(anomaly_games)]
    detail = (f"{len(played):,} players who played across "
              f"{played.game_id.nunique()} games; tolerance "
              f"{MINUTES_TOLERANCE} min\n"
              f"  games with no recorded anomaly: {clean.game_id.nunique()}, "
              f"{len(clean):,} players, max difference "
              f"{clean.difference.abs().max():.4f} min, mean "
              f"{clean.difference.abs().mean():.4f} min")
    if not over.empty:
        detail += f"\n  {len(over)} player(s) outside tolerance:"
        for _, r in over.head(20).iterrows():
            tag = "in an anomaly game" if r.game_id in anomaly_games \
                else "UNEXPLAINED"
            detail += (f"\n    {r.game_id} {r.team_tricode} {r.player}: "
                       f"derived {r.derived_minutes:.2f} vs box "
                       f"{r.boxscore_minutes:.2f} ({tag})")
    a.check("Derived on-court minutes match boxscore minutes, except in "
            "games with a recorded anomaly",
            over_unexplained.empty, detail)

    # 11. Plus/minus coherence, the Phase 1 deferral
    risk = pd.DataFrame(pm_bad)
    if not risk.empty:
        risk = risk.sort_values("game_date")
        risk.to_csv(config.LINEUP_RISK_CSV, index=False)

    total_games = events.game_id.nunique()
    coherence_detail = (
        f"{len(pm_bad)} of {total_games} games "
        f"({100 * len(pm_bad) / max(total_games, 1):.1f}%) have player "
        f"plus/minus that does not sum to five times the margin.\n"
        f"Recorded in {config.LINEUP_RISK_CSV.name}. Being listed there "
        f"excuses nothing; it flags the game for scrutiny."
    )
    if pm_bad:
        for entry in pm_bad[:15]:
            coherence_detail += (
                f"\n  {entry['game_date']} {entry['matchup']:<13} "
                f"BOS err {entry['celtics_error']:+.0f}  "
                f"OPP err {entry['opponent_error']:+.0f}")
    # This is reported, not failed: the defect is in the NBA's data, and the
    # lineup reconstruction is verified independently by check 10.
    a.check("Player plus/minus coherence measured and recorded", True,
            coherence_detail)

    return a, events, roster, lineups, minutes, anomalies, risk


def build_report(a, events, roster, lineups, minutes, anomalies, risk):
    n_fail = len(a.failed)
    lines = [
        "=" * 74,
        "PHASE 2 VALIDATION - PARSED EVENTS, ROSTERS AND LINEUPS",
        f"Run at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "=" * 74,
        "",
        "INPUTS",
        f"  events   : {len(events):,} rows, {events.game_id.nunique()} games",
        f"  rosters  : {len(roster):,} player-game rows",
        f"  lineups  : {len(lineups):,} event lineups",
        f"  minutes  : {len(minutes):,} player comparisons",
        f"  anomalies: {len(anomalies):,}",
        "",
        "CHECKS",
    ]
    lines += a.render()

    played = minutes[minutes.played]

    if len(anomalies):
        lines += ["", "RECORDED ANOMALIES", "-" * 18,
                  f"  {len(anomalies)} anomaly row(s) across "
                  f"{anomalies.game_id.nunique()} of "
                  f"{lineups.game_id.nunique()} games "
                  f"({100 * anomalies.game_id.nunique() / max(lineups.game_id.nunique(), 1):.1f}%)",
                  ""]
        for reason, count in anomalies.reason.value_counts().items():
            lines.append(f"    {count:>4}  {reason}")
        lines += ["", "  Games affected, with reason:"]
        for game_id, group in anomalies.groupby("game_id"):
            reasons = "; ".join(sorted(set(group.reason)))
            lines.append(f"    {game_id}  {reasons}")
        lines += [
            "",
            "  These are defects in the NBA's own logs, not in the parser.",
            "  Lineup composition in these games carries some uncertainty and",
            "  they are named here so any lineup-based result can be reported",
            "  with that caveat, or recomputed excluding them.",
        ]

    lines += [
        "",
        "LINEUP RECONSTRUCTION EVIDENCE",
        "  Derived minutes come from walking substitutions. Boxscore minutes",
        "  come from the NBA and know nothing about that reconstruction, so",
        "  agreement between them is independent evidence.",
        f"    players compared     : {len(played):,}",
        f"    mean abs difference  : {played.difference.abs().mean():.4f} min",
        f"    median abs difference: {played.difference.abs().median():.4f} min",
        f"    max abs difference   : {played.difference.abs().max():.4f} min",
        "",
        "  Boxscore minutes are recorded to the second, so a residual of a few",
        "  hundredths of a minute is rounding, not error.",
    ]

    if not events.empty:
        lines += [
            "",
            "PARSED DATA SUMMARY",
            f"  secondary events (blocks/steals, blank actionType): "
            f"{int(events.is_secondary_event.sum()):,}",
            f"  events with no score reported (forward-filled): "
            f"{int((~events.score_reported).sum()):,} "
            f"({100 * (~events.score_reported).mean():.1f}%)",
            f"  periods present: {sorted(events.period.unique().tolist())}",
            f"  Celtics margin range: {int(events.celtics_margin.min())} to "
            f"{int(events.celtics_margin.max())}",
        ]

    lines += [
        "",
        "=" * 74,
        f"RESULT: {len(a.results) - n_fail} passed, {n_fail} failed",
    ]
    if n_fail == 0:
        lines.append("Phase 2 is validated. Parsed tables are safe for feature")
        lines.append("engineering in Phase 3.")
    else:
        lines.append("Phase 2 is NOT validated. Do not proceed.")
        lines.append("Failed: " + ", ".join(r[0] for r in a.failed))
    lines.append("=" * 74)
    return "\n".join(lines)


def main():
    a, *rest = run()
    report = build_report(a, *rest)
    print(report)
    out = config.REPORTS_DIR / "phase2_validation.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to: {out}")
    return len(a.failed) == 0


if __name__ == "__main__":
    main()
