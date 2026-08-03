"""
Phase 1, step 3: validate the game index.

The pull script already checks structure. This script is the independent audit,
kept separate on purpose: the thing that builds data should not be the only
thing that vouches for it.

Every check either passes, or fails with the offending rows printed. A written
report goes to reports/phase1_game_index_validation.txt so the paper has a
citable record.

Checks performed
  1.  Season game counts match the documented expectations
  2.  GAME_ID values are unique
  3.  GAME_ID values have the NBA regular season format (002 prefix, 10 chars)
  4.  Opponent abbreviations are 3 letters and Boston never plays itself
  5.  Distinct opponent count is 29
  6.  Home and away games are balanced within each season
  7.  Game dates fall inside a plausible window for their season
  8.  Boston never plays two games on the same calendar date
  9.  Final scores are in a plausible NBA range
  10. Win/loss agrees with the sign of the plus/minus column

HOW TO RUN IN SPYDER
  Open scripts/02_validate_game_index.py and press F5.
"""

from datetime import datetime, timezone

import pandas as pd

from src import config, known_anomalies

# Boston played every other franchise at least once in every full season, so
# 29 distinct opponents is the correct expectation across the whole dataset.
EXPECTED_DISTINCT_OPPONENTS = 29

# Plausible bounds for an NBA team's points in a single game. Deliberately
# generous. These catch parsing disasters, not unusual games.
MIN_PLAUSIBLE_PTS = 50
MAX_PLAUSIBLE_PTS = 175


class Auditor:
    """Collects pass/fail results so one failure does not hide the others."""

    def __init__(self):
        self.results = []      # (name, passed, detail)
        self.deferred = []     # (name, reason)

    def check(self, name, passed, detail=""):
        self.results.append((name, bool(passed), detail))
        return passed

    def defer(self, name, reason):
        """
        Record a check that cannot run yet.

        Deferred checks are printed in the report so that an unrun check is
        visible rather than simply absent. A check nobody remembers is a check
        that never happens.
        """
        self.deferred.append((name, reason))

    @property
    def failed(self):
        return [r for r in self.results if not r[1]]

    def render(self):
        lines = []
        for name, passed, detail in self.results:
            status = "PASS" if passed else "FAIL"
            lines.append(f"  [{status}] {name}")
            if detail:
                for dline in str(detail).splitlines():
                    lines.append(f"         {dline}")
        for name, reason in self.deferred:
            lines.append(f"  [DEFER] {name}")
            for rline in str(reason).splitlines():
                lines.append(f"         {rline}")
        return lines


def season_date_window(season: str):
    """
    Return a generous (start, end) window for a season label like "2016-17".

    NBA seasons run roughly October to April. 2019-20 extended into a summer
    restart, and 2020-21 started late in December, so the window is wide on
    purpose. This check is only meant to catch a season label attached to the
    wrong year, not to police the schedule.
    """
    start_year = int(season.split("-")[0])
    start = pd.Timestamp(year=start_year, month=8, day=1)
    end = pd.Timestamp(year=start_year + 1, month=10, day=31)
    return start, end


def run_audit(df: pd.DataFrame) -> Auditor:
    a = Auditor()

    # 1. Season counts
    counts = df.groupby("SEASON").size()
    mismatches = []
    for season in config.SEASONS:
        actual = int(counts.get(season, 0))
        expected = config.EXPECTED_GAME_COUNTS[season]
        if actual != expected:
            mismatches.append(f"{season}: actual {actual}, expected {expected}")
    a.check(
        f"Season game counts match expectations (total {config.EXPECTED_TOTAL_GAMES})",
        not mismatches and len(df) == config.EXPECTED_TOTAL_GAMES,
        "\n".join(mismatches) if mismatches else f"total rows = {len(df)}",
    )

    # 2. Unique game IDs
    dupes = df.loc[df["GAME_ID"].duplicated(keep=False)]
    a.check(
        "GAME_ID values are unique",
        dupes.empty,
        dupes[["SEASON", "GAME_DATE", "GAME_ID", "MATCHUP"]].to_string(index=False)
        if not dupes.empty else "",
    )

    # 3. Game ID format. NBA regular season game IDs are 10 characters and
    #    begin with 002. A 001 prefix would mean preseason, 004 playoffs.
    ids = df["GAME_ID"].astype(str)
    bad_format = df.loc[~(ids.str.len().eq(10) & ids.str.startswith("002"))]
    a.check(
        "GAME_ID format is regular season (10 chars, 002 prefix)",
        bad_format.empty,
        bad_format[["SEASON", "GAME_ID", "MATCHUP"]].to_string(index=False)
        if not bad_format.empty else "",
    )

    # 4. Opponent abbreviations
    opp = df["OPPONENT_ABBREV"].astype(str)
    bad_opp = df.loc[~opp.str.fullmatch(r"[A-Z]{3}") | opp.eq(config.CELTICS_ABBREV)]
    a.check(
        "Opponent abbreviations are valid and never BOS",
        bad_opp.empty,
        bad_opp[["SEASON", "GAME_ID", "MATCHUP"]].to_string(index=False)
        if not bad_opp.empty else "",
    )

    # 5. Distinct opponents
    n_opp = df["OPPONENT_ABBREV"].nunique()
    a.check(
        f"Distinct opponents equals {EXPECTED_DISTINCT_OPPONENTS}",
        n_opp == EXPECTED_DISTINCT_OPPONENTS,
        f"found {n_opp}: {sorted(df['OPPONENT_ABBREV'].unique())}",
    )

    # 6. Home/away balance per season
    imbalances = []
    for season, grp in df.groupby("SEASON"):
        home = int(grp["IS_HOME"].sum())
        away = int((~grp["IS_HOME"]).sum())
        # A full 82-game season is exactly 41/41. Shortened seasons may be
        # uneven, so allow a small tolerance there and report the number.
        tolerance = 0 if config.EXPECTED_GAME_COUNTS[season] == 82 else 4
        if abs(home - away) > tolerance:
            imbalances.append(
                f"{season}: {home} home, {away} away (tolerance {tolerance})"
            )
    a.check(
        "Home and away games are balanced within each season",
        not imbalances,
        "\n".join(imbalances) if imbalances else "",
    )

    # 7. Dates inside a plausible season window
    out_of_window = []
    for season, grp in df.groupby("SEASON"):
        start, end = season_date_window(season)
        bad = grp.loc[(grp["GAME_DATE"] < start) | (grp["GAME_DATE"] > end)]
        for _, row in bad.iterrows():
            out_of_window.append(
                f"{season}: {row['GAME_DATE'].date()} {row['MATCHUP']}"
            )
    a.check(
        "Game dates fall inside their season's plausible window",
        not out_of_window,
        "\n".join(out_of_window) if out_of_window else "",
    )

    # 8. No two games on the same date
    same_day = df.loc[df["GAME_DATE"].duplicated(keep=False)]
    a.check(
        "Boston never plays two games on the same calendar date",
        same_day.empty,
        same_day[["SEASON", "GAME_DATE", "MATCHUP"]].to_string(index=False)
        if not same_day.empty else "",
    )

    # 9. Plausible point totals
    bad_pts = df.loc[
        (df["PTS"] < MIN_PLAUSIBLE_PTS) | (df["PTS"] > MAX_PLAUSIBLE_PTS)
        | df["PTS"].isna()
    ]
    a.check(
        f"Boston point totals are between {MIN_PLAUSIBLE_PTS} and {MAX_PLAUSIBLE_PTS}",
        bad_pts.empty,
        bad_pts[["SEASON", "GAME_DATE", "MATCHUP", "PTS"]].to_string(index=False)
        if not bad_pts.empty else
        f"range {int(df['PTS'].min())} to {int(df['PTS'].max())}",
    )

    # --- Checks 10 and 11 concern PLUS_MINUS. ---
    #
    # History worth keeping in the code, because it explains the shape of these
    # checks. The original version of check 11 tested only that the SIGN of
    # PLUS_MINUS matched the recorded win or loss. It flagged one game. Looking
    # at the file revealed five games with a FRACTIONAL PLUS_MINUS, which is
    # impossible for a final margin. Four of them slipped through purely because
    # their sign happened to be right.
    #
    # A diagnostic (src/diagnose_plusminus.py) then verified all five against
    # BoxScoreSummaryV2, with ten clean games as controls. Result: WL and PTS
    # are correct everywhere, and team PLUS_MINUS is the sum of player-level
    # plus/minus divided by five, which does not reconcile on those games.
    #
    # So: check the value is a possible margin at all, not just its sign, and
    # excuse only the games in the evidence-carrying registry.
    excused = known_anomalies.excused_game_ids("PLUS_MINUS")
    pm = pd.to_numeric(df["PLUS_MINUS"], errors="coerce")
    is_excused = df["GAME_ID"].isin(excused)

    # 10. PLUS_MINUS must be a whole number. A margin is the difference of two
    #     integer scores.
    non_integer = df.loc[
        (pm.isna() | ((pm - pm.round()).abs() > 1e-9)) & ~is_excused
    ]
    a.check(
        "PLUS_MINUS is a whole number (documented anomalies excused)",
        non_integer.empty,
        non_integer[["SEASON", "GAME_DATE", "MATCHUP", "PLUS_MINUS"]]
        .to_string(index=False) if not non_integer.empty else
        f"{int(is_excused.sum())} game(s) excused via the anomaly registry",
    )

    # 11. Win/loss agrees with the sign of PLUS_MINUS. NBA games cannot end tied.
    inconsistent = df.loc[
        (pm.isna()
         | ((df["CELTICS_WON"] == 1) & (pm <= 0))
         | ((df["CELTICS_WON"] == 0) & (pm >= 0)))
        & ~is_excused
    ]
    a.check(
        "Win/loss agrees with the sign of PLUS_MINUS (documented anomalies excused)",
        inconsistent.empty,
        inconsistent[["SEASON", "GAME_DATE", "MATCHUP", "WL", "PLUS_MINUS"]]
        .to_string(index=False) if not inconsistent.empty else "",
    )

    # Deferred, and recorded here so it cannot be quietly forgotten. Validating
    # the target variable against a summary column is weaker than validating it
    # against the actual game data. Once the boxscores are downloaded, WL gets
    # checked against real final scores for all 636 games, not a 15 game sample.
    a.defer(
        "WL verified against final scores for all games",
        "Phase 2. Requires the boxscore pull. A 15 game sample has been "
        "verified so far (5 suspects plus 10 controls), all correct.",
    )
    a.defer(
        "Player plus/minus sums to 5 x margin for all games",
        "Phase 2. Tests whether the substitution logs are internally "
        "consistent, which lineup reconstruction in Phase 3 depends on. "
        "The five registry games are known to fail this; the question is "
        "how many others do.",
    )

    return a


def summarise(df: pd.DataFrame):
    lines = ["", "DESCRIPTIVE SUMMARY"]
    lines.append(f"  Games: {len(df)}")
    lines.append(f"  Date range: {df['GAME_DATE'].min().date()} to "
                 f"{df['GAME_DATE'].max().date()}")
    lines.append(f"  Overall record: {int(df['CELTICS_WON'].sum())}-"
                 f"{int((1 - df['CELTICS_WON']).sum())} "
                 f"({df['CELTICS_WON'].mean():.3f})")
    lines.append("")
    lines.append("  Per season record (this is the target variable's base rate,")
    lines.append("  and it is what any model has to beat):")
    for season, grp in df.groupby("SEASON"):
        w = int(grp["CELTICS_WON"].sum())
        l = len(grp) - w
        home_w = int(grp.loc[grp["IS_HOME"], "CELTICS_WON"].sum())
        home_n = int(grp["IS_HOME"].sum())
        away_w = int(grp.loc[~grp["IS_HOME"], "CELTICS_WON"].sum())
        away_n = int((~grp["IS_HOME"]).sum())
        lines.append(
            f"    {season}  {w:>2}-{l:<2} ({w / len(grp):.3f})   "
            f"home {home_w}/{home_n}   away {away_w}/{away_n}"
        )
    return lines


def main():
    config.ensure_dirs()

    if not config.GAME_INDEX_CSV.exists():
        raise FileNotFoundError(
            f"{config.GAME_INDEX_CSV} not found. Run the game index pull first."
        )

    df = pd.read_csv(config.GAME_INDEX_CSV, parse_dates=["GAME_DATE"])
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df["IS_HOME"] = df["IS_HOME"].astype(bool)

    auditor = run_audit(df)

    header = [
        "=" * 70,
        "PHASE 1 VALIDATION - CELTICS GAME INDEX",
        f"Run at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Source: {config.GAME_INDEX_CSV}",
        "=" * 70,
        "",
        "CHECKS",
    ]

    body = auditor.render()
    registry = [""] + known_anomalies.render_registry()
    tail = summarise(df)

    n_fail = len(auditor.failed)
    verdict = [
        "",
        "=" * 70,
        f"RESULT: {len(auditor.results) - n_fail} passed, {n_fail} failed, "
        f"{len(auditor.deferred)} deferred",
    ]
    if n_fail == 0:
        verdict.append("Game index is validated. Ready for the raw data pull.")
        if auditor.deferred:
            verdict.append("")
            verdict.append("Note: the deferred checks above are not optional.")
            verdict.append("They require data that does not exist yet and must")
            verdict.append("run in Phase 2 before any model is trained.")
    else:
        verdict.append("Game index is NOT validated. Do not proceed.")
        verdict.append("Failed checks: " + ", ".join(r[0] for r in auditor.failed))
    verdict.append("=" * 70)

    report = "\n".join(header + body + registry + tail + verdict)
    print(report)

    out = config.REPORTS_DIR / "phase1_game_index_validation.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to: {out}")

    return n_fail == 0


if __name__ == "__main__":
    main()
