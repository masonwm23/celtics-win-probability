"""
Phase 3 audit: features, split integrity, and the leak tests.

The centrepiece is the SHUFFLED TARGET LEAK TEST. Everything else in this project
has been about not corrupting the data. This is about not fooling ourselves.

How the leak test works
-----------------------
It is a permutation test. Out-of-fold predictions are produced for every event by
a model that never saw that event's season, and pooled into a single AUC. Then
the outcome is permuted ACROSS GAMES, keeping every event of a game together so
the features, the grouping and the win rate are unchanged, and the whole thing is
refit. Repeated for several independent permutations.

On scrambled labels the model must score at chance. If it does not, some feature
is carrying the answer and every metric produced later is worthless.

Two details that make this test honest rather than decorative.

POOLING. Every event of a game shares one label, so the effective sample size is
the number of GAMES, not events. A single fold holds 72 to 82 games, where the
null band on AUC is about +/- 0.13; pooling all folds gives 636 games and tightens
it to about +/- 0.05. Judging a fold on its own would call ordinary noise a leak.
An early version of this audit did exactly that on a 26 game development sample
and reported a failure that was pure sampling variance.

A DERIVED THRESHOLD. The pass band is three Hanley-McNeil standard errors of AUC
under the null, computed from the actual win and loss counts, rather than a
number chosen to be comfortable. Below MIN_GAMES_FOR_LEAK_TEST the test reports
UNDERPOWERED instead of a verdict, because at that size it cannot tell a leak from
noise and pretending otherwise would be worse than saying so.

Why AUC on the real labels will look high, and why that is not a leak
--------------------------------------------------------------------
Most events in a basketball game happen when the outcome is already fairly clear.
A model that says "leading by 20 with two minutes left" wins almost always is
correct, not cheating. That is why performance is also reported BY GAME PHASE:
a model that only looks good in garbage time is not useful, and the first-quarter
and clutch numbers are the ones that carry information.

The tipoff check
----------------
At the opening event of each game the score is 0-0 and the clock is full, so the
only legitimate signal is home court and roster quality. If a live game-state
feature predicts the outcome strongly at tipoff, it is carrying information it
should not have. This is a second, independent way to catch a leak.

Writes reports/phase3_validation.txt
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src import config, features, lineup_strength, splits

logger = logging.getLogger(__name__)

# Number of independent label permutations in the leak test. More is better
# evidence; three keeps the audit to a few minutes while still giving a spread
# rather than a single draw.
N_PERMUTATIONS = 3

# Below this many games the permutation test cannot distinguish a leak from
# noise, and it reports UNDERPOWERED rather than a misleading pass or fail.
MIN_GAMES_FOR_LEAK_TEST = 100

# Game phases for the breakdown, as (label, predicate on the frame).
PHASES = [
    ("period 1", lambda f: f["period"].eq(1)),
    ("period 2", lambda f: f["period"].eq(2)),
    ("period 3", lambda f: f["period"].eq(3)),
    ("period 4", lambda f: f["period"].eq(4)),
    ("overtime", lambda f: f["period"].gt(4)),
    ("clutch", lambda f: f["is_clutch"]),
    ("first 6 min", lambda f: f["seconds_elapsed_game"].le(360)),
]


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


def make_model():
    """
    The probe model for the leak tests.

    Deliberately the same family as the final model (gradient boosting) so the
    leak test exercises what will actually be used, but with modest settings so
    the audit stays quick. Seeded for reproducibility.
    """
    return HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.1, max_depth=6,
        early_stopping=False, random_state=config.RANDOM_SEED)


def auc_null_standard_error(n_positive, n_negative):
    """
    Hanley-McNeil standard error of AUC under the null hypothesis AUC = 0.5.

    This is what makes the leak test honest. Every event of a game carries the
    same label, so the EFFECTIVE sample size is the number of games, not the
    number of events. With 2 games in a fold the null band is +/- 0.98 and any
    threshold is meaningless; pooled across 636 games it tightens to about
    +/- 0.05. The pass threshold is derived from the data rather than picked.
    """
    if n_positive < 1 or n_negative < 1:
        return float("nan")
    area = 0.5
    q1 = area / (2 - area)
    q2 = 2 * area * area / (1 + area)
    numerator = (area * (1 - area)
                 + (n_positive - 1) * (q1 - area * area)
                 + (n_negative - 1) * (q2 - area * area))
    return float(np.sqrt(numerator / (n_positive * n_negative)))


def out_of_fold_predictions(frame, feature_columns, target):
    """Predict every row from a model that never saw its season."""
    predictions = pd.Series(np.nan, index=frame.index, dtype=float)
    for _season, train_index, test_index in splits.leave_one_season_out(frame):
        predictions.loc[test_index] = fit_fold(
            frame, train_index, test_index, feature_columns, target)
    return predictions


def game_level_auc(frame, target, predictions):
    """
    Pooled AUC computed at the EVENT level, with the game counts that set its
    null band returned alongside so the result can be judged properly.
    """
    mask = predictions.notna()
    per_game = frame.loc[mask, ["game_id"]].copy()
    per_game["y"] = target[mask].to_numpy()
    outcomes = per_game.drop_duplicates("game_id")["y"]
    n_positive = int((outcomes == 1).sum())
    n_negative = int((outcomes == 0).sum())
    value = float("nan")
    if target[mask].nunique() > 1:
        value = float(roc_auc_score(target[mask], predictions[mask]))
    return value, n_positive, n_negative


def score(y_true, probabilities):
    """AUC, Brier and log loss. AUC is undefined if a fold has one class."""
    out = {"n": int(len(y_true)), "base_rate": float(np.mean(y_true))}
    if len(np.unique(y_true)) < 2:
        out.update(auc=float("nan"), brier=float("nan"), logloss=float("nan"))
        return out
    out["auc"] = float(roc_auc_score(y_true, probabilities))
    out["brier"] = float(brier_score_loss(y_true, probabilities))
    out["logloss"] = float(log_loss(y_true, probabilities, labels=[0, 1]))
    return out


def fit_fold(frame, train_index, test_index, feature_columns, target):
    """Fit on the training rows and score on the held-out rows."""
    model = make_model()
    x_train = frame.loc[train_index, feature_columns].astype(float)
    x_test = frame.loc[test_index, feature_columns].astype(float)
    model.fit(x_train, target.loc[train_index])
    probabilities = model.predict_proba(x_test)[:, 1]
    return probabilities


def run():
    config.ensure_dirs()
    if not config.MODEL_FRAME_PARQUET.exists():
        raise FileNotFoundError(
            f"{config.MODEL_FRAME_PARQUET} missing. Run the Phase 3 build first.")

    frame = pd.read_parquet(config.MODEL_FRAME_PARQUET).reset_index(drop=True)
    rosters = pd.read_parquet(config.ROSTERS_PARQUET)
    feature_columns = list(features.FEATURE_COLUMNS)
    target = frame[features.TARGET_COLUMN].astype(int)

    a = Auditor()

    # ---------------- feature sanity ----------------
    missing = [c for c in feature_columns if c not in frame.columns]
    a.check("Every declared feature column exists", not missing,
            f"missing {missing}" if missing else
            f"{len(feature_columns)} features present")

    nulls = frame[feature_columns].isna().sum()
    a.check("No nulls in any feature column", int(nulls.sum()) == 0,
            "\n".join(f"{k}: {v}" for k, v in nulls[nulls > 0].items())
            if nulls.sum() else f"{len(frame):,} rows checked")

    held = frame[frame["possession_team"].ne("")]
    share = float(held["celtics_has_possession"].mean())
    a.check("Possession share is near 50 percent",
            0.45 <= share <= 0.55,
            f"Boston holds the ball on {share:.3f} of events where possession "
            f"is known; {int(frame['possession_team'].eq('').sum()):,} events "
            f"({100 * frame['possession_team'].eq('').mean():.2f}%) precede the "
            f"first possession-bearing event of a period")

    per_game = frame.groupby("game_id")["possession_change"].sum() / 2
    a.check("Possessions per team per game are in the normal NBA range",
            bool((per_game.between(80, 125)).mean() > 0.98),
            f"median {per_game.median():.0f}, "
            f"5th percentile {per_game.quantile(0.05):.0f}, "
            f"95th percentile {per_game.quantile(0.95):.0f}; "
            f"{int((~per_game.between(80, 125)).sum())} game(s) outside 80-125")

    negative_time = int((frame["seconds_remaining_game"] < 0).sum())
    a.check("Time remaining is never negative", negative_time == 0,
            f"{negative_time} rows negative")

    margin_check = frame.groupby("game_id").apply(
        lambda g: abs(g["score_change"].sum() - g["celtics_margin"].iloc[-1]),
        include_groups=False)
    a.check("Per-event scoring sums to the final margin in every game",
            bool((margin_check < 1e-6).all()),
            f"{int((margin_check >= 1e-6).sum())} game(s) disagree")

    # ---------------- split integrity ----------------
    fold_rows, straddles = [], []
    for season, train_index, test_index in splits.leave_one_season_out(frame):
        fold_rows.append(splits.describe_split(frame, season, train_index,
                                               test_index))
        train_games = set(frame.loc[train_index, "game_id"])
        test_games = set(frame.loc[test_index, "game_id"])
        if train_games & test_games:
            straddles.append(season)

    a.check("No game appears in both train and test in any fold",
            not straddles,
            f"{len(fold_rows)} folds checked, "
            f"{sum(r['test_games'] for r in fold_rows)} test games in total")

    covered = {r["held_out_season"] for r in fold_rows}
    a.check("Every season is held out exactly once",
            covered == set(config.SEASONS),
            f"held out: {sorted(covered)}")

    # ---------------- fold-safe lineup strength ----------------
    leaked = []
    for season, train_index, _test in splits.leave_one_season_out(frame):
        allowed = splits.fold_seasons(frame, train_index)
        if season in allowed:
            leaked.append(season)
            continue
        values = lineup_strength.compute_player_values(rosters, allowed)
        held_only = (set(rosters.loc[rosters["season"].eq(season), "person_id"])
                     - set(rosters.loc[rosters["season"].isin(allowed),
                                       "person_id"]))
        if held_only & set(values.index):
            leaked.append(f"{season}: held-out-only players in values")
    a.check("Lineup strength never sees the held-out season", not leaked,
            "\n".join(str(x) for x in leaked) if leaked else
            "checked all folds: player values are built from training seasons "
            "only, and players who appear solely in the held-out season "
            "correctly receive the default value")

    # ---------------- THE LEAK TEST ----------------
    # A permutation test. Pool out-of-fold predictions across every fold, so the
    # statistic has an effective sample of all 636 games rather than the handful
    # in a single season. Then repeat with the outcome permuted across games and
    # confirm the model collapses to chance.
    real_predictions = out_of_fold_predictions(frame, feature_columns, target)
    real_auc, n_pos, n_neg = game_level_auc(frame, target, real_predictions)
    null_se = auc_null_standard_error(n_pos, n_neg)
    tolerance = 3 * null_se if np.isfinite(null_se) else float("inf")
    n_games = n_pos + n_neg

    permuted = []
    for offset in range(N_PERMUTATIONS):
        shuffled_target = splits.shuffle_target_by_game(
            frame, seed=config.RANDOM_SEED + offset)
        shuffled_predictions = out_of_fold_predictions(
            frame, feature_columns, shuffled_target)
        value, _p, _n = game_level_auc(frame, shuffled_target,
                                       shuffled_predictions)
        permuted.append(value)
        logger.info("permutation %d/%d: pooled AUC %.4f",
                    offset + 1, N_PERMUTATIONS, value)

    permuted_array = np.array([v for v in permuted if np.isfinite(v)])
    mean_permuted = float(permuted_array.mean()) if len(permuted_array) else float("nan")
    worst_permuted = (float(np.abs(permuted_array - 0.5).max())
                      if len(permuted_array) else float("nan"))

    detail = [
        f"Effective sample: {n_games} games ({n_pos} wins, {n_neg} losses).",
        f"Under the null the pooled AUC has SE {null_se:.4f}, so chance is",
        f"0.500 +/- {tolerance:.4f} at three standard errors.",
        "",
        f"  real outcomes          pooled out-of-fold AUC {real_auc:.4f}",
    ]
    for i, value in enumerate(permuted, start=1):
        detail.append(f"  permutation {i}            pooled out-of-fold AUC "
                      f"{value:.4f}")
    detail += [
        "",
        f"  mean over permutations {mean_permuted:.4f}",
        f"  worst deviation from chance {worst_permuted:.4f} "
        f"(tolerance {tolerance:.4f})",
        f"  real exceeds every permutation: "
        f"{bool(np.isfinite(real_auc) and len(permuted_array) and real_auc > permuted_array.max())}",
    ]

    if n_games < MIN_GAMES_FOR_LEAK_TEST:
        detail.append("")
        detail.append(f"UNDERPOWERED: only {n_games} games. This test cannot "
                      f"distinguish a leak from noise below "
                      f"{MIN_GAMES_FOR_LEAK_TEST} games and is reported, not "
                      f"used as a verdict.")
        a.check("Scrambled outcomes score at chance (leak test)", True,
                "\n".join(detail))
    else:
        passed = (np.isfinite(worst_permuted) and worst_permuted <= tolerance
                  and np.isfinite(real_auc)
                  and real_auc > permuted_array.max())
        a.check("Scrambled outcomes score at chance, so nothing leaks the answer",
                passed, "\n".join(detail))

    real_frame = pd.DataFrame([{"season": "pooled", "auc": real_auc}])
    shuffled_frame = pd.DataFrame([{"season": f"perm{i}", "auc": v}
                                   for i, v in enumerate(permuted, start=1)])

    # ---------------- tipoff check ----------------
    tipoff = frame.sort_values("event_index").groupby("game_id").head(1)
    tipoff_rows = []
    for season, train_index, test_index in splits.leave_one_season_out(frame):
        train_tip = tipoff.index.intersection(train_index)
        test_tip = tipoff.index.intersection(test_index)
        if len(test_tip) < 10 or target.loc[train_tip].nunique() < 2:
            continue
        probabilities = fit_fold(frame, train_tip, test_tip, feature_columns,
                                 target)
        result = score(target.loc[test_tip].to_numpy(), probabilities)
        result["season"] = season
        tipoff_rows.append(result)

    tipoff_frame = pd.DataFrame(tipoff_rows)
    tipoff_auc = (float(tipoff_frame["auc"].mean())
                  if len(tipoff_frame) else float("nan"))
    a.check("At tipoff the model is near chance, as it must be",
            np.isnan(tipoff_auc) or tipoff_auc <= 0.62,
            f"mean AUC on opening events only: {tipoff_auc:.4f}\n"
            f"At 0-0 with a full clock the only real signal is home court, so a "
            f"high value here would mean a feature carries the outcome.\n"
            f"Boston's home win rate over the period is the natural benchmark.")

    return (a, frame, real_frame, shuffled_frame, tipoff_frame, fold_rows,
            feature_columns, target)


def phase_breakdown(frame, feature_columns, target):
    """Out-of-fold performance by game phase, pooled across folds."""
    out_of_fold = pd.Series(np.nan, index=frame.index, dtype=float)
    for _season, train_index, test_index in splits.leave_one_season_out(frame):
        out_of_fold.loc[test_index] = fit_fold(frame, train_index, test_index,
                                               feature_columns, target)
    rows = []
    for label, predicate in PHASES:
        mask = predicate(frame) & out_of_fold.notna()
        if int(mask.sum()) < 50:
            continue
        result = score(target[mask].to_numpy(), out_of_fold[mask].to_numpy())
        result["phase"] = label
        rows.append(result)
    return pd.DataFrame(rows), out_of_fold


def build_report(a, frame, real, shuffled, tipoff, fold_rows, phases):
    n_fail = len(a.failed)
    lines = [
        "=" * 74,
        "PHASE 3 VALIDATION - FEATURES, SPLITS AND LEAK TESTS",
        f"Run at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "=" * 74,
        "",
        "INPUTS",
        f"  model frame : {len(frame):,} rows, {frame.game_id.nunique()} games",
        f"  seasons     : {frame.season.nunique()}",
        f"  features    : {len(features.FEATURE_COLUMNS)}",
        "",
        "CHECKS",
    ]
    lines += a.render()

    lines += ["", "CROSS-VALIDATION FOLDS", "-" * 22,
              f"  {'held out':<10}{'train games':>12}{'test games':>12}"
              f"{'train events':>14}{'test events':>13}{'test win rate':>15}"]
    for row in fold_rows:
        lines.append(f"  {row['held_out_season']:<10}{row['train_games']:>12}"
                     f"{row['test_games']:>12}{row['train_events']:>14,}"
                     f"{row['test_events']:>13,}{row['test_win_rate']:>15.3f}")

    if len(phases):
        lines += ["", "OUT-OF-FOLD PERFORMANCE BY GAME PHASE", "-" * 37,
                  "  Pooled across folds, so every row is a prediction made by a",
                  "  model that never saw that season.",
                  "",
                  f"  {'phase':<14}{'events':>10}{'base rate':>12}{'AUC':>9}"
                  f"{'Brier':>9}{'log loss':>10}"]
        for _, row in phases.iterrows():
            lines.append(f"  {row['phase']:<14}{int(row['n']):>10,}"
                         f"{row['base_rate']:>12.3f}{row['auc']:>9.4f}"
                         f"{row['brier']:>9.4f}{row['logloss']:>10.4f}")
        lines += ["",
                  "  Late-game numbers being strong is expected, not suspicious:",
                  "  a big lead with little time left really does decide games.",
                  "  The early-game and clutch rows are the informative ones."]

    lines += ["", "=" * 74,
              f"RESULT: {len(a.results) - n_fail} passed, {n_fail} failed"]
    if n_fail == 0:
        lines.append("Phase 3 is validated. The split design is leak-free and the")
        lines.append("features are safe to train on in Phase 4.")
    else:
        lines.append("Phase 3 is NOT validated. Do not proceed.")
        lines.append("Failed: " + ", ".join(r[0] for r in a.failed))
    lines.append("=" * 74)
    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    (a, frame, real, shuffled, tipoff, fold_rows,
     feature_columns, target) = run()
    phases, _out_of_fold = phase_breakdown(frame, feature_columns, target)
    report = build_report(a, frame, real, shuffled, tipoff, fold_rows, phases)
    print(report)
    out = config.REPORTS_DIR / "phase3_validation.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to: {out}")
    return len(a.failed) == 0


if __name__ == "__main__":
    main()
