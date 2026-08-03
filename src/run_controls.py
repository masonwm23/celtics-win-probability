"""
Phase 6 runner: execute the pre-registered controls and report every one.

Fold structure is unchanged, leave-one-season-out, so nothing here is graded on
a season it trained on. What is new is that TRAINING Brier is recorded alongside
out-of-fold Brier. That pair is the memorisation test: a model that has learned
something real improves on both, while a model that has memorised training games
improves sharply in-sample and gets worse out of sample.

Outputs
-------
reports/phase6_controls.txt
data/processed/control_predictions.parquet
"""

import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import (config, controls, evaluate, features, lineup_strength, models,
                 opponent_strength, splits)

logger = logging.getLogger(__name__)


def load_frame():
    """The model frame with opponent strength attached. Opponent is required."""
    frame = pd.read_parquet(config.MODEL_FRAME_PARQUET).reset_index(drop=True)
    rosters = pd.read_parquet(config.ROSTERS_PARQUET)
    lineups = pd.read_parquet(config.LINEUPS_PARQUET)

    path = config.INTERIM_DIR / "opponent_strength.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Phase 6 tests the opponent result, so opponent "
            "strength must exist. Run scripts/14_build_opponent_strength.py.")
    strength = pd.read_csv(path, dtype={"GAME_ID": str})
    strength["GAME_ID"] = strength["GAME_ID"].str.zfill(10)
    frame = opponent_strength.attach_opponent_strength(frame, strength)
    frame = frame.reset_index(drop=True)

    missing = int(frame[opponent_strength.OPPONENT_FEATURE_COLUMNS]
                  .isna().sum().sum())
    if missing:
        raise ValueError(f"{missing} null opponent feature values; refusing to "
                         "run controls on an incomplete frame")

    frame = controls.add_random_game_constants(frame)
    frame = controls.add_opponent_buckets(frame)
    return frame, rosters, lineups


def run_controls(frame, rosters, lineups):
    """
    Fit every control spec in every fold.

    Returns (out_of_fold_predictions, train_scores, fold_seconds).
    `train_scores` holds pooled in-sample Brier per spec, accumulated as a sum of
    squared errors so folds of different sizes combine correctly.
    """
    target = frame[features.TARGET_COLUMN].astype(int)
    predictions = {s["key"]: pd.Series(np.nan, index=frame.index, dtype=float)
                   for s in controls.CONTROL_SPECS}
    train_sse = {s["key"]: 0.0 for s in controls.CONTROL_SPECS}
    train_n = {s["key"]: 0 for s in controls.CONTROL_SPECS}
    fold_seconds = []

    for season, train_index, test_index in splits.leave_one_season_out(frame):
        started = time.time()
        allowed = splits.fold_seasons(frame, train_index)

        # Player values from TRAINING seasons only, same discipline as Phase 4.
        values = lineup_strength.compute_player_values(rosters, allowed)
        train_frame = lineup_strength.attach_lineup_strength(
            frame.loc[train_index], lineups, values)
        test_frame = lineup_strength.attach_lineup_strength(
            frame.loc[test_index], lineups, values)
        train_frame.index, test_frame.index = train_index, test_index

        y_train = target.loc[train_index].to_numpy()
        for spec in controls.CONTROL_SPECS:
            probabilities, model = models.fit_predict(
                spec, train_frame, target.loc[train_index], test_frame)
            predictions[spec["key"]].loc[test_index] = probabilities

            # In-sample predictions on the SAME fitted model.
            x_train = train_frame[spec["features"]].astype(float).to_numpy()
            if spec["transform"] is not None:
                x_train = spec["transform"](x_train)
            in_sample = model.predict_proba(x_train)[:, 1]
            train_sse[spec["key"]] += float(((in_sample - y_train) ** 2).sum())
            train_n[spec["key"]] += len(y_train)

        fold_seconds.append(round(time.time() - started, 1))
        logger.info("controls done for fold %s in %.1fs", season,
                    fold_seconds[-1])

    train_scores = {k: train_sse[k] / train_n[k] for k in train_sse}
    return predictions, train_scores, fold_seconds


def build_report(frame, predictions, train_scores, comparison, bootstraps,
                 phase_tables):
    target = frame[features.TARGET_COLUMN].astype(int)
    by_key = {row["tier"]: row for _, row in comparison.iterrows()}

    lines = [
        "=" * 78,
        "PHASE 6 - CONTROLS FOR THE GAME-CONSTANT MEMORISATION ARTEFACT",
        f"Run at: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        "=" * 78,
        "",
        "  Phase 5 reported that every opponent formulation made the model far",
        "  worse out of fold. A single pregame column moved Brier 0.1630 ->",
        "  0.1998. That is too much damage for one constant-per-game column,",
        "  so before writing it up as a finding it is tested as an artefact.",
        "",
        "  The accusation: opponent_point_diff_prior takes 608 distinct values",
        "  across 636 games, so it is functionally a game identifier. A game",
        "  averages 486 events that all share one label, and min_child_weight",
        "  is 20, so a leaf holding one whole game clears the floor 24x over.",
        "  The tree can isolate a training game into a PURE node and memorise",
        "  it, spending its capacity on that instead of on margin and clock.",
        "",
        "  Every prediction below was written down BEFORE the run.",
        "",
        "=" * 78,
        "TRAINING VERSUS OUT-OF-FOLD BRIER",
        "=" * 78,
        "",
        "  The memorisation signature. A model that learns something real",
        "  improves on BOTH. A model that memorises training games improves",
        "  in-sample and gets worse out of sample, so the gap widens.",
        "",
        f"  {'spec':<52}{'train':>9}{'out-of-fold':>13}{'gap':>9}",
    ]
    for spec in controls.CONTROL_SPECS:
        key = spec["key"]
        if key not in by_key:
            continue
        train = train_scores[key]
        oof = float(by_key[key]["brier"])
        lines.append(f"  {spec['name']:<52}{train:>9.4f}{oof:>13.4f}"
                     f"{oof - train:>9.4f}")

    lines += [
        "",
        "  A larger gap means more of the fit did not survive the season",
        "  boundary. Read it against the reference rows, not in isolation.",
        "",
        "=" * 78,
        "CONTROL RESULTS, OUT OF FOLD",
        "=" * 78,
        "",
        f"  {'spec':<52}{'AUC':>8}{'Brier':>9}{'skill':>8}{'ECE':>8}",
    ]
    for spec in controls.CONTROL_SPECS:
        key = spec["key"]
        if key not in by_key:
            continue
        row = by_key[key]
        lines.append(f"  {spec['name']:<52}{row['auc']:>8.4f}"
                     f"{row['brier']:>9.4f}{row['brier_skill']:>7.1%}"
                     f"{row['ece']:>8.4f}")

    lines += ["", "  Pre-registered prediction for each control:", ""]
    for spec in controls.CONTROL_SPECS:
        if spec["key"] in by_key:
            lines.append(f"    {spec['key']:<26} {spec['prediction']}")

    lines += [
        "",
        "=" * 78,
        "PAIRED COMPARISONS, CLUSTER BOOTSTRAP ON GAMES",
        "=" * 78,
        "",
        "  Positive difference means the SECOND model is better.",
        "  Each control is compared against ITS OWN baseline: the game-aware",
        "  specs against C5, not against tier 3, so the parameter change is",
        "  not mistaken for an opponent effect.",
        "",
        f"  {'comparison':<58}{'diff':>9}{'95% CI':>22}{'real?':>8}",
    ]
    for label, result in bootstraps.items():
        ci = f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]"
        real = "yes" if result["excludes_zero"] else "no"
        lines.append(f"  {label:<58}{result['observed_difference']:>+9.4f}"
                     f"{ci:>22}{real:>8}")

    # The decisive read.
    tier3 = float(by_key["ref_tier3"]["brier"])
    tier5 = float(by_key["ref_tier5"]["brier"])
    random_brier = float(by_key["c1_random_unique"]["brier"])
    bucket_brier = float(by_key["c2_random_bucket"]["brier"])
    lines += [
        "",
        "=" * 78,
        "THE DECISIVE CONTROL",
        "=" * 78,
        "",
        f"  tier 3, no added column                    Brier {tier3:.4f}",
        f"  tier 5, real opponent context              Brier {tier5:.4f}"
        f"   ({tier5 - tier3:+.4f})",
        f"  C1, a RANDOM near-unique game constant     Brier {random_brier:.4f}"
        f"   ({random_brier - tier3:+.4f})",
        f"  C2, a RANDOM 5-value game constant         Brier {bucket_brier:.4f}"
        f"   ({bucket_brier - tier3:+.4f})",
        "",
        "  C1 contains no information about anything. Any damage it does is",
        "  the mechanism alone.",
        "",
    ]
    for line in _wrap(controls.verdict(random_brier - tier3, tier5 - tier3), 74):
        lines.append(f"  {line}")

    lines += ["", "=" * 78, "PERFORMANCE BY GAME PHASE", "=" * 78, ""]
    for spec in controls.CONTROL_SPECS:
        key = spec["key"]
        if key not in phase_tables:
            continue
        lines.append(f"  {spec['name']}")
        lines.append(f"    {'phase':<14}{'events':>9}{'AUC':>9}{'Brier':>9}"
                     f"{'skill':>8}{'ECE':>8}")
        for _, row in phase_tables[key].iterrows():
            lines.append(f"    {row['phase']:<14}{int(row['n']):>9,}"
                         f"{row['auc']:>9.4f}{row['brier']:>9.4f}"
                         f"{row['brier_skill']:>7.1%}{row['ece']:>8.4f}")
        lines.append("")

    lines += ["=" * 78]
    return "\n".join(lines)


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    config.ensure_dirs()
    started = time.time()

    frame, rosters, lineups = load_frame()
    target = frame[features.TARGET_COLUMN].astype(int)
    print(f"Loaded {len(frame):,} events, {frame.game_id.nunique()} games")
    print(f"Distinct values of opponent_point_diff_prior across games: "
          f"{frame.groupby('game_id')['opponent_point_diff_prior'].first().nunique()}")
    print(f"Mean events per game: {len(frame) / frame.game_id.nunique():.0f}")
    print(f"min_child_weight, default {models.DEFAULT_MIN_CHILD_WEIGHT}, "
          f"game-aware {controls.GAME_AWARE_MIN_CHILD_WEIGHT}")

    print(f"\nFitting {len(controls.CONTROL_SPECS)} control specs across 8 "
          f"folds. This takes a while.")
    predictions, train_scores, _seconds = run_controls(frame, rosters, lineups)

    comparison = evaluate.compare_tiers(
        {k: v.to_numpy() for k, v in predictions.items()}, target.to_numpy())
    phase_tables = {key: evaluate.phase_table(frame, target, series.to_numpy())
                    for key, series in predictions.items()}

    print("Bootstrapping the paired comparisons (resampling games)...")
    bootstraps = {}
    for base_key, candidate_key in controls.CONTROL_COMPARISONS:
        bootstraps[f"{base_key} vs {candidate_key}"] = (
            evaluate.bootstrap_brier_difference(
                frame["game_id"].to_numpy(), target.to_numpy(),
                predictions[base_key].to_numpy(),
                predictions[candidate_key].to_numpy(),
                n_boot=2000, seed=config.RANDOM_SEED))

    frame_out = pd.DataFrame({k: v for k, v in predictions.items()})
    frame_out.insert(0, "game_id", frame["game_id"])
    frame_out.insert(1, "event_index", frame["event_index"])
    frame_out.insert(2, features.TARGET_COLUMN, target)
    frame_out.to_parquet(config.PROCESSED_DIR / "control_predictions.parquet",
                         index=False)

    report = build_report(frame, predictions, train_scores, comparison,
                          bootstraps, phase_tables)
    print(report)
    out = config.REPORTS_DIR / "phase6_controls.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to: {out}")
    print(f"Total time {(time.time() - started) / 60:.1f} minutes")
    return comparison


if __name__ == "__main__":
    main()
