"""
Phase 4: train and evaluate the four tiers, out of fold.

The whole comparison runs inside leave-one-season-out folds. For each fold:

  1. Player values are computed from that fold's TRAINING seasons only.
  2. Lineup strength is attached to both the training and the test rows using
     those training-derived values. The test rows get lineup strength computed
     from seasons that exclude their own, which is the point.
  3. Every tier is fitted on the same training rows and predicts the same test
     rows, so the comparison is like-for-like.

Step 1 is inside the loop deliberately. Computing player values once, outside,
would be faster and would leak: a held-out season's games would have helped build
a feature used to predict them.

Outputs
-------
data/processed/oof_predictions.parquet   out-of-fold probability per tier per row
reports/phase4_results.txt               the comparison, calibration and phases
models/                                  final model, feature schema, metadata
"""

import json
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import (config, evaluate, features, lineup_strength, models,
                 opponent_strength, splits)

logger = logging.getLogger(__name__)


def load_inputs():
    frame = pd.read_parquet(config.MODEL_FRAME_PARQUET).reset_index(drop=True)
    rosters = pd.read_parquet(config.ROSTERS_PARQUET)
    lineups = pd.read_parquet(config.LINEUPS_PARQUET)

    # Opponent strength, if it has been built. Attached ONCE, outside the fold
    # loop, and that is correct rather than lazy: an as-of-date record is a
    # function of one point in time and the games before it, so it does not
    # depend on which seasons the model trains on. Lineup strength is different
    # and must stay inside the loop.
    path = config.INTERIM_DIR / "opponent_strength.csv"
    if path.exists():
        strength = pd.read_csv(path, dtype={"GAME_ID": str})
        strength["GAME_ID"] = strength["GAME_ID"].str.zfill(10)
        frame = opponent_strength.attach_opponent_strength(frame, strength)
        missing = frame[opponent_strength.OPPONENT_FEATURE_COLUMNS].isna().sum()
        if int(missing.sum()):
            logger.warning("opponent features have nulls: %s",
                           missing[missing > 0].to_dict())
        frame = frame.reset_index(drop=True)
    return frame, rosters, lineups


def run_opponent_tiers(frame):
    """
    Fit the opponent tiers out of fold.

    No per-fold feature rebuild is needed here, for the reason given in
    load_inputs. The fold structure is still leave-one-season-out, so the model
    never trains on the season it is graded on.
    """
    available = [t for t in models.OPPONENT_TIERS
                 if all(c in frame.columns for c in t["features"])]
    if not available:
        logger.info("opponent features not present; skipping opponent tiers")
        return {}

    target = frame[features.TARGET_COLUMN].astype(int)
    predictions = {t["key"]: pd.Series(np.nan, index=frame.index, dtype=float)
                   for t in available}
    for season, train_index, test_index in splits.leave_one_season_out(frame):
        for tier in available:
            probabilities, _model = models.fit_predict(
                tier, frame.loc[train_index], target.loc[train_index],
                frame.loc[test_index])
            predictions[tier["key"]].loc[test_index] = probabilities
        logger.info("opponent tiers done for fold %s", season)
    return predictions


def run_folds(frame, rosters, lineups, exclude_games=None):
    """
    Fit every tier in every fold and collect out-of-fold predictions.

    Returns (predictions, fold_records, fitted_on_last_fold).
    """
    target = frame[features.TARGET_COLUMN].astype(int)
    predictions = {tier["key"]: pd.Series(np.nan, index=frame.index, dtype=float)
                   for tier in models.TIERS}
    fold_records = []
    last_models = {}

    for season, train_index, test_index in splits.leave_one_season_out(frame):
        started = time.time()
        allowed = splits.fold_seasons(frame, train_index)

        # Player values from TRAINING seasons only. Inside the loop on purpose.
        values = lineup_strength.compute_player_values(
            rosters, allowed, exclude_games=exclude_games)

        train_frame = lineup_strength.attach_lineup_strength(
            frame.loc[train_index], lineups, values)
        test_frame = lineup_strength.attach_lineup_strength(
            frame.loc[test_index], lineups, values)
        train_frame.index = train_index
        test_frame.index = test_index

        for tier in models.TIERS:
            probabilities, model = models.fit_predict(
                tier, train_frame, target.loc[train_index], test_frame)
            predictions[tier["key"]].loc[test_index] = probabilities
            last_models[tier["key"]] = model

        fold_records.append({
            "held_out_season": season,
            "train_games": int(frame.loc[train_index, "game_id"].nunique()),
            "test_games": int(frame.loc[test_index, "game_id"].nunique()),
            "train_events": int(len(train_index)),
            "test_events": int(len(test_index)),
            "players_valued": int(len(values)),
            "seconds": round(time.time() - started, 1),
        })
        logger.info("fold %s done in %.1fs (%d players valued)",
                    season, fold_records[-1]["seconds"], len(values))

    return predictions, fold_records, last_models


def run_lineup_variants(frame, rosters, lineups):
    """
    Fit the three PRE-REGISTERED lineup variants, out of fold.

    Player values are recomputed per fold AND per shrinkage level, since variant
    B changes the shrinkage. Every variant is reported regardless of outcome.
    """
    target = frame[features.TARGET_COLUMN].astype(int)
    predictions = {v["key"]: pd.Series(np.nan, index=frame.index, dtype=float)
                   for v in models.LINEUP_VARIANTS}

    shrinkages = sorted({v["shrinkage_minutes"] for v in models.LINEUP_VARIANTS})

    for season, train_index, test_index in splits.leave_one_season_out(frame):
        allowed = splits.fold_seasons(frame, train_index)
        prepared = {}
        for shrinkage in shrinkages:
            values = lineup_strength.compute_player_values(
                rosters, allowed, shrinkage_minutes=shrinkage)
            train_frame = lineup_strength.attach_lineup_strength(
                frame.loc[train_index], lineups, values)
            test_frame = lineup_strength.attach_lineup_strength(
                frame.loc[test_index], lineups, values)
            train_frame.index, test_frame.index = train_index, test_index
            prepared[shrinkage] = (train_frame, test_frame)

        for variant in models.LINEUP_VARIANTS:
            train_frame, test_frame = prepared[variant["shrinkage_minutes"]]
            if variant["needs_interactions"]:
                train_frame = models.add_lineup_interactions(train_frame)
                test_frame = models.add_lineup_interactions(test_frame)
                train_frame.index, test_frame.index = train_index, test_index
            probabilities, _model = models.fit_predict(
                variant, train_frame, target.loc[train_index], test_frame)
            predictions[variant["key"]].loc[test_index] = probabilities
        logger.info("variants done for fold %s", season)

    return predictions


def fit_final_model(frame, rosters, lineups, tier_key):
    """
    Fit the deliverable model on ALL seasons.

    This is the artefact the API will serve. It is NOT the model any reported
    metric comes from: every number in the results table is out of fold. Fitting
    on everything is correct for deployment and wrong for evaluation, and keeping
    those two things separate is the point.
    """
    target = frame[features.TARGET_COLUMN].astype(int)
    all_seasons = sorted(frame["season"].unique())
    values = lineup_strength.compute_player_values(rosters, all_seasons)
    full = lineup_strength.attach_lineup_strength(frame, lineups, values)
    full.index = frame.index

    # The deliverable tier is chosen by measured out-of-fold performance, not
    # fixed in advance. An earlier version hardcoded the lineup tier and would
    # have shipped the WORST of the in-game models.
    tier = models.TIER_BY_KEY[tier_key]
    x = full[tier["features"]].astype(float).to_numpy()
    model = tier["factory"]()
    model.fit(x, target.to_numpy())
    return model, values, tier


def _jsonable(value):
    """
    Convert numpy scalars to plain Python types for JSON.

    numpy int64 and float64 are not JSON serialisable, and pandas hands them
    back from almost every aggregation, so metadata writing fails without this.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def save_final_model(model, values, tier, frame, results_row):
    """
    Save the model together with everything needed to use it correctly.

    The feature schema is saved WITH the model on purpose. Serving features in a
    different order than training silently produces garbage, and an API that
    reads its column order from a file cannot make that mistake.
    """
    import joblib

    config.ensure_dirs()
    model_path = config.MODELS_DIR / "win_probability_model.joblib"
    joblib.dump(model, model_path)

    values_path = config.MODELS_DIR / "player_values.csv"
    values.rename("player_value").to_frame().to_csv(values_path)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_file": model_path.name,
        "model_class": type(model).__name__,
        "tier": tier["key"],
        "tier_name": tier["name"],
        "feature_order": list(tier["features"]),
        "n_features": len(tier["features"]),
        "target": features.TARGET_COLUMN,
        "trained_on_seasons": sorted(frame["season"].unique().tolist()),
        "trained_on_games": int(frame["game_id"].nunique()),
        "trained_on_events": int(len(frame)),
        "random_seed": config.RANDOM_SEED,
        "shrinkage_minutes": lineup_strength.DEFAULT_SHRINKAGE_MINUTES,
        "player_values_file": values_path.name,
        "out_of_fold_performance": results_row,
        "note": ("Metrics in out_of_fold_performance come from "
                 "leave-one-season-out cross validation, NOT from this model. "
                 "This model is fitted on every season for deployment; using it "
                 "to score its own training data would be meaningless."),
    }
    metadata_path = config.MODELS_DIR / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")
    return model_path, metadata_path, values_path


def build_report(frame, predictions, fold_records, comparison, phase_tables,
                 calibration, sensitivity=None, bootstraps=None,
                 final_key="tier3_celtics"):
    target = frame[features.TARGET_COLUMN].astype(int)
    lines = [
        "=" * 78,
        "PHASE 4 RESULTS - MODEL COMPARISON, OUT OF FOLD",
        f"Run at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "=" * 78,
        "",
        "Every number below is out-of-fold: each prediction comes from a model",
        f"that never saw that season. Leave-one-season-out, "
        f"{frame.season.nunique()} folds.",
        "",
        f"  events {len(frame):,}   games {frame.game_id.nunique()}   "
        f"seasons {frame.season.nunique()}   base rate "
        f"{target.mean():.4f}",
        "",
        "FOLDS",
        f"  {'held out':<10}{'train g':>9}{'test g':>8}{'train ev':>11}"
        f"{'test ev':>10}{'players':>9}{'secs':>7}",
    ]
    for row in fold_records:
        lines.append(f"  {row['held_out_season']:<10}{row['train_games']:>9}"
                     f"{row['test_games']:>8}{row['train_events']:>11,}"
                     f"{row['test_events']:>10,}{row['players_valued']:>9}"
                     f"{row['seconds']:>7.1f}")

    lines += ["", "=" * 78, "TIER COMPARISON (pooled over all folds)", "=" * 78, ""]
    lines.append(f"  {'tier':<46}{'AUC':>8}{'Brier':>9}{'skill':>8}"
                 f"{'logloss':>10}{'ECE':>8}")
    for tier in models.TIERS:
        row = comparison.loc[comparison["tier"].eq(tier["key"])]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(f"  {tier['name']:<46}{r['auc']:>8.4f}{r['brier']:>9.4f}"
                     f"{r['brier_skill']:>7.1%}{r['logloss']:>10.4f}"
                     f"{r['ece']:>8.4f}")
    baseline = comparison.iloc[0]
    lines += [
        "",
        f"  Always predicting the base rate scores Brier "
        f"{baseline['baseline_brier']:.4f} and log loss "
        f"{baseline['baseline_logloss']:.4f}.",
        "  'skill' is the Brier improvement over that, which is the honest way",
        "  to read a Brier score.",
        "",
        "  Tier 2 is a GENERIC BASELINE, not ESPN's model. ESPN does not publish",
        "  theirs, so no like-for-like comparison to it is possible. Tier 2 is a",
        "  transparent reimplementation of the standard margin-and-time approach.",
    ]

    # What each step buys.
    def get(key, column):
        row = comparison.loc[comparison["tier"].eq(key)]
        return float(row.iloc[0][column]) if not row.empty else float("nan")

    lines += ["", "WHAT EACH STEP ADDS", "-" * 19]
    steps = [
        ("In-game information over a pregame prior",
         "tier1_pregame", "tier2_generic"),
        ("Celtics-specific model over the generic baseline",
         "tier2_generic", "tier3_celtics"),
        ("Lineup strength over game state alone",
         "tier3_celtics", "tier4_lineup"),
    ]
    for label, before, after in steps:
        b_brier, a_brier = get(before, "brier"), get(after, "brier")
        b_auc, a_auc = get(before, "auc"), get(after, "auc")
        change = (b_brier - a_brier) / b_brier if b_brier else float("nan")
        lines.append(f"  {label}")
        lines.append(f"    Brier {b_brier:.4f} -> {a_brier:.4f}  "
                     f"({change:+.1%})    AUC {b_auc:.4f} -> {a_auc:.4f}")

    if bootstraps:
        lines += ["", "=" * 78,
                  "ARE THESE DIFFERENCES REAL? CLUSTER BOOTSTRAP", "=" * 78, "",
                  "  Resamples GAMES, not events. Events inside a game share an",
                  "  outcome and are highly dependent, so resampling events would",
                  "  treat correlated rows as independent and give an",
                  "  interval far too narrow. The unit is the game, not the event.",
                  "",
                  "  Positive difference means the SECOND model is better.",
                  "",
                  f"  {'comparison':<52}{'diff':>9}{'95% CI':>20}{'real?':>8}"]
        for label, b in bootstraps.items():
            ci = f"[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]"
            verdict = "yes" if b["excludes_zero"] else "no"
            lines.append(f"  {label:<52}{b['observed_difference']:>+9.4f}"
                         f"{ci:>20}{verdict:>8}")
        lines += ["",
                  "  'real?' means the interval excludes zero. A 'no' says the",
                  "  two models are not distinguishable on this data, which is a",
                  "  result worth stating rather than hiding behind a point"
                  " estimate."]

    lines += ["", "=" * 78, "PRE-REGISTERED LINEUP VARIANTS", "=" * 78, "",
              "  The first lineup tier made the model worse. Three alternative",
              "  parameterisations were fixed IN ADVANCE and all are reported",
              "  here regardless of outcome. Searching until something works and",
              "  reporting only the winner manufactures false findings; this",
              "  does not.", "",
              f"  {'model':<52}{'AUC':>8}{'Brier':>9}{'skill':>8}{'ECE':>8}"]
    baseline_row = comparison.loc[comparison["tier"].eq("tier3_celtics")]
    if not baseline_row.empty:
        r = baseline_row.iloc[0]
        lines.append(f"  {'Tier 3 baseline (no lineup)':<52}{r['auc']:>8.4f}"
                     f"{r['brier']:>9.4f}{r['brier_skill']:>7.1%}"
                     f"{r['ece']:>8.4f}")
    for variant in models.LINEUP_VARIANTS:
        row = comparison.loc[comparison["tier"].eq(variant["key"])]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(f"  {variant['name']:<52}{r['auc']:>8.4f}"
                     f"{r['brier']:>9.4f}{r['brier_skill']:>7.1%}"
                     f"{r['ece']:>8.4f}")
    tier3 = baseline_row.iloc[0]["brier"] if not baseline_row.empty else None
    if tier3 is not None:
        better = [v["name"] for v in models.LINEUP_VARIANTS
                  if not comparison.loc[comparison["tier"].eq(v["key"])].empty
                  and float(comparison.loc[comparison["tier"].eq(v["key"])]
                            .iloc[0]["brier"]) < tier3]
        lines.append("")
        if better:
            lines.append(f"  Variants that improve on tier 3: {', '.join(better)}")
        else:
            lines.append("  NO variant improves on tier 3. The negative result on")
            lines.append("  lineup strength stands, across all three "
                         "pre-registered")
            lines.append("  parameterisations, and is reported as a finding.")

    opponent_rows = [t for t in models.OPPONENT_TIERS
                     if not comparison.loc[comparison["tier"].eq(t["key"])].empty]
    if opponent_rows:
        lines += ["", "=" * 78, "OPPONENT CONTEXT (PRE-REGISTERED)", "=" * 78, "",
                  "  Every opponent measure is computed AS OF THE GAME DATE from",
                  "  prior games only. A full-season record includes games played",
                  "  after the one being predicted, which would be a leak.",
                  "",
                  "  Variants fixed in advance, all reported.", "",
                  f"  {'model':<58}{'AUC':>8}{'Brier':>9}{'skill':>8}{'ECE':>8}"]
        base_row = comparison.loc[comparison["tier"].eq("tier3_celtics")]
        if not base_row.empty:
            r = base_row.iloc[0]
            lines.append(f"  {'Tier 3 baseline (no opponent context)':<58}"
                         f"{r['auc']:>8.4f}{r['brier']:>9.4f}"
                         f"{r['brier_skill']:>7.1%}{r['ece']:>8.4f}")
        for tier in opponent_rows:
            r = comparison.loc[comparison["tier"].eq(tier["key"])].iloc[0]
            lines.append(f"  {tier['name']:<58}{r['auc']:>8.4f}"
                         f"{r['brier']:>9.4f}{r['brier_skill']:>7.1%}"
                         f"{r['ece']:>8.4f}")
        if not base_row.empty:
            base_brier = float(base_row.iloc[0]["brier"])
            better = [t["name"] for t in opponent_rows
                      if float(comparison.loc[comparison["tier"].eq(t["key"])]
                               .iloc[0]["brier"]) < base_brier]
            lines.append("")
            if better:
                lines.append("  Improves on tier 3: " + "; ".join(better))
                lines.append("  Check the bootstrap above before calling it real.")
            else:
                lines.append("  NO opponent formulation here improves on tier 3.")
                lines.append("")
                lines.append("  DO NOT READ THAT AS A FINDING ABOUT OPPONENTS.")
                lines.append("  Phases 6 and 7 established that these tiers are")
                lines.append("  CONTAMINATED. Every measure above is constant")
                lines.append("  within a game and takes a distinct value in nearly")
                lines.append("  every game, so the tree can isolate one training")
                lines.append("  game into a pure leaf and memorise it: tier 5's")
                lines.append("  training Brier is 0.0076 against a 0.2290 baseline.")
                lines.append("  A column of RANDOM numbers at the same resolution")
                lines.append("  does the same damage, and the damage scales with")
                lines.append("  the number of distinct values for both.")
                lines.append("")
                lines.append("  The usable measurement is in reports/")
                lines.append("  phase7_clean_tests.txt: at about 5 distinct values")
                lines.append("  opponent context is neutral to slightly positive,")
                lines.append("  and in the linear model it is a small improvement.")
                lines.append("  Run scripts/16_run_clean_tests.py.")

    lines += ["", "=" * 78, "PERFORMANCE BY GAME PHASE", "=" * 78, "",
              "  A pooled number is dominated by late-game events, where any model",
              "  looks good. The early and clutch rows are the informative ones.", ""]
    for tier in models.TIERS + models.LINEUP_VARIANTS + models.OPPONENT_TIERS:
        table = phase_tables.get(tier["key"])
        if table is None or table.empty:
            continue
        lines.append(f"  {tier['name']}")
        lines.append(f"    {'phase':<14}{'events':>9}{'base':>8}{'AUC':>9}"
                     f"{'Brier':>9}{'skill':>8}{'ECE':>8}")
        for _, r in table.iterrows():
            lines.append(f"    {r['phase']:<14}{int(r['n']):>9,}"
                         f"{r['base_rate']:>8.3f}{r['auc']:>9.4f}"
                         f"{r['brier']:>9.4f}{r['brier_skill']:>7.1%}"
                         f"{r['ece']:>8.4f}")
        lines.append("")

    lines += ["=" * 78, f"CALIBRATION OF THE SELECTED MODEL ({final_key})",
              "=" * 78, "",
              "  Does a stated probability mean what it says? For a dashboard this",
              "  matters more than ranking: a number that reads 70 percent while",
              "  meaning 90 is worse than useless to someone watching.", "",
              f"    {'bin':<12}{'events':>10}{'share':>9}{'predicted':>12}"
              f"{'observed':>11}{'gap':>9}"]
    for _, r in calibration.iterrows():
        lines.append(f"    {r['bin']:<12}{int(r['n']):>10,}{r['share']:>8.1%}"
                     f"{r['mean_predicted']:>12.4f}{r['observed']:>11.4f}"
                     f"{r['gap']:>+9.4f}")

    if sensitivity is not None:
        lines += ["", "=" * 78, "SENSITIVITY: EXCLUDING THE FLAGGED GAMES",
                  "=" * 78, "",
                  "  Twenty games carry data-quality flags from Phases 1 and 2: ten",
                  "  whose player plus/minus does not reconcile, and ten with",
                  "  residual lineup uncertainty. Player values were recomputed",
                  "  without them and the lineup tier refitted.", ""]
        for key, row in sensitivity.items():
            lines.append(f"    {key:<24} Brier {row['brier']:.4f}  "
                         f"AUC {row['auc']:.4f}  skill {row['brier_skill']:.1%}")

    lines += ["", "=" * 78]
    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    config.ensure_dirs()
    started = time.time()

    frame, rosters, lineups = load_inputs()
    target = frame[features.TARGET_COLUMN].astype(int)
    print(f"Loaded {len(frame):,} events, {frame.game_id.nunique()} games")

    print(f"\nFitting 4 tiers across {frame.season.nunique()} "
          "leave-one-season-out folds...")
    predictions, fold_records, _ = run_folds(frame, rosters, lineups)

    out_of_fold = pd.DataFrame({k: v for k, v in predictions.items()})
    out_of_fold.insert(0, "game_id", frame["game_id"])
    out_of_fold.insert(1, "event_index", frame["event_index"])
    out_of_fold.insert(2, "season", frame["season"])
    out_of_fold.insert(3, features.TARGET_COLUMN, target)
    out_of_fold.to_parquet(config.PROCESSED_DIR / "oof_predictions.parquet",
                           index=False)

    print("\nFitting the 3 pre-registered lineup variants...")
    variant_predictions = run_lineup_variants(frame, rosters, lineups)
    predictions.update(variant_predictions)

    print("\nFitting the opponent tiers...")
    opponent_predictions = run_opponent_tiers(frame)
    predictions.update(opponent_predictions)

    comparison = evaluate.compare_tiers(
        {k: v.to_numpy() for k, v in predictions.items()}, target.to_numpy())
    phase_tables = {key: evaluate.phase_table(frame, target, series.to_numpy())
                    for key, series in predictions.items()}

    # Choose the deliverable by measured out-of-fold Brier, among the in-game
    # models. Hardcoding it is how the worst model gets shipped.
    candidates = comparison.loc[comparison["tier"].ne("tier1_pregame")]
    final_key = str(candidates.loc[candidates["brier"].idxmin(), "tier"])
    print(f"\nBest in-game model by out-of-fold Brier: {final_key}")

    calibration = evaluate.calibration_table(
        target.to_numpy(), predictions[final_key].to_numpy())

    # Cluster bootstrap on the differences that matter, resampling GAMES.
    print("Bootstrapping tier differences (resampling games)...")
    bootstraps = {}
    pairs = [("tier2_generic", "tier3_celtics"),
             ("tier3_celtics", "tier5_opponent"),
             ("tier3_celtics", "variant_d_opp_point_diff"),
             ("tier3_celtics", "variant_e_opp_recent_form"),
             ("tier3_celtics", "variant_f_strength_diff"),
             ("tier3_celtics", "tier4_lineup"),
             ("tier3_celtics", "variant_a_diff_only"),
             ("tier3_celtics", "variant_b_heavy_shrinkage"),
             ("tier3_celtics", "variant_c_time_interaction")]
    for a_key, b_key in pairs:
        if a_key not in predictions or b_key not in predictions:
            continue
        bootstraps[f"{a_key} vs {b_key}"] = evaluate.bootstrap_brier_difference(
            frame["game_id"].to_numpy(), target.to_numpy(),
            predictions[a_key].to_numpy(), predictions[b_key].to_numpy(),
            n_boot=2000, seed=config.RANDOM_SEED)

    # Sensitivity: drop the flagged games from player-value estimation.
    flagged = set()
    if config.LINEUP_RISK_CSV.exists():
        flagged |= set(pd.read_csv(config.LINEUP_RISK_CSV,
                                   dtype={"game_id": str})["game_id"])
    if config.LINEUP_ANOMALIES_CSV.exists():
        anomalies = pd.read_csv(config.LINEUP_ANOMALIES_CSV,
                                dtype={"game_id": str})
        if len(anomalies):
            flagged |= set(anomalies["game_id"])
    sensitivity = None
    if flagged:
        print(f"\nSensitivity check: recomputing player values without "
              f"{len(flagged)} flagged game(s)...")
        sens_predictions, _records, _m = run_folds(
            frame, rosters, lineups, exclude_games=flagged)
        sens_comparison = evaluate.compare_tiers(
            {k: v.to_numpy() for k, v in sens_predictions.items()},
            target.to_numpy())
        sensitivity = {}
        for key in ("tier3_celtics", "tier4_lineup"):
            base = comparison.loc[comparison["tier"].eq(key)].iloc[0]
            sens = sens_comparison.loc[sens_comparison["tier"].eq(key)].iloc[0]
            sensitivity[f"{key} (all games)"] = base
            sensitivity[f"{key} (flagged excluded)"] = sens

    report = build_report(frame, predictions, fold_records, comparison,
                          phase_tables, calibration, sensitivity,
                          bootstraps=bootstraps, final_key=final_key)
    print(report)
    out = config.REPORTS_DIR / "phase4_results.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to: {out}")

    print("\nFitting the deliverable model on all seasons...")
    model, values, tier = fit_final_model(frame, rosters, lineups, final_key)
    results_row = comparison.loc[comparison["tier"].eq(final_key)].iloc[0]
    model_path, metadata_path, values_path = save_final_model(
        model, values, tier, frame, dict(results_row))
    print(f"  model    -> {model_path}")
    print(f"  metadata -> {metadata_path}")
    print(f"  values   -> {values_path}")
    print(f"\nTotal time {(time.time() - started) / 60:.1f} minutes")
    return comparison


if __name__ == "__main__":
    main()
