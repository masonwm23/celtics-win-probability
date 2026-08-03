"""
Phase 7 runner: the dose-response ladders, the clean lineup re-test, and the
deliverable decision.

Same 8 leave-one-season-out folds as everything else. Training Brier is recorded
alongside out-of-fold Brier again, because the gap between them is what exposed
the problem in the first place.

Outputs
-------
reports/phase7_clean_tests.txt
data/processed/clean_test_predictions.parquet
"""

import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import (clean_tests, config, evaluate, features, lineup_strength,
                 models, opponent_strength, splits)

logger = logging.getLogger(__name__)


def load_frame():
    frame = pd.read_parquet(config.MODEL_FRAME_PARQUET).reset_index(drop=True)
    rosters = pd.read_parquet(config.ROSTERS_PARQUET)
    lineups = pd.read_parquet(config.LINEUPS_PARQUET)

    path = config.INTERIM_DIR / "opponent_strength.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/14_build_opponent_strength.py.")
    strength = pd.read_csv(path, dtype={"GAME_ID": str})
    strength["GAME_ID"] = strength["GAME_ID"].str.zfill(10)
    frame = opponent_strength.attach_opponent_strength(frame, strength)
    frame = frame.reset_index(drop=True)

    frame = clean_tests.add_opponent_ladder(frame)
    frame = clean_tests.add_random_ladder(frame)
    return frame, rosters, lineups


def ladder_cardinality(frame):
    """Distinct values per game, which is the quantity under test."""
    per_game = frame.groupby("game_id").first()
    out = {}
    for name, _step, _label in clean_tests.OPPONENT_LADDER:
        out[name] = int(per_game[name].nunique())
    out[clean_tests.OPPONENT_SOURCE] = int(
        per_game[clean_tests.OPPONENT_SOURCE].nunique())
    for name, _k in clean_tests.RANDOM_LADDER:
        out[name] = int(per_game[name].nunique())
    out[clean_tests.RANDOM_RAW] = int(per_game[clean_tests.RANDOM_RAW].nunique())
    return out


def run_clean(frame, rosters, lineups):
    target = frame[features.TARGET_COLUMN].astype(int)
    predictions = {s["key"]: pd.Series(np.nan, index=frame.index, dtype=float)
                   for s in clean_tests.CLEAN_SPECS}
    train_sse = {s["key"]: 0.0 for s in clean_tests.CLEAN_SPECS}
    train_n = {s["key"]: 0 for s in clean_tests.CLEAN_SPECS}

    for season, train_index, test_index in splits.leave_one_season_out(frame):
        started = time.time()
        allowed = splits.fold_seasons(frame, train_index)

        values = lineup_strength.compute_player_values(rosters, allowed)
        train_frame = lineup_strength.attach_lineup_strength(
            frame.loc[train_index], lineups, values)
        test_frame = lineup_strength.attach_lineup_strength(
            frame.loc[test_index], lineups, values)
        train_frame.index, test_frame.index = train_index, test_index

        # Lineup bins from TRAINING rows only. The held-out season does not get
        # to influence where the cut points sit.
        train_bins, test_bins = clean_tests.bin_by_training_quantiles(
            train_frame[clean_tests.LINEUP_SOURCE],
            test_frame[clean_tests.LINEUP_SOURCE])
        train_frame[clean_tests.LINEUP_BINNED] = train_bins
        test_frame[clean_tests.LINEUP_BINNED] = test_bins

        y_train = target.loc[train_index].to_numpy()
        for spec in clean_tests.CLEAN_SPECS:
            probabilities, model = models.fit_predict(
                spec, train_frame, target.loc[train_index], test_frame)
            predictions[spec["key"]].loc[test_index] = probabilities

            x_train = train_frame[spec["features"]].astype(float).to_numpy()
            if spec["transform"] is not None:
                x_train = spec["transform"](x_train)
            in_sample = model.predict_proba(x_train)[:, 1]
            train_sse[spec["key"]] += float(((in_sample - y_train) ** 2).sum())
            train_n[spec["key"]] += len(y_train)

        logger.info("clean tests done for fold %s in %.1fs", season,
                    time.time() - started)

    train_scores = {k: train_sse[k] / train_n[k] for k in train_sse}
    return predictions, train_scores


def _wrap(text, width=74):
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


def build_report(frame, predictions, train_scores, comparison, feature_boots,
                 deliverable_boots, cardinality, phase_tables):
    by_key = {row["tier"]: row for _, row in comparison.iterrows()}
    tier3 = float(by_key["p7_tier3"]["brier"])
    tier2 = float(by_key["p7_tier2"]["brier"])

    lines = [
        "=" * 78,
        "PHASE 7 - CLEAN RE-TESTS AFTER THE PHASE 6 ARTEFACT",
        f"Run at: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        "=" * 78,
        "",
        "  Phase 6 showed that a feature which is constant within a game and",
        "  near-unique across games lets a tree memorise training games. Tier 5",
        "  scored 0.0076 on its own training data against a 0.2290 baseline.",
        "  A column of random numbers reproduced 85% of the damage.",
        "",
        "  Phase 6 also got something wrong: a min_child_weight of 4,000 did",
        "  NOT neutralise the mechanism, so its 'fair' opponent and lineup",
        "  tests were not fair. Phase 7 replaces them.",
        "",
        "=" * 78,
        "DOSE-RESPONSE: DAMAGE AGAINST FEATURE RESOLUTION",
        "=" * 78,
        "",
        "  The mechanism claim predicts the SHAPE of this table in advance:",
        "  damage should grow with the number of distinct values a feature can",
        "  take, and a meaningless column should track a real one rung for rung.",
        "  Both ladders are nested, so only resolution changes.",
        "",
        f"  {'resolution':<16}{'opp values':>11}{'opp Brier':>11}{'vs t3':>9}"
        f"{'rand values':>13}{'rand Brier':>12}{'vs t3':>9}",
    ]
    opponent_gaps, random_gaps = [], []
    for label, opp_key, rand_key in clean_tests.LADDER_PAIRS:
        opp_column = (clean_tests.OPPONENT_SOURCE if opp_key == "p7_opp_raw"
                      else opp_key.replace("p7_", ""))
        rand_column = (clean_tests.RANDOM_RAW if rand_key == "p7_rand_raw"
                       else rand_key.replace("p7_", ""))
        opp_brier = float(by_key[opp_key]["brier"])
        rand_brier = float(by_key[rand_key]["brier"])
        opponent_gaps.append(opp_brier - tier3)
        random_gaps.append(rand_brier - tier3)
        lines.append(
            f"  {label:<16}{cardinality[opp_column]:>11,}{opp_brier:>11.4f}"
            f"{opp_brier - tier3:>+9.4f}{cardinality[rand_column]:>13,}"
            f"{rand_brier:>12.4f}{rand_brier - tier3:>+9.4f}")

    lines += ["", f"  Tier 3, no added column: Brier {tier3:.4f}", ""]
    for line in _wrap(clean_tests.dose_response_verdict(opponent_gaps,
                                                        random_gaps)):
        lines.append(f"  {line}")

    lines += [
        "",
        "=" * 78,
        "TRAINING VERSUS OUT-OF-FOLD BRIER",
        "=" * 78,
        "",
        f"  {'spec':<58}{'train':>8}{'oof':>8}{'gap':>8}",
    ]
    for spec in clean_tests.CLEAN_SPECS:
        key = spec["key"]
        oof = float(by_key[key]["brier"])
        lines.append(f"  {spec['name']:<58}{train_scores[key]:>8.4f}"
                     f"{oof:>8.4f}{oof - train_scores[key]:>8.4f}")

    lines += [
        "",
        "=" * 78,
        "ALL SPECIFICATIONS, OUT OF FOLD",
        "=" * 78,
        "",
        f"  {'spec':<58}{'AUC':>8}{'Brier':>9}{'skill':>8}{'ECE':>8}",
    ]
    for spec in clean_tests.CLEAN_SPECS:
        row = by_key[spec["key"]]
        lines.append(f"  {spec['name']:<58}{row['auc']:>8.4f}"
                     f"{row['brier']:>9.4f}{row['brier_skill']:>7.1%}"
                     f"{row['ece']:>8.4f}")

    lines += [
        "",
        "=" * 78,
        "FEATURE COMPARISONS (both sides share a model configuration)",
        "=" * 78,
        "",
        "  Positive difference means the SECOND model is better.",
        "",
        f"  {'comparison':<56}{'diff':>9}{'95% CI':>22}{'real?':>7}",
    ]
    for label, result in feature_boots.items():
        ci = f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]"
        lines.append(f"  {label:<56}{result['observed_difference']:>+9.4f}"
                     f"{ci:>22}{'yes' if result['excludes_zero'] else 'no':>7}")

    # The clean lineup verdict.
    lineup_tree = feature_boots.get("p7_tier3 vs p7_lineup_bins5")
    lineup_linear = feature_boots.get("p7_tier2 vs p7_linear_lineup")
    lines += [
        "",
        "=" * 78,
        "CLEAN LINEUP RE-TEST",
        "=" * 78,
        "",
        "  Phase 4 reported that lineup strength genuinely degrades the model",
        "  and blamed player values failing to transfer across seasons. Phase 6",
        "  showed most of that penalty was reproducible with noise. These two",
        "  tests block memorisation instead of assuming it away.",
        "",
        f"  in the tree, 5 bins from training quantiles: "
        f"{lineup_tree['observed_difference']:+.4f} "
        f"[{lineup_tree['ci_low']:+.4f}, {lineup_tree['ci_high']:+.4f}]",
        f"  in the linear model, no splits available:   "
        f"{lineup_linear['observed_difference']:+.4f} "
        f"[{lineup_linear['ci_low']:+.4f}, {lineup_linear['ci_high']:+.4f}]",
        "",
    ]
    if not lineup_tree["excludes_zero"] and not lineup_linear["excludes_zero"]:
        verdict = ("Neither clean test finds a real effect. Phase 4's claim "
                   "that lineup strength hurts is NOT supported once "
                   "memorisation is blocked, and the seasons-do-not-transfer "
                   "explanation must be withdrawn.")
    elif (lineup_linear["excludes_zero"]
          and lineup_linear["observed_difference"] > 0):
        verdict = ("Lineup strength HELPS in the clean linear test. Phase 4's "
                   "negative result was an artefact and must be reversed.")
    else:
        verdict = ("At least one clean test still finds a real effect; see the "
                   "signs above before rewriting Phase 4.")
    for line in _wrap(verdict):
        lines.append(f"  {line}")

    lines += [
        "",
        "=" * 78,
        "THE DELIVERABLE DECISION",
        "=" * 78,
        "",
        "  Rule fixed in code BEFORE this run: the deliverable changes only if",
        "  the interval excludes zero in the challenger's favour. The same",
        "  standard that stopped tier 4 shipping, applied symmetrically.",
        "",
        f"  incumbent  tier 3, gradient boosting on 13 features   Brier {tier3:.4f}",
        f"  reference  tier 2, generic linear baseline            Brier {tier2:.4f}",
        "",
        f"  {'comparison':<56}{'diff':>9}{'95% CI':>22}{'real?':>7}",
    ]
    for label, result in deliverable_boots.items():
        ci = f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]"
        lines.append(f"  {label:<56}{result['observed_difference']:>+9.4f}"
                     f"{ci:>22}{'yes' if result['excludes_zero'] else 'no':>7}")
    lines.append("")
    for challenger in ("p7_linear_opp", "p7_linear_strength"):
        result = deliverable_boots[f"p7_tier3 vs {challenger}"]
        for line in _wrap(clean_tests.deliverable_verdict(
                result, clean_tests.SPEC_BY_KEY[challenger]["name"], "tier 3")):
            lines.append(f"  {line}")
        lines.append("")

    # The null control has to be null, or the linear result means nothing.
    null = feature_boots["p7_tier2 vs p7_linear_random"]
    lines += [
        "=" * 78,
        "THE LINEAR NULL CONTROL",
        "=" * 78,
        "",
        "  Adding a RANDOM per-game value to the linear model. If this improved",
        "  the model, the opponent result above would be an artefact of a",
        "  different kind and could not be trusted.",
        "",
        f"  difference {null['observed_difference']:+.4f}  "
        f"CI [{null['ci_low']:+.4f}, {null['ci_high']:+.4f}]  "
        f"real? {'yes' if null['excludes_zero'] else 'no'}",
        "",
    ]
    if null["excludes_zero"]:
        lines.append("  CONTROL FAILED. A meaningless column moved the linear")
        lines.append("  model. Do not trust the linear opponent result.")
    else:
        lines.append("  Control passes. A meaningless column does nothing here,")
        lines.append("  so the linear opponent result is signal, not structure.")

    lines += ["", "=" * 78, "PERFORMANCE BY GAME PHASE", "=" * 78, ""]
    for spec in clean_tests.CLEAN_SPECS:
        key = spec["key"]
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


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    config.ensure_dirs()
    started = time.time()

    frame, rosters, lineups = load_frame()
    target = frame[features.TARGET_COLUMN].astype(int)
    cardinality = ladder_cardinality(frame)
    print(f"Loaded {len(frame):,} events, {frame.game_id.nunique()} games")
    print("Distinct values per game on each ladder rung:")
    for name, count in cardinality.items():
        print(f"    {name:<32}{count:>6,}")

    print(f"\nFitting {len(clean_tests.CLEAN_SPECS)} specs across 8 folds...")
    predictions, train_scores = run_clean(frame, rosters, lineups)

    comparison = evaluate.compare_tiers(
        {k: v.to_numpy() for k, v in predictions.items()}, target.to_numpy())
    phase_tables = {key: evaluate.phase_table(frame, target, series.to_numpy())
                    for key, series in predictions.items()}

    print("Bootstrapping (resampling games)...")

    def boot(a_key, b_key):
        return evaluate.bootstrap_brier_difference(
            frame["game_id"].to_numpy(), target.to_numpy(),
            predictions[a_key].to_numpy(), predictions[b_key].to_numpy(),
            n_boot=2000, seed=config.RANDOM_SEED)

    feature_boots = {f"{a} vs {b}": boot(a, b)
                     for a, b in clean_tests.FEATURE_COMPARISONS}
    deliverable_boots = {f"{a} vs {b}": boot(a, b)
                         for a, b in clean_tests.DELIVERABLE_COMPARISONS}

    out_frame = pd.DataFrame({k: v for k, v in predictions.items()})
    out_frame.insert(0, "game_id", frame["game_id"])
    out_frame.insert(1, "event_index", frame["event_index"])
    out_frame.insert(2, features.TARGET_COLUMN, target)
    out_frame.to_parquet(
        config.PROCESSED_DIR / "clean_test_predictions.parquet", index=False)

    # Train and out-of-fold scores as data, not only as report text, so the
    # figures can be rebuilt without re-parsing a formatted file.
    scores = comparison.copy()
    scores["train_brier"] = scores["tier"].map(train_scores)
    scores["name"] = scores["tier"].map(
        {s["key"]: s["name"] for s in clean_tests.CLEAN_SPECS})
    scores["cardinality"] = scores["tier"].map({
        **{opp: cardinality[(clean_tests.OPPONENT_SOURCE
                             if opp == "p7_opp_raw" else opp.replace("p7_", ""))]
           for _l, opp, _r in clean_tests.LADDER_PAIRS},
        **{rand: cardinality[(clean_tests.RANDOM_RAW
                              if rand == "p7_rand_raw" else rand.replace("p7_", ""))]
           for _l, _o, rand in clean_tests.LADDER_PAIRS},
    })
    scores.to_csv(config.REPORTS_DIR / "phase7_scores.csv", index=False)

    report = build_report(frame, predictions, train_scores, comparison,
                          feature_boots, deliverable_boots, cardinality,
                          phase_tables)
    print(report)
    out = config.REPORTS_DIR / "phase7_clean_tests.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to: {out}")
    print(f"Total time {(time.time() - started) / 60:.1f} minutes")
    return comparison


if __name__ == "__main__":
    main()
