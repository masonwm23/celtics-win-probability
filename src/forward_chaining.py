"""
Phase 8a: the deployment-analogue split.

WHY THIS EXISTS
---------------
Every result so far uses leave-one-season-out, which lets a fold train on the
2023-24 season to predict 2016-17. That is legitimate for measuring how much
team-specific structure exists in the data, and it is NOT how a live model would
ever be used: on 12 November 2018 you do not have 2023 games.

The paper currently states that as a limitation. Running it converts a statement
into a measurement, which is the whole difference between a caveat and a result.

`splits.forward_chaining` trains only on EARLIER seasons, expanding the window
each time. With eight seasons and a three-season minimum it produces five folds:

    train 2016-17..2018-19          test 2019-20
    train 2016-17..2019-20          test 2020-21
    train 2016-17..2020-21          test 2021-22
    train 2016-17..2021-22          test 2022-23
    train 2016-17..2022-23          test 2023-24

THE COMPARISON HAS TO BE LIKE FOR LIKE
--------------------------------------
Leave-one-season-out is scored on all eight seasons, forward chaining on five.
Comparing the published 0.1630 against a five-season number would be comparing
different test sets, not different split designs. So leave-one-season-out is
re-run here and BOTH are scored on exactly the same five held-out seasons.

A CONFOUND THAT MUST BE STATED, NOT HIDDEN
------------------------------------------
Forward chaining differs from leave-one-season-out in two ways at once: it never
sees the future, and it has less training data (three to seven seasons instead of
a constant seven). A gap between them cannot be attributed to directionality
alone. The per-fold table below reports the training-season count for exactly
this reason, and the earliest fold, with three training seasons, is where the
data-volume effect should be largest if it is driving anything.

Outputs
-------
reports/phase8_forward_chaining.txt
data/processed/forward_chaining_predictions.parquet
"""

import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import (config, evaluate, features, lineup_strength, models,
                 opponent_strength, splits)

logger = logging.getLogger(__name__)

# The specifications compared under both split designs. Deliberately small: the
# question is about the SPLIT, so only models that matter to the paper's claims
# are run. Adding more would multiply runtime without answering anything.
SPLIT_SPECS = ["tier2_generic", "tier3_celtics"]

LINEAR_OPPONENT = {
    "key": "linear_opponent",
    "name": "Linear baseline plus opponent differential",
    "features": list(models.GENERIC_FEATURES) + ["opponent_point_diff_prior"],
    "factory": models.make_generic_model,
    "transform": models.add_generic_interaction,
}


def load_frame():
    frame = pd.read_parquet(config.MODEL_FRAME_PARQUET).reset_index(drop=True)
    path = config.INTERIM_DIR / "opponent_strength.csv"
    has_opponent = path.exists()
    if has_opponent:
        strength = pd.read_csv(path, dtype={"GAME_ID": str})
        strength["GAME_ID"] = strength["GAME_ID"].str.zfill(10)
        frame = opponent_strength.attach_opponent_strength(frame, strength)
        frame = frame.reset_index(drop=True)
    return frame, has_opponent


def specs_to_run(has_opponent):
    out = [models.TIER_BY_KEY[key] for key in SPLIT_SPECS]
    if has_opponent:
        out.append(LINEAR_OPPONENT)
    for spec in out:
        spec.setdefault("transform", None)
    return out


def run_split(frame, splitter, specs, label):
    """
    Fit every spec on every fold of one split design.

    Returns (predictions keyed by spec, fold records). Rows outside any fold's
    test set stay NaN and are excluded when scoring.
    """
    target = frame[features.TARGET_COLUMN].astype(int)
    predictions = {s["key"]: pd.Series(np.nan, index=frame.index, dtype=float)
                   for s in specs}
    records = []

    for season, train_index, test_index in splitter(frame):
        started = time.time()
        for spec in specs:
            probabilities, _model = models.fit_predict(
                spec, frame.loc[train_index], target.loc[train_index],
                frame.loc[test_index])
            predictions[spec["key"]].loc[test_index] = probabilities
        record = splits.describe_split(frame, season, train_index, test_index)
        record["split"] = label
        record["seconds"] = round(time.time() - started, 1)
        records.append(record)
        logger.info("%s fold %s: %d train seasons, %d train games, %.1fs",
                    label, season, record["train_seasons"],
                    record["train_games"], record["seconds"])
    return predictions, records


def common_test_seasons(records_a, records_b):
    a = {r["held_out_season"] for r in records_a}
    b = {r["held_out_season"] for r in records_b}
    return sorted(a & b)


def score_on(frame, target, predictions, mask):
    """Score one spec's predictions restricted to a subset of rows."""
    values = predictions.to_numpy()
    usable = mask & ~np.isnan(values)
    if usable.sum() == 0:
        return None
    return evaluate.score_all(target.to_numpy()[usable], values[usable])


def build_report(frame, target, loso_predictions, fc_predictions,
                 loso_records, fc_records, seasons, mask, bootstraps, specs):
    lines = [
        "=" * 78,
        "PHASE 8a - FORWARD CHAINING VERSUS LEAVE-ONE-SEASON-OUT",
        f"Run at: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        "=" * 78,
        "",
        "  Leave-one-season-out lets a fold train on 2023-24 to predict 2016-17.",
        "  That measures how much team-specific structure exists in the data. It",
        "  is not how a live model would be used, because on any given night you",
        "  only have the past.",
        "",
        "  Forward chaining trains only on EARLIER seasons, expanding the window.",
        "  Both designs are scored here on exactly the same held-out seasons, so",
        "  the comparison is of split designs and not of different test sets.",
        "",
        f"  Common test seasons: {', '.join(seasons)}",
        f"  Rows scored: {int(mask.sum()):,} of {len(frame):,}",
        "",
        "=" * 78,
        "FOLD STRUCTURE",
        "=" * 78,
        "",
        f"  {'split':<22}{'held out':<10}{'train seasons':>15}"
        f"{'train games':>13}{'test games':>12}",
    ]
    for record in loso_records + fc_records:
        if record["held_out_season"] not in seasons:
            continue
        lines.append(f"  {record['split']:<22}{record['held_out_season']:<10}"
                     f"{record['train_seasons']:>15}{record['train_games']:>13,}"
                     f"{record['test_games']:>12,}")

    lines += [
        "",
        "  THE CONFOUND, STATED PLAINLY. Forward chaining differs in two ways at",
        "  once: it never sees the future AND it trains on less data, three to",
        "  seven seasons instead of a constant seven. Any gap below cannot be",
        "  attributed to directionality alone. The earliest fold is where a",
        "  data-volume effect should bite hardest if it is driving anything.",
        "",
        "=" * 78,
        "PERFORMANCE UNDER BOTH SPLIT DESIGNS",
        "=" * 78,
        "",
        f"  {'model':<46}{'split':<22}{'AUC':>8}{'Brier':>9}{'skill':>8}"
        f"{'ECE':>8}",
    ]
    for spec in specs:
        for label, predictions in (("leave-one-season-out", loso_predictions),
                                   ("forward chaining", fc_predictions)):
            row = score_on(frame, target, predictions[spec["key"]], mask)
            if row is None:
                continue
            lines.append(f"  {spec['name'][:45]:<46}{label:<22}{row['auc']:>8.4f}"
                         f"{row['brier']:>9.4f}{row['brier_skill']:>7.1%}"
                         f"{row['ece']:>8.4f}")
        lines.append("")

    lines += [
        "=" * 78,
        "IS THE DIFFERENCE REAL? CLUSTER BOOTSTRAP ON GAMES",
        "=" * 78,
        "",
        "  Positive difference means FORWARD CHAINING is better.",
        "",
        f"  {'model':<52}{'diff':>9}{'95% CI':>22}{'real?':>7}",
    ]
    for label, result in bootstraps.items():
        ci = f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]"
        lines.append(f"  {label:<52}{result['observed_difference']:>+9.4f}"
                     f"{ci:>22}{'yes' if result['excludes_zero'] else 'no':>7}")

    lines += ["", verdict(bootstraps), "", "=" * 78]
    return "\n".join(lines)


def verdict(bootstraps):
    """Read the comparison without editorialising past what was measured."""
    if not bootstraps:
        return "  No comparison available."
    decisive = [k for k, v in bootstraps.items() if v["excludes_zero"]]
    worse = [k for k, v in bootstraps.items()
             if v["excludes_zero"] and v["observed_difference"] < 0]
    if not decisive:
        return ("  NO MEASURABLE PENALTY. On the seasons both designs cover, "
                "training\n  only on the past is not distinguishable from "
                "training on all other\n  seasons. The published leave-one-season-out "
                "numbers are therefore a\n  reasonable guide to deployment "
                "performance, which is worth stating\n  because it could easily "
                "have gone the other way.")
    if worse:
        return ("  FORWARD CHAINING IS MEASURABLY WORSE for: "
                + "; ".join(worse)
                + "\n  The published leave-one-season-out numbers are therefore "
                  "optimistic\n  relative to deployment, and the paper must say "
                  "so with this number\n  attached. Remember the confound: less "
                  "training data, not only\n  direction of time.")
    return ("  FORWARD CHAINING IS MEASURABLY BETTER for: "
            + "; ".join(decisive)
            + "\n  Unexpected, and worth investigating before it is reported.")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    config.ensure_dirs()
    started = time.time()

    frame, has_opponent = load_frame()
    target = frame[features.TARGET_COLUMN].astype(int)
    specs = specs_to_run(has_opponent)
    print(f"Loaded {len(frame):,} events, {frame.game_id.nunique()} games")
    print(f"Specifications: {', '.join(s['key'] for s in specs)}")
    if not has_opponent:
        print("  opponent strength not built; running without the linear "
              "opponent spec")

    print("\nLeave-one-season-out...")
    loso_predictions, loso_records = run_split(
        frame, splits.leave_one_season_out, specs, "leave-one-season-out")

    print("Forward chaining (expanding window, past only)...")
    fc_predictions, fc_records = run_split(
        frame, splits.forward_chaining, specs, "forward chaining")

    seasons = common_test_seasons(loso_records, fc_records)
    if not seasons:
        raise RuntimeError("the two split designs share no test season")
    mask = frame["season"].isin(seasons).to_numpy()

    print(f"\nScoring both on the {len(seasons)} shared seasons...")
    bootstraps = {}
    game_ids = frame["game_id"].to_numpy()[mask]
    y = target.to_numpy()[mask]
    for spec in specs:
        a = loso_predictions[spec["key"]].to_numpy()[mask]
        b = fc_predictions[spec["key"]].to_numpy()[mask]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        bootstraps[spec["name"][:51]] = evaluate.bootstrap_brier_difference(
            game_ids, y, a, b, n_boot=2000, seed=config.RANDOM_SEED)

    out_frame = pd.DataFrame({"game_id": frame["game_id"],
                              "event_index": frame["event_index"],
                              "season": frame["season"],
                              features.TARGET_COLUMN: target})
    for spec in specs:
        out_frame[f"loso_{spec['key']}"] = loso_predictions[spec["key"]]
        out_frame[f"fc_{spec['key']}"] = fc_predictions[spec["key"]]
    out_frame.to_parquet(
        config.PROCESSED_DIR / "forward_chaining_predictions.parquet",
        index=False)

    report = build_report(frame, target, loso_predictions, fc_predictions,
                          loso_records, fc_records, seasons, mask, bootstraps,
                          specs)
    print(report)
    out = config.REPORTS_DIR / "phase8_forward_chaining.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to: {out}")
    print(f"Total time {(time.time() - started) / 60:.1f} minutes")
    return report


if __name__ == "__main__":
    main()
