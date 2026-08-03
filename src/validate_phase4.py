"""
Phase 4 audit: is the saved model correct, usable and reproducible?

This is checklist item 6. A model file that cannot be reloaded and reproduced is
not a research artefact, it is a one-off.

Checks
  1.  The model file, metadata and player values all exist
  2.  Metadata records the exact feature order, seed and training scope
  3.  The saved model reloads and predicts
  4.  Reloaded predictions match a freshly fitted model exactly, same seed
  5.  Feature order in metadata matches what the model was trained on
  6.  Scrambling the feature order changes predictions, proving order matters
  7.  Probabilities are in [0, 1] and not degenerate
  8.  Out-of-fold predictions exist for every row and every tier
  9.  The lineup tier beat, or did not beat, the game-state tier, reported as
      measured rather than assumed

Check 6 deserves a note. Serving features in a different order than training is
the single most common way a working model silently produces nonsense in
production. The metadata carries the order for exactly that reason, and this
check proves the order is load-bearing rather than decorative.

Writes reports/phase4_validation.txt
"""

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config, evaluate, features, lineup_strength, models

logger = logging.getLogger(__name__)


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


def run(sample_rows=20000):
    import joblib

    a = Auditor()
    model_path = config.MODELS_DIR / "win_probability_model.joblib"
    metadata_path = config.MODELS_DIR / "model_metadata.json"
    values_path = config.MODELS_DIR / "player_values.csv"
    oof_path = config.PROCESSED_DIR / "oof_predictions.parquet"

    present = [p.exists() for p in (model_path, metadata_path, values_path)]
    a.check("Model, metadata and player values are all saved", all(present),
            "\n".join(f"{p.name}: {'present' if e else 'MISSING'}"
                      for p, e in zip((model_path, metadata_path, values_path),
                                      present)))
    if not all(present):
        return a, None, None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = ["feature_order", "random_seed", "trained_on_seasons",
                "trained_on_games", "model_class", "target"]
    missing = [k for k in required if k not in metadata]
    a.check("Metadata records everything needed to reproduce the model",
            not missing,
            f"missing {missing}" if missing else
            f"{metadata['model_class']} on {metadata['n_features']} features, "
            f"seed {metadata['random_seed']}, "
            f"{metadata['trained_on_games']} games across "
            f"{len(metadata['trained_on_seasons'])} seasons")

    frame = pd.read_parquet(config.MODEL_FRAME_PARQUET).reset_index(drop=True)
    rosters = pd.read_parquet(config.ROSTERS_PARQUET)
    lineups = pd.read_parquet(config.LINEUPS_PARQUET)
    target = frame[features.TARGET_COLUMN].astype(int)

    values = lineup_strength.compute_player_values(
        rosters, sorted(frame["season"].unique()))
    full = lineup_strength.attach_lineup_strength(frame, lineups, values)
    full.index = frame.index

    feature_order = metadata["feature_order"]
    tier = models.TIER_BY_KEY[metadata["tier"]]
    a.check("Metadata feature order matches the tier definition",
            feature_order == list(tier["features"]),
            f"{len(feature_order)} features, first three "
            f"{feature_order[:3]}")

    sample = full.sample(n=min(sample_rows, len(full)),
                         random_state=config.RANDOM_SEED)
    x_sample = sample[feature_order].astype(float).to_numpy()

    loaded = joblib.load(model_path)
    loaded_probabilities = loaded.predict_proba(x_sample)[:, 1]
    a.check("The saved model reloads and predicts", True,
            f"predicted {len(loaded_probabilities):,} sampled rows")

    # Refit from scratch with the same seed and compare.
    refit = tier["factory"]()
    refit.fit(full[feature_order].astype(float).to_numpy(), target.to_numpy())
    refit_probabilities = refit.predict_proba(x_sample)[:, 1]
    max_difference = float(np.abs(loaded_probabilities
                                  - refit_probabilities).max())
    a.check("A fresh fit with the same seed reproduces the saved model exactly",
            max_difference < 1e-9,
            f"max absolute difference across {len(x_sample):,} rows: "
            f"{max_difference:.3e}")

    # Feature order must matter.
    shuffled_order = list(reversed(feature_order))
    x_shuffled = sample[shuffled_order].astype(float).to_numpy()
    shuffled_probabilities = loaded.predict_proba(x_shuffled)[:, 1]
    order_difference = float(np.abs(loaded_probabilities
                                    - shuffled_probabilities).max())
    a.check("Feature order is load-bearing, so the saved schema is required",
            order_difference > 1e-6,
            f"reversing the column order changes predictions by up to "
            f"{order_difference:.4f}. This is why model_metadata.json carries "
            f"feature_order and why the API must read it rather than guess.")

    a.check("Probabilities are valid and not degenerate",
            bool((loaded_probabilities >= 0).all()
                 and (loaded_probabilities <= 1).all()
                 and loaded_probabilities.std() > 0.01),
            f"range {loaded_probabilities.min():.4f} to "
            f"{loaded_probabilities.max():.4f}, sd "
            f"{loaded_probabilities.std():.4f}")

    comparison = None
    if oof_path.exists():
        oof = pd.read_parquet(oof_path)
        tier_keys = [t["key"] for t in models.TIERS]
        missing_predictions = {k: int(oof[k].isna().sum()) for k in tier_keys
                               if k in oof}
        a.check("Out-of-fold predictions exist for every row and every tier",
                len(oof) == len(frame)
                and all(v == 0 for v in missing_predictions.values()),
                f"{len(oof):,} rows; missing per tier {missing_predictions}")

        comparison = evaluate.compare_tiers(
            {k: oof[k].to_numpy() for k in tier_keys if k in oof},
            oof[features.TARGET_COLUMN].to_numpy())

        game_state = comparison.loc[comparison["tier"].eq("tier3_celtics")]
        with_lineup = comparison.loc[comparison["tier"].eq("tier4_lineup")]
        if not game_state.empty and not with_lineup.empty:
            before = float(game_state.iloc[0]["brier"])
            after = float(with_lineup.iloc[0]["brier"])
            helped = after < before
            a.check("Lineup strength contribution measured (either direction "
                    "is a valid finding)", True,
                    f"Brier {before:.4f} -> {after:.4f} "
                    f"({(before - after) / before:+.2%}). "
                    f"Lineup strength "
                    f"{'improves' if helped else 'does NOT improve'} "
                    f"out-of-sample prediction.\n"
                    f"Reported as measured. A negative result here is a real "
                    f"finding, not a failure, and must not be tuned away.")

    return a, metadata, comparison


def build_report(a, metadata, comparison):
    n_fail = len(a.failed)
    lines = [
        "=" * 74,
        "PHASE 4 VALIDATION - SAVED MODEL AND REPRODUCIBILITY",
        f"Run at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "=" * 74,
        "",
        "CHECKS",
    ]
    lines += a.render()

    if metadata:
        lines += ["", "SAVED ARTEFACT", "-" * 14,
                  f"  model        : {metadata.get('model_file')}",
                  f"  class        : {metadata.get('model_class')}",
                  f"  tier         : {metadata.get('tier_name')}",
                  f"  features     : {metadata.get('n_features')}",
                  f"  seed         : {metadata.get('random_seed')}",
                  f"  trained on   : {metadata.get('trained_on_games')} games, "
                  f"{metadata.get('trained_on_events'):,} events",
                  f"  seasons      : "
                  f"{', '.join(metadata.get('trained_on_seasons', []))}",
                  "",
                  "  The saved model is fitted on EVERY season, which is correct",
                  "  for deployment and wrong for evaluation. Every metric in",
                  "  phase4_results.txt is out of fold and comes from different",
                  "  models. The metadata says so explicitly."]

    if comparison is not None and not comparison.empty:
        lines += ["", "OUT-OF-FOLD TIER COMPARISON", "-" * 27,
                  f"  {'tier':<18}{'AUC':>9}{'Brier':>10}{'skill':>9}"
                  f"{'logloss':>10}{'ECE':>9}"]
        for _, r in comparison.iterrows():
            lines.append(f"  {r['tier']:<18}{r['auc']:>9.4f}{r['brier']:>10.4f}"
                         f"{r['brier_skill']:>8.1%}{r['logloss']:>10.4f}"
                         f"{r['ece']:>9.4f}")

    lines += ["", "=" * 74,
              f"RESULT: {len(a.results) - n_fail} passed, {n_fail} failed"]
    if n_fail == 0:
        lines.append("The model is saved, reloadable and reproducible.")
        lines.append("Checklist item 6 is satisfied.")
    else:
        lines.append("NOT validated. Failed: "
                     + ", ".join(r[0] for r in a.failed))
    lines.append("=" * 74)
    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    a, metadata, comparison = run()
    report = build_report(a, metadata, comparison)
    print(report)
    out = config.REPORTS_DIR / "phase4_validation.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to: {out}")
    return len(a.failed) == 0


if __name__ == "__main__":
    main()
