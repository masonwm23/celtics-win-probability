"""
Phase 4: the model tiers.

Four models, arranged so that each one isolates what the next adds. Reporting a
single number for a win probability model tells you almost nothing; the useful
question is what each ingredient is worth.

  Tier 1  PREGAME ONLY
          Home court only. No in-game information at all. This is the floor: any
          in-game model must beat it, and by how much is the measure of what
          watching the game is worth.

  Tier 2  GENERIC IN-GAME
          Logistic regression on score margin and time remaining, plus their
          interaction. This is the stand-in for a generic public win probability
          model.

          IT IS NOT ESPN'S MODEL. ESPN does not publish theirs, so calling this
          "ESPN" would misrepresent it. It is labelled a generic baseline
          everywhere it appears, and the research plan's "compare against ESPN if
          feasible" is answered honestly: a like-for-like comparison against the
          real thing is not feasible without their model, so a transparent
          reimplementation of the standard approach stands in its place.

  Tier 3  CELTICS-SPECIFIC GAME STATE
          Gradient boosting on all 13 validated game-state features. This is the
          research plan's core model.

  Tier 4  CELTICS-SPECIFIC PLUS LINEUP
          Tier 3 plus lineup strength, computed inside each fold from training
          seasons only. Separated from tier 3 on purpose: the contribution of
          lineup context is a finding to be measured, not an assumption.

Every tier is fitted and scored inside the same leave-one-season-out folds, on
identical rows, so the comparison is like-for-like.
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import config, features, lineup_strength, opponent_strength

# Tier 2's feature set. Deliberately minimal: this is what a generic public model
# is understood to use, and adding more would stop it being a baseline.
GENERIC_FEATURES = ["celtics_margin", "seconds_remaining_game"]

PREGAME_FEATURES = ["celtics_is_home"]

GAME_STATE_FEATURES = list(features.FEATURE_COLUMNS)

LINEUP_FEATURES = (GAME_STATE_FEATURES
                   + list(lineup_strength.LINEUP_FEATURE_COLUMNS))


def make_pregame_model():
    """
    Predicts the training base rate, split by home and away.

    A logistic regression on a single binary feature, which is exactly the
    home-court prior and nothing more.
    """
    return LogisticRegression(max_iter=1000)


def make_generic_model():
    """
    Logistic regression on margin, time, and their interaction.

    The interaction matters: a five-point lead means something very different
    with two minutes left than with thirty. Without it this baseline would be
    unfairly weak, and a straw man is not a useful comparison.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000)),
    ])


def add_generic_interaction(x):
    """Append margin divided by remaining minutes to the generic feature matrix."""
    x = np.asarray(x, dtype=float)
    margin = x[:, 0]
    seconds = np.clip(x[:, 1], 1.0, None)
    urgency = margin / (seconds / 60.0)
    return np.column_stack([x, urgency])


DEFAULT_MIN_CHILD_WEIGHT = 20


def make_gradient_boosting(min_child_weight=DEFAULT_MIN_CHILD_WEIGHT):
    """
    The Celtics-specific model. Gradient boosting, as the research plan specifies.

    Settings are modest and fixed rather than tuned on the test seasons, because
    tuning against the held-out season is itself a leak. If hyperparameters are
    tuned later it must be inside the training folds only.

    ON min_child_weight
    -------------------
    The default of 20 assumes every feature varies within a game. It is toothless
    against a feature that is CONSTANT across a game: a game averages 486 events
    that all share one outcome label, so a leaf holding one whole game holds 486
    rows and clears a threshold of 20 twenty-four times over. The tree can then
    isolate a single training game into a pure node and memorise its result.

    src/controls.py tests exactly this, and the parameter is exposed here so a
    game-aware floor can be used where game-constant features are involved.
    """
    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=min_child_weight,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )


TIERS = [
    {
        "key": "tier1_pregame",
        "name": "Tier 1: pregame only (home court)",
        "features": PREGAME_FEATURES,
        "factory": make_pregame_model,
        "transform": None,
        "needs_lineup": False,
    },
    {
        "key": "tier2_generic",
        "name": "Tier 2: generic in-game baseline (margin and time)",
        "features": GENERIC_FEATURES,
        "factory": make_generic_model,
        "transform": add_generic_interaction,
        "needs_lineup": False,
    },
    {
        "key": "tier3_celtics",
        "name": "Tier 3: Celtics-specific, game state",
        "features": GAME_STATE_FEATURES,
        "factory": make_gradient_boosting,
        "transform": None,
        "needs_lineup": False,
    },
    {
        "key": "tier4_lineup",
        "name": "Tier 4: Celtics-specific plus lineup strength",
        "features": LINEUP_FEATURES,
        "factory": make_gradient_boosting,
        "transform": None,
        "needs_lineup": True,
    },
]

TIER_BY_KEY = {tier["key"]: tier for tier in TIERS}


def fit_predict(tier, train_frame, train_target, test_frame):
    """Fit one tier on a fold's training rows and predict its test rows."""
    x_train = train_frame[tier["features"]].astype(float).to_numpy()
    x_test = test_frame[tier["features"]].astype(float).to_numpy()
    if tier["transform"] is not None:
        x_train = tier["transform"](x_train)
        x_test = tier["transform"](x_test)

    model = tier["factory"]()
    model.fit(x_train, np.asarray(train_target, dtype=int))
    return model.predict_proba(x_test)[:, 1], model


# ---------------------------------------------------------------------------
# PRE-REGISTERED LINEUP VARIANTS
# ---------------------------------------------------------------------------
#
# The first lineup tier made the model measurably WORSE: pooled Brier 0.1630 ->
# 0.1823, and in the first six minutes Brier skill fell from +1.4% to -14.5%.
#
# Before concluding that lineup context does not help, three alternative
# parameterisations are tested. They are FIXED IN ADVANCE and ALL of them are
# reported whether they help or not. That distinction matters: searching until
# something works and reporting only the winner is how false findings are
# manufactured. Pre-registering the set and reporting all outcomes is not.
#
#   Variant A  DIFFERENCE ONLY. The original tier added three correlated columns
#              (both teams' strength and their difference). Only the difference
#              should carry signal, so the other two may be adding variance.
#
#   Variant B  HEAVIER SHRINKAGE. 2000 minutes instead of 500, roughly 28 games.
#              If the problem is noisy player estimates failing to transfer to a
#              held-out season, stronger shrinkage toward zero should help.
#
#   Variant C  INTERACTED WITH TIME. Roster quality should matter most when
#              there is time left for it to express itself, and hardly at all
#              with thirty seconds on the clock. A flat feature cannot express
#              that, and the damage being concentrated early is consistent with
#              the model misusing it.
#
# If none of the three improves on tier 3, the negative result stands and is
# reported as a finding.

REGULATION_SECONDS = 2880.0


def add_lineup_interactions(frame):
    """Add the time-weighted lineup term used by variant C."""
    frame = frame.copy()
    weight = (frame["seconds_remaining_game"] / REGULATION_SECONDS).clip(0, 1)
    frame["lineup_diff_x_time"] = frame["lineup_strength_diff"] * weight
    return frame


LINEUP_VARIANTS = [
    {
        "key": "variant_a_diff_only",
        "name": "Variant A: lineup difference only",
        "shrinkage_minutes": 500.0,
        "features": GAME_STATE_FEATURES + ["lineup_strength_diff"],
        "needs_interactions": False,
    },
    {
        "key": "variant_b_heavy_shrinkage",
        "name": "Variant B: heavier shrinkage (2000 minutes)",
        "shrinkage_minutes": 2000.0,
        "features": LINEUP_FEATURES,
        "needs_interactions": False,
    },
    {
        "key": "variant_c_time_interaction",
        "name": "Variant C: lineup difference interacted with time remaining",
        "shrinkage_minutes": 500.0,
        "features": (GAME_STATE_FEATURES
                     + ["lineup_strength_diff", "lineup_diff_x_time"]),
        "needs_interactions": True,
    },
]

for _variant in LINEUP_VARIANTS:
    _variant.setdefault("factory", make_gradient_boosting)
    _variant.setdefault("transform", None)
    _variant.setdefault("needs_lineup", True)


# ---------------------------------------------------------------------------
# PRE-REGISTERED OPPONENT TIERS
# ---------------------------------------------------------------------------
#
# The registration form's premise is that the opponent should be part of the
# matchup context. This is the last untested element of that hypothesis.
#
# Every opponent measure is computed AS OF THE GAME DATE from prior games only.
# A full-season record includes games played after the one being predicted, and
# joining that on would be a leak. See src/opponent_strength.py.
#
# Unlike lineup strength, these features do NOT need to be rebuilt inside each
# fold. Lineup strength is an average over a set of seasons, so which seasons you
# average over determines the value, and using the held-out season would leak.
# An as-of-date opponent record is a function of one specific point in time and
# the games before it. It does not depend on which seasons the model trains on,
# so computing it once is correct rather than convenient.
#
# Same discipline as the lineup work: the variants are FIXED IN ADVANCE and ALL
# are reported, whichever way they come out.
#
#   Tier 5     ALL opponent measures on top of tier 3.
#   Variant D  Opponent season-to-date point differential only. The single most
#              standard measure of team quality.
#   Variant E  Opponent recent form only, last 10 games. If season-long averages
#              are too static, recent form is the natural alternative.
#   Variant F  Celtics minus opponent differential only. Encodes the matchup as
#              one number rather than two.

OPPONENT_FEATURES = (GAME_STATE_FEATURES
                     + list(opponent_strength.OPPONENT_FEATURE_COLUMNS))

OPPONENT_TIERS = [
    {
        "key": "tier5_opponent",
        "name": "Tier 5: Celtics-specific plus opponent context",
        "features": OPPONENT_FEATURES,
    },
    {
        "key": "variant_d_opp_point_diff",
        "name": "Variant D: opponent season-to-date point differential only",
        "features": GAME_STATE_FEATURES + ["opponent_point_diff_prior"],
    },
    {
        "key": "variant_e_opp_recent_form",
        "name": "Variant E: opponent recent form only (last 10 games)",
        "features": GAME_STATE_FEATURES + ["opponent_recent_form"],
    },
    {
        "key": "variant_f_strength_diff",
        "name": "Variant F: Celtics minus opponent differential only",
        "features": GAME_STATE_FEATURES + ["strength_diff_prior"],
    },
]

for _tier in OPPONENT_TIERS:
    _tier.setdefault("factory", make_gradient_boosting)
    _tier.setdefault("transform", None)
    _tier.setdefault("needs_lineup", False)
    _tier.setdefault("needs_opponent", True)

for _tier in TIERS + LINEUP_VARIANTS:
    _tier.setdefault("needs_opponent", False)

ALL_TIER_BY_KEY = {t["key"]: t for t in TIERS + LINEUP_VARIANTS + OPPONENT_TIERS}
