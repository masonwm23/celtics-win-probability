"""
Phase 6: controls for the game-constant memorisation artefact.

WHY THIS FILE EXISTS
--------------------
Phase 5 reported that every opponent formulation made the model dramatically
worse out of fold: tier 5 Brier 0.2119 against tier 3's 0.1630, and variant D,
which adds a SINGLE pregame column, 0.1998. A one-column pregame feature cannot
legitimately cost 0.037 Brier. That is not a finding about basketball; it is a
symptom.

The mechanism, stated so it can be tested rather than argued:

  1. `opponent_point_diff_prior` takes 608 distinct values across 636 games. It
     is an as-of-date float, so nearly every game gets its own value. It is
     functionally a game identifier.
  2. A game averages 486 events, and every event in a game carries the SAME
     win/loss label.
  3. XGBoost runs with min_child_weight = 20. A leaf holding one entire game
     holds 486 rows, which clears that threshold twenty-four times over.
  4. So the tree can split on that column, isolate one training game, and land
     on a PURE node. Gradient boosting finds pure nodes irresistible because the
     training gain is enormous. Trees get spent memorising which training game
     is which instead of learning margin and clock.
  5. In the held-out season those values are new, the splits are meaningless,
     and the predictions are noise.

Everything observed in Phase 5 fits: damage is worst in the first six minutes
(-26.1% skill), where margin carries least information, and recovers to +43.7%
by the fourth quarter where margin dominates enough to override the noise.
Calibration error goes 0.0140 -> 0.1276, which is the signature of memorisation
rather than of a weak feature.

PRE-REGISTERED PREDICTIONS
--------------------------
These are written down BEFORE the controls are run, so the result cannot be
rationalised after the fact. If the memorisation account is right:

  C1  A RANDOM near-unique number, constant within each game and carrying zero
      information about anything, degrades the model by roughly as much as the
      real opponent features do. This is the decisive control. A random column
      cannot contain a real effect, so any damage it does is pure mechanism.

  C2  The same random number COARSENED to five buckets does roughly nothing.
      Five values cannot isolate a game, so if cardinality is the culprit this
      control should be near-neutral.

  C3  Real opponent strength coarsened to five fixed quality buckets is neutral
      or better than tier 3, not catastrophically worse.

  C4  vs C5  With a game-aware min_child_weight, so that no leaf can hold fewer
      than several games, the opponent features stop being able to memorise.
      C5 is the same floor applied to tier 3 with no opponent features, which is
      the control for the parameter change itself. C4 against C5 is the FAIR
      test of whether opponent context helps.

  C8  vs tier 2  A logistic regression cannot isolate individual games by
      splitting, so it is immune to this mechanism entirely. If opponent
      strength carries real signal, it should show up here.

If C1 comes out near-neutral instead, the memorisation account is WRONG, the
Phase 5 result stands as measured, and that is what gets written up.

THIS ALSO PUTS THE PHASE 4 LINEUP CONCLUSION IN QUESTION
--------------------------------------------------------
Lineup strength is constant across each stint rather than each game, so the same
mechanism applies in weaker form, and the measured degradation was correspondingly
smaller (0.1823 against tier 5's 0.2119). Phase 4 attributed that to player values
failing to transfer across seasons. That explanation may be partly or entirely
wrong. C6 applies the same game-aware floor to the lineup tier so the two
explanations can be told apart.
"""

import logging

import numpy as np
import pandas as pd

from src import config, features, lineup_strength, models, opponent_strength

logger = logging.getLogger(__name__)

# A game averages 486 events. This floor is roughly eight games' worth, so no
# leaf can be built around a single game or even a handful of them. Fixed in
# advance, not tuned against the held-out seasons.
GAME_AWARE_MIN_CHILD_WEIGHT = 4000

# Opponent quality cut points, in season-to-date point differential. Chosen from
# the known shape of NBA team differentials, NOT from the observed distribution
# of this dataset, and fixed before any control was run. Five buckets:
#   0: weak (< -5)   1: below average   2: average   3: above average   4: strong (> +5)
OPPONENT_BUCKET_EDGES = (-5.0, -2.0, 2.0, 5.0)

RANDOM_FEATURES = ["random_game_constant", "random_game_bucket"]
BUCKET_FEATURES = ["opponent_quality_bucket", "strength_diff_bucket"]


# ---------------------------------------------------------------------------
# The control features
# ---------------------------------------------------------------------------

def add_random_game_constants(frame: pd.DataFrame,
                              seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """
    Two columns that are constant within each game and carry NO information.

    `random_game_constant` is drawn once per game from a normal distribution, so
    it is near-unique across games: the same shape as the real opponent feature,
    with none of the meaning.

    `random_game_bucket` is drawn once per game from five values, so it is
    game-constant but CANNOT isolate a game. The pair separates "game-constant"
    from "game-constant and high-cardinality", which are different accusations.
    """
    games = np.sort(frame["game_id"].unique())
    rng = np.random.default_rng(seed)
    lookup = pd.DataFrame({
        "game_id": games,
        "random_game_constant": rng.normal(0.0, 5.0, size=len(games)),
        "random_game_bucket": rng.integers(0, 5, size=len(games)).astype(float),
    })
    out = frame.merge(lookup, on="game_id", how="left", validate="many_to_one")
    if out[RANDOM_FEATURES].isna().any().any():
        raise ValueError("random game constants failed to attach to every row")
    out.index = frame.index
    return out


def add_opponent_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Coarsen the real opponent measures to five fixed quality buckets.

    Deliberately destroys the feature's ability to identify a game while keeping
    its ability to say "this is a strong opponent". If the Phase 5 damage came
    from cardinality rather than from opponent quality, this should be harmless.
    """
    needed = ["opponent_point_diff_prior", "strength_diff_prior"]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise KeyError(f"opponent features absent: {missing}. "
                       "Run scripts/14_build_opponent_strength.py first.")
    out = frame.copy()
    out["opponent_quality_bucket"] = np.digitize(
        out["opponent_point_diff_prior"].to_numpy(),
        OPPONENT_BUCKET_EDGES).astype(float)
    out["strength_diff_bucket"] = np.digitize(
        out["strength_diff_prior"].to_numpy(),
        OPPONENT_BUCKET_EDGES).astype(float)
    return out


def make_gb_game_aware():
    """Gradient boosting with a floor that no single game can satisfy."""
    return models.make_gradient_boosting(
        min_child_weight=GAME_AWARE_MIN_CHILD_WEIGHT)


# ---------------------------------------------------------------------------
# The control specifications, FIXED IN ADVANCE
# ---------------------------------------------------------------------------

GAME_STATE = list(models.GAME_STATE_FEATURES)

CONTROL_SPECS = [
    {
        "key": "ref_tier3",
        "name": "Reference: tier 3, game state only",
        "features": GAME_STATE,
        "factory": models.make_gradient_boosting,
        "prediction": "reference point, should reproduce Phase 4 exactly",
    },
    {
        "key": "ref_tier5",
        "name": "Reference: tier 5, plus opponent context",
        "features": None,          # filled in below, needs opponent columns
        "factory": models.make_gradient_boosting,
        "needs_opponent": True,
        "prediction": "reference point, should reproduce Phase 5 exactly",
    },
    {
        "key": "c1_random_unique",
        "name": "C1: plus a RANDOM near-unique game constant",
        "features": GAME_STATE + ["random_game_constant"],
        "factory": models.make_gradient_boosting,
        "needs_random": True,
        "prediction": "DECISIVE. If this degrades like tier 5, the damage is "
                      "mechanism, not opponents.",
    },
    {
        "key": "c2_random_bucket",
        "name": "C2: plus a RANDOM 5-value game constant",
        "features": GAME_STATE + ["random_game_bucket"],
        "factory": models.make_gradient_boosting,
        "needs_random": True,
        "prediction": "near-neutral, since 5 values cannot isolate a game",
    },
    {
        "key": "c3_opponent_bucketed",
        "name": "C3: plus REAL opponent quality in 5 fixed buckets",
        "features": GAME_STATE + ["opponent_quality_bucket"],
        "factory": models.make_gradient_boosting,
        "needs_opponent": True,
        "needs_buckets": True,
        "prediction": "neutral or better than tier 3, not catastrophic",
    },
    {
        "key": "c4_opponent_game_aware",
        "name": "C4: opponent context, game-aware min_child_weight",
        "features": None,          # opponent features, filled in below
        "factory": make_gb_game_aware,
        "needs_opponent": True,
        "prediction": "the FAIR opponent test; compare against C5, not tier 3",
    },
    {
        "key": "c5_tier3_game_aware",
        "name": "C5: tier 3, game-aware min_child_weight (control for C4)",
        "features": GAME_STATE,
        "factory": make_gb_game_aware,
        "prediction": "isolates what the parameter change alone does",
    },
    {
        "key": "c6_lineup_game_aware",
        "name": "C6: lineup strength, game-aware min_child_weight",
        "features": None,          # lineup features, filled in below
        "factory": make_gb_game_aware,
        "needs_lineup": True,
        "prediction": "re-tests the Phase 4 lineup conclusion against C5",
    },
    {
        "key": "c7_random_game_aware",
        "name": "C7: random near-unique constant, game-aware floor",
        "features": GAME_STATE + ["random_game_constant"],
        "factory": make_gb_game_aware,
        "needs_random": True,
        "prediction": "should be neutral; confirms the floor is the fix",
    },
    {
        "key": "c8_linear_opponent",
        "name": "C8: LINEAR baseline plus opponent differential",
        "features": models.GENERIC_FEATURES + ["opponent_point_diff_prior"],
        "factory": models.make_generic_model,
        "transform": models.add_generic_interaction,
        "needs_opponent": True,
        "prediction": "immune to this mechanism; compare against tier 2",
    },
    {
        "key": "c9_linear_strength_diff",
        "name": "C9: LINEAR baseline plus Celtics-minus-opponent differential",
        "features": models.GENERIC_FEATURES + ["strength_diff_prior"],
        "factory": models.make_generic_model,
        "transform": models.add_generic_interaction,
        "needs_opponent": True,
        "prediction": "immune to this mechanism; compare against tier 2",
    },
    {
        "key": "ref_tier2",
        "name": "Reference: tier 2, generic linear baseline",
        "features": list(models.GENERIC_FEATURES),
        "factory": models.make_generic_model,
        "transform": models.add_generic_interaction,
        "prediction": "reference point for C8 and C9",
    },
]

# Fill in the specs that depend on the opponent and lineup feature lists.
for _spec in CONTROL_SPECS:
    _spec.setdefault("transform", None)
    _spec.setdefault("needs_lineup", False)
    _spec.setdefault("needs_opponent", False)
    _spec.setdefault("needs_random", False)
    _spec.setdefault("needs_buckets", False)
    if _spec["key"] in ("ref_tier5", "c4_opponent_game_aware"):
        _spec["features"] = (GAME_STATE
                             + list(opponent_strength.OPPONENT_FEATURE_COLUMNS))
    elif _spec["key"] == "c6_lineup_game_aware":
        _spec["features"] = (GAME_STATE
                             + list(lineup_strength.LINEUP_FEATURE_COLUMNS))

SPEC_BY_KEY = {s["key"]: s for s in CONTROL_SPECS}

# The comparisons that answer the question. Each is (baseline, candidate).
CONTROL_COMPARISONS = [
    ("ref_tier3", "c1_random_unique"),
    ("ref_tier3", "c2_random_bucket"),
    ("ref_tier3", "c3_opponent_bucketed"),
    ("c5_tier3_game_aware", "c4_opponent_game_aware"),
    ("c5_tier3_game_aware", "c6_lineup_game_aware"),
    ("c5_tier3_game_aware", "c7_random_game_aware"),
    ("ref_tier2", "c8_linear_opponent"),
    ("ref_tier2", "c9_linear_strength_diff"),
]


def verdict(random_gap: float, opponent_gap: float,
            tolerance: float = 0.5) -> str:
    """
    Read the decisive control.

    `random_gap` is how much C1 (a meaningless column) degraded Brier relative to
    tier 3; `opponent_gap` is how much tier 5 degraded it. Both are positive when
    the addition made things worse.

    If a random column does at least `tolerance` of the damage the real opponent
    features do, the Phase 5 result is dominated by mechanism and cannot be read
    as a statement about opponent quality.
    """
    if opponent_gap <= 0:
        return ("The opponent features did not degrade the model, so there is "
                "no artefact to explain.")
    ratio = random_gap / opponent_gap
    if ratio >= tolerance:
        return (f"ARTEFACT CONFIRMED. A meaningless random column reproduces "
                f"{ratio:.0%} of the damage the opponent features do. The "
                f"Phase 5 result is a property of the model configuration, not "
                f"of opponent quality, and must not be reported as a finding "
                f"about basketball.")
    return (f"ARTEFACT NOT SUPPORTED. A random column reproduces only "
            f"{ratio:.0%} of the damage. The memorisation account does not "
            f"explain the Phase 5 result, which therefore stands as measured.")
