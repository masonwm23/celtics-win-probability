"""
Phase 7: clean re-tests, after Phase 6 showed the earlier conclusions were
contaminated.

WHAT PHASE 6 ESTABLISHED
------------------------
A feature that is constant within a game, and takes a distinct value in nearly
every game, lets a tree isolate one training game into a pure leaf and memorise
its outcome. Event rows are not independent: a game averages 486 of them and all
carry the same label. Tier 5's TRAINING Brier was 0.0076 against a baseline of
0.2290, which is not learning, it is recall.

A column of random numbers reproduced 85 percent of the damage that real opponent
context appeared to do. Coarsened to five values, the same random column did
almost nothing, and real opponent quality in five buckets was exactly neutral.

WHAT PHASE 6 GOT WRONG
----------------------
The prediction that a min_child_weight of 4,000, about eight games, would
neutralise the mechanism FAILED. A random column at that floor still cost 0.0101
with an interval excluding zero. So C4 and C6 were never the fair tests they were
advertised as, and the lineup question was left open rather than answered.

WHAT PHASE 7 DOES
-----------------
1. DOSE-RESPONSE. If cardinality is the mechanism, damage should grow with the
   number of distinct values a feature can take, and it should grow the SAME way
   for a real feature and for a meaningless one. Both ladders are run at matched
   resolutions: about 5, 20, 100 and 608 distinct values.

   This is a stronger claim than "a random column also hurts". It predicts the
   shape of the curve in advance, and a shape can fail to appear.

2. CLEAN LINEUP RE-TEST. Phase 4 reported that lineup strength genuinely hurts
   and attributed it to player values not transferring across seasons. That
   explanation is now suspect. Lineup strength is tested here at five buckets in
   the tree and as a plain term in the linear model, neither of which can
   memorise a game.

3. THE DELIVERABLE QUESTION. C8, a logistic regression on margin, time, their
   interaction and one opponent number, beat tier 3 on Brier, AUC, calibration
   and seven of eight game phases. It has never been compared to tier 3 through
   the bootstrap. The rule is fixed here, in code, BEFORE the run: the deliverable
   changes only if the interval excludes zero in the challenger's favour. A point
   estimate is not enough, which is the same rule that kept tier 4 from shipping.

4. A LINEAR NULL CONTROL. If adding a random per-game column to the linear model
   also improved it, the C8 result would be an artefact of a different kind. It
   must come out null.
"""

import logging

import numpy as np
import pandas as pd

from src import config, controls, features, models

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Controlled cardinality
# ---------------------------------------------------------------------------
#
# The opponent ladder is built by ROUNDING, which uses no data-dependent
# parameter at all: a step of 5.0 points is a step of 5.0 points whatever the
# data does. Measured against the real 636 games these steps give 5, 20 and 88
# distinct values against a raw 608.

OPPONENT_LADDER = [
    ("opp_bins5", 5.0, "about 5 values"),
    ("opp_bins20", 1.0, "about 20 values"),
    ("opp_bins100", 0.2, "about 88 values"),
]
OPPONENT_SOURCE = "opponent_point_diff_prior"

# The random ladder is NESTED, exactly like the opponent ladder: one uniform draw
# per game, then coarsened. Same information at every rung, only the resolution
# changes. Drawing independently at each rung would compare different things.
RANDOM_LADDER = [
    ("rand_bins5", 5),
    ("rand_bins20", 20),
    ("rand_bins100", 100),
]
RANDOM_RAW = "rand_raw"

LINEUP_BINS = 5
LINEUP_SOURCE = "lineup_strength_diff"
LINEUP_BINNED = "lineup_diff_bin"


def add_opponent_ladder(frame: pd.DataFrame) -> pd.DataFrame:
    """Round the opponent differential to a ladder of coarser resolutions."""
    if OPPONENT_SOURCE not in frame.columns:
        raise KeyError(f"{OPPONENT_SOURCE} absent; run "
                       "scripts/14_build_opponent_strength.py first")
    out = frame.copy()
    values = out[OPPONENT_SOURCE].to_numpy(dtype=float)
    for name, step, _label in OPPONENT_LADDER:
        out[name] = np.round(values / step) * step
    return out


def add_random_ladder(frame: pd.DataFrame,
                      seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """
    One uniform draw per game, then the same draw coarsened.

    Nested by construction, so the ladder isolates resolution and nothing else.
    """
    games = np.sort(frame["game_id"].unique())
    rng = np.random.default_rng(seed)
    draw = rng.random(len(games))
    lookup = pd.DataFrame({"game_id": games, RANDOM_RAW: draw})
    for name, k in RANDOM_LADDER:
        lookup[name] = np.floor(draw * k).astype(float)
    out = frame.merge(lookup, on="game_id", how="left", validate="many_to_one")
    if out[[RANDOM_RAW] + [n for n, _ in RANDOM_LADDER]].isna().any().any():
        raise ValueError("random ladder failed to attach to every row")
    out.index = frame.index
    return out


def bin_by_training_quantiles(train_values, test_values, n_bins=LINEUP_BINS):
    """
    Cut a feature into equal-count bins using edges from the TRAINING rows only.

    Rounding cannot be used for lineup strength: its scale is derived per fold
    from that fold's player values, so no fixed step can be stated in advance.
    Taking the edges from training rows keeps the held-out season out of the
    binning decision, which is the property that matters.
    """
    train_values = np.asarray(train_values, dtype=float)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(train_values, quantiles)
    edges = np.unique(edges)
    return (np.digitize(train_values, edges).astype(float),
            np.digitize(np.asarray(test_values, dtype=float),
                        edges).astype(float))


# ---------------------------------------------------------------------------
# The specifications, FIXED IN ADVANCE
# ---------------------------------------------------------------------------

GAME_STATE = list(models.GAME_STATE_FEATURES)
GENERIC = list(models.GENERIC_FEATURES)


def _gb(features_list, key, name, prediction, **extra):
    spec = {"key": key, "name": name, "features": features_list,
            "factory": models.make_gradient_boosting, "transform": None,
            "prediction": prediction}
    spec.update(extra)
    return spec


def _linear(extra_columns, key, name, prediction):
    return {"key": key, "name": name,
            "features": GENERIC + list(extra_columns),
            "factory": models.make_generic_model,
            "transform": models.add_generic_interaction,
            "prediction": prediction}


CLEAN_SPECS = [
    _gb(GAME_STATE, "p7_tier3", "Reference: tier 3, game state only",
        "reference; must reproduce 0.1630 exactly"),
    _linear([], "p7_tier2", "Reference: tier 2, generic linear baseline",
            "reference; must reproduce 0.1641 exactly"),

    # Dose-response, real opponent quality.
    _gb(GAME_STATE + ["opp_bins5"], "p7_opp_bins5",
        "Opponent differential, ~5 distinct values",
        "neutral; too coarse to name a game"),
    _gb(GAME_STATE + ["opp_bins20"], "p7_opp_bins20",
        "Opponent differential, ~20 distinct values",
        "slightly worse than 5 if cardinality is the mechanism"),
    _gb(GAME_STATE + ["opp_bins100"], "p7_opp_bins100",
        "Opponent differential, ~88 distinct values",
        "clearly worse; approaching one value per game"),
    _gb(GAME_STATE + [OPPONENT_SOURCE], "p7_opp_raw",
        "Opponent differential, raw (~608 distinct values)",
        "worst of the ladder; this is Phase 5's variant D"),

    # Dose-response, a meaningless column at matched resolutions.
    _gb(GAME_STATE + ["rand_bins5"], "p7_rand_bins5",
        "RANDOM per-game value, 5 distinct values",
        "neutral; the floor of the random ladder"),
    _gb(GAME_STATE + ["rand_bins20"], "p7_rand_bins20",
        "RANDOM per-game value, 20 distinct values",
        "should track the opponent ladder at the same rung"),
    _gb(GAME_STATE + ["rand_bins100"], "p7_rand_bins100",
        "RANDOM per-game value, 100 distinct values",
        "should track the opponent ladder at the same rung"),
    _gb(GAME_STATE + [RANDOM_RAW], "p7_rand_raw",
        "RANDOM per-game value, one per game",
        "worst of the random ladder; damage with zero information"),

    # Clean lineup re-test.
    _gb(GAME_STATE + [LINEUP_BINNED], "p7_lineup_bins5",
        "Lineup strength difference, 5 bins from training quantiles",
        "re-tests Phase 4's negative result where memorisation is blocked",
        needs_lineup=True),
    _linear([LINEUP_SOURCE], "p7_linear_lineup",
            "LINEAR baseline plus lineup strength difference",
            "lineup in a model with no splits; cannot memorise"),

    # The deliverable question, and its null control.
    _linear([OPPONENT_SOURCE], "p7_linear_opp",
            "LINEAR baseline plus opponent differential",
            "the challenger; must beat tier 3 through the bootstrap to ship"),
    _linear(["strength_diff_prior"], "p7_linear_strength",
            "LINEAR baseline plus Celtics-minus-opponent differential",
            "second challenger, same standard"),
    _linear([RANDOM_RAW], "p7_linear_random",
            "LINEAR baseline plus a RANDOM per-game value",
            "MUST be null; if this improves, the linear result is not signal"),
]

for _spec in CLEAN_SPECS:
    _spec.setdefault("needs_lineup", False)

SPEC_BY_KEY = {s["key"]: s for s in CLEAN_SPECS}

# Comparisons that isolate a FEATURE. Both sides must share a model factory, or
# the difference would confound the feature with the model.
FEATURE_COMPARISONS = [
    ("p7_tier3", "p7_opp_bins5"),
    ("p7_tier3", "p7_opp_bins20"),
    ("p7_tier3", "p7_opp_bins100"),
    ("p7_tier3", "p7_opp_raw"),
    ("p7_tier3", "p7_rand_bins5"),
    ("p7_tier3", "p7_rand_bins20"),
    ("p7_tier3", "p7_rand_bins100"),
    ("p7_tier3", "p7_rand_raw"),
    ("p7_tier3", "p7_lineup_bins5"),
    ("p7_tier2", "p7_linear_lineup"),
    ("p7_tier2", "p7_linear_opp"),
    ("p7_tier2", "p7_linear_strength"),
    ("p7_tier2", "p7_linear_random"),
]

# Comparisons ACROSS model families, run deliberately to choose what ships. The
# factories differ on purpose here, and that is stated rather than hidden.
DELIVERABLE_COMPARISONS = [
    ("p7_tier3", "p7_linear_opp"),
    ("p7_tier3", "p7_linear_strength"),
]

LADDER_PAIRS = [
    ("about 5", "p7_opp_bins5", "p7_rand_bins5"),
    ("about 20", "p7_opp_bins20", "p7_rand_bins20"),
    ("about 100", "p7_opp_bins100", "p7_rand_bins100"),
    ("one per game", "p7_opp_raw", "p7_rand_raw"),
]


def deliverable_verdict(result: dict, challenger: str, incumbent: str) -> str:
    """
    The switching rule, fixed before the run.

    `result` comes from evaluate.bootstrap_brier_difference(incumbent,
    challenger), so a POSITIVE difference means the challenger is better.

    The deliverable changes only when the interval excludes zero in the
    challenger's favour. This is the same standard that stopped tier 4 shipping,
    and applying it symmetrically is the point: a rule that only bites when the
    answer is inconvenient is not a rule.
    """
    better = result["observed_difference"] > 0
    decisive = result["excludes_zero"]
    if better and decisive:
        return (f"SWITCH. {challenger} beats {incumbent} by "
                f"{result['observed_difference']:+.4f} Brier and the interval "
                f"excludes zero. It becomes the deliverable.")
    if better:
        return (f"KEEP {incumbent}. {challenger} is ahead by "
                f"{result['observed_difference']:+.4f} Brier but the interval "
                f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] spans "
                f"zero, so the two are not distinguishable on 636 games. A "
                f"point estimate is not enough to change what ships.")
    return (f"KEEP {incumbent}. {challenger} does not lead on the point "
            f"estimate.")


def dose_response_verdict(opponent_gaps, random_gaps) -> str:
    """
    Read the two ladders.

    `*_gaps` are Brier increases over tier 3 at matched resolutions, coarsest
    first. The mechanism claim predicts BOTH rise with cardinality and that they
    rise together.
    """
    opponent_gaps = [float(g) for g in opponent_gaps]
    random_gaps = [float(g) for g in random_gaps]
    rising = (opponent_gaps[-1] > opponent_gaps[0]
              and random_gaps[-1] > random_gaps[0])
    if not rising:
        return ("DOSE-RESPONSE NOT FOUND. Damage does not grow with the number "
                "of distinct values, so cardinality is not the mechanism and "
                "the Phase 6 account needs revising.")
    span = random_gaps[-1] - random_gaps[0]
    if span <= 0:
        return "DOSE-RESPONSE INCONCLUSIVE on the random ladder."
    tracking = abs((opponent_gaps[-1] - opponent_gaps[0]) - span) / span
    if tracking <= 0.5:
        return ("DOSE-RESPONSE CONFIRMED, AND MATCHED. Damage grows with "
                "resolution for both ladders and the meaningless column tracks "
                "the real one. The degradation is a property of feature "
                "cardinality against non-independent rows, not of opponent "
                "quality.")
    return ("DOSE-RESPONSE CONFIRMED, PARTIALLY MATCHED. Damage grows with "
            "resolution on both ladders, but not at the same rate, so "
            "cardinality explains much of the effect and not all of it.")
