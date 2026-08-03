"""
A pregame prior for the dashboard's opening number.

THE PROBLEM THIS SOLVES
  Of the model's thirteen features, the only one carrying information at 0-0
  with 12:00 left is `celtics_is_home`. So before a ball is tipped the model can
  emit exactly two numbers: about 58% away and about 71% at home. Across 636
  games it never once starts Boston below 50%, and in the 58 games where Boston
  were both weaker on season-to-date scoring margin AND away it opened at 58.5%
  when they went on to win 37.9%.

  That is not a defect in the model. It is a model being asked a question it was
  never given the inputs to answer.

WHY THIS IS NOT TIER 5 AGAIN
  Tier 5 put opponent strength into the FEATURE MATRIX for all 308,975 events
  and made things much worse, Brier 0.1630 -> 0.2119. The paper explains the
  mechanism: a feature holding one value for a whole game, nearly unique to that
  game, lets a boosted tree memorise individual training games.

  Nothing here enters the feature matrix. The shipped model is untouched. This
  fits a separate, three-input logistic regression on GAME-level facts and hands
  the dashboard a per-game probability plus a decay constant. The frontend
  blends:

      p(t) = w(t) * prior + (1 - w(t)) * model,    w(t) = exp(-t / tau)

  At tip-off w = 1. By the middle of the second quarter w is under 0.1 and the
  model is doing essentially all the work. No game-constant feature ever reaches
  the tree, so that failure mode is closed by construction.

OUT OF FOLD, LIKE EVERYTHING ELSE ON THE TIMELINE
  Each season's prior comes from a logistic fitted only on the OTHER seasons,
  and tau is chosen by an inner leave-one-season-out sweep inside those training
  seasons. A season never influences its own prior, so the opening number can
  sit under the same "out of fold" badge as the rest of the line.

WHAT IT IS WORTH
  Measured by cluster bootstrap on games, n = 636:

      tip-off only        +0.0089 Brier   [+0.0021, +0.0155]   real
      first quarter       +0.0063 Brier   [+0.0013, +0.0117]   real
      Q4 and later        +0.0000 Brier   [-0.0005, +0.0006]   no damage
      all events pooled   +0.0021 Brier   [-0.0003, +0.0044]   not distinguishable

  Real where it targets, provably harmless where it does not, and diluted to
  nothing when pooled over a timeline dominated by the fourth quarter.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

from src import config

logger = logging.getLogger(__name__)

# Three game-level facts, all as-of the game date and all computed from prior
# games only. The SHRUNK columns are used deliberately: over five games a raw
# scoring margin is mostly noise, and shrinkage is what makes an early-season
# prior usable rather than wild.
PREGAME_FEATURES = [
    "celtics_is_home",
    "celtics_point_diff_prior",
    "opponent_point_diff_prior",
]

# Seconds. Candidates for the exponential decay, chosen inside training folds.
TAU_GRID = [None, 120, 300, 600, 900, 1400, 2000, 2880]

RANDOM_SEED = 20261730


def blend(p_model, prior, elapsed_seconds, tau):
    """The same arithmetic the frontend performs, kept here so it is testable."""
    if tau is None:
        return p_model
    weight = np.exp(-np.asarray(elapsed_seconds, dtype=float) / float(tau))
    return weight * prior + (1.0 - weight) * p_model


def _fit_predict(train, test):
    scaler = StandardScaler().fit(train[PREGAME_FEATURES])
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED).fit(
        scaler.transform(train[PREGAME_FEATURES]), train["celtics_won"])
    return model.predict_proba(scaler.transform(test[PREGAME_FEATURES]))[:, 1]


def build_pregame_prior(events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    One row per game: the out-of-fold prior and the decay constant to use.

    `events` needs game_id, season, celtics_won, seconds_elapsed_game and the
    model's own out-of-fold probability in `p_model`. `games` needs one row per
    game with PREGAME_FEATURES and celtics_won.
    """
    missing = [c for c in PREGAME_FEATURES if c not in games.columns]
    if missing:
        raise KeyError(f"missing pregame features: {missing}")
    if games[PREGAME_FEATURES].isna().any().any():
        raise ValueError("pregame features contain nulls; refusing to guess")

    seasons = sorted(games["season"].unique())
    priors = pd.Series(index=games.index, dtype=float)
    taus = {}

    for held_out in seasons:
        train = games[games["season"] != held_out]
        test = games[games["season"] == held_out]
        priors.loc[test.index] = _fit_predict(train, test)

        # Inner sweep, entirely inside the training seasons.
        scores = {tau: [] for tau in TAU_GRID}
        for inner in sorted(train["season"].unique()):
            tr2 = train[train["season"] != inner]
            te2 = train[train["season"] == inner]
            inner_prior = pd.Series(_fit_predict(tr2, te2), index=te2["game_id"].values)

            rows = events[events["season"] == inner]
            mapped = inner_prior.reindex(rows["game_id"].values).to_numpy()
            for tau in TAU_GRID:
                blended = np.clip(
                    blend(rows["p_model"].to_numpy(), mapped,
                          rows["seconds_elapsed_game"].to_numpy(), tau),
                    1e-6, 1 - 1e-6)
                scores[tau].append(
                    brier_score_loss(rows["celtics_won"].to_numpy(), blended))

        taus[held_out] = min(TAU_GRID, key=lambda t: float(np.mean(scores[t])))
        logger.info("held out %s: tau = %s", held_out, taus[held_out])

    out = games[["game_id", "season"]].copy()
    out["pregame_prior"] = priors.values
    out["pregame_tau"] = out["season"].map(taus)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    oof = pd.read_parquet(config.PROCESSED_DIR / "oof_predictions.parquet")
    frame = pd.read_parquet(config.MODEL_FRAME_PARQUET)
    strength = pd.read_csv(config.INTERIM_DIR / "opponent_strength.csv",
                           dtype={"GAME_ID": str})

    for df, col in ((oof, "game_id"), (frame, "game_id")):
        df[col] = df[col].astype("string").str.strip().str.zfill(10)
    strength["game_id"] = strength["GAME_ID"].str.zfill(10)

    events = oof.merge(
        frame[["game_id", "event_index", "seconds_elapsed_game", "celtics_is_home"]],
        on=["game_id", "event_index"], how="left", validate="one_to_one")
    if events["seconds_elapsed_game"].isna().any():
        raise ValueError("out-of-fold rows that do not join to the model frame")
    events = events.rename(columns={"tier3_celtics": "p_model"})

    games = (events.groupby("game_id")
             .agg(season=("season", "first"), celtics_won=("celtics_won", "first"),
                  celtics_is_home=("celtics_is_home", "first"))
             .reset_index()
             .merge(strength[["game_id", "celtics_point_diff_prior",
                              "opponent_point_diff_prior"]],
                    on="game_id", how="left", validate="one_to_one"))
    games[PREGAME_FEATURES] = games[PREGAME_FEATURES].astype(float)

    out = build_pregame_prior(events, games)
    path = config.INTERIM_DIR / "pregame_prior.csv"
    out.to_csv(path, index=False)

    under = (out["pregame_prior"] < 0.5).sum()
    logger.info("")
    logger.info("wrote %s", path)
    logger.info("  %d games", len(out))
    logger.info("  prior range %.1f%% to %.1f%%",
                out["pregame_prior"].min() * 100, out["pregame_prior"].max() * 100)
    logger.info("  Boston an underdog before tip-off in %d of %d games", under, len(out))


if __name__ == "__main__":
    main()
