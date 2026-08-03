"""
Phase 9c: the Python API the dashboard talks to.

WHAT IT SERVES, AND FROM WHERE
------------------------------
Two different things, and the distinction is the whole point of this file.

  REPLAY   /api/games and /api/games/{id} serve the precomputed files from
           data/serving. Every probability in them is OUT OF FOLD: predicted by
           a model that never saw that game's season. This is what the timeline
           displays.

  WHAT-IF  /api/whatif and /api/predict run the SAVED DEPLOYMENT MODEL, which
           was fitted on all eight seasons. For any historical game that model
           is in-sample, so its number is not a fair estimate of accuracy. It is
           the right tool for "what would this model say about a state that
           never happened" and the wrong tool for "how good is this model".

           Every response from those two endpoints carries an explicit caveat
           field saying so. The frontend renders it. A viewer should never have
           to guess which of the two numbers they are looking at.

WHY WHAT-IF TAKES A GAME AND AN EVENT
-------------------------------------
A win probability model needs all thirteen features. A caller who supplies only
a score margin would need the other twelve invented, and invented features
produce a confident number about nothing.

So /api/whatif takes a real (game_id, event_index), loads that event's ACTUAL
feature row, applies only the overrides the caller names, and predicts. The
untouched features stay real. /api/predict exists for the explicit case and
requires all thirteen by name, rejecting anything partial.
"""

import json
import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from src import config
from src.features import DERIVED_FEATURES, DERIVED_INPUTS, recompute_derived

logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

IN_SAMPLE_CAVEAT = (
    "This number comes from the deployment model, which was fitted on all "
    "eight seasons. For any game in this dataset that is in-sample, so it is "
    "not a fair estimate of accuracy. The timeline probabilities are out of "
    "fold; this one is not."
)

OUT_OF_FOLD_NOTE = (
    "Every probability in the timeline is out of fold: predicted by a model "
    "that never saw this game's season."
)


# ---------------------------------------------------------------------------
# Lazily loaded artefacts
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_metadata() -> dict:
    path = config.MODELS_DIR / "model_metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/11_train_model.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_model():
    import joblib
    path = config.MODELS_DIR / "win_probability_model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/11_train_model.py first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_feature_frame() -> pd.DataFrame:
    """
    The model frame, indexed by (game_id, event_index), for what-if lookups.

    Held in memory because scrubbing a timeline issues many small requests and
    re-reading a 309,000-row parquet per request would make the interface feel
    broken.
    """
    frame = pd.read_parquet(config.MODEL_FRAME_PARQUET)
    frame["game_id"] = (frame["game_id"].astype("string").str.strip()
                        .str.zfill(10))
    return frame.set_index(["game_id", "event_index"]).sort_index()


def feature_order() -> list:
    return list(load_metadata()["feature_order"])


def _feature_value(row: pd.DataFrame, name: str):
    """One feature off a single-row frame, as a float or None for a missing."""
    value = row.iloc[0][name]
    return None if pd.isna(value) else float(value)


def predict_rows(rows: pd.DataFrame) -> np.ndarray:
    """Probabilities for a frame that already carries every feature."""
    order = feature_order()
    missing = [c for c in order if c not in rows.columns]
    if missing:
        raise KeyError(f"missing features: {missing}")
    matrix = rows[order].astype(float).to_numpy()
    return load_model().predict_proba(matrix)[:, 1]


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------

def create_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field

    app = FastAPI(
        title="Celtics Live Win Probability API",
        description=(
            "Serves out-of-fold win probability replays for 636 Boston Celtics "
            "regular season games, 2016-17 to 2023-24, plus what-if queries "
            "against the saved deployment model."),
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    class WhatIf(BaseModel):
        game_id: str
        event_index: int
        overrides: dict = Field(
            default_factory=dict,
            description=("Feature names to replace on this event's REAL "
                         "feature row. Everything not named stays as it "
                         "actually was."))

    class ExplicitPredict(BaseModel):
        features: dict = Field(
            description="All thirteen features by name. Partial input is "
                        "rejected rather than filled in.")

    def serving_file(*parts):
        path = config.SERVING_DIR.joinpath(*parts)
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=(f"{path.name} not found. Run "
                        "scripts/20_build_serving.py to build the serving "
                        "data."))
        return path

    @app.get("/api/health")
    def health():
        index = config.SERVING_DIR / "index.json"
        model = config.MODELS_DIR / "win_probability_model.joblib"
        return {
            "status": "ok",
            "serving_data": index.exists(),
            "model": model.exists(),
            "games": (json.loads(index.read_text())["count"]
                      if index.exists() else 0),
            "note": OUT_OF_FOLD_NOTE,
        }

    @app.get("/api/model")
    def model_info():
        """
        What is actually deployed, including the out-of-fold metrics and the
        note that those metrics do not come from this model.
        """
        try:
            return load_metadata()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    @app.get("/api/games")
    def games():
        return FileResponse(serving_file("index.json"),
                            media_type="application/json")

    @app.get("/api/games/{game_id}")
    def game(game_id: str):
        game_id = game_id.strip().zfill(10)
        return FileResponse(serving_file("games", f"{game_id}.json"),
                            media_type="application/json")

    @app.get("/api/coverage")
    def coverage():
        """What display data is missing, measured in minutes rather than heads."""
        return FileResponse(serving_file("coverage.json"),
                            media_type="application/json")

    @app.post("/api/whatif")
    def whatif(request: WhatIf):
        """
        Re-predict a REAL event with some features replaced.

        The untouched features keep their real values, so the answer is about
        the change the caller named and not about twelve invented numbers.
        """
        try:
            frame = load_feature_frame()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        key = (request.game_id.strip().zfill(10), int(request.event_index))
        if key not in frame.index:
            raise HTTPException(
                status_code=404,
                detail=f"no event {key[1]} in game {key[0]}")

        row = frame.loc[[key]].copy()
        unknown = [k for k in request.overrides if k not in feature_order()]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=(f"not model features: {unknown}. Allowed: "
                        f"{feature_order()}"))
        for name, value in request.overrides.items():
            row[name] = float(value)

        # Two of the thirteen features are FUNCTIONS of the others:
        # margin_per_minute_remaining is margin over minutes left, and is_clutch
        # is the NBA definition, which tests the margin. Overriding the margin
        # and leaving those two alone builds a row that describes no possible
        # game, and the model answers a question nobody asked. Before this, a
        # real second-quarter event moved from -12 to -32 and the win
        # probability went UP, 31.9% to 32.5%; asking for +20 returned 21.8%.
        #
        # A feature the caller named explicitly is left exactly as they set it.
        # They asked for that value and are told which columns were rebuilt.
        before = {name: _feature_value(row, name) for name in DERIVED_FEATURES}
        row = recompute_derived(row, skip=set(request.overrides))
        recomputed = {
            name: {
                "from": before[name],
                "to": _feature_value(row, name),
                "because": [c for c in DERIVED_INPUTS[name]
                            if c in request.overrides],
            }
            for name in DERIVED_FEATURES
            if name not in request.overrides
            and before[name] != _feature_value(row, name)
        }

        try:
            probability = float(predict_rows(row)[0])
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        return {
            "game_id": key[0],
            "event_index": key[1],
            "probability": round(probability, 5),
            "overrides_applied": request.overrides,
            "derived_recomputed": recomputed,
            "features_used": {
                name: _feature_value(row, name)
                for name in feature_order()},
            "model": "deployment model, fitted on all eight seasons",
            "caveat": IN_SAMPLE_CAVEAT,
        }

    @app.post("/api/predict")
    def predict(request: ExplicitPredict):
        """
        Predict from an explicit, complete feature vector.

        Partial input is REJECTED. Filling the gaps with defaults would produce
        a confident number about a state nobody described.
        """
        order = feature_order()
        missing = [name for name in order if name not in request.features]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=(f"missing features: {missing}. All {len(order)} are "
                        "required; none are defaulted."))
        row = pd.DataFrame([{name: float(request.features[name])
                             for name in order}])
        try:
            probability = float(predict_rows(row)[0])
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        return {
            "probability": round(probability, 5),
            "model": "deployment model, fitted on all eight seasons",
            "caveat": IN_SAMPLE_CAVEAT,
        }

    return app


def main(host="127.0.0.1", port=8000, reload=False):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        raise SystemExit(
            "fastapi and uvicorn are not installed in this environment.\n"
            "In a Terminal, with your conda environment active, run:\n\n"
            "    pip install fastapi uvicorn\n")

    index = config.SERVING_DIR / "index.json"
    if not index.exists():
        raise SystemExit(
            f"{index} not found. Run scripts/20_build_serving.py first.")

    count = json.loads(index.read_text())["count"]
    print("=" * 70)
    print("CELTICS WIN PROBABILITY API")
    print("=" * 70)
    print(f"  games available : {count}")
    print(f"  serving data    : {config.SERVING_DIR}")
    print(f"  docs            : http://{host}:{port}/docs")
    print()
    print("  Timeline probabilities are OUT OF FOLD.")
    print("  /api/whatif and /api/predict use the deployment model, which is")
    print("  in-sample for these games. Both responses say so explicitly.")
    print()
    print("  Leave this running and start the frontend in another Terminal.")
    print("  Ctrl-C to stop.")
    print("=" * 70)

    import uvicorn
    uvicorn.run("src.api:create_app", host=host, port=port, reload=reload,
                factory=True)


if __name__ == "__main__":
    main()
