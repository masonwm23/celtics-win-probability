"""
Phase 9b, step 0: report the schema of every table the dashboard will serve.

WHY THIS IS A SCRIPT AND NOT A GUESS
------------------------------------
The API layer joins five tables that were written across four phases: events,
rosters, lineups, the model frame and the out-of-fold predictions. Writing that
join against remembered column names is how a serving layer ends up silently
dropping a column or matching on the wrong key.

This prints what is actually on disk, so the serving code is written against
observed schemas. It reads nothing into a model and changes no file.

It also checks the join keys the dashboard depends on, because a key that looks
right and has a different dtype (string game ids against integer game ids, say)
produces an empty join rather than an error.
"""

import logging

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

TABLES = [
    ("events", lambda: config.EVENTS_PARQUET),
    ("rosters", lambda: config.ROSTERS_PARQUET),
    ("lineups", lambda: config.LINEUPS_PARQUET),
    ("model_frame", lambda: config.MODEL_FRAME_PARQUET),
    ("oof_predictions", lambda: config.PROCESSED_DIR / "oof_predictions.parquet"),
    ("game_index", lambda: config.GAME_INDEX_CSV),
    ("player_values", lambda: config.MODELS_DIR / "player_values.csv"),
    ("player_bios", lambda: config.RAW_DIR / "player_bios.csv"),
    ("opponent_strength",
     lambda: config.INTERIM_DIR / "opponent_strength.csv"),
]


def load_head(path, rows=3):
    if path.suffix == ".csv":
        return pd.read_csv(path, nrows=rows), None
    frame = pd.read_parquet(path)
    return frame.head(rows), len(frame)


def describe(name, path):
    if not path.exists():
        return {"name": name, "path": str(path), "exists": False}
    head, total = load_head(path)
    return {
        "name": name,
        "path": str(path),
        "exists": True,
        "rows": total,
        "columns": list(head.columns),
        "dtypes": {c: str(t) for c, t in head.dtypes.items()},
        "head": head,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("=" * 78)
    print("SERVING SCHEMA PROBE")
    print("=" * 78)
    print()
    print("  Reports what is actually on disk, so the API is written against")
    print("  observed schemas rather than remembered ones. Reads nothing into")
    print("  a model and changes no file.")
    print()

    reports = []
    for name, getter in TABLES:
        report = describe(name, getter())
        reports.append(report)
        print("-" * 78)
        if not report["exists"]:
            print(f"{name}: NOT FOUND at {report['path']}")
            continue
        count = f"{report['rows']:,} rows" if report["rows"] else "csv"
        print(f"{name}  ({count})")
        print(f"  {report['path']}")
        for column, dtype in report["dtypes"].items():
            print(f"    {column:<32}{dtype}")

    print("-" * 78)
    print()
    print("JOIN KEYS THE DASHBOARD DEPENDS ON")
    print()
    by_name = {r["name"]: r for r in reports if r["exists"]}
    for name in ("events", "rosters", "lineups", "model_frame",
                 "oof_predictions"):
        report = by_name.get(name)
        if not report:
            continue
        keys = [c for c in report["columns"]
                if c.lower() in {"game_id", "gameid", "person_id", "player_id",
                                 "personid", "event_index", "season"}]
        types = ", ".join(f"{k} ({report['dtypes'][k]})" for k in keys)
        print(f"  {name:<18}{types or 'NO RECOGNISED KEY'}")

    print()
    print("SAMPLE ROWS")
    for report in reports:
        if not report["exists"]:
            continue
        print()
        print(f"--- {report['name']} ---")
        with pd.option_context("display.max_columns", 50,
                               "display.width", 200):
            print(report["head"].to_string())

    return reports


if __name__ == "__main__":
    main()
