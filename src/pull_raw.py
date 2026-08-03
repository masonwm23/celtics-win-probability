"""
Phase 1, step 5: pull raw play-by-play and boxscore for every game.

This is the long job. For each game in the index it fetches two endpoints and
saves each response to disk as JSON, byte-for-byte as the API returned it.
Nothing is parsed, filtered, or reshaped here. Parsing happens in Phase 2, from
these cached files, so re-running the analysis never re-hits the API.

Design decisions that matter
----------------------------
Resumable. If a game's JSON already exists and loads cleanly, it is skipped.
You can stop this script and restart it without losing work or duplicating
calls.

Both teams are saved. The boxscore contains Boston's players AND the
opponent's. Filtering to Boston at download time would throw away the opponent
roster data the matchup context needs, and would mean re-pulling 636 games
later. So the full payload is kept.

Failures are recorded, not hidden. Every game gets a manifest row with a status.
A game that fails all retries is written to the manifest as failed, with the
error, and the script keeps going. At the end it prints the failures explicitly.
An empty result is never treated as success.

Schema is observed, not assumed. The smoke test prints the actual structure of
the first response, including which periods appear and how many events. The
Phase 2 parser gets written against what the API really returns rather than
against a guess.

Outputs
-------
data/raw/playbyplay/{game_id}.json
data/raw/boxscore/{game_id}.json
data/raw/raw_pull_manifest.csv
reports/phase1_raw_pull_summary.txt
reports/phase1_schema_probe.txt   (smoke test only)
"""

import json
import logging
import time
from datetime import datetime, timezone

import pandas as pd

from src import config
from src.nba_client import call_endpoint, NBARequestError

logger = logging.getLogger(__name__)

# Periods 1 through 14 covers regulation plus ten overtimes, which is more than
# any NBA game has ever needed. Passing an explicit range avoids relying on an
# endpoint default that might return only the first quarter.
PBP_START_PERIOD = 1
PBP_END_PERIOD = 14


# ---------------------------------------------------------------------------
# Disk helpers
# ---------------------------------------------------------------------------

def _path_for(kind: str, game_id: str):
    directory = config.RAW_PBP_DIR if kind == "pbp" else config.RAW_BOX_DIR
    return directory / f"{game_id}.json"


def _load_cached(kind: str, game_id: str):
    """
    Return cached JSON for this game, or None if absent or unreadable.

    A file that exists but does not parse is treated as absent, so a run that
    was killed mid-write gets repaired rather than silently trusted.
    """
    path = _path_for(kind, game_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        logger.warning("Cached %s for %s is unreadable, will re-fetch",
                       kind, game_id)
        return None


def _save(kind: str, game_id: str, payload: dict):
    """Write JSON atomically, so an interrupted write cannot leave a half file."""
    path = _path_for(kind, game_id)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Payload inspection
# ---------------------------------------------------------------------------

def count_pbp_events(payload: dict):
    """
    Count play-by-play rows and report which periods appear.

    Written defensively because the V3 response shape is being confirmed, not
    assumed. Returns (n_events, sorted_periods, note).
    """
    try:
        actions = payload["game"]["actions"]
    except (KeyError, TypeError):
        # Fall back to the older resultSets shape if the response differs.
        rs = payload.get("resultSets") or payload.get("resultSet")
        if isinstance(rs, list) and rs:
            headers = rs[0].get("headers", [])
            rows = rs[0].get("rowSet", [])
            periods = []
            if "PERIOD" in headers:
                pi = headers.index("PERIOD")
                periods = sorted({r[pi] for r in rows if r[pi] is not None})
            return len(rows), periods, "resultSets shape"
        return 0, [], "UNRECOGNISED SHAPE"

    periods = sorted({a.get("period") for a in actions
                      if a.get("period") is not None})
    return len(actions), periods, "game.actions shape"


def count_box_players(payload: dict):
    """Count player rows in a boxscore payload and report the shape found."""
    try:
        box = payload["boxScoreTraditional"]
        home = box["homeTeam"]["players"]
        away = box["awayTeam"]["players"]
        return len(home) + len(away), "boxScoreTraditional shape"
    except (KeyError, TypeError):
        rs = payload.get("resultSets")
        if isinstance(rs, list) and rs:
            return len(rs[0].get("rowSet", [])), "resultSets shape"
        return 0, "UNRECOGNISED SHAPE"


# ---------------------------------------------------------------------------
# Per game fetch
# ---------------------------------------------------------------------------

def fetch_one_game(game_id: str, force: bool = False):
    """
    Fetch (or load from cache) both endpoints for one game.

    Returns a dict of manifest fields. Never raises for a network failure. The
    failure is recorded in the returned row instead, so one bad game cannot
    abort a 636 game run.
    """
    row = {
        "game_id": game_id,
        "pbp_status": "", "pbp_events": 0, "pbp_periods": "", "pbp_shape": "",
        "box_status": "", "box_players": 0, "box_shape": "",
        "error": "",
    }

    # --- play by play ---
    payload = None if force else _load_cached("pbp", game_id)
    if payload is not None:
        row["pbp_status"] = "cached"
    else:
        try:
            from nba_api.stats.endpoints import playbyplayv3
            endpoint = call_endpoint(
                playbyplayv3.PlayByPlayV3,
                game_id=game_id,
                start_period=PBP_START_PERIOD,
                end_period=PBP_END_PERIOD,
            )
            payload = endpoint.get_dict()
            _save("pbp", game_id, payload)
            row["pbp_status"] = "fetched"
        except NBARequestError as exc:
            row["pbp_status"] = "FAILED"
            row["error"] += f"pbp: {exc} | "
        except Exception as exc:  # noqa: BLE001
            row["pbp_status"] = "FAILED"
            row["error"] += f"pbp: {type(exc).__name__}: {exc} | "

    if payload is not None:
        n, periods, shape = count_pbp_events(payload)
        row["pbp_events"] = n
        row["pbp_periods"] = ",".join(str(p) for p in periods)
        row["pbp_shape"] = shape
        if n == 0:
            row["pbp_status"] = "EMPTY"
            row["error"] += "pbp: zero events returned | "

    # --- boxscore ---
    payload = None if force else _load_cached("box", game_id)
    if payload is not None:
        row["box_status"] = "cached"
    else:
        try:
            from nba_api.stats.endpoints import boxscoretraditionalv3
            endpoint = call_endpoint(
                boxscoretraditionalv3.BoxScoreTraditionalV3,
                game_id=game_id,
            )
            payload = endpoint.get_dict()
            _save("box", game_id, payload)
            row["box_status"] = "fetched"
        except NBARequestError as exc:
            row["box_status"] = "FAILED"
            row["error"] += f"box: {exc} | "
        except Exception as exc:  # noqa: BLE001
            row["box_status"] = "FAILED"
            row["error"] += f"box: {type(exc).__name__}: {exc} | "

    if payload is not None:
        n, shape = count_box_players(payload)
        row["box_players"] = n
        row["box_shape"] = shape
        if n == 0:
            row["box_status"] = "EMPTY"
            row["error"] += "box: zero player rows returned | "

    row["error"] = row["error"].strip(" |")
    return row


# ---------------------------------------------------------------------------
# Schema probe, run during the smoke test only
# ---------------------------------------------------------------------------

def write_schema_probe(game_id: str):
    """
    Dump the real structure of one game's two payloads to a report.

    This exists so the Phase 2 parser is written against observed field names
    instead of assumed ones. It prints top level keys, the keys of one event,
    and the keys of one player row.
    """
    lines = [
        "=" * 70,
        "SCHEMA PROBE",
        f"Game ID: {game_id}",
        f"Run at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "=" * 70,
        "",
        "Purpose: record the ACTUAL response structure so the Phase 2 parser",
        "is written against observed field names, not assumed ones.",
        "",
    ]

    pbp = _load_cached("pbp", game_id)
    lines.append("PLAY BY PLAY (PlayByPlayV3)")
    if pbp is None:
        lines.append("  no cached payload")
    else:
        lines.append(f"  top level keys: {sorted(pbp.keys())}")
        n, periods, shape = count_pbp_events(pbp)
        lines.append(f"  detected shape: {shape}")
        lines.append(f"  event count   : {n}")
        lines.append(f"  periods seen  : {periods}")
        actions = (pbp.get("game") or {}).get("actions")
        if actions:
            lines.append(f"  event field names ({len(actions[0])} fields):")
            for k in sorted(actions[0].keys()):
                lines.append(f"    {k}")
            lines.append("")
            lines.append("  first 3 events, raw:")
            for a in actions[:3]:
                lines.append(f"    {json.dumps(a)[:300]}")

    lines.append("")
    box = _load_cached("box", game_id)
    lines.append("BOXSCORE (BoxScoreTraditionalV3)")
    if box is None:
        lines.append("  no cached payload")
    else:
        lines.append(f"  top level keys: {sorted(box.keys())}")
        n, shape = count_box_players(box)
        lines.append(f"  detected shape : {shape}")
        lines.append(f"  player rows    : {n}")
        try:
            bt = box["boxScoreTraditional"]
            lines.append(f"  boxScoreTraditional keys: {sorted(bt.keys())}")
            for side in ("homeTeam", "awayTeam"):
                team = bt[side]
                lines.append(f"  {side}: "
                             f"{team.get('teamTricode')} "
                             f"({len(team.get('players', []))} players)")
            p = bt["homeTeam"]["players"][0]
            lines.append("")
            lines.append(f"  player field names ({len(p)} fields):")
            for k in sorted(p.keys()):
                lines.append(f"    {k}")
            lines.append("")
            lines.append("  first player row, raw:")
            lines.append(f"    {json.dumps(p)[:600]}")
        except (KeyError, TypeError, IndexError) as exc:
            lines.append(f"  could not walk structure: {type(exc).__name__}")

    report = "\n".join(lines)
    out = config.REPORTS_DIR / "phase1_schema_probe.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nSchema probe saved to: {out}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def sample_across_seasons(index: pd.DataFrame, limit: int) -> pd.DataFrame:
    """
    Take roughly equal numbers of games from each season, up to `limit` total.

    Why not just the first N games: the NBA's play-by-play format changed over
    the 2016-2024 window. A sample drawn only from 2016-17 would not reveal a
    2023-24 shape difference, so the smoke test has to span all eight seasons.
    """
    n_seasons = index["SEASON"].nunique()
    per_season = max(1, limit // n_seasons)
    sampled = index.groupby("SEASON", group_keys=False).head(per_season)
    return (sampled.sort_values(["GAME_DATE", "GAME_ID"])
                   .head(limit)
                   .reset_index(drop=True))


def run(limit=None, force=False, probe=False):
    """
    Pull raw data for games in the index.

    Parameters
    ----------
    limit : int or None
        Process only the first N games. Used for the smoke test.
    force : bool
        Ignore the cache and re-fetch.
    probe : bool
        Write a schema probe report from the first game processed.
    """
    config.ensure_dirs()

    if not config.GAME_INDEX_CSV.exists():
        raise FileNotFoundError(
            f"{config.GAME_INDEX_CSV} not found. Run the game index pull first."
        )

    index = pd.read_csv(config.GAME_INDEX_CSV, parse_dates=["GAME_DATE"])
    index["GAME_ID"] = index["GAME_ID"].astype(str).str.zfill(10)

    if limit is not None:
        index = sample_across_seasons(index, limit)

    total = len(index)
    print(f"Games to process: {total}")
    print(f"Cache: {'IGNORED (force)' if force else 'used where valid'}")
    print(f"Raw play-by-play dir: {config.RAW_PBP_DIR}")
    print(f"Raw boxscore dir    : {config.RAW_BOX_DIR}")
    print()

    rows = []
    t_start = time.time()
    first_game_id = None

    for i, rec in enumerate(index.itertuples(index=False), start=1):
        game_id = rec.GAME_ID
        if first_game_id is None:
            first_game_id = game_id

        row = fetch_one_game(game_id, force=force)
        row.update({
            "season": rec.SEASON,
            "game_date": rec.GAME_DATE.date().isoformat(),
            "matchup": rec.MATCHUP,
            "opponent": rec.OPPONENT_ABBREV,
        })
        rows.append(row)

        bad = ("FAILED" in (row["pbp_status"], row["box_status"])
               or "EMPTY" in (row["pbp_status"], row["box_status"]))
        if bad or i % 25 == 0 or i == total:
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed else 0
            eta = (total - i) / rate if rate else 0
            flag = "  <-- PROBLEM" if bad else ""
            print(f"  [{i:>4}/{total}] {row['season']} {row['game_date']} "
                  f"{row['matchup']:<12} "
                  f"pbp={row['pbp_events']:>5} box={row['box_players']:>3} "
                  f"({elapsed/60:.1f}m elapsed, ETA {eta/60:.1f}m){flag}")
            if bad and row["error"]:
                print(f"         {row['error'][:200]}")

    manifest = pd.DataFrame(rows)
    ordered = ["game_id", "season", "game_date", "matchup", "opponent",
               "pbp_status", "pbp_events", "pbp_periods", "pbp_shape",
               "box_status", "box_players", "box_shape", "error"]
    manifest = manifest[ordered]

    # Merge with any existing manifest so a resumed run does not lose the
    # record of games processed in an earlier session.
    if config.RAW_MANIFEST_CSV.exists() and limit is None:
        try:
            prior = pd.read_csv(config.RAW_MANIFEST_CSV, dtype={"game_id": str})
            keep = prior.loc[~prior["game_id"].isin(manifest["game_id"])]
            manifest = pd.concat([keep, manifest], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not merge prior manifest: %s", exc)

    manifest = manifest.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    manifest.to_csv(config.RAW_MANIFEST_CSV, index=False)

    summary = build_summary(manifest, total, time.time() - t_start, limit)
    print(summary)
    out = config.REPORTS_DIR / "phase1_raw_pull_summary.txt"
    out.write_text(summary + "\n", encoding="utf-8")
    print(f"\nSummary saved to: {out}")
    print(f"Manifest saved to: {config.RAW_MANIFEST_CSV}")

    if probe and first_game_id:
        print("\n")
        write_schema_probe(first_game_id)

    n_problem = int(
        manifest["pbp_status"].isin(["FAILED", "EMPTY"]).sum()
        + manifest["box_status"].isin(["FAILED", "EMPTY"]).sum()
    )
    return n_problem == 0


def build_summary(manifest, processed, elapsed, limit):
    lines = [
        "",
        "=" * 70,
        "PHASE 1 RAW PULL SUMMARY",
        f"Run at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Mode: {'SMOKE TEST (limit=%d)' % limit if limit else 'FULL RUN'}",
        f"Games processed this run: {processed}",
        f"Elapsed: {elapsed/60:.1f} minutes",
        "=" * 70,
        "",
        f"Manifest rows total: {len(manifest)}",
        "",
        "Play-by-play status counts:",
    ]
    for status, n in manifest["pbp_status"].value_counts().items():
        lines.append(f"  {status:<10} {n}")
    lines.append("")
    lines.append("Boxscore status counts:")
    for status, n in manifest["box_status"].value_counts().items():
        lines.append(f"  {status:<10} {n}")

    lines.append("")
    lines.append("Response shapes observed (should be consistent):")
    for col in ("pbp_shape", "box_shape"):
        counts = manifest.loc[manifest[col].ne(""), col].value_counts()
        for shape, n in counts.items():
            lines.append(f"  {col}: {shape}  ({n} games)")

    ok = manifest.loc[manifest["pbp_events"] > 0, "pbp_events"]
    if len(ok):
        lines.append("")
        lines.append("Play-by-play event counts per game:")
        lines.append(f"  min {int(ok.min())}   median {int(ok.median())}   "
                     f"max {int(ok.max())}   total {int(ok.sum()):,}")
        lines.append("  A normal NBA game has roughly 400 to 600 logged events.")
        thin = manifest.loc[(manifest["pbp_events"] > 0)
                            & (manifest["pbp_events"] < 300)]
        if len(thin):
            lines.append(f"  {len(thin)} game(s) under 300 events, listed for review:")
            for _, r in thin.iterrows():
                lines.append(f"    {r['game_date']} {r['matchup']} "
                             f"{r['pbp_events']} events")

    box_ok = manifest.loc[manifest["box_players"] > 0, "box_players"]
    if len(box_ok):
        lines.append("")
        lines.append("Boxscore player rows per game (both teams):")
        lines.append(f"  min {int(box_ok.min())}   median {int(box_ok.median())}"
                     f"   max {int(box_ok.max())}")
        lines.append("  Both teams combined is normally in the high 20s to 30s.")

    # Period coverage. If this only ever shows 1, the period parameters are
    # wrong and we are silently downloading first quarters only.
    periods_seen = set()
    for val in manifest["pbp_periods"]:
        if isinstance(val, str) and val:
            periods_seen.update(int(p) for p in val.split(","))
    if periods_seen:
        lines.append("")
        lines.append(f"Periods present across games: {sorted(periods_seen)}")
        if max(periods_seen) < 4:
            lines.append("  WARNING: no game reaches period 4. The period")
            lines.append("  parameters are probably wrong. STOP and report.")
        else:
            lines.append("  Periods 1 to 4 plus any overtimes is correct.")

    problems = manifest.loc[
        manifest["pbp_status"].isin(["FAILED", "EMPTY"])
        | manifest["box_status"].isin(["FAILED", "EMPTY"])
    ]
    lines.append("")
    lines.append("=" * 70)
    if problems.empty:
        lines.append("RESULT: no failures and no empty payloads.")
    else:
        lines.append(f"RESULT: {len(problems)} game(s) need attention. "
                     f"Listed below and in the manifest.")
        for _, r in problems.iterrows():
            lines.append(f"  {r['game_date']} {r['matchup']} "
                         f"pbp={r['pbp_status']} box={r['box_status']}")
            if r["error"]:
                lines.append(f"    {str(r['error'])[:200]}")
        lines.append("")
        lines.append("Re-running this script retries only the missing games,")
        lines.append("because completed games are served from cache.")
    lines.append("=" * 70)
    return "\n".join(lines)


def main_smoke():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    return run(limit=16, probe=True)


def main_full():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    return run(limit=None, probe=False)


if __name__ == "__main__":
    main_smoke()
