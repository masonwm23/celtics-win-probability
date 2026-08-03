"""
Phase 11b: find out WHY the video probe came back completely empty.

WHAT HAPPENED
-------------
The probe requested 535 events across 24 games and 8 seasons. Zero request
errors. Zero clips. Every single event, every season, every event type, 0.0%.

A uniform zero with zero errors has three possible causes and they are not the
same answer:

  A. the endpoint returned an empty body            -> the NBA really has no
                                                       clip at this address
  B. the endpoint returned a body in a shape my     -> my parser is wrong and
     parser does not recognise                         the clips were there
  C. `videoevents` is the wrong endpoint            -> ask the right one

The probe recorded a verdict but not the raw response, so it cannot tell the
three apart. That is a gap in the probe, and this script closes it by printing
exactly what came back, unparsed.

WHY C IS THE LEADING SUSPECT
----------------------------
nba_api exposes both `videoevents` and `videoeventsasset`. The nba.com site
itself calls videoeventsasset for a single play's clip. The probe used
videoevents. This script calls BOTH for the same events and prints both raw
bodies side by side, so the difference is observed rather than argued about.

It also issues one plain requests.get to stats.nba.com, bypassing nba_api
entirely, so an empty body can be distinguished from nba_api quietly discarding
something.

READ ONLY
---------
About 14 network calls. Writes one file, reports/video_diagnose.txt. Touches
nothing else.
"""

import json
import logging

import pandas as pd

from src import config
from src.nba_client import call_endpoint

logger = logging.getLogger(__name__)

# One recent game and one old one. If clips exist anywhere they exist in
# 2023-24; if they exist only there, that is itself the finding.
DIAGNOSTIC_GAMES = ["0022300906", "0021600311"]

# Made field goals only. A substitution having no clip proves nothing. A made
# shot in a nationally televised NBA game having no clip means the address is
# wrong.
EVENTS_PER_GAME = 3

BODY_PREVIEW = 1500


def pick_events(events: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """The made field goals most likely to have a highlight clip."""
    game = events.loc[events["game_id"].eq(game_id)]
    made = game.loc[game["action_type"].eq("Made Shot")]
    if made.empty:                       # fall back rather than return nothing
        made = game
    return made.sort_values("event_index").head(EVENTS_PER_GAME)


def describe_payload(payload) -> str:
    """What is actually in the response, before any interpretation."""
    if payload is None:
        return "      payload is None"
    if not isinstance(payload, dict):
        return f"      payload is {type(payload).__name__}, not a dict"

    lines = [f"      top level keys : {sorted(payload.keys())}"]
    results = payload.get("resultSets")
    lines.append(f"      resultSets type: {type(results).__name__}")
    if isinstance(results, dict):
        lines.append(f"      resultSets keys: {sorted(results.keys())}")
        meta = results.get("Meta")
        if isinstance(meta, dict):
            urls = meta.get("videoUrls") or []
            lines.append(f"      Meta.videoUrls : {len(urls)} entr(ies)")
            if urls:
                lines.append(f"        first: {json.dumps(urls[0])[:400]}")
        playlist = results.get("playlist") or []
        lines.append(f"      playlist       : {len(playlist)} entr(ies)")
        if playlist:
            lines.append(f"        first: {json.dumps(playlist[0])[:400]}")
    elif isinstance(results, list):
        lines.append(f"      resultSets len : {len(results)}")
        if results:
            lines.append(f"        first: {json.dumps(results[0])[:400]}")
    return "\n".join(lines)


def try_endpoint(endpoint_cls, game_id, action_number):
    try:
        endpoint = call_endpoint(endpoint_cls, game_id=str(game_id),
                                 game_event_id=int(action_number))
        try:
            url = endpoint.nba_response.get_url()
        except Exception:                          # noqa: BLE001
            url = "(url unavailable)"
        return {"ok": True, "url": url, "payload": endpoint.get_dict(),
                "raw": endpoint.nba_response.get_response(), "error": None}
    except Exception as exc:                       # noqa: BLE001
        return {"ok": False, "url": None, "payload": None, "raw": None,
                "error": f"{type(exc).__name__}: {exc}"}


def raw_http(game_id, action_number, endpoint_name="videoevents"):
    """
    One request with no nba_api in the way.

    If nba_api is discarding something, this is where it shows up, because the
    status code and the untouched body are both printed.
    """
    import requests
    from nba_api.stats.library.http import NBAStatsHTTP
    url = f"https://stats.nba.com/stats/{endpoint_name}"
    try:
        response = requests.get(
            url, params={"GameID": str(game_id),
                         "GameEventID": int(action_number)},
            headers=dict(NBAStatsHTTP.headers), timeout=config.REQUEST_TIMEOUT)
        return {"status": response.status_code, "url": response.url,
                "body": response.text, "error": None}
    except Exception as exc:                       # noqa: BLE001
        return {"status": None, "url": url, "body": "", "error":
                f"{type(exc).__name__}: {exc}"}


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    config.ensure_dirs()

    from nba_api.stats.endpoints import videoevents, videoeventsasset

    events = pd.read_parquet(config.EVENTS_PARQUET)
    events["game_id"] = events["game_id"].astype(str).str.zfill(10)

    lines = [
        "=" * 78,
        "PHASE 11b - WHY DID THE VIDEO PROBE RETURN NOTHING",
        "=" * 78,
        "",
        "  535 events, 24 games, 8 seasons, 0 errors, 0 clips. That pattern is",
        "  produced equally well by 'there is no video' and by 'we asked the",
        "  wrong address'. This prints the raw response so the two can be told",
        "  apart.",
        "",
    ]

    for game_id in DIAGNOSTIC_GAMES:
        chosen = pick_events(events, game_id)
        if chosen.empty:
            lines.append(f"  {game_id}: not in events.parquet, skipped")
            continue
        season = chosen["season"].iloc[0]
        lines += ["=" * 78, f"GAME {game_id}  ({season})", "=" * 78, ""]

        for event in chosen.itertuples():
            lines += [
                f"  event_index {event.event_index}  "
                f"action_number {event.action_number}  "
                f"[{event.action_type}]",
                f"    ours: {event.description}",
                "",
            ]
            for label, cls in (("videoevents", videoevents.VideoEvents),
                               ("videoeventsasset",
                                videoeventsasset.VideoEventsAsset)):
                result = try_endpoint(cls, game_id, event.action_number)
                lines.append(f"    --- {label} ---")
                if not result["ok"]:
                    lines.append(f"      FAILED: {result['error']}")
                    lines.append("")
                    continue
                lines.append(f"      url: {result['url']}")
                lines.append(describe_payload(result["payload"]))
                body = result["raw"]
                if isinstance(body, bytes):
                    body = body.decode("utf-8", "replace")
                lines.append(f"      raw body ({len(body or '')} chars), "
                             f"first {BODY_PREVIEW}:")
                lines.append("      " + (body or "")[:BODY_PREVIEW]
                             .replace("\n", " "))
                lines.append("")
            logger.info("  %s event %s probed on both endpoints",
                        game_id, event.action_number)

        # One raw call per game, nba_api removed from the picture entirely.
        first = chosen.iloc[0]
        for endpoint_name in ("videoevents", "videoeventsasset"):
            raw = raw_http(game_id, first["action_number"], endpoint_name)
            lines += [
                f"    --- raw requests.get, {endpoint_name}, no nba_api ---",
                f"      HTTP {raw['status']}   {raw['error'] or ''}",
                f"      {raw['url']}",
                f"      body ({len(raw['body'])} chars), first {BODY_PREVIEW}:",
                "      " + raw["body"][:BODY_PREVIEW].replace("\n", " "),
                "",
            ]
        logger.info("  %s raw calls done", game_id)

    lines += [
        "=" * 78,
        "HOW TO READ THIS",
        "=" * 78,
        "",
        "  If videoeventsasset returns a playlist and videoevents does not,",
        "  the probe asked the wrong endpoint and its result means nothing.",
        "  The probe gets fixed and re-run.",
        "",
        "  If BOTH return an empty playlist and the raw HTTP status is 200,",
        "  the clips genuinely are not addressable this way and the video",
        "  feature is not worth building from this source.",
        "",
        "  If the raw status is 403 or the body is not JSON, we are being",
        "  blocked, which is a third answer again and says nothing about",
        "  whether the clips exist.",
        "",
        "  Nothing was changed by this script.",
        "=" * 78,
    ]

    report = "\n".join(lines)
    print(report)
    out = config.REPORTS_DIR / "video_diagnose.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved to: {out}")
    return report


if __name__ == "__main__":
    main()
