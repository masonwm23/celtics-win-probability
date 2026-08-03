"""
Phase 11: a READ-ONLY probe of NBA play-clip availability.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This measures whether video clips exist for individual plays, how well they can
be matched to our event table, and whether the URLs actually resolve. It writes
one report and one CSV. It changes NOTHING about the dashboard, the API, the
model, the serving data or any research artefact.

The dashboard must never depend on video, so nothing here is wired into it.

A FALSE NEGATIVE WORTH RECORDING
-------------------------------
The first run of this probe reported 0 clips out of 535 events across all
eight seasons, with zero request errors. That was wrong, and the way it was
wrong is worth keeping.

It asked `videoevents`. That endpoint returns HTTP 200 and a playlist entry
that is correct in every respect except that its videoUrls entry has no URL
fields at all, only nulls beside a placeholder uuid that is byte-identical for
every event in every game. The right endpoint is `videoeventsasset`, which
takes the same two parameters and returns three encodings, thumbnails and
caption tracks. Phase 11b established this by printing both raw bodies.

The lesson is not "check the endpoint name". It is that the failure produced a
clean, uniform, plausible-looking 0.0% rather than an error, and the probe as
first written stored a verdict rather than the raw response, so it could not
tell a real absence from a wrong address. build_report now refuses to present
a zero-with-no-errors as a result, and points at scripts/25_diagnose_video.py.

WHY THE MATCHING QUESTION IS THE HARD ONE
-----------------------------------------
The video endpoint is keyed by `game_event_id`, which is the play-by-play
EVENTNUM. In our tables that is `action_number`. Phase 2 established that
**action_number is NOT unique within a game**: one game carried 521 events with
duplicate action numbers, and sorting on it corrupted the clock.

That means a clip requested by action number can be ambiguous, and a clip shown
against the wrong play would be worse than showing no clip at all. So this probe
does not merely ask "did a clip come back". For every clip it checks:

  1. the returned game id equals the one requested;
  2. the returned event id equals the one requested;
  3. the returned description matches our stored description for that event.

A clip failing any of those is recorded as a MISMATCH, not as coverage. The
report separates the three so the failure mode is visible rather than averaged
away.

Sampling
--------
Three games per season, chosen at fixed positions in the season by date (first
quarter, middle, third quarter), so the sample is reproducible and spans all
eight seasons rather than clustering in the recent, better-covered ones.

Within a game the events are sampled STRATIFIED BY ACTION TYPE, up to two per
type, because coverage almost certainly differs between a made three and a
substitution and a single blended percentage would hide that.

Outputs
-------
reports/video_probe.txt
data/interim/video_probe.csv
"""

import logging
import re
import time
from collections import Counter, defaultdict

import pandas as pd

from src import config
from src.nba_client import call_endpoint

logger = logging.getLogger(__name__)

GAMES_PER_SEASON = 3
EVENTS_PER_TYPE = 2

# Where in each season the sampled games sit, as fractions through the schedule.
# Fixed so a re-run probes the same games.
SEASON_POSITIONS = (0.25, 0.50, 0.75)

# A subsample of returned URLs gets a HEAD request to see whether it actually
# resolves. Kept small because this is a probe, not a download. Spread evenly
# across seasons by sample_for_playback, because clip retention almost
# certainly differs between 2016-17 and 2023-24.
PLAYBACK_CHECKS = 40

REQUEST_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"),
    "Referer": "https://www.nba.com/",
}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_games(events: pd.DataFrame) -> pd.DataFrame:
    """
    Three games per season at fixed positions through the schedule.

    Deterministic on purpose: a probe whose sample moves between runs cannot be
    compared with its own earlier result.
    """
    rows = []
    per_game = (events.groupby(["season", "game_id"])["game_date"].min()
                .reset_index().sort_values(["season", "game_date", "game_id"]))
    for season, group in per_game.groupby("season"):
        group = group.reset_index(drop=True)
        n = len(group)
        for position in SEASON_POSITIONS:
            index = min(int(round(position * (n - 1))), n - 1)
            row = group.loc[index]
            rows.append({"season": season, "game_id": row["game_id"],
                         "game_date": row["game_date"],
                         "position_in_season": position})
    return pd.DataFrame(rows).drop_duplicates("game_id").reset_index(drop=True)


def sample_events(game_events: pd.DataFrame) -> pd.DataFrame:
    """
    Up to two events per action type.

    Blending a made shot and a substitution into one coverage percentage would
    hide the only distinction that matters for a video feature.
    """
    picked = []
    for action_type, group in game_events.groupby("action_type"):
        if not str(action_type).strip():
            continue
        group = group.sort_values("event_index")
        picked.append(group.head(EVENTS_PER_TYPE))
    if not picked:
        return game_events.head(0)
    return pd.concat(picked).sort_values("event_index")


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def normalise_description(text) -> str:
    """Lowercase, strip punctuation and collapse whitespace, for comparison only."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def description_agreement(ours, theirs) -> float:
    """
    Share of our description's words that appear in theirs.

    Deliberately crude. It is not trying to prove the descriptions are the same
    sentence; it is trying to catch a clip that is plainly about a different
    play, which is the failure that matters.
    """
    a = set(normalise_description(ours).split())
    b = set(normalise_description(theirs).split())
    if not a:
        return 0.0
    return len(a & b) / len(a)


AGREEMENT_THRESHOLD = 0.5


def classify(requested_game, requested_event, payload, our_description) -> dict:
    """
    Turn one endpoint response into a verdict.

    Statuses:
      no_clip     the endpoint returned nothing playable
      matched     game id, event id and description all agree
      mismatch    a clip came back but it is not demonstrably this play
    """
    result = {
        "status": "no_clip",
        "returned_game_id": None,
        "returned_event_id": None,
        "returned_description": None,
        "description_agreement": None,
        "url": None,
        "url_small": None,
        "url_medium": None,
        "url_large": None,
        "thumbnail": None,
        "captions": None,
        "duration": None,
    }
    if not payload:
        return result

    urls = (payload.get("resultSets", {}) or {}).get("Meta", {}) \
        .get("videoUrls", []) or []
    playlist = (payload.get("resultSets", {}) or {}).get("playlist", []) or []
    if not urls or not playlist:
        return result

    entry = urls[0] or {}
    play = playlist[0] or {}

    # Medium first. 960x540 is the size a dashboard panel would actually use,
    # so that is the one whose reliability is worth measuring. The others are
    # recorded but not tested.
    url = entry.get("murl") or entry.get("lurl") or entry.get("surl")
    if not url:
        # This is the videoevents stub: a well-formed response with every URL
        # field null. No URL means no clip, whatever else came back.
        return result

    result.update({
        "url": url,
        "url_small": entry.get("surl"),
        "url_medium": entry.get("murl"),
        "url_large": entry.get("lurl"),
        "thumbnail": entry.get("mth") or entry.get("sth"),
        "captions": entry.get("vtt"),
        "duration": entry.get("mdur") or entry.get("ldur") or entry.get("sdur"),
        "returned_game_id": str(play.get("gi") or ""),
        "returned_event_id": play.get("ei"),
        "returned_description": play.get("dsc"),
    })
    agreement = description_agreement(our_description, play.get("dsc"))
    result["description_agreement"] = round(agreement, 3)

    same_game = str(play.get("gi") or "").zfill(10) == str(requested_game).zfill(10)
    same_event = str(play.get("ei")) == str(requested_event)
    if same_game and same_event and agreement >= AGREEMENT_THRESHOLD:
        result["status"] = "matched"
    else:
        result["status"] = "mismatch"
        reasons = []
        if not same_game:
            reasons.append("game id differs")
        if not same_event:
            reasons.append("event id differs")
        if agreement < AGREEMENT_THRESHOLD:
            reasons.append("description disagrees")
        result["mismatch_reason"] = "; ".join(reasons)
    return result


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------

def fetch_event(game_id: str, action_number: int):
    """
    Ask videoeventsASSET, not videoevents.

    The first run of this probe used `videoevents` and returned 0 clips out of
    535 across all eight seasons. That endpoint answers HTTP 200 with a
    playlist entry that is correct in every respect except that its videoUrls
    entry carries no URL fields at all: no surl, no murl, no lurl, only nulls
    and a placeholder uuid that is byte-identical for every event in every
    game. It is a stub.

    `videoeventsasset` takes the same two parameters and returns the real
    thing: three encodings, thumbnails and caption tracks. Phase 11b confirmed
    this on a 2016-17 game and a 2023-24 game, six events, all six populated.

    Keeping this comment because the failure mode is the interesting part: the
    wrong endpoint did not error, did not 403 and did not return an empty
    playlist. It returned a well-formed answer that was silently empty where it
    mattered, and the resulting 0.0% looked exactly like a finding.
    """
    from nba_api.stats.endpoints import videoeventsasset
    endpoint = call_endpoint(videoeventsasset.VideoEventsAsset,
                             game_id=str(game_id),
                             game_event_id=int(action_number))
    return endpoint.get_dict()


def sample_for_playback(playable: pd.DataFrame, limit: int) -> pd.DataFrame:
    """
    An even spread of clips across seasons, up to `limit`.

    Whether an eight-year-old clip still resolves is a different question from
    whether last season's does, and it is the question that decides how much of
    the dashboard a video panel could ever cover.
    """
    if playable.empty or limit <= 0:
        return playable.head(0)
    seasons = sorted(playable["season"].unique())
    per_season = max(1, limit // len(seasons))
    picked = pd.concat([playable.loc[playable["season"].eq(season)]
                        .head(per_season) for season in seasons])
    if len(picked) < limit:                 # top up from whatever is left
        remainder = playable.drop(index=picked.index)
        picked = pd.concat([picked, remainder.head(limit - len(picked))])
    return picked.head(limit).sort_index()


def check_playback(url: str) -> dict:
    """A HEAD request. Reports what happened; never raises."""
    import requests
    try:
        response = requests.head(url, headers=REQUEST_HEADERS, timeout=15,
                                 allow_redirects=True)
        return {
            "http_status": response.status_code,
            "content_type": response.headers.get("Content-Type"),
            "content_length": response.headers.get("Content-Length"),
            "error": None,
        }
    except Exception as exc:                       # noqa: BLE001
        return {"http_status": None, "content_type": None,
                "content_length": None, "error": f"{type(exc).__name__}: {exc}"}


def run_probe() -> pd.DataFrame:
    events = pd.read_parquet(config.EVENTS_PARQUET)
    events["game_id"] = events["game_id"].astype(str).str.zfill(10)

    games = sample_games(events)
    logger.info("probing %d games across %d seasons",
                len(games), games["season"].nunique())

    # Duplicate action numbers make a clip request ambiguous. Phase 2 found one
    # game with 521 of them, so this is measured rather than hoped about.
    duplicate_counts = (events.groupby("game_id")["action_number"]
                        .apply(lambda s: int(s.duplicated().sum())))

    records = []
    for _, game in games.iterrows():
        game_id = game["game_id"]
        game_events = events.loc[events["game_id"].eq(game_id)]
        chosen = sample_events(game_events)
        duplicated = set(game_events.loc[
            game_events["action_number"].duplicated(keep=False), "action_number"])

        for event in chosen.itertuples():
            try:
                payload = fetch_event(game_id, event.action_number)
                error = None
            except Exception as exc:               # noqa: BLE001
                payload, error = None, f"{type(exc).__name__}: {exc}"

            verdict = classify(game_id, event.action_number, payload,
                               event.description)
            records.append({
                "season": game["season"],
                "game_id": game_id,
                "game_date": str(pd.Timestamp(game["game_date"]).date()),
                "event_index": event.event_index,
                "action_number": event.action_number,
                "action_type": event.action_type,
                "period": event.period,
                "our_description": event.description,
                "action_number_is_duplicated": event.action_number in duplicated,
                "duplicates_in_game": int(duplicate_counts.get(game_id, 0)),
                "request_error": error,
                **verdict,
            })
        logger.info("  %s %s: %d events probed", game["season"], game_id,
                    len(chosen))

    frame = pd.DataFrame(records)

    # Playback: a subsample of the matched clips, SPREAD ACROSS SEASONS.
    #
    # Taking the first N rows would have taken them all from 2016-17, because
    # the frame is built in season order. If old clips have been pruned and
    # recent ones have not, that is exactly the pattern a head() sample would
    # report as "everything is broken" and a tail() sample would report as
    # "everything is fine".
    playable = frame.loc[frame["status"].eq("matched") & frame["url"].notna()]
    sample = sample_for_playback(playable, PLAYBACK_CHECKS)
    logger.info("checking %d clip URLs actually resolve, spread over %d seasons",
                len(sample), sample["season"].nunique() if len(sample) else 0)
    playback = {}
    for row in sample.itertuples():
        playback[row.Index] = check_playback(row.url)
        time.sleep(0.2)
    for column in ("http_status", "content_type", "content_length", "error"):
        frame[f"playback_{column}"] = frame.index.map(
            lambda i: playback.get(i, {}).get(column))

    return frame


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(frame: pd.DataFrame) -> str:
    total = len(frame)
    matched = int(frame["status"].eq("matched").sum())
    mismatched = int(frame["status"].eq("mismatch").sum())
    none = int(frame["status"].eq("no_clip").sum())
    errors = int(frame["request_error"].notna().sum())

    lines = [
        "=" * 78,
        "PHASE 11 - VIDEO COVERAGE PROBE (READ ONLY)",
        "=" * 78,
        "",
        "  This probe changed NOTHING. No app code, no API, no model, no",
        "  serving data, no research artefact was touched. It exists to answer",
        "  whether a video feature is worth building at all.",
        "",
        f"  events probed          {total:,}",
        f"  games                  {frame['game_id'].nunique()}",
        f"  seasons                {frame['season'].nunique()}",
        "",
        f"  MATCHED   {matched:>5,}  ({matched / total:.1%})  clip returned AND "
        "game id, event id and description all agree",
        f"  MISMATCH  {mismatched:>5,}  ({mismatched / total:.1%})  a clip came "
        "back but is not demonstrably this play",
        f"  NO CLIP   {none:>5,}  ({none / total:.1%})",
        f"  ERRORS    {errors:>5,}  request failed after retries",
        "",
        "  A MISMATCH is not coverage. Showing a clip of the wrong play would",
        "  be worse than showing none, so the two are never added together.",
        "",
    ]

    # A blocked or rate-limited endpoint returns nothing for every request,
    # which reads identically to "the NBA has no clips for these plays". Those
    # are completely different answers, so the difference is stated rather than
    # left for the reader to infer from the ERRORS line.
    if total and errors / total >= 0.5:
        lines += [
            "  !! WARNING: " + f"{errors / total:.0%} of requests FAILED.",
            "",
            "  This report does not measure clip availability. It measures a",
            "  broken connection to the endpoint. A blocked, rate-limited or",
            "  unreachable endpoint produces exactly the same NO CLIP counts",
            "  as an NBA that genuinely has no video, and those are different",
            "  answers. Read the error examples below before concluding",
            "  anything about coverage.",
            "",
        ]
        for row in frame.loc[frame["request_error"].notna()].head(3).itertuples():
            lines.append(f"    {row.game_id} event {row.action_number}: "
                         f"{str(row.request_error)[:90]}")
        lines.append("")

    # The blind spot the first run of this probe walked straight into. Zero
    # clips with zero errors READS like a finding and is not one. An endpoint
    # that answers 200 with an empty body for every request produces exactly
    # this, and so does a response shape this parser does not recognise. The
    # probe stores a verdict rather than the raw response, so it cannot tell
    # those apart, and it must say so rather than let the 0.0% be quoted.
    if total and none == total and errors == 0:
        lines += [
            "  !! WARNING: every request SUCCEEDED and every one returned",
            "  nothing. That is not evidence that there is no video.",
            "",
            "  A clean zero across every season and every event type is what",
            "  you also get from asking the wrong endpoint, or from a response",
            "  shape this parser does not recognise. This probe records a",
            "  verdict, not the raw body, so it cannot distinguish those from",
            "  a genuine absence of clips.",
            "",
            "  Run scripts/25_diagnose_video.py before quoting the 0.0%.",
            "",
        ]

    lines += [
        "=" * 78,
        "COVERAGE BY SEASON",
        "=" * 78,
        "",
        f"  {'season':<10}{'probed':>8}{'matched':>9}{'mismatch':>10}"
        f"{'no clip':>9}{'match rate':>12}",
    ]
    for season, group in frame.groupby("season"):
        m = int(group["status"].eq("matched").sum())
        lines.append(
            f"  {season:<10}{len(group):>8}{m:>9}"
            f"{int(group['status'].eq('mismatch').sum()):>10}"
            f"{int(group['status'].eq('no_clip').sum()):>9}"
            f"{m / len(group):>11.1%}")

    lines += [
        "",
        "=" * 78,
        "COVERAGE BY EVENT TYPE",
        "=" * 78,
        "",
        "  The number that decides the feature. If clips exist only for made",
        "  shots, the panel is empty most of the time.",
        "",
        f"  {'action type':<24}{'probed':>8}{'matched':>9}{'match rate':>12}",
    ]
    for action_type, group in frame.groupby("action_type"):
        m = int(group["status"].eq("matched").sum())
        lines.append(f"  {str(action_type)[:23]:<24}{len(group):>8}{m:>9}"
                     f"{m / len(group):>11.1%}")

    lines += ["", "=" * 78, "EVENT MATCHING", "=" * 78, ""]
    if mismatched:
        reasons = Counter(frame.loc[frame["status"].eq("mismatch"),
                                    "mismatch_reason"].dropna())
        lines.append("  Why clips failed to match:")
        for reason, count in reasons.most_common():
            lines.append(f"    {count:>4}  {reason}")
        lines.append("")
        lines.append("  Examples:")
        for row in frame.loc[frame["status"].eq("mismatch")].head(5).itertuples():
            lines.append(f"    {row.game_id} event {row.action_number}")
            lines.append(f"      ours   : {str(row.our_description)[:70]}")
            lines.append(f"      theirs : {str(row.returned_description)[:70]}")
    else:
        lines.append("  No mismatches. Every clip returned agreed with our")
        lines.append("  record on game, event and description.")

    duplicated = frame.loc[frame["action_number_is_duplicated"]]
    lines += [
        "",
        "  ACTION NUMBER AMBIGUITY. Phase 2 found action_number is not unique",
        "  within a game, and the video endpoint is keyed by it.",
        f"    events probed whose action number is duplicated: {len(duplicated)}",
    ]
    if len(duplicated):
        m = int(duplicated["status"].eq("matched").sum())
        lines.append(f"    of those, matched: {m} ({m / len(duplicated):.0%})")
        lines.append("    This is the group where a wrong clip is most likely.")

    lines += ["", "=" * 78, "PLAYBACK RELIABILITY", "=" * 78, ""]
    checked = frame.loc[frame["playback_http_status"].notna()
                        | frame["playback_error"].notna()]
    if not len(checked):
        lines.append("  No URLs were checked, because nothing matched.")
    else:
        statuses = Counter(checked["playback_http_status"].dropna().astype(int))
        lines.append(f"  URLs checked: {len(checked)}")
        for status, count in sorted(statuses.items()):
            lines.append(f"    HTTP {status}: {count}")
        failed = checked.loc[checked["playback_error"].notna()]
        if len(failed):
            lines.append(f"    network errors: {len(failed)}")
            for row in failed.head(3).itertuples():
                lines.append(f"      {row.playback_error}")
        sizes = pd.to_numeric(checked["playback_content_length"],
                              errors="coerce").dropna()
        if len(sizes):
            lines.append(f"    median clip size: {sizes.median() / 1e6:.1f} MB")
        types = Counter(checked["playback_content_type"].dropna())
        for content_type, count in types.most_common(3):
            lines.append(f"    {content_type}: {count}")

        # An old clip that has been pruned and a new one that plays are the
        # difference between a feature that works on 636 games and one that
        # works on the last two seasons. A single pooled percentage hides it.
        lines += ["", f"  {'season':<10}{'checked':>9}{'HTTP 200':>10}"
                      f"{'ok rate':>10}"]
        for season, group in checked.groupby("season"):
            ok = int(pd.to_numeric(group["playback_http_status"],
                                   errors="coerce").eq(200).sum())
            lines.append(f"  {season:<10}{len(group):>9}{ok:>10}"
                         f"{ok / len(group):>9.0%}")

    lines += [
        "",
        "=" * 78,
        "WHAT THIS DOES NOT TELL YOU",
        "=" * 78,
        "",
        "  These are NBA-hosted clips. Playing them locally for a directed",
        "  research project is one thing; a paper or public demo that embeds",
        "  them is another. That is a question for Prof. Namini, not a",
        "  technical one, and this probe does not answer it.",
        "",
        "  A HEAD request resolving is not proof a browser will play the clip.",
        "  Hotlink protection, expiring URLs and geographic restriction would",
        "  all show as a clean 200 here and fail on screen.",
        "",
        "  Nothing was changed. The dashboard does not know this probe exists.",
        "",
        "=" * 78,
    ]
    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    config.ensure_dirs()

    frame = run_probe()
    out_csv = config.INTERIM_DIR / "video_probe.csv"
    frame.to_csv(out_csv, index=False)

    report = build_report(frame)
    print(report)
    out = config.REPORTS_DIR / "video_probe.txt"
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nRow-level detail: {out_csv}")
    print(f"Report saved to : {out}")
    print()
    print("READ ONLY. Nothing in the app, the API, the model or the research")
    print("outputs was modified. Stopping here for review.")
    return frame


if __name__ == "__main__":
    main()
