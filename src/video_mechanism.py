"""
Phase 11e: why does every clip URL return the same file?

WHERE WE ARE
------------
Phase 11d hashed sixteen genuinely distinct clip URLs, spanning eight seasons,
eight different games, sixteen different uuids. Every one returned byte
identical content: same header hash, same hash 1 MB in, same total of
31,580,089 bytes, complete download.

So videos.nba.com is serving ONE file for every address we ask for. Before
declaring the idea dead, three things are worth knowing, because two of them
would change the answer.

THE THREE QUESTIONS, AND WHY EACH MATTERS
-----------------------------------------
1. IS IT A PLACEHOLDER?
   A URL is fabricated by taking a real one and replacing the uuid with
   nonsense. If that also returns the same 31.58 MB file, the server is
   answering everything with a default, our requests are not authorised, and
   the 200s mean nothing. If the fabricated URL 404s while real ones return
   the file, something stranger is going on and it is worth more digging.

2. ARE THE THUMBNAILS GATED TOO?
   Every matched clip carries a still image alongside the video, and all 336
   are distinct URLs. If the stills come back distinct while the videos do
   not, a version of the feature survives: the real still frame of the real
   play beside the probability line, which is most of the intent without the
   video pipeline. This is the question worth the most.

3. IS IT ONLY THE 960x540 ENCODING?
   The same play is published at 320x180 and 1280x720 as well. If the small
   encoding returns distinct bytes, the medium one is simply broken and the
   feature is fine.

Each URL is tried with three header sets, because the difference between them
is the difference between "blocked" and "broken":

  browser     what the probe has been sending, nba.com Referer
  bare        no Referer at all
  origin      Referer plus an explicit Origin, like a real page fetch

VERDICT RULES, written before the data
--------------------------------------
  placeholder confirmed  the fabricated URL returns the same hash as the real
                         ones. Nothing served from this host can be trusted to
                         be the play it claims.
  stills viable          the thumbnails return distinct hashes AND are real
                         images. A still-frame panel is buildable.
  small encoding viable  the 320x180 URLs return distinct hashes.

READ ONLY. About 30 small requests. Writes reports/video_mechanism.txt and
data/interim/video_mechanism.csv. Nothing else is touched.
"""

import hashlib
import logging
import re
import time

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

CHUNK = 4096
CLIPS_TO_TEST = 3

BASE_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")

HEADER_SETS = {
    "browser": {"User-Agent": BASE_AGENT, "Referer": "https://www.nba.com/"},
    "bare": {"User-Agent": BASE_AGENT},
    "origin": {"User-Agent": BASE_AGENT, "Referer": "https://www.nba.com/",
               "Origin": "https://www.nba.com"},
}

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG"


def digest(payload: bytes) -> str:
    return hashlib.md5(payload).hexdigest()[:16] if payload else ""


def fabricate(url: str) -> str:
    """
    A real URL with the uuid replaced by nonsense.

    Same host, same path shape, same encoding suffix. The only thing wrong with
    it is that it names a video that does not exist, which is exactly the
    control this needs.
    """
    return re.sub(
        r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_",
        "/00000000-0000-0000-0000-000000000000_", url, count=1)


def looks_like_image(payload: bytes) -> bool:
    return bool(payload) and (payload.startswith(JPEG_MAGIC)
                              or payload.startswith(PNG_MAGIC))


def looks_like_mp4(payload: bytes) -> bool:
    return bool(payload) and len(payload) >= 12 and b"ftyp" in payload[:64]


def fetch(url: str, header_set: str) -> dict:
    """One ranged GET, recording where it ended up. Never raises."""
    import requests
    headers = dict(HEADER_SETS[header_set])
    headers["Range"] = f"bytes=0-{CHUNK - 1}"
    try:
        response = requests.get(url, headers=headers, timeout=25,
                                allow_redirects=True)
        body = response.content or b""
        return {
            "status": response.status_code,
            "final_url": response.url,
            "redirected": response.url != url,
            "redirect_hops": len(response.history),
            "content_type": response.headers.get("Content-Type"),
            "content_range": response.headers.get("Content-Range"),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "server": response.headers.get("Server"),
            "bytes": len(body),
            "hash": digest(body),
            "is_mp4": looks_like_mp4(body),
            "is_image": looks_like_image(body),
            "error": None,
        }
    except Exception as exc:                       # noqa: BLE001
        return {"status": None, "final_url": None, "redirected": None,
                "redirect_hops": None, "content_type": None,
                "content_range": None, "etag": None, "last_modified": None,
                "server": None, "bytes": 0, "hash": "", "is_mp4": False,
                "is_image": False, "error": f"{type(exc).__name__}: {exc}"}


def verdicts(frame: pd.DataFrame) -> dict:
    """The three questions, answered by the rules stated in the docstring."""
    out = {}

    real = frame.loc[frame["kind"].eq("clip_medium")
                     & frame["header_set"].eq("browser")]
    fake = frame.loc[frame["kind"].eq("clip_fabricated")
                     & frame["header_set"].eq("browser")]
    real_hashes = {h for h in real["hash"] if h}
    fake_hashes = {h for h in fake["hash"] if h}
    if fake_hashes and fake_hashes & real_hashes:
        out["placeholder"] = (
            "CONFIRMED", "a fabricated uuid returns the same bytes as real "
                         "clips, so this host answers everything with one file")
    elif fake_hashes:
        out["placeholder"] = (
            "NO", "the fabricated uuid returns different bytes from the real "
                  "clips")
    else:
        out["placeholder"] = (
            "NOT SERVED", "the fabricated uuid returned nothing, so real URLs "
                          "are being resolved rather than defaulted")

    for kind, label, test in (
            ("thumbnail", "stills", "is_image"),
            ("clip_small", "small encoding", "is_mp4")):
        rows = frame.loc[frame["kind"].eq(kind)
                         & frame["header_set"].eq("browser")]
        hashes = {h for h in rows["hash"] if h}
        valid = int(rows[test].sum())
        if len(rows) and len(hashes) == len(rows) and valid == len(rows):
            out[kind] = ("VIABLE",
                         f"{len(rows)} {label} URLs returned {len(hashes)} "
                         f"distinct, well-formed responses")
        elif not len(rows):
            out[kind] = ("NOT TESTED", "no URLs of this kind were available")
        else:
            out[kind] = ("NOT VIABLE",
                         f"{len(rows)} {label} URLs returned only "
                         f"{len(hashes)} distinct hash(es), {valid} well-formed")
    return out


def build_report(frame: pd.DataFrame) -> str:
    calls = verdicts(frame)
    lines = [
        "=" * 78,
        "PHASE 11e - WHY DOES EVERY CLIP URL RETURN THE SAME FILE",
        "=" * 78,
        "",
        "  Phase 11d: sixteen distinct URLs, eight seasons, eight games,",
        "  sixteen uuids, all returning byte-identical content of exactly",
        "  31,580,089 bytes. This asks what mechanism produces that.",
        "",
        "=" * 78,
        "VERDICTS",
        "=" * 78,
        "",
    ]
    for key, label in (("placeholder", "placeholder served for everything"),
                       ("thumbnail", "still images usable instead"),
                       ("clip_small", "320x180 encoding usable")):
        result, reason = calls[key]
        lines += [f"  {label:<36} {result}", f"      {reason}", ""]

    if calls["placeholder"][0] == "CONFIRMED":
        lines += [
            "  Nothing served from videos.nba.com under these conditions can",
            "  be trusted to be the play it names. The video panel is not",
            "  buildable from this source as things stand.",
            "",
        ]
    if calls["thumbnail"][0] == "VIABLE":
        lines += [
            "  The stills ARE distinct. A panel showing the real frame of the",
            "  real play beside the probability line remains possible, and",
            "  carries most of the original intent.",
            "",
        ]

    lines += ["=" * 78, "RESPONSES BY ASSET AND HEADER SET", "=" * 78, "",
              f"  {'kind':<18}{'headers':<10}{'status':>7}{'bytes':>8}"
              f"{'hash':<18}{'type':<12}{'hops':>5}"]
    for row in frame.itertuples():
        lines.append(
            f"  {str(row.kind):<18}{str(row.header_set):<10}"
            f"{str(row.status):>7}{row.bytes:>8}  {(row.hash or '-'):<16}"
            f"{str(row.content_type or '-')[:11]:<12}"
            f"{str(row.redirect_hops):>5}")

    lines += ["", "=" * 78, "DISTINCT HASHES PER ASSET KIND", "=" * 78, "",
              f"  {'kind':<18}{'headers':<10}{'urls':>6}{'distinct':>10}"]
    for (kind, header_set), group in frame.groupby(["kind", "header_set"]):
        hashes = {h for h in group["hash"] if h}
        lines.append(f"  {str(kind):<18}{str(header_set):<10}{len(group):>6}"
                     f"{len(hashes):>10}")

    redirected = frame.loc[frame["redirected"].fillna(False)]
    lines += ["", "=" * 78, "REDIRECTS", "=" * 78, ""]
    if not len(redirected):
        lines.append("  None. Every request was answered at the URL asked for,")
        lines.append("  so a redirect to a generic asset is not the mechanism.")
    else:
        lines.append(f"  {len(redirected)} request(s) were redirected:")
        for row in redirected.head(6).itertuples():
            lines.append(f"    {row.kind}/{row.header_set} -> "
                         f"{str(row.final_url)[:95]}")

    errors = frame.loc[frame["error"].notna()]
    if len(errors):
        lines += ["", f"  request errors: {len(errors)}"]
        for row in errors.head(3).itertuples():
            lines.append(f"    {row.kind}/{row.header_set}: "
                         f"{str(row.error)[:80]}")

    lines += [
        "",
        "=" * 78,
        "WHAT THIS DOES NOT TELL YOU",
        "=" * 78,
        "",
        "  Whether the NBA intends these assets to be reachable this way at",
        "  all. A host that answers every address with one file is a host that",
        "  is not serving us the content, and working around that is a",
        "  different kind of decision from a technical one.",
        "",
        "  Nothing was changed by this script.",
        "=" * 78,
    ]
    return "\n".join(lines)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    config.ensure_dirs()

    source = config.INTERIM_DIR / "video_probe.csv"
    if not source.exists():
        raise SystemExit(
            f"{source} not found. Run scripts/24_probe_video.py first.")

    probe = pd.read_csv(source)
    matched = probe.loc[probe["status"].eq("matched") & probe["url"].notna()]
    if matched.empty:
        raise SystemExit("no matched clips in video_probe.csv")

    # One clip from each of three well-separated seasons.
    picked = (matched.sort_values(["season", "game_date", "event_index"])
              .groupby("season", group_keys=False).head(1))
    picked = picked.iloc[[0, len(picked) // 2, -1]] if len(picked) >= 3 \
        else picked.head(CLIPS_TO_TEST)

    targets = []
    for row in picked.itertuples():
        targets.append(("clip_medium", row.url))
        if isinstance(getattr(row, "url_small", None), str):
            targets.append(("clip_small", row.url_small))
        if isinstance(getattr(row, "thumbnail", None), str):
            targets.append(("thumbnail", row.thumbnail))
    targets.append(("clip_fabricated", fabricate(picked.iloc[0]["url"])))

    logger.info("testing %d URLs against %d header sets",
                len(targets), len(HEADER_SETS))

    records = []
    for kind, url in targets:
        for header_set in HEADER_SETS:
            result = fetch(url, header_set)
            records.append({"kind": kind, "header_set": header_set,
                            "url": url, **result})
            logger.info("  %-16s %-8s %s %s", kind, header_set,
                        result["status"], result["hash"] or "-")
            time.sleep(0.2)

    frame = pd.DataFrame(records)
    report = build_report(frame)
    print(report)
    out = config.REPORTS_DIR / "video_mechanism.txt"
    out.write_text(report + "\n", encoding="utf-8")
    frame.to_csv(config.INTERIM_DIR / "video_mechanism.csv", index=False)
    print(f"\nSaved to: {out}")
    print("\nREAD ONLY. Nothing in the app, the API, the model or the research")
    print("outputs was modified.")
    return frame


if __name__ == "__main__":
    main()
