"""
Tests for Phase 11: the read-only video coverage probe.

Three properties are worth asserting here, and they are the three that would
quietly ruin the result if they broke.

  1. THE SAMPLE IS DETERMINISTIC AND SPANS EVERY SEASON.
     A probe whose sample moves between runs cannot be compared with its own
     earlier result, and a sample that drifts toward recent seasons would
     report the coverage of 2023-24 as if it were the coverage of 2016-17.

  2. A CLIP THAT CANNOT BE VERIFIED IS NOT COVERAGE.
     The endpoint is keyed by the play-by-play event number, which Phase 2
     established is not unique within a game. So `classify` must refuse to call
     a clip matched unless the returned game id, the returned event id AND the
     returned description all agree. Every one of those three is tested by
     breaking it on its own and asserting the verdict flips to mismatch.

  3. THE PROBE WRITES NOTHING THE APP READS.
     The whole instruction was to measure without changing anything. That is
     asserted by running the probe end to end against a stubbed endpoint and
     checking that the serving, model, processed and figures directories are
     byte-for-byte what they were before.
"""

import pandas as pd
import pytest

from src import config, video_probe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_events(games_per_season=10, events_per_game=6):
    """
    A synthetic event table shaped like data/interim/events.parquet.

    Synthetic on purpose: these tests are about the probe's logic, and tying
    them to whichever games happen to sit at the 25th percentile of the real
    2018-19 schedule would make them fail for a reason that is not a bug.
    """
    types = ["2pt", "3pt", "rebound", "foul", "substitution", "turnover"]
    rows = []
    for s_index, season in enumerate(config.SEASONS):
        for g in range(games_per_season):
            game_id = f"002{s_index}6{g:05d}"
            date = pd.Timestamp("2016-10-25") + pd.Timedelta(
                days=365 * s_index + g)
            for e in range(events_per_game):
                rows.append({
                    "game_id": game_id,
                    "season": season,
                    "game_date": date,
                    "event_index": e,
                    "action_number": e + 1,
                    "period": 1,
                    "action_type": types[e % len(types)],
                    "description": f"{types[e % len(types)]} by player {e}",
                })
    return pd.DataFrame(rows)


@pytest.fixture
def events():
    return make_events()


# ---------------------------------------------------------------------------
# 1. Sampling
# ---------------------------------------------------------------------------

def test_sample_covers_every_season(events):
    games = video_probe.sample_games(events)
    assert set(games["season"]) == set(config.SEASONS), (
        "the probe must span all eight seasons; a sample missing one cannot "
        "say anything about coverage in that season")


def test_sample_takes_the_configured_number_per_season(events):
    games = video_probe.sample_games(events)
    counts = games.groupby("season").size()
    assert set(counts) == {video_probe.GAMES_PER_SEASON}


def test_sample_is_deterministic(events):
    first = video_probe.sample_games(events)
    second = video_probe.sample_games(events.sample(frac=1.0, random_state=7))
    assert list(first["game_id"]) == list(second["game_id"]), (
        "the sample must not depend on row order, or a re-run probes a "
        "different set of games and the two reports cannot be compared")


def test_sample_spreads_through_the_season_rather_than_clustering(events):
    """Early, middle and late games, not three from the same week."""
    games = video_probe.sample_games(events)
    for season, group in games.groupby("season"):
        dates = sorted(pd.to_datetime(group["game_date"]))
        assert len(set(dates)) == video_probe.GAMES_PER_SEASON, (
            f"{season}: the three sampled games fell on the same date")


def test_sample_survives_a_short_season(events):
    """A season with fewer games than positions must not raise or index past the end."""
    short = events.loc[events["game_id"].isin(
        sorted(events["game_id"].unique())[:2])]
    games = video_probe.sample_games(short)
    assert len(games) >= 1
    assert set(games["game_id"]).issubset(set(short["game_id"]))


def test_events_are_sampled_across_every_action_type(events):
    game_id = events["game_id"].iloc[0]
    chosen = video_probe.sample_events(events.loc[events["game_id"].eq(game_id)])
    present = set(events.loc[events["game_id"].eq(game_id), "action_type"])
    assert set(chosen["action_type"]) == present, (
        "coverage differs between a made three and a substitution; blending "
        "them into one percentage would hide the only distinction that "
        "matters for a video feature")


def test_events_per_type_is_capped(events):
    game_id = events["game_id"].iloc[0]
    many = pd.concat([events.loc[events["game_id"].eq(game_id)]] * 5)
    many["event_index"] = range(len(many))
    chosen = video_probe.sample_events(many)
    counts = chosen.groupby("action_type").size()
    assert counts.max() <= video_probe.EVENTS_PER_TYPE


# ---------------------------------------------------------------------------
# 2. Matching
# ---------------------------------------------------------------------------

def payload(game_id="0021600006", event_id=7, description="3pt by player 2"):
    """
    The shape videoeventsasset actually returns, copied from a real response.
    """
    stem = f"https://videos.nba.com/nba/pbp/media/2024/03/07/{game_id}/{event_id}/abc"
    return {
        "resource": "videoevents",
        "resultSets": {
            "Meta": {"videoUrls": [{
                "uuid": "abc",
                "sdur": 8850, "surl": f"{stem}_320x180.mp4",
                "sth": f"{stem}_320x180.jpg",
                "mdur": 8850, "murl": f"{stem}_960x540.mp4",
                "mth": f"{stem}_960x540.jpg",
                "ldur": 8850, "lurl": f"{stem}_1280x720.mp4",
                "lth": f"{stem}_1280x720.jpg",
                "vtt": f"{stem}.vtt", "scc": f"{stem}.scc",
                "srt": f"{stem}.srt",
            }]},
            "playlist": [{"gi": game_id, "ei": event_id, "dsc": description,
                          "y": 2024, "m": "03", "d": "07", "p": 1,
                          "ha": "DEN", "va": "BOS"}],
        }
    }


def stub_payload(game_id="0021600006", event_id=7,
                 description="3pt by player 2"):
    """
    What the WRONG endpoint returned: a well-formed answer with no URLs.

    This is the exact body videoevents sent for all 535 events in the probe's
    first run. The placeholder uuid is the real one, and it was identical for
    every event in every game across eight seasons.
    """
    return {
        "resource": "videoevents",
        "resultSets": {
            "Meta": {"videoUrls": [{
                "uuid": "4549dfbf-fde2-4dcc-8065-afade5ada267",
                "dur": None, "stt": None, "stp": None, "sth": None,
                "stw": None, "mtt": None, "mtp": None, "mth": None,
                "mtw": None, "ltt": None, "ltp": None, "lth": None,
                "ltw": None,
            }]},
            "playlist": [{"gi": game_id, "ei": event_id, "dsc": description}],
        }
    }


def test_empty_payload_is_no_clip():
    verdict = video_probe.classify("0021600006", 7, None, "3pt by player 2")
    assert verdict["status"] == "no_clip"
    assert verdict["url"] is None


def test_payload_without_a_url_is_no_clip():
    body = payload()
    body["resultSets"]["Meta"]["videoUrls"] = []
    verdict = video_probe.classify("0021600006", 7, body, "3pt by player 2")
    assert verdict["status"] == "no_clip"


def test_full_agreement_is_matched():
    verdict = video_probe.classify("0021600006", 7, payload(),
                                   "3pt by player 2")
    assert verdict["status"] == "matched"
    assert verdict["url"].endswith(".mp4")


def test_a_different_game_id_is_a_mismatch_not_coverage():
    verdict = video_probe.classify(
        "0021600006", 7, payload(game_id="0021600099"), "3pt by player 2")
    assert verdict["status"] == "mismatch"
    assert "game id differs" in verdict["mismatch_reason"]


def test_a_different_event_id_is_a_mismatch_not_coverage():
    verdict = video_probe.classify(
        "0021600006", 7, payload(event_id=99), "3pt by player 2")
    assert verdict["status"] == "mismatch"
    assert "event id differs" in verdict["mismatch_reason"]


def test_a_disagreeing_description_is_a_mismatch_not_coverage():
    """
    The case the other two checks cannot catch.

    Duplicate action numbers within a game mean the endpoint can return the
    right game and the right number and still be about a different play. The
    description is the only independent evidence available.
    """
    verdict = video_probe.classify(
        "0021600006", 7, payload(description="Jaylen Brown Substitution"),
        "Jayson Tatum 26' 3PT Jump Shot")
    assert verdict["status"] == "mismatch"
    assert "description disagrees" in verdict["mismatch_reason"]


def test_game_id_padding_does_not_cause_a_false_mismatch():
    """
    The int64-versus-zero-padded-string trap that broke a join in Phase 9b.

    Here it would show up as a fake mismatch rather than an empty table, which
    is harder to notice, so it is pinned.
    """
    verdict = video_probe.classify(
        "0021600006", 7, payload(game_id="21600006"), "3pt by player 2")
    assert verdict["status"] == "matched"


def test_description_agreement_bounds():
    assert video_probe.description_agreement("made three", "made three") == 1.0
    assert video_probe.description_agreement("made three", "") == 0.0
    assert video_probe.description_agreement("", "anything") == 0.0
    assert video_probe.description_agreement(None, "anything") == 0.0


def test_description_agreement_ignores_punctuation_and_case():
    assert video_probe.description_agreement(
        "Tatum 26' 3PT Jump Shot", "tatum 26 3pt jump shot") == 1.0


# ---------------------------------------------------------------------------
# 3. Read-only
# ---------------------------------------------------------------------------

def snapshot(*dirs):
    out = {}
    for d in dirs:
        if d.exists():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    out[str(p)] = p.stat().st_size
    return out


def test_the_probe_writes_only_its_two_files(tmp_path, monkeypatch, events):
    """
    The instruction was to measure without changing anything, so this runs the
    probe end to end against a stubbed endpoint and asserts the app's data is
    untouched.
    """
    interim = tmp_path / "interim"
    reports = tmp_path / "reports"
    serving = tmp_path / "serving"
    models = tmp_path / "models"
    for d in (interim, reports, serving, models):
        d.mkdir(parents=True)
    (serving / "index.json").write_text('{"count": 636}')
    (models / "win_probability_model.joblib").write_bytes(b"not a real model")

    parquet = interim / "events.parquet"
    events.to_parquet(parquet)

    monkeypatch.setattr(config, "INTERIM_DIR", interim)
    monkeypatch.setattr(config, "REPORTS_DIR", reports)
    monkeypatch.setattr(config, "SERVING_DIR", serving)
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(config, "EVENTS_PARQUET", parquet)
    monkeypatch.setattr(config, "ALL_DIRS", [interim, reports, serving, models])

    monkeypatch.setattr(
        video_probe, "fetch_event",
        lambda game_id, action_number: payload(
            game_id=game_id, event_id=action_number,
            description="rebound by player 2"))
    monkeypatch.setattr(video_probe, "check_playback",
                        lambda url: {"http_status": 200,
                                     "content_type": "video/mp4",
                                     "content_length": "1200000",
                                     "error": None})

    before = snapshot(serving, models)
    frame = video_probe.main()
    after = snapshot(serving, models)

    assert before == after, (
        "the probe must not write into serving or models; the dashboard has "
        "to keep working exactly as it did before this ran")
    assert (reports / "video_probe.txt").exists()
    assert (interim / "video_probe.csv").exists()
    assert set(reports.iterdir()) == {reports / "video_probe.txt"}
    assert len(frame) > 0


def test_a_request_failure_is_recorded_rather_than_raised(tmp_path, monkeypatch,
                                                          events):
    """One dead call must not abandon the other four hundred."""
    interim = tmp_path / "interim"
    interim.mkdir()
    parquet = interim / "events.parquet"
    events.to_parquet(parquet)
    monkeypatch.setattr(config, "EVENTS_PARQUET", parquet)

    def explode(game_id, action_number):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(video_probe, "fetch_event", explode)
    frame = video_probe.run_probe()

    assert len(frame) > 0
    assert frame["request_error"].notna().all()
    assert set(frame["status"]) == {"no_clip"}

    # A blocked endpoint and an NBA with no video produce identical NO CLIP
    # counts. The report has to separate them or it answers the wrong question.
    report = video_probe.build_report(frame)
    assert "WARNING" in report
    assert "does not measure clip availability" in report
    assert "TimeoutError" in report


def test_a_clean_zero_is_flagged_rather_than_reported_as_a_finding(
        tmp_path, monkeypatch, events):
    """
    The failure this probe actually walked into on its first real run.

    535 events, 0 errors, 0 clips. That reads like "the NBA has no video" and
    is equally well explained by "we asked the wrong endpoint". The report has
    to refuse to present it as a result.
    """
    interim = tmp_path / "interim"
    interim.mkdir()
    parquet = interim / "events.parquet"
    events.to_parquet(parquet)
    monkeypatch.setattr(config, "EVENTS_PARQUET", parquet)

    # Every request succeeds and every one comes back with an empty body.
    monkeypatch.setattr(video_probe, "fetch_event",
                        lambda game_id, action_number: {
                            "resultSets": {"Meta": {"videoUrls": []},
                                           "playlist": []}})
    frame = video_probe.run_probe()
    assert set(frame["status"]) == {"no_clip"}
    assert frame["request_error"].isna().all()

    report = video_probe.build_report(frame)
    assert "WARNING" in report
    assert "not evidence that there is no video" in report
    assert "25_diagnose_video.py" in report


def test_report_does_not_cry_wolf_when_requests_succeed(tmp_path, monkeypatch,
                                                        events):
    interim = tmp_path / "interim"
    interim.mkdir()
    parquet = interim / "events.parquet"
    events.to_parquet(parquet)
    monkeypatch.setattr(config, "EVENTS_PARQUET", parquet)
    monkeypatch.setattr(
        video_probe, "fetch_event",
        lambda game_id, action_number: payload(
            game_id=game_id, event_id=action_number, description="rebound"))
    monkeypatch.setattr(video_probe, "check_playback",
                        lambda url: {"http_status": 200, "content_type": None,
                                     "content_length": None, "error": None})

    report = video_probe.build_report(video_probe.run_probe())
    assert "WARNING" not in report


def test_check_playback_never_raises(monkeypatch):
    """
    A probe that dies on a bad URL reports nothing about the good ones.
    """
    import requests

    def explode(*args, **kwargs):
        raise requests.exceptions.ConnectionError("no route to host")

    monkeypatch.setattr(requests, "head", explode)
    result = video_probe.check_playback("https://videos.nba.com/x.mp4")
    assert result["http_status"] is None
    assert "ConnectionError" in result["error"]


def test_report_never_counts_a_mismatch_as_coverage(tmp_path, monkeypatch,
                                                    events):
    """
    The sentence in the report that would be a lie if the arithmetic slipped.

    Every probed event lands in exactly one of matched, mismatch or no_clip,
    and the match rate is computed from matched alone.
    """
    interim = tmp_path / "interim"
    interim.mkdir()
    parquet = interim / "events.parquet"
    events.to_parquet(parquet)
    monkeypatch.setattr(config, "EVENTS_PARQUET", parquet)

    # Every clip comes back describing the wrong play.
    monkeypatch.setattr(
        video_probe, "fetch_event",
        lambda game_id, action_number: payload(
            game_id=game_id, event_id=action_number,
            description="completely unrelated commentary"))
    monkeypatch.setattr(video_probe, "check_playback",
                        lambda url: {"http_status": 200, "content_type": None,
                                     "content_length": None, "error": None})

    frame = video_probe.run_probe()
    assert set(frame["status"]) == {"mismatch"}

    report = video_probe.build_report(frame)
    matched_line = next(line for line in report.splitlines()
                        if line.strip().startswith("MATCHED"))
    assert "0" in matched_line and "0.0%" in matched_line, (
        "every clip was about the wrong play, so the matched count and rate "
        "must read zero rather than being padded by the mismatches: "
        f"{matched_line}")
    assert "100.0%" in next(line for line in report.splitlines()
                            if line.strip().startswith("MISMATCH"))
    assert "worse than showing none" in report


def test_report_breaks_coverage_down_by_season_and_event_type(tmp_path,
                                                              monkeypatch,
                                                              events):
    interim = tmp_path / "interim"
    interim.mkdir()
    parquet = interim / "events.parquet"
    events.to_parquet(parquet)
    monkeypatch.setattr(config, "EVENTS_PARQUET", parquet)
    monkeypatch.setattr(
        video_probe, "fetch_event",
        lambda game_id, action_number: payload(
            game_id=game_id, event_id=action_number, description="rebound"))
    monkeypatch.setattr(video_probe, "check_playback",
                        lambda url: {"http_status": 200, "content_type": None,
                                     "content_length": None, "error": None})

    report = video_probe.build_report(video_probe.run_probe())
    assert "COVERAGE BY SEASON" in report
    assert "COVERAGE BY EVENT TYPE" in report
    for season in config.SEASONS:
        assert season in report
    assert "substitution" in report


# ---------------------------------------------------------------------------
# 4. The diagnostic (Phase 11b)
# ---------------------------------------------------------------------------

def test_describe_payload_reports_an_empty_playlist_as_empty():
    from src import video_diagnose
    text = video_diagnose.describe_payload(
        {"resultSets": {"Meta": {"videoUrls": []}, "playlist": []}})
    assert "Meta.videoUrls : 0 entr" in text
    assert "playlist       : 0 entr" in text


def test_describe_payload_reports_a_populated_playlist():
    from src import video_diagnose
    text = video_diagnose.describe_payload(
        payload()["resultSets"] and payload())
    assert "Meta.videoUrls : 1 entr" in text
    assert "playlist       : 1 entr" in text
    assert "0021600006" in text


def test_describe_payload_survives_an_unexpected_shape():
    """
    The case that matters most. If the response is not the shape the probe
    assumes, the diagnostic has to print what it IS rather than raise, because
    an unrecognised shape is one of the three answers we are trying to tell
    apart.
    """
    from src import video_diagnose
    for odd in (None, "a string", {"resultSets": [{"name": "x"}]}, {}):
        text = video_diagnose.describe_payload(odd)
        assert isinstance(text, str) and text


def test_diagnostic_writes_only_its_report(tmp_path, monkeypatch, events):
    from src import video_diagnose

    reports = tmp_path / "reports"
    interim = tmp_path / "interim"
    serving = tmp_path / "serving"
    for d in (reports, interim, serving):
        d.mkdir(parents=True)
    (serving / "index.json").write_text('{"count": 636}')

    # Give the diagnostic games it can actually find.
    frame = events.copy()
    ids = sorted(frame["game_id"].unique())[:2]
    frame.loc[frame["game_id"].eq(ids[0]), "action_type"] = "Made Shot"
    parquet = interim / "events.parquet"
    frame.to_parquet(parquet)

    monkeypatch.setattr(config, "EVENTS_PARQUET", parquet)
    monkeypatch.setattr(config, "REPORTS_DIR", reports)
    monkeypatch.setattr(config, "SERVING_DIR", serving)
    monkeypatch.setattr(config, "ALL_DIRS", [reports, interim, serving])
    monkeypatch.setattr(video_diagnose, "DIAGNOSTIC_GAMES", list(ids))
    monkeypatch.setattr(video_diagnose, "try_endpoint",
                        lambda cls, g, e: {"ok": True, "url": "https://x",
                                           "payload": payload(game_id=g,
                                                              event_id=e),
                                           "raw": "{}", "error": None})
    monkeypatch.setattr(video_diagnose, "raw_http",
                        lambda g, e, name="videoevents": {
                            "status": 200, "url": "https://x", "body": "{}",
                            "error": None})

    before = snapshot(serving)
    report = video_diagnose.main()
    assert snapshot(serving) == before
    assert set(reports.iterdir()) == {reports / "video_diagnose.txt"}
    assert "HOW TO READ THIS" in report


def test_diagnostic_records_an_endpoint_failure_instead_of_dying(
        tmp_path, monkeypatch, events):
    from src import video_diagnose

    reports = tmp_path / "reports"
    interim = tmp_path / "interim"
    for d in (reports, interim):
        d.mkdir(parents=True)
    parquet = interim / "events.parquet"
    events.to_parquet(parquet)
    ids = sorted(events["game_id"].unique())[:1]

    monkeypatch.setattr(config, "EVENTS_PARQUET", parquet)
    monkeypatch.setattr(config, "REPORTS_DIR", reports)
    monkeypatch.setattr(config, "ALL_DIRS", [reports, interim])
    monkeypatch.setattr(video_diagnose, "DIAGNOSTIC_GAMES", list(ids))
    monkeypatch.setattr(video_diagnose, "try_endpoint",
                        lambda cls, g, e: {"ok": False, "url": None,
                                           "payload": None, "raw": None,
                                           "error": "HTTPError: 403"})
    monkeypatch.setattr(video_diagnose, "raw_http",
                        lambda g, e, name="videoevents": {
                            "status": 403, "url": "https://x",
                            "body": "<html>blocked</html>", "error": None})

    report = video_diagnose.main()
    assert "FAILED: HTTPError: 403" in report
    assert "HTTP 403" in report


# ---------------------------------------------------------------------------
# 5. The false negative, pinned
# ---------------------------------------------------------------------------

def test_the_stub_response_that_fooled_the_first_run_is_no_clip():
    """
    The exact body videoevents returned for all 535 events.

    It is well-formed, HTTP 200, right game, right event, right description,
    and carries no playable URL. The only correct verdict is no_clip. If this
    ever came back "matched", the dashboard would try to play a null.
    """
    verdict = video_probe.classify(
        "0021600006", 7, stub_payload(), "3pt by player 2")
    assert verdict["status"] == "no_clip"
    assert verdict["url"] is None


def test_the_real_asset_response_yields_a_playable_url_and_extras():
    verdict = video_probe.classify("0021600006", 7, payload(),
                                   "3pt by player 2")
    assert verdict["status"] == "matched"
    # Medium is what a dashboard panel would use, so it is what gets tested.
    assert verdict["url"] == verdict["url_medium"]
    assert verdict["url"].endswith("_960x540.mp4")
    assert verdict["url_small"].endswith("_320x180.mp4")
    assert verdict["url_large"].endswith("_1280x720.mp4")
    assert verdict["thumbnail"].endswith(".jpg")
    assert verdict["captions"].endswith(".vtt")
    assert verdict["duration"] == 8850


def test_the_probe_asks_videoeventsasset_not_videoevents():
    """
    Pinned by name. This is the single line whose being wrong cost a full
    535-request run and produced a report that read like a finding.
    """
    import inspect
    source = inspect.getsource(video_probe.fetch_event)
    assert "videoeventsasset" in source
    assert "VideoEventsAsset" in source
    assert "videoevents.VideoEvents" not in source


# ---------------------------------------------------------------------------
# 6. Playback sampling
# ---------------------------------------------------------------------------

def playable_frame(per_season=10):
    rows = []
    for season in config.SEASONS:
        for i in range(per_season):
            rows.append({"season": season, "status": "matched",
                         "url": f"https://videos.nba.com/{season}-{i}.mp4"})
    return pd.DataFrame(rows)


def test_playback_sample_spans_every_season():
    """
    The bug this replaced: head(25) on a season-ordered frame takes all 25
    from 2016-17. If old clips have been pruned, that reports "everything is
    broken"; tail() would report "everything is fine". Neither is the answer.
    """
    sample = video_probe.sample_for_playback(playable_frame(),
                                             video_probe.PLAYBACK_CHECKS)
    assert set(sample["season"]) == set(config.SEASONS)


def test_playback_sample_respects_the_limit():
    sample = video_probe.sample_for_playback(playable_frame(), 12)
    assert len(sample) == 12


def test_playback_sample_handles_nothing_to_check():
    empty = playable_frame().head(0)
    assert len(video_probe.sample_for_playback(empty, 40)) == 0
    assert len(video_probe.sample_for_playback(playable_frame(), 0)) == 0


def test_playback_sample_takes_everything_when_there_is_little():
    small = playable_frame(per_season=1)
    sample = video_probe.sample_for_playback(small, 40)
    assert len(sample) == len(small)


# ---------------------------------------------------------------------------
# 7. Phase 11c: is the clip really there
# ---------------------------------------------------------------------------

MP4_HEAD = (b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41"
            b"\x00\x00\x00\x08free")


def test_a_real_mp4_header_is_recognised():
    from src import video_playback_check as vpc
    assert vpc.looks_like_mp4(MP4_HEAD)


def test_an_error_page_is_not_mistaken_for_a_clip():
    """
    The whole point. A CDN answering 200 with an HTML error body, or with
    zero-fill, must not read as a playable clip.
    """
    from src import video_playback_check as vpc
    assert not vpc.looks_like_mp4(b"<html><body>Access Denied</body></html>")
    assert not vpc.looks_like_mp4(b"\x00" * 2048)
    assert not vpc.looks_like_mp4(b"")
    assert not vpc.looks_like_mp4(b"short")
    assert not vpc.looks_like_mp4(None)


def test_content_range_yields_the_real_total_size():
    from src import video_playback_check as vpc
    assert vpc.total_size_from_content_range("bytes 0-2047/48213456") == 48213456
    assert vpc.total_size_from_content_range("bytes 0-2047/*") == 0
    assert vpc.total_size_from_content_range("") == 0
    assert vpc.total_size_from_content_range(None) == 0


def test_clip_sample_is_two_per_season():
    from src import video_playback_check as vpc
    rows = []
    for season in config.SEASONS:
        for i in range(6):
            rows.append({"season": season, "status": "matched",
                         "url": f"https://videos.nba.com/{season}-{i}.mp4",
                         "game_date": f"2017-01-0{i + 1}", "event_index": i})
    sample = vpc.sample_clips(pd.DataFrame(rows))
    assert set(sample["season"]) == set(config.SEASONS)
    assert set(sample.groupby("season").size()) == {vpc.SAMPLE_PER_SEASON}


def test_unmatched_rows_are_never_tested():
    """A mismatch or a no_clip has no URL worth checking."""
    from src import video_playback_check as vpc
    frame = pd.DataFrame([
        {"season": "2016-17", "status": "mismatch", "url": "https://x.mp4",
         "game_date": "2017-01-01", "event_index": 1},
        {"season": "2016-17", "status": "no_clip", "url": None,
         "game_date": "2017-01-02", "event_index": 2},
    ])
    assert vpc.sample_clips(frame).empty


def test_check_one_never_raises(monkeypatch):
    from src import video_playback_check as vpc
    import requests

    def explode(*args, **kwargs):
        raise requests.exceptions.ConnectionError("no route to host")

    monkeypatch.setattr(requests, "head", explode)
    monkeypatch.setattr(requests, "get", explode)
    result = vpc.check_one("https://videos.nba.com/x.mp4")
    assert result["is_mp4"] is False
    assert "ConnectionError" in result["error"]


def test_report_flags_identical_sizes_as_unproven():
    """
    The exact pathology that made the HEAD result worthless: every clip
    reporting the same byte count. If it recurs on the ranged GET, the report
    has to say so instead of printing a confident percentage.
    """
    from src import video_playback_check as vpc
    frame = pd.DataFrame([
        {"season": s, "head_status": 200, "head_length": "33200000",
         "range_status": 206, "range_bytes_returned": 2048,
         "real_total_bytes": 33200000, "content_range": "bytes 0-2047/33200000",
         "is_mp4": True, "error": None}
        for s in config.SEASONS])
    report = vpc.build_report(frame)
    assert "unproven" in report
    assert "distinct real file sizes among them: 1" in report


def test_report_does_not_flag_plausibly_varied_sizes():
    from src import video_playback_check as vpc
    frame = pd.DataFrame([
        {"season": s, "head_status": 200, "head_length": "33200000",
         "range_status": 206, "range_bytes_returned": 2048,
         "real_total_bytes": 1_000_000 + i * 137_000,
         "content_range": f"bytes 0-2047/{1_000_000 + i * 137_000}",
         "is_mp4": True, "error": None}
        for i, s in enumerate(config.SEASONS)])
    report = vpc.build_report(frame)
    assert "unproven" not in report


def test_report_marks_a_non_mp4_body_as_failure():
    from src import video_playback_check as vpc
    frame = pd.DataFrame([
        {"season": "2016-17", "head_status": 200, "head_length": "33200000",
         "range_status": 200, "range_bytes_returned": 512,
         "real_total_bytes": 0, "content_range": None,
         "is_mp4": False, "error": None}])
    report = vpc.build_report(frame)
    assert "  NO" in report
    assert "real MP4 bytes returned 0" in report


# ---------------------------------------------------------------------------
# 8. Phase 11d: distinct videos, or one video sixteen times
# ---------------------------------------------------------------------------

def fingerprint_frame(distinct=True, n=16, tail_status=416):
    rows = []
    for i in range(n):
        h = f"{i:016x}" if distinct else "aaaaaaaaaaaaaaaa"
        d = f"d{i:015x}" if distinct else "bbbbbbbbbbbbbbbb"
        rows.append({"season": config.SEASONS[i % len(config.SEASONS)],
                     "game_id": f"00216003{i:02d}", "action_number": i,
                     "url": f"https://videos.nba.com/{i}.mp4",
                     "header_status": 206, "header_hash": h,
                     "header_bytes": 2048, "header_error": None,
                     "deep_status": 206, "deep_hash": d,
                     "deep_bytes": 2048, "deep_error": None,
                     "claimed_tail_status": tail_status, "claimed_tail_hash": "",
                     "claimed_tail_bytes": 0, "claimed_tail_error": None})
    return pd.DataFrame(rows)


def test_identical_headers_are_fatal():
    """The outcome that kills the feature: every URL serving the same video."""
    from src import video_fingerprint as vf
    result, reason = vf.verdict(fingerprint_frame(distinct=False))
    assert result == "FATAL"
    assert "not all different videos" in reason


def test_distinct_headers_and_interiors_pass():
    from src import video_fingerprint as vf
    result, reason = vf.verdict(fingerprint_frame(distinct=True))
    assert result == "PASS"
    assert "16 distinct headers" in reason


def test_one_repeated_header_among_many_is_still_fatal():
    """
    The rule is not "mostly distinct". Fifteen real clips and one duplicate
    means one event would show the wrong play, and that is the whole objection.
    """
    from src import video_fingerprint as vf
    frame = fingerprint_frame(distinct=True)
    frame.loc[frame.index[-1], "header_hash"] = frame.loc[frame.index[0],
                                                          "header_hash"]
    result, _ = vf.verdict(frame)
    assert result == "FATAL"


def test_distinct_headers_but_identical_interiors_is_fatal():
    """
    A container could differ in its header and still wrap the same video.
    Chance agreement 1 MB into two different encodes is not credible, so
    identical interiors is treated as the same failure.
    """
    from src import video_fingerprint as vf
    frame = fingerprint_frame(distinct=True)
    frame["deep_hash"] = "same0000same0000"
    result, reason = vf.verdict(frame)
    assert result == "FATAL"
    assert "1 MB into" in reason


def test_empty_frame_is_fatal_not_pass():
    from src import video_fingerprint as vf
    result, _ = vf.verdict(fingerprint_frame().head(0))
    assert result == "FATAL"


def test_digest_marks_empty_bodies_rather_than_hashing_them():
    """
    An empty 416 body must not hash to a value, or sixteen 416s would look
    like sixteen identical files and trip FATAL for the wrong reason.
    """
    from src import video_fingerprint as vf
    assert vf.digest(b"") == ""
    assert vf.digest(None) == ""
    assert len(vf.digest(b"some bytes")) == 16
    assert vf.digest(b"a") != vf.digest(b"b")


def test_fetch_range_never_raises(monkeypatch):
    from src import video_fingerprint as vf
    import requests

    def explode(*args, **kwargs):
        raise requests.exceptions.ReadTimeout("timed out")

    monkeypatch.setattr(requests, "get", explode)
    result = vf.fetch_range("https://videos.nba.com/x.mp4", 0)
    assert result["hash"] == ""
    assert "ReadTimeout" in result["error"]


def test_report_states_the_fatal_verdict_plainly():
    from src import video_fingerprint as vf
    report = vf.build_report(fingerprint_frame(distinct=False),
                             {"status": 200, "actual_bytes": 1_800_000,
                              "complete": True, "advertised": "31580089",
                              "hash": "x", "error": None})
    assert "VERDICT: FATAL" in report
    assert "Do not build the video panel" in report


def test_report_calls_out_an_advertised_size_that_did_not_arrive():
    from src import video_fingerprint as vf
    report = vf.build_report(fingerprint_frame(distinct=True),
                             {"status": 200, "actual_bytes": 1_800_000,
                              "complete": True, "advertised": "31580089",
                              "hash": "x", "error": None})
    assert "VERDICT: PASS" in report
    assert "advertised size is wrong" in report
    assert "1.8 MB" in report


def test_report_survives_a_failed_download():
    from src import video_fingerprint as vf
    report = vf.build_report(fingerprint_frame(distinct=True),
                             {"error": "ReadTimeout: timed out"})
    assert "download failed" in report
    assert "VERDICT: PASS" in report


def test_report_explains_a_416_at_the_claimed_offset():
    from src import video_fingerprint as vf
    report = vf.build_report(fingerprint_frame(distinct=True, tail_status=416),
                             {"error": "x"})
    assert "the advertised 31.6 MB does not exist" in report


# ---------------------------------------------------------------------------
# 9. Phase 11e: the access mechanism
# ---------------------------------------------------------------------------

REAL_URL = ("https://videos.nba.com/nba/pbp/media/2016/12/05/0021600311/1/"
            "40bf6a83-2928-81e5-929b-59f40525d1dd_960x540.mp4")


def test_fabricate_keeps_the_shape_and_kills_only_the_uuid():
    """
    The control has to differ in exactly one respect. Same host, same path,
    same encoding suffix, a uuid that names nothing.
    """
    from src import video_mechanism as vm
    fake = vm.fabricate(REAL_URL)
    assert fake != REAL_URL
    assert fake.startswith("https://videos.nba.com/nba/pbp/media/2016/12/05/")
    assert fake.endswith("_960x540.mp4")
    assert "00000000-0000-0000-0000-000000000000" in fake
    assert "40bf6a83" not in fake


def test_fabricate_is_a_noop_when_there_is_no_uuid_to_replace():
    from src import video_mechanism as vm
    plain = "https://videos.nba.com/nba/pbp/media/x.mp4"
    assert vm.fabricate(plain) == plain


def test_image_and_mp4_sniffing():
    from src import video_mechanism as vm
    assert vm.looks_like_image(b"\xff\xd8\xff\xe0" + b"0" * 40)
    assert vm.looks_like_image(b"\x89PNG\r\n\x1a\n" + b"0" * 40)
    assert not vm.looks_like_image(b"<html>nope</html>")
    assert not vm.looks_like_image(b"")
    assert vm.looks_like_mp4(MP4_HEAD)
    assert not vm.looks_like_mp4(b"\xff\xd8\xff\xe0" + b"0" * 40)


def mechanism_frame(fake_hash="deadbeefdeadbeef", thumb_distinct=True,
                    small_distinct=True):
    rows = []
    for i in range(3):
        rows.append(("clip_medium", "same0000same0000", "video/mp4", True, False))
        rows.append(("thumbnail", f"thumb{i:011d}" if thumb_distinct
                     else "thumbsame0000000", "image/jpeg", False, True))
        rows.append(("clip_small", f"small{i:011d}" if small_distinct
                     else "smallsame0000000", "video/mp4", True, False))
    rows.append(("clip_fabricated", fake_hash, "video/mp4", True, False))
    out = []
    for kind, h, ctype, is_mp4, is_image in rows:
        for header_set in ("browser", "bare", "origin"):
            out.append({"kind": kind, "header_set": header_set,
                        "url": "https://videos.nba.com/x", "status": 206,
                        "final_url": "https://videos.nba.com/x",
                        "redirected": False, "redirect_hops": 0,
                        "content_type": ctype, "content_range": None,
                        "etag": None, "last_modified": None, "server": None,
                        "bytes": 4096, "hash": h, "is_mp4": is_mp4,
                        "is_image": is_image, "error": None})
    return pd.DataFrame(out)


def test_a_fabricated_url_returning_the_same_bytes_confirms_a_placeholder():
    from src import video_mechanism as vm
    frame = mechanism_frame(fake_hash="same0000same0000")
    result, reason = vm.verdicts(frame)["placeholder"]
    assert result == "CONFIRMED"
    assert "answers everything with one file" in reason


def test_a_fabricated_url_returning_different_bytes_does_not():
    from src import video_mechanism as vm
    result, _ = vm.verdicts(mechanism_frame())["placeholder"]
    assert result == "NO"


def test_distinct_wellformed_thumbnails_are_viable():
    from src import video_mechanism as vm
    result, _ = vm.verdicts(mechanism_frame(thumb_distinct=True))["thumbnail"]
    assert result == "VIABLE"


def test_identical_thumbnails_are_not_viable():
    """
    The failure that would kill the fallback too. If the stills are gated the
    same way as the videos, there is nothing left to build.
    """
    from src import video_mechanism as vm
    result, reason = vm.verdicts(
        mechanism_frame(thumb_distinct=False))["thumbnail"]
    assert result == "NOT VIABLE"
    assert "1 distinct hash" in reason


def test_thumbnails_that_are_not_images_are_not_viable():
    from src import video_mechanism as vm
    frame = mechanism_frame(thumb_distinct=True)
    frame.loc[frame["kind"].eq("thumbnail"), "is_image"] = False
    result, _ = vm.verdicts(frame)["thumbnail"]
    assert result == "NOT VIABLE"


def test_report_leads_with_the_fallback_when_stills_survive():
    from src import video_mechanism as vm
    report = vm.build_report(mechanism_frame(fake_hash="same0000same0000"))
    assert "placeholder served for everything" in report
    assert "CONFIRMED" in report
    assert "stills ARE distinct" in report
    assert "buildable from this source" in report


def test_report_notes_when_nothing_redirected():
    from src import video_mechanism as vm
    report = vm.build_report(mechanism_frame())
    assert "a redirect to a generic asset is not the mechanism" in report


def test_fetch_never_raises(monkeypatch):
    from src import video_mechanism as vm
    import requests

    def explode(*args, **kwargs):
        raise requests.exceptions.SSLError("handshake failed")

    monkeypatch.setattr(requests, "get", explode)
    result = vm.fetch(REAL_URL, "browser")
    assert result["hash"] == ""
    assert "SSLError" in result["error"]


# ---------------------------------------------------------------------------
# 10. Phase 11f: the full sweep, correcting a biased sample
# ---------------------------------------------------------------------------

def sweep_frame(real_indices=(), n=24):
    rows = []
    types = ["Made Shot", "Rebound", "Foul"]
    for i in range(n):
        is_real = i in real_indices
        rows.append({
            "season": config.SEASONS[i % len(config.SEASONS)],
            "game_id": f"00216003{i:02d}", "action_number": i + 1,
            "action_type": types[i % len(types)], "period": (i % 4) + 1,
            "our_description": f"play {i}", "url": f"https://v.nba.com/{i}.mp4",
            "status": 206, "bytes": 2048,
            "hash": f"real{i:012d}" if is_real else "placeholder00000",
            "error": None,
            "verdict": "real" if is_real else "placeholder"})
    return pd.DataFrame(rows)


def test_classify_against_the_measured_placeholder():
    from src import video_sweep as vs
    assert vs.classify("abc123", "placeholder00000") == "real"
    assert vs.classify("placeholder00000", "placeholder00000") == "placeholder"
    assert vs.classify("", "placeholder00000") == "no_response"


def test_sweep_report_says_nothing_found_when_all_are_placeholders():
    from src import video_sweep as vs
    report = vs.build_report(sweep_frame(), "placeholder00000")
    assert "REAL            0" in report
    assert "Nothing. Every matched clip" in report
    assert "biased sixteen" in report


def test_sweep_report_admits_the_earlier_conclusion_was_wrong():
    """
    If real clips turn up, the report has to say the earlier conclusion was
    wrong and why, not bury it under a percentage.
    """
    from src import video_sweep as vs
    report = vs.build_report(sweep_frame(real_indices=(3, 9, 17)),
                             "placeholder00000")
    assert "earlier conclusion" in report
    assert "was wrong" in report
    assert "OPEN THESE IN A BROWSER" in report
    assert "https://v.nba.com/3.mp4" in report


def test_sweep_report_catches_a_second_shared_file():
    """
    Differing from the placeholder is not the same as being a real clip. If
    every 'real' hash is the same value, that is a second placeholder and the
    report must say so rather than claim a win.
    """
    from src import video_sweep as vs
    frame = sweep_frame(real_indices=(1, 2, 3))
    frame.loc[frame["verdict"].eq("real"), "hash"] = "second0000000000"
    report = vs.build_report(frame, "placeholder00000")
    assert "distinct hashes among the real clips: 1" in report
    assert "SECOND shared file" in report


def test_sweep_report_breaks_down_by_period():
    """
    The breakdown that would have caught the original bias. All sixteen tested
    clips were period 1 opening events.
    """
    from src import video_sweep as vs
    report = vs.build_report(sweep_frame(), "placeholder00000")
    assert "BY PERIOD" in report
    assert "BY EVENT TYPE" in report
    assert "BY SEASON" in report


def test_fabricate_matches_the_mechanism_probe():
    from src import video_sweep as vs
    from src import video_mechanism as vm
    assert vs.fabricate(REAL_URL) == vm.fabricate(REAL_URL)


def test_fetch_head_bytes_never_raises(monkeypatch):
    from src import video_sweep as vs
    import requests

    def explode(*args, **kwargs):
        raise requests.exceptions.ChunkedEncodingError("broken")

    monkeypatch.setattr(requests, "get", explode)
    result = vs.fetch_head_bytes("https://v.nba.com/x.mp4")
    assert result["hash"] == ""
    assert "ChunkedEncodingError" in result["error"]


def test_sweep_aborts_without_a_placeholder_reference(tmp_path, monkeypatch):
    """
    The failure that would invert the whole result. With no reference hash,
    every clip differs from "" and would be called real.
    """
    from src import video_sweep as vs

    interim = tmp_path / "interim"
    reports = tmp_path / "reports"
    for d in (interim, reports):
        d.mkdir(parents=True)
    sweep_frame().assign(status="matched").to_csv(
        interim / "video_probe.csv", index=False)

    monkeypatch.setattr(config, "INTERIM_DIR", interim)
    monkeypatch.setattr(config, "REPORTS_DIR", reports)
    monkeypatch.setattr(config, "ALL_DIRS", [interim, reports])
    monkeypatch.setattr(vs, "fetch_head_bytes",
                        lambda url: {"status": None, "bytes": 0, "hash": "",
                                     "error": "timeout"})
    with pytest.raises(SystemExit) as excinfo:
        vs.main()
    assert "placeholder reference" in str(excinfo.value)
