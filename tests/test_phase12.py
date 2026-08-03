"""
Tests for Phase 12: official, embeddable game highlights.

The properties worth pinning are the ones that keep this inside the rules the
project was given.

  1. METADATA ONLY. No download, no scrape, no re-host. Asserted structurally:
     the module must not reference any downloader, and the only host it talks
     to is the official API.

  2. A REEL FOR THE WRONG GAME IS NOT A MATCH. Five independent conditions,
     each tested by breaking it on its own.

  3. THE KEY IS NEVER LEAKED. Errors must not echo it, and the loader must
     fail with instructions rather than proceeding without one.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src import config, youtube_probe as yt


GAME = {
    "season": "2023-24",
    "game_id": "0022300906",
    "game_date": datetime(2023, 12, 8, tzinfo=timezone.utc),
    "opponent_tricode": "DEN",
    "matchup": "BOS @ DEN",
    "is_home": False,
}

OFFICIAL = {"UC_nba", "UC_celtics"}


def item(title="Celtics vs Nuggets Full Game Highlights | Dec 8, 2023",
         channel_id="UC_nba", published="2023-12-08T06:30:00Z",
         embeddable=True, privacy="public", region=None, video_id="abc123"):
    return {
        "id": video_id,
        "snippet": {"title": title, "channelId": channel_id,
                    "channelTitle": "NBA", "publishedAt": published},
        "status": {"embeddable": embeddable, "privacyStatus": privacy},
        "contentDetails": {"duration": "PT9M12S",
                           **({"regionRestriction": region} if region else {})},
    }


# ---------------------------------------------------------------------------
# 1. Metadata only
# ---------------------------------------------------------------------------

def test_module_contains_no_downloader_or_scraper():
    """
    The instruction was explicit: no downloading, scraping or re-hosting. This
    fails if anyone later reaches for the obvious tools.
    """
    import inspect
    source = inspect.getsource(yt).lower()
    for forbidden in ("yt_dlp", "youtube_dl", "pytube", "ffmpeg",
                      "beautifulsoup", "bs4", "selenium", "scrapy"):
        assert forbidden not in source, f"{forbidden} must not appear here"


def test_only_the_official_api_host_is_contacted():
    assert yt.API_ROOT == "https://www.googleapis.com/youtube/v3"
    import inspect
    source = inspect.getsource(yt)
    assert "youtube.com/watch?v=" in source, (
        "a human-readable watch link in the report is fine; it is a link, "
        "not a fetch")
    assert "urlopen" in source
    # The only place a URL is built for fetching is api_get.
    assert source.count("urlopen(") == 1


def test_channel_ids_are_resolved_not_hardcoded():
    """
    Hardcoded channel ids are how a probe silently starts trusting the wrong
    channel. The handles are the only constants allowed.
    """
    import inspect
    source = inspect.getsource(yt)
    assert "forHandle" in source
    assert yt.OFFICIAL_HANDLES == ("@NBA", "@celtics")
    # No 24-character UC... literals anywhere.
    import re
    assert not re.search(r"['\"]UC[A-Za-z0-9_-]{22}['\"]", source)


# ---------------------------------------------------------------------------
# 2. Matching
# ---------------------------------------------------------------------------

def test_a_clean_candidate_matches():
    result = yt.classify(item(), GAME, OFFICIAL)
    assert result["verdict"] == "matched"
    assert result["reasons"] == ""
    assert result["watch_url"].endswith("abc123")


def test_an_unofficial_channel_is_rejected():
    result = yt.classify(item(channel_id="UC_some_fan"), GAME, OFFICIAL)
    assert result["verdict"] == "rejected"
    assert "not an official channel" in result["reasons"]


def test_embedding_disabled_is_rejected():
    """The whole premise. If it cannot be embedded it cannot be used."""
    result = yt.classify(item(embeddable=False), GAME, OFFICIAL)
    assert result["verdict"] == "rejected"
    assert "embedding disabled" in result["reasons"]


def test_a_non_public_video_is_rejected():
    result = yt.classify(item(privacy="unlisted"), GAME, OFFICIAL)
    assert result["verdict"] == "rejected"
    assert "privacy is unlisted" in result["reasons"]


def test_a_video_published_long_after_the_game_is_rejected():
    """
    A season retrospective naming both teams would otherwise sail through and
    sit under the wrong scoreboard.
    """
    result = yt.classify(item(published="2024-06-01T00:00:00Z"), GAME,
                         OFFICIAL)
    assert result["verdict"] == "rejected"
    assert "outside the game-date window" in result["reasons"]


def test_a_video_published_before_the_game_is_rejected():
    result = yt.classify(item(published="2023-11-20T00:00:00Z"), GAME,
                         OFFICIAL)
    assert result["verdict"] == "rejected"
    assert "outside the game-date window" in result["reasons"]


def test_a_title_naming_only_one_team_is_rejected():
    """
    "Celtics highlights" could be any of 82 games that season. Both teams or
    nothing.
    """
    result = yt.classify(item(title="Celtics Best Plays"), GAME, OFFICIAL)
    assert result["verdict"] == "rejected"
    assert "does not name both teams" in result["reasons"]


def test_a_title_naming_the_wrong_opponent_is_rejected():
    result = yt.classify(item(title="Celtics vs Lakers Highlights"), GAME,
                         OFFICIAL)
    assert result["verdict"] == "rejected"
    assert "does not name both teams" in result["reasons"]


def test_city_names_count_as_naming_the_team():
    assert yt.title_names_both_teams("Boston at Denver, full highlights", "DEN")
    assert yt.title_names_both_teams("Celtics vs Nuggets", "DEN")
    assert not yt.title_names_both_teams("Nuggets vs Suns", "DEN")


def test_two_word_city_names_are_matched():
    assert yt.title_names_both_teams("Celtics vs Golden State", "GSW")
    assert yt.title_names_both_teams("Celtics vs Trail Blazers", "POR")
    assert yt.title_names_both_teams("Boston Celtics at Oklahoma City", "OKC")


def test_every_opponent_in_the_dataset_has_a_name_mapping():
    """
    29 opponents appear across the 636 games. A missing entry would silently
    reject every game against that team.
    """
    expected = {"ATL", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
                "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP",
                "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR",
                "UTA", "WAS"}
    assert expected.issubset(set(yt.TEAM_NAMES))


def test_region_restrictions_are_reported_not_swallowed():
    """
    Embeddable but blocked where the viewer is means unusable. It does not
    fail the match, because that depends on the viewer, but it must surface.
    """
    result = yt.classify(item(region={"blocked": ["US", "CA"]}), GAME,
                         OFFICIAL)
    assert result["verdict"] == "matched"
    assert "blocked in 2 region" in result["region_restriction"]

    allowed = yt.classify(item(region={"allowed": ["DE"]}), GAME, OFFICIAL)
    assert "allowed in only 1 region" in allowed["region_restriction"]


def test_a_malformed_publish_date_is_rejected_rather_than_crashing():
    result = yt.classify(item(published="not a date"), GAME, OFFICIAL)
    assert result["verdict"] == "rejected"
    assert "outside the game-date window" in result["reasons"]


def test_all_five_failures_are_reported_together():
    """A near miss should be legible, not just absent."""
    result = yt.classify(
        item(title="Some Other Game", channel_id="UC_fan", embeddable=False,
             privacy="private", published="2020-01-01T00:00:00Z"),
        GAME, OFFICIAL)
    for reason in ("not an official channel", "embedding disabled",
                   "privacy is private", "outside the game-date window",
                   "does not name both teams"):
        assert reason in result["reasons"]


# ---------------------------------------------------------------------------
# 3. Credentials
# ---------------------------------------------------------------------------

def test_missing_key_fails_with_instructions_not_a_stack_trace(monkeypatch,
                                                               tmp_path):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        yt.load_api_key()
    message = str(excinfo.value)
    assert "YouTube Data API v3" in message
    assert ".youtube_api_key" in message
    assert "never downloads or scrapes" in message


def test_key_is_read_from_the_environment_first(monkeypatch, tmp_path):
    monkeypatch.setenv("YOUTUBE_API_KEY", "from-env")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    assert yt.load_api_key() == "from-env"


def test_key_is_read_from_the_gitignored_file(monkeypatch, tmp_path):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    (tmp_path / yt.KEY_FILE).write_text("from-file\n")
    assert yt.load_api_key() == "from-file"


def test_the_key_file_is_gitignored():
    """
    A committed API key is a real incident, not a style problem.
    """
    ignore = (config.PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert yt.KEY_FILE in ignore


def test_errors_never_echo_the_key(monkeypatch):
    """
    The API key travels in the query string, so a urllib error message can
    contain the full URL. It must be redacted before it reaches a log or a
    report.
    """
    def explode(endpoint, key, **params):
        raise RuntimeError(f"HTTP 403 for {yt.API_ROOT}/search?key={key}&q=x")

    monkeypatch.setattr(yt, "api_get", explode)
    payload, error = yt.safe_api_get("search", "SUPER_SECRET_KEY", q="x")
    assert payload is None
    assert "SUPER_SECRET_KEY" not in error
    assert "<redacted>" in error


# ---------------------------------------------------------------------------
# 4. Sampling and reporting
# ---------------------------------------------------------------------------

def index_frame():
    rows = []
    for s_index, season in enumerate(config.SEASONS):
        for g in range(41):
            rows.append({
                "SEASON": season,
                "GAME_ID": int(f"002{s_index}6{g:05d}"),
                "GAME_DATE": (pd.Timestamp("2016-10-25")
                              + pd.Timedelta(days=365 * s_index + g)),
                "OPPONENT_ABBREV": sorted(yt.TEAM_NAMES)[g % 29],
                "MATCHUP": "BOS vs. XXX",
                "IS_HOME": True,
            })
    return pd.DataFrame(rows)


def test_three_games_come_from_three_different_seasons():
    games = yt.pick_games(index_frame())
    assert len(games) == 3
    assert len({g["season"] for g in games}) == 3


def test_the_sample_spans_early_middle_and_late_seasons():
    """
    Two earlier phases were misled by samples that clustered. This one is
    explicit about spreading.
    """
    games = yt.pick_games(index_frame())
    seasons = [g["season"] for g in games]
    assert seasons[0] == config.SEASONS[0]
    assert seasons[-1] == config.SEASONS[-1]


def test_sampling_is_deterministic():
    first = yt.pick_games(index_frame())
    second = yt.pick_games(index_frame().sample(frac=1.0, random_state=3))
    assert [g["game_id"] for g in first] == [g["game_id"] for g in second]


def test_report_says_matched_when_a_reel_is_found():
    games = [GAME]
    frame = pd.DataFrame([{**yt.game_fields(GAME), **yt.classify(
        item(), GAME, OFFICIAL), "searched_channel": ""}])
    report = yt.build_report(frame, games,
                             {"@NBA": {"channel_id": "UC_nba",
                                       "title": "NBA"}})
    assert "games with a usable reel 1 of 1" in report
    assert "MATCHED" in report
    assert "Game highlights" in report
    assert "Never 'Current play'" in report


def test_report_explains_why_when_nothing_matched():
    games = [GAME]
    frame = pd.DataFrame([{**yt.game_fields(GAME), **yt.classify(
        item(embeddable=False), GAME, OFFICIAL), "searched_channel": ""}])
    report = yt.build_report(frame, games,
                             {"@NBA": {"channel_id": "UC_nba",
                                       "title": "NBA"}})
    assert "NO USABLE REEL" in report
    assert "embedding disabled" in report


def test_report_flags_that_no_channel_resolved():
    """
    With no official channel resolved every candidate is rejected for the same
    reason, which would read as "no highlights exist". It is not that.
    """
    report = yt.build_report(pd.DataFrame(), [GAME], {})
    assert "NONE RESOLVED" in report
    assert "proves nothing" in report


def test_report_states_three_games_is_not_a_coverage_figure():
    report = yt.build_report(pd.DataFrame(), [GAME], {})
    assert "not a coverage figure" in report
    assert "636" in report


def test_report_requires_the_panel_to_degrade_to_nothing():
    report = yt.build_report(pd.DataFrame(), [GAME], {})
    assert "degrade to nothing" in report


# ---------------------------------------------------------------------------
# 5. Phase 12b: listing instead of guessing at queries
# ---------------------------------------------------------------------------

def test_listing_is_still_metadata_only():
    from src import youtube_listing as yl
    import inspect
    source = inspect.getsource(yl).lower()
    for forbidden in ("yt_dlp", "youtube_dl", "pytube", "ffmpeg",
                      "beautifulsoup", "bs4", "selenium", "scrapy"):
        assert forbidden not in source


def test_listing_uses_no_search_query_at_all():
    """
    The entire point. A query is what hid the reel last time, so the listing
    must order by date and take everything in the window.
    """
    from src import youtube_listing as yl
    import inspect
    source = inspect.getsource(yl.list_uploads)
    assert 'order="date"' in source
    assert "q=" not in source
    assert f"maxResults=MAX_RESULTS" in source or "maxResults=50" in source


def test_region_detail_prints_codes_not_a_count():
    """
    Phase 12 reported "allowed in only 24 region(s)" and never said which. If
    the US is missing the feature is dead here, so the codes must survive.
    """
    from src import youtube_listing as yl
    allowed = yl.region_detail(
        {"contentDetails": {"regionRestriction": {"allowed": ["GB", "DE", "FR"]}}})
    assert allowed["region_mode"] == "allowlist"
    assert allowed["region_codes"] == "DE,FR,GB"
    assert allowed["us_ok"] is False

    with_us = yl.region_detail(
        {"contentDetails": {"regionRestriction": {"allowed": ["US", "CA"]}}})
    assert with_us["us_ok"] is True


def test_region_detail_handles_a_blocklist():
    from src import youtube_listing as yl
    blocked = yl.region_detail(
        {"contentDetails": {"regionRestriction": {"blocked": ["US"]}}})
    assert blocked["region_mode"] == "blocklist"
    assert blocked["us_ok"] is False

    elsewhere = yl.region_detail(
        {"contentDetails": {"regionRestriction": {"blocked": ["CN"]}}})
    assert elsewhere["us_ok"] is True


def test_no_restriction_is_the_good_case_and_reads_as_unknown_not_false():
    """
    us_ok must be None, not False, when there is no restriction. False would
    make an unrestricted video look blocked.
    """
    from src import youtube_listing as yl
    clean = yl.region_detail({"contentDetails": {}})
    assert clean["region_mode"] == "none"
    assert clean["us_ok"] is None


def test_reel_detection_separates_game_reels_from_play_compilations():
    from src import youtube_listing as yl
    assert yl.looks_like_a_game_reel(
        "CELTICS at NETS | FULL GAME HIGHLIGHTS | March 11, 2021")
    assert yl.looks_like_a_game_reel("Celtics vs Knicks Game Recap")
    assert not yl.looks_like_a_game_reel("Top 10 NBA Plays of the Night: 01.18.17")
    assert not yl.looks_like_a_game_reel("Marvin Williams Tomahawk Slam in Boston")
    assert not yl.looks_like_a_game_reel(None)


def listing_frame(rows):
    base = {**yt.game_fields(GAME), "handle": "@NBA", "video_id": "v",
            "title": "t", "published_at": "2023-12-08T06:00:00Z",
            "duration": "PT9M", "embeddable": True, "privacy": "public",
            "names_both_teams": True, "looks_like_reel": True,
            "phase12_verdict": "matched", "phase12_reasons": "",
            "watch_url": "https://www.youtube.com/watch?v=v",
            "region_mode": "none", "region_codes": "", "us_ok": None}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_listing_report_distinguishes_a_rule_miss_from_an_absence():
    """
    The distinction the whole phase exists to draw.
    """
    from src import youtube_listing as yl
    frame = listing_frame([{
        "title": "CELTICS at NUGGETS | FULL GAME HIGHLIGHTS",
        "phase12_verdict": "rejected",
        "phase12_reasons": "published outside the game-date window"}])
    report = yl.build_report(frame, [GAME], {})
    assert "EXISTS but the Phase 12 rule" in report
    assert "The rule is wrong for this game" in report


def test_listing_report_calls_an_absence_an_absence():
    from src import youtube_listing as yl
    frame = listing_frame([{
        "title": "Top 10 Plays of the Night", "names_both_teams": False,
        "looks_like_reel": False, "phase12_verdict": "rejected",
        "phase12_reasons": "title does not name both teams"}])
    report = yl.build_report(frame, [GAME], {})
    assert "the answer is availability, not" in report


def test_listing_report_shouts_when_the_us_is_excluded():
    from src import youtube_listing as yl
    frame = listing_frame([{
        "title": "CELTICS at NETS | FULL GAME HIGHLIGHTS",
        "region_mode": "allowlist", "region_codes": "DE,FR,GB", "us_ok": False}])
    report = yl.build_report(frame, [GAME], {})
    assert "US playable: False" in report
    assert "NOT playable in the United States" in report
    assert "DE,FR,GB" in report


def test_listing_report_stays_quiet_when_nothing_is_restricted():
    from src import youtube_listing as yl
    report = yl.build_report(listing_frame([{}]), [GAME], {})
    assert "No upload in the window carries any region restriction" in report
    assert "NOT playable in the United States" not in report


def test_listing_report_reports_an_empty_window_honestly():
    from src import youtube_listing as yl
    report = yl.build_report(pd.DataFrame(), [GAME], {})
    assert "genuine absence, not a search miss" in report


# ---------------------------------------------------------------------------
# 6. Phase 12c: constructing the title instead of guessing it
# ---------------------------------------------------------------------------

HOME_GAME = {**GAME, "matchup": "BOS vs. DEN", "is_home": True}
AWAY_GAME = {**GAME, "matchup": "BOS @ DEN", "is_home": False}


def test_home_and_away_is_not_backwards():
    """
    The NBA titles these AWAY at HOME. Getting it the wrong way round searches
    for a title that does not exist, and would look like the reel is missing.
    """
    from src import youtube_targeted as yg
    assert yg.home_and_away(HOME_GAME) == ("Nuggets", "Celtics")
    assert yg.home_and_away(AWAY_GAME) == ("Celtics", "Nuggets")


def test_the_primary_variant_reproduces_the_observed_convention():
    """
    Built against a title actually seen in the March 2021 listing:
    "CELTICS at NETS | FULL GAME HIGHLIGHTS | March 11, 2021".
    """
    from src import youtube_targeted as yg
    from datetime import datetime, timezone
    game = {"opponent_tricode": "BKN", "is_home": False,
            "game_date": datetime(2021, 3, 11, tzinfo=timezone.utc)}
    primary = yg.title_variants(game)[0][0]
    assert primary == "Celtics at Nets | FULL GAME HIGHLIGHTS | March 11, 2021"


def test_variants_are_ordered_most_specific_first():
    from src import youtube_targeted as yg
    variants = yg.title_variants(HOME_GAME)
    assert len(variants) >= 3
    assert "FULL GAME HIGHLIGHTS" in variants[0][0]
    # The last one is the loosest and must not carry the exact date.
    assert "December 8, 2023" not in variants[-1][0]


def test_every_variant_names_both_teams():
    """
    Otherwise a variant could surface a video the acceptance rule then
    rejects, which wastes quota and muddies the report.
    """
    from src import youtube_targeted as yg
    for game in (HOME_GAME, AWAY_GAME):
        for query, _ in yg.title_variants(game):
            assert yt.title_names_both_teams(query, game["opponent_tricode"])


def test_month_names_are_correct_at_the_boundaries():
    from src import youtube_targeted as yg
    from datetime import datetime, timezone
    for month, name in ((1, "January"), (6, "June"), (12, "December")):
        game = {"opponent_tricode": "DEN", "is_home": True,
                "game_date": datetime(2024, month, 5, tzinfo=timezone.utc)}
        assert f"{name} 5, 2024" in yg.title_variants(game)[0][0]


def test_targeted_search_does_not_filter_by_channel():
    """
    Deliberate. Relevance across all of YouTube finds the reel; the official
    channel test is applied afterwards to what comes back, so precision is
    kept without letting a channel filter suppress relevance.
    """
    from src import youtube_targeted as yg
    import inspect
    source = inspect.getsource(yg.search_titles)
    assert "channelId" not in source
    assert "q=query" in source


def test_targeted_module_is_still_metadata_only():
    from src import youtube_targeted as yg
    import inspect
    source = inspect.getsource(yg).lower()
    for forbidden in ("yt_dlp", "youtube_dl", "pytube", "ffmpeg",
                      "beautifulsoup", "bs4", "selenium", "scrapy"):
        assert forbidden not in source


def targeted_frame(rows):
    base = {**yt.game_fields(GAME), "found_by": "2021 convention",
            "video_id": "v", "title": "t", "channel_title": "NBA",
            "channel_id": "UC_nba", "published_at": "2023-12-08T06:00:00Z",
            "embeddable": True, "privacy": "public", "duration": "PT9M",
            "region_restriction": "", "verdict": "matched", "reasons": "",
            "watch_url": "https://www.youtube.com/watch?v=v"}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_targeted_report_records_which_convention_worked():
    """
    A 636-game run needs to know which title format applies to which era.
    """
    from src import youtube_targeted as yg
    report = yg.build_report(targeted_frame([{}]), [GAME])
    assert "WHICH TITLE CONVENTION WORKED" in report
    assert "2021 convention" in report
    assert "2023-24" in report


def test_targeted_report_owns_both_earlier_mistakes():
    from src import youtube_targeted as yg
    report = yg.build_report(pd.DataFrame(), [GAME])
    assert "invented query" in report
    assert "search index, not an enumeration" in report


def test_targeted_report_states_absence_is_weak_evidence():
    """
    The honest framing. Not finding a reel under four guessed titles is not
    the same as there being none.
    """
    from src import youtube_targeted as yg
    report = yg.build_report(pd.DataFrame(), [GAME])
    assert "weaker evidence" in report
    assert "NO USABLE REEL FROM ANY VARIANT" in report


def test_targeted_report_lists_the_titles_it_searched():
    """So a failure can be judged rather than taken on trust."""
    from src import youtube_targeted as yg
    report = yg.build_report(pd.DataFrame(), [GAME])
    assert "titles searched:" in report
    assert "FULL GAME HIGHLIGHTS" in report


# ---------------------------------------------------------------------------
# 7. Phase 12d: coverage by season, and the quota trap
# ---------------------------------------------------------------------------

def test_quota_exhaustion_is_recognised():
    """
    A spent quota returns 403 for everything. Reading that as "no video" would
    turn the tail of a run into a fabricated coverage cliff. This project has
    now been caught by a failure-that-looks-like-a-finding three times.
    """
    from src import youtube_coverage as yc
    assert yc.looks_like_quota_exhaustion("HTTPError: HTTP Error 403: Forbidden")
    assert yc.looks_like_quota_exhaustion("quotaExceeded")
    assert yc.looks_like_quota_exhaustion("dailyLimitExceeded")
    assert not yc.looks_like_quota_exhaustion("HTTP Error 404: Not Found")
    assert not yc.looks_like_quota_exhaustion("")
    assert not yc.looks_like_quota_exhaustion(None)


def test_coverage_sample_is_three_per_season_across_all_eight():
    from src import youtube_coverage as yc
    games = yc.pick_games(index_frame())
    assert len({g["season"] for g in games}) == len(config.SEASONS)
    counts = {}
    for game in games:
        counts[game["season"]] = counts.get(game["season"], 0) + 1
    assert set(counts.values()) == {yc.GAMES_PER_SEASON}


def test_coverage_sample_spreads_through_each_season():
    """
    The mistake made twice in Phase 11: a sample that clustered at the start.
    """
    from src import youtube_coverage as yc
    games = yc.pick_games(index_frame())
    for season in config.SEASONS:
        dates = sorted(g["game_date"] for g in games
                       if g["season"] == season)
        assert len(set(dates)) == yc.GAMES_PER_SEASON
    assert {g["position_in_season"] for g in games} == set(yc.SEASON_POSITIONS)


def test_coverage_sampling_is_deterministic():
    from src import youtube_coverage as yc
    first = yc.pick_games(index_frame())
    second = yc.pick_games(index_frame().sample(frac=1.0, random_state=11))
    assert [g["game_id"] for g in first] == [g["game_id"] for g in second]


def coverage_frame(rows):
    base = {"season": "2020-21", "game_id": "x", "game_date": "2021-03-11",
            "matchup": "BOS @ BKN", "opponent_tricode": "BKN",
            "away": "Celtics", "home": "Nets", "position_in_season": 0.5,
            "video_id": None, "title": None, "channel_title": None,
            "published_at": None, "duration": None, "found_by": None,
            "watch_url": None, "region_restriction": None,
            "unofficial_candidates": 0, "outcome": "nothing_found", "note": ""}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_untested_games_are_never_counted_as_absences():
    """
    The whole point of the quota guard. `not_tested` must be excluded from the
    denominator, or a quota stop reads as a coverage collapse.
    """
    from src import youtube_coverage as yc
    frame = coverage_frame([
        {"outcome": "official_reel", "found_by": "2021 convention",
         "title": "CELTICS at NETS | FULL GAME HIGHLIGHTS"},
        {"outcome": "not_tested", "season": "2023-24"},
        {"outcome": "not_tested", "season": "2023-24"},
    ])
    report = yc.build_report(frame)
    assert "games tested       1 of 3" in report
    assert "NOT TESTED" in report
    assert "never as absences" in report
    # The one tested game found a reel, so the rate is 100%, not 33%.
    assert "(100% of tested)" in report


def test_report_separates_unofficial_only_from_nothing_found():
    """
    'The reel exists but only on a re-upload channel' is a different fact from
    'no coverage', and the decision differs.
    """
    from src import youtube_coverage as yc
    frame = coverage_frame([
        {"outcome": "unofficial_only", "unofficial_candidates": 4},
        {"outcome": "nothing_found"},
    ])
    report = yc.build_report(frame)
    assert "unofficial only    1" in report
    assert "nothing found      1" in report
    assert "different fact from no coverage" in report


def test_report_names_the_usable_seasons():
    from src import youtube_coverage as yc
    frame = coverage_frame([
        {"outcome": "official_reel", "season": "2020-21",
         "found_by": "2021 convention", "title": "t"},
        {"outcome": "unofficial_only", "season": "2016-17"},
    ])
    report = yc.build_report(frame)
    assert "Seasons with at least one official reel: 2020-21" in report
    assert "NOTHING AT ALL elsewhere" in report
    assert "never be" in report and "Current play" in report


def test_report_says_so_when_no_season_works():
    from src import youtube_coverage as yc
    report = yc.build_report(coverage_frame([{"outcome": "unofficial_only"},
                                             {"outcome": "nothing_found"}]))
    assert "nothing to" in report
    assert "build a highlights panel on" in report


def test_report_warns_that_three_games_is_not_a_coverage_figure():
    from src import youtube_coverage as yc
    report = yc.build_report(coverage_frame([{"outcome": "official_reel",
                                              "found_by": "c", "title": "t"}]))
    assert "wide error bars" in report
    assert "could still be 60%" in report


def test_run_marks_everything_after_a_quota_stop_as_untested(monkeypatch):
    """
    End to end: the third game kills the quota, and games four and five must
    come back untested rather than empty.
    """
    from src import youtube_coverage as yc
    games = yc.pick_games(index_frame())[:5]
    calls = {"n": 0}

    def fake_probe(key, game, official_ids):
        calls["n"] += 1
        if calls["n"] < 3:
            return {**yc.base_row(game), "outcome": "official_reel",
                    "note": ""}, False
        return {**yc.base_row(game), "outcome": "not_tested",
                "note": "quota exhausted before this game"}, True

    monkeypatch.setattr(yc, "probe_game", fake_probe)
    frame = yc.run("k", games, {"@NBA": {"channel_id": "UC_nba",
                                         "title": "NBA"}})
    assert list(frame["outcome"]) == ["official_reel", "official_reel",
                                      "not_tested", "not_tested", "not_tested"]
    assert calls["n"] == 3, "probing must stop, not keep burning calls"


def test_coverage_module_is_still_metadata_only():
    from src import youtube_coverage as yc
    import inspect
    source = inspect.getsource(yc).lower()
    for forbidden in ("yt_dlp", "youtube_dl", "pytube", "ffmpeg",
                      "beautifulsoup", "bs4", "selenium", "scrapy"):
        assert forbidden not in source
