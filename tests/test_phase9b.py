"""
Tests for Phase 9b: the serving build.

Two failure modes matter here and neither raises on its own.

The first is the game-id dtype trap: `game_index.csv` stores GAME_ID as an
integer and every parquet stores it as a zero-padded string, so a direct join
returns nothing and reports nothing.

The second is a merge that silently drops rows, which shows up as a timeline
with holes rather than as an error.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import build_serving


# ---------------------------------------------------------------------------
# The join-key trap
# ---------------------------------------------------------------------------

def test_integer_and_string_game_ids_normalise_to_the_same_value():
    """
    The exact failure the schema probe found. 21600006 and '0021600006' are the
    same game and must become the same key.
    """
    from_int = build_serving.normalise_game_id([21600006])
    from_str = build_serving.normalise_game_id(["0021600006"])
    assert from_int.iloc[0] == from_str.iloc[0] == "0021600006"


def test_float_game_ids_do_not_keep_a_decimal_suffix():
    """
    A CSV read can produce 21600006.0. Zero-padding that string without
    stripping the suffix yields '21600006.0', which matches nothing.
    """
    assert build_serving.normalise_game_id([21600006.0]).iloc[0] == "0021600006"


def test_already_padded_ids_are_left_alone():
    values = ["0021600006", "0022301202"]
    out = list(build_serving.normalise_game_id(values))
    assert out == values


def test_normalised_ids_join_across_the_two_dtypes():
    """The end-to-end version: a merge that would otherwise return zero rows."""
    parquet_side = pd.DataFrame({"game_id": ["0021600006"], "events": [486]})
    csv_side = pd.DataFrame({"GAME_ID": [21600006], "MATCHUP": ["BOS vs. BKN"]})
    csv_side["game_id"] = build_serving.normalise_game_id(csv_side["GAME_ID"])

    naive = parquet_side.merge(
        csv_side.assign(game_id=csv_side["GAME_ID"].astype(str)),
        on="game_id", how="inner", suffixes=("", "_y"))
    assert len(naive) == 0          # this is what the bug looks like

    correct = parquet_side.merge(csv_side, on="game_id", how="inner")
    assert len(correct) == 1


# ---------------------------------------------------------------------------
# Probabilities must be out of fold, and complete
# ---------------------------------------------------------------------------

def small_frame(n=6):
    return pd.DataFrame({
        "game_id": ["0021600006"] * n,
        "event_index": list(range(n)),
    })


def test_probability_join_preserves_every_row():
    frame = small_frame()
    oof = pd.DataFrame({
        "game_id": ["0021600006"] * 6,
        "event_index": list(range(6)),
        "tier3_celtics": np.linspace(0.4, 0.9, 6),
        "tier2_generic": np.linspace(0.4, 0.9, 6),
    })
    out = build_serving.attach_probabilities(frame, oof)
    assert len(out) == len(frame)
    assert out["tier3_celtics"].notna().all()


def test_a_missing_probability_raises_rather_than_showing_a_default():
    """
    A gap must stop the build. Filling it with 0.5, or with the previous value,
    would put a number on screen that no model produced.
    """
    frame = small_frame()
    oof = pd.DataFrame({
        "game_id": ["0021600006"] * 4,
        "event_index": [0, 1, 2, 3],
        "tier3_celtics": [0.5] * 4,
        "tier2_generic": [0.5] * 4,
    })
    with pytest.raises(ValueError, match="no out-of-fold probability"):
        build_serving.attach_probabilities(frame, oof)


def test_the_dashboard_reads_the_out_of_fold_column_not_a_refit():
    """
    Naming discipline. The dashboard must replay stored out-of-fold
    predictions, never the all-seasons deployment model, which would be
    in-sample for every game in the dataset.
    """
    assert build_serving.PRIMARY_TIER == "tier3_celtics"
    assert build_serving.BASELINE_TIER == "tier2_generic"


# ---------------------------------------------------------------------------
# Lineups
# ---------------------------------------------------------------------------

def test_lineup_strings_split_into_player_ids():
    assert build_serving.split_lineup("1,2,3,4,5") == ["1", "2", "3", "4", "5"]


def test_an_empty_lineup_is_an_empty_list_not_a_list_with_a_blank():
    for value in ("", None, np.nan, float("nan")):
        assert build_serving.split_lineup(value) == []


# ---------------------------------------------------------------------------
# Coverage measured in minutes
# ---------------------------------------------------------------------------

def test_coverage_is_weighted_by_minutes_not_by_headcount():
    """
    The distinction that matters. One starter and one call-up count the same in
    a headcount and very differently in what a viewer actually sees.
    """
    rosters = pd.DataFrame({
        "season": ["2021-22"] * 3,
        "person_id": [1, 2, 3],
        "minutes_played": [36.0, 34.0, 2.0],
    })
    bios = pd.DataFrame({"season": ["2021-22"] * 2, "person_id": [1, 2]})
    coverage = build_serving.bio_coverage_by_minutes(rosters, bios)
    row = coverage.iloc[0]

    assert row["players_without_bio"] == 1
    assert row["players"] == 3
    assert row["minutes_without_bio"] == 2.0
    # A third of the players, but under three percent of the minutes.
    assert row["share_of_minutes"] < 0.03


def test_coverage_reports_the_median_minutes_of_missing_players():
    """
    This is the number that tests the hardship-contract hypothesis. Call-ups
    have a low median; missing rotation players would not.
    """
    rosters = pd.DataFrame({
        "season": ["2021-22"] * 4,
        "person_id": [1, 2, 3, 4],
        "minutes_played": [30.0, 4.0, 6.0, 8.0],
    })
    bios = pd.DataFrame({"season": ["2021-22"], "person_id": [1]})
    coverage = build_serving.bio_coverage_by_minutes(rosters, bios)
    assert coverage.iloc[0]["median_minutes_of_missing"] == 6.0


def test_coverage_matches_within_a_season():
    """A 2023-24 bio row does not cover a 2016-17 appearance."""
    rosters = pd.DataFrame({
        "season": ["2016-17"], "person_id": [9], "minutes_played": [20.0]})
    bios = pd.DataFrame({"season": ["2023-24"], "person_id": [9]})
    coverage = build_serving.bio_coverage_by_minutes(rosters, bios)
    assert coverage.iloc[0]["players_without_bio"] == 1


def test_coverage_is_empty_without_bios():
    assert build_serving.bio_coverage_by_minutes(
        pd.DataFrame(), pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# Player cards
# ---------------------------------------------------------------------------

def roster_row(**overrides):
    row = {
        "person_id": 1628369, "first_name": "Jayson", "family_name": "Tatum",
        "team_tricode": "BOS", "is_celtics_team": True, "jersey_number": "0",
        "coarse_position": "F", "is_starter": True, "minutes_played": 36.5,
        "points": 30, "reboundsTotal": 8, "assists": 4,
        "plusMinusPoints": 12.0, "season": "2016-17",
    }
    row.update(overrides)
    return row


def test_a_player_without_a_bio_still_gets_a_card():
    """
    The headshot URL is built from person_id alone, so a missing bio row costs
    height and the granular position and nothing else.
    """
    rosters = pd.DataFrame([roster_row()])
    players = build_serving.build_player_table(rosters, pd.DataFrame(),
                                               pd.DataFrame())
    card = players["1628369"]
    assert card["name"] == "Jayson Tatum"
    assert card["jersey"] == "0"
    assert card["coarse_position"] == "F"
    assert "1628369.png" in card["headshot"]
    assert card["has_bio"] is False


def test_missing_fields_are_null_rather_than_a_plausible_default():
    """A blank card is honest. An invented height is not."""
    rosters = pd.DataFrame([roster_row()])
    card = build_serving.build_player_table(
        rosters, pd.DataFrame(), pd.DataFrame())["1628369"]
    assert card["height"] is None
    assert card["height_inches"] is None
    assert card["position"] is None
    assert card["player_value"] is None


def test_a_bio_row_supplies_height_and_the_granular_position():
    rosters = pd.DataFrame([roster_row()])
    bios = pd.DataFrame([{
        "season": "2016-17", "person_id": 1628369, "full_name": "Jayson Tatum",
        "position": "F-G", "height": "6-8", "height_inches": 80.0,
    }])
    card = build_serving.build_player_table(rosters, bios,
                                            pd.DataFrame())["1628369"]
    assert card["position"] == "F-G"
    assert card["height"] == "6-8"
    assert card["height_inches"] == 80.0
    assert card["has_bio"] is True


def test_a_bio_row_from_another_season_is_not_used():
    rosters = pd.DataFrame([roster_row()])
    bios = pd.DataFrame([{
        "season": "2023-24", "person_id": 1628369, "full_name": "Jayson Tatum",
        "position": "F-G", "height": "6-8", "height_inches": 80.0,
    }])
    card = build_serving.build_player_table(rosters, bios,
                                            pd.DataFrame())["1628369"]
    assert card["has_bio"] is False
    assert card["height"] is None


def test_player_value_is_attached_when_available():
    rosters = pd.DataFrame([roster_row()])
    values = pd.DataFrame({"person_id": [1628369], "player_value": [0.0421]})
    card = build_serving.build_player_table(rosters, pd.DataFrame(),
                                            values)["1628369"]
    assert card["player_value"] == pytest.approx(0.0421)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_numpy_types_serialise():
    """
    numpy int64 is not JSON serialisable and this bit the metadata writer in
    Phase 4. It must not bite the serving layer too.
    """
    payload = {"a": np.int64(3), "b": np.float64(0.5), "c": np.bool_(True)}
    text = json.dumps(payload, default=build_serving._jsonable)
    assert json.loads(text) == {"a": 3, "b": 0.5, "c": True}


def test_an_unserialisable_type_raises_rather_than_becoming_a_string():
    with pytest.raises(TypeError):
        json.dumps({"x": object()}, default=build_serving._jsonable)


# ---------------------------------------------------------------------------
# End-to-end payload
# ---------------------------------------------------------------------------

def synthetic_game(n=8, celtics_home=True):
    game_id = "0021600006"
    events = pd.DataFrame({
        "game_id": [game_id] * n,
        "event_index": range(n),
        "season": ["2016-17"] * n,
        "game_date": [pd.Timestamp("2016-10-26")] * n,
        "opponent_tricode": ["BKN"] * n,
        "celtics_is_home": [celtics_home] * n,
        "celtics_won": [1] * n,
        "period": [1] * n,
        "clock_raw": ["PT12M00.00S"] * n,
        "seconds_elapsed_game": np.linspace(0, 700, n),
        "celtics_score": range(n),
        "opponent_score": range(n),
        "celtics_margin": [0] * n,
        "description": ["something"] * n,
        "action_type": ["Made Shot"] * n,
        "team_tricode": ["BOS"] * n,
        "person_id": [1628369] * n,
        "shot_result": ["Made"] * n,
        "shot_value": [2] * n,
        "loc_x": [10] * n,
        "loc_y": [20] * n,
        "is_clutch": [False] * n,
        "tier3_celtics": np.linspace(0.4, 0.9, n),
        "tier2_generic": np.linspace(0.4, 0.9, n),
    })
    lineups = pd.DataFrame({
        "game_id": [game_id] * n,
        "event_index": range(n),
        "home_lineup": ["1,2,3,4,5"] * (n // 2) + ["1,2,3,4,6"] * (n - n // 2),
        "away_lineup": ["7,8,9,10,11"] * n,
    })
    rosters = pd.DataFrame([
        roster_row(person_id=1628369, is_celtics_team=True,
                   team_tricode="BOS"),
        roster_row(person_id=201572, first_name="Brook", family_name="Lopez",
                   is_celtics_team=False, team_tricode="BKN"),
    ])
    rosters["team_id"] = [1610612738, 1610612751]
    index_row = pd.Series({"MATCHUP": "BOS vs. BKN"})
    return game_id, events, lineups, rosters, index_row


def test_payload_builds_end_to_end():
    game_id, events, lineups, rosters, index_row = synthetic_game()
    payload = build_serving.build_game_payload(
        game_id, events, lineups, rosters, pd.DataFrame(), pd.DataFrame(),
        index_row, None)

    assert payload["meta"]["game_id"] == game_id
    assert payload["meta"]["opponent"] == "BKN"
    assert len(payload["events"]["wp"]) == len(events)
    assert len(payload["players"]) == 2
    json.dumps(payload, default=build_serving._jsonable)


def test_lineups_are_stored_once_and_referenced_by_index():
    """
    A lineup changes perhaps 40 times in a game. Repeating five ids on every
    one of 486 events is waste the browser has to parse.
    """
    game_id, events, lineups, rosters, index_row = synthetic_game(n=8)
    payload = build_serving.build_game_payload(
        game_id, events, lineups, rosters, pd.DataFrame(), pd.DataFrame(),
        index_row, None)

    # Two distinct Celtics lineups, one opponent lineup, across 8 events.
    assert len(payload["lineup_table"]) == 3
    assert len(payload["events"]["celtics_lineup"]) == 8
    assert len(set(payload["events"]["celtics_lineup"])) == 2
    assert len(set(payload["events"]["opponent_lineup"])) == 1


def test_the_celtics_lineup_follows_home_or_away():
    """
    Getting this backwards would put the opponent's five on the Celtics bench
    for every away game, which is 318 of 636.
    """
    game_id, events, lineups, rosters, index_row = synthetic_game(
        celtics_home=True)
    home_payload = build_serving.build_game_payload(
        game_id, events, lineups, rosters, pd.DataFrame(), pd.DataFrame(),
        index_row, None)
    celtics_at_home = home_payload["lineup_table"][
        home_payload["events"]["celtics_lineup"][0]]

    game_id, events, lineups, rosters, index_row = synthetic_game(
        celtics_home=False)
    away_payload = build_serving.build_game_payload(
        game_id, events, lineups, rosters, pd.DataFrame(), pd.DataFrame(),
        index_row, None)
    celtics_away = away_payload["lineup_table"][
        away_payload["events"]["celtics_lineup"][0]]

    assert celtics_at_home == ["1", "2", "3", "4", "5"]
    assert celtics_away == ["7", "8", "9", "10", "11"]


def test_a_lineup_join_that_misses_events_raises():
    """
    Caught a real bug. A LEFT join does not lose rows when the right side is
    short, it fills them with nulls, so a row-count check passes while the
    events silently have no lineup. The symptom is a null, not a short frame.
    """
    game_id, events, lineups, rosters, index_row = synthetic_game()
    with pytest.raises(ValueError, match="without a lineup"):
        build_serving.build_game_payload(
            game_id, events, lineups.iloc[:3], rosters, pd.DataFrame(),
            pd.DataFrame(), index_row, None)


def test_a_row_count_check_alone_would_not_have_caught_it():
    """
    Demonstrates why the null check is needed, so the guard is not removed
    later as redundant.
    """
    left = pd.DataFrame({"k": [1, 2, 3]})
    right = pd.DataFrame({"k": [1], "v": ["x"]})
    merged = left.merge(right, on="k", how="left")
    assert len(merged) == len(left)          # row count is unchanged
    assert merged["v"].isna().sum() == 2     # but two rows have nothing


def test_payload_records_where_the_probability_came_from():
    """
    A viewer, and a reader of the paper, should be able to tell from the
    payload itself that these are out-of-fold numbers.
    """
    game_id, events, lineups, rosters, index_row = synthetic_game()
    payload = build_serving.build_game_payload(
        game_id, events, lineups, rosters, pd.DataFrame(), pd.DataFrame(),
        index_row, None)
    assert "never saw this season" in payload["meta"]["probability_source"]


def test_jersey_falls_back_to_the_bio_when_the_boxscore_is_blank():
    """
    The boxscore jersey field is blank for entire seasons. The court draws
    jersey numbers, so an empty circle would be a worse answer than a number
    that exists in a second source.
    """
    rosters = pd.DataFrame([roster_row(jersey_number="")])
    bios = pd.DataFrame([{
        "season": "2016-17", "person_id": 1628369, "full_name": "Jayson Tatum",
        "position": "F", "height": "6-8", "height_inches": 80.0, "jersey": 0,
    }])
    card = build_serving.build_player_table(rosters, bios,
                                            pd.DataFrame())["1628369"]
    assert card["jersey"] == "0"


def test_the_boxscore_jersey_wins_when_both_exist():
    rosters = pd.DataFrame([roster_row(jersey_number="7")])
    bios = pd.DataFrame([{
        "season": "2016-17", "person_id": 1628369, "full_name": "Jayson Tatum",
        "position": "F", "height": "6-8", "height_inches": 80.0, "jersey": 99,
    }])
    card = build_serving.build_player_table(rosters, bios,
                                            pd.DataFrame())["1628369"]
    assert card["jersey"] == "7"


def test_a_player_with_no_jersey_anywhere_gets_null():
    rosters = pd.DataFrame([roster_row(jersey_number="")])
    card = build_serving.build_player_table(rosters, pd.DataFrame(),
                                            pd.DataFrame())["1628369"]
    assert card["jersey"] is None


# ---------------------------------------------------------------------------
# Jersey numbers
# ---------------------------------------------------------------------------

def test_jersey_float_artefact_is_stripped():
    """
    player_bios.csv stores jerseys as clean text, but 14 of its 4,009 rows are
    blank, so pandas types the column float64 and jersey 7 arrives as 7.0.
    That reached the dashboard as "7.0" on every player card.
    """
    assert build_serving.normalise_jersey(7.0) == "7"
    assert build_serving.normalise_jersey("7.0") == "7"
    assert build_serving.normalise_jersey(50.0) == "50"
    assert build_serving.normalise_jersey("0.0") == "0"


def test_double_zero_is_not_collapsed_to_zero():
    """
    The reason the READ is done as text rather than fixed afterwards.

    "00" is a real NBA jersey and a different one from "0". Once a value has
    been through float64 both are 0.0 and no amount of later cleaning can tell
    them apart. This asserts the normaliser never destroys the distinction on a
    value that is still text.
    """
    assert build_serving.normalise_jersey("00") == "00"
    assert build_serving.normalise_jersey("0") == "0"
    assert build_serving.normalise_jersey("00") != build_serving.normalise_jersey("0")


def test_the_bios_read_types_jersey_as_text():
    """
    A test on the source of the bug rather than its symptom. If someone later
    drops the dtype, this fails.
    """
    import inspect
    source = inspect.getsource(build_serving)
    assert 'dtype={"jersey": "string"}' in source


def test_blank_and_missing_jerseys_become_empty():
    import numpy as np
    for blank in (None, "", "   ", np.nan, float("nan"), "nan", "<NA>"):
        assert build_serving.normalise_jersey(blank) == "", repr(blank)


def test_a_non_numeric_jersey_is_left_alone():
    """Nothing here should be inventing or discarding an unusual value."""
    assert build_serving.normalise_jersey("1A") == "1A"
    assert build_serving.normalise_jersey(" 23 ") == "23"


def test_real_bios_file_round_trips_when_read_as_text(tmp_path):
    """
    End to end on the actual failure: blanks in the column force float64, and
    reading as text is what prevents it.
    """
    import pandas as pd
    path = tmp_path / "bios.csv"
    path.write_text("person_id,jersey\n1,7\n2,00\n3,\n4,50\n")

    naive = pd.read_csv(path)
    assert str(naive["jersey"].dtype).startswith("float"), (
        "the fixture must reproduce the float inference the bug depends on")
    assert build_serving.normalise_jersey(naive["jersey"].iloc[1]) == "0", (
        "read naively, 00 has already become 0 and is unrecoverable")

    typed = pd.read_csv(path, dtype={"jersey": "string"})
    assert build_serving.normalise_jersey(typed["jersey"].iloc[0]) == "7"
    assert build_serving.normalise_jersey(typed["jersey"].iloc[1]) == "00"
    assert build_serving.normalise_jersey(typed["jersey"].iloc[2]) == ""
    assert build_serving.normalise_jersey(typed["jersey"].iloc[3]) == "50"


# ---------------------------------------------------------------------------
# Season-correct headshots
#
# The dashboard was showing every player in the jersey he wears TODAY, so a
# 2017-18 game had Kyrie Irving in a Mavericks shirt. scripts/39 fetches and
# confirms a season photo for each player-season; these tests cover the lookup
# that decides which URL a given player gets in a given game.
#
# The case that matters most is a mid-season trade. That player has two
# confirmed rows for one season, and picking either one blindly would put him
# in the wrong shirt for half his games.
# ---------------------------------------------------------------------------

def _headshot_frame():
    return pd.DataFrame([
        # Stayed put all season.
        {"person_id": 1, "season": "2017-18", "team_abbrev": "BOS",
         "url": "https://cdn/1610612738/2017/1040x760/1.png", "usable": True},
        # Traded mid-season: two confirmed rows, one season.
        {"person_id": 2, "season": "2017-18", "team_abbrev": "CLE",
         "url": "https://cdn/1610612739/2017/1040x760/2.png", "usable": True},
        {"person_id": 2, "season": "2017-18", "team_abbrev": "BOS",
         "url": "https://cdn/1610612738/2017/1040x760/2.png", "usable": True},
        # Fetched and refused: 403, no image. Must never be offered.
        {"person_id": 3, "season": "2017-18", "team_abbrev": "BOS",
         "url": "https://cdn/1610612738/2017/1040x760/3.png", "usable": False},
    ])


def test_season_headshot_uses_the_team_the_player_was_on_in_that_game():
    lookup = build_serving.build_headshot_lookup(_headshot_frame())
    assert build_serving.season_headshot(lookup, 2, "2017-18", "BOS") == (
        "https://cdn/1610612738/2017/1040x760/2.png")
    assert build_serving.season_headshot(lookup, 2, "2017-18", "CLE") == (
        "https://cdn/1610612739/2017/1040x760/2.png")


def test_a_traded_player_gets_nothing_when_the_team_is_unknown():
    """
    The team-blind fallback exists for tricode mismatches, but for a player
    with two shirts in one season it would be a coin flip. It must decline
    rather than guess, and the caller then shows the current photo.
    """
    lookup = build_serving.build_headshot_lookup(_headshot_frame())
    assert build_serving.season_headshot(lookup, 2, "2017-18", "PHO") is None
    # A player who never moved is unambiguous, so the fallback does answer.
    assert build_serving.season_headshot(lookup, 1, "2017-18", "PHO") == (
        "https://cdn/1610612738/2017/1040x760/1.png")


def test_rows_that_did_not_return_an_image_are_never_used():
    """
    329 of 4,009 player-seasons came back 403. Those must fall through to the
    current photo, not be handed to the browser as a broken image.
    """
    lookup = build_serving.build_headshot_lookup(_headshot_frame())
    assert build_serving.season_headshot(lookup, 3, "2017-18", "BOS") is None


def test_no_map_at_all_is_not_an_error():
    """
    A checkout that has never run script 39 must still build, with everyone on
    the current photo.
    """
    empty = build_serving.build_headshot_lookup(pd.DataFrame())
    assert build_serving.season_headshot(empty, 1, "2017-18", "BOS") is None
    assert build_serving.season_headshot(None, 1, "2017-18", "BOS") is None


def test_player_ids_and_seasons_are_compared_as_the_same_type():
    """
    The map is read from CSV, so person_id arrives as int64 and season as str,
    while the roster supplies a Python int. A type mismatch here would silently
    return None for every player and look like "the NBA has no photos".
    """
    frame = _headshot_frame()
    frame["person_id"] = frame["person_id"].astype("int64")
    lookup = build_serving.build_headshot_lookup(frame)
    assert build_serving.season_headshot(lookup, np.int64(1), "2017-18", "BOS")
    assert build_serving.season_headshot(lookup, 1, "2017-18", "BOS")


# ---------------------------------------------------------------------------
# Full team names in the payload
#
# The dashboard's completed-game state reads "Boston Celtics defeat Denver
# Nuggets", and the payload previously carried only tricodes. The map is read
# from the bios file rather than typed into the source, because a hand-written
# list of thirty team names is a list nothing checks.
# ---------------------------------------------------------------------------

def test_team_names_are_read_from_the_bios_file():
    bios = pd.DataFrame([
        {"team_abbrev": "BOS", "team_name": "Boston Celtics"},
        {"team_abbrev": "DEN", "team_name": "Denver Nuggets"},
        {"team_abbrev": "BOS", "team_name": "Boston Celtics"},
    ])
    names = build_serving.team_names(bios)
    assert names == {"BOS": "Boston Celtics", "DEN": "Denver Nuggets"}


def test_a_missing_bios_file_does_not_break_the_build():
    assert build_serving.team_names(pd.DataFrame()) == {}
    assert build_serving.team_names(None) == {}
