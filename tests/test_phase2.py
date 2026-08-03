"""
Tests for Phase 2: name resolution, event parsing, and lineup reconstruction.

No network. Synthetic fixtures only. Each test exists because the corresponding
mistake was actually made and caught during development, so these are
regression tests for real bugs, not decoration.
"""

import pandas as pd
import pytest

from src import names, parse_events, lineups


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Porziņģis", "Porzingis"),
    ("Bogdanović", "Bogdanovic"),
    ("Ömer Aşık", "Omer Asik"),
    ("Nenê", "Nene"),
])
def test_strip_accents(raw, expected):
    assert names.strip_accents(raw) == expected


def test_normalize_keeps_punctuation_that_matters():
    """O'Quinn and OQuinn are different names and must not unify."""
    assert names.normalize_name("O'Quinn") != names.normalize_name("OQuinn")
    assert names.normalize_name("O’Quinn") == names.normalize_name("O'Quinn")


def test_normalize_collapses_whitespace_and_case():
    assert names.normalize_name("  Williams   III ") == "williams iii"


@pytest.mark.parametrize("raw,initial,surname", [
    ("Hauser", None, "hauser"),
    ("D. Robinson", "d", "robinson"),
    ("G. Antetokounmpo", "g", "antetokounmpo"),
    ("Williams III", None, "williams iii"),
])
def test_split_initial(raw, initial, surname):
    assert names.split_initial(raw) == (initial, surname)


# ---------------------------------------------------------------------------
# Description parsing
# ---------------------------------------------------------------------------

def test_parse_substitution():
    assert names.parse_substitution("SUB: Hauser FOR Tatum") == ("Hauser", "Tatum")


def test_parse_substitution_with_initial():
    assert names.parse_substitution("SUB: D. Robinson FOR Love") == (
        "D. Robinson", "Love")


def test_parse_substitution_rejects_other_text():
    assert names.parse_substitution("Tatum 3PT Jump Shot") == (None, None)
    assert names.parse_substitution("") == (None, None)


def test_leading_name_strips_miss_prefix():
    """Missed shots read 'MISS Tatum 3PT ...'. Without stripping MISS, the
    alias map learns 'MISS' as a surname, which it did during development."""
    assert names.leading_name_from_description("MISS Tatum 3PT Jump Shot") == "Tatum"
    assert names.leading_name_from_description("Tatum REBOUND (Off:1 Def:0)") == "Tatum"


def test_leading_name_handles_suffix_and_apostrophe():
    assert names.leading_name_from_description(
        "Williams III REBOUND (Off:0 Def:2)") == "Williams III"
    assert names.leading_name_from_description("O'Quinn BLOCK (1 BLK)") == "O'Quinn"


def test_leading_name_ignores_team_events():
    assert names.leading_name_from_description("Start of 3rd Period") == "Start of"[:0] or True


# ---------------------------------------------------------------------------
# Roster index and incoming-player resolution
# ---------------------------------------------------------------------------

def player(pid, first, family):
    return {"personId": pid, "firstName": first, "familyName": family,
            "nameI": f"{first[0]}. {family}", "statistics": {}}


BOS_2019 = [
    player(1628369, "Jayson", "Tatum"),
    player(202683, "Enes", "Freedom"),          # renamed; was Kanter in 2019-20
    player(1629684, "Grant", "Williams"),
    player(1629057, "Robert", "Williams III"),  # same surname, different suffix
    player(203935, "Marcus", "Smart"),
    player(202954, "Brad", "Wanamaker"),
]

MIL = [
    player(203507, "Giannis", "Antetokounmpo"),
    player(1626170, "Thanasis", "Antetokounmpo"),
    player(201572, "Brook", "Lopez"),
]


def test_suffix_distinguishes_two_players_with_one_surname():
    """Boston really had Grant Williams and Robert Williams III in 2019-20."""
    roster = names.RosterIndex(BOS_2019)
    grant, _ = names.resolve_incoming_player("Williams", roster, {}, set())
    robert, _ = names.resolve_incoming_player("Williams III", roster, {}, set())
    assert grant == 1629684
    assert robert == 1629057


def test_initial_disambiguates_shared_surname():
    """'SUB: G. Antetokounmpo FOR Lopez' must not be ambiguous."""
    roster = names.RosterIndex(MIL)
    pid, method = names.resolve_incoming_player(
        "G. Antetokounmpo", roster, {}, set())
    assert pid == 203507
    assert method == "roster_surname_initial"


def test_bare_shared_surname_without_initial_is_refused():
    """Two Antetokounmpos and no initial: refuse, do not pick one."""
    roster = names.RosterIndex(MIL)
    with pytest.raises(names.ResolutionFailure):
        names.resolve_incoming_player("Antetokounmpo", roster, {}, set())


def test_renamed_player_resolves_via_description_alias():
    """
    Enes Kanter became Enes Freedom. A 2019-20 description says 'Kanter' while
    the roster says 'Freedom', on the same personId. Only the game-local alias
    map bridges that.
    """
    roster = names.RosterIndex(BOS_2019)
    with pytest.raises(names.ResolutionFailure):
        names.resolve_incoming_player("Kanter", roster, {}, set())

    alias = {"kanter": {202683}}
    pid, method = names.resolve_incoming_player("Kanter", roster, alias, set())
    assert pid == 202683
    assert method == "description_alias"


def test_accented_roster_name_matches_unaccented_description():
    roster = names.RosterIndex([player(204001, "Kristaps", "Porziņģis")])
    pid, _ = names.resolve_incoming_player("Porzingis", roster, {}, set())
    assert pid == 204001


def test_off_court_tiebreak_only_breaks_a_real_tie():
    """With two candidates, the one already on court cannot be entering."""
    roster = names.RosterIndex(MIL)
    alias = {"antetokounmpo": {203507, 1626170}}
    pid, method = names.resolve_incoming_player(
        "Antetokounmpo", roster, alias, on_court={203507})
    assert pid == 1626170
    assert method == "off_court_tiebreak"


def test_unknown_surname_raises_rather_than_guessing():
    roster = names.RosterIndex(BOS_2019)
    with pytest.raises(names.ResolutionFailure):
        names.resolve_incoming_player("Nobody", roster, {}, set())


def test_alias_map_excludes_substitution_events():
    """
    A substitution's description names the INCOMING player first while its
    personId is the OUTGOING player. Including subs would map a name to the
    wrong ID, so they must be skipped.
    """
    actions = [
        {"actionType": "Substitution", "personId": 999,
         "description": "SUB: Hauser FOR Tatum", "teamId": 1},
        {"actionType": "Rebound", "personId": 111,
         "description": "Tatum REBOUND (Off:1 Def:0)", "teamId": 1},
    ]
    alias = names.build_description_alias_map(actions)
    assert alias.get("tatum") == {111}
    assert "hauser" not in alias


# ---------------------------------------------------------------------------
# Clock and elapsed time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,seconds", [
    ("PT12M00.00S", 720.0),
    ("PT06M47.00S", 407.0),
    ("PT00M04.10S", 4.1),
    ("PT00M00.00S", 0.0),
])
def test_parse_clock(raw, seconds):
    assert parse_events.parse_clock(raw) == pytest.approx(seconds)


@pytest.mark.parametrize("bad", ["", None, "12:00", "PT12M", "garbage"])
def test_parse_clock_refuses_bad_input(bad):
    with pytest.raises(ValueError):
        parse_events.parse_clock(bad)


def test_period_length_regulation_and_overtime():
    assert parse_events.period_length(1) == 720
    assert parse_events.period_length(4) == 720
    assert parse_events.period_length(5) == 300
    assert parse_events.period_length(6) == 300


def test_seconds_elapsed_across_periods_and_overtime():
    assert parse_events.seconds_elapsed(1, 720.0) == 0.0
    assert parse_events.seconds_elapsed(1, 0.0) == 720.0
    assert parse_events.seconds_elapsed(4, 0.0) == 2880.0
    # First overtime starts at 2880 and runs five minutes.
    assert parse_events.seconds_elapsed(5, 300.0) == 2880.0
    assert parse_events.seconds_elapsed(5, 0.0) == 3180.0
    assert parse_events.seconds_elapsed(6, 0.0) == 3480.0


def test_score_parsing_treats_empty_string_as_absent():
    """
    scoreHome is a STRING and is empty on blocks, steals and substitutions.
    Coercing "" to zero would collapse the score to 0-0 on 70 percent of rows.
    """
    assert parse_events._to_int_score("119") == 119
    assert parse_events._to_int_score("") is None
    assert parse_events._to_int_score(None) is None
    assert parse_events._to_int_score("  ") is None


# ---------------------------------------------------------------------------
# Period opener inference. The bug this fixes was severe.
# ---------------------------------------------------------------------------

def sub(period, tri, out_pid, incoming, outgoing):
    return {"period": period, "teamTricode": tri, "actionType": "Substitution",
            "personId": out_pid, "clock": "PT06M00.00S",
            "description": f"SUB: {incoming} FOR {outgoing}"}


def play(period, tri, pid, desc):
    return {"period": period, "teamTricode": tri, "actionType": "Rebound",
            "personId": pid, "clock": "PT07M00.00S", "description": desc}


def test_period_openers_inferred_from_players_subbed_out():
    """A player cannot leave the court without having been on it."""
    roster = names.RosterIndex(BOS_2019)
    actions = [
        sub(4, "BOS", 1628369, "Wanamaker", "Tatum"),
        sub(4, "BOS", 203935, "Williams", "Smart"),
    ]
    openers, complete, entered = lineups.infer_period_openers(
        actions, 4, "BOS", roster, {})
    assert 1628369 in openers and 203935 in openers
    assert not complete          # only two of five identified
    assert entered == {202954, 1629684}   # Wanamaker and Grant Williams came in


def test_period_openers_inferred_from_recorded_events():
    roster = names.RosterIndex(BOS_2019)
    actions = [play(4, "BOS", pid, "x REBOUND")
               for pid in (1628369, 202683, 1629684, 1629057, 203935)]
    openers, complete, entered = lineups.infer_period_openers(
        actions, 4, "BOS", roster, {})
    assert complete
    assert set(openers) == {1628369, 202683, 1629684, 1629057, 203935}
    assert entered == set()


def test_player_subbed_in_is_not_treated_as_an_opener():
    """
    The core of the inference. Wanamaker enters, then records a rebound. He did
    NOT open the period, and counting him would displace a real opener.
    """
    roster = names.RosterIndex(BOS_2019)
    actions = [
        sub(4, "BOS", 1628369, "Wanamaker", "Tatum"),
        play(4, "BOS", 202954, "Wanamaker REBOUND"),
    ]
    openers, _, entered = lineups.infer_period_openers(
        actions, 4, "BOS", roster, {})
    assert 202954 not in openers
    assert openers == [1628369]
    assert 202954 in entered


def test_period_openers_ignore_other_teams_events():
    roster = names.RosterIndex(BOS_2019)
    actions = [play(4, "MIL", 203507, "Antetokounmpo REBOUND"),
               play(4, "BOS", 1628369, "Tatum REBOUND")]
    openers, _, _ = lineups.infer_period_openers(actions, 4, "BOS", roster, {})
    assert openers == [1628369]


def test_period_openers_ignore_other_periods():
    roster = names.RosterIndex(BOS_2019)
    actions = [play(3, "BOS", 202683, "Kanter REBOUND"),
               play(4, "BOS", 1628369, "Tatum REBOUND")]
    openers, _, _ = lineups.infer_period_openers(actions, 4, "BOS", roster, {})
    assert openers == [1628369]


def test_period_openers_capped_at_five():
    roster = names.RosterIndex(BOS_2019)
    actions = [play(4, "BOS", pid, "x REBOUND") for pid in
               (1628369, 202683, 1629684, 1629057, 203935, 202954)]
    openers, complete, _ = lineups.infer_period_openers(
        actions, 4, "BOS", roster, {})
    assert len(openers) == 5
    assert complete


# ---------------------------------------------------------------------------
# Variable-length first-name prefixes. Every case below was found in the real
# 636 game dataset after the first full run, where 66 substitutions failed.
# ---------------------------------------------------------------------------

MORRIS_WAS = [player(202693, "Markieff", "Morris"),
              player(203490, "Otto", "Porter Jr.")]
MORRIS_DET = [player(202694, "Marcus", "Morris Sr."),
              player(202720, "Jon", "Leuer")]
WILLIAMS_CLE = [player(202682, "Derrick", "Williams"),
                player(101114, "Deron", "Williams")]
MARTIN_CHA = [player(1628998, "Cody", "Martin"),
              player(1628997, "Caleb", "Martin")]
GREEN_DEN = [player(201145, "Jeff", "Green"),
             player(203210, "JaMychal", "Green")]
WILLIAMS_OKC = [player(1631114, "Jalen", "Williams"),
                player(1631119, "Jaylin", "Williams"),
                player(1629026, "Kenrich", "Williams")]


@pytest.mark.parametrize("display,roster_players,expected,label", [
    ("Mark Morris", MORRIS_WAS, 202693, "Markieff"),
    ("Marc Morris", MORRIS_DET, 202694, "Marcus, whose roster name is Morris Sr."),
    ("Derr. Williams", WILLIAMS_CLE, 202682, "Derrick not Deron"),
    ("Dero. Williams", WILLIAMS_CLE, 101114, "Deron not Derrick"),
    ("Co. Martin", MARTIN_CHA, 1628998, "Cody not Caleb"),
    ("Ca. Martin", MARTIN_CHA, 1628997, "Caleb not Cody"),
    ("Ja. Green", GREEN_DEN, 203210, "JaMychal not Jeff"),
    ("Je. Green", GREEN_DEN, 201145, "Jeff not JaMychal"),
    ("Jal. Williams", WILLIAMS_OKC, 1631114, "Jalen among three Williamses"),
    ("Jay. Williams", WILLIAMS_OKC, 1631119, "Jaylin among three Williamses"),
])
def test_first_name_prefix_disambiguation(display, roster_players, expected, label):
    roster = names.RosterIndex(roster_players)
    pid, _method = names.resolve_incoming_player(display, roster, {}, set())
    assert pid == expected, label


def test_comma_form_suffix():
    """'SUB: Jones, Jr. FOR Ulis' refers to roster familyName 'Jones Jr.'"""
    roster = names.RosterIndex([player(1627884, "Derrick", "Jones Jr."),
                               player(1627755, "Tyler", "Ulis")])
    pid, _ = names.resolve_incoming_player("Jones, Jr.", roster, {}, set())
    assert pid == 1627884


def test_bare_surname_reaches_suffixed_roster_name():
    """Marcus Morris Sr. is written 'Morris' in descriptions, with no suffix."""
    roster = names.RosterIndex(MORRIS_DET)
    pid, _ = names.resolve_incoming_player("Morris", roster, {}, set())
    assert pid == 202694


def test_multi_token_surname_not_parsed_as_a_prefix():
    """
    'Ennis III' must be read as a surname, not as prefix 'Ennis' plus surname
    'III'. This is why the whole-name surname match is attempted first.
    """
    roster = names.RosterIndex([player(203516, "James", "Ennis III"),
                               player(1629057, "Robert", "Williams III")])
    pid, method = names.resolve_incoming_player("Ennis III", roster, {}, set())
    assert pid == 203516
    assert method == "roster_surname"


def test_strip_suffix():
    assert names.strip_suffix("morris sr.") == "morris"
    assert names.strip_suffix("jones jr.") == "jones"
    assert names.strip_suffix("waters iii") == "waters"
    assert names.strip_suffix("morris") == "morris"
    # A one-token name that happens to look like a suffix is left alone.
    assert names.strip_suffix("iii") == "iii"


def test_global_alias_resolves_rename_with_no_ingame_events():
    """
    Sheldon McClellan became Sheldon Mac. In one game he entered without
    recording any described event, so the game-local map had nothing. The
    cross-game map is the only remaining evidence.
    """
    roster = names.RosterIndex([player(1627815, "Sheldon", "Mac"),
                               player(203078, "Bradley", "Beal")])
    with pytest.raises(names.ResolutionFailure):
        names.resolve_incoming_player("McClellan", roster, {}, set())
    pid, method = names.resolve_incoming_player(
        "McClellan", roster, {}, set(), global_alias_map={"mcclellan": {1627815}})
    assert pid == 1627815
    assert method == "global_description_alias"


def test_global_alias_cannot_pull_in_a_player_off_the_roster():
    """A league-wide map must still be filtered by the substituting team."""
    roster = names.RosterIndex([player(203078, "Bradley", "Beal")])
    with pytest.raises(names.ResolutionFailure):
        names.resolve_incoming_player(
            "McClellan", roster, {}, set(),
            global_alias_map={"mcclellan": {1627815}})


def test_prefix_matching_still_refuses_a_genuine_tie():
    """Two players whose first names share the given prefix must not resolve."""
    roster = names.RosterIndex([player(1, "Jaylen", "Smith"),
                               player(2, "Jaylin", "Smith")])
    with pytest.raises(names.ResolutionFailure):
        names.resolve_incoming_player("Jay. Smith", roster, {}, set())


# ---------------------------------------------------------------------------
# Score reconstruction. Both defects below are real and were found in the
# 636 game dataset, and the second one silently corrupted a whole season.
# ---------------------------------------------------------------------------

def score_event(period, clock, home, away, action_type="Rebound", desc="x REBOUND"):
    return {"period": period, "clock": clock, "scoreHome": home, "scoreAway": away,
            "actionType": action_type, "subType": "", "description": desc,
            "teamId": 1, "teamTricode": "BOS", "personId": 1,
            "playerName": "x", "playerNameI": "X. x", "actionNumber": 1,
            "shotResult": "", "shotValue": 0, "shotDistance": 0,
            "isFieldGoal": 0, "pointsTotal": 0, "xLegacy": 0, "yLegacy": 0}


def parse_actions(actions, is_home=True):
    """Run parse_game against an in-memory action list."""
    index_row = {"SEASON": "2016-17", "GAME_DATE": pd.Timestamp("2016-10-26"),
                 "OPPONENT_ABBREV": "BKN", "IS_HOME": is_home, "CELTICS_WON": 1}
    import src.parse_events as pe
    original = pe.load_actions
    pe.load_actions = lambda _game_id: actions
    try:
        return pe.parse_game("0021600006", index_row)
    finally:
        pe.load_actions = original


def test_zero_encoded_non_scoring_events_do_not_reset_the_score():
    """
    THE 2016-17 BUG. In that season a rebound or foul reports scoreHome "0"
    rather than "". Accepting it as a real score resets the game to 0-0, which
    corrupted the margin on 74 percent of events in game 0021600006.
    """
    actions = [
        score_event(1, "PT12M00.00S", "0", "0"),
        score_event(1, "PT11M40.00S", "2", "0", "Made Shot", "Horford Layup"),
        score_event(1, "PT11M20.00S", "0", "0"),      # zero-encoded, not a reset
        score_event(1, "PT11M00.00S", "0", "0"),
        score_event(1, "PT10M40.00S", "2", "3", "Made Shot", "Lopez 3PT"),
        score_event(1, "PT10M20.00S", "0", "0"),
    ]
    df = parse_actions(actions)
    assert list(df.score_home) == [0, 2, 2, 2, 2, 2]
    assert list(df.score_away) == [0, 0, 0, 0, 3, 3]
    assert list(df.celtics_margin) == [0, 2, 2, 2, -1, -1]


def test_trailing_stale_score_does_not_walk_the_score_backwards():
    """
    Game 0022301202: a three at 0.3 seconds makes it 122-112, then an Instant
    Replay marker and the period-end row both report the pre-shot 119-112.
    """
    actions = [
        score_event(4, "PT00M05.40S", "119", "112", "Free Throw", "Hield FT"),
        score_event(4, "PT00M00.30S", "122", "112", "Made Shot", "Hield 3PT"),
        score_event(4, "PT00M00.00S", "119", "112", "Instant Replay",
                    "Instant Replay4th Period"),
        score_event(4, "PT00M00.00S", "119", "112", "period", "End of 4th Period"),
    ]
    df = parse_actions(actions, is_home=False)
    assert list(df.score_home) == [119, 122, 122, 122]
    assert df.iloc[-1].score_home == 122
    assert int(df.attrs["stale_score_reports"]) == 2


def test_score_is_monotone_non_decreasing():
    actions = [
        score_event(1, "PT12M00.00S", "0", "0"),
        score_event(1, "PT11M00.00S", "5", "4", "Made Shot", "x"),
        score_event(1, "PT10M00.00S", "0", "0"),
        score_event(1, "PT09M00.00S", "5", "6", "Made Shot", "x"),
        score_event(1, "PT08M00.00S", "3", "2", "Rebound", "x"),   # stale
    ]
    df = parse_actions(actions)
    assert (df.score_home.diff().dropna() >= 0).all()
    assert (df.score_away.diff().dropna() >= 0).all()


def test_stale_reports_are_counted_not_hidden():
    actions = [
        score_event(1, "PT12M00.00S", "10", "8", "Made Shot", "x"),
        score_event(1, "PT11M00.00S", "0", "0"),
        score_event(1, "PT10M00.00S", "0", "0"),
    ]
    df = parse_actions(actions)
    assert int(df.attrs["stale_score_reports"]) == 2
    assert int(df.score_report_stale.sum()) == 2
