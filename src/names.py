"""
Player name normalization and substitution resolution.

Why this module is necessary
----------------------------
NBA play-by-play identifies the player LEAVING a substitution by `personId`,
which is authoritative. It identifies the player ENTERING only as a name inside
a free-text description:

    SUB: Hauser FOR Tatum        personId = Tatum's ID

Resolving that incoming name to a `personId` is unavoidable if on-court lineups
are to be reconstructed, and four separate hazards were found in the real data
for these eight seasons. All four were observed, not anticipated.

1. Diacritics differ between fields. Player 204001 appears as `K. Porziņģis` in
   `playerNameI` and as `Porzingis` in description text. Also `Bogdanović` and
   `Bogdanovic`. 28 occurrences in an 18 game sample.

2. Shared surnames are disambiguated with an initial, inconsistently. Usually the
   description carries a bare surname, but it becomes `G. Antetokounmpo` when the
   team has two Antetokounmpos. Both forms must be handled.

3. Suffixes are part of the surname. The roster's `familyName` is literally
   `Williams III` and `Ennis III`. This is load-bearing: in 2019-20 Boston had
   both Grant Williams (`Williams`) and Robert Williams III (`Williams III`), so
   the suffix is what tells them apart.

4. Name fields are CURRENT, description text is HISTORICAL. Enes Kanter legally
   changed his name to Enes Freedom. For a 2019-20 game the roster and event
   fields say `Freedom` while the description says `Kanter`, on the same
   `personId` 202683. Matching description text against roster names therefore
   fails outright for any player who later changed their name.

Strategy
--------
Two independent name sources, used together because each covers the other's
blind spot:

  A. The team's own boxscore roster (`familyName`, `firstName`). Handles suffixes
     and, crucially, players who enter a game without recording any statistic.

  B. A game-local map from description surname to `personId`, built from the
     descriptions of non-substitution events, which carry both. Immune to renames
     and diacritics because both sides come from the same text field in the same
     game.

Then one structural constraint, applied only to break a genuine tie:

  C. A substitution brings in a player who is currently OFF the court. If naming
     leaves several candidates but only one is off court, that is the answer.

If none of these produces a single answer, the resolution FAILS and is recorded.
Nothing is guessed. A wrong guess would silently corrupt a lineup for the rest of
a game, and lineup strength is a model feature.
"""

import re
import unicodedata
from collections import defaultdict

# Description text for a substitution. Verified against all 880 substitution
# events in the development sample: every one matches this pattern.
SUB_PATTERN = re.compile(r"^SUB:\s+(?P<incoming>.+?)\s+FOR\s+(?P<outgoing>.+)$")

# A missed shot description begins with "MISS " before the player's name, so the
# prefix has to be stripped before the leading token can be read as a surname.
MISS_PREFIX = re.compile(r"^MISS\s+")

# Name suffixes that form part of the surname in NBA data.
#
# ORDER MATTERS. These are joined into a regex alternation, and alternation is
# first-match-wins, so a shorter suffix listed earlier would win against a longer
# one. With "II" before "III", the name "Williams III" was being truncated to
# "Williams II" and then failed to match any roster entry. Longest first.
SUFFIXES = ("Jr.", "Sr.", "III", "II", "IV", "VI", "V")

# Latin letters that Unicode decomposition does NOT split into base plus
# combining mark, so stripping combining characters leaves them untouched.
# Turkish dotless i is the one that occurs in this dataset: Ömer Aşık became
# "Omer Asık" rather than "Omer Asik", which then failed to match.
NON_DECOMPOSING = str.maketrans({
    "ı": "i", "İ": "I", "ł": "l", "Ł": "L", "ø": "o", "Ø": "O",
    "đ": "d", "Đ": "D", "ð": "d", "Ð": "D", "ħ": "h", "Ħ": "H",
    "ŋ": "n", "Ŋ": "N", "ß": "ss", "æ": "ae", "Æ": "AE",
    "œ": "oe", "Œ": "OE", "þ": "th", "Þ": "Th",
})

# "K. O'Quinn" or "G. Antetokounmpo": a single initial, a dot, then the surname.
INITIAL_FORM = re.compile(r"^(?P<initial>[A-Za-z])\.\s+(?P<surname>.+)$")

# A VARIABLE-LENGTH first-name prefix followed by a surname. The NBA lengthens
# the prefix until the teammate ambiguity is resolved, and the trailing dot is
# not always present. All of these are real, from the 636 game dataset:
#
#   Marc Morris    -> Marcus Morris Sr.     Mark Morris   -> Markieff Morris
#   Derr. Williams -> Derrick Williams      Dero. Williams-> Deron Williams
#   Co. Martin     -> Cody Martin           Ca. Martin    -> Caleb Martin
#   Ja. Green      -> JaMychal Green        Je. Green     -> Jeff Green
#   Jal. Williams  -> Jalen Williams        Jay. Williams -> Jaylin Williams
#
# The prefix is capped at six letters so a genuine surname is not mistaken for
# one. This form is only tried AFTER a whole-name surname match fails, so
# "Ennis III" is matched as a surname rather than parsed as prefix "Ennis".
FIRST_PREFIX_FORM = re.compile(
    r"^(?P<prefix>[A-Za-z]{1,6})\.?\s+(?P<surname>[A-Za-z][A-Za-z'’\-\. ]*)$")

# The leading name in a description: word characters, apostrophes and hyphens,
# optionally followed by a suffix token.
LEADING_NAME = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z'’\-\.]*(?:\s+(?:" + "|".join(
        re.escape(s) for s in SUFFIXES) + r"))?)"
)


def strip_accents(text: str) -> str:
    """
    Remove diacritics. Porziņģis -> Porzingis, Bogdanović -> Bogdanovic.

    Applies the non-decomposing translation first, because characters like the
    Turkish dotless i survive Unicode decomposition and would otherwise remain:
    Aşık -> Asık rather than Asik.
    """
    if not text:
        return ""
    text = text.translate(NON_DECOMPOSING)
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize_name(text: str) -> str:
    """
    Canonical comparison form for a name.

    Strips diacritics, normalizes the Unicode right single quote used in some
    names to a plain apostrophe, collapses whitespace, and casefolds. Punctuation
    is deliberately KEPT: O'Quinn and OQuinn should not silently unify, and the
    period in a suffix carries meaning.
    """
    if not text:
        return ""
    text = strip_accents(text).replace("’", "'")
    # A comma before a suffix is a formatting variant, not part of the name:
    # the description "Jones, Jr." refers to roster familyName "Jones Jr.".
    text = text.replace(",", " ")
    return re.sub(r"\s+", " ", text).strip().casefold()


def strip_suffix(surname: str) -> str:
    """
    Remove a trailing name suffix from an already-normalized surname.

    "morris sr." -> "morris",  "jones jr." -> "jones",  "waters iii" -> "waters"

    Needed because a description can carry the bare surname while the roster
    carries the suffixed form. Marcus Morris Sr. appears in descriptions as
    "Marc Morris", with no suffix at all.
    """
    normalized_suffixes = {normalize_name(s) for s in SUFFIXES}
    parts = (surname or "").split()
    while len(parts) > 1 and parts[-1] in normalized_suffixes:
        parts.pop()
    return " ".join(parts)


def split_initial(display_name: str):
    """
    Split "G. Antetokounmpo" into ("g", "antetokounmpo").

    Returns (initial_or_None, normalized_surname). A bare surname yields
    (None, surname).
    """
    raw = (display_name or "").strip()
    match = INITIAL_FORM.match(raw)
    if match:
        return (match.group("initial").casefold(),
                normalize_name(match.group("surname")))
    return None, normalize_name(raw)


def leading_name_from_description(description: str) -> str:
    """
    Extract the acting player's display surname from an event description.

    Descriptions begin with the player's surname, except missed shots which are
    prefixed with "MISS ". Returns "" when no name can be read, which is correct
    for team-level events such as timeouts and period markers.
    """
    if not description:
        return ""
    text = MISS_PREFIX.sub("", description.strip())
    match = LEADING_NAME.match(text)
    if not match:
        return ""
    name = match.group("name").strip()
    # Guard against reading an all-caps event keyword as a name. Real NBA
    # surnames are not written in block capitals in this feed.
    if name.isupper() and len(name) > 1:
        return ""
    return name


def parse_substitution(description: str):
    """
    Split a substitution description into (incoming_name, outgoing_name).

    Returns (None, None) if the text does not match the expected form, so the
    caller can record a parse failure rather than act on a bad split.
    """
    match = SUB_PATTERN.match((description or "").strip())
    if not match:
        return None, None
    return match.group("incoming").strip(), match.group("outgoing").strip()


def build_description_alias_map(actions, team_id=None) -> dict:
    """
    Map normalized description surname -> set of personIds, for one game.

    Built from non-substitution events, which carry both a `personId` and a
    description beginning with that player's name as written AT THE TIME. This is
    the source that survives renames: for a 2019-20 game it maps `kanter` to
    person 202683, whose current name field reads `Freedom`.

    Substitution events are excluded on purpose. Their description names the
    INCOMING player first while `personId` refers to the OUTGOING one, so
    including them would map a name to the wrong ID.
    """
    alias = defaultdict(set)
    for action in actions:
        if action.get("actionType") == "Substitution":
            continue
        person_id = action.get("personId") or 0
        if not person_id:
            continue
        if team_id is not None and action.get("teamId") != team_id:
            continue
        name = leading_name_from_description(action.get("description") or "")
        if name:
            alias[normalize_name(name)].add(int(person_id))
    return dict(alias)


class RosterIndex:
    """
    Name lookup for one team's players in one game.

    Built from that team's boxscore player rows, so the candidate pool is around
    thirteen players rather than the whole league. A small pool is what makes
    surname matching safe.
    """

    def __init__(self, players):
        self.players = list(players)
        self.by_id = {int(p["personId"]): p for p in self.players}
        self._by_surname = defaultdict(list)
        self._by_base_surname = defaultdict(list)
        for player in self.players:
            person_id = int(player["personId"])
            family = normalize_name(player.get("familyName") or "")
            if not family:
                continue
            self._by_surname[family].append(person_id)
            base = strip_suffix(family)
            if base and base != family:
                self._by_base_surname[base].append(person_id)

    @property
    def person_ids(self) -> set:
        return set(self.by_id)

    def first_initial(self, person_id: int) -> str:
        player = self.by_id.get(int(person_id))
        first = normalize_name((player or {}).get("firstName") or "")
        return first[:1]

    def first_name(self, person_id: int) -> str:
        player = self.by_id.get(int(person_id))
        return normalize_name((player or {}).get("firstName") or "")

    def match_surname(self, surname: str, initial: str = None) -> list:
        """
        Candidate personIds for a normalized surname, optionally filtered by
        first initial.

        Tries the exact surname first, so `Williams` does not match
        `Williams III`. Only if that yields nothing does it try the
        suffix-stripped form, which is what lets a description's bare "Morris"
        reach roster familyName "Morris Sr.".
        """
        candidates = list(self._by_surname.get(surname, []))
        if not candidates:
            candidates = list(self._by_base_surname.get(surname, []))
        if initial:
            candidates = [pid for pid in candidates
                          if self.first_initial(pid) == initial]
        return candidates

    def match_first_prefix(self, prefix: str, surname: str) -> list:
        """
        Candidates whose surname matches and whose FIRST name starts with
        `prefix`.

        This is how the NBA disambiguates teammates who share a surname, using a
        prefix long enough to separate them: "Marc Morris" is Marcus and "Mark
        Morris" is Markieff, "Co. Martin" is Cody and "Ca. Martin" is Caleb.
        """
        prefix = normalize_name(prefix).rstrip(".")
        if not prefix:
            return []
        pool = (list(self._by_surname.get(surname, []))
                or list(self._by_base_surname.get(surname, [])))
        if not pool:
            base = strip_suffix(surname)
            pool = (list(self._by_surname.get(base, []))
                    or list(self._by_base_surname.get(base, [])))
        return [pid for pid in pool if self.first_name(pid).startswith(prefix)]

    def display(self, person_id: int) -> str:
        player = self.by_id.get(int(person_id))
        if not player:
            return f"<unknown {person_id}>"
        return f"{player.get('firstName','')} {player.get('familyName','')}".strip()


class ResolutionFailure(Exception):
    """Raised when an incoming substitution player cannot be identified."""

    def __init__(self, name, reason, candidates=None):
        self.name = name
        self.reason = reason
        self.candidates = candidates or []
        super().__init__(f"could not resolve {name!r}: {reason}")


def resolve_incoming_player(display_name, roster: RosterIndex,
                            alias_map: dict, on_court: set,
                            global_alias_map: dict = None):
    """
    Identify the personId of a player entering the game.

    Parameters
    ----------
    display_name : str
        The incoming name exactly as written in the description, for example
        "Hauser", "D. Robinson", or "Williams III".
    roster : RosterIndex
        The substituting team's players for this game.
    alias_map : dict
        Output of build_description_alias_map, scoped to this team.
    on_court : set
        personIds currently on the floor for this team. Used only to break a tie,
        never to invent an answer.

    Returns
    -------
    (person_id, method) where method records which source resolved it, so the
    validation report can show how often each was needed.

    Raises
    ------
    ResolutionFailure
        When no single candidate can be justified. Deliberately not a guess.
    """
    whole = normalize_name(display_name)
    if not whole:
        raise ResolutionFailure(display_name, "empty name")

    # A. Treat the entire name as a surname. Tried FIRST so that a genuine
    #    multi-token surname like "Ennis III" or "Jones Jr." is matched as such
    #    rather than misparsed as a first-name prefix plus surname.
    candidates = roster.match_surname(whole)
    if len(candidates) == 1:
        return candidates[0], "roster_surname"

    # B. Single initial plus surname: "G. Antetokounmpo".
    initial, surname = split_initial(display_name)
    if initial:
        narrowed = roster.match_surname(surname, initial)
        if len(narrowed) == 1:
            return narrowed[0], "roster_surname_initial"
        if narrowed:
            candidates = narrowed

    # C. Variable-length first-name prefix plus surname: "Marc Morris",
    #    "Derr. Williams", "Co. Martin". This is how teammates sharing a surname
    #    are told apart, and the prefix is exactly as long as it needs to be.
    if len(candidates) != 1:
        match = FIRST_PREFIX_FORM.match(display_name.strip())
        if match:
            prefixed = roster.match_first_prefix(
                match.group("prefix"), normalize_name(match.group("surname")))
            if len(prefixed) == 1:
                return prefixed[0], "roster_first_prefix"
            if prefixed:
                candidates = prefixed

    # D. Game-local description alias. Handles renamed players, whose roster
    #    name no longer matches the historical description text.
    if not candidates:
        aliased = [pid for pid in alias_map.get(whole, set())
                   if pid in roster.person_ids]
        if len(aliased) == 1:
            return aliased[0], "description_alias"
        if len(aliased) > 1:
            candidates = aliased

    # E. Cross-game description alias. Same idea, but built from every game in
    #    the dataset. Needed when a renamed player enters a game without
    #    recording any described event, so the game-local map never saw them.
    #    Sheldon McClellan, later Sheldon Mac, is the observed case.
    if not candidates and global_alias_map:
        aliased = [pid for pid in global_alias_map.get(whole, set())
                   if pid in roster.person_ids]
        if len(aliased) == 1:
            return aliased[0], "global_description_alias"
        if len(aliased) > 1:
            candidates = aliased

    if not candidates:
        raise ResolutionFailure(
            display_name,
            f"no roster or description match for {whole!r}",
        )

    # C. Structural tie-break. A substitution brings on a player who is off the
    #    court. If exactly one tied candidate is off court, that is the answer.
    off_court = [pid for pid in candidates if pid not in on_court]
    if len(off_court) == 1:
        return off_court[0], "off_court_tiebreak"

    raise ResolutionFailure(
        display_name,
        f"{len(candidates)} candidates remain and {len(off_court)} are off "
        f"court, so the choice is not determined",
        candidates=[roster.display(pid) for pid in candidates],
    )
