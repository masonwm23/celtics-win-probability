"""
Reconstruct the five players on court for both teams at every event.

Method
------
Start each game with the five starters per team, identified by roster row order
(see src/rosters.py for why the `position` field cannot be used). Then walk the
play-by-play in order and apply each substitution as a swap: the outgoing player
is given directly by `personId`, and the incoming player is resolved by
src/names.py.

Because every substitution event is a complete swap, a coherent log keeps each
lineup at exactly five players for the whole game, across period boundaries and
overtime, with no special handling required.

Handling incoherent logs
------------------------
Real logs are not always coherent. A substitution can name an outgoing player who
is not currently on the court. This project does not paper over that:

  - The swap is applied as faithfully as possible: remove the outgoing player if
    present, add the incoming player.
  - The event is RECORDED in an anomaly log with the reason.
  - Lineup size is checked after every event. Any deviation from five is counted
    and reported per game.

Nothing is guessed to force a lineup back to five, because a fabricated lineup
would feed straight into the lineup-strength model feature.

The independent check
---------------------
Substitution tracking is verified by a measurement that does not depend on it:
summing the game-clock time each player spends on court should reproduce that
player's boxscore minutes. If the reconstruction is wrong, derived minutes drift
away from the boxscore. This is the real test, and it runs per player per game.

Output
------
data/interim/lineups.parquet         one row per event with both lineups
data/interim/lineup_anomalies.csv    every incoherent substitution, with reason
data/interim/derived_minutes.csv     derived vs boxscore minutes per player
"""

import json
import logging

import pandas as pd

from src import config
from src.names import (RosterIndex, build_description_alias_map,
                       parse_substitution, resolve_incoming_player,
                       ResolutionFailure)
from src.parse_events import load_actions, parse_clock, seconds_elapsed

logger = logging.getLogger(__name__)

LINEUP_SIZE = 5


class GameLineupResult:
    """Everything reconstruct_game produces for a single game."""

    def __init__(self, game_id):
        self.game_id = game_id
        self.event_lineups = []   # per event, both teams
        self.anomalies = []       # incoherent substitutions
        self.minutes = {}         # (team, person_id) -> derived seconds
        self.methods = {}         # resolution method -> count
        self.undetermined_openers = {}   # (period, team) -> (candidates, needed)
        self.explorable_openers = {}     # (period, team) -> (fixed, pool, needed)
        self.contradicted_periods = {}   # (period, team) -> players who must be on
        self.period_opener_sets = {}     # (period, team) -> opener list used

    @property
    def n_bad_size(self):
        return sum(1 for row in self.event_lineups
                   if row["home_lineup_size"] != LINEUP_SIZE
                   or row["away_lineup_size"] != LINEUP_SIZE)


def _team_context(game_id: str, actions: list):
    """
    Build per-team roster index, alias map, tricode/id mapping and starters.

    Starters come from boxscore row order. Tricode to teamId mapping is taken from
    the events themselves so the two sources are tied together rather than
    assumed to agree.
    """
    path = config.RAW_BOX_DIR / f"{game_id}.json"
    with path.open("r", encoding="utf-8") as fh:
        box = json.load(fh)["boxScoreTraditional"]

    context = {}
    for side in ("homeTeam", "awayTeam"):
        team = box[side]
        tricode = team["teamTricode"]
        players = team["players"]
        context[tricode] = {
            "side": side,
            "is_home": side == "homeTeam",
            "team_id": int(team.get("teamId") or 0),
            "roster": RosterIndex(players),
            "starters": [int(p["personId"]) for p in players[:LINEUP_SIZE]],
            "boxscore_minutes": {
                int(p["personId"]): (p.get("statistics") or {}).get("minutes")
                for p in players
            },
            # Players who actually appeared. A player cannot be on court during a
            # period without recording minutes for the game.
            "played_ids": {
                int(p["personId"]) for p in players
                if str((p.get("statistics") or {}).get("minutes") or "").strip()
                not in ("", "0:00", "00:00")
            },
        }

    # Tie tricodes to the teamIds actually present in the play-by-play.
    for action in actions:
        tri, tid = action.get("teamTricode"), action.get("teamId")
        if tri in context and tid:
            context[tri]["team_id"] = int(tid)

    for tricode, ctx in context.items():
        ctx["alias"] = build_description_alias_map(actions, team_id=ctx["team_id"])

    return context


def build_global_alias_map(game_ids) -> dict:
    """
    Map description surname -> set of personIds, across EVERY game.

    Needed for one specific case: a player whose name changed, who enters a game
    without recording any described event. The game-local map never sees them, so
    the only remaining evidence is how they were named in other games. Sheldon
    McClellan, later Sheldon Mac, is the observed instance.

    Candidates are still filtered against the substituting team's roster at
    resolution time, so a league-wide map cannot pull in an unrelated player.
    """
    combined = {}
    for game_id in game_ids:
        actions = load_actions(game_id)
        for name, ids in build_description_alias_map(actions).items():
            combined.setdefault(name, set()).update(ids)
    return combined


def infer_period_openers(actions, period, tricode, roster, alias):
    """
    Infer which five players opened a given period for one team.

    WHY THIS IS NECESSARY. Substitutions made between periods are frequently NOT
    logged as substitution events. Verified case, game 0021600006: period 3 ends
    with Isaiah Thomas on court, and period 4's own substitutions prove Avery
    Bradley opened period 4 in his place. No event records the swap. Carrying the
    previous period's closing lineup forward is therefore invalid, and doing so
    corrupts every lineup for the rest of the game.

    INFERENCE RULE, using only that period's events. A player was on court at the
    period's start if, before being substituted IN during the period, they either

      - are substituted OUT (you cannot leave without having been there), or
      - record any non-substitution event (shot, rebound, foul, turnover), which
        requires being on the floor.

    Returns (openers, complete, subbed_in). `openers` is a list of up to five
    personIds in the order the evidence appeared, `complete` says whether five
    were positively identified, and `subbed_in` is everyone who entered during
    the period, which the caller needs to fill an incomplete set correctly.

    Fewer than five happens when a player opens a period, never touches the ball,
    and is never subbed out. Observed in 11 of 636 games, almost all in overtime,
    where five minutes leaves little time to record anything.
    """
    openers, subbed_in = [], set()

    for action in actions:
        if int(action.get("period") or 0) != period:
            continue
        if (action.get("teamTricode") or "") != tricode:
            continue

        if action.get("actionType") == "Substitution":
            incoming_name, _ = parse_substitution(action.get("description"))
            outgoing_id = int(action.get("personId") or 0)
            if outgoing_id and outgoing_id not in subbed_in \
                    and outgoing_id not in openers:
                openers.append(outgoing_id)
            if incoming_name:
                try:
                    incoming_id, _method = resolve_incoming_player(
                        incoming_name, roster, alias, set())
                    subbed_in.add(incoming_id)
                except ResolutionFailure:
                    pass
        else:
            person_id = int(action.get("personId") or 0)
            if person_id and person_id not in subbed_in \
                    and person_id not in openers:
                openers.append(person_id)

    return (openers[:LINEUP_SIZE], len(openers) >= LINEUP_SIZE, subbed_in)


def reconstruct_game(game_id: str, global_alias_map: dict = None,
                     opener_overrides: dict = None) -> GameLineupResult:
    """Walk one game and produce per-event lineups for both teams."""
    # The payload's delivered order is authoritative and must not be re-sorted.
    # `actionNumber` is not unique within a game, so sorting on it silently
    # reorders events and breaks clock monotonicity. See parse_events.parse_game.
    actions = load_actions(game_id)
    context = _team_context(game_id, actions)
    result = GameLineupResult(game_id)

    ctx_by_tri = context
    home_tri = next(t for t, c in context.items() if c["is_home"])
    away_tri = next(t for t, c in context.items() if not c["is_home"])

    periods = sorted({int(a.get("period") or 0) for a in actions} - {0})

    # Anchor every period independently. Period 1's openers are the boxscore
    # starters, which is a known fact. Later periods are inferred from their own
    # events, because between-period substitutions are often unlogged.
    period_openers = {}
    for period in periods:
        for tri, ctx in context.items():
            if period == periods[0]:
                openers, complete, entered = list(ctx["starters"]), True, set()
            else:
                openers, complete, entered = infer_period_openers(
                    actions, period, tri, ctx["roster"], ctx["alias"])
            period_openers[(period, tri)] = (openers, complete, entered)

    on_court = {tri: set() for tri in context}
    seconds_on = {tri: {pid: 0.0 for pid in ctx["roster"].person_ids}
                  for tri, ctx in context.items()}

    previous_elapsed = 0.0
    current_period = None

    for event_index, action in enumerate(actions):
        period = int(action.get("period") or 0)
        clock = parse_clock(action.get("clock"))
        elapsed = seconds_elapsed(period, clock)

        if period and period != current_period:
            # Re-anchor at the period boundary.
            for tri in context:
                openers, complete, entered = period_openers.get(
                    (period, tri), ([], False, set()))
                openers = list(openers)
                determined, source, needed = True, "", 0
                full_override = (opener_overrides or {}).get((period, tri))
                if full_override is not None and len(full_override) == LINEUP_SIZE:
                    on_court[tri] = set(full_override)
                    result.period_opener_sets[(period, tri)] = list(full_override)
                    continue
                if not complete:
                    needed = LINEUP_SIZE - len(openers)
                    # First choice: the previous period's closing lineup, minus
                    # anyone already identified and anyone who entered LATER in
                    # this period, since they cannot also have opened it.
                    candidates = sorted(
                        pid for pid in on_court[tri]
                        if pid not in openers and pid not in entered)
                    source = "previous period"

                    # Widening. Observed case, game 0022100688 period 3: every
                    # player from the previous period's close is either already
                    # identified or entered later, leaving ZERO candidates and a
                    # four-player lineup. That happens when an unlogged
                    # between-period substitution brings in a player who then
                    # records nothing and is never subbed out. Widen to anyone
                    # who appeared in the game, since being on court requires it.
                    if len(candidates) != needed:
                        widened = sorted(
                            pid for pid in ctx_by_tri[tri]["played_ids"]
                            if pid not in openers and pid not in entered)
                        if len(widened) == needed or not candidates:
                            candidates = widened
                            source = "players who appeared in the game"

                    # A caller-supplied override lets the wrapper retry this
                    # choice and keep whichever assignment reconciles against
                    # boxscore minutes.
                    override = (opener_overrides or {}).get((period, tri))
                    if override is not None:
                        chosen = list(override)
                        source = "minutes reconciliation"
                    else:
                        chosen = candidates[:needed]

                    determined = len(candidates) == needed
                    # Record the WIDE pool for reconciliation regardless of
                    # whether the narrow choice looked determined.
                    #
                    # Game 0021900892 is why. Its overtime openers were "1
                    # candidate for 1 slot", so it looked settled, yet Eric
                    # Gordon came out 5.00 minutes short and Jeff Green 5.00
                    # minutes long: a whole overtime credited to the wrong
                    # player. Unambiguous is not the same as correct, and
                    # because it looked settled no alternative was ever tried.
                    wide_pool = sorted(
                        pid for pid in ctx_by_tri[tri]["played_ids"]
                        if pid not in openers and pid not in entered)
                    result.explorable_openers[(period, tri)] = (
                        list(openers), wide_pool, needed)
                    openers.extend(chosen)
                    result.anomalies.append({
                        "game_id": game_id,
                        "event_index": event_index,
                        "action_number": action.get("actionNumber"),
                        "period": period,
                        "reason": ("period openers incomplete, filled from "
                                   + source
                                   + ("" if determined else " (NOT DETERMINED)")),
                        "detail": f"team {tri}, "
                                  f"{LINEUP_SIZE - needed} of {LINEUP_SIZE} "
                                  f"positively identified; "
                                  f"{len(candidates)} candidate(s) for "
                                  f"{needed} slot(s)",
                        "description": action.get("description") or "",
                    })
                on_court[tri] = set(openers[:LINEUP_SIZE])
                result.period_opener_sets[(period, tri)] = \
                    list(openers[:LINEUP_SIZE])
            current_period = period

        # Accrue time for everyone on court over the interval just ended. The
        # lineup was constant across it, since it only changes on a substitution.
        delta = elapsed - previous_elapsed
        if delta > 0:
            for tri, players in on_court.items():
                for pid in players:
                    seconds_on[tri][pid] = seconds_on[tri].get(pid, 0.0) + delta
        previous_elapsed = elapsed

        if action.get("actionType") == "Substitution":
            tri = action.get("teamTricode") or ""
            ctx = context.get(tri)
            if ctx is None:
                result.anomalies.append({
                    "game_id": game_id, "action_number": action.get("actionNumber"),
                    "period": period, "reason": "substitution for unknown team",
                    "detail": f"tricode {tri!r}",
                    "description": action.get("description") or "",
                })
            else:
                incoming_name, _ = parse_substitution(action.get("description"))
                outgoing_id = int(action.get("personId") or 0)

                if incoming_name is None:
                    result.anomalies.append({
                        "game_id": game_id,
                        "action_number": action.get("actionNumber"),
                        "period": period, "reason": "unparseable description",
                        "detail": "", "description": action.get("description") or "",
                    })
                else:
                    try:
                        incoming_id, method = resolve_incoming_player(
                            incoming_name, ctx["roster"], ctx["alias"],
                            on_court[tri], global_alias_map)
                        result.methods[method] = result.methods.get(method, 0) + 1
                    except ResolutionFailure as failure:
                        incoming_id = None
                        result.anomalies.append({
                            "game_id": game_id,
                            "action_number": action.get("actionNumber"),
                            "period": period,
                            "reason": "unresolved incoming player",
                            "detail": failure.reason,
                            "description": action.get("description") or "",
                        })

                    if outgoing_id and outgoing_id not in on_court[tri]:
                        # The log says this player left the floor, so they were
                        # on it. Our opener set for this period must be wrong.
                        # Record it so reconciliation can try putting them in.
                        result.contradicted_periods.setdefault(
                            (period, tri), set()).add(outgoing_id)
                        result.anomalies.append({
                            "game_id": game_id,
                            "action_number": action.get("actionNumber"),
                            "period": period,
                            "reason": "outgoing player was not on court",
                            "detail": ctx["roster"].display(outgoing_id),
                            "description": action.get("description") or "",
                        })

                    on_court[tri].discard(outgoing_id)
                    if incoming_id is not None:
                        on_court[tri].add(incoming_id)

        result.event_lineups.append({
            "game_id": game_id,
            "event_index": event_index,
            "action_number": int(action.get("actionNumber") or 0),
            "period": period,
            "home_lineup": tuple(sorted(on_court[home_tri])),
            "away_lineup": tuple(sorted(on_court[away_tri])),
            "home_lineup_size": len(on_court[home_tri]),
            "away_lineup_size": len(on_court[away_tri]),
        })

    result.minutes = {
        (tri, pid): secs for tri, players in seconds_on.items()
        for pid, secs in players.items()
    }
    result.context = context
    return result


def total_minutes_error(result: GameLineupResult) -> float:
    """Sum of absolute derived-versus-boxscore minute differences for one game."""
    frame = minutes_comparison(result)
    return float(frame["difference"].abs().sum())


def reconstruct_game_reconciled(game_id: str, global_alias_map: dict = None,
                                max_combinations: int = 400):
    """
    Reconstruct a game, resolving uncertain period openers by reconciliation.

    Two kinds of uncertainty are explored:

      1. Periods whose opening five could not be fully identified from that
         period's own events. Every way of filling the open slots from the
         players who appeared in the game is tried. Note this runs even when the
         narrow candidate list looked "determined": in game 0021900892 the single
         obvious candidate was simply wrong, costing Eric Gordon a full overtime.

      2. Periods contradicted by a later substitution. If the log says a player
         left the floor, they were on it, so an opener set that excludes them is
         wrong. Each of the five is tried in that player's place.

    The assignment whose derived on-court minutes best match the boxscore is
    kept. This is a real inference, not a guess, because the boxscore fixes every
    player's total time and usually only one assignment reproduces it.

    Two honest caveats, both surfaced in the validation report. For these games
    the minutes check is no longer fully independent, since minutes were used to
    choose. And if two assignments tie on minutes error, the ambiguity is genuine
    and the first is kept.

    Returns (result, reconciled_keys).
    """
    from itertools import combinations, product

    base = reconstruct_game(game_id, global_alias_map)
    if not base.explorable_openers and not base.contradicted_periods:
        return base, []

    baseline_error = total_minutes_error(base)
    if baseline_error <= 1e-6:
        return base, []          # already exact, nothing to improve

    options_by_key = {}

    for key, (fixed, pool, needed) in base.explorable_openers.items():
        opts = []
        for combo in combinations(pool, needed):
            candidate = list(fixed) + list(combo)
            if len(candidate) == LINEUP_SIZE:
                opts.append(tuple(candidate))
        if opts:
            options_by_key[key] = opts

    for key, must_be_on in base.contradicted_periods.items():
        current = None
        for row in base.event_lineups:
            if row["period"] == key[0]:
                current = row
                break
        if current is None:
            continue
        existing = list(base.period_opener_sets.get(key, []))
        if len(existing) != LINEUP_SIZE:
            continue
        opts = [tuple(existing)]
        for player in sorted(must_be_on):
            if player in existing:
                continue
            for index in range(LINEUP_SIZE):
                swapped = list(existing)
                swapped[index] = player
                opts.append(tuple(swapped))
        options_by_key.setdefault(key, [])
        options_by_key[key] = opts if not options_by_key[key] else \
            options_by_key[key] + [o for o in opts if o not in options_by_key[key]]

    if not options_by_key:
        return base, []

    total = 1
    for opts in options_by_key.values():
        total *= max(1, len(opts))
    if total > max_combinations:
        logger.warning("%s has %d opener combinations, above the cap of %d; "
                       "keeping the deterministic choice", game_id, total,
                       max_combinations)
        return base, []

    keys = sorted(options_by_key)
    best, best_error, best_keys = base, baseline_error, []
    for choice in product(*[options_by_key[k] for k in keys]):
        overrides = {k: list(v) for k, v in zip(keys, choice)}
        candidate_result = reconstruct_game(game_id, global_alias_map, overrides)
        error = total_minutes_error(candidate_result)
        if error < best_error - 1e-9:
            best, best_error, best_keys = candidate_result, error, list(overrides)
    return best, best_keys


def minutes_comparison(result: GameLineupResult) -> pd.DataFrame:
    """
    Derived on-court minutes versus boxscore minutes, per player.

    This is the independent verification of substitution tracking. It uses the
    boxscore, which knows nothing about how the lineups were reconstructed.
    """
    from src.rosters import parse_minutes

    rows = []
    for (tricode, person_id), seconds in result.minutes.items():
        ctx = result.context[tricode]
        boxscore = parse_minutes(ctx["boxscore_minutes"].get(person_id))
        derived = seconds / 60.0
        rows.append({
            "game_id": result.game_id,
            "team_tricode": tricode,
            "person_id": person_id,
            "player": ctx["roster"].display(person_id),
            "derived_minutes": round(derived, 3),
            "boxscore_minutes": round(boxscore, 3),
            "difference": round(derived - boxscore, 3),
            "played": boxscore > 0,
        })
    return pd.DataFrame(rows)


def build_lineups(game_ids=None):
    """
    Reconstruct lineups for every cached game, or a subset.

    Returns (lineups_df, anomalies_df, minutes_df, method_counts).
    """
    if game_ids is None:
        game_ids = sorted(p.stem for p in config.RAW_PBP_DIR.glob("*.json"))

    logger.info("building cross-game description alias map over %d games",
                len(game_ids))
    global_alias_map = build_global_alias_map(game_ids)

    lineup_frames, anomaly_rows, minute_frames = [], [], []
    methods = {}

    reconciled_games = []
    for n, game_id in enumerate(game_ids, start=1):
        result, reconciled = reconstruct_game_reconciled(game_id, global_alias_map)
        if reconciled:
            reconciled_games.append(game_id)
        lineup_frames.append(pd.DataFrame(result.event_lineups))
        anomaly_rows.extend(result.anomalies)
        minute_frames.append(minutes_comparison(result))
        for key, count in result.methods.items():
            methods[key] = methods.get(key, 0) + count
        if n % 100 == 0:
            logger.info("reconstructed %d/%d games", n, len(game_ids))

    lineups = pd.concat(lineup_frames, ignore_index=True)
    anomalies = pd.DataFrame(anomaly_rows) if anomaly_rows else pd.DataFrame(
        columns=["game_id", "action_number", "period", "reason", "detail",
                 "description"])
    minutes = pd.concat(minute_frames, ignore_index=True)
    if reconciled_games:
        logger.info("opener choice resolved by minutes reconciliation in "
                    "%d game(s): %s", len(reconciled_games),
                    ", ".join(reconciled_games))
    methods = dict(methods)
    methods["_games_reconciled_by_minutes"] = len(reconciled_games)
    return lineups, anomalies, minutes, methods
