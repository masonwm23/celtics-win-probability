/**
 * Reading the play-by-play description properly.
 *
 * The feed writes a compressed, semi-structured string per event and the panel
 * used to collapse most of it to the bare action type: every rebound read
 * "Rebound", every foul read "Foul". The detail was there and being thrown
 * away.
 *
 * Every pattern below was derived from the descriptions in all 636 games in
 * this dataset, not from documentation and not from memory. Two of those
 * measurements decide the design:
 *
 *   1. NO foul description names the player who was fouled. Zero of 22,000+
 *      foul events contain "on" or "drawn". The trailing parenthetical, which
 *      looks like a second player, is the OFFICIAL: "(B.Spooner)", "(Z.Zarba)",
 *      "(S.Foster)". Treating it as the fouled player would have put a
 *      referee's name in the panel as a victim. So the fouled player is
 *      reported as not recorded, always, rather than guessed at from the
 *      surrounding events.
 *
 *   2. NO turnover description mentions a steal. Zero of 21,000+ turnover
 *      events contain "STEAL". Steals arrive as their own events with a BLANK
 *      action type and the word in the description. They are therefore
 *      labelled separately and never joined to a turnover, because pairing
 *      them would mean inferring from an adjacent row.
 *
 * A rebound's type is a third measured case. The feed gives the player's
 * running totals, "REBOUND (Off:0 Def:1)", not the type of THIS rebound. In
 * 25,649 of 66,600 rebounds both counters are non-zero, so a single row cannot
 * say which one just incremented. It is recovered by differencing the same
 * player's own previous counters, which is arithmetic on a recorded field
 * rather than inference about anything else.
 */

// ---------------------------------------------------------------------------
// Fouls
// ---------------------------------------------------------------------------

/**
 * Longest and most specific first. "Offensive Charge Foul" has to be matched
 * before anything that would also match the bare word "Foul".
 */
export const FOUL_TYPES = [
  [/FLAGRANT\.FOUL\.TYPE\s*2/i, "Flagrant 2 foul"],
  [/FLAGRANT\.FOUL\.TYPE\s*1/i, "Flagrant 1 foul"],
  [/AWAY\.FROM\.PLAY\.FOUL/i, "Away-from-play foul"],
  [/HANGING\.TECH\.FOUL/i, "Hanging technical"],
  [/Non-Unsportsmanlike/i, "Non-unsportsmanlike technical"],
  [/\bTaunting\b/i, "Taunting technical"],
  [/\bDelay\b/i, "Delay of game"],
  [/Double Technical/i, "Double technical"],
  [/Double Personal/i, "Double personal foul"],
  [/Offensive Charge Foul/i, "Offensive charge foul"],
  [/Shooting Block Foul/i, "Shooting block foul"],
  [/Personal Take Foul/i, "Take foul"],
  [/Personal Block Foul/i, "Personal block foul"],
  [/Transition Take Foul/i, "Transition take foul"],
  [/Loose Ball Foul/i, "Loose ball foul"],
  [/OFF\.Foul/i, "Offensive foul"],
  [/L\.B\.FOUL/i, "Loose ball foul"],
  [/IN\.FOUL/i, "Inbound foul"],
  [/S\.FOUL/i, "Shooting foul"],
  [/P\.FOUL/i, "Personal foul"],
  [/T\.FOUL/i, "Technical foul"],
  [/T\.Foul/i, "Technical foul"],
  [/Technical/i, "Technical foul"],
];

/** The recorded foul type, or null when the feed did not name one. */
export function foulType(description) {
  const text = String(description || "");
  for (const [pattern, label] of FOUL_TYPES) {
    if (pattern.test(text)) return label;
  }
  return null;
}

/**
 * Who the foul was called ON. Always null, and deliberately so.
 *
 * Kept as a function rather than omitted, because the honest answer to "who
 * got fouled" is a measurement about this feed and not an oversight. If a
 * future season starts recording it, this is where it goes, and the test that
 * pins the current behaviour will fail and say so.
 */
export function fouledPlayer() {
  return null;
}

export const FOULED_UNKNOWN = "Player fouled not recorded";

// ---------------------------------------------------------------------------
// Turnovers
// ---------------------------------------------------------------------------

/** Again longest first: "Out of Bounds - Bad Pass" outranks "Bad Pass". */
export const TURNOVER_CAUSES = [
  [/Out of Bounds - Bad Pass/i, "Bad pass out of bounds"],
  [/Out of Bounds Lost Ball/i, "Lost ball out of bounds"],
  [/Step Out of Bounds/i, "Stepped out of bounds"],
  [/Poss Lost Ball/i, "Lost ball"],
  [/Offensive Foul/i, "Offensive foul"],
  [/Discontinue Dribble/i, "Discontinued dribble"],
  [/Double Dribble/i, "Double dribble"],
  [/3 Second Violation/i, "Three-second violation"],
  [/5 Second Violation/i, "Five-second violation"],
  [/8 Second Violation/i, "Eight-second violation"],
  [/Jump Ball Violation/i, "Jump ball violation"],
  [/Illegal Screen/i, "Illegal screen"],
  [/Illegal Assist/i, "Illegal assist"],
  [/Basket from Below/i, "Basket from below"],
  [/Swinging Elbows/i, "Swinging elbows"],
  [/Too Many Players/i, "Too many players"],
  [/Lane Violation/i, "Lane violation"],
  [/Kicked Ball/i, "Kicked ball"],
  [/Punched Ball/i, "Punched ball"],
  [/Shot Clock/i, "Shot clock"],
  [/Goaltending/i, "Offensive goaltending"],
  [/Traveling/i, "Traveling"],
  [/Backcourt/i, "Backcourt violation"],
  [/Palming/i, "Palming"],
  [/Inbound/i, "Inbound violation"],
  [/Lost Ball/i, "Lost ball"],
  [/Bad Pass/i, "Bad pass"],
  [/\bFoul Turnover\b/i, "Foul"],
];

/** The recorded cause, or null when the feed did not name one. */
export function turnoverCause(description) {
  const text = String(description || "");
  for (const [pattern, label] of TURNOVER_CAUSES) {
    if (pattern.test(text)) return label;
  }
  return null;
}

/**
 * Is this a TEAM turnover rather than a player's?
 *
 * "RAPTORS Turnover: Shot Clock (T#5)" has no player. The feed writes the team
 * where the name would go.
 */
export function isTeamEvent(description) {
  return /^[A-Z0-9][A-Za-z0-9\s.'-]*\s(?:Turnover:|Rebound$)/.test(
    String(description || "").trim()
  );
}

// ---------------------------------------------------------------------------
// Rebounds
// ---------------------------------------------------------------------------

/** The player's running offensive and defensive totals, or null for a team rebound. */
export function reboundCounters(description) {
  const match = String(description || "").match(/\(Off:(\d+)\s+Def:(\d+)\)/);
  if (!match) return null;
  return { off: Number(match[1]), def: Number(match[2]) };
}

/**
 * Which counter just went up.
 *
 * `previous` is the SAME PLAYER's counters from their last rebound in this
 * game, or nothing if this is their first. Returns null only when the two
 * rows are indistinguishable, which should not happen on a well-formed feed
 * and is reported honestly rather than defaulted to one side.
 */
export function reboundKind(counters, previous = { off: 0, def: 0 }) {
  if (!counters) return "team";
  const before = previous || { off: 0, def: 0 };
  if (counters.off > before.off) return "offensive";
  if (counters.def > before.def) return "defensive";
  return null;
}

/**
 * Rebound types for a whole game, in one pass.
 *
 * Walks the events once, keeping each player's last seen counters, so every
 * rebound is resolved by differencing that player's own history. Returns a
 * Map from event index to "offensive", "defensive", "team" or null.
 */
export function reboundKindsForGame(events) {
  const kinds = new Map();
  const last = new Map();
  const n = events?.action_type?.length || 0;
  for (let i = 0; i < n; i += 1) {
    if (events.action_type[i] !== "Rebound") continue;
    const counters = reboundCounters(events.description[i]);
    if (!counters) {
      kinds.set(i, "team");
      continue;
    }
    const person = events.person_id[i];
    const kind = reboundKind(counters, last.get(person));
    kinds.set(i, kind);
    last.set(person, counters);
  }
  return kinds;
}

// ---------------------------------------------------------------------------
// Steals and blocks
// ---------------------------------------------------------------------------

/**
 * Steals and blocks arrive with a BLANK action type and the word in the text.
 * 23 of the 490 events on game 0022300906 are this shape.
 */
export function blankActionKind(description) {
  const text = String(description || "");
  if (/\bSTEAL\b/.test(text)) return "Steal";
  if (/\bBLOCK\b/.test(text)) return "Block";
  return null;
}

// ---------------------------------------------------------------------------
// The label the interface shows
// ---------------------------------------------------------------------------

/**
 * A detailed label for one event.
 *
 * Returns the pieces rather than a sentence, so the card can lay them out and
 * the ribbon can use a shorter form without the two drifting apart.
 *
 *   label     what happened, as specifically as the feed allows
 *   detail    a qualifier, such as the turnover's cause
 *   note      something the feed does NOT record, stated plainly
 */
export function describePlay(event, { reboundType = null } = {}) {
  const action = String(event?.action_type || "").trim();
  const description = String(event?.description || "");
  const value = Number(event?.shot_value) || 2;

  if (action === "Made Shot") return { label: `Made ${value}PT` };
  if (action === "Missed Shot") return { label: `Missed ${value}PT` };

  if (action === "Free Throw") {
    return {
      label: /^MISS\b/i.test(description.trim())
        ? "Missed free throw"
        : "Made free throw",
    };
  }

  if (action === "Rebound") {
    if (reboundType === "offensive") return { label: "Offensive rebound" };
    if (reboundType === "defensive") return { label: "Defensive rebound" };
    if (reboundType === "team") return { label: "Team rebound" };
    // The counters could not separate the two. Say so rather than pick one.
    return { label: "Rebound", note: "Offensive or defensive not recoverable" };
  }

  if (action === "Foul") {
    return {
      label: foulType(description) || "Foul",
      // Measured across all 636 games: the feed never names the fouled player.
      note: FOULED_UNKNOWN,
    };
  }

  if (action === "Turnover") {
    const cause = turnoverCause(description);
    return {
      label: "Turnover",
      detail: cause,
      // Steals are separate events. Pairing one to this turnover would mean
      // reading an adjacent row, which is not something the feed states.
      note: "Any steal is recorded as its own event",
    };
  }

  if (action === "period") return { label: "Period break" };
  if (action) return { label: action };

  const blank = blankActionKind(description);
  if (blank) return { label: blank };
  return { label: "Play" };
}

/** The one-line form, for the ribbon and the timeline read-out. */
export function playLabel(event, options) {
  const parts = describePlay(event, options);
  return parts.detail ? `${parts.label} — ${parts.detail}` : parts.label;
}
