/**
 * Who got the assist.
 *
 * The feed records assists ONLY inside made-shot descriptions, as
 * "Porter Jr. 1' Cutting Layup Shot (2 PTS) (Gordon 1 AST)". Crediting one
 * means turning a bare surname into a person, and that is where it gets hard.
 *
 * HOW THIS WAS ARRIVED AT
 * -----------------------
 * Measured against the boxscore across all 636 games at each step:
 *
 *   surname match only .......................... 290 / 636
 *   plus accent folding ......................... 475 / 636
 *   plus first-initial and prefix forms ......... 554 / 636
 *   plus suffixes stripped on BOTH sides ........ 444 / 636   WORSE
 *   layered: exact surname first, base after .... 593 / 636
 *   plus a game-derived alias map ............... 635 / 636
 *
 * The fourth line is the interesting one. Stripping "III" and "Jr." before
 * comparing looks like an obvious improvement and made things considerably
 * worse, because Boston carried Grant Williams and Robert Williams III at the
 * same time: the suffix is part of the surname and is load-bearing. The exact
 * form has to be tried first and the stripped form only as a fallback, which
 * is the order src/names.py already uses for the substitution feed.
 *
 * The last line needs no hardcoding. Enes Kanter appears in descriptions as
 * "Kanter" while the roster carries his current name, Enes Freedom, so a
 * surname match cannot work. But his OWN events carry both the description
 * text and his person id, so the game states the alias itself. Nothing is
 * typed in here.
 *
 * The one game that still disagrees is 0021600006, where the play-by-play and
 * the boxscore credit one assist to different teammates. That is two NBA
 * sources disagreeing, the same shape as the rebound anomaly in lib/scoring.js.
 */

const SUFFIXES = new Set(["jr", "sr", "ii", "iii", "iv", "v"]);

const ASSIST_PATTERN = /\(([^()]+?)\s+\d+\s+AST\)/;

/** Accents folded, punctuation dropped, lower case. */
export function fold(text) {
  return String(text || "")
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[.'’]/g, "")
    .trim();
}

/** "morris sr" to "morris". Only ever used as a FALLBACK. */
export function stripSuffix(surname) {
  const parts = String(surname || "").split(/\s+/).filter(Boolean);
  while (parts.length > 1 && SUFFIXES.has(parts[parts.length - 1])) parts.pop();
  return parts.join(" ");
}

/** The assisting player's name as the feed wrote it, or null. */
export function assistLabel(description) {
  const match = String(description || "").match(ASSIST_PATTERN);
  return match ? match[1].trim() : null;
}

function surnameOf(player) {
  const parts = fold(player?.name).split(/\s+/).filter(Boolean);
  return parts.length > 1 ? parts.slice(1).join(" ") : parts[0] || "";
}

function firstOf(player) {
  return fold(player?.name).split(/\s+/)[0] || "";
}

/**
 * How this game's descriptions name each player, derived from the game itself.
 *
 * Every event that carries a person id also carries text that begins with that
 * player's name as the feed writes it. Collecting the one- and two-token
 * leading forms gives a map from written name to person, and any form that
 * points at more than one person is discarded rather than guessed at.
 */
export function descriptionAliases(events) {
  const seen = new Map();
  const n = events?.person_id?.length || 0;
  for (let i = 0; i < n; i += 1) {
    const person = events.person_id[i];
    if (!person) continue;
    const tokens = String(events.description[i] || "").trim().split(/\s+/);
    for (const take of [1, 2]) {
      if (tokens.length < take + 1) continue;
      const lead = tokens.slice(0, take).join(" ");
      if (!/^[A-Za-z][\w.'’-]*( [A-Za-z][\w.'’-]*)?$/.test(lead)) continue;
      const key = fold(lead);
      if (!seen.has(key)) seen.set(key, new Set());
      seen.get(key).add(person);
    }
  }
  const aliases = new Map();
  for (const [key, people] of seen) {
    if (people.size === 1) aliases.set(key, [...people][0]);
  }
  return aliases;
}

/**
 * The assisting player, or null.
 *
 * Four passes in a fixed order, then the alias map. Order is the whole point:
 * the exact surname is tried before the suffix-stripped one, so Robert
 * Williams III is never collapsed into Grant Williams.
 */
export function resolveAssister(label, roster, aliases) {
  const wanted = fold(label);
  if (!wanted || !roster?.length) return null;
  const parts = wanted.split(/\s+/);

  const exact = roster.filter((p) => surnameOf(p) === wanted);
  if (exact.length === 1) return exact[0];

  if (parts.length > 1) {
    const prefix = parts[0];
    const rest = parts.slice(1).join(" ");
    const narrowed = roster.filter(
      (p) => surnameOf(p) === rest && firstOf(p).startsWith(prefix)
    );
    if (narrowed.length === 1) return narrowed[0];
  }

  const based = roster.filter(
    (p) => stripSuffix(surnameOf(p)) === stripSuffix(wanted)
  );
  if (based.length === 1) return based[0];

  if (parts.length > 1) {
    const prefix = parts[0];
    const rest = stripSuffix(parts.slice(1).join(" "));
    const narrowed = roster.filter(
      (p) => stripSuffix(surnameOf(p)) === rest && firstOf(p).startsWith(prefix)
    );
    if (narrowed.length === 1) return narrowed[0];
  }

  const aliased = aliases?.get(wanted);
  if (aliased) {
    const hit = roster.find((p) => p.person_id === aliased);
    if (hit) return hit;
  }
  return null;
}

/**
 * Running assist totals, in the same shape lib/scoring.js uses.
 *
 * The assister is always on the SHOOTING team, so the candidate pool is one
 * roster of about thirteen rather than the whole game. A small pool is what
 * makes surname matching safe.
 */
export function assistTimeline(events, players) {
  const byPlayer = new Map();
  const aliases = descriptionAliases(events);

  const byTeam = new Map();
  for (const player of Object.values(players || {})) {
    const team = String(player.team || "");
    if (!byTeam.has(team)) byTeam.set(team, []);
    byTeam.get(team).push(player);
  }

  const n = events?.action_type?.length || 0;
  for (let i = 0; i < n; i += 1) {
    if (events.action_type[i] !== "Made Shot") continue;
    const label = assistLabel(events.description[i]);
    if (!label) continue;

    const assister = resolveAssister(
      label,
      byTeam.get(String(events.team[i])) || [],
      aliases
    );
    if (!assister) continue;

    const history = byPlayer.get(assister.person_id) || [];
    history.push({
      index: i,
      total: (history.length ? history[history.length - 1].total : 0) + 1,
      points: 1,
    });
    byPlayer.set(assister.person_id, history);
  }
  return byPlayer;
}
