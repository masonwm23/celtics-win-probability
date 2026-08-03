/**
 * Has the game finished, and who won.
 *
 * The panel used to show "End of 1st OT" at the last event of an overtime
 * game, which reads as though play is about to resume. The final event is the
 * end of the game and should say so.
 *
 * The winner is read from the SCORE, never from a team. Boston loses 223 of
 * the 636 games in this dataset, so anything that assumes the Celtics won is
 * wrong more than a third of the time.
 *
 * Two independent sources agree on the final score, and this checks them
 * against each other rather than trusting one: the last event's running score
 * columns, and `meta.celtics_final` / `meta.opponent_final`, which come from
 * the boxscore. When they disagree the panel says so instead of picking a
 * winner, because a disagreement there means something upstream is wrong and
 * quietly choosing a side would hide it.
 */

/** Ordinal overtime wording. Period 5 is the first overtime. */
export function overtimeLabel(periods) {
  const extra = Number(periods) - 4;
  if (!Number.isFinite(extra) || extra <= 0) return null;
  if (extra === 1) return "in overtime";
  if (extra === 2) return "in double overtime";
  if (extra === 3) return "in triple overtime";
  return `in ${extra} overtimes`;
}

/**
 * "End of OT", kept as secondary context rather than as the status.
 *
 * The ordinal is only written when it tells the reader something. A game with
 * one overtime said "End of 1st OT", which invites the question "so when is the
 * second?" — of a game that has already finished. There was no second. In a
 * double overtime game "End of 2nd OT" is worth saying, because a first one
 * happened and the reader may be looking for it.
 *
 * `totalPeriods` is how many periods the game ran to. It defaults to `period`,
 * which is the right assumption for the only caller: this label is attached to
 * the FINAL event, where the current period is by definition the last one.
 */
export function periodEndLabel(period, totalPeriods = period) {
  const value = Number(period);
  if (!Number.isFinite(value)) return null;
  if (value <= 4) return `End of Q${value}`;

  const total = Number(totalPeriods);
  const overtimes = Number.isFinite(total) ? total - 4 : value - 4;
  if (overtimes <= 1) return "End of OT";

  const extra = value - 4;
  const ordinal = extra === 1 ? "1st" : extra === 2 ? "2nd"
    : extra === 3 ? "3rd" : `${extra}th`;
  return `End of ${ordinal} OT`;
}

/** A team's full name when the payload carries one, otherwise its tricode. */
function teamName(name, abbrev) {
  const value = String(name || "").trim();
  return value || String(abbrev || "").trim();
}

/**
 * The state of the game at this cursor.
 *
 * `isFinal` is true only on the LAST event. Everything else on the object is
 * null until then, so a caller cannot accidentally show a winner mid-game.
 */
export function gameOutcome(events, meta, cursor) {
  const total = events?.wp?.length || 0;
  const isFinal = total > 0 && cursor === total - 1;
  if (!isFinal) return { isFinal: false };

  const celtics = Number(events.celtics_score[cursor]);
  const opponent = Number(events.opponent_score[cursor]);

  // The boxscore's own figures, for cross-checking. Absent on an older
  // payload, in which case the event columns stand alone and the check is
  // reported as not performed rather than as passed.
  const metaCeltics = meta?.celtics_final;
  const metaOpponent = meta?.opponent_final;
  const checked = metaCeltics !== undefined && metaOpponent !== undefined;
  const agrees = !checked
    || (Number(metaCeltics) === celtics && Number(metaOpponent) === opponent);

  const celticsName = teamName(meta?.celtics_name, "BOS");
  const opponentName = teamName(meta?.opponent_name, meta?.opponent);

  const base = {
    isFinal: true,
    celtics,
    opponent,
    celticsName,
    opponentName,
    opponentAbbrev: String(meta?.opponent || ""),
    periods: Number(meta?.periods) || 4,
    overtime: overtimeLabel(meta?.periods),
    periodEnd: periodEndLabel(events.period[cursor], Number(meta?.periods) || undefined),
    scoresAgree: agrees,
    scoreCheck: checked,
  };

  // A tie is not a valid NBA result. If the two columns say one, something is
  // wrong with the data and no winner is claimed.
  if (celtics === opponent) {
    return { ...base, tie: true, winnerName: null, loserName: null };
  }

  const celticsWon = celtics > opponent;
  return {
    ...base,
    tie: false,
    celticsWon,
    winnerName: celticsWon ? celticsName : opponentName,
    loserName: celticsWon ? opponentName : celticsName,
    winnerAbbrev: celticsWon ? "BOS" : base.opponentAbbrev,
    winningScore: Math.max(celtics, opponent),
    losingScore: Math.min(celtics, opponent),
  };
}

/** "FINAL · Celtics win 116-109 in overtime", or the honest alternative. */
export function finalHeadline(outcome) {
  if (!outcome?.isFinal) return null;
  if (outcome.tie) {
    return `FINAL · scores tied ${outcome.celtics}-${outcome.opponent}, `
      + "which is not a valid result";
  }
  const short = outcome.celticsWon ? "Celtics win" : `${outcome.opponentAbbrev} win`;
  const overtime = outcome.overtime ? ` ${outcome.overtime}` : "";
  return `FINAL · ${short} ${outcome.winningScore}-${outcome.losingScore}${overtime}`;
}

/** "Game complete · Boston Celtics defeat Denver Nuggets". */
export function finalSentence(outcome) {
  if (!outcome?.isFinal) return null;
  if (outcome.tie) return "Game complete · the recorded scores are tied";
  return `Game complete · ${outcome.winnerName} defeat ${outcome.loserName}`;
}
