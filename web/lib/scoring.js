/**
 * Points, accumulated play by play.
 *
 * WHY THIS SUMS RATHER THAN READS
 * -------------------------------
 * The feed writes each scorer's running total into the description: "Horford
 * 3PT Jump Shot (12 PTS)". That looks like the obvious source, and it is the
 * wrong one.
 *
 * Checked across all 636 games in this dataset:
 *
 *   summing the shot values   matches the boxscore in 636 of 636 games
 *   the feed's own counter     matches the boxscore in 632 of 636 games
 *
 * Game 0022200107 shows what goes wrong. Al Horford's counter reads 2, 4, then
 * jumps to 12, then comes BACK DOWN to 6, 7, 9. It is not even monotone. His
 * boxscore total is 12, which is what summing 2 + 2 + 3 + 2 + 1 + 2 gives.
 *
 * So every total here is built from `shot_value` on made field goals and one
 * point per made free throw, and the feed's running counter is never read. A
 * number that is right 99.4% of the time is a number that is wrong on screen
 * four times, with nothing to indicate which four.
 *
 * FREE THROWS
 * -----------
 * They carry `shot_value` 0 and an empty `shot_result`; whether one went in is
 * written as a leading "MISS" in the description, the same convention
 * lib/playby.js handles. One point each when they went in.
 */

/** Points scored on this event: 0, 1, 2 or 3. */
export function eventPoints(event) {
  const action = String(event?.action_type || "");
  if (action === "Made Shot") {
    const value = Number(event?.shot_value);
    // A made field goal is worth 2 or 3. Anything else is a corrupt row and is
    // counted as 2 rather than as 0, because a made shot scored nothing is a
    // claim the feed never makes.
    return value === 3 ? 3 : 2;
  }
  if (action === "Free Throw") {
    return /^MISS\b/i.test(String(event?.description || "").trim()) ? 0 : 1;
  }
  return 0;
}

/**
 * Every scorer's running total, as a sparse timeline.
 *
 * Returns a Map from person id to an ascending array of `{ index, total }`,
 * one entry per scoring event. Sparse rather than a full per-event matrix
 * because a player scores perhaps a dozen times in a game and the cursor needs
 * a lookup, not a copy.
 */
export function pointsTimeline(events) {
  const byPlayer = new Map();
  const n = events?.action_type?.length || 0;

  for (let i = 0; i < n; i += 1) {
    const points = eventPoints({
      action_type: events.action_type[i],
      shot_value: events.shot_value[i],
      description: events.description[i],
    });
    if (!points) continue;

    const person = events.person_id[i];
    if (person === null || person === undefined || person === 0) continue;

    const history = byPlayer.get(person) || [];
    const running = (history.length ? history[history.length - 1].total : 0) + points;
    history.push({ index: i, total: running, points });
    byPlayer.set(person, history);
  }
  return byPlayer;
}

/**
 * Every rebounder's running count.
 *
 * Counted from Rebound EVENTS, one each, rather than read from the running
 * "(Off:n Def:n)" pair, for the same reason points are summed: a counter can
 * be wrong and a count of events cannot disagree with itself.
 *
 * Reconciled against the boxscore across all 636 games: 635 match. The single
 * exception is game 0021601219, where the play-by-play credits Isaiah Thomas
 * four rebounds (its own counters reach Off:2 Def:2) and the boxscore records
 * three. That is a disagreement between two NBA sources in one 2016-17 game,
 * the season this project already documents for feed defects, and not a
 * failure of the method.
 *
 * Team rebounds carry no player and are excluded, which is why a team's player
 * rebound totals do not sum to its team rebound count.
 */
export function reboundTimeline(events) {
  const byPlayer = new Map();
  const n = events?.action_type?.length || 0;
  for (let i = 0; i < n; i += 1) {
    if (events.action_type[i] !== "Rebound") continue;
    const person = events.person_id[i];
    if (person === null || person === undefined || person === 0) continue;
    const history = byPlayer.get(person) || [];
    history.push({
      index: i,
      total: (history.length ? history[history.length - 1].total : 0) + 1,
      points: 1,
    });
    byPlayer.set(person, history);
  }
  return byPlayer;
}

/**
 * A player's points as of an event, inclusive.
 *
 * Binary search over that player's own scoring events, so scrubbing to the
 * middle of a game costs a handful of comparisons rather than a re-scan.
 */
export function pointsAt(timeline, personId, index) {
  const history = timeline?.get(personId);
  if (!history || !history.length || index < history[0].index) return 0;

  let low = 0;
  let high = history.length - 1;
  let best = 0;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (history[mid].index <= index) {
      best = history[mid].total;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return best;
}

/** Points scored by this player on THIS event, for the "+3" flash. */
export function pointsOnEvent(timeline, personId, index) {
  const history = timeline?.get(personId);
  if (!history) return 0;
  const hit = history.find((entry) => entry.index === index);
  return hit ? hit.points : 0;
}

/**
 * The scoreboard for one team as of an event.
 *
 * Sorted by points, then by name, so the order is stable while the numbers
 * move. Players who have not scored are kept, with zero, because a live
 * boxscore that hides them makes it look like they are not in the game.
 */
export function teamScoring(timeline, players, index, { celtics }) {
  return Object.values(players || {})
    .filter((player) => Boolean(player.is_celtics) === celtics)
    .map((player) => ({
      player,
      points: pointsAt(timeline, player.person_id, index),
      justScored: pointsOnEvent(timeline, player.person_id, index),
    }))
    .sort((a, b) => {
      if (b.points !== a.points) return b.points - a.points;
      return String(a.player.name).localeCompare(String(b.player.name));
    });
}

/**
 * Do the summed player points equal the scoreboard?
 *
 * The scoreboard column is written by the feed independently of the
 * descriptions, so agreement between the two is real evidence rather than a
 * tautology. Exposed so the interface can show whether it reconciles at the
 * current event instead of asserting that it does.
 */
export function reconcile(rows, teamScore) {
  const total = rows.reduce((sum, row) => sum + row.points, 0);
  return { total, expected: teamScore, agrees: total === teamScore };
}
