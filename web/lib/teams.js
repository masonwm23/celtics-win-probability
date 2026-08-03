/**
 * Which team goes on the left.
 *
 * ONE rule, in one place: the HOME team is on the left, the way a broadcast
 * graphic shows it. The scoreboard has always done this. The current-play
 * card did not, and printed Boston's score first no matter who was at home,
 * so in Boston's 318 away games the two disagreed about which number belonged
 * to whom. "54-57" under a scoreboard reading "DEN 57 ... 54 BOS" is not a
 * typo a viewer can spot; it just quietly reverses the game.
 *
 * Anything that renders a score pair should use this rather than deciding for
 * itself, which is why it is a module and not a couple of ternaries.
 */

export const CELTICS_ABBREV = "BOS";

/**
 * The two sides of a score, in display order.
 *
 * Each side carries everything needed to draw it: score, tricode, logo, and
 * whether it is Boston, so a caller never has to work that out again.
 */
export function scoreSides({ meta, celticsScore, opponentScore }) {
  const celtics = {
    abbrev: CELTICS_ABBREV,
    score: Number(celticsScore),
    logo: meta?.celtics_logo || null,
    name: meta?.celtics_name || CELTICS_ABBREV,
    isCeltics: true,
    isHome: Boolean(meta?.celtics_is_home),
  };
  const opponent = {
    abbrev: String(meta?.opponent || ""),
    score: Number(opponentScore),
    logo: meta?.opponent_logo || null,
    name: meta?.opponent_name || String(meta?.opponent || ""),
    isCeltics: false,
    isHome: !meta?.celtics_is_home,
  };

  return meta?.celtics_is_home
    ? { left: celtics, right: opponent }
    : { left: opponent, right: celtics };
}

/** "DEN 57 – 54 BOS", for a title attribute or a screen reader. */
export function scoreLine(sides) {
  return `${sides.left.abbrev} ${sides.left.score} – ${sides.right.score} ${sides.right.abbrev}`;
}
