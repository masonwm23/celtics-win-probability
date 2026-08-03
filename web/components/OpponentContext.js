"use client";

/**
 * Who the Celtics were playing, and how that team had actually been going.
 *
 * The panel this replaces was one sentence with three problems.
 *
 * It said the opponent "were 1.5 points per game on the season to that date",
 * which reads as though Washington scored a point and a half a night. The
 * number is a point DIFFERENTIAL, how much they outscored opponents by, and it
 * is the only figure on screen for a reader with no background to misread.
 *
 * Worse, 1.5 was not what Washington did. Every *_prior field in the payload is
 * SHRUNK toward a neutral centre by SHRINKAGE_GAMES before it is stored, which
 * is correct for a model feature and wrong to narrate as history: Washington
 * had outscored opponents by 4.4 a game and were 4-1, and 1.5 is that 4.4
 * pulled two thirds of the way to zero because only five games had been played.
 * This panel reads the *_raw fields, which are the season-to-date figures, and
 * mentions the shrunk version as what a model would have been given.
 *
 * And it ended "so nothing from this game or later leaks into the feature",
 * true about the leakage and wrong about the feature. Opponent context is NOT a
 * model input. The shipped model has thirteen features and none describes the
 * opponent, because every opponent formulation tested made out-of-sample
 * prediction worse: the full opponent tier moved Brier from 0.1630 to 0.2119,
 * and three pre-registered variants were worse too. The lineup panel says its
 * numbers are descriptive in as many words. This one implied the opposite.
 *
 * Everything here degrades to nothing rather than guessing. A raw figure is
 * null before a team has played a game, because a record over no games is not
 * zero, it is undefined, and a card that cannot be filled honestly is not
 * drawn.
 */

const signed = (value, digits = 1) =>
  `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;

const isNum = (value) => typeof value === "number" && Number.isFinite(value);

export default function OpponentContext({ context, opponent, opponentName }) {
  if (!context) return null;

  const {
    opponent_games_played_prior: games,
    opponent_win_pct_prior_raw: theirWinPct,
    opponent_point_diff_prior_raw: theirDiff,
    celtics_point_diff_prior_raw: ourDiff,
    opponent_point_diff_prior: theirDiffShrunk,
    shrinkage_games: shrinkage,
  } = context;

  const name = opponentName || opponent;

  // Wins come from the RAW rate, so the record is the real one. The shrunk rate
  // would have printed 3-2 for a team that went 4-1.
  const wins = isNum(theirWinPct) && isNum(games)
    ? Math.round(theirWinPct * games)
    : null;
  const losses = wins === null ? null : games - wins;

  // Computed here rather than read from the payload, because strength_diff_prior
  // is the difference of the two SHRUNK figures and would not match the two raw
  // cards sitting either side of it.
  const gap = isNum(theirDiff) && isNum(ourDiff) ? ourDiff - theirDiff : null;
  const bostonAhead = gap !== null && gap > 0;

  const played = isNum(games) && games > 0;

  return (
    <div className="panel__body">
      {played ? (
        <p className="mq__p" style={{ marginBottom: 12 }}>
          Going into this game, {name} had played <b>{games}</b>{" "}
          {games === 1 ? "game" : "games"} that season
          {wins !== null && (
            <>
              {" "}
              and won <b>{wins}</b> of {games === 1 ? "it" : "them"}
            </>
          )}
          {isNum(theirDiff) && (
            <>
              , {theirDiff >= 0 ? "outscoring" : "being outscored by"} their
              opponents by an average of{" "}
              <b>{Math.abs(theirDiff).toFixed(1)} points a night</b> &mdash;
              that is the <em>margin</em>, not what they scored
            </>
          )}
          .
        </p>
      ) : (
        <p className="mq__p" style={{ marginBottom: 12 }}>
          This was {name}&apos;s first game of the season, so there is no
          season-to-date form to show. A record over no games is undefined
          rather than zero, and the cards below are left out rather than filled
          with a number nobody measured.
        </p>
      )}

      {played && (
        <div className="impact">
          {wins !== null && (
            <div className="impact__card">
              <div className="impact__label">{opponent} record</div>
              <div className="impact__value">
                {wins}&ndash;{losses}
              </div>
              <div className="impact__sub">going into this game</div>
            </div>
          )}

          {isNum(theirDiff) && (
            <div className="impact__card">
              <div className="impact__label">{opponent} scoring margin</div>
              <div
                className="impact__value"
                style={{ color: theirDiff >= 0 ? "var(--text)" : "var(--opponent)" }}
              >
                {signed(theirDiff)}
              </div>
              <div className="impact__sub">points a game, season to date</div>
            </div>
          )}

          {isNum(ourDiff) && (
            <div className="impact__card">
              <div className="impact__label">Boston scoring margin</div>
              <div
                className="impact__value"
                style={{ color: ourDiff >= 0 ? "var(--celtics)" : "var(--opponent)" }}
              >
                {signed(ourDiff)}
              </div>
              <div className="impact__sub">points a game, same date</div>
            </div>
          )}

          {gap !== null && (
            <div className="impact__card">
              <div className="impact__label">On paper</div>
              <div
                className="impact__value"
                style={{ color: bostonAhead ? "var(--celtics)" : "var(--opponent)" }}
              >
                {signed(gap)}
              </div>
              <div className="impact__sub">
                Boston {bostonAhead ? "ahead of" : "behind"} {opponent}
              </div>
            </div>
          )}
        </div>
      )}

      <p className="note" style={{ marginTop: played ? 14 : 0, marginBottom: 0 }}>
        <strong>
          These numbers describe the matchup. The model never sees them.
        </strong>{" "}
        Opponent strength was built, tested and left out, because every version
        of it made the forecasts worse on games the model had not seen &mdash;
        the error score went from 0.1630 to 0.2119, and three alternatives fixed
        in advance were all worse too. That is reported as a finding rather than
        quietly dropped.
        {played && isNum(theirDiffShrunk) && isNum(theirDiff) && (
          <>
            {" "}
            Had it been used, the model would have been handed{" "}
            <b>{signed(theirDiffShrunk)}</b> rather than{" "}
            <b>{signed(theirDiff)}</b>: an early-season average is pulled toward
            the league&apos;s middle
            {isNum(shrinkage) && <> by {shrinkage} games&apos; worth of doubt</>}
            , so that {games} games do not get treated as though they proved
            something.
          </>
        )}{" "}
        Everything above counts only games played <em>before</em> this one, so a
        team&apos;s later results cannot creep backwards into how they looked on
        the night.
      </p>
    </div>
  );
}
