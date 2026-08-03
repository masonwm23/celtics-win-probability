"use client";

import { useMemo } from "react";
import { jerseyNumber } from "@/lib/format";
import { pointsTimeline, reconcile, teamScoring } from "@/lib/scoring";

/**
 * Points as the game runs.
 *
 * A live boxscore tied to the same cursor as everything else: scrub, step or
 * press Play and each player's total climbs. Nothing is projected or smoothed;
 * a bar moves when that player actually scored.
 *
 * Every total is SUMMED from made field goals and made free throws. The feed
 * also writes each scorer's running total into the description, and that
 * counter disagrees with the boxscore in 4 of the 636 games, so it is not
 * read. See lib/scoring.js for the game where it goes 2, 4, 12, 6, 7, 9.
 *
 * The panel checks itself. The scoreboard column in the event table is written
 * independently of the descriptions these totals come from, so it is a genuine
 * second source, and the footer reports whether the two agree AT THE CURRENT
 * EVENT rather than claiming they always will. Measured over game 0022300906:
 * 980 team-events, zero disagreements.
 */
export default function ScoringRun({
  events,
  cursor,
  players,
  meta,
  celticsLineup,
  opponentLineup,
}) {
  const timeline = useMemo(() => pointsTimeline(events), [events]);

  const boston = teamScoring(timeline, players, cursor, { celtics: true });
  const opponent = teamScoring(timeline, players, cursor, { celtics: false });

  const bosCheck = reconcile(boston, events.celtics_score[cursor]);
  const oppCheck = reconcile(opponent, events.opponent_score[cursor]);

  // One scale for both teams, so a 20-point bar is the same length on either
  // side. Scaling each team to its own leader would make a blowout look level.
  const most = Math.max(
    1,
    ...boston.map((r) => r.points),
    ...opponent.map((r) => r.points)
  );

  const onCourt = new Set([...(celticsLineup || []), ...(opponentLineup || [])]);

  return (
    <div className="scoring">
      <Column
        label="BOS"
        rows={boston}
        most={most}
        check={bosCheck}
        onCourt={onCourt}
        celtics
      />
      <Column
        label={meta.opponent}
        rows={opponent}
        most={most}
        check={oppCheck}
        onCourt={onCourt}
      />
    </div>
  );
}

function Column({ label, rows, most, check, onCourt, celtics = false }) {
  const scorers = rows.filter((row) => row.points > 0).length;
  return (
    <div className={`scorecol ${celtics ? "scorecol--bos" : "scorecol--opp"}`}>
      <div className="scorecol__head">
        <span className="scorecol__team">{label}</span>
        <span className="scorecol__total">{check.total}</span>
      </div>

      <ul className="scorecol__list">
        {rows.map((row) => (
          <li
            key={row.player.person_id}
            className={[
              "scorerow",
              row.points === 0 ? "scorerow--zero" : "",
              row.justScored ? "scorerow--scored" : "",
              onCourt.has(row.player.person_id) ? "scorerow--oncourt" : "",
            ].join(" ")}
          >
            <span className="scorerow__name">
              {row.player.name}
              {row.player.jersey && (
                <span className="scorerow__num">
                  {" "}
                  #{jerseyNumber(row.player.jersey)}
                </span>
              )}
            </span>
            <span className="scorerow__bar" aria-hidden="true">
              <span
                className="scorerow__fill"
                style={{ width: `${(row.points / most) * 100}%` }}
              />
            </span>
            <span className="scorerow__pts">
              {row.points}
              {row.justScored > 0 && (
                <span className="scorerow__plus">+{row.justScored}</span>
              )}
            </span>
          </li>
        ))}
      </ul>

      <p className="scorecol__foot">
        {scorers} of {rows.length} have scored ·{" "}
        {check.agrees ? (
          <span className="is-ok">matches the scoreboard ({check.expected})</span>
        ) : (
          <span className="is-warn">
            summed {check.total}, scoreboard {check.expected}
          </span>
        )}
      </p>
    </div>
  );
}
