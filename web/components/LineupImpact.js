"use client";

import { signed, percent } from "@/lib/format";

/**
 * Lineup impact cards.
 *
 * THE HONEST VERSION. The design brief asked for cards showing what the
 * on-court five is doing to the win probability. The model that ships does not
 * use lineup features, because lineup strength measurably HURT out-of-sample
 * prediction, and that negative result survived every clean re-test:
 *
 *   tier 3, no lineup                       Brier 0.1630
 *   plus lineup strength                    Brier 0.1823   (interval excludes zero)
 *   lineup in 5 bins, memorisation blocked   -0.0081  [-0.0131, -0.0030]
 *   lineup in a linear model, no splits      -0.0059  [-0.0108, -0.0009]
 *
 * So these cards describe the lineup on the floor. They do not claim to move
 * the probability, and the panel says so in as many words. Wiring them to the
 * lineup model would have matched the brief and contradicted the research.
 */
export default function LineupImpact({ players, lineup, opponentLineup, wp }) {
  const onCourt = lineup.map((id) => players[id]).filter(Boolean);
  const opponentOnCourt = opponentLineup
    .map((id) => players[id])
    .filter(Boolean);

  const sumValue = (group) =>
    group.reduce((total, p) => total + (p.player_value ?? 0), 0);
  const withValue = (group) => group.filter((p) => p.player_value !== null);

  const celticsValue = sumValue(onCourt);
  const opponentValue = sumValue(opponentOnCourt);
  const starters = onCourt.filter((p) => p.is_starter).length;
  const covered = withValue(onCourt).length;

  return (
    <div className="panel">
      <div className="panel__head">
        <h2 className="panel__title">Lineup on the floor</h2>
        <span className="badge">descriptive</span>
      </div>
      <div className="panel__body">
        <div className="impact">
          <div className="impact__card">
            <div className="impact__label">Boston lineup value</div>
            <div className="impact__value" style={{ color: "var(--celtics)" }}>
              {signed(celticsValue, 3)}
            </div>
            <div className="impact__sub">
              {covered} of {onCourt.length} players valued
            </div>
          </div>

          <div className="impact__card">
            <div className="impact__label">Opponent lineup value</div>
            <div className="impact__value" style={{ color: "var(--opponent)" }}>
              {signed(opponentValue, 3)}
            </div>
            <div className="impact__sub">
              {withValue(opponentOnCourt).length} of {opponentOnCourt.length} valued
            </div>
          </div>

          <div className="impact__card">
            <div className="impact__label">Difference</div>
            <div className="impact__value">
              {signed(celticsValue - opponentValue, 3)}
            </div>
            <div className="impact__sub">Boston minus opponent</div>
          </div>

          <div className="impact__card">
            <div className="impact__label">Starters on court</div>
            <div className="impact__value">{starters} / 5</div>
            <div className="impact__sub">from the boxscore</div>
          </div>

          <div className="impact__card">
            <div className="impact__label">Win probability</div>
            <div className="impact__value" style={{ color: "var(--celtics)" }}>
              {percent(wp)}
            </div>
            <div className="impact__sub">out of fold</div>
          </div>
        </div>

        <p className="note" style={{ marginTop: 14, marginBottom: 0 }}>
          <strong>These cards describe the lineup. They do not drive the
          probability.</strong>{" "}
          Knowing who was on the floor was added to the model and tested
          honestly, and it made the forecasts measurably <em>worse</em> on games
          the model had not seen. Its error score went from 0.1630 to 0.1823,
          and lower is better, so that is a clear step backwards rather than a
          rounding error &mdash; the &ldquo;How good is this model?&rdquo; panel
          explains what those numbers are and what counts as good. Three
          pre-registered alternative parameterisations were all worse too, and
          the result survived a clean re-test that blocked the memorisation
          artefact described in the paper. Player value here is a shrunk
          per-minute plus/minus, shown because it is real information about who
          is on the floor, not because the model uses it.
        </p>
      </div>
    </div>
  );
}
