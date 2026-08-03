"""
Phase 8b: the paper's figures.

Every figure is built from a saved prediction table or score file, never from a
number typed in by hand. If a result changes, the figures change with it. Nothing
here refits a model.

A figure is not decoration. Each one below exists to make a specific claim
legible in a way a table cannot:

  Figure 1  Reliability. Does a stated probability mean what it says? This is the
            property a live dashboard lives or dies on, and a diagram shows the
            shape of the miscalibration where a table shows only its size.

  Figure 2  THE SIGNATURE FIGURE. Damage against feature resolution, for real
            opponent quality and for a meaningless random column. Two curves that
            converge is an argument no table makes as quickly.

  Figure 3  Brier skill by game phase. The pooled number is dominated by
            fourth-quarter events where any model looks good. This shows where
            the model actually earns anything.

  Figure 4  A single game's win probability trace, chosen as the largest genuine
            comeback in the dataset. Sanity as much as illustration: if the
            curve is implausible on a game a human remembers, something is wrong.

  Figure 5  Training against out-of-fold Brier. The memorisation signature, and
            the clearest single image of the Phase 6 artefact.

Outputs: figures/fig1..fig5 as PNG at 200 dpi, plus a manifest.
"""

import logging

import matplotlib
matplotlib.use("Agg")          # write files; do not depend on a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

from src import config, evaluate, features  # noqa: E402

logger = logging.getLogger(__name__)

DPI = 200
FIGSIZE = (7.5, 5.0)

# Boston green and a neutral grey, so the figures read in print and greyscale.
CELTICS = "#007A33"
ACCENT = "#BA9653"
GREY = "#5A5A5A"
LIGHT = "#B8B8B8"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, name, tight=True):
    config.ensure_dirs()
    path = config.FIGURES_DIR / name
    if tight:
        # Figures that set their own gridspec spacing manage their own layout.
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# Figure 1: reliability
# ---------------------------------------------------------------------------

def figure_reliability(oof: pd.DataFrame):
    y = oof[features.TARGET_COLUMN].to_numpy()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot([0, 1], [0, 1], linestyle="--", color=LIGHT, linewidth=1.2,
            label="perfect calibration")

    for key, label, colour in (("tier2_generic", "Tier 2: generic baseline", GREY),
                               ("tier3_celtics", "Tier 3: Celtics-specific", CELTICS)):
        if key not in oof.columns:
            continue
        table = evaluate.calibration_table(y, oof[key].to_numpy())
        ax.plot(table["mean_predicted"], table["observed"], marker="o",
                markersize=4.5, linewidth=1.6, color=colour, label=label)

    _style(ax, "Figure 1. Reliability, out of fold",
           "Predicted probability of a Celtics win",
           "Observed frequency of a Celtics win")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.text(0.98, 0.02,
            "Points above the diagonal are under-confident,\nbelow are "
            "over-confident.",
            transform=ax.transAxes, fontsize=7.5, color=GREY,
            ha="right", va="bottom")
    return _save(fig, "fig1_reliability.png")


# ---------------------------------------------------------------------------
# Figure 2: the dose-response curve
# ---------------------------------------------------------------------------

def figure_dose_response(scores: pd.DataFrame):
    """
    Damage against feature resolution, real feature versus meaningless column.

    Both ladders are nested, so the only thing changing along the x axis is how
    many distinct values the feature can take.
    """
    baseline = float(scores.loc[scores["tier"].eq("p7_tier3"), "brier"].iloc[0])
    opponent_keys = ["p7_opp_bins5", "p7_opp_bins20", "p7_opp_bins100",
                     "p7_opp_raw"]
    random_keys = ["p7_rand_bins5", "p7_rand_bins20", "p7_rand_bins100",
                   "p7_rand_raw"]

    def series(keys):
        rows = scores.set_index("tier").loc[keys]
        return (rows["cardinality"].to_numpy(dtype=float),
                rows["brier"].to_numpy(dtype=float) - baseline)

    opp_x, opp_y = series(opponent_keys)
    rand_x, rand_y = series(random_keys)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhline(0.0, color=LIGHT, linewidth=1.2, linestyle="--")
    ax.plot(opp_x, opp_y, marker="o", markersize=6, linewidth=2.0,
            color=CELTICS, label="Real opponent strength")
    ax.plot(rand_x, rand_y, marker="s", markersize=6, linewidth=2.0,
            color=ACCENT, linestyle="--", label="Random numbers (no information)")

    ax.set_xscale("log")
    ax.set_xticks([5, 20, 100, 636])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    _style(ax, "Figure 2. Damage grows with feature resolution,\n"
                "whether or not the feature means anything",
           "Distinct values the feature takes across 636 games (log scale)",
           "Increase in out-of-fold Brier vs game state alone")
    ax.margins(y=0.16)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.annotate("the curves converge:\ninformation content stops mattering",
                xy=(opp_x[-1], opp_y[-1]), xytext=(-186, -92),
                textcoords="offset points", fontsize=7.5, color=GREY,
                ha="left",
                arrowprops=dict(arrowstyle="->", color=GREY, linewidth=0.8))
    ax.annotate("the gap here is what real\nopponent information is worth",
                xy=((opp_x[0] + rand_x[0]) / 2, (opp_y[0] + rand_y[0]) / 2),
                xytext=(30, 46), textcoords="offset points", fontsize=7.5,
                color=GREY, ha="left",
                arrowprops=dict(arrowstyle="->", color=GREY, linewidth=0.8))
    return _save(fig, "fig2_dose_response.png")


# ---------------------------------------------------------------------------
# Figure 3: skill by game phase
# ---------------------------------------------------------------------------

def figure_phase_skill(frame: pd.DataFrame, oof: pd.DataFrame):
    y = oof[features.TARGET_COLUMN]
    tiers = [("tier1_pregame", "Tier 1: pregame only", LIGHT),
             ("tier2_generic", "Tier 2: generic baseline", GREY),
             ("tier3_celtics", "Tier 3: Celtics-specific", CELTICS)]
    tiers = [t for t in tiers if t[0] in oof.columns]

    tables = {key: evaluate.phase_table(frame, y, oof[key].to_numpy())
              for key, _label, _colour in tiers}
    phases = list(tables[tiers[0][0]]["phase"])
    positions = np.arange(len(phases))
    width = 0.8 / len(tiers)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for i, (key, label, colour) in enumerate(tiers):
        values = tables[key]["brier_skill"].to_numpy() * 100
        ax.bar(positions + i * width - 0.4 + width / 2, values, width * 0.92,
               label=label, color=colour)
    ax.axhline(0.0, color="black", linewidth=0.8)

    _style(ax, "Figure 3. Where the model actually earns anything",
           "", "Brier skill over predicting the base rate (%)")
    # Headroom so the note does not sit on top of the tallest bars.
    ax.set_ylim(top=ax.get_ylim()[1] * 1.22)
    ax.set_xticks(positions)
    ax.set_xticklabels(phases, rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.text(0.99, 0.97,
            "A pooled number is dominated by period 4,\nwhere a large lead "
            "genuinely does decide the game.",
            transform=ax.transAxes, fontsize=7.5, color=GREY,
            ha="right", va="top")
    return _save(fig, "fig3_phase_skill.png")


# ---------------------------------------------------------------------------
# Figure 4: one game's trace
# ---------------------------------------------------------------------------

def choose_comeback_game(frame: pd.DataFrame, oof: pd.DataFrame,
                         key="tier3_celtics"):
    """
    The largest genuine comeback: the game Boston won after its out-of-fold win
    probability fell lowest.

    Chosen by a rule rather than by eye, so the figure is not cherry-picked.
    """
    joined = pd.DataFrame({
        "game_id": oof["game_id"],
        "probability": oof[key],
        "won": oof[features.TARGET_COLUMN],
    })
    wins = joined.loc[joined["won"].eq(1)]
    lowest = wins.groupby("game_id")["probability"].min()
    return str(lowest.idxmin()), float(lowest.min())


def figure_game_trace(frame: pd.DataFrame, oof: pd.DataFrame,
                      game_id=None, key="tier3_celtics"):
    if game_id is None:
        game_id, _low = choose_comeback_game(frame, oof, key)

    rows = frame.loc[frame["game_id"].eq(game_id)].copy()
    probabilities = oof.loc[oof["game_id"].eq(game_id), key].to_numpy()
    if len(rows) != len(probabilities):
        raise ValueError("event frame and prediction frame disagree on length "
                         f"for game {game_id}")
    rows["probability"] = probabilities
    rows = rows.sort_values("event_index")

    elapsed = rows["seconds_elapsed_game"].to_numpy() / 60.0
    probability = rows["probability"].to_numpy()
    margin = rows["celtics_margin"].to_numpy()
    season = rows["season"].iloc[0]

    # Name the game if the index is available. "BOS @ PHX, 8 November 2018" is
    # a great deal more use to a reader than a ten-digit identifier.
    caption = f"game {game_id} ({season})"
    if config.GAME_INDEX_CSV.exists():
        index = pd.read_csv(config.GAME_INDEX_CSV, dtype={"GAME_ID": str},
                            parse_dates=["GAME_DATE"])
        index["GAME_ID"] = index["GAME_ID"].str.zfill(10)
        match = index.loc[index["GAME_ID"].eq(game_id)]
        if len(match):
            row = match.iloc[0]
            caption = (f"{row['MATCHUP']}, "
                       f"{row['GAME_DATE']:%-d %B %Y} ({season})")

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.5, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.12})

    top.axhline(0.5, color=LIGHT, linewidth=1.0, linestyle="--")
    top.plot(elapsed, probability, color=CELTICS, linewidth=1.6)
    top.fill_between(elapsed, 0.5, probability,
                     where=probability >= 0.5, color=CELTICS, alpha=0.18)
    top.fill_between(elapsed, 0.5, probability,
                     where=probability < 0.5, color=ACCENT, alpha=0.22)
    top.set_ylim(0, 1)
    _style(top, f"Figure 4. Win probability, {caption}",
           "", "P(Celtics win), out of fold")

    low = float(np.min(probability))
    low_at = float(elapsed[int(np.argmin(probability))])
    top.annotate(f"low point {low:.3f}", xy=(low_at, low),
                 xytext=(-30, 62), textcoords="offset points", fontsize=8,
                 color=GREY, ha="right",
                 arrowprops=dict(arrowstyle="->", color=GREY, linewidth=0.8))
    top.text(0.015, 0.96,
             "Prediction comes from a model that never saw this season.",
             transform=top.transAxes, fontsize=7.5, color=GREY,
             ha="left", va="top")

    bottom.axhline(0.0, color=LIGHT, linewidth=1.0)
    bottom.plot(elapsed, margin, color=GREY, linewidth=1.2)
    _style(bottom, "", "Minutes elapsed", "Score margin")

    for period_end in (12, 24, 36):
        for ax in (top, bottom):
            ax.axvline(period_end, color=LIGHT, linewidth=0.7, linestyle=":")

    return _save(fig, "fig4_game_trace.png", tight=False)


# ---------------------------------------------------------------------------
# Figure 5: the memorisation signature
# ---------------------------------------------------------------------------

def figure_memorisation(scores: pd.DataFrame):
    keys = ["p7_tier3", "p7_opp_bins5", "p7_opp_bins20", "p7_opp_bins100",
            "p7_opp_raw", "p7_rand_raw", "p7_linear_opp", "p7_tier2"]
    keys = [k for k in keys if k in set(scores["tier"])]
    rows = scores.set_index("tier").loc[keys]

    labels = [str(n) for n in rows["name"]]
    train = rows["train_brier"].to_numpy(dtype=float)
    oof = rows["brier"].to_numpy(dtype=float)
    positions = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, (t, o) in enumerate(zip(train, oof)):
        ax.plot([t, o], [i, i], color=LIGHT, linewidth=1.6, zorder=1)
    ax.scatter(train, positions, s=48, color=ACCENT, zorder=2, label="training")
    ax.scatter(oof, positions, s=48, color=CELTICS, zorder=2,
               label="out of fold")

    ax.set_yticks(positions)
    ax.set_yticklabels([label[:52] for label in labels], fontsize=8)
    ax.invert_yaxis()
    _style(ax, "Figure 5. The memorisation signature", "Brier score", "")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.text(0.01, 0.02,
            "A long bar means the fit did not survive the season boundary.",
            transform=ax.transAxes, fontsize=7.5, color=GREY,
            ha="left", va="bottom")
    return _save(fig, "fig5_memorisation.png")


# ---------------------------------------------------------------------------

def build_all():
    config.ensure_dirs()
    written, skipped = [], []

    frame = pd.read_parquet(config.MODEL_FRAME_PARQUET).reset_index(drop=True)

    oof_path = config.PROCESSED_DIR / "oof_predictions.parquet"
    if oof_path.exists():
        oof = pd.read_parquet(oof_path)
        written.append(figure_reliability(oof))
        written.append(figure_phase_skill(frame, oof))
        written.append(figure_game_trace(frame, oof))
    else:
        skipped.append((oof_path.name, "run scripts/11_train_model.py"))

    scores_path = config.REPORTS_DIR / "phase7_scores.csv"
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
        written.append(figure_dose_response(scores))
        written.append(figure_memorisation(scores))
    else:
        skipped.append((scores_path.name, "re-run scripts/16_run_clean_tests.py"))

    return written, skipped


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    written, skipped = build_all()

    print()
    print("=" * 70)
    print("FIGURES")
    print("=" * 70)
    for path in written:
        print(f"  wrote  {path}")
    for name, hint in skipped:
        print(f"  SKIPPED, {name} not found: {hint}")
    if not skipped:
        print()
        print("  All five figures built. Figure 2 is the paper's signature "
              "image.")
    return written


if __name__ == "__main__":
    main()
