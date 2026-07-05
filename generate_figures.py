"""
generate_figures.py

Regenerates both publication figures for:
    Explainable Reinforcement Learning through Information-Theoretic
    Policy Compression (IEEE TAI, manuscript TAI-2026-Apr-A-00737)

Figures produced (each as 300-DPI PNG + vector PDF):
  1. pareto-frontier.png / .pdf
     Main manuscript, Figure 1 ("Pareto Frontier of Explainability vs.
     Performance with Dual-Fidelity Metrics"). Data source: Table I
     (tab:results).
  2. information_capacity_plot.png / .pdf
     Supplementary Material, Appendix A, Figure 1 ("Empirical Validation
     of Information-Theoretic Bounds"). Data source: Table I's PTG (%)
     column, re-expressed against Information Capacity H(C) = tree depth
     d, per Proposition 1 (H(C) <= d bits).

Both figures use the same serif/IEEE house style so they read as a
matched pair regardless of which one Reviewer 2 originally meant by
"Fig. 3".
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from google.colab import drive
drive.mount('/content/drive')



# ============================================================================
# Shared publication style (applies to both figures)
# ============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

OUTDIR = "/content/drive/My Drive/paper/XRL_Experiments"


# Shared color palette across both figures, for visual consistency
COLOR_FB     = "#1f5fb4"   # blue   -- primary metric line
COLOR_FS     = "#0a9396"   # teal   -- secondary fidelity line
COLOR_REWARD = "#d9622b"   # orange -- performance / PTG line
COLOR_GOOD   = "#2e8b57"   # green  -- "solved" / functional-competence zone
COLOR_BAD    = "#b03a2e"   # muted red -- underfitting / failure zone
COLOR_GREY   = "#6b6b6b"   # grey   -- neutral threshold markers


# ============================================================================
# FIGURE 1: Pareto Frontier (main manuscript, Figure 1)
# ============================================================================
def make_pareto_figure():
    # ---- Data from Table I (tab:results) ----
    depth        = np.array([2, 4, 6, 8, 10, 12, 14])
    F_b          = np.array([5.0, 10.5, 11.8, 35.4, 39.1, 42.1, 44.2])       # %
    F_s          = np.array([0.89, 0.91, 0.93, 0.95, 0.96, 0.97, 0.97]) * 100  # -> %
    reward_mean  = np.array([-670.0, -614.3, -623.8, 9.1, -10.4, 50.9, 103.9])
    reward_std   = np.array([57.5, 71.1, 62.4, 183.8, 185.9, 191.9, 144.1])

    ppo_reward_mean = 215.9

    fig, ax1 = plt.subplots(figsize=(3.45, 2.75), dpi=300)
    ax2 = ax1.twinx()

    # --- Reward axis (right), drawn first so it sits behind fidelity lines ---
    reward_ylim = (-750, 260)
    ax2.set_ylim(*reward_ylim)

    # Functional competence zone (reward > 100)
    ax2.axhspan(100, reward_ylim[1], color=COLOR_GOOD, alpha=0.10, zorder=0,
                label="Functional competence zone (R > 100)")

    # PPO teacher baseline
    ax2.axhline(ppo_reward_mean, color=COLOR_GOOD, linestyle=":", linewidth=1.3,
                zorder=1, label="PPO teacher baseline")

    # Reward line with CI shading
    ax2.fill_between(depth, reward_mean - reward_std, reward_mean + reward_std,
                      color=COLOR_REWARD, alpha=0.15, linewidth=0, zorder=2)
    line_reward, = ax2.plot(depth, reward_mean, color=COLOR_REWARD, marker="D",
                             markersize=3.5, linewidth=1.4, zorder=4,
                             label="Reward (bootstrap CI, 100 ep.)")

    ax2.set_ylabel("Episode Reward", color=COLOR_REWARD)
    ax2.tick_params(axis="y", colors=COLOR_REWARD)
    ax2.spines["right"].set_color(COLOR_REWARD)

    # --- Fidelity axis (left) ---
    line_fb, = ax1.plot(depth, F_b, color=COLOR_FB, marker="o", markersize=3.5,
                         linewidth=1.4, zorder=5, label="$F_b$ (Behavioral Fidelity)")
    line_fs, = ax1.plot(depth, F_s, color=COLOR_FS, marker="s", markersize=3.5,
                         linewidth=1.4, linestyle="--", zorder=5,
                         label="$F_s \\times 100$ (Structural Fidelity)")

    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Fidelity (\\%)")
    ax1.set_xlabel("Tree Depth ($d$)")
    ax1.set_xticks(depth)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax1.grid(True, axis="y", linewidth=0.4, alpha=0.35, zorder=0)

    # Phase-transition marker at depth 8
    ax1.axvline(8, color=COLOR_GREY, linewidth=0.7, linestyle=(0, (1, 2)), zorder=1)
    ax1.annotate("phase\ntransition", xy=(8, 102), ha="center", va="bottom",
                 fontsize=6.5, color=COLOR_GREY)

    # --- Legend (single combined, placed below plot) ---
    handles = [line_fb, line_fs, line_reward,
               plt.Line2D([0], [0], color=COLOR_GOOD, linestyle=":", linewidth=1.3),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_GOOD, alpha=0.10)]
    labels = ["$F_b$", "$F_s \\times 100$", "Reward (bootstrap CI, 100 ep.)",
              "PPO teacher baseline", "Functional competence zone"]
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=6.3,
               bbox_to_anchor=(0.55, -0.08), columnspacing=1.0, handlelength=1.6)

    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/pareto-frontier.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/pareto-frontier.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved pareto-frontier.png / .pdf")


# ============================================================================
# FIGURE 2: Empirical Validation of Information-Theoretic Bounds
# (Supplementary Material, Appendix A, Figure 1)
# ============================================================================
def make_information_capacity_figure():
    # ---- Data from Table I's PTG (%) column, re-expressed against ----
    # ---- Information Capacity H(C) = tree depth d (Proposition 1) ----
    H_C = np.array([2, 4, 6, 8, 10, 12, 14])
    PTG = np.array([410.2, 384.5, 388.9, 95.8, 104.8, 76.4, 51.9])

    H_MIN = 8  # empirical phase-transition threshold (Section V.A)

    fig, ax = plt.subplots(figsize=(3.45, 2.75), dpi=300)

    xlim = (1, 15)
    ylim = (0, 460)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Background regions: underfitting (H(C) < H_min) vs. functional
    # competence (H(C) >= H_min)
    ax.axvspan(xlim[0], H_MIN, color=COLOR_BAD, alpha=0.08, zorder=0,
               label="Underfitting region")
    ax.axvspan(H_MIN, xlim[1], color=COLOR_GOOD, alpha=0.08, zorder=0,
               label="Functional competence region")

    # Phase-transition threshold
    ax.axvline(H_MIN, color=COLOR_GOOD, linewidth=1.3, linestyle="--", zorder=2,
               label=f"Phase transition ($H_{{min}} \\approx {H_MIN}$ bits)")

    # Task-solved threshold (PTG < 100%)
    ax.axhline(100, color=COLOR_REWARD, linewidth=1.0, linestyle=":", zorder=2,
               label="Task solved (PTG $<$ 100\\%)")

    # Main PTG curve
    line_ptg, = ax.plot(H_C, PTG, color=COLOR_FB, marker="o", markersize=4,
                         linewidth=1.6, zorder=5, label="Observed PTG")

    ax.set_xlabel("Information Capacity $H(C)$ (bits)")
    ax.set_ylabel("Performance-Transparency Gap (\\%)")
    ax.set_xticks(H_C)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.35, zorder=0)

    # Annotations
    ax.annotate("Catastrophic\nfailure", xy=(6, 388.9), xytext=(4.3, 430),
                fontsize=6.5, color=COLOR_BAD, ha="center",
                arrowprops=dict(arrowstyle="->", color=COLOR_BAD, linewidth=0.8))
    ax.annotate("Functional\ncompetence", xy=(14, 51.9), xytext=(11.3, 130),
                fontsize=6.5, color=COLOR_GOOD, ha="center",
                arrowprops=dict(arrowstyle="->", color=COLOR_GOOD, linewidth=0.8))

    # --- Legend (combined, placed below plot) ---
    handles = [line_ptg,
               plt.Line2D([0], [0], color=COLOR_GOOD, linestyle="--", linewidth=1.3),
               plt.Line2D([0], [0], color=COLOR_REWARD, linestyle=":", linewidth=1.0),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_BAD, alpha=0.08),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_GOOD, alpha=0.08)]
    labels = ["Observed PTG", f"Phase transition ($H_{{min}} \\approx {H_MIN}$ bits)",
              "Task solved (PTG $<$ 100\\%)", "Underfitting region",
              "Functional competence region"]
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=6.0,
               bbox_to_anchor=(0.55, -0.14), columnspacing=1.0, handlelength=1.6)

    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/information_capacity_plot.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUTDIR}/information_capacity_plot.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved information_capacity_plot.png / .pdf")


if __name__ == "__main__":
    make_pareto_figure()
    make_information_capacity_figure()
    print("done")