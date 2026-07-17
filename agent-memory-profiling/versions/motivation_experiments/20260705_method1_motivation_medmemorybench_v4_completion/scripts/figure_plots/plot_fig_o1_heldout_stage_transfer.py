#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

from plot_common import save_figure

FIGURE_NAME = "Fig_heldout_stage_transfer"

DATA = [
    {
        "stage": "ground",
        "stage_advantage_mean": 0.075879410388406,
        "ci95_low": 0.0379836272216124,
        "ci95_high": 0.1193721808408519,
    },
    {
        "stage": "plan",
        "stage_advantage_mean": 0.3699615656009573,
        "ci95_low": 0.234190331800693,
        "ci95_high": 0.5209805039211418,
    },
    {
        "stage": "reason",
        "stage_advantage_mean": 0.0357938400037705,
        "ci95_low": -0.0025431287974175,
        "ci95_high": 0.0736272642180722,
    },
    {
        "stage": "reflect",
        "stage_advantage_mean": 0.0752291548211425,
        "ci95_low": 0.0367577979320255,
        "ci95_high": 0.1154981135621589,
    },
    {
        "stage": "aggregate",
        "stage_advantage_mean": 0.1392159927035691,
        "ci95_low": 0.1011941180732279,
        "ci95_high": 0.181556342284289,
    },
]


def configure_font() -> str:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    script_dir = Path(__file__).resolve().parent
    font_path = script_dir / "fonts" / "Arial.ttf"

    if not font_path.exists():
        raise FileNotFoundError(f"Font not found: {font_path}")

    font_manager.fontManager.addfont(str(font_path))

    font_properties = font_manager.FontProperties(fname=str(font_path))
    font_name = font_properties.get_name()

    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "hatch.linewidth": 1.0,
        }
    )

    return font_name


def main() -> None:
    import matplotlib.pyplot as plt

    font_name = configure_font()
    print(f"Using font: {font_name}")

    stages = [row["stage"] for row in DATA]
    means = [row["stage_advantage_mean"] for row in DATA]

    err_low = [
        row["stage_advantage_mean"] - row["ci95_low"]
        for row in DATA
    ]
    err_high = [
        row["ci95_high"] - row["stage_advantage_mean"]
        for row in DATA
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        stages,
        means,
        width=0.65,
        yerr=[err_low, err_high],
        capsize=5,
        color="#B0B0B0",
        edgecolor="black",
        linewidth=0.9,
        error_kw={
            "elinewidth": 1.2,
            "capthick": 1.2,
            "ecolor": "black",
        },
    )

    # stage styling
    for bar in bars[:4]:
        bar.set_facecolor("#A8A8A8")
        bar.set_edgecolor("black")
        bar.set_hatch(None)

    # aggregate bar highlight
    bars[-1].set_facecolor("none")   # 关键：去掉背景色
    bars[-1].set_edgecolor("black")
    bars[-1].set_hatch("///")
    # zero baseline
    ax.axhline(0, color="black", linewidth=1.0, zorder=0)

    # ====== 핵心修正（你要的图形优化）======

    # y-axis 更紧凑（关键）
    ax.set_ylim(0.0, 0.55)

    # grid 提升论文可读性
    ax.grid(axis='y', linestyle='--', alpha=0.35)

    # x-axis 规范
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages)
    plt.xticks(rotation=0, ha='center')

    # labels
    ax.set_ylabel("Held-out Stage Advantage", fontsize=22)
    ax.set_title("Held-Out Stage Transfer", fontsize=22)

    # ticks styling
    ax.tick_params(axis="x", labelsize=20, width=1.1, length=4)
    ax.tick_params(axis="y", labelsize=16, width=1.1, length=4)

    # # bottom spacing (避免挤压)
    # fig.subplots_adjust(bottom=0.18)

    # layout
    fig.tight_layout()

    save_figure(fig, FIGURE_NAME)
    plt.close(fig)

    print(f"wrote {FIGURE_NAME}.pdf")


if __name__ == "__main__":
    main()