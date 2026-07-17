#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from plot_common import save_figure


FIGURE_NAME = "Fig_same_cross_stage_locality"

DATA = [
    {"stage": "ground", "same_stage_overlap": 0.0609944751381215, "cross_stage_overlap": 0.0470811744386873},
    {"stage": "plan", "same_stage_overlap": 0.1615469613259668, "cross_stage_overlap": 0.0415544041450777},
    {"stage": "reason", "same_stage_overlap": 0.0512707182320441, "cross_stage_overlap": 0.0445250431778929},
    {"stage": "reflect", "same_stage_overlap": 0.0380110497237569, "cross_stage_overlap": 0.0414853195164076},
]


# =========================
# Font config (方案 A：全局统一)
# =========================
def configure_font() -> str:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    script_dir = Path(__file__).resolve().parent
    font_path = script_dir / "fonts" / "Arial.ttf"

    if not font_path.exists():
        raise FileNotFoundError(f"Font not found: {font_path}")

    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()

    plt.rcParams.update({
        "font.family": font_name,
        "font.size": 16,

        # 保证 PDF/PS 输出一致性
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        # 防止负号乱码
        "axes.unicode_minus": False,

        # 关键：统一粗细策略（避免 title/label 视觉不一致）
        "axes.titleweight": "regular",
        "axes.labelweight": "regular",
    })

    return font_name


def main() -> None:
    import matplotlib.pyplot as plt

    font_name = configure_font()
    print(f"[INFO] Using font: {font_name}")

    x = list(range(len(DATA)))

    fig, ax = plt.subplots(figsize=(7, 5))

    # =========================
    # bars
    # =========================
    ax.bar(
        [i - 0.2 for i in x],
        [row["same_stage_overlap"] for row in DATA],
        width=0.4,
        label="same-stage",
        color="#A8A8A8",
        edgecolor="black",
        linewidth=0.9
    )

    ax.bar(
        [i + 0.2 for i in x],
        [row["cross_stage_overlap"] for row in DATA],
        width=0.4,
        label="cross-stage",
        facecolor="none",
        edgecolor="black",
        linewidth=0.9,
        hatch="///"
    )

    # =========================
    # axis (统一字体后不再需要 fontproperties)
    # =========================
    ax.set_xticks(x)
    ax.set_xticklabels([row["stage"] for row in DATA])

    ax.set_ylabel("Overlap", fontsize=24)
    ax.set_title("Same vs Cross Stage Locality", fontsize=22)

    # =========================
    # ticks
    # =========================
    ax.set_yticks([
        0.00,
        0.05,
        0.10,
        0.14,
        0.16,
        0.17
    ])

    ax.tick_params(axis="x", labelsize=22, width=1.1, length=4)
    ax.tick_params(axis="y", labelsize=16, width=1.1, length=4)

    # =========================
    # grid (paper style)
    # =========================
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # =========================
    # legend（统一字体后不会再变粗）
    # =========================
    ax.legend(fontsize=18, frameon=True)

    fig.tight_layout()

    save_figure(fig, FIGURE_NAME)
    plt.close(fig)

    print(f"[INFO] wrote {FIGURE_NAME}.pdf")


if __name__ == "__main__":
    main()