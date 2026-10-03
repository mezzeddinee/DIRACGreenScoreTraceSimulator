from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "four_run_results.csv"
OUTPUT_STEM = BASE / "five_run_environmental_comparison"

STYLE = {
    "Green-hydric": {"color": "#009E73", "marker": "o", "offset": (9, -9), "align": "left"},
    "Random": {"color": "#E69F00", "marker": "o", "offset": (9, 9), "align": "left"},
    "SARA-MATRIX": {"color": "#0072B2", "marker": "D", "offset": (9, 9), "align": "left"},
    "TR-03-METU": {"color": "#D55E00", "marker": "D", "offset": (-9, -9), "align": "right"},
    "SARA + TR-03": {"color": "#CC79A7", "marker": "^", "offset": (-9, -9), "align": "right"},
}


def load_rows() -> list[dict[str, str]]:
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = load_rows()
    by_name = {row["scenario"]: row for row in rows}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(9.4, 7.4))
    grid = fig.add_gridspec(2, 1, height_ratios=[5.2, 1.25], hspace=0.18)
    ax = fig.add_subplot(grid[0])
    table_ax = fig.add_subplot(grid[1])
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for row in rows:
        name = row["scenario"]
        carbon = float(row["total_carbon_kgco2"])
        water = float(row["total_water_stress_l"])
        makespan = int(row["makespan_min"])
        style = STYLE[name]

        ax.scatter(
            carbon,
            water,
            s=150,
            marker=style["marker"],
            color=style["color"],
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
        )
        ax.annotate(
            name,
            (carbon, water),
            xytext=style["offset"],
            textcoords="offset points",
            ha=style["align"],
            va="top" if style["offset"][1] < 0 else "bottom",
            color=style["color"],
            fontsize=9.5,
            fontweight="semibold",
            linespacing=1.25,
        )

    green = by_name["Green-hydric"]
    random = by_name["Random"]
    ax.annotate(
        "",
        xy=(float(green["total_carbon_kgco2"]), float(green["total_water_stress_l"])),
        xytext=(float(random["total_carbon_kgco2"]), float(random["total_water_stress_l"])),
        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#59636e", "linestyle": "--"},
        zorder=2,
    )
    ax.text(
        0.035,
        0.955,
        "Preferred direction  ↙\nlower carbon · lower water",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="#36424d",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f4f6f7", "edgecolor": "#c8cdd2"},
    )

    ax.set_yscale("log")
    ax.set_xlim(13, 75)
    ax.set_ylim(0.4, 9000)
    ax.set_xlabel("Total carbon footprint (kgCO$_2$)")
    ax.set_ylabel("Total water-scarcity impact (stress-L, log scale)")
    ax.grid(True, which="major", color="#d6dadd", linestyle="-", linewidth=0.7, alpha=0.8)
    ax.grid(True, which="minor", axis="y", color="#e8eaec", linestyle=":", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#6f7b85",
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=8,
            label="Multi-site policy (4 × 750 slots)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="#6f7b85",
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=7,
            label="Single-site extreme (750 slots)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#6f7b85",
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=8,
            label="Two-site run (2 × 750 slots)",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.96, edgecolor="#c8cdd2")

    table_ax.axis("off")
    columns = ["Configuration", "Run type", "Carbon (kgCO$_2$)", "Water (stress-L)", "Makespan (min)"]
    table_rows = []
    for row in rows:
        water = float(row["total_water_stress_l"])
        water_text = f"{water:,.2f}" if water >= 100.0 else f"{water:.3f}"
        table_rows.append(
            [
                row["scenario"],
                row["run_type"],
                f"{float(row['total_carbon_kgco2']):.2f}",
                water_text,
                f"{int(row['makespan_min']):,}",
            ]
        )
    result_table = table_ax.table(
        cellText=table_rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.12, 1.0, 0.82],
        colWidths=[0.20, 0.20, 0.19, 0.22, 0.19],
    )
    result_table.auto_set_font_size(False)
    result_table.set_fontsize(8.8)
    for (row_index, column_index), cell in result_table.get_celld().items():
        cell.set_edgecolor("#c8cdd2")
        cell.set_linewidth(0.65)
        if row_index == 0:
            cell.set_facecolor("#35424d")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f7f8f9" if row_index % 2 else "white")
            if column_index == 0:
                scenario = rows[row_index - 1]["scenario"]
                cell.get_text().set_color(STYLE[scenario]["color"])
                cell.get_text().set_weight("bold")

    fig.suptitle("Carbon–water trade-offs across five configurations", y=0.975, fontsize=15, weight="bold")
    fig.text(
        0.5,
        0.018,
        "Same 133,631-job trace; water metric is AWARE-weighted scarcity impact. Lower-left is preferable.\n"
        "Green/Random sites: SARA-MATRIX, IN2P3-IRES, FZK-LCG2, RAL-LCG2.",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#53606b",
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.92, bottom=0.065)

    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=320)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"))
    plt.close(fig)

    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".pdf"))
    print(OUTPUT_STEM.with_suffix(".svg"))


if __name__ == "__main__":
    main()
