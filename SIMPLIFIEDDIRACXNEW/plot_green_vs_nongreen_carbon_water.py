from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = {
    "Green": {"carbon_kg": 17.58051481, "water_stress_l": 2.89563681},
    "Non-green": {"carbon_kg": 18.85414930, "water_stress_l": 3.87732560},
}


def main() -> None:
    output = Path(__file__).resolve().parent / "plots" / "green_vs_nongreen_carbon_water.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    green = RUNS["Green"]
    non_green = RUNS["Non-green"]
    carbon_reduction = 100.0 * (non_green["carbon_kg"] - green["carbon_kg"]) / non_green["carbon_kg"]
    water_reduction = (
        100.0
        * (non_green["water_stress_l"] - green["water_stress_l"])
        / non_green["water_stress_l"]
    )

    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor("#f7f8fa")
    ax.set_facecolor("#ffffff")

    ax.scatter(
        non_green["carbon_kg"],
        non_green["water_stress_l"],
        s=230,
        color="#d95f02",
        edgecolor="white",
        linewidth=2,
        zorder=3,
    )
    ax.scatter(
        green["carbon_kg"],
        green["water_stress_l"],
        s=260,
        color="#17823b",
        edgecolor="white",
        linewidth=2,
        zorder=4,
    )

    ax.annotate(
        "",
        xy=(green["carbon_kg"], green["water_stress_l"]),
        xytext=(non_green["carbon_kg"], non_green["water_stress_l"]),
        arrowprops={"arrowstyle": "->", "color": "#52606d", "lw": 2, "linestyle": "--"},
        zorder=2,
    )
    ax.annotate(
        f"Green\n{green['carbon_kg']:.2f} kgCO$_2$\n{green['water_stress_l']:.2f} stress-L",
        (green["carbon_kg"], green["water_stress_l"]),
        xytext=(12, -12),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#126b31",
    )
    ax.annotate(
        f"Non-green\n{non_green['carbon_kg']:.2f} kgCO$_2$\n{non_green['water_stress_l']:.2f} stress-L",
        (non_green["carbon_kg"], non_green["water_stress_l"]),
        xytext=(-12, -12),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#a44700",
    )

    ax.text(
        0.5,
        0.95,
        f"Green reduction: {carbon_reduction:.2f}% carbon  |  {water_reduction:.2f}% water",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        color="#25313c",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#eaf5ed", "edgecolor": "#9bc9a6"},
    )
    ax.set_title("Green vs Non-green: Total Carbon and Water Impact", fontsize=16, fontweight="bold", pad=16)
    ax.set_xlabel("Total carbon footprint (kgCO$_2$)", fontsize=12)
    ax.set_ylabel("Total water impact (stress-L)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.22, y=0.28)

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
