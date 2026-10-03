#!/usr/bin/env python3
"""Create the publication diagram for DIRAC GreenSiteDirector placement."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


HERE = Path(__file__).resolve().parent
GEOJSON = HERE / "countries.geojson"
GEOJSON_URL = (
    "https://raw.githubusercontent.com/datasets/geo-countries/"
    "master/data/countries.geojson"
)

COUNTRIES = {
    "Belgium", "Denmark", "France", "Germany", "Ireland", "Italy",
    "Luxembourg", "Netherlands", "Norway", "Poland", "Portugal",
    "Spain", "Sweden", "Switzerland", "United Kingdom",
}

SITES = {
    "RAL-LCG2\nUnited Kingdom": (-1.31, 51.57),
    "IN2P3-IRES\nFrance": (4.87, 45.78),
    "FZK-LCG2\nGermany": (8.43, 49.10),
    "SARA-MATRIX\nNetherlands": (4.90, 52.37),
}


def iter_rings(geometry):
    coords = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        yield from coords
    elif geometry["type"] == "MultiPolygon":
        for polygon in coords:
            yield from polygon


def main() -> None:
    if not GEOJSON.exists():
        with urlopen(GEOJSON_URL, timeout=30) as response:
            GEOJSON.write_bytes(response.read())

    data = json.loads(GEOJSON.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(12.0, 6.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fbfd")

    for feature in data["features"]:
        props = feature["properties"]
        name = props.get("ADMIN") or props.get("name")
        if name not in COUNTRIES:
            continue
        for ring in iter_rings(feature["geometry"]):
            xs, ys = zip(*ring)
            ax.fill(xs, ys, facecolor="#edf3f6", edgecolor="#a9bfcc",
                    linewidth=0.65, zorder=1)

    navy, teal, green = "#12315b", "#087f83", "#2b9348"

    # Central scheduler, placed above the geographic sites.
    outer = FancyBboxPatch(
        (-0.1, 57.25), 7.1, 3.5,
        boxstyle="round,pad=0.35,rounding_size=0.35",
        facecolor="white", edgecolor=navy, linewidth=2.0, zorder=5,
    )
    inner = FancyBboxPatch(
        (0.55, 57.75), 5.8, 1.35,
        boxstyle="round,pad=0.2,rounding_size=0.25",
        facecolor=teal, edgecolor="none", zorder=6,
    )
    ax.add_patch(outer)
    ax.add_patch(inner)
    ax.text(3.45, 60.1, "DIRAC WMS", ha="center", va="center",
            fontsize=17, fontweight="bold", color=navy, zorder=7)
    ax.text(3.45, 58.42, "GreenSiteDirector", ha="center", va="center",
            fontsize=13.5, fontweight="bold", color="white", zorder=7)

    # Generic ranked preference indicator; it does not encode a fixed ordering.
    ax.text(8.0, 60.15, "Green Score ranking", color=navy,
            fontsize=11.5, fontweight="bold", va="center", zorder=7)
    for i, alpha in enumerate((1.0, 0.78, 0.56, 0.34), start=1):
        y = 59.55 - (i - 1) * 0.55
        ax.text(8.25, y, str(i), ha="center", va="center", color="white",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.28", fc=green,
                          ec="none", alpha=alpha), zorder=7)
        ax.plot([8.8, 10.8 - 0.18 * (i - 1)], [y, y], color=green,
                linewidth=6, alpha=alpha, solid_capstyle="round", zorder=7)

    label_offsets = {
        "RAL-LCG2\nUnited Kingdom": (-2.8, 0.55),
        "IN2P3-IRES\nFrance": (-2.8, -1.15),
        "FZK-LCG2\nGermany": (1.3, -1.2),
        "SARA-MATRIX\nNetherlands": (1.25, 0.75),
    }

    origin = (3.45, 57.72)
    for label, (lon, lat) in SITES.items():
        ax.annotate(
            "", xy=(lon, lat + 0.2), xytext=origin,
            arrowprops=dict(arrowstyle="-|>", color=teal, lw=1.8,
                            mutation_scale=13, shrinkA=4, shrinkB=5),
            zorder=3,
        )
        ax.scatter(lon, lat, s=180, facecolor="white", edgecolor=navy,
                   linewidth=2.0, zorder=5)
        ax.scatter(lon, lat, s=65, marker="s", facecolor=teal,
                   edgecolor="white", linewidth=0.8, zorder=6)
        dx, dy = label_offsets[label]
        ax.annotate(
            label, xy=(lon, lat), xytext=(lon + dx, lat + dy),
            fontsize=9.5, fontweight="bold", color=navy,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=navy,
                      lw=1.3),
            arrowprops=dict(arrowstyle="-", color=navy, lw=0.8), zorder=7,
        )

    ax.text(-7.8, 56.3, "Pilot submission", color=teal, fontsize=9.5,
            fontweight="bold", rotation=12, zorder=4)
    ax.text(10.4, 55.8, "Pilot submission", color=teal, fontsize=9.5,
            fontweight="bold", rotation=-13, zorder=4)

    ax.set_xlim(-12.5, 15.5)
    ax.set_ylim(42.8, 61.4)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(HERE / f"dirac_greensitedirector_europe.{suffix}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
