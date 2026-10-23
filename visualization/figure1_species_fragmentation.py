"""Plot Figure 1b-d from species-level fragmentation summary CSV files.

The changes directory contains AMPH_changes.csv, BIRD_changes.csv,
MA_changes.csv and RE_changes.csv, with the fraction columns 增加比例,
下降比例 and 不变比例. The summary CSV contains group, species_id, status,
mean_1985_common, mean_2020_common and mean_change; only status='ok' is used.
Each species has equal weight. Absolute mean changes <= 1e-6 are unchanged.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


GROUPS = ["AMPH", "BIRD", "MA", "RE"]
NAMES = ["Amphibians", "Birds", "Mammals", "Reptiles"]
FIELDS = ["增加比例", "下降比例", "不变比例"]
LABELS = ["Increase", "Decrease", "Unchanged"]
COLOURS = ["#E74C3C", "#3498DB", "#FAAC61"]
EPS = 1e-6
STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.5, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "legend.frameon": False,
}


def require_columns(frame, columns, path):
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def load_pixel_proportions(changes_dir):
    """Average within-species pixel fractions, without pooling pixel counts."""
    rows = []
    for group, name in zip(GROUPS, NAMES):
        path = changes_dir / f"{group}_changes.csv"
        frame = pd.read_csv(path)
        require_columns(frame, FIELDS, path)
        values = frame[FIELDS].apply(pd.to_numeric, errors="raise").dropna()
        if values.empty:
            raise ValueError(f"{path}: no complete species proportions")
        array = values.to_numpy()
        if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ValueError(f"{path}: proportions must be finite fractions in [0, 1]")
        if not np.allclose(array.sum(axis=1), 1, atol=1e-6, rtol=0):
            raise ValueError(f"{path}: each species' three proportions must sum to 1")
        means = values.mean().to_numpy()
        rows.append({"group": group, "name": name, "valid_species": len(values),
                     "increase_fraction": means[0], "decrease_fraction": means[1],
                     "unchanged_fraction": means[2]})
    return pd.DataFrame(rows)


def load_species_summary(path):
    frame = pd.read_csv(path, dtype={"species_id": str})
    numeric = ["mean_1985_common", "mean_2020_common", "mean_change"]
    require_columns(frame, ["group", "species_id", "status", *numeric], path)
    valid = frame.loc[frame["status"].eq("ok") & frame["group"].isin(GROUPS)].copy()
    if valid["species_id"].isna().any() or valid.duplicated(["group", "species_id"]).any():
        raise ValueError(f"{path}: valid species IDs must be present and unique within each group")
    valid[numeric] = valid[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(valid[numeric].to_numpy()).all():
        raise ValueError(f"{path}: valid species means must be finite")
    missing = set(GROUPS).difference(valid["group"])
    if missing:
        raise ValueError(f"{path}: no valid species for {sorted(missing)}")
    return valid


def direction_summary(valid):
    rows = []
    for group, name in zip(GROUPS, NAMES):
        changes = valid.loc[valid["group"].eq(group), "mean_change"].to_numpy()
        counts = [int((changes > EPS).sum()), int((changes < -EPS).sum()),
                  int((np.abs(changes) <= EPS).sum())]
        for label, colour, count in zip(LABELS, COLOURS, counts):
            rows.append({"group": group, "name": name, "category": label,
                         "count": count, "valid_species": len(changes),
                         "fraction": count / len(changes), "percent": count / len(changes) * 100,
                         "colour": colour, "alpha": .8})
    return pd.DataFrame(rows)


def yearly_summary(valid):
    summary = valid.groupby("group").agg(
        species_count=("species_id", "size"), mean_1985=("mean_1985_common", "mean"),
        mean_2020=("mean_2020_common", "mean"), mean_change=("mean_change", "mean"),
    ).loc[GROUPS].copy()
    if not np.allclose(summary["mean_2020"] - summary["mean_1985"],
                       summary["mean_change"], atol=1e-12):
        raise ValueError("Yearly means and mean_change are inconsistent")
    return summary


def save_figure(fig, output_dir, stem):
    for extension in ("svg", "pdf"):
        fig.savefig(output_dir / f"{stem}.{extension}")
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.tiff", dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def draw_pixel_proportions(data, output_dir, labeled):
    values = data[["increase_fraction", "decrease_fraction", "unchanged_fraction"]].to_numpy()
    fig, ax = plt.subplots(figsize=(112 / 25.4, 112 / 25.4))
    if labeled:
        fig.subplots_adjust(left=.17, right=.975, bottom=.18, top=.81)
    else:
        fig.subplots_adjust(left=.055, right=.985, bottom=.055, top=.985)
    x = np.arange(4)
    bottom = np.zeros(4)
    for col, (colour, label) in enumerate(zip(COLOURS, LABELS)):
        heights = values[:, col]
        ax.bar(x, heights, width=.4, bottom=bottom, color=colour, alpha=.8,
               edgecolor="none", label=label)
        if labeled:
            for xx, lo, height in zip(x, bottom, heights):
                ax.text(xx, lo + height / 2, f"{height * 100:.2f}%", ha="center",
                        va="center", fontsize=7, color="#26323D")
        bottom += heights
    ax.set_xlim(-.5, 3.5)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(axis="y", length=3 if labeled else 8, width=.8 if labeled else 2)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(.8 if labeled else 2)
    if labeled:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1, decimals=0))
        ax.set_xticks(x, NAMES)
        ax.tick_params(axis="x", length=0, pad=7)
        ax.set_ylabel("Mean within-species pixel proportion (%)", labelpad=5)
        fig.text(.17, .94, "Mean pixel-change proportions by group", fontsize=8.5, weight="bold")
        fig.legend(loc="upper center", bbox_to_anchor=(.58, .91), ncol=3,
                   fontsize=7, columnspacing=1.4, handlelength=1.3)
    else:
        ax.set_xticks([])
        ax.set_yticklabels([])
    save_figure(fig, output_dir, "figure1b_pixel_proportions" + ("" if labeled else "_clean"))


def draw_species_pies(data, output_dir, labeled):
    pie_order = [2, 1, 0]
    fig, axes = plt.subplots(2, 2, figsize=(160 / 25.4, 160 / 25.4))
    fig.subplots_adjust(left=.04, right=.96, bottom=.12 if labeled else .04,
                        top=.91 if labeled else .96, wspace=.19,
                        hspace=.30 if labeled else .12)
    for ax, group, name in zip(axes.flat, GROUPS, NAMES):
        rows = data.loc[data["group"].eq(group)].set_index("category").loc[LABELS]
        fractions = rows["fraction"].to_numpy()[pie_order]
        colours = [to_rgba(COLOURS[i], .8) for i in pie_order]
        wedges, _ = ax.pie(fractions, colors=colours, explode=[.035] * 3,
                           startangle=90, counterclock=False, radius=.95,
                           wedgeprops={"edgecolor": "white", "linewidth": 1.7})
        for wedge, fraction in zip(wedges, fractions):
            if not labeled:
                continue
            angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
            dx, dy = np.cos(angle), np.sin(angle)
            cx, cy = wedge.center
            label = f"{fraction * 100:.1f}%"
            if fraction >= .05:
                ax.text(cx + .59 * dx, cy + .59 * dy, label,
                        ha="center", va="center", fontsize=9, color="black")
            else:
                ax.annotate(label, xy=(cx + .91 * dx, cy + .91 * dy),
                            xytext=(.52, 1.035), ha="left", va="center", fontsize=9,
                            color="black", arrowprops={"arrowstyle": "-", "color": "black",
                                                       "lw": .65, "shrinkA": 2, "shrinkB": 1})
        if labeled:
            ax.set_title(f'{name}\n$n$ = {int(rows.iloc[0]["valid_species"]):,}',
                         fontsize=10, pad=7, linespacing=1.55)
        ax.set_xlim(-1.17, 1.24)
        ax.set_ylim(-1.13, 1.17)
        ax.set_aspect("equal")
        ax.set_axis_off()
    if labeled:
        handles = [Patch(facecolor=to_rgba(c, .8), edgecolor="none", label=label)
                   for c, label in zip(COLOURS, LABELS)]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035),
                   ncol=3, fontsize=9, columnspacing=1.8, handlelength=1.3)
    save_figure(fig, output_dir, "figure1c_species_directions" + ("" if labeled else "_clean"))


def draw_yearly_means(summary, output_dir, limits):
    """Save the clean paired-point panel for manuscript assembly."""
    if limits[0] >= limits[1]:
        raise ValueError("--mean-limits must be increasing")
    values = summary[["mean_1985", "mean_2020"]].to_numpy()
    if np.any((values < limits[0]) | (values > limits[1])):
        raise ValueError("Yearly means fall outside --mean-limits; widen the plotting range")
    fig, ax = plt.subplots(figsize=(112 / 25.4, 90 / 25.4))
    fig.subplots_adjust(left=.075, right=.97, bottom=.075, top=.97)
    for y, (_, row) in zip([3., 2., 1., 0.], summary.iterrows()):
        old, new = float(row["mean_1985"]), float(row["mean_2020"])
        ax.hlines(y, min(old, new), max(old, new), color="black", lw=1.2, zorder=2)
        for value, colour in [(old, "#8DABAF"), (new, "#DA7765")]:
            ax.scatter([value], [y], s=150, facecolor=colour, edgecolor=colour,
                       linewidth=1., zorder=4)
    ax.set_xlim(*limits)
    ax.set_ylim(-.5, 3.5)
    ax.set_xticks(np.linspace(*limits, 6))
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(2)
        ax.spines[side].set_color("black")
    ax.tick_params(axis="both", length=8, width=2, color="black",
                   labelbottom=False, labelleft=False)
    ax.grid(False)
    save_figure(fig, output_dir, "figure1d_yearly_means_clean")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changes-dir", type=Path, default=Path("data/visualization/figure1"))
    parser.add_argument("--summary-csv", type=Path,
                        default=Path("data/visualization/figure1/all_species_mean_changes.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("D:/hzl/code/outputs/figures/figure1"))
    parser.add_argument("--mean-limits", type=float, nargs=2, default=(.13, .18),
                        metavar=("LOWER", "UPPER"), help="Horizontal axis range of panel d")
    args = parser.parse_args()
    pixels = load_pixel_proportions(args.changes_dir)
    species = load_species_summary(args.summary_csv)
    directions = direction_summary(species)
    years = yearly_summary(species)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for labeled in (True, False):
            draw_pixel_proportions(pixels, args.output_dir, labeled)
            draw_species_pies(directions, args.output_dir, labeled)
        draw_yearly_means(years, args.output_dir, args.mean_limits)
    pixels.to_csv(args.output_dir / "figure1b_source_data.csv", index=False, encoding="utf-8-sig")
    directions.to_csv(args.output_dir / "figure1c_source_data.csv", index=False, encoding="utf-8-sig")
    years.to_csv(args.output_dir / "figure1d_source_data.csv", encoding="utf-8-sig")
    print(f"Figure 1b-d and source summaries saved to {args.output_dir}")


if __name__ == "__main__":
    main()
