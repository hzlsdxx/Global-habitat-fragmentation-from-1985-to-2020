"""Figure 2c–e: threshold donuts, habitat-loss GAMs and metric correlations.

Reads prepared rasters/CSV tables; it does not collect raw species pixels.
GAM loss_rate = (habitat_1985 - habitat_2020) / habitat_1985:
positive values mean habitat loss. Fragmentation changes are 2020 minus 1985.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = ["0-25%", "25-50%", "50-75%", "75-100%"]
GAM_COLORS = ["#999DCB", "#80b4d5", "#f57f5a", "#e16c66"]
METRICS = ["CLUMPY", "nLSI", "ED", "PD"]
PLOT_ORDER = ["PD", "ED", "CLUMPY", "nLSI"]
METRIC_COLORS = {"PD": "#EB9794", "ED": "#FCD5AE", "CLUMPY": "#B9D6E7", "nLSI": "#999DCB"}


def read_numeric_csv(path, columns):
    data = pd.read_csv(path)
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
    data = data.loc[:, columns].apply(pd.to_numeric, errors="raise")
    if data.empty or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Input must contain finite numeric observations in every required column.")
    return data


def save_figure(fig, output_dir, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / (name + "." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_donuts(paths, output_dir):
    """Count positive, zero and negative valid pixels in four threshold rasters."""
    import rasterio

    percentages = []
    for path in paths:
        with rasterio.open(path) as src:
            values = src.read(1, masked=True).astype(np.float32).compressed()
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("No valid raster pixels: " + str(path))
        percentages.append(np.array([(values > 0).sum(), (values == 0).sum(),
                                     (values < 0).sum()]) / values.size * 100)
    labels = ["0%", "1%", "5%", "10%"]
    pd.DataFrame(percentages, columns=["increase_pct", "no_change_pct", "decrease_pct"],
                 index=pd.Index(labels, name="threshold")).to_csv(output_dir / "figure2c_percentages.csv")
    colors = ["#e16c66", "#ebecec", "#8cafd3"]
    for show_text in (True, False):
        fig, axes = plt.subplots(2, 2, figsize=(6, 6), facecolor="white")
        for ax, values, label in zip(axes.flat, percentages, labels):
            ax.pie(values, colors=colors, startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.35, edgecolor="none"),
                   autopct="%.2f%%" if show_text else None, pctdistance=1.17)
            if show_text:
                ax.text(0, 0, label, ha="center", va="center")
            ax.axis("equal")
        if show_text:
            fig.legend(["Increase", "No change", "Decrease"], loc="upper center", ncol=3, frameon=False)
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        save_figure(fig, output_dir, "figure2c_threshold_donuts_" + ("with_text" if show_text else "no_text"))


def draw_gam(csv_path, output_dir):
    """Fit the existing LinearGAM(s(0)) separately in four initial-amount bins."""
    from pygam import LinearGAM, s

    data = read_numeric_csv(csv_path, ["amount", "loss_rate", "frag_diff"])
    amount, loss, frag = (data[c].to_numpy() for c in data.columns)
    groups = np.full(len(amount), -1, dtype=int)
    groups[(amount > 0) & (amount <= .25)] = 0
    groups[(amount > .25) & (amount <= .50)] = 1
    groups[(amount > .50) & (amount <= .75)] = 2
    # Preserve the source workflow's exclusion of fully occupied habitat (amount=1).
    groups[(amount > .75) & (amount < 1)] = 3
    results, rows = [], []
    for qi in range(4):
        mask = groups == qi
        x, y = loss[mask], frag[mask]
        if len(x) < 2 or np.unique(x).size < 2:
            raise ValueError("Insufficient variation for GAM in " + LABELS[qi])
        gam = LinearGAM(s(0)).fit(x.reshape(-1, 1), y)
        x_pred = np.linspace(x.min(), x.max(), 300)
        y_pred = gam.predict(x_pred.reshape(-1, 1))
        ci = gam.confidence_intervals(x_pred.reshape(-1, 1), width=.95)
        if not np.isfinite(y_pred).all() or not np.isfinite(ci).all():
            raise ValueError("Non-finite GAM output in " + LABELS[qi])
        results.append((qi, x_pred, y_pred, ci))
        rows.extend(dict(group=LABELS[qi], loss_rate=float(a), predicted_fragmentation_change=float(b),
                         ci95_lower=float(lo), ci95_upper=float(hi))
                    for a, b, lo, hi in zip(x_pred, y_pred, ci[:, 0], ci[:, 1]))
    pd.DataFrame(rows).to_csv(output_dir / "figure2d_gam_curves.csv", index=False)
    for show_text in (True, False):
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
        for qi, x, y, ci in results:
            ax.fill_between(x, ci[:, 0], ci[:, 1], alpha=.15, color=GAM_COLORS[qi], edgecolor="none")
            ax.plot(x, ci[:, 0], color=GAM_COLORS[qi], linewidth=1.5)
            ax.plot(x, ci[:, 1], color=GAM_COLORS[qi], linewidth=1.5)
            ax.plot(x, y, color=GAM_COLORS[qi], linewidth=3.5, label=LABELS[qi])
        ax.axhline(0, color="gray", linestyle="--", linewidth=.8)
        ax.axvline(0, color="gray", linestyle="--", linewidth=.8)
        ax.tick_params(direction="in", length=5, labelbottom=show_text, labelleft=show_text)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if show_text:
            ax.set_xlabel("Habitat loss rate (1985–2020)")
            ax.set_ylabel("Fragmentation change (MIX diff)")
            ax.legend(title="Habitat amount (1985)", fontsize=9, frameon=False)
        fig.tight_layout()
        save_figure(fig, output_dir, "figure2d_habitat_gam_" + ("with_text" if show_text else "no_text"))


def draw_spearman(csv_path, output_dir):
    """Plot absolute Spearman r; the accompanying table retains signed r and p."""
    from scipy.stats import spearmanr

    data = read_numeric_csv(csv_path, ["amount", "mix_diff"] + [m + "_diff" for m in METRICS])
    amount = data["amount"].to_numpy()
    corr = {metric: [] for metric in METRICS}
    rows = []
    for i, (lo, hi) in enumerate(zip([0, .25, .50, .75], [.25, .50, .75, 1])):
        mask = (amount > lo) & (amount <= hi)
        for metric in METRICS:
            if mask.sum() > 10:
                r, p = spearmanr(data.loc[mask, metric + "_diff"], data.loc[mask, "mix_diff"])
            else:
                r, p = np.nan, np.nan
            corr[metric].append(r)
            rows.append(dict(bin=LABELS[i], metric=metric, n=int(mask.sum()), r=r, p=p))
    pd.DataFrame(rows).to_csv(output_dir / "figure2e_spearman.csv", index=False)
    for show_text in (True, False):
        fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor="white")
        x, width = np.arange(4), .18
        for j, metric in enumerate(PLOT_ORDER):
            ax.bar(x + (j - 1.5) * width, np.abs(corr[metric]), width,
                   color=METRIC_COLORS[metric], label=metric, edgecolor="white", linewidth=.5)
        ax.set_xticks(x)
        ax.set_ylim(0, 1)
        ax.tick_params(direction="in", length=5, labelbottom=show_text, labelleft=show_text)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if show_text:
            ax.set_xticklabels(LABELS)
            ax.set_xlabel("Habitat amount (1985)")
            ax.set_ylabel("|Spearman r| with MIX change")
            ax.legend(fontsize=9, frameon=False, loc="lower center",
                      bbox_to_anchor=(.5, 1.01), ncol=4)
        fig.tight_layout()
        save_figure(fig, output_dir, "figure2e_metric_spearman_" + ("with_text" if show_text else "no_text"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=["all", "donuts", "gam", "spearman"], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data/visualization/figure2"))
    parser.add_argument("--threshold-rasters", type=Path, nargs=4, metavar=("T0", "T1", "T5", "T10"))
    parser.add_argument("--gam-csv", type=Path, help="CSV with amount, loss_rate, frag_diff")
    parser.add_argument("--metrics-csv", type=Path, help="CSV with amount, mix_diff and four metric differences")
    parser.add_argument("--output-dir", type=Path, default=Path("D:/hzl/code/outputs/figures/figure2"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "pdf.fonttype": 42})
    if args.panel in ("all", "donuts"):
        paths = args.threshold_rasters or [
            args.data_dir / "composite_index_0pct/ALL_composite_diff_all.tif",
            args.data_dir / "composite_index_1pct/ALL_composite_diff_1_all.tif",
            args.data_dir / "composite_index_5pct/ALL_composite_diff_5_all.tif",
            args.data_dir / "composite_index_10pct/ALL_composite_diff_10_all.tif",
        ]
        draw_donuts(paths, args.output_dir)
    if args.panel in ("all", "gam"):
        draw_gam(args.gam_csv or args.data_dir / "hab_gam_pixels.csv", args.output_dir)
    if args.panel in ("all", "spearman"):
        draw_spearman(args.metrics_csv or args.data_dir / "hab_metric_pixels.csv", args.output_dir)
    print("Figure panels saved to", args.output_dir)


if __name__ == "__main__":
    main()
