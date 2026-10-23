"""Export Figure 4 regional GLMM coefficient panels from the R results TXT.

Each region produces a labeled SVG and an SVG without text for layout.
Bars are shown for p <= 0.1; all six predictors retain their supplied 95% CIs.
"""

import argparse
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "D:/hzl/code/outputs/GLMM_ALL_COMPOSITE/GLMM_ALL_COMPOSITE_results.txt"
)
DEFAULT_OUTPUT_DIR = Path("D:/hzl/code/outputs/figures/figure4")
VARIABLES = (
    "Fire frequency",
    "Drought frequency",
    "Temperature change",
    "Precipitation change",
    "Population change",
    "Nighttime light change",
)
COLORS = ("#ee3b20", "#ee3b20", "#3282bd", "#3282bd", "#f4b040", "#f4b040")
NUMERIC_COLUMNS = ("Coefficient", "Std_Error", "p_value", "CI_lower", "CI_upper")
HEADER = re.compile(
    r"区域:\s*(.*?)\s*\|\s*因变量:\s*(.*?)\s*\|\s*n=(\d+)"
)


def _decode_header_escapes(text):
    """Read Chinese headers escaped by R under a non-Chinese locale."""
    return re.sub(
        r"<U\+([0-9a-fA-F]{4,6})>",
        lambda match: chr(int(match.group(1), 16)),
        text,
    )


def parse_txt_results(filepath):
    """Read and validate one complete six-predictor table per region/response."""
    results = []
    current = None
    rows = []
    seen = set()

    def finish_region():
        if current is None:
            return
        label = f"{current['region']} / {current['response']}"
        row_names = [row["Variable"] for row in rows]
        missing = sorted(set(VARIABLES) - set(row_names))
        duplicates = sorted({name for name in row_names if row_names.count(name) > 1})
        if missing or duplicates or len(rows) != len(VARIABLES):
            raise ValueError(
                f"Incomplete or duplicate predictor table for {label}: "
                f"missing={missing}, duplicates={duplicates}. "
                "All five numeric columns must be on the same line per predictor."
            )
        frame = pd.DataFrame(rows).set_index("Variable").loc[list(VARIABLES)].reset_index()
        values = frame[list(NUMERIC_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite coefficient results for {label}.")
        if (frame["Std_Error"] < 0).any() or not frame["p_value"].between(0, 1).all():
            raise ValueError(f"Invalid standard errors or p-values for {label}.")
        if (
            (frame["CI_lower"] > frame["Coefficient"]).any()
            or (frame["CI_upper"] < frame["Coefficient"]).any()
        ):
            raise ValueError(f"Confidence interval does not contain its estimate for {label}.")
        key = (current["region"], current["response"])
        if key in seen:
            raise ValueError(f"Duplicate region/response table: {label}.")
        seen.add(key)
        results.append(dict(current, df=frame))

    text = _decode_header_escapes(Path(filepath).read_text(encoding="utf-8-sig"))
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        match = HEADER.search(line)
        if match:
            finish_region()
            current = {
                "region": match.group(1).strip(),
                "response": match.group(2).strip(),
                "n_total": int(match.group(3)),
            }
            if not current["region"] or not current["response"] or current["n_total"] <= 0:
                raise ValueError(f"Invalid region header on line {line_number}.")
            rows = []
            continue

        for variable in VARIABLES:
            if not line.startswith(variable):
                continue
            if current is None:
                raise ValueError(f"Predictor table has no region header on line {line_number}.")
            tokens = line[len(variable):].split()
            if len(tokens) != len(NUMERIC_COLUMNS):
                raise ValueError(
                    f"Expected five numeric fields for {variable} on line {line_number}; "
                    "export the R table with a wide print width if it is wrapped."
                )
            try:
                values = [float(token) for token in tokens]
            except ValueError as exc:
                raise ValueError(f"Invalid numeric field on line {line_number}.") from exc
            rows.append(dict(zip(NUMERIC_COLUMNS, values), Variable=variable))
            break
    finish_region()
    if not results:
        raise ValueError(f"No GLMM region tables found in {filepath}.")
    return results


def auto_axis(ci_low_arr, ci_high_arr):
    data_range = ci_high_arr.max() - ci_low_arr.min()
    step = 0.05 if data_range < 0.15 else (0.1 if data_range < 0.35 else 0.15)
    # Keep the bar origin visible even when every coefficient has the same sign.
    x_min = np.floor(min(0.0, ci_low_arr.min()) / step) * step - step * 0.3
    x_max = np.ceil(max(0.0, ci_high_arr.max()) / step) * step + step * 0.3
    ticks = np.arange(
        np.ceil(x_min / step) * step,
        np.floor(x_max / step) * step + step * 0.5,
        step,
    )
    return x_min, x_max, np.round(ticks, 10)


def _filename_label(label):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._") or "region"


def render_region(region_info, show_text=True):
    """Return one Matplotlib figure without changing the supplied statistics."""
    import matplotlib.pyplot as plt

    frame = region_info["df"]
    x_min, x_max, ticks = auto_axis(frame["CI_lower"].values, frame["CI_upper"].values)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if show_text:
        fig.suptitle(
            f"GLMM Coefficients — {region_info['response']} | {region_info['region']}",
            fontsize=13,
        )
    for i, row in frame.iterrows():
        coefficient = row["Coefficient"]
        ci_low, ci_high = row["CI_lower"], row["CI_upper"]
        if row["p_value"] <= 0.1:
            ax.barh(
                i, abs(coefficient), height=0.4, left=min(0.0, coefficient),
                color=COLORS[i], alpha=0.8, edgecolor="white", linewidth=1,
            )
        ax.plot([ci_low, ci_high], [i, i], color="black", linewidth=2, alpha=0.8)
        for endpoint in (ci_low, ci_high):
            ax.plot([endpoint, endpoint], [i - 0.05, i + 0.05],
                    color="black", linewidth=2, alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(ticks)
    ax.set_yticks(np.arange(len(VARIABLES)))
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    if show_text:
        ax.set_title(f"n={region_info['n_total']}", fontsize=11)
        ax.set_yticklabels(VARIABLES, fontsize=10)
        ax.set_xticklabels([f"{tick:.2f}" for tick in ticks], fontsize=9)
    else:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    ax.tick_params(axis="both", which="major", length=4, direction="out", color="black")
    fig.tight_layout()
    return fig


def export_panels(results, output_dir):
    output_dir = Path(output_dir)
    labels = [
        f"GLMM_{_filename_label(item['response'])}_{_filename_label(item['region'])}"
        for item in results
    ]
    if len({label.casefold() for label in labels}) != len(labels):
        raise ValueError("Region/response labels produce duplicate output filenames.")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep Matplotlib's generated cache beside the requested figure output.
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        for item, label in zip(results, labels):
            for show_text in (True, False):
                suffix = "with_text" if show_text else "no_text"
                path = output_dir / f"{label}_{suffix}.svg"
                fig = render_region(item, show_text=show_text)
                try:
                    fig.savefig(path, format="svg", dpi=300, bbox_inches="tight")
                finally:
                    plt.close(fig)
                paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="GLMM results TXT produced by the R analysis.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for the labeled and unlabeled regional SVGs.")
    args = parser.parse_args(argv)
    try:
        results = parse_txt_results(args.input)
        paths = export_panels(results, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved {len(paths)} SVG panels for {len(results)} regions to {args.output_dir}")


if __name__ == "__main__":
    main()
