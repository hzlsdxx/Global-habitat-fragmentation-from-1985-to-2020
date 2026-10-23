"""Calculate species-level composite fragmentation from four habitat metrics.

Inputs: {species_id}_{CLUMPY,nLSI,ED,PD}.tif in group/year subdirectories.
Output: {species_id}_MIX.tif beside each input set, or under --output-dir with
the same directory structure. AREA_MN and MESH are not composite components.

Each metric has one shared min/max pair computed from all valid input pixels
under --input-dir. Include all comparison groups and years in that directory.
No empirical normalization bounds are hard-coded or printed.

For x clipped to [min, max], normalization is (x - min) / (max - min),
except CLUMPY, which uses (max - x) / (max - min). MIX is the equal-weight
mean only where all four normalized metrics are valid. Larger MIX indicates
greater fragmentation. Each group/year is processed independently; this script
does not pair species between years or calculate temporal differences.
"""

import argparse
import multiprocessing
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio

METRICS = ("CLUMPY", "nLSI", "ED", "PD")
INVERT = {"CLUMPY": True, "nLSI": False, "ED": False, "PD": False}
FILL = -9999.0
INPUT_FOLDER = Path("D:/hzl/code/outputs/six_habitat_indices")
MAX_WORKERS = max(1, min(16, multiprocessing.cpu_count() - 1))
FILE_PATTERN = re.compile(r"^(.+)_(CLUMPY|nLSI|ED|PD)\.tiff?$", re.IGNORECASE)
METRIC_NAMES = {metric.upper(): metric for metric in METRICS}


def scan_and_group(folder):
    """Group files by (relative parent directory, species ID), without collisions."""
    root = Path(folder).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    groups = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        species_id, raw_metric = match.groups()
        metric = METRIC_NAMES[raw_metric.upper()]
        key = (str(path.parent.relative_to(root)), species_id)
        files = groups.setdefault(key, {})
        if metric in files:
            raise ValueError(f"Duplicate {metric} raster for {key}: {path}")
        files[metric] = path
    return groups


def find_global_bounds(groups):
    """Compute per-metric min/max over finite, unmasked pixels in all input sets."""
    bounds = {metric: {"min": np.inf, "max": -np.inf} for metric in METRICS}
    for files in groups.values():
        for metric, path in files.items():
            with rasterio.open(path) as src:
                if src.count != 1:
                    raise ValueError(f"Expected a single-band metric raster: {path}")
                for _, window in src.block_windows(1):
                    values = src.read(1, window=window, masked=True).astype("float32")
                    values = values.compressed()
                    values = values[np.isfinite(values)]
                    if values.size:
                        bounds[metric]["min"] = min(bounds[metric]["min"], float(values.min()))
                        bounds[metric]["max"] = max(bounds[metric]["max"], float(values.max()))
    for metric, limits in bounds.items():
        metric_min, metric_max = limits["min"], limits["max"]
        if not np.isfinite(metric_min) or not np.isfinite(metric_max):
            raise ValueError(f"{metric}: no finite valid input values for min/max.")
        if metric_max <= metric_min:
            raise ValueError(f"{metric}: normalization requires max > min.")
    return bounds


def read_and_norm(path, vmin, vmax, invert=False):
    """Mask NoData, round to four decimals, clip and normalize; valid zero is kept."""
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("Normalization requires finite min/max and max > min.")
    with rasterio.open(path) as src:
        if src.count != 1:
            raise ValueError(f"Expected a single-band metric raster: {path}")
        data = src.read(1, masked=True).astype("float32")
        array = data.filled(np.nan)
        profile = src.profile.copy()
    array[~np.isfinite(array)] = np.nan
    array = np.clip(np.round(array, 4), vmin, vmax)
    if invert:
        array = (vmax - array) / (vmax - vmin)
    else:
        array = (array - vmin) / (vmax - vmin)
    return array, profile


def composite_from_stack(stack):
    """Return an equal-weight mean where all four layers are finite."""
    if stack.ndim != 3 or stack.shape[0] != len(METRICS):
        raise ValueError("MIX requires exactly four metric layers.")
    valid = np.all(np.isfinite(stack), axis=0)
    result = np.full(stack.shape[1:], np.nan, dtype="float32")
    result[valid] = np.mean(stack[:, valid], axis=0)
    return result


def _check_grid(reference, profile, path):
    """Do not combine equally sized rasters from different spatial grids."""
    for key in ("width", "height", "transform", "crs"):
        if profile[key] != reference[key]:
            raise ValueError(f"Metric grids differ in {key}: {path}")


def _write_mix(array, path, profile):
    """Write atomically so a failed write cannot leave a partial MIX raster."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output_profile = profile.copy()
    output_profile.update(driver="GTiff", dtype="float32", count=1, nodata=FILL,
                          compress="lzw", tiled=True, blockxsize=512, blockysize=512)
    data = np.where(np.isfinite(array), array, FILL).astype("float32")
    with tempfile.NamedTemporaryFile(prefix=".mix-", suffix=".tif", dir=path.parent,
                                     delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with rasterio.open(temporary_path, "w", **output_profile) as dst:
            dst.write(data, 1)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def process_single_group(args):
    """Normalize a species/year's four metric rasters and write its MIX."""
    group_key, files, bounds, out_path = args
    try:
        arrays = []
        reference = None
        for metric in METRICS:
            limits = bounds[metric]
            array, profile = read_and_norm(files[metric], limits["min"], limits["max"],
                                           invert=INVERT[metric])
            if reference is None:
                reference = profile
            else:
                _check_grid(reference, profile, files[metric])
            arrays.append(array)
        mix = composite_from_stack(np.stack(arrays, axis=0))
        _write_mix(mix, out_path, reference)
        return True, str(out_path), None
    except Exception as exc:
        return False, str(out_path), f"{group_key}: {exc}"


def batch_merge_norm(in_folder, out_folder=None, max_workers=MAX_WORKERS):
    """Scan once for shared bounds, then write one four-metric MIX per input set."""
    if max_workers < 1:
        raise ValueError("Worker count must be at least one.")
    groups = scan_and_group(in_folder)
    if not groups:
        raise RuntimeError("No CLUMPY/nLSI/ED/PD input rasters found.")
    incomplete = []
    for (relative_dir, species_id), files in groups.items():
        missing = [metric for metric in METRICS if metric not in files]
        if missing:
            incomplete.append(f"{relative_dir}/{species_id}: missing {', '.join(missing)}")
    if incomplete:
        raise ValueError("Incomplete four-metric input sets; no outputs written:\n" +
                         "\n".join(incomplete))

    bounds = find_global_bounds(groups)
    print(f"Shared min/max calculated for {len(METRICS)} metrics across {len(groups)} input sets.")
    tasks = []
    output_targets = set()
    for group_key, files in groups.items():
        relative_dir, species_id = group_key
        parent = Path(out_folder).resolve() / relative_dir if out_folder else files["ED"].parent
        out_path = parent / f"{species_id}_MIX.tif"
        target = os.path.normcase(str(out_path.resolve()))
        if target in output_targets:
            raise ValueError(f"Multiple input sets would overwrite the same MIX output: {out_path}")
        output_targets.add(target)
        tasks.append((group_key, files, bounds, out_path))

    outputs, failures = [], []

    def collect(results):
        for success, out_path, error in results:
            if success:
                outputs.append(out_path)
                print(f"Written: {out_path}")
            else:
                failures.append(error)
                print(f"Failed: {error}")

    if max_workers == 1:
        collect(map(process_single_group, tasks))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            collect(executor.map(process_single_group, tasks))
    if failures:
        raise RuntimeError(f"{len(failures)} input sets failed; exclude their existing MIX files "
                           "from downstream analysis until corrected:\n" + "\n".join(failures))
    print(f"Completed: {len(outputs)} species/year MIX rasters.")
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=INPUT_FOLDER,
                        help="Root containing metric rasters for all comparison years/groups.")
    parser.add_argument("--output-dir", type=Path,
                        help="Optional output root; preserves subdirectories. Default: beside inputs.")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help="Number of parallel species processes.")
    args = parser.parse_args(argv)
    try:
        batch_merge_norm(args.input_dir, args.output_dir, args.workers)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
