"""Calculate six habitat metrics from prepared species land-cover rasters.

Inputs must already be restricted to species ranges, elevation limits and suitable
land-cover classes. This script does not repeat that preprocessing. Supply
single-band, north-up, 300 m square-cell rasters in a projected CRS with metre
units. All retained classes become habitat (1) in memory; the input NoData value
represents background, which is included in the window area denominator. No
separate binary raster is written. This script does not reproject input data.

The formulas follow the local wz_all_metrics.py workflow. Non-overlapping 10 x 10
windows start at each input's upper-left corner, without global 3 km alignment.
Partial edge windows use their actual cell count but retain a 3 km output cell.
Undefined metrics, including CLUMPY and nLSI for a fully occupied 10 x 10 window,
are written as -9999, preserving the local calculation.

Run all tasks and years, or select a task and optionally a year:
    python fragmentation/species_habitat_indices.py
    python fragmentation/species_habitat_indices.py AMPH 2020
"""

import argparse
import os
import json
import multiprocessing
from datetime import datetime
from pathlib import Path

import rasterio
import numpy as np
import geopandas as gpd
from affine import Affine
from scipy import ndimage

N_WORKERS = 16
PIXEL_SIZE = 300
PIXEL_AREA_HA = PIXEL_SIZE * PIXEL_SIZE / 10000  # 9 ha
STRUCT8 = np.ones((3, 3), dtype=int)
WINDOW_SIZE = 3000
FILL = -9999.0
YEARS = ["1985", "2020"]
METRICS = ["CLUMPY", "nLSI", "ED", "MESH", "AREA_MN", "PD"]

# Resolve inputs relative to this repository, regardless of the working directory.
# Edit these settings for another data installation; outputs default to drive D.
REPO_ROOT = Path(__file__).resolve().parents[1]
GDB_FILE = REPO_ROOT / "data" / "sp_data.gdb"
BASE_LC_DIR = REPO_ROOT / "data" / "species_habitat"
BASE_OUT_DIR = Path("D:/hzl/code/outputs/six_habitat_indices")
LC_REF = {
    year: REPO_ROOT / "data" / "land_cover" / f"land_cover_{year}_300m.tif"
    for year in YEARS
}

# (task ID, GDB layer, input folders by year, output group prefix).
# Split bird/reptile layers share their group's folders.
TASKS = [
    ("AMPH", "AMPHIBIANS_population_trend_final_2025_05_02",
     {year: BASE_LC_DIR / f"AMPH_{year}_kc" for year in YEARS}, "AMPH"),
    ("BIRD_A", "Birds_population_trend_final_2025_03_27A_2000",
     {year: BASE_LC_DIR / f"BIRD_{year}_kc" for year in YEARS}, "BIRD"),
    ("BIRD_B", "Birds_population_trend_final_2025_03_27B_4000",
     {year: BASE_LC_DIR / f"BIRD_{year}_kc" for year in YEARS}, "BIRD"),
    ("BIRD_C", "Birds_population_trend_final_2025_03_27C_6000",
     {year: BASE_LC_DIR / f"BIRD_{year}_kc" for year in YEARS}, "BIRD"),
    ("BIRD_D", "Birds_population_trend_final_2025_03_27D_8000",
     {year: BASE_LC_DIR / f"BIRD_{year}_kc" for year in YEARS}, "BIRD"),
    ("MA", "MAMMALS_population_trend_final_2025_03_23",
     {year: BASE_LC_DIR / f"MA_{year}_kc" for year in YEARS}, "MA"),
    ("RE_A", "REPTILES_population_trend_final_2025_03_23_A",
     {year: BASE_LC_DIR / f"RE_{year}_kc" for year in YEARS}, "RE"),
    ("RE_B", "REPTILES_population_trend_final_2025_03_23_B",
     {year: BASE_LC_DIR / f"RE_{year}_kc" for year in YEARS}, "RE"),
]


def load_progress(path):
    """Load restart information; output files are checked independently."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as stream:
                progress = json.load(stream)
            progress["completed"] = [str(fid) for fid in progress["completed"]]
            progress["failed"] = [str(fid) for fid in progress["failed"]]
            return progress
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Could not load progress from {path}: {exc}")
    return {"completed": [], "failed": [], "last_update": None}


def save_progress(path, progress):
    progress["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(progress, stream, indent=2, ensure_ascii=False)


def _validate_nodata(nodata):
    """Reject sentinels incompatible with the equality-based binary habitat mask."""
    if nodata is None or not np.isscalar(nodata) or not np.isfinite(nodata):
        raise ValueError("NoData must be a finite numeric value, not NaN or infinity.")
    if nodata == 1:
        raise ValueError("NoData=1 conflicts with binary habitat=1; use another sentinel.")


# Habitat patch areas in hectares, using eight-neighbour connectivity.
def _patch_areas_ha(data, nodata):
    habitat = (data != nodata)
    labeled, n = ndimage.label(habitat, structure=STRUCT8)
    if n == 0:
        return np.array([], dtype=np.float64)
    sizes = ndimage.sum(habitat, labeled, range(1, n + 1))
    return np.asarray(sizes, dtype=np.float64) * PIXEL_AREA_HA


# -------------------------------------------------------
# -------------------------------------------------------
def _same_adj_and_mine(data, nodata):
    n_1 = int(np.sum(data != nodata))
    a_h, b_h = data[:, :-1], data[:, 1:]
    a_v, b_v = data[:-1, :], data[1:, :]
    same_h = int(np.sum((a_h != nodata) & (b_h != nodata)))
    same_v = int(np.sum((a_v != nodata) & (b_v != nodata)))
    G_11 = 2 * (same_h + same_v)
    m = int(np.floor(np.sqrt(n_1)))
    if n_1 == m * m:
        min_e = 4 * m
    elif n_1 <= m * (m + 1):
        min_e = 4 * m + 2
    else:
        min_e = 4 * (m + 1)
    return G_11, min_e, n_1


# -------------------------------------------------------
# -------------------------------------------------------
def calc_all(data, nodata):
    result = {m: np.nan for m in METRICS}
    n_1 = int(np.sum(data != nodata))
    if n_1 == 0:
        return result

    n_total  = data.size
    p_i      = n_1 / n_total
    total_ha = n_total * PIXEL_AREA_HA

    # ---------- CLUMPY ----------
    G_11, min_e, _ = _same_adj_and_mine(data, nodata)
    denom = 4 * n_1 - min_e
    if denom != 0:
        g_i = G_11 / denom
        if g_i >= p_i or p_i >= 0.5:
            if (1.0 - p_i) != 0:
                result["CLUMPY"] = float((g_i - p_i) / (1.0 - p_i))
        else:
            if p_i != 0:
                result["CLUMPY"] = float((g_i - p_i) / (-p_i))

    # ---------- nLSI ----------
    e_i = 4 * n_1 - G_11
    rows, cols = data.shape
    cells_boundary = rows * 2 + cols * 2
    threshold = (0.5 * n_total + 0.5 * cells_boundary) / n_total
    if p_i <= 0.5:
        max_e = 4 * n_1
    elif n_total % 2 == 0 and p_i <= threshold:
        max_e = 3 * n_total - 2 * n_1
    elif n_total % 2 != 0 and p_i <= threshold:
        max_e = 3 * n_total - 2 * n_1 + 3
    else:
        max_e = e_i + 4 * (n_total - n_1)
    nlsi_denom = max_e - min_e
    if nlsi_denom != 0:
        v = (e_i - min_e) / nlsi_denom
        if np.isfinite(v):
            result["nLSI"] = float(max(0.0, min(1.0, v)))

    # ED excludes the window perimeter; nLSI above includes it.
    # ---------- ED ----------
    habitat = (data != nodata)
    h_edges = int(np.sum(habitat[:, :-1] != habitat[:, 1:]))
    v_edges = int(np.sum(habitat[:-1, :] != habitat[1:, :]))
    result["ED"] = float((h_edges + v_edges) * PIXEL_SIZE / total_ha)

    # ---------- MESH / AREA_MN / PD ----------
    areas = _patch_areas_ha(data, nodata)
    if len(areas) > 0:
        result["MESH"]    = float(np.sum(areas ** 2) / total_ha)
        result["AREA_MN"] = float(np.mean(areas))
        result["PD"]      = float(len(areas) / total_ha * 100)

    return result


# -------------------------------------------------------
# -------------------------------------------------------
def create_metric_rasters(landcover_path, out_paths, lc_nodata=0):
    """Binarize one prepared input in memory and write six metric rasters."""
    with rasterio.open(landcover_path) as src:
        if src.count != 1:
            raise ValueError("Species habitat inputs must be single-band rasters.")
        raster    = src.read(1)
        nodata_val = src.nodata if src.nodata is not None else lc_nodata
        _validate_nodata(nodata_val)
        transform  = src.transform
        profile    = src.profile.copy()

    rows, cols = raster.shape
    nw = int(WINDOW_SIZE / PIXEL_SIZE)
    out_rows = (rows + nw - 1) // nw
    out_cols = (cols + nw - 1) // nw

    binary = np.where(raster != nodata_val, 1, nodata_val).astype(raster.dtype)

    arrays = {m: np.full((out_rows, out_cols), FILL, dtype=np.float32) for m in METRICS}

    for i in range(0, rows, nw):
        for j in range(0, cols, nw):
            w = binary[i:i + nw, j:j + nw]
            if w.size == 0:
                continue
            oi, oj = i // nw, j // nw
            if int(np.sum(w != nodata_val)) < 1:
                continue
            res = calc_all(w, nodata_val)
            for m, v in res.items():
                if not np.isnan(v):
                    arrays[m][oi, oj] = v

    out_profile = profile.copy()
    out_profile.update({
        'dtype': rasterio.float32,
        'width': out_cols,
        'height': out_rows,
        'transform': transform * Affine.scale(nw, nw),
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
        'nodata': FILL,
        'count': 1,
    })

    for m, path in out_paths.items():
        with rasterio.open(path, 'w', **out_profile) as dst:
            dst.write(arrays[m], 1)



def _outputs_complete(paths):
    """A completed species must have all six nonempty output files."""
    return len(paths) == len(METRICS) and all(
        os.path.isfile(path) and os.path.getsize(path) > 0 for path in paths
    )


def process_feature(args):
    feature_id, lc_folder, out_folder, lc_nodata = args
    landcover_path = os.path.join(lc_folder, f"{feature_id}_landcover.tif")
    if not os.path.exists(landcover_path):
        return feature_id, "no_landcover"
    out_paths = {m: os.path.join(out_folder, f"{feature_id}_{m}.tif") for m in METRICS}
    try:
        create_metric_rasters(landcover_path, out_paths, lc_nodata)
        ok = _outputs_complete(list(out_paths.values()))
        return feature_id, "success" if ok else "empty_output"
    except Exception as exc:
        for path in out_paths.values():
            if os.path.exists(path):
                os.remove(path)
        return feature_id, f"error: {exc}"


def run_task(task_id, layer_name, lc_folders, out_prefix, year, lc_nodata):
    """Run one GDB layer/year, rechecking files for previously completed IDs."""
    lc_folder = lc_folders[year]
    out_folder = os.path.join(BASE_OUT_DIR, f"{out_prefix}_{year}")
    gdf = gpd.read_file(GDB_FILE, layer=layer_name)
    if "id_no" not in gdf.columns or gdf["id_no"].isna().any():
        raise ValueError(f"{layer_name} must contain a non-missing id_no for every species.")
    feature_ids = list(dict.fromkeys(str(fid) for fid in gdf["id_no"]))
    os.makedirs(out_folder, exist_ok=True)
    progress_file = os.path.join(out_folder, f"progress_{task_id}_{year}.json")
    progress = load_progress(progress_file)
    completed = set(progress["completed"])
    failed = set(progress["failed"])
    todo = []
    for feature_id in feature_ids:
        paths = [os.path.join(out_folder, f"{feature_id}_{m}.tif") for m in METRICS]
        if _outputs_complete(paths):
            completed.add(feature_id)
            failed.discard(feature_id)
        else:
            completed.discard(feature_id)
            todo.append(feature_id)
    progress["completed"] = sorted(completed)
    progress["failed"] = sorted(failed)
    save_progress(progress_file, progress)
    print(f"Task {task_id}, year {year}: {len(feature_ids)} species, {len(todo)} pending")
    print(f"Input: {lc_folder}")
    print(f"Output: {out_folder}")
    if not todo:
        return 0
    args_list = [(fid, lc_folder, out_folder, lc_nodata) for fid in todo]
    failure_count = 0
    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        for feature_id, status in pool.imap_unordered(process_feature, args_list):
            if status == "success":
                completed.add(feature_id)
                failed.discard(feature_id)
            else:
                completed.discard(feature_id)
                failed.add(feature_id)
                failure_count += 1
            print(f"  {feature_id}: {status}")
            progress["completed"] = sorted(completed)
            progress["failed"] = sorted(failed)
            save_progress(progress_file, progress)
    return failure_count


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("task", nargs="?", choices=[task[0] for task in TASKS],
                        help="Task to run; omit to run all configured animal groups.")
    parser.add_argument("year", nargs="?", choices=YEARS,
                        help="Year to run; omit to run both 1985 and 2020.")
    args = parser.parse_args(argv)
    failures = 0
    for year in YEARS:
        if args.year and year != args.year:
            continue
        # Optional reference supplies fallback NoData only when an input lacks it.
        lc_nodata = 0
        try:
            with rasterio.open(LC_REF[year]) as src:
                lc_nodata = src.nodata if src.nodata is not None else 0
        except rasterio.errors.RasterioIOError:
            print(f"Reference unavailable: {LC_REF[year]}; fallback NoData is 0.")
        for task_id, layer_name, lc_folders, out_prefix in TASKS:
            if args.task and task_id != args.task:
                continue
            failures += run_task(task_id, layer_name, lc_folders, out_prefix, year, lc_nodata)
    print(f"Finished. Failed or missing species inputs: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
