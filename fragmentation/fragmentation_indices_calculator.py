# 1. Find global min and max values for normalization
import os
import glob
import rasterio
import numpy as np
import concurrent.futures
import multiprocessing

def process_file(file_info):
    fp, metric = file_info
    try:
        with rasterio.open(fp) as src:
            arr = src.read(1, masked=True)
        if arr.count() == 0:
            return None
        return (metric, float(arr.min()), float(arr.max()))
    except Exception as e:
        print(f"File processing error: {fp}, error: {e}")
        return None

def main():
    folder_path = "data/landscape_indices"
    pattern = os.path.join(folder_path, '**', '*.tif')
    files = glob.glob(pattern, recursive=True)

    metrics = ['PD', 'ED', 'AREA', 'MESH']
    valid_files = []
    for fp in files:
        for metric in metrics:
            if metric in os.path.basename(fp):
                valid_files.append((fp, metric))
                break

    results = {metric: {'min': np.inf, 'max': -np.inf} for metric in metrics}

    cpu_count = multiprocessing.cpu_count()
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count - 1) as executor:
        for result in executor.map(process_file, valid_files):
            if result is None:
                continue
            metric, file_min, file_max = result
            results[metric]['min'] = min(results[metric]['min'], file_min)
            results[metric]['max'] = max(results[metric]['max'], file_max)

    for metric in metrics:
        min_val = results[metric]['min']
        max_val = results[metric]['max']
        if min_val == np.inf or max_val == -np.inf:
            print(f"{metric}: No valid data found")
        else:
            print(f"{metric} - Global minimum: {min_val:.4f}, Global maximum: {max_val:.4f}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()


# 2. Calculate composite indices
import os
import re
import numpy as np
import rasterio
import concurrent.futures
import multiprocessing
from datetime import datetime


def read_and_norm(path, vmin, vmax, invert=False):
    """
    1) Read as MaskedArray, automatically mask src.nodata;
    2) Mark all data==0 as mask;
    3) Set mask regions to np.nan;
    4) Keep 4 decimal places first (consistent with first code);
    5) Clip to [vmin, vmax];
    6) Forward/inverse normalization.
    """
    with rasterio.open(path) as src:
        # 1. Read as MaskedArray, automatically remove src.nodata
        data = src.read(1, masked=True).astype('float32')

        # 3. Fill all masks with NaN
        arr = data.filled(np.nan)

        # 4. Keep 4 decimal places first (key modification: consistent with first code)
        arr = np.round(arr, 4)

        # 5. Clip to specified range
        arr = np.clip(arr, vmin, vmax)

        # 6. Normalization: forward or inverse
        if invert:
            return (vmax - arr) / (vmax - vmin)
        else:
            return (arr - vmin) / (vmax - vmin)


def scan_and_group(folder):
    """Recursively scan folder, support nested folder structure"""
    pattern = re.compile(r'(?i)^(.+?)_(ED|PD|AREA_MN|MESH)\.tiff?$', re.IGNORECASE)
    groups = {}

    # Recursively traverse all subfolders
    for root, dirs, files in os.walk(folder):
        for fn in files:
            if not fn.lower().endswith(('.tif', '.tiff')):
                continue
            m = pattern.match(fn)
            if not m:
                continue

            grp, typ = m.group(1), m.group(2).upper()
            # Use full path as file path, group name relative to subfolder
            full_path = os.path.join(root, fn)

            # Create unique group identifier (including subfolder info)
            rel_path = os.path.relpath(root, folder)
            if rel_path == '.':
                group_key = grp
            else:
                group_key = f"{rel_path.replace(os.sep, '_')}_{grp}"

            groups.setdefault(group_key, {})[typ] = full_path

    return {g: files for g, files in groups.items()
            if set(files.keys()) == {'ED', 'PD', 'AREA_MN', 'MESH'}}


def process_single_group(args):
    """Function to process a single group, for parallel processing"""
    grp, files, ed_min, ed_max, pd_min, pd_max, area_mn_min, area_mn_max, mesh_min, mesh_max = args

    try:
        print(f"[Process {os.getpid()}] Processing group: {grp}")
        arrs = []
        profile = None

        for typ, (vmin, vmax) in zip(
                ['ED', 'PD', 'AREA_MN', 'MESH'],
                [(ed_min, ed_max),
                 (pd_min, pd_max),
                 (area_mn_min, area_mn_max),
                 (mesh_min, mesh_max)]
        ):
            path = files[typ]  # Now files[typ] is already the full path
            invert = typ in ('AREA_MN', 'MESH')
            normed = read_and_norm(path, vmin, vmax, invert=invert)
            arrs.append(normed)

            if profile is None:
                with rasterio.open(path) as src:
                    profile = src.profile.copy()

        # Average (NaN does not participate)
        mean_arr = np.nanmean(np.stack(arrs, axis=0), axis=0)
        # Replace NaN with my_nodata
        my_nodata = -9999
        mean_arr = np.where(np.isnan(mean_arr), my_nodata, mean_arr).astype('float32')

        # Output filename: ED → MIX
        ed_fn = os.path.basename(files['ED'])
        mix_fn = re.sub(r'(?i)ED', 'MIX', ed_fn, count=1)

        # Get original file's subfolder
        original_dir = os.path.dirname(files['ED'])
        # Ensure output subfolder exists
        os.makedirs(original_dir, exist_ok=True)
        # Output to corresponding subfolder
        out_path = os.path.join(original_dir, mix_fn)

        # Update profile and write out
        profile.update(dtype=rasterio.float32, count=1, nodata=my_nodata)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(mean_arr, 1)

        print(f"[Process {os.getpid()}] ✅ Completed: {ed_fn} → {mix_fn} (saved in {original_dir})")
        return True, grp, None

    except Exception as e:
        error_msg = f"Error processing group {grp}: {str(e)}"
        print(f"[Process {os.getpid()}] ❌ {error_msg}")
        return False, grp, error_msg


def batch_merge_norm(
        in_folder,
        ed_min, ed_max,
        pd_min, pd_max,
        area_mn_min, area_mn_max,
        mesh_min, mesh_max,
        out_folder=None
):
    groups = scan_and_group(in_folder)
    if not groups:
        raise RuntimeError("No complete ED/PD/AREA_MN/MESH groups found.")
    print(f"Found {len(groups)} groups in total: {', '.join(groups)}")

    # Prepare task list for parallel processing
    tasks = []
    for grp, files in groups.items():
        tasks.append((grp, files, ed_min, ed_max, pd_min, pd_max,
                      area_mn_min, area_mn_max, mesh_min, mesh_max))

    # Get CPU core count, reserve 1 core for system
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Detected {multiprocessing.cpu_count()} CPU cores, will use {max_workers} processes for parallel processing")

    # Record start time
    start_time = datetime.now()
    success_count = 0
    failure_count = 0
    failed_groups = []

    # Parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_single_group, tasks)

        # Count results
        for success, grp, error_msg in results:
            if success:
                success_count += 1
            else:
                failure_count += 1
                failed_groups.append(grp)

    # Calculate processing time
    end_time = datetime.now()
    processing_time = end_time - start_time

    # Output statistics
    print(f"\n=== Processing Complete ===")
    print(f"Processing time: {processing_time}")
    print(f"Total groups: {len(groups)}")
    print(f"Successfully processed: {success_count}")
    print(f"Processing failed: {failure_count}")
    if processing_time.total_seconds() > 0:
        print(f"Average processing speed: {len(tasks) / processing_time.total_seconds():.2f} groups/second")
    if failed_groups:
        print(f"Failed groups: {failed_groups}")
    print("============================")


if __name__ == '__main__':
    # Multiprocessing support for Windows environment
    multiprocessing.freeze_support()

    INPUT_FOLDER = "data/landscape_indices"
    OUTPUT_FOLDER = None

    # Min and max values with 4 decimal places obtained from the first code
    ED_MIN, ED_MAX = 13.3333, 133.3333
    PD_MIN, PD_MAX = 0.1111, 11.1111
    AREA_MN_MIN, AREA_MN_MAX = 9.0000, 900.0000
    MESH_MIN, MESH_MAX = 9.0000, 900.0000

    batch_merge_norm(
        INPUT_FOLDER,
        ed_min=ED_MIN, ed_max=ED_MAX,
        pd_min=PD_MIN, pd_max=PD_MAX,
        area_mn_min=AREA_MN_MIN, area_mn_max=AREA_MN_MAX,
        mesh_min=MESH_MIN, mesh_max=MESH_MAX,
        out_folder=OUTPUT_FOLDER
    )