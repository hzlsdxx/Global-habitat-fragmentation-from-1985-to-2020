import os
import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from skimage.measure import label, regionprops
from datetime import datetime
from multiprocessing import Pool, cpu_count
import functools

# -------------------------------
# File path settings (modify according to your actual situation)
# -------------------------------
dem_path = "data/dem2024_clip.tif"
gdb_file = "data/sp_data.gdb"  # GDB database file
layer_name = "AMPHIBIANS_population_trend_final_2025_05_02"  # Layer name
landcover_path = "data/land_cover_1985_300m.tif"
output_folder = "data/landscape_indices/AMPH_1985"  # Output results folder

# Parallel processing settings
MAX_WORKERS = 16  # Use CPU cores - 1, leave one core for the system
# MAX_WORKERS = 4  # Can also set manually, e.g., use only 4 cores

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)


# -------------------------------
# File checking functions
# -------------------------------
def check_output_files_exist(feature_id, output_folder):
    """Check if all output files for a feature exist and are complete"""
    required_files = [
        f"{feature_id}_landcover.tif",
        f"{feature_id}_PD.tif",
        f"{feature_id}_ED.tif",
        f"{feature_id}_AREA_MN.tif",
        f"{feature_id}_MESH.tif"
    ]

    for filename in required_files:
        filepath = os.path.join(output_folder, filename)
        if not os.path.exists(filepath):
            return False

        # Additional check if file is empty or corrupted
        try:
            if os.path.getsize(filepath) == 0:
                return False
        except OSError:
            return False

    return True


def clean_incomplete_files(feature_id, output_folder):
    """Clean up incomplete output files"""
    files_to_clean = [
        f"{feature_id}_landcover.tif",
        f"{feature_id}_PD.tif",
        f"{feature_id}_ED.tif",
        f"{feature_id}_AREA_MN.tif",
        f"{feature_id}_MESH.tif"
    ]

    for filename in files_to_clean:
        filepath = os.path.join(output_folder, filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"Deleted incomplete file: {filename}")
            except Exception as e:
                print(f"Error deleting file {filename}: {e}")


def get_processing_statistics(gdf, output_folder):
    """Get processing statistics"""
    total_features = len(gdf)
    completed_count = 0
    pending_features = []

    for idx, row in gdf.iterrows():
        feature_id = row['id_no']
        if check_output_files_exist(feature_id, output_folder):
            completed_count += 1
        else:
            pending_features.append((idx, feature_id))

    return {
        'total': total_features,
        'completed': completed_count,
        'pending': len(pending_features),
        'pending_features': pending_features
    }


# -------------------------------
# Define functions for calculating indices
# -------------------------------
def calculate_indices_in_window(data, pixel_size, no_data_value):
    data_copy = data.copy()
    data_copy2 = data_copy
    data_copy[data_copy == no_data_value] = 0  # Set no_data pixels to 0
    labeled_image, num_patches = label(data_copy, return_num=True)
    regions = regionprops(labeled_image)
    patch_areas = [region.area * (pixel_size ** 2) for region in regions]

    def calculate_total_edge_length(data_arr, nodata_value, count_boundary=True):
        """Calculate total edge length in landscape, not considering resolution."""
        if count_boundary:
            # Use padding method as in PyLandStats
            padded_arr = np.pad(
                data_arr,
                pad_width=1,
                mode="constant",
                constant_values=nodata_value
            )

            # Calculate all horizontal edge differences
            horizontal_edges = np.sum(padded_arr[1:, :] != padded_arr[:-1, :])
            # Calculate all vertical edge differences
            vertical_edges = np.sum(padded_arr[:, 1:] != padded_arr[:, :-1])

            return horizontal_edges + vertical_edges

        else:
            # Don't calculate boundary - create adjacency matrix like PyLandStats
            classes = np.unique(data_arr)
            classes = classes[classes != nodata_value]

            # Create adjacency matrix
            adjacency_matrix = np.zeros((len(classes), len(classes)), dtype=np.int32)

            # Create mapping from class value to index
            class_to_idx = {class_val: idx for idx, class_val in enumerate(classes)}

            # Calculate horizontal adjacency
            for i in range(data_arr.shape[0]):
                for j in range(data_arr.shape[1] - 1):
                    val1 = data_arr[i, j]
                    val2 = data_arr[i, j + 1]
                    if (val1 != nodata_value and val2 != nodata_value and val1 != val2):
                        idx1 = class_to_idx[val1]
                        idx2 = class_to_idx[val2]
                        adjacency_matrix[idx1, idx2] += 1
                        adjacency_matrix[idx2, idx1] += 1  # Add symmetrically

            # Calculate vertical adjacency
            for i in range(data_arr.shape[0] - 1):
                for j in range(data_arr.shape[1]):
                    val1 = data_arr[i, j]
                    val2 = data_arr[i + 1, j]
                    if (val1 != nodata_value and val2 != nodata_value and val1 != val2):
                        idx1 = class_to_idx[val1]
                        idx2 = class_to_idx[val2]
                        adjacency_matrix[idx1, idx2] += 1
                        adjacency_matrix[idx2, idx1] += 1  # Add symmetrically

            # Get upper triangular matrix (avoid double counting)
            upper_triangle = np.triu(adjacency_matrix)

            # Set diagonal elements to 0 (no edges within same class)
            np.fill_diagonal(upper_triangle, 0)

            # Total edge length is sum of upper triangle elements
            return np.sum(upper_triangle)

    valid_mask = (data_copy != 0)  # Assume 0 is nodata value

    # Calculate valid landscape area
    valid_pixels = np.sum(valid_mask)
    total_area = valid_pixels * (pixel_size ** 2)

    total_edge_length = calculate_total_edge_length(data_copy2, 0) * pixel_size

    PD = (num_patches / total_area) * 1e6 if total_area > 0 else 0
    ED = (total_edge_length / total_area) * 1e4 if total_area > 0 else 0
    AREA_MN = (np.mean(patch_areas) / 1e4) if patch_areas else 0
    if total_area == 0 or len(patch_areas) == 0:
        MESH = 0
    else:
        sum_a = sum((int(area)) ** 2 for area in patch_areas)
        MESH = (sum_a / total_area) * (1 / 1e4)
    return PD, ED, AREA_MN, MESH


def create_indices_raster(image_path, output_paths, window_size=3000, pixel_size=300, lc_nodata=0):
    with rasterio.open(image_path) as src:
        raster = src.read(1)
        rows, cols = raster.shape
        num_rows = int(window_size / pixel_size)
        num_cols = int(window_size / pixel_size)

        # Calculate output image size (downsampled to 3km resolution)
        out_rows = (rows + num_rows - 1) // num_rows
        out_cols = (cols + num_cols - 1) // num_cols

        # Set nodata value for landscape indices raster (choose an impossible negative value)
        indices_nodata = -9999.0

        # Initialize rasters with nodata value, not 0
        PD_raster = np.full((out_rows, out_cols), indices_nodata, dtype=np.float32)
        ED_raster = np.full((out_rows, out_cols), indices_nodata, dtype=np.float32)
        AREA_MN_raster = np.full((out_rows, out_cols), indices_nodata, dtype=np.float32)
        MESH_raster = np.full((out_rows, out_cols), indices_nodata, dtype=np.float32)

        for i in range(0, rows, num_rows):
            for j in range(0, cols, num_cols):
                window = raster[i:i + num_rows, j:j + num_cols]
                if window.size == 0:
                    continue

                # Calculate output position (position after downsampling)
                out_i = i // num_rows
                out_j = j // num_cols

                # Check if window has enough valid data (non-nodata values)
                valid_pixels = np.sum(window != lc_nodata)
                if valid_pixels > 0:  # Only calculate indices on windows with valid data
                    PD, ED, AREA_MN, MESH = calculate_indices_in_window(window, pixel_size, lc_nodata)
                    PD_raster[out_i, out_j] = PD
                    ED_raster[out_i, out_j] = ED
                    AREA_MN_raster[out_i, out_j] = AREA_MN
                    MESH_raster[out_i, out_j] = MESH
                # If no valid data, keep nodata value

        # Update profile to reflect new resolution (3km×3km) and set nodata value
        from affine import Affine
        profile = src.profile
        profile.update(
            dtype=rasterio.float32,
            width=out_cols,
            height=out_rows,
            transform=src.transform * Affine.scale(num_cols, num_rows),
            compress='lzw',
            tiled=True,
            blockxsize=512,
            blockysize=512,
            nodata=indices_nodata  # Explicitly set nodata value
        )

        # Save output files
        with rasterio.open(output_paths[0], 'w', **profile) as dst:
            dst.write(PD_raster, 1)
        with rasterio.open(output_paths[1], 'w', **profile) as dst:
            dst.write(ED_raster, 1)
        with rasterio.open(output_paths[2], 'w', **profile) as dst:
            dst.write(AREA_MN_raster, 1)
        with rasterio.open(output_paths[3], 'w', **profile) as dst:
            dst.write(MESH_raster, 1)


# -------------------------------
# Single feature processing function (for parallel processing)
# -------------------------------
def process_single_feature(args):
    """Function to process a single feature"""
    row_dict, idx, total_features, dem_path, landcover_path, output_folder, dem_nodata, lc_nodata = args

    feature_id = row_dict['id_no']

    # Check if output files already exist and are complete
    if check_output_files_exist(feature_id, output_folder):
        print(f"[Process {os.getpid()}] Skipping completed feature {idx} (id_no={feature_id}) - files already exist")
        return True, feature_id, "Files already exist"

    print(f"[Process {os.getpid()}] Starting to process feature {idx} (id_no={feature_id})")

    try:
        # Reconstruct geometry object
        from shapely.geometry import shape
        geom = [shape(row_dict['geometry'])]

        # Clip DEM
        with rasterio.open(dem_path) as src:
            clipped_dem, dem_transform = mask(src, geom, crop=True)
            clipped_dem = clipped_dem[0]  # Assume DEM is single band
            dem_meta_clipped = src.meta.copy()
            dem_meta_clipped.update({
                "height": clipped_dem.shape[0],
                "width": clipped_dem.shape[1],
                "transform": dem_transform
            })

        # Clip land cover
        with rasterio.open(landcover_path) as src:
            clipped_lc, lc_transform = mask(src, geom, crop=True)
            clipped_lc = clipped_lc[0]  # Assume land cover data is single band
            lc_meta_clipped = src.meta.copy()
            lc_meta_clipped.update({
                "height": clipped_lc.shape[0],
                "width": clipped_lc.shape[1],
                "transform": lc_transform
            })

        # Step 1: Filter clipped DEM based on attribute altitude range
        altitude_l = row_dict.get('altitude_low')
        altitude_u = row_dict.get('altitude_upper')
        if pd.isnull(altitude_l) or pd.isnull(altitude_u):
            dem_modified = clipped_dem.copy()
        else:
            dem_modified = np.where((clipped_dem >= altitude_l) & (clipped_dem <= altitude_u),
                                    clipped_dem, dem_nodata)
        dem_modified = dem_modified.astype(dem_meta_clipped['dtype'])

        # Step 2: Build allowed land cover code set based on classification fields
        allowed_codes = set()
        if row_dict.get('Artificial_Terrestrial') == '1':
            allowed_codes.update([1, 6])
        if row_dict.get('Desert') == '1':
            allowed_codes.add(7)
        if row_dict.get('Forest') == '1':
            allowed_codes.add(2)
        if row_dict.get('Grassland') == '1':
            allowed_codes.add(4)
        if row_dict.get('Rocky_areas') == '1':
            allowed_codes.update([9, 200])
        if row_dict.get('Savanna') == '1':
            allowed_codes.add(100)
        if row_dict.get('Shrubland') == '1':
            allowed_codes.add(3)
        if row_dict.get('Wetlands__inland_') == '1':
            allowed_codes.update([5, 8])

        # If allowed codes is empty, skip this feature
        if not allowed_codes:
            error_msg = "Allowed land cover codes are empty"
            print(f"[Process {os.getpid()}] Feature (id_no={feature_id}): {error_msg}")
            return False, feature_id, error_msg

        # If clipped DEM and land cover data have different shapes, resample land cover data to DEM grid
        if dem_modified.shape != clipped_lc.shape:
            resampled_lc = np.empty(dem_modified.shape, dtype=clipped_lc.dtype)
            reproject(
                source=clipped_lc,
                destination=resampled_lc,
                src_transform=lc_transform,
                src_crs=lc_meta_clipped['crs'],
                dst_transform=dem_meta_clipped['transform'],
                dst_crs=dem_meta_clipped['crs'],
                resampling=Resampling.nearest
            )
            clipped_lc = resampled_lc

        # Step 3: Use clipped DEM to mask land cover data
        lc_intermediate = np.where(dem_modified != dem_nodata, clipped_lc, lc_nodata)
        lc_intermediate = lc_intermediate.astype(lc_meta_clipped['dtype'])

        # Step 4: Further filter, keep only pixels corresponding to allowed codes, assign others as nodata
        lc_final = np.where(np.isin(lc_intermediate, list(allowed_codes)), lc_intermediate, lc_nodata)
        lc_final = lc_final.astype(lc_meta_clipped['dtype'])

        # Step 5: Write processed land cover data to intermediate file, filename using id_no
        intermediate_file = os.path.join(output_folder, f"{feature_id}_landcover.tif")
        # Add compression settings for land cover file
        lc_meta_compressed = lc_meta_clipped.copy()
        lc_meta_compressed.update({
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512
        })
        with rasterio.open(intermediate_file, 'w', **lc_meta_compressed) as dst:
            dst.write(lc_final, 1)

        # Step 6: Use intermediate file to calculate indices, output four raster files, filename using id_no and index name
        output_paths = [
            os.path.join(output_folder, f"{feature_id}_PD.tif"),
            os.path.join(output_folder, f"{feature_id}_ED.tif"),
            os.path.join(output_folder, f"{feature_id}_AREA_MN.tif"),
            os.path.join(output_folder, f"{feature_id}_MESH.tif")
        ]
        create_indices_raster(intermediate_file, output_paths, window_size=3000, pixel_size=300, lc_nodata=lc_nodata)

        # Verify all output files were successfully created
        if check_output_files_exist(feature_id, output_folder):
            print(f"[Process {os.getpid()}] Feature (id_no={feature_id}) processing completed!")
            return True, feature_id, "Processing successful"
        else:
            raise Exception("Some output files were not successfully created")

    except Exception as e:
        error_msg = str(e)
        print(f"[Process {os.getpid()}] Error processing feature (id_no={feature_id}): {error_msg}")
        # Clean up potentially incomplete files
        clean_incomplete_files(feature_id, output_folder)
        return False, feature_id, error_msg


# -------------------------------
# Main processing function
# -------------------------------
def main():
    print(f"Detected {cpu_count()} CPU cores, will use {MAX_WORKERS} processes for parallel processing")

    # Read vector data (specified layer in GDB file)
    gdf = gpd.read_file(gdb_file, layer=layer_name)
    print(f"Read {len(gdf)} features in total.")

    # Pre-obtain nodata values from DEM and land cover data
    with rasterio.open(dem_path) as src:
        dem_nodata = src.nodata
        if dem_nodata is None:
            dem_nodata = -9999

    with rasterio.open(landcover_path) as src:
        lc_nodata = src.nodata
        if lc_nodata is None:
            lc_nodata = 0

    # Get processing statistics
    stats = get_processing_statistics(gdf, output_folder)

    print(f"=== Processing Status ===")
    print(f"Total features: {stats['total']}")
    print(f"Completed: {stats['completed']}")
    print(f"Pending: {stats['pending']}")
    print(f"Completion rate: {stats['completed'] / stats['total'] * 100:.1f}%")
    print(f"========================")

    if stats['pending'] == 0:
        print("All features have been processed!")
        return

    # Prepare task list for pending processing
    tasks = []
    for idx, feature_id in stats['pending_features']:
        row = gdf.iloc[idx]
        # Convert GeoDataFrame row to dictionary, including geometry information
        row_dict = row.to_dict()
        row_dict['geometry'] = row.geometry.__geo_interface__

        tasks.append((
            row_dict, idx, len(gdf), dem_path, landcover_path,
            output_folder, dem_nodata, lc_nodata
        ))

    # Parallel processing
    print(f"Starting parallel processing of {len(tasks)} pending features...")
    start_time = datetime.now()

    success_count = 0
    failure_count = 0
    failed_features = []

    # Use process pool for parallel processing
    with Pool(processes=MAX_WORKERS) as pool:
        results = pool.map(process_single_feature, tasks)

        # Count results
        for success, feature_id, message in results:
            if success:
                success_count += 1
            else:
                failure_count += 1
                failed_features.append(feature_id)

    end_time = datetime.now()
    processing_time = end_time - start_time

    print(f"\n=== Parallel Processing Complete ===")
    print(f"Processing time: {processing_time}")
    print(f"Successfully processed: {success_count}")
    print(f"Processing failed: {failure_count}")
    if processing_time.total_seconds() > 0:
        print(f"Average processing speed: {len(tasks) / processing_time.total_seconds():.2f} features/second")

    # Final statistics
    final_stats = get_processing_statistics(gdf, output_folder)
    print(f"\n=== Final Statistics ===")
    print(f"Total features: {final_stats['total']}")
    print(f"Successfully processed: {final_stats['completed']}")
    print(f"Pending: {final_stats['pending']}")
    print(f"Completion rate: {final_stats['completed'] / final_stats['total'] * 100:.1f}%")
    if failed_features:
        print(f"Failed feature IDs: {failed_features[:10]}{'...' if len(failed_features) > 10 else ''}")
    print("=======================")


if __name__ == "__main__":
    main()