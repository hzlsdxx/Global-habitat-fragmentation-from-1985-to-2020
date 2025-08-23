import os
import glob
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge
from rasterio.transform import from_bounds
import concurrent.futures
import multiprocessing
from datetime import datetime
import tempfile
import shutil


def create_3km_reference_grid(landuse_path, output_extent=None):
    """
    Create 3km reference grid based on 300m land use data

    Parameters:
    -----------
    landuse_path : str
        Path to 300m land use data
    output_extent : tuple, optional
        Output extent (min_x, min_y, max_x, max_y), if None uses land use data extent

    Returns:
    --------
    profile : dict
        Profile information for 3km grid
    """
    with rasterio.open(landuse_path) as src:
        # Get original data information
        original_transform = src.transform
        original_width = src.width
        original_height = src.height
        original_crs = src.crs

        # Calculate 3km grid parameters
        pixel_size_300m = 300
        pixel_size_3km = 3000
        scale_factor = pixel_size_3km / pixel_size_300m  # 10

        # Calculate new dimensions
        new_width = int(np.ceil(original_width / scale_factor))
        new_height = int(np.ceil(original_height / scale_factor))

        # Create new transform matrix (keep top-left aligned)
        new_transform = rasterio.transform.Affine(
            pixel_size_3km, 0, original_transform.c,
            0, -pixel_size_3km, original_transform.f
        )

        # If output extent is specified, adjust grid
        if output_extent:
            min_x, min_y, max_x, max_y = output_extent
            # Align to 3km grid
            min_x = np.floor(min_x / pixel_size_3km) * pixel_size_3km
            min_y = np.floor(min_y / pixel_size_3km) * pixel_size_3km
            max_x = np.ceil(max_x / pixel_size_3km) * pixel_size_3km
            max_y = np.ceil(max_y / pixel_size_3km) * pixel_size_3km

            new_width = int((max_x - min_x) / pixel_size_3km)
            new_height = int((max_y - min_y) / pixel_size_3km)
            new_transform = rasterio.transform.Affine(
                pixel_size_3km, 0, min_x,
                0, -pixel_size_3km, max_y
            )

        # Create profile
        profile = {
            'driver': 'GTiff',
            'dtype': rasterio.float32,
            'nodata': -9999.0,
            'width': new_width,
            'height': new_height,
            'count': 1,
            'crs': original_crs,
            'transform': new_transform,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512
        }

    return profile


def resample_to_reference_grid(input_path, reference_profile):
    """
    Resample input raster to reference grid

    Parameters:
    -----------
    input_path : str
        Input raster file path
    reference_profile : dict
        Reference grid profile

    Returns:
    --------
    resampled_array : numpy.ndarray
        Resampled array
    """
    try:
        with rasterio.open(input_path) as src:
            # Create target array
            resampled_array = np.full(
                (reference_profile['height'], reference_profile['width']),
                reference_profile['nodata'],
                dtype=reference_profile['dtype']
            )

            # Resample
            reproject(
                source=rasterio.band(src, 1),
                destination=resampled_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=reference_profile['transform'],
                dst_crs=reference_profile['crs'],
                resampling=Resampling.nearest,  # Use nearest neighbor to preserve landscape index values
                src_nodata=src.nodata,
                dst_nodata=reference_profile['nodata']
            )

            return resampled_array

    except Exception as e:
        print(f"Error processing file {input_path}: {e}")
        return None


def process_single_file_to_disk(args):
    """Process single file and save to temporary file to avoid memory issues"""
    file_path, reference_profile, temp_dir = args

    print(f"[Process {os.getpid()}] Processing: {os.path.basename(file_path)}")

    try:
        resampled_array = resample_to_reference_grid(file_path, reference_profile)
        if resampled_array is not None:
            # Save to temporary file instead of returning array
            temp_filename = f"temp_{os.getpid()}_{os.path.basename(file_path)}"
            temp_path = os.path.join(temp_dir, temp_filename)

            with rasterio.open(temp_path, 'w', **reference_profile) as dst:
                dst.write(resampled_array, 1)

            print(f"[Process {os.getpid()}] ✅ Completed: {os.path.basename(file_path)}")
            return temp_path, os.path.basename(file_path)
        else:
            print(f"[Process {os.getpid()}] ❌ Failed: {os.path.basename(file_path)}")
            return None, os.path.basename(file_path)
    except Exception as e:
        print(f"[Process {os.getpid()}] ❌ Error: {os.path.basename(file_path)} - {e}")
        return None, os.path.basename(file_path)


def aggregate_mix_rasters(
        input_folder,
        landuse_path,
        output_path,
        output_extent=None
):
    """
    Aggregate all raster files containing MIX in the filename within the folder

    Parameters:
    -----------
    input_folder : str
        Input folder path
    landuse_path : str
        300m land use data path (for creating reference grid)
    output_path : str
        Output aggregated raster path
    output_extent : tuple, optional
        Output extent (min_x, min_y, max_x, max_y)
    """

    print("=== MIX Raster File Aggregation Processing ===")

    # 1. Find all raster files containing MIX
    pattern = os.path.join(input_folder, '**', '*MIX*.tif')
    mix_files = glob.glob(pattern, recursive=True)

    if not mix_files:
        raise RuntimeError(f"No TIF files containing 'MIX' found in folder {input_folder}")

    print(f"Found {len(mix_files)} MIX raster files")
    for i, file in enumerate(mix_files[:5]):  # Show first 5 files
        print(f"  {i + 1}. {os.path.basename(file)}")
    if len(mix_files) > 5:
        print(f"  ... and {len(mix_files)} files in total")

    # 2. Create 3km reference grid
    print(f"\nCreating 3km reference grid based on {os.path.basename(landuse_path)}...")
    reference_profile = create_3km_reference_grid(landuse_path, output_extent)
    print(f"Reference grid size: {reference_profile['width']} × {reference_profile['height']}")
    print(f"Pixel size: {reference_profile['transform'][0]}m")

    # 3. Parallel resampling of all files
    print(f"\nStarting parallel resampling processing...")
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {max_workers} processes for parallel processing")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="mix_aggregation_")
    print(f"Temporary file directory: {temp_dir}")

    start_time = datetime.now()

    try:
        # Prepare task parameters
        tasks = [(file_path, reference_profile, temp_dir) for file_path in mix_files]

        # Parallel processing
        temp_files = []
        failed_files = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(process_single_file_to_disk, tasks)

            for temp_path, filename in results:
                if temp_path is not None:
                    temp_files.append(temp_path)
                else:
                    failed_files.append(filename)

        processing_time = datetime.now() - start_time
        print(f"\nResampling completed, time elapsed: {processing_time}")
        print(f"Successfully processed: {len(temp_files)} files")
        print(f"Processing failed: {len(failed_files)} files")

        if failed_files:
            print("Failed file list:")
            for file in failed_files:
                print(f"  - {file}")

        if not temp_files:
            raise RuntimeError("No successfully processed files, cannot perform aggregation")

        # 4. Batch read and aggregate calculation (avoid loading all data into memory at once)
        print(f"\nStarting aggregation calculation...")

        # Initialize accumulators
        sum_array = np.zeros((reference_profile['height'], reference_profile['width']), dtype=np.float64)
        count_array = np.zeros((reference_profile['height'], reference_profile['width']), dtype=np.int32)
        nodata_value = reference_profile['nodata']

        # Process temporary files in batches
        batch_size = 10  # Process 10 files at a time to avoid memory overflow
        total_batches = (len(temp_files) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(temp_files))
            batch_files = temp_files[start_idx:end_idx]

            print(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_files)} files)...")

            for temp_file in batch_files:
                try:
                    with rasterio.open(temp_file) as src:
                        data = src.read(1, masked=False).astype(np.float64)

                        # Create valid data mask
                        valid_mask = (data != nodata_value) & (~np.isnan(data))

                        # Accumulate valid data
                        sum_array[valid_mask] += data[valid_mask]
                        count_array[valid_mask] += 1

                except Exception as e:
                    print(f"Error reading temporary file {temp_file}: {e}")
                    continue

        # Calculate average
        result_array = np.full(
            (reference_profile['height'], reference_profile['width']),
            nodata_value,
            dtype=reference_profile['dtype']
        )

        # For pixels with at least one valid value, calculate mean
        has_valid_data = count_array > 0
        if np.any(has_valid_data):
            mean_values = sum_array[has_valid_data] / count_array[has_valid_data]
            result_array[has_valid_data] = mean_values.astype(reference_profile['dtype'])

        # Statistics
        total_pixels = reference_profile['height'] * reference_profile['width']
        valid_pixels = np.sum(has_valid_data)
        nodata_pixels = total_pixels - valid_pixels

        print(f"Aggregation statistics:")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Valid pixels: {valid_pixels:,} ({valid_pixels / total_pixels * 100:.1f}%)")
        print(f"  NoData pixels: {nodata_pixels:,} ({nodata_pixels / total_pixels * 100:.1f}%)")
        if np.any(has_valid_data):
            valid_values = result_array[has_valid_data]
            print(f"  Aggregated value range: {np.min(valid_values):.4f} ~ {np.max(valid_values):.4f}")

    finally:
        # Clean up temporary files
        print(f"\nCleaning up temporary files...")
        try:
            shutil.rmtree(temp_dir)
            print(f"Temporary directory deleted: {temp_dir}")
        except Exception as e:
            print(f"Error deleting temporary directory: {e}")

    # 5. Save results
    print(f"\nSaving aggregation results to: {output_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write file
    with rasterio.open(output_path, 'w', **reference_profile) as dst:
        dst.write(result_array, 1)

        # Add metadata
        dst.update_tags(
            source_files_count=len(mix_files),
            successful_files_count=len(temp_files),
            failed_files_count=len(failed_files),
            processing_time=str(processing_time),
            aggregation_method="mean",
            description="Aggregated MIX raster from multiple input files"
        )

    print(f"✅ Aggregation complete!")
    print(f"Output file: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    return output_path


def main():
    """Main function"""
    # Multiprocessing support for Windows environment
    multiprocessing.freeze_support()

    # ============ Configuration Parameters ============
    INPUT_FOLDER = "data/landscape_indices/BIRD_2020"  # Folder containing MIX files
    LANDUSE_PATH = "data/land_cover_2020_300m.tif"  # 300m land use data
    OUTPUT_PATH = "data/landscape_indices/BIRD_2020_aggregated_3km.tif"  # Output path

    # Optional: specify output extent (min_x, min_y, max_x, max_y)
    # OUTPUT_EXTENT = (100000, 3000000, 200000, 4000000)  # Example extent
    OUTPUT_EXTENT = None  # Use complete extent of land use data

    # ============ Execute Aggregation ============
    try:
        result_path = aggregate_mix_rasters(
            input_folder=INPUT_FOLDER,
            landuse_path=LANDUSE_PATH,
            output_path=OUTPUT_PATH,
            output_extent=OUTPUT_EXTENT
        )

        print(f"\n🎉 Processing complete! Aggregated file saved at: {result_path}")

    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()