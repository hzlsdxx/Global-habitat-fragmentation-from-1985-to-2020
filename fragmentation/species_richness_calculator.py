import os
import glob
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
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

        # Create profile (use int32 for species count)
        profile = {
            'driver': 'GTiff',
            'dtype': rasterio.int32,
            'nodata': -9999,
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
    Resample input raster to reference grid and binarize

    Parameters:
    -----------
    input_path : str
        Input raster file path
    reference_profile : dict
        Reference grid profile

    Returns:
    --------
    binary_array : numpy.ndarray
        Binarized array (0 or 1)
    """
    try:
        with rasterio.open(input_path) as src:
            # Create target array
            resampled_array = np.zeros(
                (reference_profile['height'], reference_profile['width']),
                dtype=np.float32
            )

            # Resample
            reproject(
                source=rasterio.band(src, 1),
                destination=resampled_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=reference_profile['transform'],
                dst_crs=reference_profile['crs'],
                resampling=Resampling.nearest,
                src_nodata=src.nodata
            )

            # Binarize: non-zero and non-null values become 1, others become 0
            binary_array = np.where(
                (resampled_array != src.nodata if src.nodata is not None else True),
                1, 0
            ).astype(np.int32)

            return binary_array

    except Exception as e:
        print(f"Error processing file {input_path}: {e}")
        return None


def process_single_file_to_disk(args):
    """Process single file and save to temporary file to avoid memory issues"""
    file_path, reference_profile, temp_dir = args

    print(f"[Process {os.getpid()}] Processing: {os.path.basename(file_path)}")

    try:
        binary_array = resample_to_reference_grid(file_path, reference_profile)
        if binary_array is not None:
            # Save to temporary file instead of returning array
            temp_filename = f"temp_{os.getpid()}_{os.path.basename(file_path)}.tif"
            temp_path = os.path.join(temp_dir, temp_filename)

            # Update profile for saving binarized results
            temp_profile = reference_profile.copy()
            temp_profile['dtype'] = rasterio.int32
            temp_profile['nodata'] = -9999

            with rasterio.open(temp_path, 'w', **temp_profile) as dst:
                dst.write(binary_array, 1)

            valid_pixels = np.sum(binary_array)
            print(f"[Process {os.getpid()}] ✅ Completed: {os.path.basename(file_path)}, valid pixels: {valid_pixels}")
            return temp_path, os.path.basename(file_path), valid_pixels
        else:
            print(f"[Process {os.getpid()}] ❌ Failed: {os.path.basename(file_path)}")
            return None, os.path.basename(file_path), 0
    except Exception as e:
        print(f"[Process {os.getpid()}] ❌ Error: {os.path.basename(file_path)} - {e}")
        return None, os.path.basename(file_path), 0


def process_batch_files(args):
    """Parallel processing of a single batch of temporary files"""
    batch_files, nodata_value, height, width = args
    batch_id = os.getpid()

    print(f"[Process {batch_id}] Starting batch processing with {len(batch_files)} files")

    # Initialize accumulator for current batch
    batch_count = np.zeros((height, width), dtype=np.int32)

    processed_files = 0

    for temp_file in batch_files:
        try:
            with rasterio.open(temp_file) as src:
                data = src.read(1, masked=False).astype(np.int32)

                # Accumulate binary data (each pixel is 0 or 1)
                batch_count += data
                processed_files += 1

        except Exception as e:
            print(f"[Process {batch_id}] Error reading temporary file {temp_file}: {e}")
            continue

    print(f"[Process {batch_id}] Batch processing completed, successfully processed {processed_files}/{len(batch_files)} files")
    return batch_count


def parallel_aggregate_batches(temp_files, reference_profile, max_workers=16):
    """Parallel processing of all batches and aggregate results"""

    nodata_value = reference_profile['nodata']
    height = reference_profile['height']
    width = reference_profile['width']

    # Process temporary files in batches
    batch_size = 30  # Process 30 files at a time
    total_batches = (len(temp_files) + batch_size - 1) // batch_size

    print(f"Splitting {len(temp_files)} temporary files into {total_batches} batches for parallel processing...")

    # Prepare batch tasks
    batch_tasks = []
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(temp_files))
        batch_files = temp_files[start_idx:end_idx]
        batch_tasks.append((batch_files, nodata_value, height, width))

    # Parallel processing of all batches
    print(f"Using {min(max_workers, total_batches)} processes for parallel batch processing...")
    start_time = datetime.now()

    # Initialize total accumulator
    total_count = np.zeros((height, width), dtype=np.int32)

    # Use appropriate number of processes (not exceeding batch count)
    actual_workers = min(max_workers, total_batches)

    with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers) as executor:
        results = executor.map(process_batch_files, batch_tasks)

        # Merge results from all batches
        batch_num = 0
        for batch_count in results:
            if batch_count is not None:
                total_count += batch_count
                batch_num += 1
                print(f"Completed aggregation of batch {batch_num}/{total_batches}")

    processing_time = datetime.now() - start_time
    print(f"Parallel aggregation completed, time elapsed: {processing_time}")

    return total_count


def aggregate_species_rasters(
        input_folder,
        landuse_path,
        output_path,
        output_extent=None
):
    """
    Aggregate all species raster files containing MIX in the filename within the folder

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

    print("=== Species Raster File Aggregation Processing ===")

    # 1. Find all raster files containing MIX
    extensions = ['*.tif', '*.tiff', '*.img', '*.asc', '*.bil']
    species_files = []

    for ext in extensions:
        pattern = os.path.join(input_folder, '**', f'*MIX*{ext}')
        files = glob.glob(pattern, recursive=True)
        species_files.extend(files)

    if not species_files:
        raise RuntimeError(f"No raster files containing 'MIX' found in folder {input_folder}")

    print(f"Found {len(species_files)} MIX species raster files")
    for i, file in enumerate(species_files[:5]):  # Show first 5 files
        print(f"  {i + 1}. {os.path.basename(file)}")
    if len(species_files) > 5:
        print(f"  ... and {len(species_files)} files in total")

    # 2. Create 3km reference grid
    print(f"\nCreating 3km reference grid based on {os.path.basename(landuse_path)}...")
    reference_profile = create_3km_reference_grid(landuse_path, output_extent)
    print(f"Reference grid size: {reference_profile['width']} × {reference_profile['height']}")
    print(f"Pixel size: {reference_profile['transform'][0]}m")

    # 3. Parallel resampling of all files
    print(f"\nStarting parallel resampling processing...")
    max_workers = 16  # Adjust based on your system
    print(f"Using {max_workers} processes for parallel processing")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="species_aggregation_")
    print(f"Temporary file directory: {temp_dir}")

    start_time = datetime.now()

    try:
        # Prepare task parameters
        tasks = [(file_path, reference_profile, temp_dir) for file_path in species_files]

        # Parallel processing
        temp_files = []
        failed_files = []
        total_valid_pixels = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(process_single_file_to_disk, tasks)

            for temp_path, filename, valid_pixels in results:
                if temp_path is not None:
                    temp_files.append(temp_path)
                    total_valid_pixels += valid_pixels
                else:
                    failed_files.append(filename)

        processing_time = datetime.now() - start_time
        print(f"\nResampling completed, time elapsed: {processing_time}")
        print(f"Successfully processed: {len(temp_files)} files")
        print(f"Processing failed: {len(failed_files)} files")
        print(f"Total valid pixels: {total_valid_pixels:,}")

        if failed_files:
            print("Failed file list:")
            for file in failed_files:
                print(f"  - {file}")

        if not temp_files:
            raise RuntimeError("No successfully processed files, cannot perform aggregation")

        # 4. Parallel batch aggregation calculation
        print(f"\nStarting parallel aggregation calculation...")

        # Use parallel processing for batch aggregation
        count_array = parallel_aggregate_batches(
            temp_files, reference_profile, max_workers=16
        )

        # Statistics
        total_pixels = reference_profile['height'] * reference_profile['width']
        has_species = count_array > 0
        species_pixels = np.sum(has_species)
        zero_pixels = total_pixels - species_pixels

        print(f"Aggregation statistics:")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Pixels with species: {species_pixels:,} ({species_pixels / total_pixels * 100:.1f}%)")
        print(f"  Pixels without species: {zero_pixels:,} ({zero_pixels / total_pixels * 100:.1f}%)")
        if np.any(has_species):
            print(f"  Maximum species count: {np.max(count_array)}")
            print(f"  Average species count: {np.mean(count_array[has_species]):.2f}")

        # Species count distribution
        unique_counts, count_freq = np.unique(count_array, return_counts=True)
        print("Species count distribution:")
        for count, freq in zip(unique_counts[:10], count_freq[:10]):
            print(f"  {count} species: {freq:,} pixels")
        if len(unique_counts) > 10:
            print(f"  ... and {len(unique_counts) - 10} other count values")

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
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Write file
    with rasterio.open(output_path, 'w', **reference_profile) as dst:
        dst.write(count_array, 1)

        # Add metadata
        dst.update_tags(
            source_files_count=len(species_files),
            successful_files_count=len(temp_files),
            failed_files_count=len(failed_files),
            processing_time=str(processing_time),
            aggregation_method="species_count",
            max_species_count=int(np.max(count_array)),
            total_valid_pixels=int(total_valid_pixels),
            description="Species count aggregation from MIX raster files"
        )

    print(f"✅ Aggregation complete!")
    print(f"Output file: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    return output_path


def main():
    """Main function"""
    # Multiprocessing support for various environments

    # ============ Configuration Parameters ============
    INPUT_FOLDER = "data/landscape_indices/MA_1985"  # Folder containing MIX files
    LANDUSE_PATH = "data/land_cover_1985_300m.tif"  # 300m land use data
    OUTPUT_PATH = "data/landscape_indices/MA_1985_species_richness_3km.tif"  # Output path

    # Optional: specify output extent (min_x, min_y, max_x, max_y)
    # OUTPUT_EXTENT = (100000, 3000000, 200000, 4000000)  # Example extent
    OUTPUT_EXTENT = None  # Use complete extent of land use data

    # ============ Execute Aggregation ============
    try:
        result_path = aggregate_species_rasters(
            input_folder=INPUT_FOLDER,
            landuse_path=LANDUSE_PATH,
            output_path=OUTPUT_PATH,
            output_extent=OUTPUT_EXTENT
        )

        print(f"\nProcessing complete! Species richness raster saved at: {result_path}")

    except Exception as e:
        print(f"\nProcessing failed: {e}")
        raise


if __name__ == "__main__":
    main()