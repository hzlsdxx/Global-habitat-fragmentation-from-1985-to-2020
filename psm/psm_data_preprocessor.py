import os
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from rasterio.mask import mask
from datetime import datetime
import concurrent.futures
import multiprocessing


def get_resampling_method(file_path):
    """
    Intelligently select resampling method based on filename

    Parameters:
    -----------
    file_path : str
        File path

    Returns:
    --------
    Resampling : Resampling method
    str : Data type description
    """
    filename = os.path.basename(file_path).lower()

    if 'land_cover' in filename or 'landuse' in filename or 'lulc' in filename:
        return Resampling.mode, "Categorical data"
    elif 'dem' in filename or 'elevation' in filename or 'altitude' in filename:
        return Resampling.average, "Elevation data"
    elif 'treecover' in filename or 'forest' in filename or 'canopy' in filename:
        return Resampling.average, "Forest cover data"
    elif 'travel' in filename or 'accessibility' in filename or 'distance' in filename:
        return Resampling.average, "Accessibility data"
    elif 'hfp' in filename or 'footprint' in filename or 'human' in filename:
        return Resampling.average, "Human footprint data"
    elif 'population' in filename or 'density' in filename:
        return Resampling.sum, "Population density data"
    elif 'temperature' in filename or 'precipitation' in filename or 'climate' in filename:
        return Resampling.average, "Climate data"
    else:
        return Resampling.average, "Continuous data"


def get_reference_grid_info(reference_path):
    """
    Get reference raster grid information

    Parameters:
    -----------
    reference_path : str
        Reference raster data path (species change data)

    Returns:
    --------
    dict : Dictionary containing all reference grid information
    """

    print(f"=== Reading reference raster data ===")
    print(f"File path: {reference_path}")

    with rasterio.open(reference_path) as src:
        reference_info = {
            'path': reference_path,
            'profile': src.profile.copy(),
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'bounds': src.bounds,
            'crs': src.crs,
            'nodata': src.nodata,
            'dtype': src.dtypes[0]
        }

        print(f"Reference grid information:")
        print(f"  Dimensions: {reference_info['width']} × {reference_info['height']}")
        print(f"  Resolution: {reference_info['transform'][0]:.1f}m")
        print(f"  CRS: {reference_info['crs']}")
        print(f"  Bounds: {reference_info['bounds']}")
        print(f"  NoData: {reference_info['nodata']}")
        print(f"  Data type: {reference_info['dtype']}")

        return reference_info


def crop_and_align_raster(input_raster_path, reference_info, output_path):
    """
    Crop and align raster to reference grid (intelligent resampling)

    Parameters:
    -----------
    input_raster_path : str
        Input raster file path
    reference_info : dict
        Reference grid information
    output_path : str
        Output file path

    Returns:
    --------
    str : Output file path
    """

    print(f"\nProcessing raster: {os.path.basename(input_raster_path)}")

    # Intelligently select resampling method
    resampling_method, data_type = get_resampling_method(input_raster_path)
    print(f"  Data type: {data_type}")
    print(f"  Resampling method: {resampling_method}")

    start_time = datetime.now()

    try:
        with rasterio.open(input_raster_path) as src:

            print(f"  Original raster information:")
            print(f"    Dimensions: {src.width} × {src.height}")
            print(f"    Resolution: {src.transform[0]:.1f}m")
            print(f"    Bounds: {src.bounds}")
            print(f"    CRS: {src.crs}")

            # Get reference raster bounds information
            ref_bounds = reference_info['bounds']
            src_bounds = src.bounds

            print(f"  Reference raster bounds: {ref_bounds}")

            # Calculate overlap area
            overlap_left = max(ref_bounds.left, src_bounds.left)
            overlap_bottom = max(ref_bounds.bottom, src_bounds.bottom)
            overlap_right = min(ref_bounds.right, src_bounds.right)
            overlap_top = min(ref_bounds.top, src_bounds.top)

            # Check for overlap
            if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
                print(f"    Warning: Input raster has no overlap with reference raster!")
                print(f"    Input bounds: {src_bounds}")
                print(f"    Reference bounds: {ref_bounds}")

                # Create result filled with NoData
                # Choose appropriate NoData value based on source data type
                if src.dtypes[0] == 'uint8':
                    dst_nodata = 255
                elif src.dtypes[0] == 'uint16':
                    dst_nodata = 65535
                elif src.dtypes[0] == 'int16':
                    dst_nodata = -32768
                elif src.dtypes[0] == 'int32':
                    dst_nodata = -2147483648
                else:  # float types
                    dst_nodata = reference_info['nodata'] if reference_info['nodata'] is not None else -9999

                aligned_data = np.full(
                    (reference_info['height'], reference_info['width']),
                    dst_nodata,
                    dtype=src.dtypes[0]
                )

            else:
                print(f"  Overlap area: left={overlap_left:.2f}, right={overlap_right:.2f}")
                print(f"               bottom={overlap_bottom:.2f}, top={overlap_top:.2f}")

                # Calculate overlap area ratio
                ref_area = (ref_bounds.right - ref_bounds.left) * (ref_bounds.top - ref_bounds.bottom)
                overlap_area = (overlap_right - overlap_left) * (overlap_top - overlap_bottom)
                overlap_ratio = overlap_area / ref_area * 100
                print(f"  Overlap ratio: {overlap_ratio:.1f}% (relative to reference raster)")

                # Check if reprojection is needed
                if src.crs != reference_info['crs']:
                    print(f"    Reprojection needed: {src.crs} → {reference_info['crs']}")

                # Get appropriate NoData values
                ref_nodata = reference_info['nodata'] if reference_info['nodata'] is not None else -9999
                src_nodata = src.nodata

                # Choose appropriate NoData value based on target data type
                if src.dtypes[0] == 'uint8':
                    dst_nodata = 255
                elif src.dtypes[0] == 'uint16':
                    dst_nodata = 65535
                elif src.dtypes[0] == 'int16':
                    dst_nodata = -32768
                elif src.dtypes[0] == 'int32':
                    dst_nodata = -2147483648
                else:  # float types
                    dst_nodata = ref_nodata

                print(f"    Source data type: {src.dtypes[0]}, using NoData: {dst_nodata}")

                # Create target array (using full dimensions of reference grid)
                aligned_data = np.full(
                    (reference_info['height'], reference_info['width']),
                    dst_nodata,
                    dtype=src.dtypes[0]
                )

                # Execute cropping and resampling
                print(f"    Executing crop and resample to reference grid (method: {resampling_method})...")

                reproject(
                    source=rasterio.band(src, 1),
                    destination=aligned_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=reference_info['transform'],
                    dst_crs=reference_info['crs'],
                    resampling=resampling_method,
                    src_nodata=src_nodata,
                    dst_nodata=dst_nodata
                )

            # Statistics for results
            # Determine NoData value for statistics
            if 'dst_nodata' in locals():
                stat_nodata = dst_nodata
            else:
                stat_nodata = reference_info['nodata'] if reference_info['nodata'] is not None else -9999

            if stat_nodata is not None:
                valid_mask = aligned_data != stat_nodata
            else:
                valid_mask = ~np.isnan(aligned_data)

            valid_count = np.sum(valid_mask)
            total_count = aligned_data.size
            coverage_percentage = valid_count / total_count * 100

            print(f"  Result statistics:")
            print(f"    Valid pixels: {valid_count:,}/{total_count:,} ({coverage_percentage:.1f}%)")

            if valid_count > 0:
                data_min = np.min(aligned_data[valid_mask])
                data_max = np.max(aligned_data[valid_mask])
                data_mean = np.mean(aligned_data[valid_mask])
                print(f"    Value range: [{data_min:.4f}, {data_max:.4f}]")
                print(f"    Mean value: {data_mean:.4f}")

                # Coverage warnings
                if coverage_percentage < 10:
                    print(f"    Warning: Very low valid data coverage ({coverage_percentage:.1f}%), please check data extent!")
                elif coverage_percentage < 50:
                    print(f"    Note: Low valid data coverage ({coverage_percentage:.1f}%)")
            else:
                print(f"    Warning: No valid data!")

            # Prepare output configuration
            output_profile = reference_info['profile'].copy()

            # Determine output NoData value
            if 'dst_nodata' in locals():
                output_nodata = dst_nodata
            else:
                output_nodata = reference_info['nodata'] if reference_info['nodata'] is not None else -9999

            output_profile.update({
                'dtype': src.dtypes[0],
                'nodata': output_nodata,
                'compress': 'lzw',
                'tiled': True,
                'blockxsize': min(512, reference_info['width']),
                'blockysize': min(512, reference_info['height'])
            })

            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Save result
            with rasterio.open(output_path, 'w', **output_profile) as dst:
                dst.write(aligned_data, 1)

                # Add detailed metadata
                dst.update_tags(
                    source_file=os.path.basename(input_raster_path),
                    reference_file=os.path.basename(reference_info['path']),
                    processing_time=str(datetime.now() - start_time),
                    resampling_method=str(resampling_method),
                    data_type=data_type,
                    coverage_percentage=f"{coverage_percentage:.2f}%",
                    valid_pixels=int(valid_count),
                    total_pixels=int(total_count),
                    description=f"Cropped and aligned to reference grid using {resampling_method}"
                )

            processing_time = datetime.now() - start_time
            file_size = os.path.getsize(output_path) / 1024 / 1024  # MB

            print(f"    Completed - Time: {processing_time}, File size: {file_size:.1f} MB")

            return output_path

    except Exception as e:
        print(f"    Failed: {str(e)}")
        raise


def process_single_raster_alignment(args):
    """Function for parallel processing of single raster alignment"""
    input_path, reference_info, output_path = args

    try:
        result_path = crop_and_align_raster(input_path, reference_info, output_path)
        return True, os.path.basename(input_path), result_path
    except Exception as e:
        return False, os.path.basename(input_path), str(e)


def batch_crop_and_align_rasters(input_raster_paths, reference_info, output_folder, max_workers=None):
    """
    Batch crop and align raster data to reference grid

    Parameters:
    -----------
    input_raster_paths : list
        List of input raster file paths
    reference_info : dict
        Reference grid information
    output_folder : str
        Output folder
    max_workers : int
        Maximum number of parallel processes

    Returns:
    --------
    list : List of successfully processed files
    list : List of failed files
    """

    print(f"\n=== Batch Raster Cropping and Alignment Processing ===")
    print(f"Need to process {len(input_raster_paths)} raster files")
    print(f"Output folder: {output_folder}")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Preview resampling methods
    print(f"\nResampling method preview:")
    for input_path in input_raster_paths:
        if os.path.exists(input_path):
            method, data_type = get_resampling_method(input_path)
            print(f"  {os.path.basename(input_path)}: {method} ({data_type})")

    # Prepare task list
    tasks = []
    for input_path in input_raster_paths:
        if not os.path.exists(input_path):
            print(f"Warning: File does not exist - {input_path}")
            continue

        # Generate output filename
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_name = f"{base_name}_aligned_to_reference.tif"
        output_path = os.path.join(output_folder, output_name)

        tasks.append((input_path, reference_info, output_path))

    if not tasks:
        print("No valid input files")
        return [], []

    # Set number of parallel processes
    if max_workers is None:
        max_workers = min(len(tasks), multiprocessing.cpu_count() - 1)

    print(f"\nUsing {max_workers} processes for parallel processing...")

    start_time = datetime.now()
    successful = []
    failed = []

    # Parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(process_single_raster_alignment, tasks)

        for success, filename, result in results:
            if success:
                successful.append(result)
                print(f"Success: {filename}")
            else:
                failed.append((filename, result))
                print(f"Failed: {filename} - {result}")

    processing_time = datetime.now() - start_time

    print(f"\n=== Batch Processing Complete ===")
    print(f"Processing time: {processing_time}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("Failed files:")
        for filename, error in failed:
            print(f"  {filename}: {error}")

    return successful, failed


def create_country_raster_from_shapefile(shapefile_path, country_field, reference_info, output_path):
    """
    Create country code raster from shapefile (cropped to reference extent)

    Parameters:
    -----------
    shapefile_path : str
        Country shapefile path
    country_field : str
        Country code field name (e.g., 'ADM0_A3_CN')
    reference_info : dict
        Reference grid information
    output_path : str
        Output raster path

    Returns:
    --------
    str : Output file path
    dict : Country code mapping dictionary
    """

    print(f"\n=== Creating Country Code Raster ===")
    print(f"Shapefile: {os.path.basename(shapefile_path)}")
    print(f"Country field: {country_field}")
    print(f"Output file: {os.path.basename(output_path)}")

    start_time = datetime.now()

    try:
        # Read shapefile
        print(f"Reading shapefile...")
        gdf = gpd.read_file(shapefile_path)

        print(f"  Original CRS: {gdf.crs}")
        print(f"  Feature count: {len(gdf)}")

        # Check if field exists
        if country_field not in gdf.columns:
            available_fields = list(gdf.columns)
            raise ValueError(f"Field '{country_field}' does not exist. Available fields: {available_fields}")

        # Reproject to reference coordinate system
        if gdf.crs != reference_info['crs']:
            print(f"  Reprojecting to: {reference_info['crs']}")
            gdf = gdf.to_crs(reference_info['crs'])

        # Crop to reference extent
        ref_bounds = reference_info['bounds']
        print(f"  Cropping to reference extent: {ref_bounds}")

        # Create reference extent geometry
        from shapely.geometry import box
        ref_bbox = box(ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top)

        # Crop features to reference extent
        gdf_clipped = gdf[gdf.geometry.intersects(ref_bbox)].copy()
        gdf_clipped['geometry'] = gdf_clipped.geometry.intersection(ref_bbox)

        print(f"  Feature count after cropping: {len(gdf_clipped)}")

        # Get unique country codes and create number mapping
        unique_countries = gdf_clipped[country_field].dropna().unique()
        country_mapping = {country: idx for idx, country in enumerate(sorted(unique_countries))}

        print(f"  Found {len(unique_countries)} unique countries/regions:")
        for country, code in sorted(country_mapping.items())[:10]:  # Show only first 10
            print(f"    {code}: {country}")
        if len(unique_countries) > 10:
            print(f"    ... and {len(unique_countries)} total")

        # Add numerical encoding to each feature
        gdf_clipped['raster_value'] = gdf_clipped[country_field].map(country_mapping)

        # Handle missing values
        missing_mask = gdf_clipped['raster_value'].isna()
        if missing_mask.any():
            print(f"  Warning: {missing_mask.sum()} features have empty {country_field} field, will be ignored")
            gdf_clipped = gdf_clipped[~missing_mask]

        # Create geometry-value pairs for rasterization
        shapes = [(geom, value) for geom, value in zip(gdf_clipped.geometry, gdf_clipped['raster_value'])]

        print(f"Rasterization processing...")
        print(f"  Target dimensions: {reference_info['width']} × {reference_info['height']}")
        print(f"  Resolution: {reference_info['transform'][0]:.1f}m")

        # Execute rasterization
        raster_data = rasterize(
            shapes,
            out_shape=(reference_info['height'], reference_info['width']),
            transform=reference_info['transform'],
            fill=-9999,  # Background value
            dtype=np.int16,
            all_touched=False
        )

        # Statistics for rasterization result
        valid_mask = raster_data != -9999
        valid_count = np.sum(valid_mask)
        total_count = raster_data.size

        print(f"Rasterization statistics:")
        print(f"  Valid pixels: {valid_count:,}/{total_count:,} ({valid_count / total_count * 100:.1f}%)")

        if valid_count > 0:
            unique_values, counts = np.unique(raster_data[valid_mask], return_counts=True)
            print(f"  Number of countries in raster: {len(unique_values)}")
            # Show top 5 countries by area
            top_countries = sorted(zip(unique_values, counts), key=lambda x: x[1], reverse=True)[:5]
            for value, count in top_countries:
                country_name = [k for k, v in country_mapping.items() if v == value][0]
                print(f"    {value} ({country_name}): {count:,} pixels ({count / valid_count * 100:.1f}%)")

        # Prepare output configuration
        output_profile = reference_info['profile'].copy()
        output_profile.update({
            'dtype': rasterio.int16,
            'nodata': -9999,
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': min(512, reference_info['width']),
            'blockysize': min(512, reference_info['height'])
        })

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save raster
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(raster_data, 1)

            # Add metadata
            dst.update_tags(
                source_shapefile=os.path.basename(shapefile_path),
                country_field=country_field,
                country_count=len(unique_countries),
                valid_pixels=int(valid_count),
                total_pixels=int(total_count),
                processing_time=str(datetime.now() - start_time),
                description=f"Country codes rasterized from {country_field} field, cropped to reference"
            )

            # Add country mapping information (limit quantity to avoid oversized metadata)
            for country, code in list(country_mapping.items())[:50]:  # Only save first 50
                dst.update_tags(**{f"country_{code}": country})

        processing_time = datetime.now() - start_time
        file_size = os.path.getsize(output_path) / 1024 / 1024  # MB

        print(f"\nCountry code raster creation complete!")
        print(f"Processing time: {processing_time}")
        print(f"Output file: {output_path}")
        print(f"File size: {file_size:.1f} MB")

        return output_path, country_mapping

    except Exception as e:
        print(f"Failed to create country code raster: {str(e)}")
        raise


def main():
    """Main function"""

    # ============ Configuration Parameters ============

    # Reference raster data (species change data, 3km resolution)
    REFERENCE_RASTER_PATH = "data/species_analysis/species_fragmentation_change_1985_2020.tif"

    # Raster data to be cropped and aligned
    INPUT_RASTERS = [
        "data/land_cover/land_cover_1985_300m.tif",  # Land use (300m→3km, mode resampling)
        "data/human_footprint/human_footprint_1993.tif",  # Human footprint (average resampling)
        "data/accessibility/travel_time_to_cities.tif",  # Accessibility (average resampling)
        "data/forest_cover/hansen_treecover_2000.tif",  # Forest cover (average resampling)
        "data/elevation/dem_2024.tif"  # DEM (average resampling)
    ]

    # Output folder
    ALIGNED_OUTPUT_FOLDER = "data/psm_analysis/aligned_to_species_range"

    # Country shapefile and related parameters
    COUNTRY_SHAPEFILE_PATH = "data/boundaries/countries_boundaries.shp"
    COUNTRY_FIELD = "ADM0_A3_CN"  # Country code field
    COUNTRY_RASTER_OUTPUT = "data/psm_analysis/country_codes_aligned_to_species.tif"

    # ============ Execute Processing ============

    try:
        print("Raster Data Cropping and Alignment to Species Distribution Range")
        print("=" * 60)

        # 1. Read reference grid information (species change data)
        reference_info = get_reference_grid_info(REFERENCE_RASTER_PATH)

        # 2. Batch crop and align raster data
        print(f"\n{'=' * 60}")
        print("Starting batch cropping and alignment of raster data...")
        successful_rasters, failed_rasters = batch_crop_and_align_rasters(
            input_raster_paths=INPUT_RASTERS,
            reference_info=reference_info,
            output_folder=ALIGNED_OUTPUT_FOLDER,
            max_workers=4  # Can adjust based on CPU core count
        )

        # 3. Create country code raster (cropped to species range)
        print(f"\n{'=' * 60}")
        country_raster_path, country_mapping = create_country_raster_from_shapefile(
            shapefile_path=COUNTRY_SHAPEFILE_PATH,
            country_field=COUNTRY_FIELD,
            reference_info=reference_info,
            output_path=COUNTRY_RASTER_OUTPUT
        )

        # 4. Output processing results summary
        print(f"\n{'=' * 60}")
        print(f"All processing complete!")

        print(f"\nProcessing Results Summary:")
        print(f"  Reference raster: {os.path.basename(REFERENCE_RASTER_PATH)}")
        print(f"  Reference extent: {reference_info['bounds']}")
        print(f"  Reference resolution: {reference_info['transform'][0]:.1f}m")
        print(f"  Successfully processed: {len(successful_rasters)} raster files")
        print(f"  Failed: {len(failed_rasters)} files")
        print(f"  Output folder: {ALIGNED_OUTPUT_FOLDER}")

        print(f"\nCountry Code Raster:")
        print(f"  Output file: {country_raster_path}")
        print(f"  Number of countries: {len(country_mapping)}")

        if successful_rasters:
            print(f"\nSuccessfully aligned files:")
            for file_path in successful_rasters:
                print(f"  - {os.path.basename(file_path)}")

        if failed_rasters:
            print(f"\nFailed files:")
            for filename, error in failed_rasters:
                print(f"  - {filename}: {error}")

        print(f"\nNotes:")
        print(f"  • All rasters have been cropped to species distribution valid range")
        print(f"  • Land use data uses mode resampling to maintain categorical integrity")
        print(f"  • Continuous data uses average resampling to preserve numerical characteristics")
        print(f"  • Regions beyond species range (e.g., Arctic) have been removed")
        print(f"  • All output rasters have the same extent, resolution and grid alignment")

    except Exception as e:
        print(f"\nProcessing failed: {e}")
        raise


if __name__ == "__main__":
    main()