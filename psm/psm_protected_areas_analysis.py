import os
import numpy as np
import pandas as pd
import rasterio
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import time

warnings.filterwarnings('ignore')


def calc_single_var_pair_dist_std(args):
    """
    Calculate paired distance standard error for a single variable (for parallel processing)
    """
    var, pair_data_cache = args
    distances = []
    for pair_id, data in pair_data_cache.items():
        treatment_val = data['treatment'][var]
        control_val = data['control_mean'][var]
        distance = abs(treatment_val - control_val)
        distances.append(distance)
    return var, np.std(distances) / np.sqrt(len(distances)) if distances else 0


def check_balance(df_original, matched_df, fuzzy_vars):
    """
    Test covariate balance before and after matching (parallel version)
    """
    from scipy import stats
    from multiprocessing import Pool, cpu_count

    balance_results = {}
    all_vars = fuzzy_vars + ['country', 'landuse']

    # Pre-compute all pair data to avoid repeated calculations
    print("     Pre-computing pair data...")
    pair_data_cache = {}
    for pair_id in matched_df['pair_id'].unique():
        pair_data = matched_df[matched_df['pair_id'] == pair_id]
        treatment_data = pair_data[pair_data['match_type'] == 'treatment']
        control_data = pair_data[pair_data['match_type'] == 'control']

        if len(treatment_data) == 1 and len(control_data) > 0:
            # Get numeric columns (exclude non-numeric columns)
            numeric_columns = control_data.select_dtypes(include=[np.number]).columns.tolist()

            pair_data_cache[pair_id] = {
                'treatment': treatment_data.iloc[0],
                'control_mean': control_data[numeric_columns].mean()
            }

    # Parallel calculation of paired distance standard errors for each variable
    print("     Parallel calculation of paired distance standard errors...")
    n_cores = min(cpu_count(), 23)  # Use same core limit as main function

    with Pool(processes=n_cores) as pool:
        args_list = [(var, pair_data_cache) for var in all_vars]
        results = pool.map(calc_single_var_pair_dist_std, args_list)

    pair_dist_std_dict = dict(results)

    # Calculate other balance indicators
    for var in all_vars:
        # Balance before matching
        treatment_before = df_original[df_original['protected_area'] == 1][var]
        control_before = df_original[df_original['protected_area'] == 0][var]

        # Balance after matching
        treatment_after = matched_df[matched_df['match_type'] == 'treatment'][var]
        control_after = matched_df[matched_df['match_type'] == 'control'][var]

        # Calculate standardized mean difference and variance ratio
        def calc_smd(treatment, control):
            pooled_std = np.sqrt((treatment.var() + control.var()) / 2)
            return (treatment.mean() - control.mean()) / pooled_std if pooled_std > 0 else 0

        def calc_variance_ratio(treatment, control):
            return treatment.var() / control.var() if control.var() > 0 else np.inf

        # Calculate t-tests
        t_stat_before, p_val_before = stats.ttest_ind(treatment_before, control_before)
        t_stat_after, p_val_after = stats.ttest_ind(treatment_after, control_after)

        # Use pre-computed paired distance standard error
        pair_dist_std = pair_dist_std_dict[var]

        balance_results[var] = {
            'before': {
                'treatment_mean': treatment_before.mean(),
                'control_mean': control_before.mean(),
                'mean_diff': treatment_before.mean() - control_before.mean(),
                'std_diff': calc_smd(treatment_before, control_before),
                'variance_ratio': calc_variance_ratio(treatment_before, control_before),
                'treatment_std': treatment_before.std(),
                'control_std': control_before.std(),
                't_stat': t_stat_before,
                'p_value': p_val_before
            },
            'after': {
                'treatment_mean': treatment_after.mean(),
                'control_mean': control_after.mean(),
                'mean_diff': treatment_after.mean() - control_after.mean(),
                'std_diff': calc_smd(treatment_after, control_after),
                'variance_ratio': calc_variance_ratio(treatment_after, control_after),
                'treatment_std': treatment_after.std(),
                'control_std': control_after.std(),
                't_stat': t_stat_after,
                'p_value': p_val_after,
                'pair_dist_std': pair_dist_std  # Use parallel computed result
            }
        }

    return balance_results


def calculate_treatment_effect_stats(matched_df):
    """
    Calculate statistical test results for treatment effects
    """
    from scipy import stats

    # Get outcome variables for treatment and control groups
    treatment_outcomes = matched_df[matched_df['match_type'] == 'treatment']['species_fragmentation']
    control_outcomes = matched_df[matched_df['match_type'] == 'control']['species_fragmentation']

    # Calculate basic statistics
    treatment_mean = treatment_outcomes.mean()
    control_mean = control_outcomes.mean()
    difference = treatment_mean - control_mean

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(treatment_outcomes, control_outcomes)

    # Calculate standard error
    pooled_std = np.sqrt((treatment_outcomes.var() + control_outcomes.var()) / 2)
    n_treatment = len(treatment_outcomes)
    n_control = len(control_outcomes)
    standard_error = pooled_std * np.sqrt(1 / n_treatment + 1 / n_control)

    # Calculate confidence interval (95%)
    from scipy.stats import t as t_dist
    df = n_treatment + n_control - 2
    t_critical = t_dist.ppf(0.975, df)
    ci_lower = difference - t_critical * standard_error
    ci_upper = difference + t_critical * standard_error

    return {
        'treated_mean': treatment_mean,
        'control_mean': control_mean,
        'difference': difference,
        'standard_error': standard_error,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_treated': n_treatment,
        'n_control': n_control
    }


def process_group_matching(group_data, ratio=1, caliper=0.1, pair_id_start=0):
    """
    Process matching for a single group (for parallel processing)

    Parameters:
    -----------
    group_data : tuple
        Tuple containing (group_key, group_df)
    ratio : int
        Control:treatment matching ratio
    caliper : float
        Caliper parameter
    pair_id_start : int
        Starting value for pair IDs in this group

    Returns:
    --------
    tuple : (matched_data_list, stats_dict)
    """
    (country, landuse), group_df = group_data

    treatment_group = group_df[group_df['protected_area'] == 1].copy()
    control_group = group_df[group_df['protected_area'] == 0].copy()

    # Return empty result if no treatment or control group
    if len(treatment_group) == 0 or len(control_group) == 0:
        return [], {
            'group_key': (country, landuse),
            'treatment_count': len(treatment_group),
            'control_count': len(control_group),
            'matched_treatment': 0,
            'matched_control': 0,
            'unmatched_treatment': len(treatment_group)
        }

    matched_data = []
    matched_control_indices = set()
    pair_distances = []  # Collect pair distances

    pair_id = pair_id_start

    # Prepare KNN matching
    control_ps = control_group['propensity_score'].values.reshape(-1, 1)
    nbrs = NearestNeighbors(n_neighbors=min(ratio * 2, len(control_group)), algorithm='ball_tree')
    nbrs.fit(control_ps)

    # Match each treatment unit
    for idx, treatment_row in treatment_group.iterrows():
        treatment_ps = treatment_row['propensity_score']

        distances, indices = nbrs.kneighbors([[treatment_ps]])

        matched_controls = []
        for dist, ctrl_idx in zip(distances[0], indices[0]):
            actual_ctrl_idx = control_group.index[ctrl_idx]

            if dist <= caliper and actual_ctrl_idx not in matched_control_indices:
                matched_controls.append(actual_ctrl_idx)
                matched_control_indices.add(actual_ctrl_idx)
                pair_distances.append(dist)

                if len(matched_controls) >= ratio:
                    break

        if matched_controls:
            pair_id += 1

            # Add treatment group data
            treatment_row_copy = treatment_row.copy()
            treatment_row_copy['pair_id'] = pair_id
            treatment_row_copy['match_type'] = 'treatment'
            treatment_row_copy['ps_distance'] = 0.0
            matched_data.append(treatment_row_copy)

            # Add matched control group data
            for ctrl_idx in matched_controls:
                control_row = group_df.loc[ctrl_idx].copy()
                control_row['pair_id'] = pair_id
                control_row['match_type'] = 'control'
                control_row['ps_distance'] = abs(treatment_ps - control_row['propensity_score'])
                matched_data.append(control_row)

    # Statistics
    matched_treatment_count = len([d for d in matched_data if d['match_type'] == 'treatment'])
    matched_control_count = len([d for d in matched_data if d['match_type'] == 'control'])
    unmatched_treatment_count = len(treatment_group) - matched_treatment_count

    if pair_distances:
        mean_distance = np.mean(pair_distances)
        std_distance = np.std(pair_distances)  # Standard error
    else:
        mean_distance = 0
        std_distance = 0

    stats = {
        'group_key': (country, landuse),
        'treatment_count': len(treatment_group),
        'control_count': len(control_group),
        'matched_treatment': matched_treatment_count,
        'matched_control': matched_control_count,
        'unmatched_treatment': unmatched_treatment_count,
        'max_pair_id': pair_id,
        'pair_distances': pair_distances,
        'mean_pair_distance': mean_distance,
        'std_pair_distance': std_distance  # Standard error
    }

    return matched_data, stats


def perform_matching_parallel(df_with_ps, ratio=1, caliper=0.1, n_cores=None):
    """
    Perform propensity score matching in parallel

    Parameters:
    -----------
    df_with_ps : pandas.DataFrame
        Dataset with propensity scores
    ratio : int
        Control:treatment matching ratio, default 1:1
    caliper : float
        Caliper parameter controlling matching tolerance
    n_cores : int
        Number of CPU cores to use, None means use all available cores

    Returns:
    --------
    pandas.DataFrame : Matched dataset
    dict : Matching statistics
    """

    if n_cores is None:
        n_cores = min(cpu_count(), 23)  # Maximum 23 cores

    print(f"\n=== Parallel Propensity Score Matching (1:{ratio}) + Exact Matching ===")
    print(f"Using {n_cores} CPU cores for parallel processing")

    start_time = time.time()

    # Group by country and landuse
    grouped = df_with_ps.groupby(['country', 'landuse'])
    group_list = list(grouped)

    print(f"Total {len(group_list)} groups to process")

    # Assign pair_id starting values for each group to avoid duplicates
    pair_id_starts = []
    current_pair_id = 0
    for (key, group_df) in group_list:
        pair_id_starts.append(current_pair_id)
        # Estimate maximum possible pairs for this group (number of treatment units)
        treatment_count = len(group_df[group_df['protected_area'] == 1])
        current_pair_id += treatment_count + 100  # Add 100 as buffer

    # Create parallel processing function
    process_func = partial(process_group_matching, ratio=ratio, caliper=caliper)

    # Parallel processing
    print("Starting parallel processing...")
    with Pool(processes=n_cores) as pool:
        # Add pair_id_start parameter for each group
        args_list = []
        for i, group_data in enumerate(group_list):
            args_list.append((group_data, ratio, caliper, pair_id_starts[i]))

        # Use starmap to handle multiple parameters
        results = pool.starmap(process_group_matching, args_list)

    processing_time = time.time() - start_time
    print(f"Parallel processing completed in {processing_time:.2f} seconds")

    # Merge results
    print("Merging processing results...")
    all_pair_distances = []
    for _, stats in results:
        if 'pair_distances' in stats:
            all_pair_distances.extend(stats['pair_distances'])
    all_matched_data = []
    all_stats = []

    for matched_data, stats in results:
        all_matched_data.extend(matched_data)
        all_stats.append(stats)

        # Print each group's results
        key = stats['group_key']
        if stats['matched_treatment'] > 0:
            print(f"  Group country={key[0]}, landuse={key[1]}: "
                  f"treatment={stats['treatment_count']}, control={stats['control_count']}, "
                  f"matched={stats['matched_treatment']}/{stats['treatment_count']}")

    # Create final dataframe
    if all_matched_data:
        matched_df = pd.DataFrame(all_matched_data)
    else:
        matched_df = pd.DataFrame()

    # Calculate overall statistics
    total_stats = {
        'matched_treatment': sum(s['matched_treatment'] for s in all_stats),
        'matched_control': sum(s['matched_control'] for s in all_stats),
        'unmatched_treatment': sum(s['unmatched_treatment'] for s in all_stats),
        'total_groups_processed': len(all_stats),
        'successful_groups': len([s for s in all_stats if s['matched_treatment'] > 0]),
        'processing_time_seconds': processing_time,
        'cores_used': n_cores
    }

    if total_stats['matched_treatment'] > 0:
        total_stats['match_ratio'] = total_stats['matched_control'] / total_stats['matched_treatment']
        total_stats['treatment_match_rate'] = (total_stats['matched_treatment'] /
                                               (total_stats['matched_treatment'] + total_stats[
                                                   'unmatched_treatment'])) * 100

        if not matched_df.empty:
            control_distances = matched_df[matched_df['match_type'] == 'control']['ps_distance']
            total_stats['mean_ps_distance'] = control_distances.mean()
            total_stats['max_ps_distance'] = control_distances.max()
        else:
            total_stats['mean_ps_distance'] = 0
            total_stats['max_ps_distance'] = 0
    else:
        total_stats['match_ratio'] = 0
        total_stats['treatment_match_rate'] = 0
        total_stats['mean_ps_distance'] = 0
        total_stats['max_ps_distance'] = 0

    print(f"\n3. Parallel matching results:")
    print(f"   Groups processed: {total_stats['total_groups_processed']}")
    print(f"   Successfully matched groups: {total_stats['successful_groups']}")
    print(f"   Matched treatment units: {total_stats['matched_treatment']:,}")
    print(f"   Matched control units: {total_stats['matched_control']:,}")
    print(f"   Unmatched treatment units: {total_stats['unmatched_treatment']:,}")
    print(f"   Actual matching ratio: 1:{total_stats['match_ratio']:.1f}")
    print(f"   Treatment match rate: {total_stats['treatment_match_rate']:.1f}%")
    print(f"   Average propensity score difference: {total_stats['mean_ps_distance']:.4f}")
    print(f"   Maximum propensity score difference: {total_stats['max_ps_distance']:.4f}")
    print(f"   Parallel processing time: {processing_time:.2f} seconds")
    print(f"   CPU cores used: {n_cores}")

    # Calculate pair distance statistics
    if not matched_df.empty:
        pair_distances = matched_df[matched_df['match_type'] == 'control']['ps_distance']
        total_stats['mean_pair_distance'] = pair_distances.mean()
        total_stats['median_pair_distance'] = pair_distances.median()
        total_stats['max_pair_distance'] = pair_distances.max()

        print(f"   Average pair distance: {total_stats['mean_pair_distance']:.4f}")
        print(f"   Median pair distance: {total_stats['median_pair_distance']:.4f}")
        print(f"   Maximum pair distance: {total_stats['max_pair_distance']:.4f}")

    if all_pair_distances:
        total_stats['overall_mean_pair_distance'] = np.mean(all_pair_distances)
        total_stats['overall_std_pair_distance'] = np.std(all_pair_distances)  # Overall standard error
        total_stats['overall_median_pair_distance'] = np.median(all_pair_distances)

        print(f"   Overall pair distance statistics:")
        print(f"     Average distance: {total_stats['overall_mean_pair_distance']:.4f}")
        print(f"     Distance standard error: {total_stats['overall_std_pair_distance']:.4f}")
        print(f"     Median distance: {total_stats['overall_median_pair_distance']:.4f}")

    # Add balance testing
    if not matched_df.empty:
        print(f"\n4. Conducting balance tests...")
        balance_results = check_balance(df_with_ps, matched_df, fuzzy_vars=['dem', 'treecover', 'hfp', 'travel_time'])
        total_stats['balance_results'] = balance_results

        print("   Covariate balance before and after matching:")
        for var in balance_results:
            before = balance_results[var]['before']
            after = balance_results[var]['after']
            print(
                f"     {var}: pre-match difference={before['mean_diff']:.4f}, post-match difference={after['mean_diff']:.4f}")
    else:
        total_stats['balance_results'] = {}

    return matched_df, total_stats


def read_raster_data(file_path, variable_name):
    """
    Read raster data and intelligently detect NoData values
    """
    print(f"  Reading {variable_name}: {os.path.basename(file_path)}")

    with rasterio.open(file_path) as src:
        data = src.read(1, masked=False)

        # Get data type and NoData value
        dtype = src.dtypes[0]
        declared_nodata = src.nodata

        # Intelligently detect actual NoData values
        if declared_nodata is not None:
            actual_nodata = declared_nodata
        else:
            # Infer possible NoData values based on data type
            if dtype in ['uint8']:
                # Check common uint8 NoData values
                potential_nodata = [255, 0]
            elif dtype in ['uint16']:
                potential_nodata = [65535, 0]
            elif dtype in ['int16']:
                potential_nodata = [-32768, -9999, 32767]
            elif dtype in ['int32']:
                potential_nodata = [-2147483648, -9999, 2147483647]
            elif dtype in ['float32', 'float64']:
                potential_nodata = [-9999.0, -3.4028235e+38, np.nan]
            else:
                potential_nodata = [-9999]

            # Find most likely NoData value (extreme value with highest frequency)
            unique_values, counts = np.unique(data, return_counts=True)

            actual_nodata = None
            for candidate in potential_nodata:
                if not np.isnan(candidate):
                    if candidate in unique_values:
                        # Check if it's an extreme value with reasonable frequency
                        candidate_count = counts[unique_values == candidate][0]
                        total_count = data.size
                        # If proportion is between 0.1%-50%, might be NoData
                        if 0.001 <= candidate_count / total_count <= 0.5:
                            actual_nodata = candidate
                            break
                else:  # np.nan case
                    if np.any(np.isnan(data)):
                        actual_nodata = np.nan
                        break

            if actual_nodata is None:
                # Default to data type recommended value
                if dtype in ['uint8']:
                    actual_nodata = 255
                elif dtype in ['uint16']:
                    actual_nodata = 65535
                elif dtype in ['int16']:
                    actual_nodata = -32768
                elif dtype in ['int32']:
                    actual_nodata = -2147483648
                else:
                    actual_nodata = -9999.0

        meta = {
            'transform': src.transform,
            'crs': src.crs,
            'declared_nodata': declared_nodata,
            'actual_nodata': actual_nodata,
            'dtype': dtype,
            'shape': data.shape
        }

        # Count valid data
        if not np.isnan(actual_nodata):
            valid_mask = data != actual_nodata
        else:
            valid_mask = ~np.isnan(data)

        valid_count = np.sum(valid_mask)
        total_count = data.size

        print(f"    Data type: {dtype}")
        print(f"    Declared NoData: {declared_nodata}")
        print(f"    Actual NoData: {actual_nodata}")
        print(
            f"    Dimensions: {data.shape}, Valid pixels: {valid_count:,}/{total_count:,} ({valid_count / total_count * 100:.1f}%)")

        return data, meta


def prepare_psm_dataset(species_raster_path, protected_areas_raster_path,
                        dem_path, treecover_path, hfp_path, travel_path,
                        country_path, landuse_path):
    """
    Prepare PSM analysis dataset
    """
    print("=== Preparing PSM Analysis Dataset ===")

    # 1. Read all raster data
    print("\n1. Reading raster data...")

    data_info = {}

    species_data, species_meta = read_raster_data(species_raster_path, "Species landscape fragmentation")
    data_info['species'] = species_meta

    pa_data, pa_meta = read_raster_data(protected_areas_raster_path, "Protected areas")
    data_info['pa'] = pa_meta

    dem_data, dem_meta = read_raster_data(dem_path, "DEM")
    data_info['dem'] = dem_meta

    treecover_data, treecover_meta = read_raster_data(treecover_path, "Forest cover")
    data_info['treecover'] = treecover_meta

    hfp_data, hfp_meta = read_raster_data(hfp_path, "Human footprint")
    data_info['hfp'] = hfp_meta

    travel_data, travel_meta = read_raster_data(travel_path, "Accessibility")
    data_info['travel'] = travel_meta

    country_data, country_meta = read_raster_data(country_path, "Country codes")
    data_info['country'] = country_meta

    landuse_data, landuse_meta = read_raster_data(landuse_path, "Land use")
    data_info['landuse'] = landuse_meta

    # 2. Check data shape consistency
    print("\n2. Checking data consistency...")
    shapes = [data.shape for data in [species_data, pa_data, dem_data, treecover_data,
                                      hfp_data, travel_data, country_data, landuse_data]]

    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError("All raster data must have consistent shapes")

    print(f"   All raster shapes consistent: {shapes[0]}")

    # 3. Create valid data mask
    print("\n3. Creating valid data mask...")

    # Use actually detected NoData values
    print("   Detected NoData values:")
    for var_name, meta in data_info.items():
        print(f"     {var_name}: {meta['actual_nodata']} (type: {meta['dtype']})")

    # Create valid masks for each variable
    valid_masks = {}

    # Species data
    species_nodata = data_info['species']['actual_nodata']
    if not np.isnan(species_nodata):
        valid_masks['species'] = species_data != species_nodata
    else:
        valid_masks['species'] = ~np.isnan(species_data)

    # Protected areas data (special handling: only 0 and 1 values)
    pa_nodata = data_info['pa']['actual_nodata']
    if not np.isnan(pa_nodata):
        valid_masks['pa'] = (pa_data != pa_nodata) & np.isin(pa_data, [0, 1])
    else:
        valid_masks['pa'] = (~np.isnan(pa_data)) & np.isin(pa_data, [0, 1])

    # DEM data
    dem_nodata = data_info['dem']['actual_nodata']
    if not np.isnan(dem_nodata):
        valid_masks['dem'] = dem_data != dem_nodata
    else:
        valid_masks['dem'] = ~np.isnan(dem_data)

    # Forest cover data
    treecover_nodata = data_info['treecover']['actual_nodata']
    if not np.isnan(treecover_nodata):
        valid_masks['treecover'] = treecover_data != treecover_nodata
    else:
        valid_masks['treecover'] = ~np.isnan(treecover_data)

    # Human footprint data
    hfp_nodata = data_info['hfp']['actual_nodata']
    if not np.isnan(hfp_nodata):
        valid_masks['hfp'] = hfp_data != hfp_nodata
    else:
        valid_masks['hfp'] = ~np.isnan(hfp_data)

    # Accessibility data
    travel_nodata = data_info['travel']['actual_nodata']
    if not np.isnan(travel_nodata):
        valid_masks['travel'] = travel_data != travel_nodata
    else:
        valid_masks['travel'] = ~np.isnan(travel_data)

    # Country codes data
    country_nodata = data_info['country']['actual_nodata']
    if not np.isnan(country_nodata):
        valid_masks['country'] = country_data != country_nodata
    else:
        valid_masks['country'] = ~np.isnan(country_data)

    # Land use data
    landuse_nodata = data_info['landuse']['actual_nodata']
    if not np.isnan(landuse_nodata):
        valid_masks['landuse'] = landuse_data != landuse_nodata
    else:
        valid_masks['landuse'] = ~np.isnan(landuse_data)

    # Combined valid mask (pixels where all variables are valid)
    combined_valid_mask = np.logical_and.reduce(list(valid_masks.values()))

    valid_pixel_count = np.sum(combined_valid_mask)
    total_pixel_count = combined_valid_mask.size

    print(
        f"   Pixels valid for all variables: {valid_pixel_count:,}/{total_pixel_count:,} ({valid_pixel_count / total_pixel_count * 100:.1f}%)")

    if valid_pixel_count == 0:
        raise ValueError("No pixels are valid for all variables")

    # 4. Extract valid pixel data
    print("\n4. Extracting valid pixel data...")

    # Get row and column indices of valid pixels
    valid_rows, valid_cols = np.where(combined_valid_mask)

    # Create dataframe
    df = pd.DataFrame({
        'row': valid_rows,
        'col': valid_cols,
        'species_fragmentation': species_data[combined_valid_mask],
        'protected_area': pa_data[combined_valid_mask],
        'dem': dem_data[combined_valid_mask],
        'treecover': treecover_data[combined_valid_mask],
        'hfp': hfp_data[combined_valid_mask],
        'travel_time': travel_data[combined_valid_mask],
        'country': country_data[combined_valid_mask].astype(int),
        'landuse': landuse_data[combined_valid_mask].astype(int)
    })

    print(f"   Extracted pixel data: {len(df):,} rows")

    # 5. Analyze treatment and control groups
    print("\n5. Analyzing treatment and control groups...")

    treatment_count = np.sum(df['protected_area'] == 1)
    control_count = np.sum(df['protected_area'] == 0)

    print(f"   Treatment group (protected areas=1): {treatment_count:,} pixels")
    print(f"   Control group (protected areas=0): {control_count:,} pixels")
    print(f"   Control/treatment ratio: {control_count / treatment_count:.1f}:1")

    nodata_values = {var: meta['actual_nodata'] for var, meta in data_info.items()}

    # 6. Dataset metadata
    meta_info = {
        'total_pixels': len(df),
        'treatment_pixels': treatment_count,
        'control_pixels': control_count,
        'ratio': control_count / treatment_count if treatment_count > 0 else 0,
        'raster_shape': shapes[0],
        'nodata_values': nodata_values
    }

    print(f"\nDataset preparation complete")

    return df, meta_info


def calculate_propensity_scores(df):
    """
    Calculate propensity scores
    """
    print("\n=== Calculating Propensity Scores ===")

    fuzzy_vars = ['dem', 'treecover', 'hfp', 'travel_time']
    exact_vars = ['country', 'landuse']  # For exact matching, but not participating in score modeling

    print(f"\n1. Preparing covariates...")
    print(f"   Fuzzy matching variables: {fuzzy_vars}")
    print(f"   Exact matching variables: {exact_vars}")

    # Copy fuzzy matching variables and standardize
    X_fuzzy = df[fuzzy_vars].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fuzzy)

    # Target variable
    y = df['protected_area']

    # Fit logistic regression model
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_scaled, y)

    # Get propensity scores
    propensity_scores = model.predict_proba(X_scaled)[:, 1]
    df_with_ps = df.copy()
    df_with_ps['propensity_score'] = propensity_scores

    print(
        f"\n2. Score modeling complete: average score = {propensity_scores.mean():.4f}, median = {np.median(propensity_scores):.4f}")

    return df_with_ps, model, scaler, fuzzy_vars


def calculate_att_for_pair(pair_data):
    """
    Calculate ATT for a single pair (for parallel processing)

    Parameters:
    -----------
    pair_data : tuple
        Contains (pair_id, pair_df) tuple

    Returns:
    --------
    dict : ATT result dictionary, returns None if calculation fails
    """
    pair_id, pair_df = pair_data

    treatment_data = pair_df[pair_df['match_type'] == 'treatment']
    control_data = pair_df[pair_df['match_type'] == 'control']

    # Check data validity
    if len(treatment_data) != 1 or len(control_data) == 0:
        return None

    # Calculate treatment group outcome
    treatment_outcome = treatment_data['species_fragmentation'].iloc[0]

    # Calculate control group average outcome
    control_outcome_mean = control_data['species_fragmentation'].mean()

    # Calculate ATT
    att = treatment_outcome - control_outcome_mean

    # Get treatment group location information
    treatment_row = treatment_data.iloc[0]

    return {
        'pair_id': pair_id,
        'row': treatment_row['row'],
        'col': treatment_row['col'],
        'treatment_outcome': treatment_outcome,
        'control_outcome_mean': control_outcome_mean,
        'att': att,
        'n_controls': len(control_data),
        'propensity_score': treatment_row['propensity_score']
    }


def calculate_att_by_pair_parallel(matched_df, n_cores=None):
    """
    Calculate Average Treatment Effect on the Treated (ATT) for each pair in parallel

    Parameters:
    -----------
    matched_df : pandas.DataFrame
        Matched dataset
    n_cores : int
        Number of CPU cores to use, None means use all available cores

    Returns:
    --------
    pandas.DataFrame : Dataset containing ATT for each pair
    """

    if n_cores is None:
        n_cores = min(cpu_count(), 23)  # Maximum 23 cores

    print(f"\n=== Parallel Paired ATT Calculation ===")
    print(f"Using {n_cores} CPU cores for parallel ATT calculation")

    start_time = time.time()

    # Get all unique pair IDs
    unique_pair_ids = matched_df['pair_id'].unique()
    print(f"Total {len(unique_pair_ids):,} pairs need ATT calculation")

    # Prepare data for parallel processing
    pair_data_list = []
    for pair_id in unique_pair_ids:
        pair_df = matched_df[matched_df['pair_id'] == pair_id]
        pair_data_list.append((pair_id, pair_df))

    # Decide whether to use parallel processing based on data volume
    if len(pair_data_list) < 1000:
        print(f"Small number of pairs ({len(pair_data_list)}), using serial processing")
        # Serial processing
        att_results = []
        for pair_data in pair_data_list:
            result = calculate_att_for_pair(pair_data)
            if result is not None:
                att_results.append(result)
    else:
        print(f"Large number of pairs ({len(pair_data_list)}), using parallel processing")
        # Parallel processing
        with Pool(processes=n_cores) as pool:
            # Use map for parallel processing
            results = pool.map(calculate_att_for_pair, pair_data_list)

        # Filter out None results
        att_results = [result for result in results if result is not None]

    processing_time = time.time() - start_time

    # Create results DataFrame
    att_df = pd.DataFrame(att_results)

    print(f"   Parallel ATT calculation complete, time elapsed {processing_time:.2f} seconds")
    print(f"   Successfully calculated ATT for {len(att_df)} pairs")

    if len(att_df) > 0:
        print(f"   ATT statistics:")
        print(f"     Average ATT: {att_df['att'].mean():.6f}")
        print(f"     ATT standard deviation: {att_df['att'].std():.6f}")
        print(f"     ATT range: [{att_df['att'].min():.6f}, {att_df['att'].max():.6f}]")

        # Performance statistics
        pairs_per_second = len(att_df) / processing_time if processing_time > 0 else 0
        print(f"   Processing speed: {pairs_per_second:.0f} pairs/second")

        # Estimate speedup (assuming serial processing)
        if len(pair_data_list) >= 1000:
            estimated_serial_time = processing_time * n_cores
            speedup = estimated_serial_time / processing_time
            print(f"   Estimated speedup: {speedup:.1f}x")
    else:
        print("   Warning: No ATT values successfully calculated")

    return att_df


def create_att_raster(att_df, reference_raster_path, output_path, raster_shape):
    """
    Create ATT spatial distribution raster
    """
    print(f"\n=== Creating ATT Spatial Distribution Raster ===")

    # Read reference raster metadata
    with rasterio.open(reference_raster_path) as src:
        profile = src.profile.copy()

    # Create ATT raster array
    att_raster = np.full(raster_shape, -9999.0, dtype=np.float32)

    # Fill in ATT values
    for _, row in att_df.iterrows():
        r, c = int(row['row']), int(row['col'])
        att_raster[r, c] = row['att']

    # Update profile
    profile.update({
        'dtype': rasterio.float32,
        'nodata': -9999.0,
        'compress': 'lzw',
        'tiled': True
    })

    # Save raster
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(att_raster, 1)

        dst.update_tags(
            description="Average Treatment Effect (ATT) of protected areas on species fragmentation",
            att_mean=float(att_df['att'].mean()),
            att_std=float(att_df['att'].std()),
            n_treatment_pixels=len(att_df),
            processing_time=str(datetime.now())
        )

    print(f"   ATT raster saved: {output_path}")

    return output_path


def main():
    """Main function - Modified to use parallel matching"""

    # ============ Configuration Parameters ============

    # Input data paths
    SPECIES_RASTER = "data/species_analysis/species_fragmentation_change_1985_2020.tif"  # Dependent variable
    PROTECTED_AREAS_RASTER = "data/protected_areas/protected_areas_1985_2020_processed.tif"  # Treatment variable

    # Covariate paths
    DEM_PATH = "data/psm_analysis/aligned_to_species_range/dem_aligned_to_reference.tif"
    TREECOVER_PATH = "data/psm_analysis/aligned_to_species_range/hansen_treecover_2000_aligned_to_reference.tif"
    HFP_PATH = "data/psm_analysis/aligned_to_species_range/human_footprint_1993_aligned_to_reference.tif"
    TRAVEL_PATH = "data/psm_analysis/aligned_to_species_range/travel_time_to_cities_aligned_to_reference.tif"
    COUNTRY_PATH = "data/psm_analysis/aligned_to_species_range/country_codes_aligned_to_species.tif"
    LANDUSE_PATH = "data/psm_analysis/aligned_to_species_range/land_cover_1985_300m_aligned_to_reference.tif"

    # Output paths
    OUTPUT_FOLDER = "data/psm_analysis/results/protected_areas_effectiveness_analysis"
    ATT_RASTER_PATH = os.path.join(OUTPUT_FOLDER, "protected_areas_att_spatial.tif")

    # PSM parameters
    MATCH_RATIO = 1  # 1:1 matching
    CALIPER = 0.1  # Caliper parameter
    N_CORES = 16  # Use 16 CPU cores

    # ============ Execute PSM Analysis ============

    try:
        print("Protected Areas Effectiveness PSM Analysis (Parallel Version)")
        print("=" * 60)

        # 1. Prepare dataset
        df, meta_info = prepare_psm_dataset(
            SPECIES_RASTER, PROTECTED_AREAS_RASTER,
            DEM_PATH, TREECOVER_PATH, HFP_PATH, TRAVEL_PATH,
            COUNTRY_PATH, LANDUSE_PATH
        )
        print("Column names:", df.columns.tolist())
        print("Country non-null values count:", df['country'].notna().sum())
        print("Landuse non-null values count:", df['landuse'].notna().sum())
        print("Country unique values:", df['country'].unique())
        print("Landuse unique values:", df['landuse'].unique())

        # 2. Calculate propensity scores
        df_with_ps, ps_model, scaler, covariate_names = calculate_propensity_scores(df)

        # 3. Execute parallel matching
        matched_df, match_stats = perform_matching_parallel(
            df_with_ps,
            ratio=MATCH_RATIO,
            caliper=CALIPER,
            n_cores=N_CORES
        )

        # 4. Calculate paired ATT in parallel
        att_df = calculate_att_by_pair_parallel(matched_df, n_cores=N_CORES)

        # 5. Create ATT spatial distribution raster
        att_raster_path = create_att_raster(
            att_df, SPECIES_RASTER, ATT_RASTER_PATH, meta_info['raster_shape']
        )

        # 6. Save results
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        # Save balance test results
        if 'balance_results' in match_stats and match_stats['balance_results']:
            balance_path = os.path.join(OUTPUT_FOLDER, "balance_results.csv")
            balance_data = []

            for var, results in match_stats['balance_results'].items():
                before = results['before']
                after = results['after']
                balance_data.append({
                    'variable': var,
                    'treatment_mean_before': before['treatment_mean'],
                    'control_mean_before': before['control_mean'],
                    'mean_difference_before': before['mean_diff'],
                    'standardized_difference_before': before['std_diff'],
                    'variance_ratio_before': before['variance_ratio'],
                    't_stat_before': before['t_stat'],
                    'p_value_before': before['p_value'],
                    'treatment_mean_after': after['treatment_mean'],
                    'control_mean_after': after['control_mean'],
                    'mean_difference_after': after['mean_diff'],
                    'standardized_difference_after': after['std_diff'],
                    'variance_ratio_after': after['variance_ratio'],
                    'pair_distance_std': after['pair_dist_std'],
                    't_stat_after': after['t_stat'],
                    'p_value_after': after['p_value'],
                    'balance_improvement': abs(before['std_diff']) - abs(after['std_diff'])
                })

            balance_df = pd.DataFrame(balance_data)
            balance_df.to_csv(balance_path, index=False)
            print(f"   Balance test results: {balance_path}")

        # Calculate and save treatment effect statistical test results
        if not matched_df.empty:
            treatment_effect_stats = calculate_treatment_effect_stats(matched_df)

            # Save treatment effect results
            effect_path = os.path.join(OUTPUT_FOLDER, "treatment_effect_results.csv")
            effect_data = [{
                'sample_type': 'ATT',
                'treated_mean': treatment_effect_stats['treated_mean'],
                'control_mean': treatment_effect_stats['control_mean'],
                'difference': treatment_effect_stats['difference'],
                'standard_error': treatment_effect_stats['standard_error'],
                't_statistic': treatment_effect_stats['t_stat'],
                'p_value': treatment_effect_stats['p_value'],
                'ci_lower_95': treatment_effect_stats['ci_lower'],
                'ci_upper_95': treatment_effect_stats['ci_upper'],
                'n_treated': treatment_effect_stats['n_treated'],
                'n_control': treatment_effect_stats['n_control']
            }]

            effect_df = pd.DataFrame(effect_data)
            effect_df.to_csv(effect_path, index=False)
            print(f"   Treatment effect statistical test results: {effect_path}")

        # Save ATT data
        att_csv_path = os.path.join(OUTPUT_FOLDER, "att_results.csv")
        att_df.to_csv(att_csv_path, index=False)

        # Save matched data
        matched_csv_path = os.path.join(OUTPUT_FOLDER, "matched_dataset.csv")
        matched_df.to_csv(matched_csv_path, index=False)

        # Save statistical summary
        summary_path = os.path.join(OUTPUT_FOLDER, "psm_analysis_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Protected Areas Effectiveness PSM Analysis Summary (Parallel Version)\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Analysis time: {datetime.now()}\n\n")

            f.write("Dataset information:\n")
            f.write(f"  Total pixels: {meta_info['total_pixels']:,}\n")
            f.write(f"  Treatment pixels: {meta_info['treatment_pixels']:,}\n")
            f.write(f"  Control pixels: {meta_info['control_pixels']:,}\n\n")

            f.write("Parallel processing information:\n")
            f.write(f"  CPU cores used: {match_stats['cores_used']}\n")
            f.write(f"  Groups processed: {match_stats['total_groups_processed']}\n")
            f.write(f"  Successfully matched groups: {match_stats['successful_groups']}\n")
            f.write(f"  Parallel processing time: {match_stats['processing_time_seconds']:.2f} seconds\n\n")

            if 'mean_pair_distance' in match_stats:
                f.write("Pair distance statistics:\n")
                f.write(f"  Average pair distance: {match_stats['mean_pair_distance']:.4f}\n")
                f.write(f"  Median pair distance: {match_stats['median_pair_distance']:.4f}\n")
                f.write(f"  Maximum pair distance: {match_stats['max_pair_distance']:.4f}\n\n")

            f.write("Matching results:\n")
            f.write(f"  Matched treatment units: {match_stats['matched_treatment']:,}\n")
            f.write(f"  Matched control units: {match_stats['matched_control']:,}\n")
            f.write(f"  Matching ratio: 1:{match_stats['match_ratio']:.1f}\n")
            f.write(f"  Treatment match rate: {match_stats['treatment_match_rate']:.1f}%\n\n")

            f.write("ATT results:\n")
            f.write(f"  Average ATT: {att_df['att'].mean():.6f}\n")
            f.write(f"  ATT standard deviation: {att_df['att'].std():.6f}\n")
            f.write(f"  ATT range: [{att_df['att'].min():.6f}, {att_df['att'].max():.6f}]\n\n")

            if not matched_df.empty:
                treatment_effect_stats = calculate_treatment_effect_stats(matched_df)
                f.write("Treatment effect statistical test:\n")
                f.write(f"  Treatment group mean: {treatment_effect_stats['treated_mean']:.6f}\n")
                f.write(f"  Control group mean: {treatment_effect_stats['control_mean']:.6f}\n")
                f.write(f"  Treatment effect (ATT): {treatment_effect_stats['difference']:.6f}\n")
                f.write(f"  Standard error: {treatment_effect_stats['standard_error']:.6f}\n")
                f.write(f"  t-statistic: {treatment_effect_stats['t_stat']:.3f}\n")
                f.write(f"  p-value: {treatment_effect_stats['p_value']:.6f}\n")
                f.write(
                    f"  95% confidence interval: [{treatment_effect_stats['ci_lower']:.6f}, {treatment_effect_stats['ci_upper']:.6f}]\n\n")

            if 'balance_results' in match_stats and match_stats['balance_results']:
                f.write("Balance test results:\n")
                for var, results in match_stats['balance_results'].items():
                    before = results['before']
                    after = results['after']
                    f.write(f"  {var}:\n")
                    f.write(f"    Pre-match standardized difference: {before['std_diff']:.4f}\n")
                    f.write(f"    Post-match standardized difference: {after['std_diff']:.4f}\n")
                    f.write(f"    Pre-match variance ratio: {before['variance_ratio']:.4f}\n")
                    f.write(f"    Post-match variance ratio: {after['variance_ratio']:.4f}\n")
                    f.write(f"    Match distance standard error: {after['pair_dist_std']:.4f}\n")
                    f.write(f"    Pre-match p-value: {before['p_value']:.6f}\n")
                    f.write(f"    Post-match p-value: {after['p_value']:.6f}\n")
                    improvement = abs(before['std_diff']) - abs(after['std_diff'])
                    f.write(f"    Balance improvement: {improvement:.4f}\n")
                f.write("\n")

        print(f"\nParallel PSM analysis complete!")
        print(f"\nMain results:")
        print(f"  Average Treatment Effect (ATT): {att_df['att'].mean():.6f}")
        print(f"  Successfully matched protected areas: {len(att_df):,}")
        print(f"  Matching ratio: 1:{match_stats['match_ratio']:.1f}")
        print(f"  Parallel processing time: {match_stats['processing_time_seconds']:.2f} seconds")
        print(f"  CPU cores used: {match_stats['cores_used']}")

        print(f"\nOutput files:")
        print(f"  ATT spatial distribution raster: {att_raster_path}")
        print(f"  ATT data table: {att_csv_path}")
        print(f"  Matched dataset: {matched_csv_path}")
        print(f"  Analysis summary: {summary_path}")

        print(f"\nResult interpretation:")
        if att_df['att'].mean() < 0:
            print(f"  Protected areas have positive effect on reducing species landscape fragmentation (ATT < 0)")
        else:
            print(f"  Protected areas have positive effect on species landscape fragmentation (ATT > 0)")
        print(f"  Can use ATT spatial raster for mapping analysis")

        # Performance improvement calculation
        sequential_time_estimate = match_stats['processing_time_seconds'] * match_stats['cores_used']
        speedup = sequential_time_estimate / match_stats['processing_time_seconds']
        print(f"\nPerformance improvement:")
        print(f"  Estimated serial processing time: {sequential_time_estimate:.1f} seconds")
        print(f"  Actual parallel processing time: {match_stats['processing_time_seconds']:.1f} seconds")
        print(f"  Speedup ratio: {speedup:.1f}x")

    except Exception as e:
        print(f"\nPSM analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Set multiprocessing start method (recommended for Windows)
    import multiprocessing

    multiprocessing.set_start_method('spawn', force=True)
    main()