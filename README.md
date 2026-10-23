# Habitat Fragmentation Analysis Pipeline

A Python and R framework for analyzing habitat fragmentation patterns, environmental drivers, and protected area effectiveness using geospatial and statistical methods. The driver analysis uses an R spatial-block Gaussian generalized linear mixed model (GLMM).

## Overview

This repository provides four analytical components for studying habitat fragmentation and biodiversity. The six-metric calculation starts from prepared species habitat rasters; the GLMM uses a separately prepared composite-change raster as its response.

1. **Fragmentation Analysis** - Binarize prepared species habitat rasters and calculate six habitat metrics
2. **GLMM Analysis (R)** - Estimate associations between fragmentation change and environmental and anthropogenic drivers with spatial-block random intercepts
3. **PSM Analysis** - Evaluate protected area effectiveness using propensity score matching
4. **Visualization Analysis** - Create comprehensive multi-species comparison visualizations

## Repository Structure

```
├── fragmentation/
│   ├── species_habitat_indices.py
│   ├── fragmentation_indices_calculator.py
│   ├── global_mix_aggregator.py
│   └── species_richness_calculator.py
├── glm/
│   └── glmm_fragmentation_analysis.R
├── psm/
│   ├── psm_protected_areas_analysis.py
│   └── psm_data_preprocessor.py
├── visualization/
│   └── visualization_analysis_suite.py
├── data/
│   ├── sp_data.gdb
│   ├── species_habitat/
│   ├── land_cover/
│   ├── species_analysis/
│   ├── habitat_indices/
│   ├── glm_analysis/
│   ├── psm_analysis/
│   ├── visualization_results/
│   └── boundaries/
└── README.md
```

## Requirements

### Dependencies

Install the dependencies for the Python components:

```bash
pip install -r requirements.txt
```

Install the R dependencies for the GLMM analysis in an R session:

```r
install.packages(c("terra", "glmmTMB"))
```

### System Requirements
- Python 3.8+
- R 4.1+ for the GLMM analysis
- Multi-core CPU (recommended: 16+ cores for parallel processing)
- RAM: 16GB+ (depends on dataset size)
- Storage: Sufficient space for intermediate and output files

## Data Requirements

### Input Data Structure
```
data/
├── sp_data.gdb
├── species_habitat/
│   ├── AMPH_1985_kc/
│   │   └── {species_id}_landcover.tif
│   ├── AMPH_2020_kc/
│   └── ... # BIRD, MA and RE folders for 1985 and 2020
├── species_analysis/
│   ├── species_distribution_data.gdb
│   └── species_fragmentation_change_1985_2020.tif
├── analysis_results/
│   └── tables/
│       ├── AMPH_changes.csv
│       ├── BIRD_changes.csv
│       ├── MAMM_changes.csv
│       └── REPT_changes.csv
├── land_cover/
│   ├── land_cover_1985_300m.tif
│   └── land_cover_2020_300m.tif
├── elevation/
│   └── dem_2024.tif
├── boundaries/
│   ├── countries_boundaries.shp
└── protected_areas/
    └── protected_areas_1985_2020_processed.tif
```

## Analysis Components

## 1. Fragmentation Analysis

Calculate six habitat metrics from prepared species-specific habitat rasters, construct composite indices, and aggregate species-level results.

### Scripts Overview

#### `species_habitat_indices.py`
- **Purpose**: Calculate CLUMPY, nLSI, ED, MESH, AREA_MN and PD for individual species.
- **Input**: Prepared `{species_id}_landcover.tif` rasters and a species geodatabase with the configured layers and an `id_no` field.
- **Output**: Six metric GeoTIFFs per species, grouped by animal group and year, plus JSON progress records.
- **Processing**: In-memory habitat binarization, parallel species processing and resumable batches for 1985 and 2020.

### Habitat Inputs and Binarization

The script takes species habitat rasters that have been filtered by species range, elevation limits and suitable land-cover classes. Complete this preprocessing before running the metric calculation. Every non-NoData class in the input is treated as habitat.

The default inputs are:

- `data/sp_data.gdb`: species layers configured in `TASKS`, with species identifiers in `id_no`.
- `data/species_habitat/{GROUP}_{YEAR}_kc/{species_id}_landcover.tif`: prepared habitat rasters, where `GROUP` is `AMPH`, `BIRD`, `MA` or `RE` and `YEAR` is `1985` or `2020`.
- `data/land_cover/land_cover_{YEAR}_300m.tif`: optional reference for the NoData value when a species raster does not define one. If neither raster supplies it, the fallback is `0`.

The script merges all non-NoData habitat classes into value `1` in memory and retains the NoData value as background. Thus adjacent suitable land-cover classes belong to the same habitat patch. No separate binary TIFF is written. The effective NoData value must be finite and different from `1`; input rasters must use a projected grid with 300 m pixels.

Configure `GDB_FILE`, `BASE_LC_DIR`, `LC_REF`, `TASKS`, `BASE_OUT_DIR` and `N_WORKERS` at the top of the script for your data. Bird tasks `BIRD_A`–`BIRD_D` share `BIRD` input/output folders; reptile tasks `RE_A` and `RE_B` share `RE` folders.

### Usage Example
```bash
# Run all configured tasks for both years, from the repository root
python fragmentation/species_habitat_indices.py

# Run one task for one year
python fragmentation/species_habitat_indices.py AMPH 2020
```

Command syntax is `python fragmentation/species_habitat_indices.py [TASK] [YEAR]`. Task choices are `AMPH`, `BIRD_A`, `BIRD_B`, `BIRD_C`, `BIRD_D`, `MA`, `RE_A` and `RE_B`; year choices are `1985` and `2020`. Omitting the year runs both years for the selected task; omitting both arguments runs every configured task.

### Key Parameters
- **Window size**: Non-overlapping 10 × 10 pixel windows (3 km × 3 km), starting at each input raster's upper-left corner. Partial windows at the raster edges use their actual cell counts, but their output cells remain 3 km wide and may extend beyond the source bounds. This stage does not align rasters to a common global 3 km grid origin or reproject them.
- **Pixel size**: 300 m; each cell covers 9 ha. Density and effective mesh-size denominators use the whole window area, including background cells.
- **Connectivity**: Eight-neighbour connectivity for habitat patches; horizontal and vertical shared edges for CLUMPY, nLSI and ED. ED counts internal habitat/background edges and excludes the outside boundary of the window.
- **Parallel workers**: Configurable with `N_WORKERS` (default: 16).
- **Undefined values**: No-habitat windows and undefined metrics use output NoData `-9999`. CLUMPY and nLSI are undefined for a fully occupied 10 × 10 window. Finite nLSI values, including those from partial edge windows, are clipped to `[0, 1]`.

### Output Files

The default output directory is `D:/hzl/code/outputs/six_habitat_indices/{GROUP}_{YEAR}`. Change `BASE_OUT_DIR` to select another location.

| File | Metric | Units |
| --- | --- | --- |
| `{species_id}_CLUMPY.tif` | Clumpiness | Dimensionless |
| `{species_id}_nLSI.tif` | Normalized landscape shape index | Dimensionless |
| `{species_id}_ED.tif` | Edge density | m/ha |
| `{species_id}_MESH.tif` | Effective mesh size | ha |
| `{species_id}_AREA_MN.tif` | Mean habitat-patch area | ha |
| `{species_id}_PD.tif` | Patch density | Patches per 100 ha |

Progress is recorded in `progress_{TASK}_{YEAR}.json`. The six-metric script does not write a composite MIX raster, group means or temporal differences.

### Composite and Aggregation Scripts

These scripts construct species-level MIX indices, aggregate global maps and count species coverage. Configure their input directories for the analysis.

| Script | Purpose |
| --- | --- |
| `fragmentation_indices_calculator.py` | Normalizes CLUMPY, nLSI, ED and PD using shared global min/max, reverses CLUMPY, and averages the four values into `{species_id}_MIX.tif` where all four are valid. |
| `global_mix_aggregator.py` | Aligns species MIX rasters to a reference 3 km grid and averages valid values. It does not calculate a 1985–2020 difference. |
| `species_richness_calculator.py` | Counts valid species MIX coverage on a reference grid. Its presence/coverage binarization is separate from the habitat binarization used before calculating the six metrics. |

### Species-Level Composite Index (MIX)

MIX combines **CLUMPY, nLSI, ED and PD** with equal weights. AREA_MN and MESH are standalone metrics. For each metric, `min_metric` and `max_metric` are computed from all finite, non-NoData values in its input rasters. The same pair is used for every species, animal group and year under the input root. Include both 1985 and 2020 in a single run when comparing years; running them separately would calculate different normalization ranges.

Input values are rounded to four decimal places and clipped to their metric's `[min_metric, max_metric]` range before normalization:

```text
scaled_metric = (metric - min_metric) / (max_metric - min_metric)
MIX = ((1 - scaled_CLUMPY) + scaled_nLSI + scaled_ED + scaled_PD) / 4
```

The min/max symbols represent values calculated from the input data; no fixed empirical extrema are required. Only CLUMPY is reversed, so larger MIX indicates greater fragmentation. A pixel is valid only when all four components are finite and valid; otherwise its output is NoData `-9999`. Valid zeros are preserved. An empty or constant normalization range (`max_metric <= min_metric`) produces an error.

```bash
# Read all groups and years under the six-metric output directory
python fragmentation/fragmentation_indices_calculator.py

# Select input/output roots while preserving the group/year subdirectories
python fragmentation/fragmentation_indices_calculator.py --input-dir D:/hzl/code/outputs/six_habitat_indices --output-dir D:/hzl/code/outputs/species_mix --workers 4
```

By default, the script reads `D:/hzl/code/outputs/six_habitat_indices` recursively and writes `{species_id}_MIX.tif` beside each species/year's input rasters. `--output-dir` selects a separate output root. The four inputs in each set must share dimensions, transform and CRS. Missing or duplicate metric files are reported as errors. Each species/year set is processed independently; cross-year species pairing and temporal differences are separate analysis steps.

Configure `global_mix_aggregator.py` and `species_richness_calculator.py` for one animal-group/year directory at a time. MIX coverage requires all four metrics to be defined, so its coverage count can exclude windows where habitat is present but CLUMPY or nLSI is undefined.

### Composite Response for the GLMM

The GLMM response is a separately prepared composite-change raster based on **CLUMPY, nLSI, ED and PD**. Its calculation first averages each metric across species within each animal group, then normalizes using shared species-level global min/max, reverses CLUMPY, combines the four metrics and averages across available groups. The composite change is **2020 minus 1985**; positive values represent increased fragmentation.

Provide `ALL_composite_diff_all.tif` in `data/glm_analysis/`, or configure another response path in the R script. The species-level MIX calculator writes individual composite rasters; generating this GLMM response requires the separate aggregation and temporal-difference steps. Averaging metrics before composition and averaging species MIX afterwards can give different results when valid-data masks or clipping differ.

## 2. GLMM Analysis (R)

Estimate associations between composite habitat fragmentation change and six environmental and anthropogenic predictors. The R script `glm/glmm_fragmentation_analysis.R` fits a spatial-block Gaussian generalized linear mixed model using `glmmTMB`.

### Inputs and Configuration

Run from the repository root. Configure `x_paths`, `y_configs`, and `region_configs` at the top of the script to match your data. The default raster paths are:

| Variable | File under `data/glm_analysis/` |
| --- | --- |
| Response: composite fragmentation change (1985–2020) | `ALL_composite_diff_all.tif` |
| Fire frequency | `fire_frequency.tif` |
| Drought frequency | `drought_frequency.tif` |
| Temperature change | `global_temperature_trends.tif` |
| Precipitation change | `global_precipitation_trends.tif` |
| Population change | `population_difference_3km.tif` |
| Nighttime light change | `nightlight_change.tif` |

The default regions are AF, AU, China, EU, NA, RUS, SA, SEA, and Global. Their boundary files under `data/boundaries/` are `AF_final.shp`, `AU.shp`, `China.shp`, `EU_final.shp`, `NA_final2.shp`, `RUS.shp`, `SA.shp`, `SEA.shp`, and `countries_boundaries.shp`, respectively. Include the associated shapefile files, such as `.shx`, `.dbf`, and `.prj`.

All predictor rasters must share a CRS and grid. The response raster must use the same CRS as the predictors; within each region it is cropped, bilinearly resampled to the predictor grid, and masked. Boundary vectors are reprojected to the predictor CRS when needed. Supply the input datasets locally before running the analysis.

### Usage Example

```bash
Rscript glm/glmm_fragmentation_analysis.R
```

The default output directory is `D:/hzl/code/outputs/GLMM_ALL_COMPOSITE`. Change the `folder` field in `y_configs` to use another location.

### Model and Processing

- The model uses a Gaussian family with an identity link, six fixed-effect predictors, and a random intercept for each 5° × 5° longitude/latitude block: `fragmentation ~ fire_freq + drought_freq + temp_change + precip_change + pop_change + light_change + (1 | block_id)`.
- Each region is cropped and masked separately. Rows with missing values are removed. If more than `MAX_OBS = 1000000` complete observations remain, the script samples up to that limit using seed 42.
- Blocks containing fewer than `MIN_BLOCK_N = 5` observations are removed **after sampling**. Regions with fewer than 30 remaining observations or fewer than 3 blocks are skipped.
- The response and all six predictors are standardized within each retained regional sample using R's `scale()` (sample standard deviation).
- Projected pixel coordinates are converted to longitude/latitude for block assignment. The blocks are angular cells, not equal-area units. `BLOCK_DEG` controls their size.
- Fixed-effect results include two-sided Wald z-test p-values and approximate 95% Wald confidence intervals (`estimate ± 1.96 × standard error`). The intercept is excluded from the reported coefficient table.

The block random intercept represents variation shared by observations in the same block. It does not guarantee removal of residual spatial autocorrelation. The script reports data coverage and sample counts to the console; it does not perform a complete convergence or residual diagnostic assessment.

### Output Files

`GLMM_ALL_COMPOSITE_results.txt` contains each successfully fitted region's six predictor coefficients, standard errors, p-values, confidence intervals, observation count, and block count. One TXT file is written per response configuration. The GLMM script generates no plots and saves no fitted model objects.

## 3. PSM Analysis  

Evaluate protected area effectiveness using Propensity Score Matching to address selection bias.

### Scripts Overview

#### `psm_data_preprocessor.py`
- **Purpose**: Preprocess and align environmental data for PSM analysis
- **Input**: Multiple environmental raster datasets
- **Output**: Spatially aligned, standardized datasets
- **Key Features**: Intelligent resampling, parallel processing, quality control

#### `psm_protected_areas_analysis.py`
- **Purpose**: Evaluate protected area effectiveness using PSM
- **Input**: Species fragmentation data, protected area boundaries, covariates
- **Output**: Average Treatment Effect (ATT) estimates, spatial maps
- **Key Features**: Parallel matching, balance testing, causal inference

### Usage Example
```bash
# Step 1: Preprocess data
python psm/psm_data_preprocessor.py

# Step 2: Run PSM analysis  
python psm/psm_protected_areas_analysis.py
```

### PSM Methodology
1. **Propensity Score Estimation**: Logistic regression on covariates
2. **Matching Strategy**: 1:1 nearest neighbor with caliper
3. **Exact Matching**: Country and land use categories
4. **Balance Testing**: Pre/post-matching covariate balance
5. **Effect Estimation**: Average Treatment Effect on Treated (ATT)

### Key Parameters
- **Matching ratio**: 1:1 (treatment:control)
- **Caliper**: 0.1 (propensity score tolerance)
- **Fuzzy variables**: DEM, tree cover, human footprint, accessibility
- **Exact variables**: Country, land use

### Output Files
- `att_results.csv` - Treatment effect estimates
- `matched_dataset.csv` - Matched treatment/control pairs
- `balance_results.csv` - Pre/post matching balance tests
- `protected_areas_att_spatial.tif` - Spatial ATT distribution
- `psm_analysis_summary.txt` - Comprehensive results

## 4. Visualization Analysis

Create comprehensive multi-species habitat fragmentation comparison visualizations and statistical analysis.

### Scripts Overview

#### `multi_species_visualization_suite.py`
- **Purpose**: Generate stacked bar chart visualizations for habitat fragmentation changes across multiple species
- **Input**: Species-specific CSV files with fragmentation change data
- **Output**: Horizontal stacked bar charts and statistical summaries
- **Key Features**: Multi-species comparison, statistical validation, customizable sampling

### Core Functions

#### `create_multi_species_habitat_fragmentation_stacked()`
- **Purpose**: Create horizontal stacked bar charts comparing habitat fragmentation changes across species
- **Input**: CSV files with `Species_ID`, `Increase_Ratio`, `Decrease_Ratio`, `Unchanged_Ratio` columns
- **Output**: Horizontal stacked bar visualization showing fragmentation patterns
- **Key Features**:
  - Color-coded categories (Red: Increase, Blue: Decrease, Orange: Unchanged)
  - Reversed species order for optimal readability
  - Statistical validation and data cleaning
  - Professional styling with publication-ready output
  - SVG format for high-quality scientific graphics

### Usage Example
```bash
# Run habitat fragmentation stacked analysis
python visualization/fragmentation_stacked_analysis.py
```

### Advanced Configuration
```python
# Example species data dictionary
species_csv_files = {
    "AMPH": "data/analysis_results/tables/AMPH_changes.csv",
    "BIRD": "data/analysis_results/tables/BIRD_changes.csv",
    "MAMM": "data/analysis_results/tables/MA_changes.csv",
    "REPT": "data/analysis_results/tables/RE_changes.csv"
}

# Execute with custom parameters
create_multi_species_habitat_fragmentation_stacked(
    species_csv_files,
    output_path="data/visualization_results/multi_species_fragmentation.svg",
    sample_size=10000  # Optional sampling for large datasets
)
```

### Input Data Requirements
- CSV files with required columns: `Species_ID`, `Increase_Ratio`, `Decrease_Ratio`, `Unchanged_Ratio`
- Ratio values should be numeric and represent proportions (typically between 0 and 1)
- Species identifier consistency across datasets
- Proper null value handling and data validation

### Key Features
- **Multi-species comparison**: Simultaneous visualization of fragmentation patterns across taxonomic groups
- **Publication quality**: SVG output format for high-resolution scientific graphics
- **Statistical validation**: Comprehensive data cleaning and quality control processes
- **Professional styling**: Clean layout with Arial font and scientific color schemes
- **Flexible sampling**: Configurable random sampling for large datasets
- **Cross-species analysis**: Statistical comparisons and summary calculations

### Statistical Analysis Features
- **Data validation**: Automatic null value detection and removal
- **Sample size reporting**: Tracks data availability and processing statistics per species
- **Cross-species comparison**: Mean, standard deviation calculations across all species
- **Quality control**: Validates required fields and data completeness
- **Comprehensive reporting**: Detailed console output with processing information

### Output Files
- `multi_species_habitat_fragmentation_stacked.svg` - Main horizontal stacked bar chart
- Console output with detailed statistical summaries for each species
- Cross-species comparison statistics (means, standard deviations)
- Processing information and data quality reports

## Performance Optimization

### Parallel Processing
Parallel processing is available in the Python components that expose worker settings:
- **Default**: CPU cores - 1
- **Configurable**: Set `MAX_WORKERS` or `N_CORES` parameters
- **Memory management**: Batch processing for large datasets

The GLMM script processes regions sequentially and does not configure parallel workers. Reduce `MAX_OBS` to shorten model fitting when needed; the full regional raster data frame is created before sampling, so this setting does not cap preprocessing memory use.

### Memory Efficiency
- Temporary file usage for large raster operations
- Streaming data processing
- Automatic cleanup of intermediate files
- Optional data sampling for visualization components

### Recommended Hardware
- **CPU**: 16+ cores for optimal performance
- **RAM**: 32GB+ for large study areas
- **Storage**: SSD recommended for I/O intensive operations

## Quality Control

### Data Validation
- Consistent spatial extent and resolution
- NoData value detection and handling
- Geometric validity checks
- Coverage percentage reporting
- Statistical validation for visualization components

### Statistical Validation
- Balance testing for PSM
- GLMM data-coverage and sample-count reporting; convergence and residual diagnostics require separate assessment
- Significance testing throughout
- Confidence interval estimation
- Cross-species statistical comparisons

## Troubleshooting

### Common Issues

**Memory Errors**
- Reduce batch size in processing functions
- Increase swap space
- Process smaller study areas
- Use sampling parameters in visualization functions

**Projection Mismatches**
- All input data must be in same CRS
- Use `psm_data_preprocessor.py` for alignment

**Missing Dependencies**
```bash
# Install required packages
pip install -r requirements.txt
# Additional visualization dependencies
pip install matplotlib seaborn scipy
```

**Performance Issues**
- Adjust parallel worker counts
- Monitor CPU and memory usage
- Use SSD storage for temporary files
- Enable data sampling for large visualization datasets

**Visualization Errors**
- Verify CSV column names match requirements
- Check for sufficient data after cleaning
- Ensure output directory permissions
- Validate species data completeness

---

**Note**: Update file paths and parameters according to your specific data structure and analysis requirements.
