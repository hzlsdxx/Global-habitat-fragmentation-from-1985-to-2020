# Habitat Fragmentation Analysis Pipeline

A Python and R framework for analyzing habitat fragmentation patterns, environmental drivers, and protected area effectiveness using geospatial and statistical methods. The driver analysis uses an R spatial-block Gaussian generalized linear mixed model (GLMM).

## Overview

This pipeline provides end-to-end analysis capabilities for understanding habitat fragmentation impacts on biodiversity through four integrated analytical components:

1. **Fragmentation Analysis** - Calculate habitat indices and species-specific fragmentation metrics
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

Calculate habitat fragmentation indices and aggregate species-level metrics.

### Scripts Overview

#### `species_landscape_indices.py`
- **Purpose**: Calculate habitat indices (PD, ED, AREA_MN, MESH) for individual species
- **Input**: Species distribution data, DEM, land cover
- **Output**: Species-specific habitat index rasters
- **Key Features**: Parallel processing, habitat filtering, altitude constraints

#### `fragmentation_indices_calculator.py`  
- **Purpose**: Create normalized composite fragmentation indices per species
- **Input**: Individual species landscape index files
- **Output**: Standardized MIX (composite) indices
- **Key Features**: Global normalization, batch processing

#### `global_mix_aggregator.py`
- **Purpose**: Aggregate all species into global-scale fragmentation maps
- **Input**: Species-specific MIX files  
- **Output**: Global mean fragmentation raster (3km resolution)
- **Key Features**: Memory-efficient processing, 3km grid standardization

#### `species_richness_calculator.py`
- **Purpose**: Calculate species richness from individual species distributions
- **Input**: Species-specific MIX files
- **Output**: Species count raster showing biodiversity patterns
- **Key Features**: Binary aggregation, parallel batch processing

### Usage Example
```bash
# Step 1: Calculate individual species indices
python fragmentation/species_landscape_indices.py

# Step 2: Create normalized composite indices  
python fragmentation/fragmentation_indices_calculator.py

# Step 3: Aggregate to global scale
python fragmentation/global_mix_aggregator.py

# Step 4: Calculate species richness
python fragmentation/species_richness_calculator.py
```

### Key Parameters
- **Window size**: 3000m (3km resolution for habitat indices)
- **Pixel size**: 300m (input land cover resolution)
- **Parallel workers**: Configurable (default: CPU cores - 1)

### Output Files
- `{species_id}_PD.tif` - Patch density
- `{species_id}_ED.tif` - Edge density  
- `{species_id}_AREA_MN.tif` - Mean patch area
- `{species_id}_MESH.tif` - Effective mesh size
- `{species_id}_MIX.tif` - Composite fragmentation index
- Global aggregation and species richness rasters

## 2. GLMM Analysis (R)

Estimate associations between composite habitat fragmentation change and six environmental and anthropogenic predictors using `glmmTMB`. The R script `glm/glmm_fragmentation_analysis.R` replaces the former Python GLM script. The `glm/` directory name is retained.

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
