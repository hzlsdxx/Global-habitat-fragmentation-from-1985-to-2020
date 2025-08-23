# Habitat Fragmentation Analysis Pipeline

A comprehensive Python-based framework for analyzing habitat fragmentation patterns, environmental drivers, and protected area effectiveness using advanced geospatial and statistical methods.

## Overview

This pipeline provides end-to-end analysis capabilities for understanding habitat fragmentation impacts on biodiversity through four integrated analytical components:

1. **Fragmentation Analysis** - Calculate habitat indices and species-specific fragmentation metrics
2. **GLM Analysis** - Identify environmental and anthropogenic drivers of fragmentation change  
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
│   └── glm_fragmentation_analysis.py
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
```bash
pip install rasterio numpy pandas geopandas scikit-learn statsmodels matplotlib seaborn
pip install multiprocessing functools datetime warnings tempfile shutil scipy
```

### System Requirements
- Python 3.8+
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

## 2. GLM Analysis

Identify environmental and anthropogenic drivers of habitat fragmentation change using Generalized Linear Models.

### Scripts Overview

#### `glm_fragmentation_analysis.py`
- **Purpose**: Analyze drivers of fragmentation change using GLM
- **Input**: Fragmentation change data, environmental predictors
- **Output**: Statistical models, coefficient plots, significance tests
- **Key Features**: Variable standardization, spatial masking, comprehensive diagnostics

### Usage Example
```bash
# Run GLM analysis
python glm/glm_fragmentation_analysis.py
```

### Input Variables
- **Response**: Species fragmentation change (1985-2020)
- **Predictors**: 
  - Fire frequency
  - Drought frequency
  - Temperature change
  - Precipitation change
  - Population change  
  - Nighttime light change

### Key Features
- Gaussian GLM with standardized variables
- 95% confidence intervals
- Significance testing (p ≤ 0.1)
- Coefficient visualization
- Comprehensive model diagnostics

### Output Files
- Statistical model summary
- Coefficient statistics table
- Visualization plots (optional SVG export)

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

 **Output**: Stacked visualization showing increase, decrease, and unchanged ratios
- **Key Features**:
  - Horizontal stacked bar chart format
  - Color-coded categories (Red: Increase, Blue: Decrease, Orange: Unchanged)
  - Statistical validation and data cleaning
  - SVG output format for high-quality publication

#### `create_simple_legend()`
- **Purpose**: Generate standalone color legend for fragmentation charts
- **Output**: Publication-ready legend with color coding
- **Features**: Clean, professional styling with customizable layout

#### `analyze_fragmentation_patterns()`
- **Purpose**: Comprehensive statistical analysis of fragmentation patterns
- **Input**: Multi-species CSV data
- **Output**: Statistical summaries, cross-species comparisons
- **Features**:
  - Mean, standard deviation, and median calculations
  - Species-wise and cross-species statistical comparisons
  - Sample size reporting and validation

#### `main_fragmentation_visualization()`
- **Purpose**: Execute complete visualization pipeline
- **Features**: 
  - Automated folder structure creation
  - Comprehensive output generation
  - Error handling and logging
  - Statistical summary file generation

### Usage Example
```bash
# Run complete visualization analysis
python visualization/visualization_analysis_suite.py
```

### Advanced Usage
```python
# Custom species analysis
species_csv_files = {
    "AMPH": "data/analysis_results/tables/AMPH_changes.csv",
    "BIRD": "data/analysis_results/tables/BIRD_changes.csv", 
    "MAMM": "data/analysis_results/tables/MAMM_changes.csv",
    "REPT": "data/analysis_results/tables/REPT_changes.csv"
}

# Execute with custom parameters
all_species_data, analysis_results = main_fragmentation_visualization(
    species_csv_files,
    output_folder="data/visualization_results",
    sample_size=10000  # Optional sampling for large datasets
)
```

### Input Data Requirements
Each species CSV file must contain the following columns:
- `Species_ID`: Unique identifier for each species record
- `Increase_Ratio`: Proportion of habitat showing increased fragmentation
- `Decrease_Ratio`: Proportion of habitat showing decreased fragmentation  
- `Unchanged_Ratio`: Proportion of habitat showing no fragmentation change

### Key Parameters
- **Sample size**: Configurable random sampling for large datasets (default: all data)
- **Color scheme**: 
  - Red (#E74C3C): Increase ratio
  - Blue (#3498DB): Decrease ratio
  - Orange (#faac61): Unchanged ratio
- **Output format**: SVG for high-quality, scalable graphics
- **Chart orientation**: Horizontal stacked bars for better species name readability

### Output Files
- `multi_species_habitat_fragmentation_stacked.svg` - Main comparative visualization
- `fragmentation_legend.svg` - Standalone color legend
- `fragmentation_analysis_summary.txt` - Comprehensive statistical summary
- Console output with detailed processing information and statistics

### Statistical Analysis Features
- **Data validation**: Automatic null value detection and removal
- **Sample size reporting**: Tracks data availability per species
- **Cross-species comparison**: Statistical summaries across all analyzed species
- **Quality control**: Validates required fields and data completeness
- **Error handling**: Graceful failure handling with informative error messages

### Visualization Features
- **Publication-ready quality**: High-resolution SVG output
- **Professional styling**: Clean, scientific visualization aesthetics
- **Customizable layout**: Adjustable figure sizes and spacing
- **Statistical annotations**: Comprehensive statistical reporting
- **Memory efficient**: Optimized for large dataset processing

## Performance Optimization

### Parallel Processing
All scripts support multi-core processing:
- **Default**: CPU cores - 1
- **Configurable**: Set `MAX_WORKERS` or `N_CORES` parameters
- **Memory management**: Batch processing for large datasets

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
- Model diagnostics for GLM
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
