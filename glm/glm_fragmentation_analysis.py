import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ----------------------------
# 0. Read first band raster metadata for reference
# ----------------------------
tif_paths = [
    "data/glm_analysis/fire_frequency.tif",
    "data/glm_analysis/drought_frequency.tif",
    "data/glm_analysis/global_temperature_trends.tif",
    "data/glm_analysis/global_precipitation_trends.tif",
    "data/glm_analysis/population_difference_3km.tif",
    "data/glm_analysis/nightlight_change.tif",
    "data/glm_analysis/species_fragmentation_change_1985_2020.tif"  # Response variable
]

with rasterio.open(tif_paths[0]) as src0:
    ref_transform = src0.transform
    ref_crs = src0.crs
    ref_height = src0.height
    ref_width = src0.width

# ----------------------------
# 1. Load raster data and convert to np.nan
# ----------------------------
data_arrays = []
for path in tif_paths[:-1]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        data_arrays.append(arr)

with rasterio.open(tif_paths[-1]) as src:
    y_arr = src.read(1).astype(np.float32)
    if src.nodata is not None:
        y_arr[y_arr == src.nodata] = np.nan

# ----------------------------
# 2. Use shapefile mask only (no longer filtering for value==3 regions)
# ----------------------------
shp_path = "data/boundaries/study_area_boundaries.shp"
gdf = gpd.read_file(shp_path).to_crs(ref_crs)
mask_arr = rasterize(
    [(geom, 1) for geom in gdf.geometry],
    out_shape=(ref_height, ref_width),
    transform=ref_transform,
    fill=0,
    dtype='uint8'
).astype(bool)

# ----------------------------
# 3. Create comprehensive valid mask & extract samples
# ----------------------------
stacked = np.stack(data_arrays, axis=-1)  # (H, W, C)
data_valid_mask = np.all(~np.isnan(stacked), axis=-1)  # All bands have values
y_valid_mask = ~np.isnan(y_arr)  # Response variable is valid
final_mask = data_valid_mask & mask_arr & y_valid_mask  # Comprehensive mask

if final_mask.sum() == 0:
    raise ValueError("No pixels meet the criteria, please check shapefile and rasters.")

samples = stacked[final_mask]
y_data = y_arr[final_mask]
print(f"Total pixels before filtering: {mask_arr.sum()}")
print(f"Valid pixels for predictor variables: {data_valid_mask.sum()}")
print(f"Valid pixels for response variable: {y_valid_mask.sum()}")
print(f"Final valid pixels: {final_mask.sum()}")
print(f"Using all {samples.shape[0]} points for modeling")

# ----------------------------
# 4. Build DataFrame
# ----------------------------
cols = ['fire_freq', 'drought_freq', 'temp_change', 'precip_change', 'pop_change', 'light_change']
df = pd.DataFrame(samples, columns=cols)
df['fragmentation'] = y_data

X = df[cols]
y = df['fragmentation']

# ----------------------------
# 5. Data cleaning and validity check
# ----------------------------
# Check and remove samples containing NaN or infinite values
print(f"Sample count before data cleaning: {len(df)}")

# Check for NaN or infinite values
nan_mask = df.isnull().any(axis=1)
inf_mask = np.isinf(df.select_dtypes(include=[np.number])).any(axis=1)
invalid_mask = nan_mask | inf_mask

if invalid_mask.sum() > 0:
    print(f"Found {invalid_mask.sum()} invalid samples (containing NaN or infinite values), will be removed")
    df_clean = df[~invalid_mask].copy()
else:
    df_clean = df.copy()

print(f"Sample count after data cleaning: {len(df_clean)}")

if len(df_clean) == 0:
    raise ValueError("No valid samples after cleaning, please check data quality.")

X = df_clean[cols]
y = df_clean['fragmentation']

# Re-check data validity
print(f"Number of NaNs in predictor variables: {X.isnull().sum().sum()}")
print(f"Number of NaNs in response variable: {y.isnull().sum()}")
print(f"Number of infinite values in predictor variables: {np.isinf(X).sum().sum()}")
print(f"Number of infinite values in response variable: {np.isinf(y).sum()}")

# ----------------------------
# 6. Standardize features and response variable
# ----------------------------
# Standardize X (predictor variables)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=cols, index=X.index)

# Standardize y (response variable)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

print(f"Predictor variable standardization complete, response variable standardization complete\n")

# ----------------------------
# 7. GLM fitting (using standardized X and y, with intercept)
# ----------------------------
# Final check: ensure no NaN or infinite values
if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
    raise ValueError("Standardized predictor variables still contain NaN or infinite values")
if np.any(np.isnan(y_scaled)) or np.any(np.isinf(y_scaled)):
    raise ValueError("Standardized response variable still contains NaN or infinite values")

X_const = sm.add_constant(X_scaled)
try:
    glm_model = sm.GLM(y_scaled, X_const, family=sm.families.Gaussian()).fit()
    print("GLM model fitting successful!")
except Exception as e:
    print(f"GLM fitting failed: {e}")
    print("Data statistics:")
    print(f"X_scaled shape: {X_scaled.shape}")
    print(f"y_scaled shape: {y_scaled.shape}")
    print(f"X_scaled descriptive statistics:\n{X_scaled.describe()}")
    print(f"y_scaled descriptive statistics: mean={np.mean(y_scaled):.4f}, std={np.std(y_scaled):.4f}")
    raise

# Output model summary
print("GLM model fitting results:")
print("=" * 50)
print(glm_model.summary())

# ----------------------------
# 8. Plot coefficient chart (bar chart with error bars)
# ----------------------------
# Extract model parameters (excluding intercept)
coefficients = glm_model.params[1:]  # Exclude intercept
std_errors = glm_model.bse[1:]  # Exclude intercept standard errors
p_values = glm_model.pvalues[1:]  # Exclude intercept p-values

# Variable names (English)
var_names = [
    'Fire frequency',
    'Drought frequency',
    'Temperature change',
    'Precipitation change',
    'Population change',
    'Nighttime light change'
]

# Define colors
colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#1abc9c', '#9b59b6']

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot coefficients and standard errors for each variable
y_positions = np.arange(len(var_names))

for i, (coef, se, color, p_val) in enumerate(zip(coefficients, std_errors, colors, p_values)):
    # Calculate 95% confidence interval
    ci_95_low = coef - 1.96 * se
    ci_95_high = coef + 1.96 * se

    # Determine if significant (p <= 0.1)
    is_significant = p_val <= 0.1

    if is_significant:
        # Significant variables: draw bar chart
        if coef >= 0:
            # Positive coefficient: draw bar from 0
            ax.barh(y_positions[i], coef, height=0.4, left=0,
                    color=color, alpha=0.8, edgecolor='white', linewidth=1)
        else:
            # Negative coefficient: draw from coefficient value to 0
            ax.barh(y_positions[i], -coef, height=0.4, left=coef,
                    color=color, alpha=0.8, edgecolor='white', linewidth=1)

    # Draw error bars for all variables (black)
    ax.plot([ci_95_low, ci_95_high], [y_positions[i], y_positions[i]],
            color='black', linewidth=2, alpha=0.8)

    # Add endpoints to error bars
    ax.plot([ci_95_low, ci_95_low], [y_positions[i] - 0.05, y_positions[i] + 0.05],
            color='black', linewidth=2, alpha=0.8)
    ax.plot([ci_95_high, ci_95_high], [y_positions[i] - 0.05, y_positions[i] + 0.05],
            color='black', linewidth=2, alpha=0.8)

# Add vertical reference line (x=0)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)

# Set y-axis labels
ax.set_yticks(y_positions)
ax.invert_yaxis()  # Invert y-axis so first variable is at top

ax.set_xlim(-0.05, 0.25)  # Set x-axis display from -0.05 to 0.25
ax.set_xticks(np.arange(-0.05, 0.26, 0.05))
ax.set_xticklabels([])  # Hide x-axis labels
ax.set_yticklabels([])  # Hide y-axis labels

# Adjust layout
plt.tight_layout()

# Export as SVG format
# plt.savefig("data/glm_results/coefficient_plot.svg", format='svg', dpi=300, bbox_inches='tight')

# Display figure
plt.show()

# ----------------------------
# 9. Print coefficient statistics table
# ----------------------------
print("\n" + "=" * 60)
print("GLM Coefficient Statistics Table")
print("=" * 60)
coef_df = pd.DataFrame({
    'Variable': var_names,
    'Coefficient': coefficients.values,
    'Std_Error': std_errors.values,
    'p_value': p_values.values,
    'CI_lower': (coefficients - 1.96 * std_errors).values,
    'CI_upper': (coefficients + 1.96 * std_errors).values
})
print(coef_df.round(4))