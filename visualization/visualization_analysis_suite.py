import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')


def create_multi_species_habitat_fragmentation_stacked(species_csv_dict, output_path=None, sample_size=None):
    """
    Create stacked bar chart comparison of habitat fragmentation changes across multiple species

    Parameters:
    -----------
    species_csv_dict : dict
        Dictionary format: {"Species Name": "CSV file path", ...}
    output_path : str, optional
        Output image path
    sample_size : int, optional
        Random sample size for each group (if data is too large)
    """

    # Set plot style
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 11,
        'font.weight': 'normal'
    })

    # Create figure
    fig, ax = plt.subplots(figsize=(24, 8))

    species_names = list(species_csv_dict.keys())
    num_species = len(species_names)

    all_species_data = {}  # Store all species data

    print("Starting to process multiple species data...")
    print("=" * 80)

    for species_idx, (species_name, csv_path) in enumerate(species_csv_dict.items()):
        print(f"\nProcessing species: {species_name}")
        print("-" * 40)

        try:
            # 1. Read CSV data
            print("Reading CSV file...")
            df = pd.read_csv(csv_path)

            # Check required fields
            required_fields = ['Species_ID', 'Increase_Ratio', 'Decrease_Ratio', 'Unchanged_Ratio']
            missing_fields = [field for field in required_fields if field not in df.columns]
            if missing_fields:
                print(f"Warning: {species_name} missing required fields: {missing_fields}, skipping this species")
                continue

            print(f"Original data shape: {df.shape}")

            # 2. Data cleaning
            initial_count = len(df)
            df_clean = df.dropna(subset=['Increase_Ratio', 'Decrease_Ratio', 'Unchanged_Ratio']).copy()
            final_count = len(df_clean)
            print(
                f"After removing null values: {initial_count} -> {final_count} ({initial_count - final_count} records removed)")

            if len(df_clean) == 0:
                print(f"Warning: {species_name} has no valid data after cleaning, skipping")
                continue

            # 3. Data sampling (if needed)
            if sample_size and len(df_clean) > sample_size:
                df_clean = df_clean.sample(sample_size, random_state=42)
                print(f"Random sampling of {sample_size} samples")

            # 4. Calculate overall statistics for each species
            mean_increase = df_clean['Increase_Ratio'].mean()
            mean_decrease = df_clean['Decrease_Ratio'].mean()
            mean_unchanged = df_clean['Unchanged_Ratio'].mean()

            print(f"Species statistics:")
            print(f"  Average increase ratio: {mean_increase:.6f}")
            print(f"  Average decrease ratio: {mean_decrease:.6f}")
            print(f"  Average unchanged ratio: {mean_unchanged:.6f}")

            # Store current species statistical data
            all_species_data[species_name] = {
                'increase': mean_increase,
                'decrease': mean_decrease,
                'unchanged': mean_unchanged,
                'raw_data': df_clean
            }

        except Exception as e:
            print(f"Error processing species {species_name}: {e}")
            continue

    # 6. Start drawing stacked chart
    print("\nStarting to draw stacked chart...")

    # Set colors: increase-red, decrease-blue, unchanged-orange
    colors = ['#E74C3C', '#3498DB', '#faac61']  # Red, blue, orange

    # Prepare data
    species_list = list(reversed(all_species_data.keys()))
    increase_values = [all_species_data[species]['increase'] for species in species_list]
    decrease_values = [all_species_data[species]['decrease'] for species in species_list]
    unchanged_values = [all_species_data[species]['unchanged'] for species in species_list]

    # Set y-axis positions (y-coordinate is species)
    y_positions = np.arange(len(species_list))
    bar_height = 0.4

    # Create horizontal stacked bar chart (order: increase, decrease, unchanged)
    p1 = ax.barh(y_positions, increase_values, bar_height,
                 color=colors[0], alpha=0.8, label='Increase Ratio')

    p2 = ax.barh(y_positions, decrease_values, bar_height,
                 left=increase_values, color=colors[1], alpha=0.8, label='Decrease Ratio')

    # Calculate left position for unchanged ratio
    left_unchanged = np.array(increase_values) + np.array(decrease_values)
    p3 = ax.barh(y_positions, unchanged_values, bar_height,
                 left=left_unchanged, color=colors[2], alpha=0.8, label='Unchanged Ratio')

    # Set axes
    ax.set_ylim(-0.5, len(species_list) - 0.5)
    ax.set_xlim(0, 1)  # Ratio range 0-1

    # Remove all text labels and tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Set tick marks
    ax.tick_params(axis='x', length=8, width=2, labelbottom=False)
    ax.tick_params(axis='y', length=15, width=2, labelleft=False)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.tick_params(axis='y', labelsize=12, labelleft=False, length=15, width=2)

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Set background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Adjust layout
    plt.tight_layout()

    # 8. Save image
    if output_path:
        plt.savefig(output_path, format='svg',
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none')
        print(f"\nImage saved to: {output_path}")

    plt.show()

    # 9. Statistical testing and output
    print(f"\n=== Species Comparison Statistics ===")
    print("=" * 80)

    for species_name, data in all_species_data.items():
        print(f"\n{species_name}:")
        print(f"  Increase ratio: {data['increase']:.4f}")
        print(f"  Decrease ratio: {data['decrease']:.4f}")
        print(f"  Unchanged ratio: {data['unchanged']:.4f}")
        print(f"  Total: {data['increase'] + data['decrease'] + data['unchanged']:.4f}")

    # Cross-species comparison
    print(f"\n=== Cross-Species Comparison ===")
    increase_means = [data['increase'] for data in all_species_data.values()]
    decrease_means = [data['decrease'] for data in all_species_data.values()]
    unchanged_means = [data['unchanged'] for data in all_species_data.values()]

    print(f"Increase ratio - Mean: {np.mean(increase_means):.4f}, Std: {np.std(increase_means):.4f}")
    print(f"Decrease ratio - Mean: {np.mean(decrease_means):.4f}, Std: {np.std(decrease_means):.4f}")
    print(f"Unchanged ratio - Mean: {np.mean(unchanged_means):.4f}, Std: {np.std(unchanged_means):.4f}")

    return fig, ax, all_species_data


def create_simple_legend(output_path=None):
    """
    Create a simple color legend for the stacked chart

    Parameters:
    -----------
    output_path : str, optional
        Output image path
    """

    # Set plot style
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 14,
        'font.weight': 'normal'
    })

    # Create a small figure
    fig, ax = plt.subplots(figsize=(6, 2))

    # Set colors and labels
    colors = ['#E74C3C', '#3498DB', '#faac61']
    labels = ['Increase', 'Decrease', 'Unchanged']

    # Create simple legend elements
    legend_elements = []
    for color, label in zip(colors, labels):
        legend_elements.append(plt.Line2D([0], [0],
                                          marker='s',
                                          color='w',
                                          markerfacecolor=color,
                                          markeredgecolor=color,
                                          markersize=15,
                                          linestyle='None',
                                          label=label))

    # Create legend
    legend = ax.legend(handles=legend_elements,
                       loc='center',
                       frameon=True,
                       fontsize=14,
                       ncol=3)

    # Set legend box style
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.5)

    # Hide axes
    ax.axis('off')

    # Set background
    fig.patch.set_facecolor('white')

    # Adjust layout
    plt.tight_layout()

    # Save image
    if output_path:
        plt.savefig(output_path, format='svg', bbox_inches='tight',
                    facecolor='white', edgecolor='none', dpi=300)
        print(f"Legend saved to: {output_path}")

    plt.show()

    return fig, ax, legend


def analyze_fragmentation_patterns(species_csv_dict, sample_size=None):
    """
    Analyze fragmentation change patterns across multiple species

    Parameters:
    -----------
    species_csv_dict : dict
        Dictionary format: {"Species Name": "CSV file path", ...}
    sample_size : int, optional
        Random sample size for each species

    Returns:
    --------
    dict : Analysis results for all species
    """

    print("\n=== Multi-Species Fragmentation Pattern Analysis ===")
    print("=" * 80)

    all_results = {}

    for species_name, csv_path in species_csv_dict.items():
        try:
            print(f"\nAnalyzing species: {species_name}")

            # Read data
            df = pd.read_csv(csv_path)
            df_clean = df.dropna(subset=['Increase_Ratio', 'Decrease_Ratio', 'Unchanged_Ratio']).copy()

            # Apply sampling if needed
            if sample_size and len(df_clean) > sample_size:
                df_clean = df_clean.sample(sample_size, random_state=42)

            if len(df_clean) < 2:
                print(f"  Insufficient data, skipping analysis")
                continue

            # Calculate statistics for each change type
            increase_stats = {
                'mean': df_clean['Increase_Ratio'].mean(),
                'std': df_clean['Increase_Ratio'].std(),
                'median': df_clean['Increase_Ratio'].median(),
                'count': len(df_clean)
            }

            decrease_stats = {
                'mean': df_clean['Decrease_Ratio'].mean(),
                'std': df_clean['Decrease_Ratio'].std(),
                'median': df_clean['Decrease_Ratio'].median(),
                'count': len(df_clean)
            }

            unchanged_stats = {
                'mean': df_clean['Unchanged_Ratio'].mean(),
                'std': df_clean['Unchanged_Ratio'].std(),
                'median': df_clean['Unchanged_Ratio'].median(),
                'count': len(df_clean)
            }

            all_results[species_name] = {
                'increase': increase_stats,
                'decrease': decrease_stats,
                'unchanged': unchanged_stats,
                'total_samples': len(df_clean)
            }

            print(f"  Sample size: {len(df_clean)}")
            print(f"  Increase - Mean: {increase_stats['mean']:.4f}, Std: {increase_stats['std']:.4f}")
            print(f"  Decrease - Mean: {decrease_stats['mean']:.4f}, Std: {decrease_stats['std']:.4f}")
            print(f"  Unchanged - Mean: {unchanged_stats['mean']:.4f}, Std: {unchanged_stats['std']:.4f}")

        except Exception as e:
            print(f"  Analysis failed: {e}")

    return all_results


# Main execution function
def main_fragmentation_visualization(species_csv_dict, output_folder="data/visualization_results", sample_size=None):
    """
    Execute complete fragmentation visualization analysis pipeline

    Parameters:
    -----------
    species_csv_dict : dict
        Dictionary format: {"Species Name": "CSV file path", ...}
    output_folder : str
        Output folder for results
    sample_size : int, optional
        Random sample size for each species
    """

    import os

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    try:
        print("Starting multi-species habitat fragmentation stacked chart analysis...")
        print("=" * 80)

        # 1. Create stacked chart
        stacked_output = os.path.join(output_folder, "multi_species_habitat_fragmentation_stacked.svg")
        fig, ax, all_species_data = create_multi_species_habitat_fragmentation_stacked(
            species_csv_dict,
            output_path=stacked_output,
            sample_size=sample_size
        )

        # 2. Create legend
        legend_output = os.path.join(output_folder, "fragmentation_legend.svg")
        create_simple_legend(output_path=legend_output)

        # 3. Analyze patterns
        analysis_results = analyze_fragmentation_patterns(species_csv_dict, sample_size)

        # 4. Save analysis summary
        summary_output = os.path.join(output_folder, "fragmentation_analysis_summary.txt")
        with open(summary_output, 'w', encoding='utf-8') as f:
            f.write("Multi-Species Habitat Fragmentation Analysis Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write("Analysis Overview:\n")
            f.write(f"  Total species analyzed: {len(analysis_results)}\n")
            f.write(f"  Sample size per species: {sample_size if sample_size else 'All available data'}\n\n")

            f.write("Species-wise Results:\n")
            for species_name, results in analysis_results.items():
                f.write(f"\n{species_name}:\n")
                f.write(f"  Sample size: {results['total_samples']}\n")
                f.write(f"  Increase ratio: {results['increase']['mean']:.4f} ± {results['increase']['std']:.4f}\n")
                f.write(f"  Decrease ratio: {results['decrease']['mean']:.4f} ± {results['decrease']['std']:.4f}\n")
                f.write(f"  Unchanged ratio: {results['unchanged']['mean']:.4f} ± {results['unchanged']['std']:.4f}\n")

        if fig is not None:
            print(f"\nMulti-species stacked chart analysis complete!")
            print(f"Results saved to: {output_folder}")

        print("=" * 80)

        return all_species_data, analysis_results

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# Usage example
if __name__ == "__main__":
    # Define CSV file paths for four species
    species_csv_files = {
        "AMPH": "data/analysis_results/tables/AMPH_changes.csv",
        "BIRD": "data/analysis_results/tables/BIRD_changes.csv",
        "MAMM": "data/analysis_results/tables/MAMM_changes.csv",
        "REPT": "data/analysis_results/tables/REPT_changes.csv"
    }

    # Execute complete analysis
    all_species_data, analysis_results = main_fragmentation_visualization(
        species_csv_files,
        output_folder="data/visualization_results",
        sample_size=10000  # Maximum 10000 samples per species
    )