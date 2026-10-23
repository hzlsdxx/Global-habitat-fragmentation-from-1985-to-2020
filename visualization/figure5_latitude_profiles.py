"""Figure 5: plot 1-degree latitude medians of site mean_ATT and bootstrap intervals."""
from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

SEED = 20261011
N_BOOT = 1000
plt.rcParams.update({'font.family': 'Arial', 'svg.fonttype': 'none', 'pdf.fonttype': 42})

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('data/psm_analysis/results'),
                        help='Root containing ALL/PA and ALL/KBA spatial ATT GeoPackages.')
    parser.add_argument('--output-dir', type=Path, default=Path('D:/hzl/code/outputs/figures/figure5'))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'purpose': 'PA and KBA distributions of ATT by latitude',
        'archetype': 'quantitative grid; two separate latitude profiles',
        'backend': 'Python', 'canvas_inches': [2.5, 6],
        'bin_width_degrees': 1, 'bootstrap_replicates': N_BOOT, 'seed': SEED,
        'center': 'median of site mean_ATT within each latitude bin',
        'interval': '95% percentile bootstrap confidence interval of median; site resampling within bin',
        'representative_points': 'Computed after transforming polygons to EPSG:4326',
        'maps': {},
    }
    for key in ['PA', 'KBA']:
        path = args.results_dir / 'ALL' / key / f'ALL_{key}_composite_att_spatial.gpkg'
        data = gpd.read_file(path, columns=['mean_ATT'])
        data = data.loc[data['mean_ATT'].notna()].copy()
        assert np.isfinite(data['mean_ATT']).all()
        data = data.to_crs(4326)
        data['geometry'] = data.representative_point()
        data['latitude'] = data.geometry.y
        data['bin'] = pd.cut(data['latitude'], np.arange(-90, 91, 1), include_lowest=True)
        rng = np.random.RandomState(SEED)
        rows = []
        for interval, group in data.groupby('bin', observed=False):
            values = group['mean_ATT'].to_numpy()
            if not len(values):
                continue
            median = np.median(values)
            if len(values) == 1:
                lo = hi = median
            else:
                samples = rng.choice(values, size=(N_BOOT, len(values)), replace=True)
                lo, hi = np.percentile(np.median(samples, axis=1), [2.5, 97.5])
            rows.append([interval.mid, median, lo, hi, len(values)])
        summary = pd.DataFrame(rows, columns=['lat_mid', 'median', 'lower', 'upper', 'count']).sort_values('lat_mid')
        assert summary['count'].sum() == len(data)
        assert (summary['lower'] <= summary['upper']).all()
        summary.to_csv(output_dir/f'ALL_{key}_lat_profile_statistics.csv', index=False)
        data[['latitude', 'mean_ATT']].to_csv(output_dir/f'ALL_{key}_lat_profile_source_data.csv', index=False)
        xmax = np.max(np.abs(summary[['lower', 'upper']].to_numpy())) * 1.05
        for show_text in [True, False]:
            fig, ax = plt.subplots(figsize=(2.5, 6), facecolor='white')
            ax.fill_betweenx(summary['lat_mid'], summary['lower'], summary['upper'],
                             color='lightgray', alpha=.6, linewidth=0)
            ax.plot(summary['median'], summary['lat_mid'], color='black', linewidth=1)
            ax.axvline(0, color='red', linestyle='--', linewidth=1.2)
            ax.set_xlim(-xmax, xmax)
            ax.set_xticks([-0.04, -0.02, 0.00, 0.02, 0.04])
            ax.set_ylim(-55, 85)
            ax.yaxis.set_ticks_position('right')
            ticks = [-40, -20, 0, 20, 40, 60, 80]
            ax.set_yticks(ticks)
            ax.set_yticklabels([f'{abs(t)}°S' if t < 0 else f'{t}°N' if t > 0 else '0°' for t in ticks])
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(axis='y', right=True, left=False, labelright=show_text,
                           labelleft=False, direction='in', length=5, labelsize=9)
            ax.tick_params(axis='x', labelbottom=show_text, direction='in', length=5, labelsize=9)
            for spine in ax.spines.values():
                spine.set_linewidth(1)
            # Identical plot geometry in both versions; right margin reserves latitude labels.
            fig.subplots_adjust(left=.08, right=.78, bottom=.08, top=.98)
            suffix = 'with_text' if show_text else 'no_text'
            stem = output_dir/f'ALL_{key}_lat_profile_{suffix}'
            fig.savefig(stem.with_suffix('.svg'), facecolor='white')
            fig.savefig(stem.with_suffix('.png'), dpi=180, facecolor='white')
            plt.close(fig)
            root = ET.parse(stem.with_suffix('.svg')).getroot()
            ns = {'s':'http://www.w3.org/2000/svg'}
            assert not root.findall('.//s:image', ns)
            assert bool(root.findall('.//s:text', ns)) == show_text
        metadata['maps'][key] = {'source': str(path), 'sites': len(data),
                                 'nonempty_bins': len(summary), 'x_limits': [-xmax, xmax],
                                 'x_ticks': [-0.04, -0.02, 0.00, 0.02, 0.04],
                                 'vector_validation': 'No embedded images; editable text only in with_text SVG'}
        print(key, len(data), 'sites;', len(summary), 'bins; exports verified', flush=True)
    (output_dir/'latitude_profile_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')

if __name__ == '__main__':
    main()
