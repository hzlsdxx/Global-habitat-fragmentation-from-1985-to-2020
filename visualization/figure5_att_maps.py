"""Figure 5: map site-level PA/KBA mean_ATT from GeoPackage inputs."""
from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

COLORS = ['#2f7d1f', '#7fb03f', '#b8cf9a', '#d7c8d3', '#cc5c9d', '#b30078']
MARKERSIZE = 12
mpl.rcParams.update({'font.family': 'Arial', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'svg.hashsalt': 'figure5-att-maps'})

def build_scale(values):
    neg, pos = values[values < 0], values[values > 0]
    bounds = []
    if len(neg):
        bounds.extend(np.quantile(neg, [0, 1/3, 2/3]))
    bounds.append(0.)
    if len(pos):
        bounds.extend(np.quantile(pos, [1/3, 2/3, 1]))
    bounds = np.array(sorted(set(np.round(bounds, 6))))
    if len(bounds) < 4:
        bounds = np.unique(np.round(np.quantile(values, [0, .2, .4, .6, .8, 1]), 6))
        bounds = np.unique(np.append(bounds, 0.))
    count = len(bounds) - 1
    colors = COLORS if count == 6 else mpl.colors.LinearSegmentedColormap.from_list('custom', COLORS, N=count)(np.arange(count))
    cmap = mpl.colors.ListedColormap(colors)
    return bounds, cmap, mpl.colors.BoundaryNorm(bounds, cmap.N)

def draw(points, layers, key, bounds, cmap, norm, output_dir):
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.subplots_adjust(left=.01, right=.99, bottom=.15, top=.99)
    for name, frame in layers.items():
        before = len(ax.collections)
        if name == 'china_lines':
            frame.plot(ax=ax, color='#4D4D4D', linewidth=.8, zorder=1.2)
        else:
            frame.boundary.plot(ax=ax, color='#4D4D4D', linewidth=.8, zorder=1.1 if name == 'china_poly' else 1)
        for i, collection in enumerate(ax.collections[before:]):
            collection.set_gid(f'{name}_{i}')
    points.plot(ax=ax, column='mean_ATT', cmap=cmap, norm=norm,
                markersize=MARKERSIZE, linewidth=0, alpha=.95, zorder=2)
    ax.collections[-1].set_gid('att_points')
    ax.set_axis_off()
    # Reserve space below the map so the legend cannot obscure island boundaries.
    cax = fig.add_axes([.55, .055, .28, .026])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                       cax=cax, orientation='horizontal', spacing='uniform')
    cbar.set_ticks(bounds)
    cbar.ax.tick_params(length=3, width=.7, labelsize=11)
    cbar.outline.set_visible(False)
    cbar.solids.set_rasterized(False)
    for text_on in [True, False]:
        cbar.set_ticklabels([f'{v:.3f}' for v in bounds] if text_on else ['']*len(bounds))
        cax.set_title(f'Mean ATT of {key}s' if text_on else '', fontsize=14, pad=9)
        suffix = 'with_text' if text_on else 'no_text'
        stem = output_dir / f'ALL_{key}_map_{suffix}'
        fig.savefig(stem.with_suffix('.svg'), facecolor='white')
        fig.savefig(stem.with_suffix('.png'), dpi=150, facecolor='white')
    plt.close(fig)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('data/psm_analysis/results'),
                        help='Root containing ALL/PA and ALL/KBA spatial ATT GeoPackages.')
    parser.add_argument('--world-boundary', type=Path, default=Path('data/boundaries/world.shp'))
    parser.add_argument('--china-polygons', type=Path, default=Path('data/boundaries/China_poly.shp'))
    parser.add_argument('--china-lines', type=Path, default=Path('data/boundaries/China_lines.shp'))
    parser.add_argument('--output-dir', type=Path, default=Path('D:/hzl/code/outputs/figures/figure5'))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = {kind: args.results_dir / 'ALL' / kind / f'ALL_{kind}_composite_att_spatial.gpkg'
             for kind in ('PA', 'KBA')}
    layer_paths = {'world': args.world_boundary, 'china_poly': args.china_polygons,
                   'china_lines': args.china_lines}
    layers = {name:gpd.read_file(path).to_crs('EPSG:6933') for name,path in layer_paths.items()}
    report = {'crs':'EPSG:6933', 'boundary_sources':{k:str(v) for k,v in layer_paths.items()},
              'layer_features':{k:len(v) for k,v in layers.items()}, 'maps':{}}
    for key, path in tasks.items():
        data = gpd.read_file(path)
        original_count = len(data)
        data = data[data['mean_ATT'].notna()].copy()
        assert np.isfinite(data['mean_ATT']).all()
        assert data.geometry.notna().all() and not data.geometry.is_empty.any()
        data = data.to_crs('EPSG:6933')
        data['geometry'] = data.representative_point()
        data = data.sort_values('mean_ATT', key=lambda s:s.abs())
        bounds, cmap, norm = build_scale(data['mean_ATT'].to_numpy())
        draw(data, layers, key, bounds, cmap, norm, output_dir)
        source = data[['mean_ATT']].copy()
        source['x_epsg6933'], source['y_epsg6933'] = data.geometry.x, data.geometry.y
        source.to_csv(output_dir/f'ALL_{key}_point_source_data.csv', index=True, index_label='source_row')
        report['maps'][key] = {'source':str(path), 'input_features':original_count,
                               'plotted_features':len(data), 'bounds':bounds.tolist()}
        print(key, len(data), bounds, flush=True)
    ns = {'s':'http://www.w3.org/2000/svg'}
    report['svg_checks'] = {}
    for path in output_dir.glob('ALL_*_map_*.svg'):
        root = ET.parse(path).getroot()
        ids = {node.get('id') for node in root.iter()}
        image_count, text_count = len(root.findall('.//s:image', ns)), len(root.findall('.//s:text', ns))
        assert image_count == 0, path
        assert all(f'{name}_0' in ids for name in layers), path
        assert (text_count > 0) == ('with_text' in path.stem), path
        report['svg_checks'][path.name] = {'embedded_images':image_count, 'editable_text_elements':text_count,
                                          'all_three_boundary_layers_present':True}
    (output_dir/'map_metadata.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
