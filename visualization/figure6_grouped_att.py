"""Figure 6: grouped PA/KBA ATT, using all eligible site records."""
from pathlib import Path
import argparse
import json
import sqlite3
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import FormatStrFormatter

TAXA = {'AMPH': 'Amphibians', 'BIRD': 'Birds', 'MA': 'Mammals', 'RE': 'Reptiles'}
CONTINENTS = ['Europe', 'North America', 'Asia', 'Oceania', 'Africa', 'South America']
ZONES = {'A': 'Tropical', 'B': 'Arid', 'C': 'Temperate', 'D': 'Continental'}
FIGSIZE = (12.8, 10.4)
DOT_SIZE = 23
SCATTER_SIZE = 1.1
SCATTER_ALPHA = 0.12
PALETTE_Q = ['#F2AC57', '#44A678', '#128BA6', '#D94B2B']
PALETTE_CONT = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
PALETTE_TYPE = ['#E74C3C', '#2E86C1']
PALETTE_CLIMATE = ['#D94B2B', '#F2AC57', '#44A678', '#128BA6']
PANEL = {
    'a': ('PA', 'Area quartile', ['Q1', 'Q2', 'Q3', 'Q4'], PALETTE_Q, (-0.0032, 0.0012), 0.001),
    'b': ('PA', 'Continent', CONTINENTS, PALETTE_CONT, (-0.060, 0.040), 0.02),
    'c': ('PA', 'Protection type', ['Strict protection', 'Multiple use'], PALETTE_TYPE, (-0.004, 0.001), 0.001),
    'd': ('KBA', 'Area quartile', ['Q1', 'Q2', 'Q3', 'Q4'], PALETTE_Q, (-0.0022, 0.0012), 0.001),
    'e': ('KBA', 'Continent', CONTINENTS, PALETTE_CONT, (-0.040, 0.040), 0.02),
    'f': ('KBA', 'Climate zone', list(ZONES.values()), PALETTE_CLIMATE, (-0.004, 0.004), 0.002),
}
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.3,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'svg.hashsalt': 'fig6-full-data',
    'legend.frameon': False, 'savefig.facecolor': 'white',
})


def attributes(path):
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as con:
        table = con.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features'").fetchone()[0]
        geom = con.execute('SELECT column_name FROM gpkg_geometry_columns').fetchone()[0]
        columns = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")') if r[1] != geom]
        sql = 'SELECT ' + ','.join('"' + x + '"' for x in columns) + ' FROM "' + table + '" ORDER BY fid'
        return pd.read_sql_query(sql, con)


def load_groups(results_dir, continent_map, climate_membership, output_dir):
    country_map = json.loads(continent_map.read_text(encoding='utf-8'))
    climate = pd.read_csv(climate_membership)
    assert climate.sitrecid.is_unique
    groups, rows, source_rows, audit, edges = {}, [], [], [], {}
    for kind in ('PA', 'KBA'):
        for taxon, name in TAXA.items():
            path = results_dir / taxon / kind / f'{taxon}_{kind}_composite_att_spatial.gpkg'
            full = attributes(path)
            idcol = 'WDPA_PI' if kind == 'PA' else 'sitrecid'
            assert full[idcol].is_unique
            valid = full.dropna(subset=['mean_ATT']).copy()
            assert np.isfinite(valid.mean_ATT).all()
            assignments = {}
            area_col = 'AREA_KM' if kind == 'PA' else 'gis_area_k'
            area = valid.dropna(subset=[area_col]).copy()
            bins = area[area_col].quantile([0, .25, .5, .75, 1]).to_numpy()
            assert np.all(np.diff(bins) > 0), 'Repeated quartile thresholds require review.'
            area['group'] = pd.cut(area[area_col], bins=bins, labels=['Q1', 'Q2', 'Q3', 'Q4'], include_lowest=True)
            edges[f'{kind}_{taxon}'] = {'field': area_col, 'thresholds': bins.tolist()}
            assignments['a' if kind == 'PA' else 'd'] = area
            country = 'ISO3' if kind == 'PA' else 'iso3'
            cont = valid.copy()
            cont['group'] = cont[country].map(country_map)
            assignments['b' if kind == 'PA' else 'e'] = cont.dropna(subset=['group'])
            if kind == 'PA':
                category_map = {x: 'Strict protection' for x in ['Ia', 'Ib', 'II']}
                category_map.update({x: 'Multiple use' for x in ['III', 'IV', 'V', 'VI']})
                types = valid.copy()
                types['group'] = types.IUCN_CA.map(category_map)
                assignments['c'] = types.dropna(subset=['group'])
            else:
                clim = valid.merge(climate, on='sitrecid', how='left', validate='one_to_one')
                clim['group'] = clim.Cli_Zone.map(ZONES)
                assignments['f'] = clim.dropna(subset=['group'])
            for panel, data in assignments.items():
                audit.append({'Panel': panel, 'Taxon': name, 'valid_ATT': len(valid),
                              'included': len(data), 'excluded_by_grouping': len(valid) - len(data)})
                for group in PANEL[panel][2]:
                    d = data.loc[data.group == group]
                    values = d.mean_ATT.to_numpy(float)
                    assert len(values) > 1
                    groups[panel, taxon, group] = values
                    mean = float(values.mean())
                    ci = float(stats.sem(values) * stats.t.ppf(.975, len(values)-1))
                    q1, med, q3 = np.percentile(values, [25, 50, 75])
                    lo, hi = PANEL[panel][4]
                    rows.append({'Panel': panel, 'PA_KBA': kind, 'Taxon': name,
                                 'Grouping': PANEL[panel][1], 'Group': group, 'n': len(values),
                                 'mean_ATT': mean, 'CI_lower': mean-ci, 'CI_upper': mean+ci,
                                 'median': med, 'Q1': q1, 'Q3': q3,
                                 'whisker_low': max(q1-1.5*(q3-q1), values.min()),
                                 'whisker_high': min(q3+1.5*(q3-q1), values.max()),
                                 'below_axis': int((values < lo).sum()),
                                 'above_axis': int((values > hi).sum()), 'Source': str(path)})
                    source_rows.append(pd.DataFrame({'Panel': panel, 'PA_KBA': kind, 'Taxon': name,
                                                     'Group': group, 'record_id': d[idcol].astype(str).to_numpy(),
                                                     'mean_ATT': values}))
    table = pd.DataFrame(rows).sort_values(['Panel'], kind='stable')
    assert len(table) == 104
    table.to_csv(output_dir / 'Fig6_all_group_statistics.csv', index=False, encoding='utf-8-sig')
    supplement = table[table.Panel.isin(['b', 'c', 'e', 'f'])]
    assert len(supplement) == 72
    supplement[['Panel', 'PA_KBA', 'Taxon', 'Grouping', 'Group', 'n']].to_csv(
        output_dir / 'Fig6_sample_sizes.csv', index=False, encoding='utf-8-sig')
    pd.concat(source_rows, ignore_index=True).to_csv(output_dir / 'Fig6_source_data.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(audit).sort_values('Panel', kind='stable').to_csv(output_dir / 'grouping_exclusions.csv', index=False)
    (output_dir / 'area_quartile_thresholds.json').write_text(json.dumps(edges, indent=2), encoding='utf-8')
    return groups, table


def legend(ax, panel):
    _, _, labels, colors, _, _ = PANEL[panel]
    if panel in ('b', 'e'):
        handles = [(Rectangle((0, 0), 1, 1, fill=False, edgecolor=c, linewidth=1),
                    Line2D([], [], marker='o', markersize=5, markeredgecolor='black',
                           markeredgewidth=.5, color=c, linestyle='none')) for c in colors]
        ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=1)}, ncol=6,
                  loc='lower left', bbox_to_anchor=(-.012, 1.055), borderaxespad=0,
                  handlelength=1.35, handletextpad=.4, columnspacing=.7, fontsize=7.7)
    else:
        handles = [ax.errorbar([], [], yerr=[], fmt='o', color=c, markersize=4,
                              elinewidth=.8, capsize=0) for c in colors]
        ax.legend(handles, labels, ncol=len(labels), loc='lower center',
                  bbox_to_anchor=(.5, 1.015), borderaxespad=0,
                  handletextpad=.35, columnspacing=2.7)


def draw_panel(ax, panel, groups, table, standalone=False):
    kind, _, labels, colors, limits, tick_step = PANEL[panel]
    ax.set_xlim(-.52, 3.52)
    ax.set_ylim(*limits)
    ax.set_xticks(range(4), list(TAXA.values()))
    ax.tick_params(axis='x', length=0, pad=7)
    ticks = np.arange(np.ceil(limits[0]/tick_step), np.floor(limits[1]/tick_step)+1) * tick_step
    ticks[np.abs(ticks) < 1e-10] = 0
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f' if panel in ('b', 'e') else '%.3f'))
    ax.set_ylabel(f'Mean ATT of {kind}s', labelpad=10)
    ax.text(-.125, 1.045, panel, transform=ax.transAxes, fontweight='bold', fontsize=14)
    rng = np.random.default_rng(20261011 + ord(panel))
    spacing = .13 if len(labels) == 6 else (.19 if len(labels) == 4 else .31)
    offsets = (np.arange(len(labels))-(len(labels)-1)/2) * spacing
    scatter_count = 0
    for i, taxon in enumerate(TAXA):
        for j, (group, color) in enumerate(zip(labels, colors)):
            v = groups[panel, taxon, group]
            row = table[(table.Panel == panel) & (table.Taxon == TAXA[taxon]) & (table.Group == group)].iloc[0]
            x = i + offsets[j]
            if panel in ('b', 'e'):
                # Every eligible record is sent to the vector scatter artist; only x jitter is random.
                xx = x + rng.normal(0, .023, len(v))
                scatter = ax.scatter(xx, v, color=color, s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
                                     linewidths=0, zorder=1, rasterized=False, clip_on=True)
                scatter.set_gid(f'all-records-{panel}-{taxon}-{j}-n{len(v)}')
                assert len(scatter.get_offsets()) == len(v)
                scatter_count += len(v)
                width = .08
                ax.add_patch(Rectangle((x-width/2, row.Q1), width, row.Q3-row.Q1,
                                       fill=False, edgecolor=color, linewidth=1.05, zorder=3))
                ax.plot([x-width/2, x+width/2], [row['median']]*2, color=color, lw=.8, zorder=3)
                ax.plot([x, x], [row.whisker_low, row.Q1], color=color, lw=.9, zorder=3)
                ax.plot([x, x], [row.Q3, row.whisker_high], color=color, lw=.9, zorder=3)
                for end in (row.whisker_low, row.whisker_high):
                    ax.plot([x-width/4, x+width/4], [end]*2, color=color, lw=.9, zorder=3)
                edge, errcolor, cap = 'black', 'black', 1.8
                size = 33
            else:
                edge, errcolor, cap, size = color, color, 0, DOT_SIZE
            ax.errorbar(x, row.mean_ATT, yerr=[[row.mean_ATT-row.CI_lower], [row.CI_upper-row.mean_ATT]],
                        fmt='none', color=errcolor, lw=.85, capsize=cap, capthick=.7, zorder=4)
            ax.scatter(x, row.mean_ATT, s=size, color=color, edgecolors=edge, linewidths=.55, zorder=5)
    # b supplies a shared continent legend in the full composite; individual e is self-contained.
    if panel != 'e' or standalone:
        legend(ax, panel)
    return scatter_count


def save(fig, stem, output_dir):
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(output_dir / f'{stem}.{ext}', dpi=180 if ext == 'png' else 300,
                    bbox_inches='tight', pad_inches=.12)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path('data/psm_analysis/results'),
                        help='Root of the four taxonomic groups and PA/KBA spatial ATT GeoPackages.')
    parser.add_argument('--continent-map', type=Path,
                        default=Path('data/visualization/figure6/iso3_continent_mapping.json'),
                        help='JSON mapping ISO3 country codes to continent names.')
    parser.add_argument('--climate-membership', type=Path,
                        default=Path('data/visualization/figure6/kba_climate_membership.csv'),
                        help='CSV with unique sitrecid and Cli_Zone (A, B, C or D).')
    parser.add_argument('--output-dir', type=Path, default=Path('D:/hzl/code/outputs/figures/figure6'))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    groups, table = load_groups(args.results_dir, args.continent_map, args.climate_membership, output_dir)
    print('Computed all 104 group statistics without sampling.', flush=True)
    fig, axs = plt.subplots(3, 2, figsize=FIGSIZE)
    fig.subplots_adjust(left=.085, right=.985, bottom=.06, top=.95, wspace=.23, hspace=.29)
    plotted = {}
    for ax, panel in zip(axs.flat, ['a', 'd', 'b', 'e', 'c', 'f']):
        plotted[panel] = draw_panel(ax, panel, groups, table)
    save(fig, 'Fig6_grouped_att', output_dir)
    plt.close(fig)
    print('Saved combined SVG, PDF and PNG.', flush=True)
    for panel in PANEL:
        fig, ax = plt.subplots(figsize=(6.5, 3.55))
        fig.subplots_adjust(left=.145, right=.98, bottom=.16, top=.84)
        draw_panel(ax, panel, groups, table, standalone=True)
        save(fig, f'panel_{panel}_att', output_dir)
        plt.close(fig)
    ns = {'s': 'http://www.w3.org/2000/svg'}
    exports = []
    svg_names = ['Fig6_grouped_att.svg'] + [f'panel_{panel}_att.svg' for panel in PANEL]
    for name in svg_names:
        file = output_dir / name
        root = ET.parse(file).getroot()
        images = len(root.findall('.//s:image', ns))
        texts = len(root.findall('.//s:text', ns))
        assert images == 0 and texts > 0, file.name
        exports.append({'file': file.name, 'embedded_images': images, 'editable_text_elements': texts})
    for panel in ('b', 'e'):
        assert plotted[panel] == int(table.loc[table.Panel == panel, 'n'].sum())
    qa = {'sampling': False, 'scatter_sampling': False, 'scatter_records': plotted,
          'svg_checks': exports,
          'scatter_note': 'All eligible points are retained in vector artists. Display limits clip extreme values visually, not statistically.',
          'climate_join': 'External site membership table joined by unique sitrecid.',
          'quartile_area_fields': {'PA': 'AREA_KM', 'KBA': 'gis_area_k'}}
    (output_dir / 'validation.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == '__main__':
    main()
