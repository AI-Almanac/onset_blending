"""
add_adm3_pcode.py
─────────────────
Rename the 'id' column in the blend output CSV to 'adm3_name', then
join 'adm3_pcode' from the woredas shapefile.

Usage
-----
  python add_adm3_pcode.py \
      --csv  blend_output_summary_20260518.csv \
      --shp  manual_zones_woredas.shp \
      --out  blend_output_summary_20260518_adm3.csv
"""

import argparse
import pandas as pd
import geopandas as gpd

parser = argparse.ArgumentParser()
parser.add_argument('--csv', default='blend_output_summary_20260518.csv')
parser.add_argument('--shp', default='manual_zones_woredas.shp')
parser.add_argument('--out', default='blend_output_summary_20260518_adm3.csv')
args = parser.parse_args()

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(args.csv)
gdf = gpd.read_file(args.shp)[['adm3_name', 'adm3_pcode']]

# ── Rename id → adm3_name ─────────────────────────────────────────────────────
df = df.rename(columns={'id': 'adm3_name'})

# ── Join adm3_pcode ───────────────────────────────────────────────────────────
df = df.merge(gdf, on='adm3_name', how='left')

# Move adm3_pcode next to adm3_name
cols = ['adm3_name', 'adm3_pcode'] + [c for c in df.columns if c not in ('adm3_name', 'adm3_pcode')]
df = df[cols]

# ── Report any unmatched rows ─────────────────────────────────────────────────
missing = df['adm3_pcode'].isna().sum()
if missing:
    print(f"Warning: {missing} rows could not be matched to a pcode:")
    print(df[df['adm3_pcode'].isna()]['adm3_name'].tolist())

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(args.out, index=False)
print(f"Saved: {args.out}  ({len(df)} rows, {df.columns.tolist()})")
