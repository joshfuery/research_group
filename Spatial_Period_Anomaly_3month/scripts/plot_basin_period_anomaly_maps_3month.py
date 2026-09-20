"""
Precipitation anomaly spatial maps for the longest 3-month-block dry and wet
periods in each basin's timeline.

Same two-panel figure format as plot_basin_anomaly_maps.py (dry | wet, diverging
red-yellow-blue scale, basin outline, shared inverted colorbar), but each panel
shows a PERIOD-AVERAGED anomaly rather than a single month.

Periods come from the 3-month "individual_month" wet/dry detection in
ET_Wet_Dry_precip_rerun/figures/Wet_Dry/3_month/individual_month/periods_summary.csv
-- the longest Dry and longest Wet run per basin. In that CSV, start_date and
end_date are the START months of the first and last 3-month block, so a period
spans start_date through end_date + 2 months.

Differences from plot_basin_anomaly_maps.py
-------------------------------------------
1. CLIMATOLOGY: built over the full IMERG record (1998-2025), not 2001-2015, so
   it matches the climatology behind the wet/dry period detection. Several
   longest periods fall entirely outside 2001-2015.

2. UNITS: IMERG 3B-MO stores precipitation in mm/hr (verified in the file's
   Units attribute), so it is converted with * 24 * days_in_month. The reference
   script's is_monthly_product() check skips that conversion for exactly these
   files, leaving mm/hr, which is why it needed to rescale each map to the basin
   time series to reach mm. With the conversion applied the basin mean matches
   the time-series CSV directly (Murray 2010-01: 42.785465 mm both ways), so no
   rescaling is done here.

3. COLOR SCALE: per-basin symmetric vmax from the 98th percentile of the
   absolute period-averaged anomaly across both panels. Averaging over a
   multi-year period flattens anomalies well below the single-month +/-80 and
   +/-100 mm of the reference.

Values are mm/month: the mean monthly anomaly over the months in the period.
"""

from __future__ import annotations

import calendar
import os
import re
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

# ---------- Config ----------
BASE = Path(__file__).resolve().parent.parent.parent  # workspace root
IMERG_DIR = Path("/Users/josh/Documents/WaterSecurity/precip_imerg")
BASINS_VECTOR = BASE / "data" / "raw" / "basins.geojson"
BASIN_NAME_FIELD = "NAME_BASIN"
PERIODS_CSV = (
    BASE / "ET_Wet_Dry_precip_rerun" / "figures" / "Wet_Dry" / "3_month"
    / "individual_month" / "periods_summary.csv"
)
OUT_DIR = BASE / "outputs" / "spatial_maps" / "period_anomaly_maps_3month"

BLOCK_MONTHS = 3  # 3-month blocks; end_date is the start of the final block

BASIN_SLUGS = {
    "Amazon Basin": "amazon", "California": "california", "Colorado": "colorado",
    "Congo": "congo", "Danube": "danube", "Ganges / Bhramaputra Basin": "ganges_brahmaputra",
    "Mekong Basin": "mekong", "Mississippi / Missouri": "mississippi_missouri",
    "Murray": "murray", "Nile": "nile", "Yangtze Basin": "yangtze",
}


def to_slug(name):
    return BASIN_SLUGS.get(name, re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"))


# Basin-specific: extent [lon_min, lon_max, lat_min, lat_max], optional country labels.
# Extents match plot_basin_anomaly_maps.py so the figures are directly comparable.
BASIN_CONFIG = {
    "Mekong Basin": {
        "extent": [92, 110, 5, 35],
        "country_labels": [
            ("China", 102, 32),
            ("India", 94, 26),
            ("Myanmar", 96, 22),
            ("Laos", 104, 19),
            ("Thailand", 100, 14),
            ("Vietnam", 107, 16),
            ("Cambodia", 105, 12),
        ],
    },
    "Murray": {
        # Full Australia so the whole landmass is visible
        "extent": [112, 154, -44, -10],
        "country_labels": None,
    },
}

# Paper-style diverging: red - light orange - yellow - light greenish blue - blue
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "precip_anom",
    [
        "#B2182B",  # red (dry / negative)
        "#F4A582",  # light orange
        "#FFFFBF",  # yellow (neutral)
        "#91BFDB",  # light greenish blue
        "#2166AC",  # blue (wet / positive)
    ],
    N=256,
)


def filename_to_date(fpath):
    name = os.path.basename(str(fpath))
    m8 = re.search(r"(\d{8})", name)
    if m8:
        return pd.Timestamp(datetime.strptime(m8.group(1), "%Y%m%d"))
    m6 = re.search(r"(\d{6})", name)
    if m6:
        return pd.Timestamp(datetime.strptime(m6.group(1) + "01", "%Y%m%d"))
    raise ValueError(f"Cannot parse date from {name}")


def days_in_month(dt):
    return calendar.monthrange(dt.year, dt.month)[1]


def load_imerg_grid_info(imerg_dir):
    imerg_dir = Path(imerg_dir)
    files = sorted(
        list(imerg_dir.glob("*.HDF5")) + list(imerg_dir.glob("*.h5")) + list(imerg_dir.glob("*.HDF"))
    )
    if not files:
        raise FileNotFoundError(f"No IMERG files in {imerg_dir}")
    import h5py
    with h5py.File(files[0], "r") as f:
        if "Grid/precipitation" in f:
            precip_path, lat_path, lon_path = "Grid/precipitation", "Grid/lat", "Grid/lon"
        else:
            precip_path, lat_path, lon_path = "precipitation", "lat", "lon"
        lat = np.asarray(f[lat_path][()])
        lon = np.asarray(f[lon_path][()])
    dx = float(np.abs(np.mean(np.diff(lon))))
    dy = float(np.abs(np.mean(np.diff(lat))))
    return lat, lon, precip_path, files, dx, dy


def read_one_precip(fpath, precip_path, dt):
    """Monthly precipitation in mm, row 0 = north.

    IMERG 3B-MO stores mm/hr (see the dataset's Units attribute), so the value
    is converted to a monthly total. Verified against
    data/processed/precipitation/imerg_basin_monthly_precip.csv.
    """
    import h5py
    with h5py.File(fpath, "r") as f:
        precip = f[precip_path][()]
    if precip.ndim == 3:
        precip = np.squeeze(precip, axis=0).T
    else:
        precip = precip.T
    precip = np.flipud(precip)
    precip = np.where(precip < 0, np.nan, precip)
    return precip * 24 * days_in_month(dt)


def bbox_indices(lat, lon, extent):
    """Row/col indices into a row0-north grid for the given extent.

    Returns (idx, lat_desc, lon_asc) where lat_desc is north-to-south so it
    lines up with the subset's row order and with the rasterized basin mask.
    """
    lon_w, lon_e, lat_s, lat_n = extent
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    jj_lat = np.where((lat >= lat_s) & (lat <= lat_n))[0]
    ii = np.where((lon >= lon_w) & (lon <= lon_e))[0]
    rows = len(lat) - 1 - jj_lat          # lat ascending -> row indices (row0 = north)
    order = np.argsort(rows)              # rows ascending == lat descending
    return np.ix_(rows[order], ii), lat[jj_lat][order], lon[ii]


def build_basin_mask(basin_gdf, lat_full, lon_full, dx, dy, extent):
    """Basin mask on the full grid, subset to extent. Same rasterize call as
    plot_basin_anomaly_maps.py so the mask matches the basin time series."""
    transform = from_origin(float(lon_full.min() - dx / 2), float(lat_full.max() + dy / 2), dx, dy)
    geom = basin_gdf.geometry.union_all()
    full_mask = rasterize(
        [(geom, 1)],
        out_shape=(len(lat_full), len(lon_full)),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    ) == 1
    idx, _, _ = bbox_indices(lat_full, lon_full, extent)
    return full_mask[idx]


def load_period_bounds(periods_csv, basins):
    """Longest Dry and longest Wet period per basin, as (first_month, last_month).

    end_date is the start of the final 3-month block, so the period runs through
    end_date + (BLOCK_MONTHS - 1) months.
    """
    df = pd.read_csv(periods_csv, parse_dates=["start_date", "end_date"])
    df = df[df["basin"].isin(basins)]
    out = {}
    for basin in basins:
        sub = df[df["basin"] == basin]
        if sub.empty:
            raise ValueError(f"{basin} not found in {periods_csv}")
        entry = {}
        for ptype in ("Dry", "Wet"):
            rows = sub[sub["type"] == ptype]
            row = rows.loc[rows["duration_blocks"].idxmax()]
            last = row["end_date"] + pd.DateOffset(months=BLOCK_MONTHS - 1)
            entry[ptype] = (row["start_date"], last, int(row["duration_blocks"]))
        out[basin] = entry
    return out


def build_stacks(files, precip_path, lat_full, lon_full, configs):
    """One pass over the IMERG files, extracting each basin's bbox subset.

    Returns times and {basin: (n_months, n_lat, n_lon) float32 array in mm}.
    """
    idx_by_basin = {b: bbox_indices(lat_full, lon_full, c["extent"])[0] for b, c in configs.items()}
    per_basin = {b: [] for b in configs}
    times = []
    for fpath in tqdm(sorted(files), desc="Loading IMERG"):
        try:
            dt = filename_to_date(fpath)
            precip = read_one_precip(fpath, precip_path, dt)
        except Exception as exc:
            print(f"  Skipping {os.path.basename(str(fpath))}: {exc}")
            continue
        times.append(dt)
        for basin, idx in idx_by_basin.items():
            per_basin[basin].append(precip[idx].astype(np.float32))
    if not times:
        raise RuntimeError("No IMERG files read. Check the directory and date parsing.")
    times = pd.DatetimeIndex(times)
    order = np.argsort(times)
    times = times[order]
    stacks = {b: np.stack(v, axis=0)[order] for b, v in per_basin.items()}
    return times, stacks


def build_climatology(stack, times):
    """Per-pixel calendar-month climatology (12, n_lat, n_lon) over the whole record."""
    months = np.array([t.month for t in times])
    clim = np.full((12,) + stack.shape[1:], np.nan, dtype=np.float32)
    for m in range(1, 13):
        sel = months == m
        if sel.any():
            clim[m - 1] = np.nanmean(stack[sel], axis=0)
    return clim


def period_mean_anomaly(stack, times, clim, first_month, last_month):
    """Mean monthly anomaly (mm/month) over the period, per pixel.

    Each month has its calendar-month climatology removed, then the period's
    months are averaged.
    """
    sel = (times >= first_month) & (times <= last_month)
    n_sel = int(sel.sum())
    if n_sel == 0:
        raise ValueError(f"No IMERG months in {first_month:%Y-%m}..{last_month:%Y-%m}")

    idx = np.flatnonzero(sel)
    anomalies = stack[idx] - clim[[times[i].month - 1 for i in idx]]
    return np.nanmean(anomalies, axis=0), n_sel


def pick_vmax(*slices, percentile=98):
    """Symmetric color limit from a robust percentile of |anomaly|, rounded up."""
    vals = np.concatenate([s[np.isfinite(s)].ravel() for s in slices])
    if vals.size == 0:
        return 1.0
    v = float(np.percentile(np.abs(vals), percentile))
    if v <= 0:
        return 1.0
    step = 10 ** np.floor(np.log10(v))
    return float(np.ceil(v / step) * step)


def cell_edges(centers):
    centers = np.asarray(centers, dtype=float)
    if len(centers) < 2:
        return np.array([centers[0] - 0.05, centers[0] + 0.05])
    d = np.diff(centers)
    return np.concatenate([[centers[0] - d[0] / 2], centers[:-1] + d / 2, [centers[-1] + d[-1] / 2]])


def plot_two_panels(anomaly_dry, anomaly_wet, lat, lon, extent, basin_gdf,
                    dry_label, wet_label, vmax, outpath, country_labels=None):
    """Two side-by-side panels with a shared colorbar and the basin outline."""
    fig = plt.figure(figsize=(12, 6), facecolor="white")
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    lon_2d, lat_2d = np.meshgrid(cell_edges(lon), cell_edges(lat))

    mesh = None
    panels = [
        (1, anomaly_dry, dry_label, "A"),
        (2, anomaly_wet, wet_label, "B"),
    ]
    for pos, data, title, tag in panels:
        ax = fig.add_subplot(1, 2, pos, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set_facecolor("white")
        # Base map first so the data is not covered
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="lightgray", alpha=0.25)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.8)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.8)
        mesh = ax.pcolormesh(
            lon_2d, lat_2d, data, cmap=CMAP, norm=norm,
            transform=ccrs.PlateCarree(), shading="flat",
        )
        basin_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=2, transform=ccrs.PlateCarree())
        ax.set_title(title, fontsize=12, fontweight="bold")
        if country_labels:
            for name, x, y in country_labels:
                ax.text(x, y, name, transform=ccrs.PlateCarree(), fontsize=9, ha="center", va="center")
        ax.text(0.02, 0.02, tag, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")

    cbar_ax = fig.add_axes([0.02, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="vertical")
    cbar.set_label("Precipitation Anomaly (mm/month)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.invert_yaxis()  # red (dry) at top, blue (wet) at bottom
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08, wspace=0.08)
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {outpath}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Period-averaged precip anomaly maps for the longest 3-month-block dry/wet periods"
    )
    parser.add_argument("--imerg-dir", default=str(IMERG_DIR), help="IMERG HDF5 directory")
    parser.add_argument("--basins", default=str(BASINS_VECTOR), help="Basins GeoJSON")
    parser.add_argument("--periods-csv", default=str(PERIODS_CSV), help="3-month individual_month periods CSV")
    parser.add_argument("--output-dir", default=str(OUT_DIR), help="Output directory")
    parser.add_argument("--percentile", type=float, default=98, help="Percentile for the symmetric color limit")
    args = parser.parse_args()

    basins_gdf = gpd.read_file(args.basins)
    if basins_gdf.crs != "EPSG:4326":
        basins_gdf = basins_gdf.to_crs("EPSG:4326")

    imerg_dir = Path(args.imerg_dir)
    if not imerg_dir.exists():
        print(f"IMERG directory not found: {imerg_dir}")
        return

    configs = {b: c for b, c in BASIN_CONFIG.items() if not basins_gdf[basins_gdf[BASIN_NAME_FIELD] == b].empty}
    for missing in set(BASIN_CONFIG) - set(configs):
        print(f"Skipping {missing}: not found in {args.basins}")
    if not configs:
        return

    periods = load_period_bounds(args.periods_csv, list(configs))
    lat_full, lon_full, precip_path, files, dx, dy = load_imerg_grid_info(imerg_dir)
    print(f"IMERG grid: lat [{lat_full.min():.2f}, {lat_full.max():.2f}], "
          f"lon [{lon_full.min():.2f}, {lon_full.max():.2f}], {len(files)} files")

    times, stacks = build_stacks(files, precip_path, lat_full, lon_full, configs)
    print(f"Climatology record: {times.min():%Y-%m} to {times.max():%Y-%m} ({len(times)} months)\n")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for basin, config in configs.items():
        extent = config["extent"]
        basin_gdf = basins_gdf[basins_gdf[BASIN_NAME_FIELD] == basin]
        _, lat_b, lon_b = bbox_indices(lat_full, lon_full, extent)
        mask = build_basin_mask(basin_gdf, lat_full, lon_full, dx, dy, extent)
        print(f"=== {basin} ===")
        print(f"  Extent {extent} | grid {stacks[basin].shape[1:]} | {int(mask.sum())} cells inside basin")
        if not mask.any():
            print("  WARNING: empty basin mask, skipping")
            continue

        clim = build_climatology(stacks[basin], times)
        slices = {}
        for ptype in ("Dry", "Wet"):
            first, last, blocks = periods[basin][ptype]
            field, n_months = period_mean_anomaly(stacks[basin], times, clim, first, last)
            field = np.where(mask, field, np.nan)
            slices[ptype] = (field, first, last, blocks, n_months)
            basin_mean = float(np.nanmean(field))
            print(f"  {ptype}: {first:%b %Y} - {last:%b %Y} "
                  f"({blocks} blocks, {n_months} months) basin mean {basin_mean:+.1f} mm/month")
            summary.append({
                "basin": basin, "type": ptype,
                "start_month": first.strftime("%Y-%m"), "end_month": last.strftime("%Y-%m"),
                "duration_blocks": blocks, "n_months": n_months,
                "basin_mean_anomaly_mm_per_month": round(basin_mean, 3),
            })

        dry_field, dry_first, dry_last, _, _ = slices["Dry"]
        wet_field, wet_first, wet_last, _, _ = slices["Wet"]
        vmax = pick_vmax(dry_field, wet_field, percentile=args.percentile)
        print(f"  Color limit: +/-{vmax:.0f} mm/month "
              f"(dry {np.nanmin(dry_field):.1f}..{np.nanmax(dry_field):.1f}, "
              f"wet {np.nanmin(wet_field):.1f}..{np.nanmax(wet_field):.1f})")

        plot_two_panels(
            dry_field, wet_field, lat_b, lon_b, extent, basin_gdf,
            f"Dry: {dry_first:%B %Y} - {dry_last:%B %Y}",
            f"Wet: {wet_first:%B %Y} - {wet_last:%B %Y}",
            vmax,
            out_dir / f"{to_slug(basin)}_precip_period_anomaly_map_3month.png",
            country_labels=config.get("country_labels"),
        )
        print()

    pd.DataFrame(summary).to_csv(out_dir / "period_anomaly_summary.csv", index=False)
    print(f"Saved {out_dir / 'period_anomaly_summary.csv'}")
    print("Done. Outputs in", out_dir)


if __name__ == "__main__":
    main()
