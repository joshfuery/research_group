# Config — inputs for `plot_basin_period_anomaly_maps_3month.py`

The three inputs the script reads, so the figures in `../figures/` can be
reproduced, or regenerated on different data by swapping a file here.

| File | What it is | Where the script picks it up |
| --- | --- | --- |
| `periods_summary.csv` | Wet/dry periods per basin from the 3-month `individual_month` detection (547 rows, 11 basins). The longest Dry and longest Wet run per basin become the two panels. | `PERIODS_CSV` / `--periods-csv` |
| `basins.geojson` | The 11 basin polygons, EPSG:4326. Names are in the `NAME_BASIN` field. Used for the basin mask and the black outline. | `BASINS_VECTOR` / `--basins` |
| `nasa_earthdata_download_manifest.txt` | The GPM IMERG 3B-MO file list (1998-01 to 2025-04, 328 monthly HDF5 files, 5.7 GB). The data itself is too large to keep in the repo, so this is the download list. | the files go in a directory given by `IMERG_DIR` / `--imerg-dir` |

## Reproducing the committed figures

```
python3 ../scripts/plot_basin_period_anomaly_maps_3month.py \
  --periods-csv periods_summary.csv \
  --basins basins.geojson \
  --imerg-dir /path/to/your/precip_imerg \
  --output-dir ../figures
```

Needs `numpy pandas geopandas rasterio cartopy matplotlib h5py` (plus `tqdm`
for the progress bar). Verified: rerunning with these files reproduces
`../figures/*.png` and `period_anomaly_summary.csv` byte for byte.

## Running it on different data

Replace a file here (or point the matching flag elsewhere) — no code change
needed for the paths:

- **Different periods** — a CSV with columns `type` (`Dry`/`Wet`),
  `start_date`, `end_date`, `duration_blocks`, `basin`. `end_date` is the
  *start* month of the final 3-month block, so a period spans `start_date`
  through `end_date + 2` months (`BLOCK_MONTHS` in the script).
- **Different basins** — any vector file geopandas reads, reprojected to
  EPSG:4326 automatically. If the name field is not `NAME_BASIN`, change
  `BASIN_NAME_FIELD` in the script.
- **Different precipitation** — a directory of monthly HDF5 files with
  `Grid/precipitation` (or `precipitation`) plus `lat`/`lon`, and the month
  readable from the filename as `YYYYMMDD` or `YYYYMM`. Values are assumed to
  be mm/hr and converted with `* 24 * days_in_month`; for a product already in
  mm/month, drop that conversion in `read_one_precip`.

Two things still live in the script, not here: `BASIN_CONFIG`, which sets each
basin's map extent and country labels (only Mekong and Murray are filled in —
add an entry to map another basin), and `BASIN_SLUGS`, the output filename stems.
