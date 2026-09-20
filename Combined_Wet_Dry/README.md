# Combined Wet/Dry — ET, GLDAS Runoff, IMERG Precip

Wet/dry period-length boxplots in the same format as `ET_Wet_Dry_precip_rerun/figures/Wet_Dry/`,
but combining three variables per basin instead of one, at 6-month and
12-month block resolutions only.

Each basin is a 3-box cluster: **ET** (MODIS), **Runoff** (GLDAS), **Precip**
(GPM IMERG) — same box/mean-marker/jittered-dot styling as the original,
colored by variable instead of by basin, with a two-panel (a) Wet / (b) Dry
period length layout.

Uses the **individual_month** method (not accumulation): each block is
classified wet/dry purely by the sign of its own anomaly value — no running
sum — and a period is a run of consecutive blocks that share that sign. Any
sign flip ends the streak immediately regardless of magnitude. This is
`ET_wet_dry_periods.py`'s `detect_consecutive_periods`, not its Eq 1-4
accumulation/whiplash method.

## Data sources

- **ET** — `data/processed/ET/MODIS_ET_*_2002_2024.csv` (per basin, monthly).
  Own anomaly (`compute_anomaly`: raw minus calendar-month climatology) +
  individual-month detection computed in this script.
- **Precip** — `data/processed/precipitation/imerg_basin_monthly_precip.csv`
  (wide, monthly). Same treatment as ET.
- **Runoff** — pulled **pre-computed**, not recomputed here, from
  [`joshfuery/research_group`](https://github.com/joshfuery/research_group)'s
  root-level `periods_summary.csv` (6-month blocks) and `periods_summary12.csv`
  (12-month blocks) — already the individual-month method, covering all 11
  basins. Copied verbatim to
  `data/gldas_runoff_periods_summary_{6,12}month.csv`. Two basin names differ
  from this workspace's spelling and are remapped in the script:
  `"Ganges / Brahmaputra Basin"` → `"Ganges / Bhramaputra Basin"`,
  `"Murray Darling"` → `"Murray"`.

## Run

```
cd scripts
python3 combined_wet_dry_periods.py
```

## Outputs (`figures/`)

- `6_month/`, `12_month/` — each contains `figure_combined_wet_dry_period_length.png`
  and `periods_summary.csv` (every detected/loaded period, all basins × variables)
