#!/usr/bin/env python3
"""
combined_wet_dry_periods.py

Wet/dry period-length boxplots combining three variables — MODIS ET, GLDAS
mean runoff, and GPM IMERG precipitation — per basin, at 6-month and
12-month block resolutions only.

Same figure style as ET_Wet_Dry_precip_rerun/scripts/ET_wet_dry_periods.py,
using its "individual_month" method (not "accumulation"): each block is
classified wet/dry purely by the sign of its own anomaly value — no running
sum — and a period is simply a run of consecutive blocks that share that
sign ("how many blocks in a row are wet/dry"). Any sign flip ends the
streak immediately, regardless of magnitude. This mirrors
ET_wet_dry_periods.py's detect_consecutive_periods, not its
detect_wet_dry_periods (Eq 1-4 accumulation/whiplash) function.

What's different from ET_wet_dry_periods.py: instead of one variable running
at four block resolutions with one box per basin, this script runs three
variables at two block resolutions (6-month, 12-month), and for each basin
draws one box per variable (ET / Runoff / Precip) side by side, so each
basin is a 3-box cluster. Panel layout (a) Wet period length / (b) Dry
period length, box styling (solid color, mean "x" marker, jittered strip of
every period's exact duration, full frame, grid) all match the original.

Data sources
------------
ET       ../../data/processed/ET/MODIS_ET_*_2002_2024.csv           (per basin, monthly, 2002-2024) -> own anomaly + individual-month detection here
Precip   ../../data/processed/precipitation/imerg_basin_monthly_precip.csv (wide, monthly, 1998-2025) -> own anomaly + individual-month detection here
Runoff   GLDAS, pulled pre-computed (not recomputed here) from
         joshfuery/research_group's root-level periods_summary.csv (6-month
         blocks) and periods_summary12.csv (12-month blocks) — already the
         individual-month consecutive-streak method, covering all 11
         basins. Copied verbatim to
         ../data/gldas_runoff_periods_summary_{6,12}month.csv.
         Two basin names differ from this workspace's spelling and are
         remapped: "Ganges / Brahmaputra Basin" -> "Ganges / Bhramaputra
         Basin", "Murray Darling" -> "Murray".

Outputs (../figures/{6_month,12_month}/)
  figure_combined_wet_dry_period_length.png   two-panel boxplot (this script's purpose)
  periods_summary.csv                          every detected/loaded period, all basins x variables
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent

ET_DIR = REPO_ROOT / "data" / "processed" / "ET"
PRECIP_CSV = REPO_ROOT / "data" / "processed" / "precipitation" / "imerg_basin_monthly_precip.csv"
GLDAS_RUNOFF_PERIODS_CSV = {
    6:  _HERE.parent / "data" / "gldas_runoff_periods_summary_6month.csv",
    12: _HERE.parent / "data" / "gldas_runoff_periods_summary_12month.csv",
}

OUTPUT_DIR = _HERE.parent / "figures"

# ---------------------------------------------------------------------------
# Basins (canonical display name -> source filename), in the same order as
# ET_Wet_Dry_precip_rerun/scripts/whiplash33.py's BASINS dict.
# ---------------------------------------------------------------------------
ET_FILES = {
    "Amazon Basin":              "MODIS_ET_Amazon_2002_2024.csv",
    "California":                 "MODIS_ET_California_2002_2024.csv",
    "Colorado":                   "MODIS_ET_Colorado_2002_2024.csv",
    "Congo":                      "MODIS_ET_Congo_2002_2024.csv",
    "Danube":                     "MODIS_ET_Danube_2002_2024.csv",
    "Ganges / Bhramaputra Basin": "MODIS_ET_Ganga-Brahmaputra_2002_2024.csv",
    "Mekong Basin":               "MODIS_ET_Mekong_2002_2024.csv",
    "Mississippi / Missouri":     "MODIS_ET_Mississippi_2002_2024.csv",
    "Murray":                     "MODIS_ET_Murray_Darling_2002_2024.csv",
    "Nile":                       "MODIS_ET_Nile_2002_2024.csv",
    "Yangtze Basin":              "MODIS_ET_Yangtze_2002_2024 (1).csv",
}

# research_group's periods_summary CSVs spell two basins differently than
# this workspace; map their names -> this workspace's canonical names.
RESEARCH_GROUP_BASIN_MAP = {
    "Amazon Basin":               "Amazon Basin",
    "California":                  "California",
    "Colorado":                    "Colorado",
    "Congo":                       "Congo",
    "Danube":                      "Danube",
    "Ganges / Brahmaputra Basin":  "Ganges / Bhramaputra Basin",
    "Mekong Basin":                "Mekong Basin",
    "Mississippi / Missouri":      "Mississippi / Missouri",
    "Murray Darling":              "Murray",
    "Nile":                        "Nile",
    "Yangtze Basin":               "Yangtze Basin",
}

BASINS = list(ET_FILES.keys())

VARIABLES = ["ET", "Runoff", "Precip"]
VARIABLE_COLORS = {
    "ET": "seagreen",
    "Runoff": "steelblue",
    "Precip": "goldenrod",
}

BLOCK_CONFIGS = [
    ("6-Month", "6MS", "6_month", 6),
    ("12-Month", "AS-JAN", "12_month", 12),
]


def safe_name(basin: str) -> str:
    return basin.replace("/", "_").replace(" ", "_").replace("__", "_")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_et_basin(basin: str) -> pd.Series:
    df = pd.read_csv(ET_DIR / ET_FILES[basin])
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df = df.sort_values("date").set_index("date")
    return pd.to_numeric(df["ET_mm"], errors="coerce").dropna()


_PRECIP_WIDE = None


def _precip_wide() -> pd.DataFrame:
    global _PRECIP_WIDE
    if _PRECIP_WIDE is None:
        df = pd.read_csv(PRECIP_CSV)
        df["time"] = pd.to_datetime(df["time"])
        _PRECIP_WIDE = df.sort_values("time").set_index("time")
    return _PRECIP_WIDE


def load_precip_basin(basin: str) -> pd.Series:
    df = _precip_wide()
    return pd.to_numeric(df[basin], errors="coerce").dropna()


def compute_anomaly(series: pd.Series) -> pd.Series:
    climatology = series.groupby(series.index.month).mean()
    return series - series.index.map(lambda d: climatology[d.month])


def get_anomaly(basin: str, variable: str) -> pd.Series:
    if variable == "ET":
        return compute_anomaly(load_et_basin(basin))
    if variable == "Precip":
        return compute_anomaly(load_precip_basin(basin))
    raise ValueError(variable)


_GLDAS_RUNOFF_PERIODS: dict[int, pd.DataFrame] = {}


def load_gldas_runoff_periods(duration_unit_months: int) -> pd.DataFrame:
    """Pre-computed GLDAS runoff wet/dry periods (individual-month method,
    already run), pulled from research_group — not recomputed here. Basin
    names are remapped to this workspace's canonical spelling."""
    if duration_unit_months not in _GLDAS_RUNOFF_PERIODS:
        df = pd.read_csv(GLDAS_RUNOFF_PERIODS_CSV[duration_unit_months])
        df["basin"] = df["basin"].map(RESEARCH_GROUP_BASIN_MAP)
        _GLDAS_RUNOFF_PERIODS[duration_unit_months] = df
    return _GLDAS_RUNOFF_PERIODS[duration_unit_months]


# ---------------------------------------------------------------------------
# Wet/dry period detection — "individual_month" method (same algorithm as
# ET_wet_dry_periods.py's detect_consecutive_periods): each block is
# classified wet/dry purely by the sign of its own anomaly value (no
# running sum), and a period is a run of consecutive same-sign blocks. Any
# sign flip ends the streak immediately, regardless of magnitude.
# ---------------------------------------------------------------------------

def detect_consecutive_periods(anomaly: pd.Series) -> pd.DataFrame:
    vals = anomaly.values
    idx = anomaly.index
    n = len(vals)

    periods = []
    period_sign = None
    period_start = 0

    for i in range(n):
        y = vals[i]
        if np.isnan(y):
            continue

        sign_y = 1 if y >= 0 else -1
        if period_sign is None:
            period_sign = sign_y
            period_start = i
        elif sign_y != period_sign:
            periods.append({
                "start_idx": period_start, "end_idx": i - 1, "sign": period_sign,
            })
            period_sign = sign_y
            period_start = i

    if period_sign is not None:
        periods.append({"start_idx": period_start, "end_idx": n - 1, "sign": period_sign})

    records = []
    for p in periods:
        records.append({
            "type": "Wet" if p["sign"] > 0 else "Dry",
            "start_date": idx[p["start_idx"]],
            "end_date": idx[p["end_idx"]],
            "duration_blocks": p["end_idx"] - p["start_idx"] + 1,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_combined_boxplot(all_periods: pd.DataFrame, resolution_label: str,
                           duration_unit_months: int, out_dir: Path) -> None:
    """Two-panel (a) Wet / (b) Dry period-length boxplot. Each basin is a
    3-box cluster (ET, Runoff, Precip), same box/scatter/legend styling as
    ET_wet_dry_periods.py's plot_period_length_boxplot."""
    y_label = "Duration (Months)"

    slot_spacing = 0.85
    group_gap = 1.15
    box_width = 0.65

    positions_by_basin: list[list[float]] = []
    x = 1.0
    for _ in BASINS:
        positions_by_basin.append([x, x + slot_spacing, x + 2 * slot_spacing])
        x += 2 * slot_spacing + group_gap
    xticks = [gp[1] for gp in positions_by_basin]  # centered on the middle (Runoff) box

    fig, axes = plt.subplots(2, 1, figsize=(22, 13))
    rng = np.random.default_rng(42)

    for ax, period_type, panel_label in zip(
        axes, ["Wet", "Dry"], ["(a) Wet Period Length", "(b) Dry Period Length"]
    ):
        sub = all_periods[all_periods["type"] == period_type]

        flat_positions: list[float] = []
        flat_data: list[np.ndarray] = []
        flat_colors: list[str] = []
        for basin, group_positions in zip(BASINS, positions_by_basin):
            for variable, pos in zip(VARIABLES, group_positions):
                vals = (
                    sub.loc[(sub["basin"] == basin) & (sub["variable"] == variable),
                            "duration_blocks"] * duration_unit_months
                ).values
                flat_positions.append(pos)
                flat_data.append(vals)
                flat_colors.append(VARIABLE_COLORS[variable])

        box = ax.boxplot(
            flat_data, positions=flat_positions, widths=box_width,
            patch_artist=True, showmeans=True, showfliers=False,
            meanprops=dict(marker="x", markerfacecolor="black",
                           markeredgecolor="black", markersize=7, markeredgewidth=1.3),
            medianprops=dict(color="black", linewidth=1.1),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            boxprops=dict(edgecolor="black", linewidth=0.8),
        )
        for patch, color in zip(box["boxes"], flat_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.9)

        for pos, color, vals in zip(flat_positions, flat_colors, flat_data):
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals,
                       color=color, edgecolor="black",
                       linewidth=0.3, s=14, alpha=0.9, zorder=3)

        ax.set_xticks(xticks)
        ax.set_xticklabels(BASINS, rotation=30, ha="right", fontsize=8)
        ax.set_xlim(0.3, positions_by_basin[-1][-1] + 0.7)
        ax.set_xlabel("Basins")
        ax.set_ylabel(y_label)
        ax.set_title(panel_label, fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        for spine in ax.spines.values():
            spine.set_visible(True)

    legend_elements = [
        Patch(facecolor=VARIABLE_COLORS[v], edgecolor="black", alpha=0.9, label=v)
        for v in VARIABLES
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0), fontsize=10, framealpha=0.9)

    fig.suptitle(
        f"ET / GLDAS Runoff / IMERG Precip Anomaly — Consecutive Wet/Dry "
        f"Period Length (individual-month method), {resolution_label} blocks",
        fontsize=11, color="dimgray", y=1.035,
    )
    fig.text(
        0.5, -0.01,
        "Runoff periods pulled pre-computed from joshfuery/research_group "
        "(individual-month method); ET and Precip periods computed in this "
        "script from this workspace's own data using the same method.",
        ha="center", fontsize=7.5, color="dimgray",
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])

    out_path = out_dir / "figure_combined_wet_dry_period_length.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved boxplot PNG: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_resolution(label: str, rule: str, dirname: str, duration_unit_months: int) -> None:
    res_dir = OUTPUT_DIR / dirname
    res_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {label} block resolution — output folder: {res_dir} ===\n")

    all_periods = []
    for basin in BASINS:
        for variable in ["ET", "Precip"]:
            print(f"Processing: {basin} / {variable}")
            try:
                anomaly = get_anomaly(basin, variable)
                blocks = anomaly.resample(rule).sum()
                periods_df = detect_consecutive_periods(blocks)
                periods_df["basin"] = basin
                periods_df["variable"] = variable
                all_periods.append(periods_df)
            except Exception as exc:
                print(f"  ERROR ({basin} / {variable}): {exc}")

    print("Loading: Runoff (pre-computed, from research_group)")
    runoff_df = load_gldas_runoff_periods(duration_unit_months)
    runoff_df = runoff_df[runoff_df["basin"].isin(BASINS)].copy()
    runoff_df["variable"] = "Runoff"
    all_periods.append(runoff_df[["type", "start_date", "end_date", "duration_blocks", "basin", "variable"]])

    if not all_periods:
        print(f"No periods detected for {label}; skipping boxplot.\n")
        return

    all_df = pd.concat(all_periods, ignore_index=True)
    csv_path = res_dir / "periods_summary.csv"
    all_df.to_csv(csv_path, index=False)
    print(f"\nSaved periods summary CSV: {csv_path}")

    plot_combined_boxplot(all_df, label, duration_unit_months, res_dir)
    print(f"\n{label} done.\n")


def main():
    for label, rule, dirname, duration_unit_months in BLOCK_CONFIGS:
        run_resolution(label, rule, dirname, duration_unit_months)
    print("All resolutions done.")


if __name__ == "__main__":
    main()
