#!/usr/bin/env python3
"""
precip_wet_dry_periods.py

Wet/dry period identification for GPM IMERG basin precipitation anomaly data,
following the methodology of Madrigal, Bedri, Piechota et al. (2024), "Water
Whiplash in Mediterranean Regions of the World," Water 16(3), 450.
https://doi.org/10.3390/w16030450

This is a direct port of ET_wet_dry_periods.py to the IMERG monthly basin
precipitation time series produced by precip_ts.py. Only the data-loading
layer, axis labels, and output paths differ; the wet/dry detection algorithms
and figure layouts are identical.

Paper methodology (Section 2.2, Equations 1-4):
  y_i = x_i - x_bar                    (Eq 1) - deviation from the long-term
                                                 mean for period i
  sum(y_i for consecutive same-sign i) = A_j   (Eq 2) - a wet (or dry) period
                                                 is the running accumulation
                                                 of consecutive same-sign
                                                 deviations
  A dry or wet period continues until the accumulated deficit/surplus is
  switched by a single period whose opposite-sign deviation exceeds the
  accumulated magnitude - that switch is "whiplash" (Eq 3): a period ends
  and a new one begins there. A same-sign-flip period whose magnitude does
  NOT exceed the running total is simply absorbed into the ongoing period
  (it does not end it).
  IR_i = A_j / x_bar                   (Eq 4) - indicator ratio, used to
                                                 compare accumulated
                                                 deficit/surplus across
                                                 basins of different scale.

Adaptation to IMERG precip anomaly data: the paper works with a single
long-term mean of annual streamflow. Here, y_i is the precipitation anomaly
used throughout this project (raw IMERG basin-mean precip minus its
calendar-month climatology, which removes the seasonal cycle - the same
calculation as precip_ts.py / precip_results.py), summed to whichever block
resolution is being analyzed. x_bar is the long-term mean of the raw
(non-anomaly) precip series at that same resolution.

Run at four block resolutions per the request (1, 3, 6, and 12 months).

Two parallel analyses are produced at every resolution:

  accumulation/    The paper's method above (Eq 1-4): a period is a running
                    SUM of consecutive same-sign anomalies, and a same-sign
                    flip only ends the period ("whiplash") if its magnitude
                    exceeds the accumulated total so far.

  individual_month/  A second, non-accumulating analysis requested alongside
                    the above: each block is judged wet/dry purely by the
                    sign of its own anomaly value (no running sum), and a
                    period is simply how many blocks IN A ROW share that
                    sign - i.e. a consecutive wet-month or dry-month streak
                    count. Any sign flip ends the streak immediately,
                    regardless of magnitude.

For each basin, at each resolution, both analyses:
  - Detect wet/dry periods (accumulation: with whiplash transitions marked;
    individual_month: consecutive same-sign streaks).
  - Plot a period timeline diagnostic.
Aggregated across all basins, at each resolution, both analyses:
  - Produce a two-panel boxplot of wet period length (top) and dry period
    length (bottom), one box per basin - accumulation/ recreates the
    paper's Figure 2; individual_month/ is the same layout for streak length.

Data source: imerg_basin_monthly_timeseries.csv (raw, wide: one column/basin)
             imerg_basin_monthly_timeseries_anomalies.csv (used if present,
             else anomalies are recomputed from the raw CSV)
Outputs saved to: Wet_Dry_IMERG/
  1_month/    3_month/    6_month/    12_month/
  each contains:
    accumulation/
      period_timelines/            per-basin accumulated anomaly + period plot
      figure2_wet_dry_period_length.png   paper-style Figure 2 recreation
      periods_summary.csv          every detected accumulation period, all basins
    individual_month/
      period_timelines/            per-basin raw anomaly bars, colored by sign
      figure_consecutive_month_length.png   boxplot of consecutive-streak length
      periods_summary.csv          every detected streak, all basins
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_CSV = Path("imerg_basin_monthly_timeseries.csv")
ANOMALY_CSV = Path("imerg_basin_monthly_timeseries_anomalies.csv")
WET_DRY_DIR = Path("Wet_Dry_IMERG")

# label, resample rule, output dirname, duration unit (blocks -> months)
BLOCK_CONFIGS = [
    ("1-Month",  None,     "1_month",  1),
    ("3-Month",  "QS-JAN", "3_month",  3),
    ("6-Month",  "6MS",    "6_month",  6),
    ("12-Month", "12MS",   "12_month", 12),
]


def safe_name(s: str) -> str:
    """Filesystem-safe basin name (matches split_imerg_to_basins.py)."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s).strip("_")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
_RAW_DF: pd.DataFrame | None = None
_ANOM_DF: pd.DataFrame | None = None


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (raw, anomaly) wide DataFrames once, cached at module scope.

    Anomaly = raw minus calendar-month climatology (seasonal cycle removed),
    exactly as computed in precip_ts.py / precip_results.py. Uses the saved
    anomalies CSV when it exists, otherwise recomputes from the raw CSV."""
    global _RAW_DF, _ANOM_DF
    if _RAW_DF is not None:
        return _RAW_DF, _ANOM_DF

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw IMERG time series not found: {RAW_CSV}")
    raw = pd.read_csv(RAW_CSV, parse_dates=["time"], index_col="time").sort_index()

    if ANOMALY_CSV.exists():
        anom = pd.read_csv(ANOMALY_CSV, parse_dates=["time"], index_col="time").sort_index()
        anom = anom.reindex(columns=raw.columns)
    else:
        monthly_mean = raw.groupby(raw.index.month).transform("mean")
        anom = raw - monthly_mean

    _RAW_DF, _ANOM_DF = raw, anom
    return raw, anom


def get_basins() -> list[str]:
    """Basin column names, dropping any that are entirely NaN."""
    raw, _ = _load_frames()
    return [c for c in raw.columns if not raw[c].isna().all()]


BASIN_COLORS: dict[str, tuple] = {}


def _init_basin_colors() -> None:
    global BASIN_COLORS
    if BASIN_COLORS:
        return
    BASIN_COLORS = {
        basin: color
        for basin, color in zip(get_basins(), plt.get_cmap("tab20").colors)
    }


def get_block_series(basin: str, rule: str | None) -> tuple[pd.Series, pd.Series]:
    """Raw precip series and its calendar-month-climatology anomaly, aggregated
    to the requested block resolution (None = leave at Monthly)."""
    raw, anom = _load_frames()
    series = raw[basin].dropna()
    anomaly = anom[basin].reindex(series.index)
    if rule is None:
        return series, anomaly
    return series.resample(rule).sum(), anomaly.resample(rule).sum()


# ---------------------------------------------------------------------------
# Wet/dry period detection (paper Eq 1-3)
# ---------------------------------------------------------------------------

def detect_wet_dry_periods(anomaly: pd.Series) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Returns (periods_df, whiplash_flags, accumulated_series).

    accumulated_series tracks the running A_j at every block - used to plot
    the period timeline (mirrors the paper's Figure 5 "Ratio" line, but in
    raw accumulated-anomaly units rather than the indicator ratio)."""
    vals = anomaly.values
    idx = anomaly.index
    n = len(vals)

    periods = []
    whiplash_flags = np.zeros(n, dtype=bool)
    accumulated = np.full(n, np.nan)

    period_sign = None
    period_start = 0
    A = 0.0

    for i in range(n):
        y = vals[i]
        if np.isnan(y):
            accumulated[i] = A
            continue

        if period_sign is None:
            period_sign = 1 if y >= 0 else -1
            period_start = i
            A = y
        else:
            sign_y = 1 if y >= 0 else (-1 if y < 0 else 0)
            if sign_y == 0 or sign_y == period_sign:
                A += y
            elif abs(y) > abs(A):
                # Whiplash: current period ends at i-1, new one starts at i.
                periods.append({
                    "start_idx": period_start, "end_idx": i - 1,
                    "sign": period_sign, "accumulated": A,
                })
                whiplash_flags[i] = True
                period_sign = sign_y
                period_start = i
                A = y
            else:
                # Opposite-sign block absorbed into the ongoing period.
                A += y

        accumulated[i] = A

    periods.append({
        "start_idx": period_start, "end_idx": n - 1,
        "sign": period_sign, "accumulated": A,
    })

    records = []
    for p in periods:
        records.append({
            "type": "Wet" if p["sign"] > 0 else "Dry",
            "start_date": idx[p["start_idx"]],
            "end_date": idx[p["end_idx"]],
            "duration_blocks": p["end_idx"] - p["start_idx"] + 1,
            "accumulated": p["accumulated"],
        })
    periods_df = pd.DataFrame(records)

    accumulated_series = pd.Series(accumulated, index=idx)
    return periods_df, whiplash_flags, accumulated_series


def detect_consecutive_periods(anomaly: pd.Series) -> pd.DataFrame:
    """Non-accumulating counterpart to detect_wet_dry_periods: each block is
    classified wet/dry by the sign of its own anomaly value alone (no
    running sum), and a period is simply a run of consecutive blocks that
    share that sign - i.e. "how many months in a row are wet/dry." Any sign
    flip ends the streak immediately, regardless of magnitude."""
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
# Plots
# ---------------------------------------------------------------------------

def plot_individual_month_timeline(basin: str, anomaly: pd.Series, resolution_label: str,
                                  out_dir) -> None:
    """Diagnostic plot for the non-accumulating analysis: each block's own
    anomaly value as a bar, colored by sign - consecutive same-color bars
    are a wet or dry streak. No running sum involved."""
    fig, ax = plt.subplots(figsize=(14, 4.5))

    vals = anomaly.values
    colors = np.where(vals >= 0, "steelblue", "tomato")

    if len(anomaly.index) > 1:
        day_diffs = anomaly.index.to_series().diff().dropna().dt.days
        bar_width = max(day_diffs.median() * 0.8, 1)
    else:
        bar_width = 20

    ax.bar(anomaly.index, vals, width=bar_width, color=colors, align="center")
    ax.axhline(0, color="black", linewidth=0.7)

    ax.set_ylabel("Precipitation Anomaly (mm)")
    ax.set_xlabel("Date")
    ax.set_title(f"{basin} - Wet/Dry by Individual Month Value ({resolution_label})")

    legend_elements = [
        Patch(facecolor="steelblue", label="Wet Month (Anomaly >= 0)"),
        Patch(facecolor="tomato", label="Dry Month (Anomaly < 0)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.85)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    sname = safe_name(basin)
    out_path = out_dir / f"{sname}_period_timeline.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved individual-month timeline PNG: {out_path.name}")


def plot_period_timeline(basin: str, anomaly: pd.Series, accumulated: pd.Series,
                          resolution_label: str, out_dir) -> None:
    """Diagnostic plot: running accumulated surplus/deficit (shaded area +
    line), shaded by period sign - the figure used to build the periods
    table."""
    fig, ax = plt.subplots(figsize=(14, 4.5))

    wet_mask = accumulated.values >= 0
    ax.fill_between(accumulated.index, accumulated.values, 0,
                     where=wet_mask, color="steelblue", alpha=0.5, interpolate=True,
                     label="Accumulated Surplus (Wet)")
    ax.fill_between(accumulated.index, accumulated.values, 0,
                     where=~wet_mask, color="tomato", alpha=0.5, interpolate=True,
                     label="Accumulated Deficit (Dry)")
    ax.plot(accumulated.index, accumulated.values, color="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.7)

    ax.set_ylabel("Accumulated Surplus / Deficit (mm)")
    ax.set_xlabel("Date")
    ax.set_title(f"{basin} - Wet/Dry Period Accumulation ({resolution_label})")

    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.5, label="Wet Period (Surplus)"),
        Patch(facecolor="tomato", alpha=0.5, label="Dry Period (Deficit)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.85)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    sname = safe_name(basin)
    out_path = out_dir / f"{sname}_period_timeline.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved period timeline PNG: {out_path.name}")


def plot_period_length_boxplot(all_periods: pd.DataFrame, resolution_label: str,
                                duration_unit_months: int, out_dir, out_filename: str,
                                title: str) -> None:
    """Two-panel (a)/(b) boxplot of wet and dry period length, solid-colored
    boxes with a mean "X" marker and a jittered strip of every individual
    period's exact duration - recreates the paper's Figure 2 layout. Shared
    by both the accumulation analysis (period = running-sum duration) and the
    individual_month analysis (period = consecutive same-sign streak length);
    only the input periods, title, and output filename differ between the two.

    One deliberate deviation: the paper's y-axis is "Duration (Years)"
    because its data was strictly annual. This script runs at four block
    resolutions (1/3/6/12-month), so the y-axis unit here reflects
    whichever resolution produced the figure instead of forcing "Years"."""
    basins = list(get_basins())
    y_label = ("Duration (Months)" if duration_unit_months > 1
               else f"Duration ({resolution_label.lower()} blocks)")

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    rng = np.random.default_rng(42)

    for ax, period_type, panel_label in zip(
        axes, ["Wet", "Dry"], ["(a) Wet Period Length", "(b) Dry Period Length"]
    ):
        sub = all_periods[all_periods["type"] == period_type]
        data = [
            (sub.loc[sub["basin"] == b, "duration_blocks"] * duration_unit_months).values
            for b in basins
        ]
        positions = range(1, len(basins) + 1)

        box = ax.boxplot(
            data, positions=positions, widths=0.6,
            patch_artist=True, showmeans=True, showfliers=False,
            meanprops=dict(marker="x", markerfacecolor="black",
                           markeredgecolor="black", markersize=8, markeredgewidth=1.5),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            boxprops=dict(edgecolor="black", linewidth=0.8),
        )
        for patch, basin in zip(box["boxes"], basins):
            patch.set_facecolor(BASIN_COLORS[basin])
            patch.set_alpha(0.95)

        # Jittered strip of every individual period's duration (all points,
        # not just outliers) - mirrors the paper's dot-cloud styling.
        for pos, basin, vals in zip(positions, basins, data):
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals,
                       color=BASIN_COLORS[basin], edgecolor="black",
                       linewidth=0.3, s=18, alpha=0.9, zorder=3)

        counts = [len(d) for d in data]
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"{b}\n(n={n})" for b, n in zip(basins, counts)],
            rotation=30, ha="right", fontsize=8,
        )
        ax.set_xlabel("Basins")
        ax.set_ylabel(y_label)
        ax.set_title(panel_label, fontsize=12)
        ax.grid(True, axis="y", alpha=0.3)
        for spine in ax.spines.values():
            spine.set_visible(True)

    fig.suptitle(
        f"{title} - IMERG Precipitation Anomaly, {resolution_label} blocks",
        fontsize=11, color="dimgray",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = out_dir / out_filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved boxplot PNG: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_resolution(label: str, rule: str | None, dirname: str, duration_unit_months: int) -> None:
    res_dir = WET_DRY_DIR / dirname
    accum_dir = res_dir / "accumulation"
    accum_timeline_dir = accum_dir / "period_timelines"
    individual_dir = res_dir / "individual_month"
    individual_timeline_dir = individual_dir / "period_timelines"
    accum_timeline_dir.mkdir(parents=True, exist_ok=True)
    individual_timeline_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== {label} block resolution - output folder: {res_dir} ===\n")

    all_accum_periods = []
    all_individual_periods = []
    for basin in get_basins():
        print(f"Processing: {basin}")
        try:
            series, anomaly = get_block_series(basin, rule)
        except Exception as exc:
            print(f"  ERROR loading: {exc}")
            continue

        try:
            accum_periods_df, _whiplash_flags, accumulated = detect_wet_dry_periods(anomaly)
            accum_periods_df["basin"] = basin
            all_accum_periods.append(accum_periods_df)
            plot_period_timeline(basin, anomaly, accumulated, label, accum_timeline_dir)
        except Exception as exc:
            print(f"  ERROR (accumulation): {exc}")

        try:
            individual_periods_df = detect_consecutive_periods(anomaly)
            individual_periods_df["basin"] = basin
            all_individual_periods.append(individual_periods_df)
            plot_individual_month_timeline(basin, anomaly, label, individual_timeline_dir)
        except Exception as exc:
            print(f"  ERROR (individual_month): {exc}")

    if all_accum_periods:
        all_accum_df = pd.concat(all_accum_periods, ignore_index=True)
        csv_path = accum_dir / "periods_summary.csv"
        all_accum_df.to_csv(csv_path, index=False)
        print(f"\nSaved accumulation periods summary CSV: {csv_path.name}")
        plot_period_length_boxplot(
            all_accum_df, label, duration_unit_months, accum_dir,
            out_filename="figure2_wet_dry_period_length.png",
            title="Figure 2 (recreated)",
        )
    else:
        print(f"No accumulation periods detected for {label}; skipping boxplot.\n")

    if all_individual_periods:
        all_individual_df = pd.concat(all_individual_periods, ignore_index=True)
        csv_path = individual_dir / "periods_summary.csv"
        all_individual_df.to_csv(csv_path, index=False)
        print(f"Saved individual-month periods summary CSV: {csv_path.name}")
        plot_period_length_boxplot(
            all_individual_df, label, duration_unit_months, individual_dir,
            out_filename="figure_consecutive_month_length.png",
            title="Consecutive Wet/Dry Month Count",
        )
    else:
        print(f"No individual-month periods detected for {label}; skipping boxplot.\n")

    print(f"\n{label} done.\n")


def main():
    _load_frames()
    _init_basin_colors()
    for label, rule, dirname, duration_unit_months in BLOCK_CONFIGS:
        run_resolution(label, rule, dirname, duration_unit_months)

    print("All resolutions done.")


if __name__ == "__main__":
    main()
