#!/usr/bin/env python3
"""
whiplash33.py

Whiplash analysis using 33rd/66th percentile thresholds.

Methodology follows Whiplash_Study/code.ipynb (Whiplash_Identified_Plots) with:
  - 33rd and 66th percentile thresholds (instead of 10th/90th)
  - 3-month, 6-month, and 12-month panels combined into one PDF per basin
  - Basin statistics timeseries plots matching the basin_statistic_timeseries style

PRECIPITATION RERUN (adapted from the ET_Wet_Dry version in
joshfuery/research_group): instead of MODIS ET, this reads the
basin-averaged monthly GPM IMERG precipitation series from this workspace
(data/processed/precipitation/imerg_basin_monthly_precip.csv, wide format:
a `time` column plus one column per basin). Every downstream calculation
(monthly calendar climatology anomaly, p33/p66 whiplash detection, rolling
stats, decade counts) is byte-for-byte the same as the ET version — only
the input series changes. "ET" in variable names / axis labels now means
precipitation in mm.

Data source: GPM IMERG precip (../data/imerg_basin_monthly_precip.csv)
Outputs saved to: ../figures/whiplash33/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE       = Path(__file__).resolve().parent
PRECIP_CSV  = _HERE / ".." / "data" / "imerg_basin_monthly_precip.csv"
OUTPUT_DIR  = _HERE / ".." / "figures" / "whiplash33"

ID_STATS_DIR = OUTPUT_DIR / "identification_stats"
DECADE_DIR   = OUTPUT_DIR / "decade_counts"
ID_STATS_DIR.mkdir(parents=True, exist_ok=True)
DECADE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Basin → column name in the wide IMERG precip CSV. Keys are the exact
# column headers in imerg_basin_monthly_precip.csv so the mapping is an
# identity lookup (the workspace file spells two basins "Ganges /
# Bhramaputra Basin" and "Murray").
# ---------------------------------------------------------------------------
BASINS = {
    "Amazon Basin":               "Amazon Basin",
    "California":                  "California",
    "Colorado":                    "Colorado",
    "Congo":                       "Congo",
    "Danube":                      "Danube",
    "Ganges / Bhramaputra Basin":  "Ganges / Bhramaputra Basin",
    "Mekong Basin":                "Mekong Basin",
    "Mississippi / Missouri":      "Mississippi / Missouri",
    "Murray":                      "Murray",
    "Nile":                        "Nile",
    "Yangtze Basin":               "Yangtze Basin",
}

# Safe filename prefix for each basin
def safe_name(basin: str) -> str:
    return basin.replace("/", "_").replace(" ", "_").replace("__", "_")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_PRECIP_WIDE = None


def _precip_wide() -> pd.DataFrame:
    """Lazily load & cache the wide monthly IMERG precip table."""
    global _PRECIP_WIDE
    if _PRECIP_WIDE is None:
        df = pd.read_csv(PRECIP_CSV)
        df["time"] = pd.to_datetime(df["time"])
        _PRECIP_WIDE = df.sort_values("time").set_index("time")
    return _PRECIP_WIDE


def load_basin(basin: str) -> pd.Series:
    df = _precip_wide()
    series = pd.to_numeric(df[BASINS[basin]], errors="coerce").dropna()
    series.index.name = "date"
    return series


def compute_anomaly(series: pd.Series) -> pd.Series:
    climatology = series.groupby(series.index.month).mean()
    return series - series.index.map(lambda d: climatology[d.month])


# ---------------------------------------------------------------------------
# Whiplash detection (33rd / 66th percentile thresholds)
# ---------------------------------------------------------------------------

# (rule=None means no resampling — pass anomaly directly, matching whiplash_136)
WINDOW_CONFIGS = [
    ("Monthly",  None,      28,  "%b %Y"),
    ("3-Month",  "QS-JAN",  80,  "%b %Y"),
    ("6-Month",  "6MS",     160, "%b %Y"),
    ("Annual",   "AS-JAN",  300, "%Y"),
]


def detect_whiplash_33(anomaly_blocks: pd.Series):
    """Exact same logic as whiplash_136, with p33/p66 instead of p10/p90."""
    vals = anomaly_blocks.values
    n    = len(vals)

    p66 = float(np.nanpercentile(vals, 66))
    p33 = float(np.nanpercentile(vals, 33))

    bar_state = np.select(
        [vals > p66, vals < p33],
        [1, -1],
        default=0,
    )

    is_whiplash = np.zeros(n, dtype=bool)
    for i in range(1, n):
        prev, curr = int(bar_state[i - 1]), int(bar_state[i])
        if prev != 0 and curr != 0 and prev != curr:
            is_whiplash[i] = True

    return is_whiplash, p33, p66


def draw_whiplash_panel(ax, anomaly_blocks: pd.Series, win_label: str,
                        bar_width_days: int, date_fmt: str):
    vals  = anomaly_blocks.values
    dates = anomaly_blocks.index
    n = len(vals)

    is_whiplash, p33, p66 = detect_whiplash_33(anomaly_blocks)

    for i in range(n):
        val  = vals[i]
        date = dates[i]
        if np.isnan(val):
            continue
        color = "steelblue" if val >= 0 else "tomato"
        alpha = 1.0 if is_whiplash[i] else 0.3
        ax.bar(date, val, width=bar_width_days, color=color, alpha=alpha,
               align="center")

        if is_whiplash[i]:
            offset = 0.08 * abs(val) if abs(val) > 0 else 0.05
            star_y = val + offset if val >= 0 else val - offset
            ax.plot(date, star_y, marker="*", color="#d4af37",
                    markersize=12, markeredgecolor="black",
                    markeredgewidth=0.5, linestyle="none", zorder=5)

    ax.axhline(p66, color="navy",    linewidth=1, linestyle="--",
               label=f"66th pct ({p66:.2f})")
    ax.axhline(p33, color="darkred", linewidth=1, linestyle="--",
               label=f"33rd pct ({p33:.2f})")
    ax.axhline(0,   color="black",   linewidth=0.8)

    ax.set_title(f"Whiplash Identification — {win_label} Anomaly Blocks "
                 f"(33rd/66th pct thresholds)",
                 fontsize=11)
    ax.set_ylabel("Precip Anomaly (mm)")
    ax.grid(True, alpha=0.25, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_elements = [
        Patch(facecolor="steelblue",         label="Wet (positive anomaly)"),
        Patch(facecolor="tomato",            label="Dry (negative anomaly)"),
        Patch(facecolor="gray", alpha=0.3,   label="Non-whiplash (faded)"),
        plt.Line2D([0], [0], color="navy",    linestyle="--",
                   label=f"66th pct ({p66:.2f})"),
        plt.Line2D([0], [0], color="darkred", linestyle="--",
                   label=f"33rd pct ({p33:.2f})"),
        plt.Line2D([0], [0], marker="*", color="#d4af37", linestyle="none",
                   markersize=10, markeredgecolor="black", markeredgewidth=0.5,
                   label="Whiplash event"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.8)


# ---------------------------------------------------------------------------
# Whiplash PDF: 3-panel (3m / 6m / 12m) per basin
# ---------------------------------------------------------------------------

def plot_whiplash_pdf(basin: str, anomaly: pd.Series) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(20, 22), sharex=False)
    sname = safe_name(basin)
    fig.suptitle(f"{basin} — Whiplash Identified (33rd/66th pct threshold)",
                 fontsize=14, fontweight="bold", y=0.995)

    for ax, (win_label, rule, bar_width, date_fmt) in zip(axes, WINDOW_CONFIGS):
        blocks = anomaly.copy() if rule is None else anomaly.resample(rule).sum()
        draw_whiplash_panel(ax, blocks, win_label, bar_width, date_fmt)

        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=45)

    axes[-1].set_xlabel("Date")
    plt.tight_layout(rect=[0, 0, 1, 0.995])

    out_path = ID_STATS_DIR / f"{sname}_whiplash33.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved whiplash PDF: {out_path.name}")


# ---------------------------------------------------------------------------
# Basin statistics timeseries (matching basin_statistic_timeseries style)
# ---------------------------------------------------------------------------

def plot_stats_timeseries(basin: str, series: pd.Series, anomaly: pd.Series) -> None:
    # Rolling stats on raw precip — 12-month lookback window for all three
    rolling_12m = series.shift(1).rolling(window=12, min_periods=12)

    autocorr = rolling_12m.apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )
    skewness = rolling_12m.skew()
    variance = rolling_12m.apply(
        lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else np.nan,
        raw=True,
    )

    sname = safe_name(basin)
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)
    fig.suptitle(f"{basin} — Rolling Statistics Time Series", fontsize=13)

    axes[0].plot(autocorr.index, autocorr.values, color="blue", linewidth=1)
    axes[0].set_ylabel("Autocorrelation")
    axes[0].legend(["Autocorrelation (lag 1)"], loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(variance.index, variance.values, color="green", linewidth=1)
    axes[1].set_ylabel("Variance")
    axes[1].legend(["Variance"], loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    axes[2].plot(skewness.index, skewness.values, color="red", linewidth=1)
    axes[2].set_ylabel("Skewness")
    axes[2].legend(["Skewness"], loc="upper right", fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    axes[3].plot(anomaly.index, anomaly.values, color="purple", linewidth=1)
    axes[3].set_ylabel("Anomaly")
    axes[3].set_xlabel("Date")
    axes[3].legend(["Anomaly"], loc="upper right", fontsize=9)
    axes[3].grid(True, alpha=0.3)
    axes[3].spines["top"].set_visible(False)
    axes[3].spines["right"].set_visible(False)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        plt.setp(ax.get_xticklabels(), visible=True)

    plt.tight_layout()
    out_path = ID_STATS_DIR / f"{sname}_stats_timeseries.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved stats PNG:    {out_path.name}")


# ---------------------------------------------------------------------------
# Combined 7-panel figure: whiplash identification + rolling statistics
# ---------------------------------------------------------------------------

def plot_combined_figure(basin: str, series: pd.Series, anomaly: pd.Series) -> None:
    """7-panel combined figure: whiplash identification (3) on top, rolling stats (4) below."""
    rolling_12m = series.shift(1).rolling(window=12, min_periods=12)
    autocorr = rolling_12m.apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
    skewness = rolling_12m.skew()
    variance = rolling_12m.apply(
        lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else np.nan, raw=True
    )

    sname = safe_name(basin)

    height_ratios = [1.5, 1.0, 1.0, 1.0, 0.75, 0.75, 0.75, 0.75]
    fig, axes = plt.subplots(
        8, 1,
        figsize=(20, 36),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.60},
    )
    fig.suptitle(
        f"{basin} — Whiplash Identification (p33/p66) + Rolling Statistics (12-month window)",
        fontsize=14, fontweight="bold", y=1.001,
    )

    # Panels 0–3: whiplash bar charts (Monthly / 3-Month / 6-Month / Annual)
    for ax, (win_label, rule, bar_width, date_fmt) in zip(axes[:4], WINDOW_CONFIGS):
        blocks = anomaly.copy() if rule is None else anomaly.resample(rule).sum()
        draw_whiplash_panel(ax, blocks, win_label, bar_width, date_fmt)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=45)

    # Section divider on panel 4
    axes[4].set_title(
        "── Rolling Statistics (12-month backward-looking window) ──",
        fontsize=10, color="dimgray", pad=8,
    )

    # Panel 4: autocorrelation
    axes[4].plot(autocorr.index, autocorr.values, color="steelblue", linewidth=1.2)
    axes[4].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[4].set_ylabel("Autocorr (lag-1)")
    axes[4].legend(["Rolling Autocorrelation (lag-1)"], loc="upper right", fontsize=9)
    axes[4].grid(True, alpha=0.25, axis="y")
    axes[4].spines["top"].set_visible(False)
    axes[4].spines["right"].set_visible(False)

    # Panel 5: variance
    axes[5].plot(variance.index, variance.values, color="seagreen", linewidth=1.2)
    axes[5].set_ylabel("Variance (mm²)")
    axes[5].legend(["Rolling Variance"], loc="upper right", fontsize=9)
    axes[5].grid(True, alpha=0.25, axis="y")
    axes[5].spines["top"].set_visible(False)
    axes[5].spines["right"].set_visible(False)

    # Panel 6: skewness
    axes[6].plot(skewness.index, skewness.values, color="firebrick", linewidth=1.2)
    axes[6].axhline(0, color="black", linewidth=0.7, linestyle="--")
    axes[6].set_ylabel("Skewness")
    axes[6].legend(["Rolling Skewness"], loc="upper right", fontsize=9)
    axes[6].grid(True, alpha=0.25, axis="y")
    axes[6].spines["top"].set_visible(False)
    axes[6].spines["right"].set_visible(False)

    # Panel 7: monthly anomaly
    axes[7].plot(anomaly.index, anomaly.values, color="mediumpurple", linewidth=1.0)
    axes[7].axhline(0, color="black", linewidth=0.7)
    axes[7].set_ylabel("Precip Anomaly (mm)")
    axes[7].set_xlabel("Date")
    axes[7].legend(["Monthly Precip Anomaly"], loc="upper right", fontsize=9)
    axes[7].grid(True, alpha=0.25, axis="y")
    axes[7].spines["top"].set_visible(False)
    axes[7].spines["right"].set_visible(False)

    for ax in axes[4:]:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    out_path = ID_STATS_DIR / f"{sname}_combined.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved combined PNG:  {out_path.name}")


# ---------------------------------------------------------------------------
# Rolling decade whiplash counts (Monthly blocks)
# ---------------------------------------------------------------------------

DECADE_SPAN_YEARS = 10  # window is [start, start + DECADE_SPAN_YEARS] inclusive


def plot_decade_whiplash_counts(basin: str, anomaly: pd.Series) -> None:
    """For each rolling 10-year window (2002-2012, 2003-2013, ...), count how
    many Monthly-block whiplash events (33rd/66th pct) fall inside it."""
    is_whiplash, _, _ = detect_whiplash_33(anomaly)
    whiplash_years = anomaly.index[is_whiplash].year

    min_year = int(anomaly.index.year.min())
    max_year = int(anomaly.index.year.max())

    starts = list(range(min_year, max_year - DECADE_SPAN_YEARS + 1))
    if not starts:
        print(f"  Skipped decade-count plot ({basin}): not enough years "
              f"({min_year}-{max_year}) for a {DECADE_SPAN_YEARS}-year window")
        return

    counts = [
        int(((whiplash_years >= start) & (whiplash_years <= start + DECADE_SPAN_YEARS)).sum())
        for start in starts
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(starts, counts, marker="o", color="darkorange", linewidth=1.5)

    ax.set_xticks(starts)
    ax.set_xticklabels(
        [f"{s}–{s + DECADE_SPAN_YEARS}" for s in starts],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_xlabel("Rolling 10-Year Window")
    ax.set_ylabel("Number of Whiplash Events")
    ax.set_title(
        f"{basin} — Whiplash Events per Rolling Decade (Monthly blocks, 33rd/66th pct)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    sname = safe_name(basin)
    out_path = DECADE_DIR / f"{sname}_decade_whiplash_counts.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved decade count PNG: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Output folder: {OUTPUT_DIR}\n")

    for basin in BASINS:
        print(f"Processing: {basin}")

        try:
            series  = load_basin(basin)
            anomaly = compute_anomaly(series)
        except Exception as exc:
            print(f"  ERROR loading data: {exc}")
            continue

        try:
            plot_whiplash_pdf(basin, anomaly)
        except Exception as exc:
            print(f"  ERROR whiplash plot: {exc}")

        try:
            plot_stats_timeseries(basin, series, anomaly)
        except Exception as exc:
            print(f"  ERROR stats plot: {exc}")

        try:
            plot_combined_figure(basin, series, anomaly)
        except Exception as exc:
            print(f"  ERROR combined plot: {exc}")

        try:
            plot_decade_whiplash_counts(basin, anomaly)
        except Exception as exc:
            print(f"  ERROR decade count plot: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
