#!/usr/bin/env python3
"""
precip_whiplash33_ml_prediction.py

Trains and compares machine learning models that predict whether a given
period will be a "whiplash" event, using the same 33rd/66th percentile
whiplash definition as whiplash33.py.

This is a direct port of ET_whiplash33_ml_prediction.py to the GPM IMERG
monthly basin precipitation time series produced by precip_ts.py. The ET
version ran the same methodology across two data sources (MODIS ET and GPM
IMERG precipitation); this clone keeps only the precipitation source, so it
depends on nothing outside this workspace (the whiplash33 helpers it needs -
safe_name, compute_anomaly, detect_whiplash_33 - are inlined below).

The methodology still runs across one dimension:
  - Block resolution (see BLOCK_CONFIGS): Monthly anomaly blocks (the
    original analysis) and 3-Month anomaly blocks (summed the same way
    whiplash33.py's "3-Month" panel does).
Nothing about the modeling approach changes between the two - only the block
resolution of the series being fed into it.

For each basin, at each resolution:
  - Builds a leakage-safe feature table from the anomaly series - every
    feature for period t is built only from data through period t-1 (lagged
    anomalies, lagged raw value, 3-/6-/12-period trailing autocorrelation /
    variance / skewness, and calendar month).
  - Evaluates three classifiers (Logistic Regression, Random Forest,
    Gradient Boosting) two ways:
      1. Holdout - chronological 80/20 split (train on the earlier period,
         test on the later, untouched period).
      2. Leave-one-year-out cross-validation - hold out all periods of one
         calendar year, train on every other year, repeat for every year,
         then pool the out-of-fold predictions into one set of metrics.
  - Fits each model once on the full record (per basin) purely to read off
    feature importance: standardized coefficients for Logistic Regression,
    impurity-based importances for the two tree ensembles.
  - Checks each model for overfitting by comparing its metrics on the
    holdout train split against the holdout test split: accuracy, precision,
    recall, F1, ROC-AUC, RMSE (root Brier score: predicted probability vs.
    actual 0/1 outcome), and log loss. A model that scores much better on
    train than on test is overfit - memorizing the training periods rather
    than learning a pattern that generalizes.

Random Forest and Gradient Boosting are both intentionally constrained
(shallow max_depth, min_samples_leaf/split, Gradient Boosting subsampling)
to keep them from overfitting the relatively small amount of training data
available in some basins.

NOTE on the whiplash label itself: like the rest of whiplash33.py, the
33rd/66th percentile thresholds used to decide whether a period is
"extreme" are computed once over each basin's full record. That is
consistent with how whiplash events are defined everywhere else in this
project, but it does mean the *label* is informed by the full record even
though the *features* used to predict it never look past period t-1.

Data source: imerg_basin_monthly_timeseries.csv (GPM IMERG, wide: one
             column per basin), produced by precip_ts.py.

Outputs saved to: Wet_Dry_IMERG/ML/
  monthly/    analysis at Monthly anomaly-block resolution
  3_month/    identical methodology, at 3-Month anomaly-block resolution
  each folder contains:
    holdout/              chronological 80/20 split results
    loocv/                 leave-one-year-out cross-validation results
    feature_importance/    coefficients / importances, per basin and averaged
    overfitting/            train-vs-test metric comparison, per basin and averaged
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------
PRECIP_CSV = Path("imerg_basin_monthly_timeseries.csv")
ML_DIR = Path("Wet_Dry_IMERG") / "ML"

# Same methodology, run once per block resolution. `rule` is the pandas
# resample rule applied to the Monthly anomaly (and raw value) series before
# building features - None means "use the Monthly series as-is", matching
# whiplash33.py's WINDOW_CONFIGS convention exactly.
BLOCK_CONFIGS = [
    {"label": "Monthly", "rule": None, "dirname": "monthly"},
    {"label": "3-Month", "rule": "QS-JAN", "dirname": "3_month"},
]


# ---------------------------------------------------------------------------
# whiplash33.py helpers, inlined (this clone has no whiplash33 import)
# ---------------------------------------------------------------------------

def safe_name(basin: str) -> str:
    return basin.replace("/", "_").replace(" ", "_").replace("__", "_")


def compute_anomaly(series: pd.Series) -> pd.Series:
    """Raw value minus its calendar-month climatology (seasonal cycle
    removed) - identical to whiplash33.compute_anomaly / precip_ts.py."""
    climatology = series.groupby(series.index.month).mean()
    return series - series.index.map(lambda d: climatology[d.month])


def detect_whiplash_33(anomaly_blocks: pd.Series):
    """33rd/66th percentile whiplash flag: a block is 'wet-extreme' (state
    +1) above p66, 'dry-extreme' (-1) below p33, neutral otherwise; a
    whiplash is any block whose extreme state is the opposite sign of the
    previous block's extreme state. Thresholds are computed over the full
    record. Exact port of whiplash33.detect_whiplash_33."""
    vals = anomaly_blocks.values
    n = len(vals)

    p66 = float(np.nanpercentile(vals, 66))
    p33 = float(np.nanpercentile(vals, 33))

    bar_state = np.select([vals > p66, vals < p33], [1, -1], default=0)

    is_whiplash = np.zeros(n, dtype=bool)
    for i in range(1, n):
        prev, curr = int(bar_state[i - 1]), int(bar_state[i])
        if prev != 0 and curr != 0 and prev != curr:
            is_whiplash[i] = True

    return pd.Series(is_whiplash, index=anomaly_blocks.index), p33, p66


# ---------------------------------------------------------------------------
# Precipitation data source (GPM IMERG basin means)
# ---------------------------------------------------------------------------
_PRECIP_DF: pd.DataFrame | None = None


def _load_precip_frame() -> pd.DataFrame:
    global _PRECIP_DF
    if _PRECIP_DF is None:
        if not PRECIP_CSV.exists():
            raise FileNotFoundError(f"IMERG basin time series not found: {PRECIP_CSV}")
        df = pd.read_csv(PRECIP_CSV)
        df["time"] = pd.to_datetime(df["time"])
        _PRECIP_DF = df.sort_values("time").set_index("time")
    return _PRECIP_DF


def load_basin_precip(basin: str) -> pd.Series:
    df = _load_precip_frame()
    series = pd.to_numeric(df[basin], errors="coerce").dropna()
    series.index.name = None
    return series


def get_basins() -> list[str]:
    """Basin column names, dropping any that are entirely NaN."""
    df = _load_precip_frame()
    return [c for c in df.columns if not df[c].isna().all()]


TEST_FRACTION   = 0.2          # last 20% of periods (chronological) held out for testing
ROLLING_WINDOWS = [3, 6, 12]   # in units of periods - trailing windows for autocorr/variance/skew

ROLLING_FEATURE_COLS = [
    f"rolling_{stat}_{w}"
    for w in ROLLING_WINDOWS
    for stat in ("autocorr", "variance", "skew")
]

FEATURE_COLS = [
    "anomaly_lag1", "anomaly_lag2", "anomaly_lag3",
    "raw_lag1", "bar_state_lag1",
    *ROLLING_FEATURE_COLS,
    "month_sin", "month_cos",
]

MODELS = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]),
    "Random Forest": Pipeline([
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=4,
            min_samples_leaf=5, min_samples_split=10,
            class_weight="balanced", random_state=42,
        )),
    ]),
    "Gradient Boosting": Pipeline([
        ("scale", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            max_depth=2, min_samples_leaf=5, min_samples_split=10,
            subsample=0.8, random_state=42,
        )),
    ]),
}

MODEL_COLORS = {
    "Logistic Regression": "steelblue",
    "Random Forest": "seagreen",
    "Gradient Boosting": "firebrick",
}


def make_dirs(root: Path) -> dict:
    dirs = {
        "root": root,
        "holdout": root / "holdout",
        "loocv": root / "loocv",
        "feature_importance": root / "feature_importance",
        "overfitting": root / "overfitting",
    }
    for key, path in dirs.items():
        if key != "root":
            path.mkdir(parents=True, exist_ok=True)
    return dirs


def _fit(name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> None:
    """Fit a pipeline, giving Gradient Boosting manual balanced sample
    weights since (unlike the other two) it has no class_weight param."""
    if name == "Gradient Boosting":
        weights = compute_sample_weight("balanced", y)
        pipeline.fit(X, y, clf__sample_weight=weights)
    else:
        pipeline.fit(X, y)


# ---------------------------------------------------------------------------
# Data loading at a given block resolution
# ---------------------------------------------------------------------------

def get_block_series(basin: str, rule: str | None) -> tuple[pd.Series, pd.Series]:
    """Return (series, anomaly) at the requested block resolution. The
    anomaly is always computed from the Monthly climatology first (so the
    "extreme month" comparison stays meaningful), then both series and
    anomaly are summed into blocks - exactly how whiplash33.py's
    WINDOW_CONFIGS aggregate the Monthly anomaly into 3-Month blocks."""
    series = load_basin_precip(basin)
    anomaly = compute_anomaly(series)
    if rule is None:
        return series, anomaly
    return series.resample(rule).sum(), anomaly.resample(rule).sum()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(series: pd.Series, anomaly: pd.Series) -> pd.DataFrame:
    """Leakage-safe feature table: every feature for period t uses only data
    through period t-1. Target (is_whiplash) is period t's whiplash flag."""
    is_whiplash, p33, p66 = detect_whiplash_33(anomaly)
    bar_state = pd.Series(
        np.select([anomaly.values > p66, anomaly.values < p33], [1, -1], default=0),
        index=anomaly.index,
    )

    df = pd.DataFrame(index=anomaly.index)
    df["anomaly_lag1"] = anomaly.shift(1)
    df["anomaly_lag2"] = anomaly.shift(2)
    df["anomaly_lag3"] = anomaly.shift(3)
    df["raw_lag1"]       = series.shift(1)
    df["bar_state_lag1"] = bar_state.shift(1)

    for w in ROLLING_WINDOWS:
        rolling = series.shift(1).rolling(window=w, min_periods=w)
        df[f"rolling_autocorr_{w}"] = rolling.apply(
            lambda x: pd.Series(x).autocorr(lag=1), raw=False
        )
        df[f"rolling_variance_{w}"] = rolling.apply(
            lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else np.nan, raw=True
        )
        df[f"rolling_skew_{w}"] = rolling.skew()

    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    df["is_whiplash"] = is_whiplash.astype(int)

    return df[FEATURE_COLS + ["is_whiplash"]].dropna()


def time_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Chronological split - no shuffling - so the test period is strictly
    later in time than the train period."""
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


# ---------------------------------------------------------------------------
# Evaluation 1: chronological holdout split
# ---------------------------------------------------------------------------

def evaluate_holdout(name, pipeline, X_train, y_train, X_test, y_test):
    _fit(name, pipeline, X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else np.nan,
    }
    return metrics, y_proba


def run_basin_holdout(basin: str, features: pd.DataFrame, dirs: dict) -> pd.DataFrame:
    train, test = time_split(features)

    if train["is_whiplash"].nunique() < 2 or test["is_whiplash"].nunique() < 2:
        raise ValueError(
            f"not enough whiplash/non-whiplash examples in the train or test "
            f"split (train pos={int(train['is_whiplash'].sum())}/{len(train)}, "
            f"test pos={int(test['is_whiplash'].sum())}/{len(test)})"
        )

    X_train, y_train = train[FEATURE_COLS], train["is_whiplash"]
    X_test,  y_test  = test[FEATURE_COLS],  test["is_whiplash"]

    rows = []
    proba_by_model = {}
    for name, pipeline in MODELS.items():
        metrics, y_proba = evaluate_holdout(name, pipeline, X_train, y_train, X_test, y_test)
        metrics["basin"] = basin
        rows.append(metrics)
        proba_by_model[name] = y_proba

    metrics_df = pd.DataFrame(rows)

    plot_model_comparison(basin, metrics_df, dirs)
    for name, y_proba in proba_by_model.items():
        plot_prediction_timeline(basin, test.index, y_test, y_proba, name, dirs)

    return metrics_df


# ---------------------------------------------------------------------------
# Evaluation 2: leave-one-year-out cross-validation
# ---------------------------------------------------------------------------

def leave_one_year_out(features: pd.DataFrame, name: str, pipeline: Pipeline) -> pd.Series:
    """Hold out every period of one calendar year at a time, train on all
    other years, predict the held-out year, repeat for every year. Returns
    out-of-fold predicted probabilities indexed like `features` (NaN for any
    year skipped because its training fold had only one class)."""
    X = features[FEATURE_COLS]
    y = features["is_whiplash"]
    years = features.index.year

    oof_proba = pd.Series(np.nan, index=features.index)
    for yr in sorted(years.unique()):
        test_mask  = years == yr
        train_mask = ~test_mask

        y_train_fold = y[train_mask]
        if y_train_fold.nunique() < 2:
            continue  # can't fit a classifier on a single class

        X_train_fold = X[train_mask]
        X_test_fold  = X[test_mask]

        _fit(name, pipeline, X_train_fold, y_train_fold)
        oof_proba.loc[X_test_fold.index] = pipeline.predict_proba(X_test_fold)[:, 1]

    return oof_proba


def run_basin_loocv(basin: str, features: pd.DataFrame, dirs: dict) -> pd.DataFrame:
    y_full = features["is_whiplash"]
    n_years_total = features.index.year.nunique()

    rows = []
    oof_by_model = {}
    for name, pipeline in MODELS.items():
        oof_proba = leave_one_year_out(features, name, pipeline)
        valid = oof_proba.notna()

        if valid.sum() == 0 or y_full[valid].nunique() < 2:
            raise ValueError(f"LOOCV produced no usable out-of-fold predictions for {name}")

        y_true  = y_full[valid]
        y_proba = oof_proba[valid]
        y_pred  = (y_proba >= 0.5).astype(int)

        rows.append({
            "model": name,
            "basin": basin,
            "n_years_evaluated": int(features.index.year[valid].nunique()),
            "n_years_total": int(n_years_total),
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
            "roc_auc":   roc_auc_score(y_true, y_proba) if y_true.nunique() > 1 else np.nan,
        })
        oof_by_model[name] = oof_proba

    metrics_df = pd.DataFrame(rows)

    plot_loocv_model_comparison(basin, metrics_df, dirs)
    for name, oof_proba in oof_by_model.items():
        plot_loocv_timeline(basin, features.index, y_full, oof_proba, name, dirs)

    return metrics_df


# ---------------------------------------------------------------------------
# Feature importance / coefficients (fit once on the full record per basin)
# ---------------------------------------------------------------------------

def run_basin_feature_importance(basin: str, features: pd.DataFrame, dirs: dict) -> pd.DataFrame:
    X = features[FEATURE_COLS]
    y = features["is_whiplash"]

    if y.nunique() < 2:
        raise ValueError("only one class present across the full record")

    rows = []
    for name, pipeline in MODELS.items():
        _fit(name, pipeline, X, y)

        clf = pipeline.named_steps["clf"]
        values = clf.coef_[0] if hasattr(clf, "coef_") else clf.feature_importances_
        values = np.asarray(values, dtype=float)

        max_abs = np.abs(values).max()
        abs_normalized = np.abs(values) / max_abs if max_abs > 0 else np.zeros_like(values)

        for feat, val, norm in zip(FEATURE_COLS, values, abs_normalized):
            rows.append({
                "basin": basin, "model": name, "feature": feat,
                "value": val, "abs_normalized": norm,
            })

    importance_df = pd.DataFrame(rows)
    plot_feature_importance(basin, importance_df, dirs)
    return importance_df


# ---------------------------------------------------------------------------
# Overfitting check: train-split metrics vs. test-split metrics
# ---------------------------------------------------------------------------

def evaluate_split(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Score an already-fitted pipeline on a given split. RMSE here is the
    root Brier score - RMSE between predicted probability and the actual
    0/1 outcome, a standard way to score a probabilistic classifier."""
    y_pred  = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    return {
        "accuracy":  accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall":    recall_score(y, y_pred, zero_division=0),
        "f1":        f1_score(y, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y, y_proba) if y.nunique() > 1 else np.nan,
        "rmse":      float(np.sqrt(np.mean((y.values - y_proba) ** 2))),
        "log_loss":  log_loss(y, y_proba, labels=[0, 1]),
    }


def run_basin_overfit_check(basin: str, features: pd.DataFrame, dirs: dict) -> pd.DataFrame:
    """Fit each model on the same chronological train split used for the
    holdout evaluation, then score it on train AND test. A big gap between
    the two (train much better than test) is the signature of overfitting."""
    train, test = time_split(features)

    if train["is_whiplash"].nunique() < 2 or test["is_whiplash"].nunique() < 2:
        raise ValueError(
            f"not enough whiplash/non-whiplash examples in the train or test "
            f"split (train pos={int(train['is_whiplash'].sum())}/{len(train)}, "
            f"test pos={int(test['is_whiplash'].sum())}/{len(test)})"
        )

    X_train, y_train = train[FEATURE_COLS], train["is_whiplash"]
    X_test,  y_test  = test[FEATURE_COLS],  test["is_whiplash"]

    rows = []
    for name, pipeline in MODELS.items():
        _fit(name, pipeline, X_train, y_train)

        train_metrics = evaluate_split(pipeline, X_train, y_train)
        train_metrics.update(model=name, basin=basin, split="train")
        rows.append(train_metrics)

        test_metrics = evaluate_split(pipeline, X_test, y_test)
        test_metrics.update(model=name, basin=basin, split="test")
        rows.append(test_metrics)

    metrics_df = pd.DataFrame(rows)
    plot_overfit_check(basin, metrics_df, dirs)
    return metrics_df


# ---------------------------------------------------------------------------
# Plots - shared helpers
# ---------------------------------------------------------------------------

def _grouped_bar(ax, metrics_df: pd.DataFrame, value_label: str):
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metric_cols))
    width = 0.8 / len(metrics_df)

    for i, (_, row) in enumerate(metrics_df.iterrows()):
        vals = [row[m] for m in metric_cols]
        ax.bar(x + i * width, vals, width=width,
               label=row["model"], color=MODEL_COLORS.get(row["model"]))

    # Accuracy measures something different from the other four (it's inflated
    # by how rare whiplash periods are) - separate it visually so it isn't
    # read as directly comparable to Precision/Recall/F1/ROC-AUC. Each
    # category's bar cluster always spans [tick, tick + 0.8], so 0.9 sits in
    # the gap right after the Accuracy cluster and before the Precision one.
    ax.axvline(0.9, color="gray", linewidth=0.8, linestyle=":")

    ax.set_xticks(x + width * (len(metrics_df) - 1) / 2)
    ax.set_xticklabels(["Accuracy*", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(value_label)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.0, -0.16,
            "* Accuracy counts non-whiplash periods too, so it can look high "
            "even when a model misses most events - Precision/Recall/F1/ROC-AUC are class-imbalance-aware.",
            transform=ax.transAxes, fontsize=7, color="dimgray", ha="left")


def _feature_importance_axes(fig, importance_df: pd.DataFrame, feature_col: str,
                              value_col: str, norm_col: str, title_prefix: str):
    """Two stacked subplots: normalized importance across models, and signed
    Logistic Regression coefficients. Shared by per-basin and overall plots."""
    models = list(MODELS.keys())
    axes = fig.subplots(2, 1)

    ax = axes[0]
    x = np.arange(len(FEATURE_COLS))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        sub = (
            importance_df[importance_df["model"] == model]
            .set_index(feature_col)
            .reindex(FEATURE_COLS)
        )
        ax.bar(x + i * width, sub[norm_col].values, width=width,
               label=model, color=MODEL_COLORS.get(model))
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Normalized Importance (0-1, within model)")
    ax.set_title(f"{title_prefix} - Feature Importance Across Models")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    lr = (
        importance_df[importance_df["model"] == "Logistic Regression"]
        .set_index(feature_col)
        .reindex(FEATURE_COLS)[value_col]
    )
    lr_sorted = lr.reindex(lr.abs().sort_values(ascending=False).index)
    colors = ["steelblue" if v >= 0 else "tomato" for v in lr_sorted.values]
    ax2.bar(np.arange(len(lr_sorted)), lr_sorted.values, color=colors)
    ax2.set_xticks(np.arange(len(lr_sorted)))
    ax2.set_xticklabels(lr_sorted.index, rotation=45, ha="right", fontsize=8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Standardized Coefficient")
    ax2.set_title(
        f"{title_prefix} - Logistic Regression Coefficients "
        "(blue = raises whiplash odds, red = lowers)"
    )
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Plots - holdout
# ---------------------------------------------------------------------------

def plot_model_comparison(basin: str, metrics_df: pd.DataFrame, dirs: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bar(ax, metrics_df, "Score (held-out test period)")
    ax.set_title(f"{basin} - Whiplash Prediction: Model Comparison (Holdout)")
    plt.tight_layout()

    sname = safe_name(basin)
    out_path = dirs["holdout"] / f"{sname}_model_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved holdout model comparison PNG: {out_path.name}")


def plot_prediction_timeline(basin, test_dates, y_test, y_proba, model_name, dirs: dict) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test_dates, y_proba, color=MODEL_COLORS.get(model_name, "darkorange"),
            linewidth=1.3, label=f"Predicted whiplash probability ({model_name})")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8,
               label="Decision threshold (0.5)")

    actual_dates = test_dates[y_test.values.astype(bool)]
    ax.scatter(actual_dates, [1.02] * len(actual_dates), marker="*",
               color="#d4af37", edgecolor="black", s=90, zorder=5,
               label="Actual whiplash event")

    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("Predicted Probability")
    ax.set_xlabel("Date")
    ax.set_title(f"{basin} - Whiplash Prediction Timeline, Held-Out Test Period ({model_name})")
    # Legend placed below the axes (never at the top) so it can never cover a
    # star marker - stars are always plotted at y=1.02, and a whiplash event
    # can fall anywhere in the test window, including the upper-left corner.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3,
              fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    sname = safe_name(basin)
    model_slug = model_name.lower().replace(" ", "_")
    out_path = dirs["holdout"] / f"{sname}_prediction_timeline_{model_slug}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved holdout prediction timeline PNG: {out_path.name}")


def plot_overall_comparison(all_metrics: pd.DataFrame, dirs: dict) -> None:
    summary = (
        all_metrics.groupby("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bar(ax, summary, "Mean Score Across Basins (held-out test period)")
    ax.set_title("Whiplash Prediction - Model Comparison Averaged Across All Basins (Holdout)")
    plt.tight_layout()

    out_path = dirs["holdout"] / "all_basins_model_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved overall holdout comparison PNG: {out_path.name}")


# ---------------------------------------------------------------------------
# Plots - leave-one-year-out CV
# ---------------------------------------------------------------------------

def plot_loocv_model_comparison(basin: str, metrics_df: pd.DataFrame, dirs: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bar(ax, metrics_df, "Score (pooled out-of-fold predictions)")
    ax.set_title(f"{basin} - Whiplash Prediction: Model Comparison (Leave-One-Year-Out CV)")
    plt.tight_layout()

    sname = safe_name(basin)
    out_path = dirs["loocv"] / f"{sname}_loocv_model_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved LOOCV model comparison PNG: {out_path.name}")


def plot_loocv_timeline(basin, dates, y_full, oof_proba, model_name, dirs: dict) -> None:
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(dates, oof_proba.values, color=MODEL_COLORS.get(model_name, "darkorange"),
            linewidth=1.1, label=f"Out-of-fold predicted probability ({model_name})")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8,
               label="Decision threshold (0.5)")

    actual_dates = dates[y_full.values.astype(bool)]
    ax.scatter(actual_dates, [1.02] * len(actual_dates), marker="*",
               color="#d4af37", edgecolor="black", s=70, zorder=5,
               label="Actual whiplash event")

    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("Predicted Probability")
    ax.set_xlabel("Date")
    ax.set_title(f"{basin} - Leave-One-Year-Out Out-of-Fold Predictions ({model_name})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3,
              fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    sname = safe_name(basin)
    model_slug = model_name.lower().replace(" ", "_")
    out_path = dirs["loocv"] / f"{sname}_loocv_timeline_{model_slug}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved LOOCV timeline PNG: {out_path.name}")


def plot_overall_loocv_comparison(all_loocv_metrics: pd.DataFrame, dirs: dict) -> None:
    summary = (
        all_loocv_metrics.groupby("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bar(ax, summary, "Mean Score Across Basins (pooled out-of-fold predictions)")
    ax.set_title("Whiplash Prediction - Model Comparison Averaged Across All Basins (LOOCV)")
    plt.tight_layout()

    out_path = dirs["loocv"] / "all_basins_loocv_model_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overall LOOCV comparison PNG: {out_path.name}")


# ---------------------------------------------------------------------------
# Plots - feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(basin: str, importance_df: pd.DataFrame, dirs: dict) -> None:
    fig = plt.figure(figsize=(12, 10))
    _feature_importance_axes(fig, importance_df, "feature", "value", "abs_normalized", basin)
    plt.tight_layout()

    sname = safe_name(basin)
    out_path = dirs["feature_importance"] / f"{sname}_feature_importance.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved feature importance PNG: {out_path.name}")


def plot_overall_feature_importance(all_importance: pd.DataFrame, dirs: dict) -> None:
    avg = (
        all_importance.groupby(["model", "feature"])[["value", "abs_normalized"]]
        .mean()
        .reset_index()
    )

    fig = plt.figure(figsize=(12, 10))
    _feature_importance_axes(fig, avg, "feature", "value", "abs_normalized",
                              "All Basins (Averaged)")
    plt.tight_layout()

    out_path = dirs["feature_importance"] / "all_basins_feature_importance.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overall feature importance PNG: {out_path.name}")


# ---------------------------------------------------------------------------
# Plots - overfitting check
# ---------------------------------------------------------------------------

def _split_bar(ax, sub_df: pd.DataFrame, metric_cols: list, labels: list,
               base_color: str, title: str) -> None:
    """Train (faded) vs test (solid) bars, same base color, for one model."""
    train_row = sub_df[sub_df["split"] == "train"].iloc[0]
    test_row  = sub_df[sub_df["split"] == "test"].iloc[0]

    x = np.arange(len(metric_cols))
    width = 0.35
    ax.bar(x - width / 2, [train_row[m] for m in metric_cols], width=width,
           color=base_color, alpha=0.35, label="Train")
    ax.bar(x + width / 2, [test_row[m] for m in metric_cols], width=width,
           color=base_color, alpha=1.0, label="Test")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_overfit_check(basin: str, metrics_df: pd.DataFrame, dirs: dict) -> None:
    models = list(MODELS.keys())
    fig, axes = plt.subplots(len(models), 2, figsize=(13, 4 * len(models)))

    bounded_cols   = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    bounded_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    error_cols     = ["rmse", "log_loss"]
    error_labels   = ["RMSE", "Log Loss"]

    for row, model in enumerate(models):
        sub = metrics_df[metrics_df["model"] == model]

        _split_bar(axes[row, 0], sub, bounded_cols, bounded_labels,
                   MODEL_COLORS.get(model), f"{model} - Score Metrics (higher = better)")
        axes[row, 0].set_ylim(0, 1.05)

        _split_bar(axes[row, 1], sub, error_cols, error_labels,
                   MODEL_COLORS.get(model), f"{model} - Error Metrics (lower = better)")

    fig.suptitle(f"{basin} - Train vs. Test: Overfitting Check", fontsize=14, y=1.0)
    plt.tight_layout()

    sname = safe_name(basin)
    out_path = dirs["overfitting"] / f"{sname}_overfit_check.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overfit check PNG: {out_path.name}")


def plot_overall_overfit_gap(all_overfit: pd.DataFrame, dirs: dict) -> None:
    """A model that scores much better on train than on test is overfit.
    Gap is defined so that a positive bar always means 'more overfit' -
    train-minus-test for score metrics, test-minus-train for error metrics."""
    increasing_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    error_metrics = ["rmse", "log_loss"]

    train_df = all_overfit[all_overfit["split"] == "train"].set_index(["basin", "model"])
    test_df  = all_overfit[all_overfit["split"] == "test"].set_index(["basin", "model"])

    gaps = []
    for model in MODELS:
        row = {"model": model}
        for m in increasing_metrics:
            row[m] = (train_df.xs(model, level="model")[m]
                       - test_df.xs(model, level="model")[m]).mean()
        for m in error_metrics:
            row[m] = (test_df.xs(model, level="model")[m]
                       - train_df.xs(model, level="model")[m]).mean()
        gaps.append(row)
    gap_df = pd.DataFrame(gaps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.8 / len(gap_df)

    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(increasing_metrics))
    for i, (_, row) in enumerate(gap_df.iterrows()):
        vals = [row[m] for m in increasing_metrics]
        axes[0].bar(x + i * width, vals, width=width, label=row["model"],
                    color=MODEL_COLORS.get(row["model"]))
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x + width * (len(gap_df) - 1) / 2)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("Train - Test (mean across basins)")
    axes[0].set_title("Score Metrics: Higher Bar = More Overfit")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    labels2 = ["RMSE", "Log Loss"]
    x2 = np.arange(len(error_metrics))
    for i, (_, row) in enumerate(gap_df.iterrows()):
        vals = [row[m] for m in error_metrics]
        axes[1].bar(x2 + i * width, vals, width=width, label=row["model"],
                    color=MODEL_COLORS.get(row["model"]))
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x2 + width * (len(gap_df) - 1) / 2)
    axes[1].set_xticklabels(labels2, fontsize=9)
    axes[1].set_ylabel("Test - Train Error (mean across basins)")
    axes[1].set_title("Error Metrics: Higher Bar = More Overfit")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("Overfitting Gap (Train vs. Test), Averaged Across All Basins", fontsize=13)
    plt.tight_layout()

    out_path = dirs["overfitting"] / "all_basins_overfit_gap.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overall overfitting gap PNG: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_block_config(block_label: str, rule: str | None, block_dirname: str) -> None:
    dirs = make_dirs(ML_DIR / block_dirname)
    print(f"=== {block_label} block resolution - GPM IMERG precipitation - "
          f"output folder: {dirs['root']} ===\n")

    all_holdout, all_loocv, all_fi, all_overfit = [], [], [], []

    for basin in get_basins():
        print(f"Processing: {basin}")

        try:
            series, anomaly = get_block_series(basin, rule)
            features = build_features(series, anomaly)
        except Exception as exc:
            print(f"  ERROR loading/building features: {exc}")
            continue

        try:
            all_holdout.append(run_basin_holdout(basin, features, dirs))
        except Exception as exc:
            print(f"  ERROR holdout: {exc}")

        try:
            all_loocv.append(run_basin_loocv(basin, features, dirs))
        except Exception as exc:
            print(f"  ERROR LOOCV: {exc}")

        try:
            all_fi.append(run_basin_feature_importance(basin, features, dirs))
        except Exception as exc:
            print(f"  ERROR feature importance: {exc}")

        try:
            all_overfit.append(run_basin_overfit_check(basin, features, dirs))
        except Exception as exc:
            print(f"  ERROR overfit check: {exc}")

    if all_holdout:
        holdout_df = pd.concat(all_holdout, ignore_index=True)
        path = dirs["holdout"] / "model_metrics_summary.csv"
        holdout_df.to_csv(path, index=False)
        print(f"\nSaved holdout metrics summary CSV: {path.name}")
        plot_overall_comparison(holdout_df, dirs)

    if all_loocv:
        loocv_df = pd.concat(all_loocv, ignore_index=True)
        path = dirs["loocv"] / "loocv_metrics_summary.csv"
        loocv_df.to_csv(path, index=False)
        print(f"Saved LOOCV metrics summary CSV: {path.name}")
        plot_overall_loocv_comparison(loocv_df, dirs)

    if all_fi:
        fi_df = pd.concat(all_fi, ignore_index=True)
        path = dirs["feature_importance"] / "feature_importance_summary.csv"
        fi_df.to_csv(path, index=False)
        print(f"Saved feature importance summary CSV: {path.name}")
        plot_overall_feature_importance(fi_df, dirs)

    if all_overfit:
        overfit_df = pd.concat(all_overfit, ignore_index=True)
        path = dirs["overfitting"] / "overfit_metrics_summary.csv"
        overfit_df.to_csv(path, index=False)
        print(f"Saved overfit metrics summary CSV: {path.name}")
        plot_overall_overfit_gap(overfit_df, dirs)

    print(f"\n{block_label} / GPM IMERG precipitation done.\n")


def main():
    _load_precip_frame()
    for block in BLOCK_CONFIGS:
        run_block_config(block["label"], block["rule"], block["dirname"])

    print("All block resolutions done.")


if __name__ == "__main__":
    main()
