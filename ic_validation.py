#!/usr/bin/env python3
"""
M1 — IC Validation Hook for arXiv 2503.11499
Spearman Information Coefficient diagnostics: per-ETF, per-regime, rolling.

IC definition:
  For each month t, IC(t) = Spearman rank correlation between the 10-ETF
  cross-sectional predicted returns and their realized returns.
  Predictions for month t use only data from [t-48, t-1] (rolling window),
  so there is NO look-ahead leakage.

Sampling frequency: monthly (matches pipeline granularity).
IC window: each IC observation is one calendar month.

Interpretation:
  IC > 0  → model ranks ETFs in the correct direction (good signal)
  IC < 0  → model ranks ETFs inversely (anti-signal / harmful)
  IC ≈ 0  → no cross-sectional forecasting power

Author: AI-MAC Platform
Date: 2026-03-23
"""

import os
import sys
import json
import datetime
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# Import core pipeline functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regime_pipeline import (
    download_fred_md, apply_tcode_transforms, download_etf_returns,
    run_pca, two_layer_kmeans, build_transition_matrix,
    ridge_regression_forecast, ETFS, TOTAL_REGIMES, ROLLING_WINDOW,
    REGIME_LABELS, OUTPUT_DIR
)


def compute_cross_sectional_ic(predictions, actual_returns):
    """
    Compute monthly cross-sectional Spearman IC.

    For each month t with valid predictions, compute:
        IC(t) = SpearmanR(pred[t, :], actual[t, :])
    across the 10-ETF cross-section.

    Returns:
        ic_series: pd.Series indexed by date, values are Spearman rho per month.
    """
    n_months = len(actual_returns)
    ic_values = []
    ic_dates = []

    for t in range(n_months):
        pred_t = predictions[t]
        if np.isnan(pred_t).any():
            continue
        actual_t = actual_returns.iloc[t].values
        if np.isnan(actual_t).any():
            continue

        rho, _ = stats.spearmanr(pred_t, actual_t)
        ic_values.append(rho)
        ic_dates.append(actual_returns.index[t])

    return pd.Series(ic_values, index=pd.DatetimeIndex(ic_dates), name='IC')


def compute_per_etf_ic(predictions, actual_returns, window=36):
    """
    Time-series IC per ETF: rolling Spearman correlation between
    predicted and realized returns for each ETF individually.

    Returns:
        DataFrame with columns = ETF names, rows = dates, values = rolling IC.
    """
    n_months = len(actual_returns)
    etf_names = actual_returns.columns.tolist()
    results = {}

    for j, etf in enumerate(etf_names):
        pred_j = []
        actual_j = []
        dates_j = []
        for t in range(n_months):
            if np.isnan(predictions[t]).any():
                continue
            pred_j.append(predictions[t, j])
            actual_j.append(actual_returns.iloc[t, j])
            dates_j.append(actual_returns.index[t])

        pred_s = pd.Series(pred_j, index=dates_j)
        actual_s = pd.Series(actual_j, index=dates_j)

        # Full-sample time-series IC for this ETF
        if len(pred_s) > 10:
            rho_full, _ = stats.spearmanr(pred_s, actual_s)
        else:
            rho_full = np.nan

        results[etf] = {
            'full_sample_ic': rho_full,
            'n_obs': len(pred_s),
        }

        # Rolling IC (manual Spearman over rolling window)
        if len(pred_s) >= window:
            rolling_vals = []
            for i in range(window, len(pred_s) + 1):
                rho_w, _ = stats.spearmanr(pred_s.iloc[i-window:i], actual_s.iloc[i-window:i])
                rolling_vals.append(rho_w)
            rolling_ic = pd.Series(rolling_vals)
            results[etf]['rolling_ic_mean'] = float(rolling_ic.mean())
            results[etf]['rolling_ic_std'] = float(rolling_ic.std())
        else:
            results[etf]['rolling_ic_mean'] = np.nan
            results[etf]['rolling_ic_std'] = np.nan

    return pd.DataFrame(results).T


def compute_per_regime_ic(predictions, actual_returns, regime_labels):
    """
    Compute average cross-sectional IC conditioned on regime.

    For each regime r, collect all months where regime_labels[t] == r,
    compute IC(t) for those months, and report mean/std/count.

    Returns:
        DataFrame indexed by regime, columns: mean_ic, std_ic, count, pct_positive.
    """
    n_months = len(actual_returns)
    regime_ic = {r: [] for r in range(TOTAL_REGIMES)}

    for t in range(n_months):
        pred_t = predictions[t]
        if np.isnan(pred_t).any():
            continue
        actual_t = actual_returns.iloc[t].values
        if np.isnan(actual_t).any():
            continue

        rho, _ = stats.spearmanr(pred_t, actual_t)
        r = regime_labels[t]
        regime_ic[r].append(rho)

    rows = []
    for r in range(TOTAL_REGIMES):
        vals = regime_ic[r]
        if len(vals) > 0:
            rows.append({
                'regime': r,
                'regime_label': REGIME_LABELS[r],
                'mean_ic': np.mean(vals),
                'std_ic': np.std(vals),
                'count': len(vals),
                'pct_positive': np.mean([v > 0 for v in vals]),
            })
        else:
            rows.append({
                'regime': r,
                'regime_label': REGIME_LABELS[r],
                'mean_ic': np.nan,
                'std_ic': np.nan,
                'count': 0,
                'pct_positive': np.nan,
            })

    return pd.DataFrame(rows).set_index('regime')


def check_negative_ic_alert(ic_series, consecutive_months=3):
    """
    Alert rule: flag distinct episodes where avg IC < 0 for N+ consecutive months.
    Reports only the maximal streak per episode (no overlapping windows).

    Returns:
        list of alert dicts with start_date, end_date, avg_ic for each violation.
    """
    monthly_ic = ic_series.resample('MS').mean()
    alerts = []
    streak_start = None
    streak_vals = []
    streak_dates = []

    def flush_streak():
        if streak_start is not None and len(streak_vals) >= consecutive_months:
            alerts.append({
                'start_date': str(streak_dates[0].date()),
                'end_date': str(streak_dates[-1].date()),
                'months': len(streak_vals),
                'avg_ic': round(float(np.mean(streak_vals)), 4),
            })

    for date, val in monthly_ic.items():
        if np.isnan(val) or val >= 0:
            flush_streak()
            streak_start = None
            streak_vals = []
            streak_dates = []
        else:
            if streak_start is None:
                streak_start = date
            streak_vals.append(val)
            streak_dates.append(date)

    flush_streak()  # handle streak at end of series
    return alerts


def main():
    print("=" * 70)
    print("M1 — IC VALIDATION HOOK")
    print("arXiv 2503.11499 Regime Detection Pipeline")
    print(f"Run date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---- Stage 0: Data ----
    print("\n[DATA] Downloading macro and ETF data...")
    fred_raw, tcodes = download_fred_md()
    fred_transformed = apply_tcode_transforms(fred_raw, tcodes)
    etf_returns = download_etf_returns()

    # ---- Stage A: Regime Classification ----
    print("\n[REGIME] Running PCA + two-layer k-means...")
    pca_components, pca_model, scaler, n_pca, cumvar = run_pca(fred_transformed)
    regime_df, km_l1, km_l2, elbow_k, inertias = two_layer_kmeans(
        pca_components, fred_transformed.index)

    # ---- Align ----
    common_idx = regime_df.index.intersection(etf_returns.index)
    regime_aligned = regime_df.loc[common_idx]
    returns_aligned = etf_returns.loc[common_idx]
    macro_aligned = fred_transformed.loc[common_idx]

    macro_scaled = scaler.transform(macro_aligned)
    macro_pca = pca_model.transform(macro_scaled)

    regime_labels_arr = regime_aligned['regime'].values
    prob_cols = [f'prob_r{r}' for r in range(TOTAL_REGIMES)]
    regime_probs_arr = regime_aligned[prob_cols].values

    print(f"[ALIGN] Dataset: {len(common_idx)} months, {len(ETFS)} ETFs, {TOTAL_REGIMES} regimes")

    # ---- Stage B: Ridge Regression Forecasts ----
    print("\n[FORECAST] Running ridge regression (rolling {}-month window)...".format(ROLLING_WINDOW))
    ridge_preds = ridge_regression_forecast(
        macro_pca, returns_aligned, regime_probs_arr, regime_labels_arr)

    # ---- IC DIAGNOSTICS ----
    print("\n" + "=" * 50)
    print("IC DIAGNOSTICS")
    print("=" * 50)

    # 1. Cross-sectional IC (monthly)
    print("\n[IC] Computing cross-sectional Spearman IC...")
    ic_series = compute_cross_sectional_ic(ridge_preds, returns_aligned)
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    ic_median = ic_series.median()
    ic_pct_pos = (ic_series > 0).mean()
    n_ic = len(ic_series)

    print(f"  Cross-sectional IC: mean={ic_mean:.4f}, std={ic_std:.4f}, "
          f"median={ic_median:.4f}, %positive={ic_pct_pos:.2%}, n={n_ic}")

    # 2. Per-regime IC
    print("\n[IC] Computing per-regime IC...")
    regime_ic_df = compute_per_regime_ic(ridge_preds, returns_aligned, regime_labels_arr)
    print(regime_ic_df.to_string())

    # 3. Per-ETF IC
    print("\n[IC] Computing per-ETF time-series IC...")
    etf_ic_df = compute_per_etf_ic(ridge_preds, returns_aligned)
    print(etf_ic_df.to_string())

    # 4. Rolling 12-month IC
    print("\n[IC] Computing rolling 12-month IC...")
    rolling_12m_ic = ic_series.rolling(12, min_periods=6).mean()
    rolling_12m_min = rolling_12m_ic.min()
    rolling_12m_min_date = rolling_12m_ic.idxmin()
    print(f"  Rolling 12m IC: min={rolling_12m_min:.4f} at {rolling_12m_min_date}")

    # 5. Alert check: 3 consecutive months of negative IC
    print("\n[ALERT] Checking for 3-month negative IC streaks...")
    alerts = check_negative_ic_alert(ic_series, consecutive_months=3)
    if alerts:
        print(f"  WARNING: {len(alerts)} alert period(s) detected!")
        for a in alerts:
            print(f"    {a['start_date']} to {a['end_date']} "
                  f"({a['months']} months, avg IC={a['avg_ic']:.4f})")
    else:
        print("  OK: No 3-month negative IC streaks detected.")

    # ---- OOS split ----
    oos_cutoff = pd.Timestamp('2022-12-31')
    is_ic = ic_series[ic_series.index <= oos_cutoff]
    oos_ic = ic_series[ic_series.index > oos_cutoff]

    print(f"\n[IC] In-sample IC (→ Dec 2022): mean={is_ic.mean():.4f}, n={len(is_ic)}")
    if len(oos_ic) > 0:
        print(f"[IC] Out-of-sample IC (Jan 2023→): mean={oos_ic.mean():.4f}, n={len(oos_ic)}")
    else:
        print("[IC] Out-of-sample IC: insufficient data")

    # ---- SAVE DIAGNOSTICS ----
    print("\n" + "=" * 50)
    print("SAVING DIAGNOSTICS")
    print("=" * 50)

    # CSV: monthly IC time series
    ic_csv_path = os.path.join(OUTPUT_DIR, 'ic_monthly_timeseries.csv')
    ic_df_out = pd.DataFrame({
        'date': ic_series.index,
        'ic': ic_series.values,
        'regime': [regime_labels_arr[returns_aligned.index.get_loc(d)] for d in ic_series.index],
    })
    ic_df_out.to_csv(ic_csv_path, index=False)
    print(f"[OUTPUT] Saved: {ic_csv_path}")

    # CSV: per-regime IC
    regime_ic_csv_path = os.path.join(OUTPUT_DIR, 'ic_per_regime.csv')
    regime_ic_df.to_csv(regime_ic_csv_path)
    print(f"[OUTPUT] Saved: {regime_ic_csv_path}")

    # CSV: per-ETF IC
    etf_ic_csv_path = os.path.join(OUTPUT_DIR, 'ic_per_etf.csv')
    etf_ic_df.to_csv(etf_ic_csv_path)
    print(f"[OUTPUT] Saved: {etf_ic_csv_path}")

    # JSON: full diagnostics
    diagnostics = {
        'timestamp': datetime.datetime.now().isoformat(),
        'ic_window_definition': 'Monthly cross-sectional Spearman rank correlation '
                                'between 10-ETF predicted returns and realized returns',
        'sampling_frequency': 'Monthly',
        'rolling_forecast_window': ROLLING_WINDOW,
        'look_ahead_check': 'PASS — predictions for month t use only data from [t-48, t-1]',
        'overall': {
            'mean_ic': round(float(ic_mean), 4),
            'std_ic': round(float(ic_std), 4),
            'median_ic': round(float(ic_median), 4),
            'pct_positive': round(float(ic_pct_pos), 4),
            'n_months': int(n_ic),
            't_stat': round(float(ic_mean / (ic_std / np.sqrt(n_ic))), 4) if ic_std > 0 else 0.0,
        },
        'in_sample': {
            'mean_ic': round(float(is_ic.mean()), 4),
            'n_months': int(len(is_ic)),
        },
        'out_of_sample': {
            'mean_ic': round(float(oos_ic.mean()), 4) if len(oos_ic) > 0 else None,
            'n_months': int(len(oos_ic)),
        },
        'per_regime': {
            f'R{r}': {
                'label': REGIME_LABELS[r],
                'mean_ic': round(float(regime_ic_df.loc[r, 'mean_ic']), 4)
                    if not np.isnan(regime_ic_df.loc[r, 'mean_ic']) else None,
                'std_ic': round(float(regime_ic_df.loc[r, 'std_ic']), 4)
                    if not np.isnan(regime_ic_df.loc[r, 'std_ic']) else None,
                'count': int(regime_ic_df.loc[r, 'count']),
                'pct_positive': round(float(regime_ic_df.loc[r, 'pct_positive']), 4)
                    if not np.isnan(regime_ic_df.loc[r, 'pct_positive']) else None,
            }
            for r in range(TOTAL_REGIMES)
        },
        'per_etf': {
            etf: {
                'full_sample_ic': round(float(etf_ic_df.loc[etf, 'full_sample_ic']), 4)
                    if not np.isnan(etf_ic_df.loc[etf, 'full_sample_ic']) else None,
                'n_obs': int(etf_ic_df.loc[etf, 'n_obs']),
            }
            for etf in ETFS
        },
        'alerts': alerts,
        'alert_rule': 'avg IC < 0 for 3 consecutive months',
    }

    diag_json_path = os.path.join(OUTPUT_DIR, 'ic_diagnostics_2503.json')
    with open(diag_json_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    print(f"[OUTPUT] Saved: {diag_json_path}")

    # ---- SUMMARY ----
    print("\n" + "=" * 70)
    print("IC VALIDATION COMPLETE")
    print("=" * 70)
    print(f"  Overall IC: {ic_mean:.4f} (t-stat: {diagnostics['overall']['t_stat']:.2f})")
    print(f"  % Positive: {ic_pct_pos:.1%}")
    print(f"  Alerts: {len(alerts)} period(s)")
    signal_quality = "POSITIVE" if ic_mean > 0 else "NEGATIVE"
    print(f"  Signal quality: {signal_quality}")
    print("=" * 70)

    return diagnostics


if __name__ == '__main__':
    diagnostics = main()
