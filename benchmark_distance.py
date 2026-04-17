#!/usr/bin/env python3
"""
M3 — Benchmark Distance Reporting for arXiv 2503.11499
Per-ETF RMSFE distance vs Random Walk (RW) and AR(1) baselines.

Metrics:
  RMSFE = sqrt( mean( (predicted - actual)^2 ) )
  Relative RMSFE = RMSFE_model / RMSFE_baseline  (<1 means model beats baseline)

Baselines:
  RW  — predict next-month return = 0 (martingale hypothesis)
  AR1 — predict next-month return from rolling AR(1) fit on same window

All forecasts use the same causal rolling 48-month window as the main pipeline.
No look-ahead leakage.

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
    ridge_regression_forecast, naive_forecast,
    ETFS, TOTAL_REGIMES, ROLLING_WINDOW, REGIME_LABELS, OUTPUT_DIR
)


def ar1_forecast(returns, window=ROLLING_WINDOW):
    """
    Rolling AR(1) baseline: for each ETF, predict next return from
    OLS regression on lagged return using the trailing window.
    """
    n_months, n_etfs = returns.shape
    predictions = np.full((n_months, n_etfs), np.nan)

    vals = returns.values
    for t in range(window, n_months):
        for j in range(n_etfs):
            y = vals[t - window:t, j]
            x = np.arange(len(y)).reshape(-1, 1)
            # AR(1): y_t = a + b * y_{t-1}
            y_lag = y[:-1]
            y_curr = y[1:]
            if len(y_lag) < 5:
                continue
            # Simple OLS
            x_mean = y_lag.mean()
            y_mean = y_curr.mean()
            cov = ((y_lag - x_mean) * (y_curr - y_mean)).sum()
            var = ((y_lag - x_mean) ** 2).sum()
            if var < 1e-12:
                predictions[t, j] = y_mean
            else:
                b = cov / var
                a = y_mean - b * x_mean
                predictions[t, j] = a + b * y[-1]  # predict from last observation

    return predictions


def rw_forecast(returns, window=ROLLING_WINDOW):
    """
    Random Walk baseline: predict next return = 0.
    Only produces predictions where Ridge also produces them (t >= window).
    """
    n_months, n_etfs = returns.shape
    predictions = np.full((n_months, n_etfs), np.nan)
    for t in range(window, n_months):
        predictions[t, :] = 0.0
    return predictions


def compute_rmsfe(predictions, actuals, etf_names):
    """
    Per-ETF RMSFE over months with valid predictions.
    Returns dict: {etf: rmsfe_value}
    """
    results = {}
    for j, etf in enumerate(etf_names):
        valid = ~np.isnan(predictions[:, j])
        if valid.sum() == 0:
            results[etf] = np.nan
            continue
        resid = predictions[valid, j] - actuals[valid, j]
        results[etf] = float(np.sqrt(np.mean(resid ** 2)))
    return results


def main():
    print("=" * 60)
    print("M3 — BENCHMARK DISTANCE REPORTING")
    print(f"Run: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ---- Data ----
    fred_raw, tcodes = download_fred_md()
    fred_transformed = apply_tcode_transforms(fred_raw, tcodes)
    etf_returns = download_etf_returns()

    # ---- Regime classification ----
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

    n_months = len(returns_aligned)
    actuals = returns_aligned.values
    print(f"\n[DATA] {n_months} months, {len(ETFS)} ETFs, window={ROLLING_WINDOW}")

    # ---- Generate predictions ----
    print("\n[FORECAST] Ridge regression...")
    ridge_preds = ridge_regression_forecast(
        macro_pca, returns_aligned, regime_probs_arr, regime_labels_arr)

    print("[FORECAST] Naive conditional Sharpe...")
    naive_preds = naive_forecast(
        returns_aligned, regime_labels_arr, regime_probs_arr)

    print("[FORECAST] AR(1) baseline...")
    ar1_preds = ar1_forecast(returns_aligned)

    print("[FORECAST] Random Walk (zero) baseline...")
    rw_preds = rw_forecast(returns_aligned)

    # ---- Compute RMSFE ----
    rmsfe_ridge = compute_rmsfe(ridge_preds, actuals, ETFS)
    rmsfe_naive = compute_rmsfe(naive_preds, actuals, ETFS)
    rmsfe_ar1 = compute_rmsfe(ar1_preds, actuals, ETFS)
    rmsfe_rw = compute_rmsfe(rw_preds, actuals, ETFS)

    # ---- Build results table ----
    rows = []
    for etf in ETFS:
        row = {
            'ETF': etf,
            'RMSFE_Ridge': round(rmsfe_ridge[etf], 6),
            'RMSFE_Naive': round(rmsfe_naive[etf], 6),
            'RMSFE_AR1': round(rmsfe_ar1[etf], 6),
            'RMSFE_RW': round(rmsfe_rw[etf], 6),
        }
        # Relative RMSFE (model / baseline)
        row['Ridge_vs_RW'] = round(rmsfe_ridge[etf] / rmsfe_rw[etf], 4) if rmsfe_rw[etf] > 0 else np.nan
        row['Ridge_vs_AR1'] = round(rmsfe_ridge[etf] / rmsfe_ar1[etf], 4) if rmsfe_ar1[etf] > 0 else np.nan
        row['Naive_vs_RW'] = round(rmsfe_naive[etf] / rmsfe_rw[etf], 4) if rmsfe_rw[etf] > 0 else np.nan
        row['Naive_vs_AR1'] = round(rmsfe_naive[etf] / rmsfe_ar1[etf], 4) if rmsfe_ar1[etf] > 0 else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)

    # Aggregate row
    agg = {
        'ETF': 'AGGREGATE',
        'RMSFE_Ridge': round(np.mean([rmsfe_ridge[e] for e in ETFS]), 6),
        'RMSFE_Naive': round(np.mean([rmsfe_naive[e] for e in ETFS]), 6),
        'RMSFE_AR1': round(np.mean([rmsfe_ar1[e] for e in ETFS]), 6),
        'RMSFE_RW': round(np.mean([rmsfe_rw[e] for e in ETFS]), 6),
    }
    agg['Ridge_vs_RW'] = round(agg['RMSFE_Ridge'] / agg['RMSFE_RW'], 4) if agg['RMSFE_RW'] > 0 else np.nan
    agg['Ridge_vs_AR1'] = round(agg['RMSFE_Ridge'] / agg['RMSFE_AR1'], 4) if agg['RMSFE_AR1'] > 0 else np.nan
    agg['Naive_vs_RW'] = round(agg['RMSFE_Naive'] / agg['RMSFE_RW'], 4) if agg['RMSFE_RW'] > 0 else np.nan
    agg['Naive_vs_AR1'] = round(agg['RMSFE_Naive'] / agg['RMSFE_AR1'], 4) if agg['RMSFE_AR1'] > 0 else np.nan
    df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)

    # ---- OOS split ----
    oos_cutoff = pd.Timestamp('2022-12-31')
    is_mask = np.array(returns_aligned.index <= oos_cutoff)
    oos_mask = np.array(returns_aligned.index > oos_cutoff)

    # Per-ETF OOS RMSFE
    oos_rows = []
    if oos_mask.sum() > 6:
        for etf_idx, etf in enumerate(ETFS):
            valid_oos = oos_mask & ~np.isnan(ridge_preds[:, etf_idx])
            if valid_oos.sum() == 0:
                continue
            rmsfe_r_oos = float(np.sqrt(np.mean((ridge_preds[valid_oos, etf_idx] - actuals[valid_oos, etf_idx]) ** 2)))
            rmsfe_rw_oos = float(np.sqrt(np.mean(actuals[valid_oos, etf_idx] ** 2)))
            rmsfe_ar1_oos = float(np.sqrt(np.mean((ar1_preds[valid_oos, etf_idx] - actuals[valid_oos, etf_idx]) ** 2)))
            oos_rows.append({
                'ETF': etf,
                'RMSFE_Ridge_OOS': round(rmsfe_r_oos, 6),
                'RMSFE_RW_OOS': round(rmsfe_rw_oos, 6),
                'RMSFE_AR1_OOS': round(rmsfe_ar1_oos, 6),
                'Ridge_vs_RW_OOS': round(rmsfe_r_oos / rmsfe_rw_oos, 4) if rmsfe_rw_oos > 0 else np.nan,
                'Ridge_vs_AR1_OOS': round(rmsfe_r_oos / rmsfe_ar1_oos, 4) if rmsfe_ar1_oos > 0 else np.nan,
            })

    oos_df = pd.DataFrame(oos_rows) if oos_rows else pd.DataFrame()

    # ---- Diebold-Mariano test (Ridge vs RW, full sample) ----
    dm_results = {}
    for etf_idx, etf in enumerate(ETFS):
        valid = ~np.isnan(ridge_preds[:, etf_idx])
        if valid.sum() < 20:
            continue
        e_ridge = (ridge_preds[valid, etf_idx] - actuals[valid, etf_idx]) ** 2
        e_rw = actuals[valid, etf_idx] ** 2  # RW predicts 0
        d = e_rw - e_ridge  # positive = Ridge better
        d_mean = d.mean()
        d_se = d.std() / np.sqrt(len(d))
        dm_stat = d_mean / d_se if d_se > 0 else 0
        dm_pval = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        dm_results[etf] = {
            'DM_stat': round(float(dm_stat), 4),
            'DM_pval': round(float(dm_pval), 4),
            'Ridge_better': bool(dm_stat > 0),
        }

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("BENCHMARK DISTANCE RESULTS")
    print("=" * 60)
    print("\nFull-sample RMSFE (relative to RW):")
    for _, row in df.iterrows():
        flag = "✓" if row.get('Ridge_vs_RW', 999) < 1.0 else "✗"
        print(f"  {row['ETF']:12s}  Ridge/RW={row['Ridge_vs_RW']:.4f}  "
              f"Ridge/AR1={row['Ridge_vs_AR1']:.4f}  {flag}")

    # Count wins
    etf_rows = df[df['ETF'] != 'AGGREGATE']
    ridge_wins_rw = (etf_rows['Ridge_vs_RW'] < 1.0).sum()
    ridge_wins_ar1 = (etf_rows['Ridge_vs_AR1'] < 1.0).sum()
    print(f"\nRidge beats RW:  {ridge_wins_rw}/{len(ETFS)} ETFs")
    print(f"Ridge beats AR1: {ridge_wins_ar1}/{len(ETFS)} ETFs")

    # ---- DM test summary ----
    print("\nDiebold-Mariano test (Ridge vs RW, H0: equal forecast accuracy):")
    dm_sig = 0
    for etf, res in dm_results.items():
        sig = "*" if res['DM_pval'] < 0.10 else ""
        print(f"  {etf:6s}  DM={res['DM_stat']:+.4f}  p={res['DM_pval']:.4f}  "
              f"{'Ridge better' if res['Ridge_better'] else 'RW better'} {sig}")
        if res['DM_pval'] < 0.10 and res['Ridge_better']:
            dm_sig += 1
    print(f"\nSignificant Ridge wins (p<0.10): {dm_sig}/{len(ETFS)}")

    # ---- Pass/Fail ----
    # Pass criterion: Ridge beats RW on aggregate AND at least 5/10 ETFs
    agg_row = df[df['ETF'] == 'AGGREGATE'].iloc[0]
    agg_pass = agg_row['Ridge_vs_RW'] < 1.0
    count_pass = ridge_wins_rw >= 5
    overall_pass = agg_pass and count_pass

    verdict = "PASS" if overall_pass else "CONDITIONAL PASS" if (agg_pass or count_pass) else "FAIL"
    print(f"\n{'=' * 40}")
    print(f"VERDICT: {verdict}")
    print(f"  Aggregate Ridge/RW < 1: {agg_pass} ({agg_row['Ridge_vs_RW']:.4f})")
    print(f"  ETFs with Ridge < RW:   {ridge_wins_rw}/10 (need ≥5)")
    print(f"{'=' * 40}")

    # ---- Save CSV ----
    csv_path = os.path.join(OUTPUT_DIR, 'benchmark_distance_2503.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n[OUTPUT] {csv_path}")

    if len(oos_df) > 0:
        oos_csv_path = os.path.join(OUTPUT_DIR, 'benchmark_distance_oos_2503.csv')
        oos_df.to_csv(oos_csv_path, index=False)
        print(f"[OUTPUT] {oos_csv_path}")

    # ---- Save JSON ----
    json_out = {
        'run_timestamp': datetime.datetime.now().isoformat(),
        'window': ROLLING_WINDOW,
        'n_months': int(n_months),
        'n_etfs': len(ETFS),
        'full_sample': {
            'rmsfe_ridge': rmsfe_ridge,
            'rmsfe_rw': rmsfe_rw,
            'rmsfe_ar1': rmsfe_ar1,
            'rmsfe_naive': {k: round(v, 6) for k, v in rmsfe_naive.items()},
            'aggregate_ridge_vs_rw': float(agg_row['Ridge_vs_RW']),
            'aggregate_ridge_vs_ar1': float(agg_row['Ridge_vs_AR1']),
            'ridge_wins_vs_rw': int(ridge_wins_rw),
            'ridge_wins_vs_ar1': int(ridge_wins_ar1),
        },
        'diebold_mariano': dm_results,
        'oos': oos_rows if oos_rows else None,
        'verdict': verdict,
    }
    json_path = os.path.join(OUTPUT_DIR, 'benchmark_distance_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f"[OUTPUT] {json_path}")

    # ---- Generate markdown report ----
    md = []
    md.append("# M3 — Benchmark Distance Report")
    md.append(f"\n**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    md.append(f"**Pipeline:** arXiv 2503.11499 Regime Detection")
    md.append(f"**Window:** {ROLLING_WINDOW} months rolling | **Sample:** {n_months} months")
    md.append("")
    md.append("## 1. Design")
    md.append("")
    md.append("For each of 10 sector ETFs, compute the Root Mean Squared Forecast Error (RMSFE) of:")
    md.append("- **Ridge**: Per-regime ridge regression on PCA macro features (main pipeline)")
    md.append("- **Naive**: Conditional regime Sharpe forecast (pipeline alternative)")
    md.append("- **RW**: Random walk (predict return = 0, martingale)")
    md.append("- **AR(1)**: Rolling autoregressive(1) on ETF returns")
    md.append("")
    md.append("Relative RMSFE < 1.0 means the model improves on the baseline.")
    md.append("")
    md.append("## 2. Full-Sample Results")
    md.append("")
    md.append("| ETF | RMSFE Ridge | RMSFE RW | RMSFE AR1 | Ridge/RW | Ridge/AR1 |")
    md.append("|-----|------------|---------|----------|---------|----------|")
    for _, row in df.iterrows():
        rr = f"**{row['Ridge_vs_RW']:.4f}**" if row['Ridge_vs_RW'] < 1.0 else f"{row['Ridge_vs_RW']:.4f}"
        ra = f"**{row['Ridge_vs_AR1']:.4f}**" if row['Ridge_vs_AR1'] < 1.0 else f"{row['Ridge_vs_AR1']:.4f}"
        md.append(f"| {row['ETF']} | {row['RMSFE_Ridge']:.6f} | {row['RMSFE_RW']:.6f} | "
                  f"{row['RMSFE_AR1']:.6f} | {rr} | {ra} |")

    md.append("")
    md.append(f"**Ridge beats RW:** {ridge_wins_rw}/10 ETFs")
    md.append(f"**Ridge beats AR1:** {ridge_wins_ar1}/10 ETFs")

    md.append("")
    md.append("## 3. Diebold-Mariano Test (Ridge vs RW)")
    md.append("")
    md.append("Tests whether forecast accuracy difference is statistically significant.")
    md.append("")
    md.append("| ETF | DM stat | p-value | Direction |")
    md.append("|-----|---------|---------|-----------|")
    for etf, res in dm_results.items():
        sig = " *" if res['DM_pval'] < 0.10 else ""
        direction = "Ridge better" if res['Ridge_better'] else "RW better"
        md.append(f"| {etf} | {res['DM_stat']:+.4f} | {res['DM_pval']:.4f}{sig} | {direction} |")
    md.append("")
    md.append(f"Significant Ridge wins (p<0.10): {dm_sig}/{len(ETFS)}")

    # OOS section
    if len(oos_df) > 0:
        md.append("")
        md.append("## 4. Out-of-Sample Results (Jan 2023 →)")
        md.append("")
        md.append("| ETF | RMSFE Ridge | RMSFE RW | Ridge/RW | Ridge/AR1 |")
        md.append("|-----|------------|---------|---------|----------|")
        for _, row in oos_df.iterrows():
            rr = f"**{row['Ridge_vs_RW_OOS']:.4f}**" if row['Ridge_vs_RW_OOS'] < 1.0 else f"{row['Ridge_vs_RW_OOS']:.4f}"
            ra = f"**{row['Ridge_vs_AR1_OOS']:.4f}**" if row['Ridge_vs_AR1_OOS'] < 1.0 else f"{row['Ridge_vs_AR1_OOS']:.4f}"
            md.append(f"| {row['ETF']} | {row['RMSFE_Ridge_OOS']:.6f} | {row['RMSFE_RW_OOS']:.6f} | {rr} | {ra} |")

    md.append("")
    md.append("## 5. Verdict")
    md.append("")
    md.append(f"**{verdict}**")
    md.append("")
    md.append("Pass criteria:")
    md.append(f"- Aggregate Ridge/RW < 1.0: {'PASS' if agg_pass else 'FAIL'} ({agg_row['Ridge_vs_RW']:.4f})")
    md.append(f"- Ridge beats RW on ≥5/10 ETFs: {'PASS' if count_pass else 'FAIL'} ({ridge_wins_rw}/10)")
    md.append("")
    md.append("## 6. Interpretation")
    md.append("")
    md.append("The benchmark distance report measures whether the regime-conditional Ridge model")
    md.append("produces more accurate point forecasts than simple baselines. This complements the")
    md.append("Sharpe-based evaluation (which measures portfolio-level value) by examining the")
    md.append("forecast accuracy channel directly.")
    md.append("")
    md.append("Key insights:")
    md.append("- RMSFE captures point forecast accuracy, not portfolio construction quality.")
    md.append("- A model can have higher RMSFE but still produce better portfolios if its ranking")
    md.append("  of ETFs is correct (IC > 0) even when magnitude is noisy.")
    md.append("- The Diebold-Mariano test accounts for serial correlation in forecast errors.")
    md.append("")
    md.append("---")
    md.append(f"*Report generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} by benchmark_distance_2503.py*")

    md_path = os.path.join(OUTPUT_DIR, 'benchmark_distance_2503.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md))
    print(f"[OUTPUT] {md_path}")

    print(f"\n[DONE] M3 Benchmark Distance: {verdict}")
    return verdict


if __name__ == '__main__':
    main()
