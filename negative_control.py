#!/usr/bin/env python3
"""
M2 — Cross-Sectional Negative Control (T5) for arXiv 2503.11499

Design:
  The existing temporal permutation test (regime_pipeline.py) shuffles regime
  labels randomly and feeds them to the naive_forecast model. This cross-
  sectional negative control is complementary: it shuffles regime labels but
  runs the FULL ridge regression forecast pipeline, testing whether the
  regime-conditioning in the ridge model adds value over random conditioning.

  Why this complements the temporal test:
  - Temporal test (existing): uses naive_forecast (conditional Sharpe) with
    shuffled labels → tests whether regime-conditional mean/vol matters.
  - Cross-sectional control (this): uses ridge_regression_forecast with
    shuffled labels → tests whether regime-conditioned regression on macro
    features adds value. If the ridge model's Sharpe drops under shuffled
    labels, the regime detection is genuinely informative for the regression
    step, not just an artefact of the portfolio construction rule.

  Shuffle method: np.random.permutation of regime labels across time, which
  breaks the temporal association between macro state and regime label while
  preserving the marginal distribution of regimes. Regime probabilities are
  reconstructed as one-hot from shuffled labels.

  Temporal order of returns and macro features is NEVER shuffled — only the
  regime labels and derived regime probabilities are permuted.

Random seed policy:
  Base seed = 20260323 (today's date). Each permutation i uses seed =
  base_seed + i, ensuring full reproducibility.

Pass criterion: observed strategy Sharpe >= 95th percentile of shuffled
  distribution (i.e., p_empirical <= 0.05).

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')

# Import core pipeline functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regime_pipeline import (
    download_fred_md, apply_tcode_transforms, download_etf_returns,
    run_pca, two_layer_kmeans, build_transition_matrix,
    ridge_regression_forecast, long_only_portfolio,
    compute_portfolio_returns, compute_metrics,
    ETFS, TOTAL_REGIMES, ROLLING_WINDOW, REGIME_LABELS, OUTPUT_DIR
)

# ============================================================
# CONFIGURATION
# ============================================================
N_PERMUTATIONS = 200
BASE_SEED = 20260323
STRATEGY_L = 3  # top-l for long-only portfolio
PASS_PERCENTILE = 95  # observed must exceed this percentile


def run_cross_sectional_negative_control(
    macro_pca, returns_aligned, regime_labels_arr, regime_probs_arr,
    n_perms=N_PERMUTATIONS, base_seed=BASE_SEED
):
    """
    Shuffled-regime-label cross-sectional negative control.

    For each permutation:
      1. Shuffle regime labels (breaking temporal association)
      2. Build one-hot regime probabilities from shuffled labels
      3. Run full ridge_regression_forecast
      4. Build long-only portfolio (l=3)
      5. Compute Sharpe

    Returns:
        np.array of permutation Sharpe ratios
    """
    print(f"[NEG-CTRL] Running {n_perms} cross-sectional permutations...")
    print(f"[NEG-CTRL] Base seed: {base_seed}")
    perm_sharpes = []

    for p in range(n_perms):
        rng = np.random.RandomState(base_seed + p)
        shuffled_labels = rng.permutation(regime_labels_arr)

        # Build one-hot regime probs from shuffled labels
        shuffled_probs = np.zeros((len(shuffled_labels), TOTAL_REGIMES))
        for i, r in enumerate(shuffled_labels):
            shuffled_probs[i, r] = 1.0

        # Full ridge forecast with shuffled regime conditioning
        preds = ridge_regression_forecast(
            macro_pca, returns_aligned, shuffled_probs, shuffled_labels,
            window=ROLLING_WINDOW
        )

        weights = long_only_portfolio(preds, l=STRATEGY_L)
        port_ret = compute_portfolio_returns(weights, returns_aligned)
        metrics = compute_metrics(port_ret)
        perm_sharpes.append(metrics['sharpe'])

        if (p + 1) % 25 == 0:
            print(f"  Permutation {p+1}/{n_perms} — Sharpe={metrics['sharpe']:.3f}")

    return np.array(perm_sharpes)


def main():
    print("=" * 70)
    print("M2 — CROSS-SECTIONAL NEGATIVE CONTROL (T5)")
    print("arXiv 2503.11499 Regime Detection Pipeline")
    print(f"Run date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Permutations: {N_PERMUTATIONS}")
    print(f"Base seed: {BASE_SEED}")
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

    print(f"[ALIGN] Dataset: {len(common_idx)} months, {len(ETFS)} ETFs, "
          f"{TOTAL_REGIMES} regimes")

    # ---- Observed Strategy Sharpe ----
    print("\n[OBSERVED] Computing observed strategy Sharpe (Ridge_LO_l3)...")
    real_preds = ridge_regression_forecast(
        macro_pca, returns_aligned, regime_probs_arr, regime_labels_arr,
        window=ROLLING_WINDOW
    )
    real_weights = long_only_portfolio(real_preds, l=STRATEGY_L)
    real_port_ret = compute_portfolio_returns(real_weights, returns_aligned)
    real_metrics = compute_metrics(real_port_ret)
    real_sharpe = real_metrics['sharpe']
    print(f"[OBSERVED] Ridge_LO_l3 Sharpe = {real_sharpe}")

    # ---- Cross-Sectional Negative Control ----
    perm_sharpes = run_cross_sectional_negative_control(
        macro_pca, returns_aligned, regime_labels_arr, regime_probs_arr,
        n_perms=N_PERMUTATIONS, base_seed=BASE_SEED
    )

    # ---- Analysis ----
    valid_perms = perm_sharpes[~np.isnan(perm_sharpes)]
    n_valid = len(valid_perms)
    perm_mean = np.mean(valid_perms)
    perm_std = np.std(valid_perms)
    perm_median = np.median(valid_perms)
    perm_p5 = np.percentile(valid_perms, 5)
    perm_p25 = np.percentile(valid_perms, 25)
    perm_p75 = np.percentile(valid_perms, 75)
    perm_p95 = np.percentile(valid_perms, 95)

    # Percentile rank of observed Sharpe in permutation distribution
    percentile_rank = (valid_perms < real_sharpe).sum() / n_valid * 100
    p_empirical = (valid_perms >= real_sharpe).sum() / n_valid

    # t-test: is the permutation mean significantly different from observed?
    if n_valid > 1:
        t_stat, p_ttest = stats.ttest_1samp(valid_perms, real_sharpe)
    else:
        t_stat, p_ttest = 0.0, 1.0

    # Pass/fail
    passed = percentile_rank >= PASS_PERCENTILE

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Observed Sharpe:       {real_sharpe:.4f}")
    print(f"  Permutation mean:      {perm_mean:.4f}")
    print(f"  Permutation std:       {perm_std:.4f}")
    print(f"  Permutation median:    {perm_median:.4f}")
    print(f"  Permutation 5th pctl:  {perm_p5:.4f}")
    print(f"  Permutation 95th pctl: {perm_p95:.4f}")
    print(f"  Percentile rank:       {percentile_rank:.1f}%")
    print(f"  Empirical p-value:     {p_empirical:.4f}")
    print(f"  t-statistic:           {t_stat:.4f}")
    print(f"  PASS criterion:        observed >= {PASS_PERCENTILE}th percentile")
    print(f"  RESULT:                {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    # ---- Save CSV distribution ----
    dist_df = pd.DataFrame({
        'permutation': range(N_PERMUTATIONS),
        'seed': [BASE_SEED + i for i in range(N_PERMUTATIONS)],
        'sharpe': perm_sharpes,
    })
    dist_csv_path = os.path.join(OUTPUT_DIR, 'negative_control_t5_distribution.csv')
    dist_df.to_csv(dist_csv_path, index=False)
    print(f"\n[OUTPUT] Saved: {dist_csv_path}")

    # ---- Save Plot ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(valid_perms, bins=30, alpha=0.7, color='steelblue', edgecolor='white',
            label=f'Shuffled Regime Sharpe (n={n_valid})')
    ax.axvline(real_sharpe, color='red', linewidth=2.5, linestyle='--',
               label=f'Observed Sharpe = {real_sharpe:.3f}')
    ax.axvline(perm_p95, color='orange', linewidth=1.5, linestyle=':',
               label=f'95th percentile = {perm_p95:.3f}')
    ax.set_title('Cross-Sectional Negative Control (T5)\n'
                 'Ridge_LO_l3 — Observed vs Shuffled Regime Labels',
                 fontsize=12)
    ax.set_xlabel('Sharpe Ratio')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)

    # Add text annotation
    text_str = (f'Percentile: {percentile_rank:.1f}%\n'
                f'p-value: {p_empirical:.4f}\n'
                f'{"PASS" if passed else "FAIL"}')
    ax.text(0.97, 0.97, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'negative_control_t5_plot.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[OUTPUT] Saved: {plot_path}")

    # ---- Save Report ----
    report = f"""# M2 — Cross-Sectional Negative Control (T5)

**Paper:** arXiv 2503.11499 — Tactical Asset Allocation with Macroeconomic Regime Detection
**Run date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Strategy:** Ridge_LO_l3 (per-regime ridge regression, long-only top-3)

## Design

### What this test does
For each of {N_PERMUTATIONS} permutations, the regime labels are shuffled across time
(breaking the temporal association between macro state and regime classification),
while all other inputs — macro features, ETF returns, temporal order — remain intact.
The full `ridge_regression_forecast` pipeline is re-run with shuffled regime
conditioning, and the resulting portfolio Sharpe is recorded.

### Why this complements the existing temporal permutation test
The existing permutation test in `regime_pipeline.py` shuffles labels and runs
`naive_forecast` (conditional mean/vol Sharpe). That tests whether knowing the
regime identity helps a simple conditional-mean predictor.

This cross-sectional control tests a stronger hypothesis: whether regime
conditioning helps the **ridge regression** forecast model. The ridge model uses
macro PCA features AND regime-conditional training subsets. If shuffling regime
labels degrades performance, it confirms that:
1. The regime detection genuinely partitions the feature space informatively.
2. The per-regime ridge models learn different coefficient surfaces.
3. The Sharpe improvement is not an artefact of portfolio construction rules alone.

### Random seed policy
- Base seed: {BASE_SEED}
- Each permutation i uses `np.random.RandomState({BASE_SEED} + i)`
- Fully deterministic and reproducible

### Shuffle method
`np.random.permutation(regime_labels)` — uniform random permutation of the
label vector. Preserves marginal regime frequencies but destroys temporal structure.
Regime probabilities are reconstructed as one-hot from shuffled labels.

## Results

| Metric | Value |
|---|---|
| Observed Sharpe (Ridge_LO_l3) | {real_sharpe:.4f} |
| Permutation mean | {perm_mean:.4f} |
| Permutation std | {perm_std:.4f} |
| Permutation median | {perm_median:.4f} |
| Permutation 5th percentile | {perm_p5:.4f} |
| Permutation 25th percentile | {perm_p25:.4f} |
| Permutation 75th percentile | {perm_p75:.4f} |
| Permutation 95th percentile | {perm_p95:.4f} |
| Observed percentile rank | {percentile_rank:.1f}% |
| Empirical p-value | {p_empirical:.4f} |
| t-statistic (1-sample) | {t_stat:.4f} |
| Valid permutations | {n_valid} / {N_PERMUTATIONS} |

## Pass/Fail

**Criterion:** Observed Sharpe must be at or above the {PASS_PERCENTILE}th percentile
of the shuffled distribution (empirical p-value <= {1 - PASS_PERCENTILE/100:.2f}).

**Result: {'PASS ✓' if passed else 'FAIL ✗'}**

{'The observed strategy Sharpe exceeds ' + str(PASS_PERCENTILE) + '% of shuffled-regime baselines, confirming that regime conditioning provides genuine informational value to the ridge regression forecast model.' if passed else 'The observed strategy Sharpe does NOT exceed the ' + str(PASS_PERCENTILE) + 'th percentile threshold. This suggests the ridge model may derive limited benefit from regime conditioning beyond what random label assignment provides.'}

## Artifacts

- `negative_control_t5_distribution.csv` — Per-permutation Sharpe values with seeds
- `negative_control_t5_plot.png` — Histogram of shuffled distribution vs observed
- `negative_control_t5_2503.md` — This report
"""

    report_path = os.path.join(OUTPUT_DIR, 'negative_control_t5_2503.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"[OUTPUT] Saved: {report_path}")

    # ---- Summary JSON (for downstream consumption) ----
    summary = {
        'timestamp': datetime.datetime.now().isoformat(),
        'test': 'cross_sectional_negative_control_t5',
        'strategy': 'Ridge_LO_l3',
        'n_permutations': N_PERMUTATIONS,
        'n_valid': int(n_valid),
        'base_seed': BASE_SEED,
        'observed_sharpe': float(real_sharpe),
        'perm_mean': round(float(perm_mean), 4),
        'perm_std': round(float(perm_std), 4),
        'perm_median': round(float(perm_median), 4),
        'perm_p5': round(float(perm_p5), 4),
        'perm_p25': round(float(perm_p25), 4),
        'perm_p75': round(float(perm_p75), 4),
        'perm_p95': round(float(perm_p95), 4),
        'percentile_rank': round(float(percentile_rank), 2),
        'p_empirical': round(float(p_empirical), 4),
        't_stat': round(float(t_stat), 4),
        'pass_criterion': f'>= {PASS_PERCENTILE}th percentile',
        'passed': bool(passed),
    }
    json_path = os.path.join(OUTPUT_DIR, 'negative_control_t5_results.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[OUTPUT] Saved: {json_path}")

    print("\n" + "=" * 70)
    print(f"M2 COMPLETE — {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    return summary


if __name__ == '__main__':
    summary = main()
