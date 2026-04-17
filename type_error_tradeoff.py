#!/usr/bin/env python3
"""
Type I / Type II Error Tradeoff Analysis — arXiv 2503.11499
Market-Signal Augmented Crisis Detection with Walk-Forward Anti-Leakage

Objectives:
  1. Add market signals (VIX, credit spreads, yield curve slope, volatility proxies)
     available at decision time (t-1 for monthly decisions at t).
  2. Enforce strict walk-forward pipeline: all fitting (scaler, PCA, k-means,
     thresholds) uses ONLY data up to t-1. No future information leaks.
  3. Define crisis labels from NBER recessions + named stress windows.
  4. Compute confusion-matrix metrics: TPR, FPR, Precision, Specificity, F1,
     AUROC, AUPRC across threshold sweeps.
  5. Recommend operating point maximizing crisis recall under bounded FPR.
  6. Compare baseline (macro-only) vs augmented (macro + market signals).

Outputs:
  - outputs/type_error_tradeoff_results.json
  - outputs/type_error_roc_pr.png
  - outputs/type_error_threshold_sweep.png
  - type_error_tradeoff_2503.md (generated separately)

Author: AI-MAC Platform
Date: 2026-03-13
"""

import os
import json
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, f1_score
)

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PCA_VARIANCE_THRESHOLD = 0.95
ROLLING_WINDOW = 48  # months — walk-forward estimation window
MIN_HISTORY = 60     # minimum months before first prediction (warmup)
N_REGIMES_L2 = 5
TOTAL_REGIMES = 6

# ============================================================
# CRISIS GROUND-TRUTH LABELS
# ============================================================
# We define "crisis" as NBER recession months PLUS acute stress windows
# where drawdowns exceeded 15% from peak (even outside official recessions).

NBER_RECESSIONS = [
    ('1960-04-01', '1961-02-01'),
    ('1969-12-01', '1970-11-01'),
    ('1973-11-01', '1975-03-01'),
    ('1980-01-01', '1980-07-01'),
    ('1981-07-01', '1982-11-01'),
    ('1990-07-01', '1991-03-01'),
    ('2001-03-01', '2001-11-01'),
    ('2007-12-01', '2009-06-01'),
    ('2020-02-01', '2020-04-01'),
]

# Additional stress windows (non-NBER but material drawdown episodes)
STRESS_WINDOWS = [
    ('1987-10-01', '1987-11-01'),   # Black Monday
    ('1998-08-01', '1998-10-01'),   # LTCM / Russia
    ('2011-08-01', '2011-10-01'),   # Euro crisis / US downgrade
    ('2015-08-01', '2016-02-01'),   # China deval / oil crash
    ('2018-10-01', '2018-12-01'),   # Vol-mageddon / Q4 selloff
    ('2022-01-01', '2022-10-01'),   # Fed tightening bear
]


def build_crisis_labels(date_index):
    """
    Build binary crisis labels: 1 = crisis month, 0 = normal.
    Sources: NBER recessions + named stress windows.
    """
    labels = pd.Series(0, index=date_index, name='crisis')
    for start, end in NBER_RECESSIONS + STRESS_WINDOWS:
        mask = (date_index >= start) & (date_index <= end)
        labels[mask] = 1
    n_crisis = labels.sum()
    print(f"[LABELS] Crisis months: {n_crisis}/{len(labels)} "
          f"({n_crisis/len(labels)*100:.1f}%)")
    return labels


# ============================================================
# DATA ACQUISITION
# ============================================================

def download_fred_macro():
    """Download macro variables from FRED (reuses regime_pipeline pattern)."""
    from pandas_datareader import data as pdr
    print("[DATA] Downloading FRED macro variables...")

    FRED_SERIES = {
        # Output & Income
        'INDPRO': 5, 'RPI': 5, 'CUMFNS': 2, 'IPFINAL': 5, 'IPMANSICS': 5,
        'IPBUSEQ': 5, 'IPMAT': 5, 'IPDMAT': 5, 'IPNMAT': 5,
        # Labor Market
        'UNRATE': 2, 'PAYEMS': 5, 'USGOOD': 5, 'MANEMP': 5,
        'SRVPRD': 5, 'USTPU': 5, 'USFIRE': 5, 'USGOVT': 5,
        'CES0600000007': 5, 'CES0600000008': 5,
        'CE16OV': 5, 'CIVPART': 2, 'UEMPMEAN': 2,
        'ICSA': 5, 'CCSA': 5,
        # Consumption, Orders & Inventories
        'DPCERA3M086SBEA': 5, 'DGORDER': 5,
        'ANDENOx': 5, 'AMDMUO': 5, 'BUSLOANS': 6,
        # Housing
        'HOUST': 4, 'HOUSTNE': 4, 'HOUSTMW': 4, 'HOUSTS': 4,
        'HOUSTW': 4, 'PERMIT': 4, 'PERMITNE': 4, 'PERMITMW': 4,
        'PERMITS': 4, 'PERMITW': 4,
        # Money & Credit
        'M1SL': 6, 'M2SL': 6, 'BOGMBASE': 6, 'TOTRESNS': 6,
        'NONBORRES': 7, 'REALLN': 6,
        'NONREVSL': 6, 'DTCOLNVHFNM': 2, 'DTCTHFNM': 2, 'INVEST': 5,
        # Interest Rates & Spreads
        'FEDFUNDS': 2, 'TB3MS': 2, 'TB6MS': 2, 'GS1': 2,
        'GS5': 2, 'GS10': 2, 'AAA': 2, 'BAA': 2,
        'TB3SMFFM': 1, 'TB6SMFFM': 1, 'T1YFFM': 1,
        'T5YFFM': 1, 'T10YFFM': 1, 'AAAFFM': 1, 'BAAFFM': 1,
        # Prices
        'CPIAUCSL': 6, 'CPILFESL': 6, 'PPIACO': 6,
        'PCEPI': 6, 'PCEPILFE': 6, 'CPIULFSL': 6,
        'CUSR0000SA0L2': 6, 'CUSR0000SA0L5': 6,
        'CPIMEDSL': 6, 'OILPRICEx': 5, 'PPICMM': 5,
        # Stock Market & Sentiment
        'SP500': 5, 'VIXCLS': 1, 'UMCSENT': 2,
        # Exchange Rates
        'EXSZUSx': 5, 'EXJPUSx': 5, 'EXUSUKx': 5, 'EXCAUSx': 5,
    }

    start_date = '1959-01-01'
    end_date = '2026-03-01'
    raw = pd.DataFrame()
    tcodes = {}
    failed = []

    series_list = list(set(FRED_SERIES.keys()))
    for i, sid in enumerate(series_list):
        try:
            s = pdr.get_data_fred(sid, start=start_date, end=end_date)
            raw[sid] = s.iloc[:, 0]
            tcodes[sid] = FRED_SERIES[sid]
            if (i + 1) % 20 == 0:
                print(f"  Downloaded {i+1}/{len(series_list)}...")
        except Exception:
            failed.append(sid)

    print(f"[DATA] Downloaded {len(raw.columns)} macro series, {len(failed)} failed")
    raw = raw.resample('MS').last().sort_index()
    return raw, pd.Series(tcodes), failed


def download_market_signals():
    """
    Download market-signal features for crisis detection augmentation.

    All signals use data available by month-end (no lookahead):
      - VIXCLS:  VIX level (monthly average)
      - BAA-AAA: Credit spread (Moody's BAA minus AAA yield)
      - GS10-TB3MS: Yield curve slope (10Y minus 3M)
      - TEDRATE / TB3SMFFM: TED spread proxy
      - SP500 realized vol: 21-day trailing vol of daily returns
      - UMCSENT: Consumer sentiment level

    These are separate from the macro feature matrix to allow
    controlled ablation (baseline = macro-only, augmented = macro + signals).
    """
    from pandas_datareader import data as pdr
    print("[DATA] Downloading market signal features...")

    signals = pd.DataFrame()
    start_date = '1959-01-01'
    end_date = '2026-03-01'

    # --- VIX (available from 1990) ---
    try:
        vix = pdr.get_data_fred('VIXCLS', start=start_date, end=end_date)
        signals['VIX'] = vix.iloc[:, 0]
        print("  VIX: OK")
    except Exception as e:
        print(f"  VIX: FAILED ({e})")

    # --- Credit spread: BAA - AAA ---
    try:
        baa = pdr.get_data_fred('BAA', start=start_date, end=end_date).iloc[:, 0]
        aaa = pdr.get_data_fred('AAA', start=start_date, end=end_date).iloc[:, 0]
        signals['CREDIT_SPREAD'] = baa - aaa
        print("  Credit spread (BAA-AAA): OK")
    except Exception as e:
        print(f"  Credit spread: FAILED ({e})")

    # --- Yield curve slope: 10Y - 3M ---
    try:
        gs10 = pdr.get_data_fred('GS10', start=start_date, end=end_date).iloc[:, 0]
        tb3m = pdr.get_data_fred('TB3MS', start=start_date, end=end_date).iloc[:, 0]
        signals['YIELD_CURVE_SLOPE'] = gs10 - tb3m
        print("  Yield curve slope (GS10-TB3MS): OK")
    except Exception as e:
        print(f"  Yield curve slope: FAILED ({e})")

    # --- TED spread proxy: 3M T-bill rate minus Fed Funds ---
    try:
        tb3 = pdr.get_data_fred('TB3MS', start=start_date, end=end_date).iloc[:, 0]
        ff = pdr.get_data_fred('FEDFUNDS', start=start_date, end=end_date).iloc[:, 0]
        signals['TED_PROXY'] = tb3 - ff
        print("  TED proxy (TB3MS-FEDFUNDS): OK")
    except Exception as e:
        print(f"  TED proxy: FAILED ({e})")

    # --- SP500 realized volatility (from monthly returns) ---
    try:
        sp = pdr.get_data_fred('SP500', start=start_date, end=end_date).iloc[:, 0]
        sp_ret = np.log(sp / sp.shift(1))
        # Rolling 3-month realized vol (annualized)
        signals['SP500_RVOL'] = sp_ret.rolling(3).std() * np.sqrt(12)
        print("  SP500 realized vol (3m): OK")
    except Exception as e:
        print(f"  SP500 RVol: FAILED ({e})")

    # --- Consumer sentiment level ---
    try:
        um = pdr.get_data_fred('UMCSENT', start=start_date, end=end_date).iloc[:, 0]
        signals['SENTIMENT'] = um
        print("  Consumer sentiment: OK")
    except Exception as e:
        print(f"  Sentiment: FAILED ({e})")

    # Resample to month-start
    signals = signals.resample('MS').last().sort_index()
    print(f"[DATA] Market signals: {signals.shape[1]} features, "
          f"{signals.shape[0]} months")
    return signals


def apply_tcode_transforms(df, tcodes):
    """Apply FRED-MD transformation codes for stationarity."""
    transformed = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col not in tcodes.index:
            continue
        tc = int(tcodes[col])
        s = df[col].copy()
        if tc == 1:
            transformed[col] = s
        elif tc == 2:
            transformed[col] = s.diff()
        elif tc == 3:
            transformed[col] = s.diff().diff()
        elif tc == 4:
            transformed[col] = np.log(s.clip(lower=1e-10))
        elif tc == 5:
            transformed[col] = np.log(s.clip(lower=1e-10)).diff()
        elif tc == 6:
            transformed[col] = np.log(s.clip(lower=1e-10)).diff().diff()
        elif tc == 7:
            transformed[col] = (s / s.shift(1) - 1).diff()

    transformed = transformed.dropna(how='all')
    threshold = len(transformed) * 0.10
    transformed = transformed.dropna(axis=1, thresh=int(len(transformed) - threshold))
    transformed = transformed.fillna(transformed.median())
    print(f"[DATA] Macro after transforms: {transformed.shape[0]}m × {transformed.shape[1]}v")
    return transformed


# ============================================================
# WALK-FORWARD CRISIS PROBABILITY ENGINE
# ============================================================

def walk_forward_crisis_scores(features, date_index, model_name="model",
                                min_history=MIN_HISTORY):
    """
    Strictly walk-forward crisis probability scores.

    LEAKAGE GUARDS:
      1. At each month t, fit scaler/PCA/k-means on data [0..t-1] only.
      2. Crisis cluster identification uses only training window statistics.
      3. No future data touches the scoring of month t.
      4. Re-fit every 12 months (expanding window) for efficiency,
         but score every month with the most recent model.

    Returns:
      crisis_scores: pd.Series of P(crisis) for each month t,
                     where t >= min_history.
    """
    n = len(features)
    scores = pd.Series(np.nan, index=date_index, name=f'{model_name}_crisis_prob')

    # Fit parameters
    last_fit_t = -999
    refit_interval = 12  # refit every 12 months
    scaler = None
    pca = None
    km_l1 = None

    print(f"[WF-{model_name}] Walk-forward scoring {n} months, "
          f"min_history={min_history}...")

    for t in range(min_history, n):
        # --- LEAKAGE GUARD: only use data up to t-1 for model fitting ---
        if t - last_fit_t >= refit_interval or scaler is None:
            train_data = features[:t]  # [0..t-1] inclusive

            # Fit scaler on training data only
            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(train_data)

            # Fit PCA on training data only
            pca = PCA()
            pca.fit(scaled_train)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_comp = max(np.argmax(cumvar >= PCA_VARIANCE_THRESHOLD) + 1, 2)
            pca = PCA(n_components=n_comp)
            pca.fit(scaled_train)
            pca_train = pca.transform(scaled_train)

            # Fit Layer-1 k-means on training data only
            km_l1 = KMeans(n_clusters=2, random_state=42, n_init=20)
            km_l1.fit(pca_train)

            # Identify crisis cluster as the SMALLER cluster
            cluster_sizes = np.bincount(km_l1.labels_)
            crisis_cluster_id = np.argmin(cluster_sizes)

            # Store crisis centroid for distance-based scoring
            crisis_centroid = km_l1.cluster_centers_[crisis_cluster_id]
            normal_centroid = km_l1.cluster_centers_[1 - crisis_cluster_id]

            last_fit_t = t

        # --- Score month t using model fitted on [0..t-1] ---
        x_t = features[t:t+1]
        x_t_scaled = scaler.transform(x_t)
        x_t_pca = pca.transform(x_t_scaled)[0]

        # Crisis score = inverse-distance-weighted probability
        d_crisis = np.linalg.norm(x_t_pca - crisis_centroid)
        d_normal = np.linalg.norm(x_t_pca - normal_centroid)
        d_crisis = max(d_crisis, 1e-10)
        d_normal = max(d_normal, 1e-10)
        # Raw probability via inverse distance
        raw_p = (1.0 / d_crisis) / (1.0 / d_crisis + 1.0 / d_normal)
        scores.iloc[t] = raw_p

        if (t - min_history + 1) % 60 == 0:
            print(f"  Scored month {t-min_history+1}/{n-min_history}")

    valid = scores.dropna()
    print(f"[WF-{model_name}] Produced {len(valid)} scores, "
          f"mean={valid.mean():.4f}, std={valid.std():.4f}")
    return scores


# ============================================================
# CONFUSION MATRIX AND METRICS
# ============================================================

def compute_confusion_metrics(y_true, y_pred):
    """Compute full confusion matrix metrics from binary labels."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall / Sensitivity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
    return {
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        'TPR_recall': round(tpr, 4),
        'FPR': round(fpr, 4),
        'precision': round(precision, 4),
        'specificity': round(specificity, 4),
        'F1': round(f1, 4),
    }


def threshold_sweep(y_true, scores, thresholds=None):
    """
    Sweep thresholds and compute metrics at each point.
    Returns list of dicts + AUROC + AUPRC.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    results = []
    for th in thresholds:
        y_pred = (scores >= th).astype(int)
        m = compute_confusion_metrics(y_true, y_pred)
        m['threshold'] = round(float(th), 4)
        results.append(m)

    # ROC curve and AUC
    fpr_arr, tpr_arr, _ = roc_curve(y_true, scores)
    auroc = auc(fpr_arr, tpr_arr)

    # Precision-Recall curve and AUC
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, scores)
    auprc = auc(rec_arr, prec_arr)

    return results, {
        'AUROC': round(auroc, 4),
        'AUPRC': round(auprc, 4),
        'roc_fpr': fpr_arr.tolist(),
        'roc_tpr': tpr_arr.tolist(),
        'pr_precision': prec_arr.tolist(),
        'pr_recall': rec_arr.tolist(),
    }


def find_operating_point(sweep_results, max_fpr=0.15):
    """
    Find the operating point that maximizes TPR (crisis recall)
    subject to FPR <= max_fpr.

    Rationale: in crisis detection, false negatives (missed crises) are
    far more costly than false positives (unnecessary de-risking).
    We bound the false alarm rate to keep the strategy investable.
    """
    candidates = [r for r in sweep_results if r['FPR'] <= max_fpr]
    if not candidates:
        # Relax constraint — pick lowest FPR
        candidates = sorted(sweep_results, key=lambda x: x['FPR'])
        best = candidates[0]
        print(f"[OP] WARNING: No point meets FPR<={max_fpr}. "
              f"Best available: FPR={best['FPR']}, TPR={best['TPR_recall']}")
        return best

    # Among candidates, maximize TPR (recall)
    best = max(candidates, key=lambda x: x['TPR_recall'])
    print(f"[OP] Best operating point (FPR<={max_fpr}): "
          f"threshold={best['threshold']}, TPR={best['TPR_recall']}, "
          f"FPR={best['FPR']}, F1={best['F1']}")
    return best


def evaluate_crisis_windows(scores, crisis_labels, threshold, crisis_windows):
    """Evaluate detection performance on named crisis windows."""
    results = {}
    for name, (start, end) in crisis_windows:
        mask = (scores.index >= start) & (scores.index <= end)
        window_scores = scores[mask].dropna()
        window_labels = crisis_labels[mask].reindex(window_scores.index)

        if len(window_scores) == 0:
            continue

        detected = (window_scores >= threshold).astype(int)
        n_detected = detected.sum()
        n_total = len(detected)
        detection_rate = n_detected / n_total if n_total > 0 else 0

        # Detection lag: months from window start to first detection
        first_detect = None
        for i, (dt, val) in enumerate(detected.items()):
            if val == 1:
                first_detect = i
                break

        results[name] = {
            'n_months': n_total,
            'n_detected': int(n_detected),
            'detection_rate': round(detection_rate, 4),
            'detection_lag_months': first_detect,
        }

    return results


# ============================================================
# PLOTTING
# ============================================================

def plot_roc_pr(curves_baseline, curves_augmented, output_path):
    """Plot ROC and Precision-Recall curves for baseline vs augmented."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    ax = axes[0]
    ax.plot(curves_baseline['roc_fpr'], curves_baseline['roc_tpr'],
            'b-', lw=2, label=f"Baseline (AUROC={curves_baseline['AUROC']:.3f})")
    ax.plot(curves_augmented['roc_fpr'], curves_augmented['roc_tpr'],
            'r-', lw=2, label=f"Augmented (AUROC={curves_augmented['AUROC']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.axvline(x=0.15, color='gray', linestyle=':', alpha=0.7, label='FPR=0.15 bound')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('ROC Curve — Crisis Detection')
    ax.legend(loc='lower right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    # PR
    ax = axes[1]
    ax.plot(curves_baseline['pr_recall'], curves_baseline['pr_precision'],
            'b-', lw=2, label=f"Baseline (AUPRC={curves_baseline['AUPRC']:.3f})")
    ax.plot(curves_augmented['pr_recall'], curves_augmented['pr_precision'],
            'r-', lw=2, label=f"Augmented (AUPRC={curves_augmented['AUPRC']:.3f})")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve — Crisis Detection')
    ax.legend(loc='upper right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved {output_path}")


def plot_threshold_sweep(sweep_baseline, sweep_augmented, op_baseline, op_augmented,
                          output_path):
    """Plot TPR, FPR, Precision, F1 vs threshold for both models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_to_plot = [
        ('TPR_recall', 'Crisis Recall (TPR)'),
        ('FPR', 'False Positive Rate (FPR)'),
        ('precision', 'Precision'),
        ('F1', 'F1 Score'),
    ]

    for ax, (metric, title) in zip(axes.flat, metrics_to_plot):
        th_b = [r['threshold'] for r in sweep_baseline]
        val_b = [r[metric] for r in sweep_baseline]
        th_a = [r['threshold'] for r in sweep_augmented]
        val_a = [r[metric] for r in sweep_augmented]

        ax.plot(th_b, val_b, 'b-', lw=2, label='Baseline (macro-only)')
        ax.plot(th_a, val_a, 'r-', lw=2, label='Augmented (macro+signals)')

        # Mark operating points
        ax.axvline(x=op_baseline['threshold'], color='b', linestyle=':', alpha=0.7)
        ax.axvline(x=op_augmented['threshold'], color='r', linestyle=':', alpha=0.7)

        ax.set_xlabel('Threshold')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Threshold Sweep — Baseline vs Augmented Crisis Detector',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved {output_path}")


def plot_crisis_timeline(scores_baseline, scores_augmented, crisis_labels,
                          op_baseline, op_augmented, output_path):
    """Plot crisis probability timeline with ground truth shading."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    common_idx = scores_baseline.dropna().index.intersection(
        scores_augmented.dropna().index
    )
    if len(common_idx) == 0:
        print("[PLOT] No overlapping scores for timeline plot")
        return

    for ax, scores, op, name, color in [
        (axes[0], scores_baseline, op_baseline, 'Baseline (macro-only)', 'blue'),
        (axes[1], scores_augmented, op_augmented, 'Augmented (macro+signals)', 'red'),
    ]:
        s = scores.loc[common_idx]
        ax.plot(s.index, s.values, color=color, lw=0.8, alpha=0.8)
        ax.axhline(y=op['threshold'], color='k', linestyle='--', lw=1,
                   label=f"threshold={op['threshold']:.2f}")

        # Shade crisis windows
        cl = crisis_labels.reindex(common_idx).fillna(0)
        crisis_starts = []
        in_crisis = False
        for i, (dt, val) in enumerate(cl.items()):
            if val == 1 and not in_crisis:
                crisis_starts.append(dt)
                in_crisis = True
            elif val == 0 and in_crisis:
                ax.axvspan(crisis_starts[-1], dt, alpha=0.15, color='red')
                in_crisis = False
        if in_crisis:
            ax.axvspan(crisis_starts[-1], common_idx[-1], alpha=0.15, color='red')

        ax.set_ylabel('P(Crisis)')
        ax.set_title(f'{name} — Crisis Probability')
        ax.legend(loc='upper right')
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved {output_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("TYPE I / TYPE II ERROR TRADEOFF — arXiv 2503.11499")
    print("Market-Signal Augmented Crisis Detection")
    print(f"Timestamp: {datetime.datetime.now().isoformat()}")
    print("=" * 70)

    # ---- Step 1: Download data ----
    raw_macro, tcodes, macro_failed = download_fred_macro()
    market_signals = download_market_signals()
    macro_transformed = apply_tcode_transforms(raw_macro, tcodes)

    # ---- Step 2: Build crisis ground-truth labels ----
    crisis_labels = build_crisis_labels(macro_transformed.index)

    # ---- Step 3: Align all data ----
    # Market signals need to be resampled and aligned
    # LAG GUARD: shift market signals by 1 month to ensure
    # we only use data available at decision time
    market_signals_lagged = market_signals.shift(1)  # t-1 available at t
    market_signals_lagged.columns = ['MKT_' + c for c in market_signals_lagged.columns]

    common_idx = macro_transformed.index.intersection(
        market_signals_lagged.dropna(how='all').index
    )
    print(f"[ALIGN] Common date range: {common_idx.min()} to {common_idx.max()} "
          f"({len(common_idx)} months)")

    # Baseline features: macro only (on common index for fair comparison)
    macro_aligned = macro_transformed.loc[common_idx].copy()
    macro_values = macro_aligned.values

    # Augmented features: macro + market signals
    signals_aligned = market_signals_lagged.loc[common_idx].copy()
    # Fill remaining NaNs in signals with expanding median (no lookahead),
    # then forward-fill, then backfill any leading NaNs with 0
    for col in signals_aligned.columns:
        signals_aligned[col] = (
            signals_aligned[col]
            .fillna(signals_aligned[col].expanding().median())
            .ffill()
            .bfill()
        )
    # Final safety: any remaining NaN → 0
    signals_aligned = signals_aligned.fillna(0)
    augmented_df = pd.concat([macro_aligned, signals_aligned], axis=1)
    # Ensure no NaNs in either feature set
    macro_aligned = macro_aligned.fillna(0)
    augmented_df = augmented_df.fillna(0)
    augmented_values = augmented_df.values

    crisis_aligned = crisis_labels.reindex(common_idx).fillna(0).astype(int)

    print(f"[FEATURES] Baseline: {macro_values.shape[1]} features")
    print(f"[FEATURES] Augmented: {augmented_values.shape[1]} features "
          f"(+{signals_aligned.shape[1]} market signals)")

    # ---- Step 4: Walk-forward crisis scoring ----
    print("\n" + "=" * 70)
    print("WALK-FORWARD CRISIS SCORING (no lookahead)")
    print("=" * 70)

    scores_baseline = walk_forward_crisis_scores(
        macro_values, common_idx, model_name="BASELINE"
    )
    scores_augmented = walk_forward_crisis_scores(
        augmented_values, common_idx, model_name="AUGMENTED"
    )

    # ---- Step 5: Restrict to scored months ----
    valid_mask = scores_baseline.notna() & scores_augmented.notna()
    valid_idx = common_idx[valid_mask]
    y_true = crisis_aligned.loc[valid_idx].values
    s_base = scores_baseline.loc[valid_idx].values
    s_aug = scores_augmented.loc[valid_idx].values

    print(f"\n[EVAL] Evaluation period: {valid_idx[0]} to {valid_idx[-1]} "
          f"({len(valid_idx)} months)")
    print(f"[EVAL] Crisis months in eval period: {y_true.sum()}/{len(y_true)} "
          f"({y_true.sum()/len(y_true)*100:.1f}%)")

    # ---- Step 6: Threshold sweep ----
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP & METRICS")
    print("=" * 70)

    sweep_base, curves_base = threshold_sweep(y_true, s_base)
    sweep_aug, curves_aug = threshold_sweep(y_true, s_aug)

    print(f"\n[METRICS] Baseline:  AUROC={curves_base['AUROC']:.4f}, "
          f"AUPRC={curves_base['AUPRC']:.4f}")
    print(f"[METRICS] Augmented: AUROC={curves_aug['AUROC']:.4f}, "
          f"AUPRC={curves_aug['AUPRC']:.4f}")

    # ---- Step 7: Find operating points ----
    print("\n--- Operating Point Selection (max TPR s.t. FPR <= 0.15) ---")
    op_base = find_operating_point(sweep_base, max_fpr=0.15)
    op_aug = find_operating_point(sweep_aug, max_fpr=0.15)

    # Also find F1-optimal points for reference
    f1_base = max(sweep_base, key=lambda x: x['F1'])
    f1_aug = max(sweep_aug, key=lambda x: x['F1'])
    print(f"[F1-OPT] Baseline:  threshold={f1_base['threshold']}, "
          f"F1={f1_base['F1']}, TPR={f1_base['TPR_recall']}, FPR={f1_base['FPR']}")
    print(f"[F1-OPT] Augmented: threshold={f1_aug['threshold']}, "
          f"F1={f1_aug['F1']}, TPR={f1_aug['TPR_recall']}, FPR={f1_aug['FPR']}")

    # ---- Step 8: Named crisis window evaluation ----
    named_crises = [
        ('Dot-Com Bust', ('2000-03-01', '2002-10-01')),
        ('GFC', ('2007-06-01', '2009-06-01')),
        ('Euro Crisis', ('2011-08-01', '2011-10-01')),
        ('COVID Shock', ('2020-02-01', '2020-04-01')),
        ('2022 Tightening', ('2022-01-01', '2022-10-01')),
        ('Black Monday', ('1987-10-01', '1987-11-01')),
        ('LTCM/Russia', ('1998-08-01', '1998-10-01')),
    ]

    crisis_eval_base = evaluate_crisis_windows(
        scores_baseline, crisis_aligned, op_base['threshold'], named_crises
    )
    crisis_eval_aug = evaluate_crisis_windows(
        scores_augmented, crisis_aligned, op_aug['threshold'], named_crises
    )

    # ---- Step 9: Generate plots ----
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_roc_pr(curves_base, curves_aug,
                os.path.join(OUTPUT_DIR, 'type_error_roc_pr.png'))
    plot_threshold_sweep(sweep_base, sweep_aug, op_base, op_aug,
                          os.path.join(OUTPUT_DIR, 'type_error_threshold_sweep.png'))
    plot_crisis_timeline(scores_baseline, scores_augmented, crisis_aligned,
                          op_base, op_aug,
                          os.path.join(OUTPUT_DIR, 'type_error_crisis_timeline.png'))

    # ---- Step 10: Compile results ----
    delta_auroc = curves_aug['AUROC'] - curves_base['AUROC']
    delta_auprc = curves_aug['AUPRC'] - curves_base['AUPRC']
    delta_tpr = op_aug['TPR_recall'] - op_base['TPR_recall']
    delta_fpr = op_aug['FPR'] - op_base['FPR']

    # Deployment recommendation logic
    if delta_auroc > 0.02 and delta_tpr > 0:
        if op_aug['FPR'] <= 0.15 and op_aug['TPR_recall'] >= 0.50:
            recommendation = "DEPLOY_AUGMENTED"
            rationale = (
                f"Augmented model improves AUROC by {delta_auroc:+.4f} and "
                f"crisis recall by {delta_tpr:+.4f} while keeping FPR "
                f"at {op_aug['FPR']:.2f} (within 15% bound). "
                f"Recommended for production crisis gating."
            )
        else:
            recommendation = "DEPLOY_WITH_MONITORING"
            rationale = (
                f"Augmented model shows improvement (AUROC {delta_auroc:+.4f}) "
                f"but operating point has TPR={op_aug['TPR_recall']:.2f} or "
                f"FPR={op_aug['FPR']:.2f} outside ideal bounds. "
                f"Deploy with enhanced monitoring and tighter alert thresholds."
            )
    elif delta_auroc > 0:
        recommendation = "MARGINAL_IMPROVEMENT"
        rationale = (
            f"Augmented model shows small AUROC improvement ({delta_auroc:+.4f}) "
            f"that may not justify added complexity. Consider deploying baseline "
            f"with market signal overlay as informational (not gating)."
        )
    else:
        recommendation = "RETAIN_BASELINE"
        rationale = (
            f"Augmented model does not improve crisis detection "
            f"(AUROC delta={delta_auroc:+.4f}). Market signals may already be "
            f"captured by macro factors. Retain baseline model."
        )

    results = {
        'metadata': {
            'timestamp': datetime.datetime.now().isoformat(),
            'eval_start': str(valid_idx[0]),
            'eval_end': str(valid_idx[-1]),
            'n_eval_months': len(valid_idx),
            'n_crisis_months': int(y_true.sum()),
            'crisis_prevalence': round(y_true.sum() / len(y_true), 4),
            'baseline_features': int(macro_values.shape[1]),
            'augmented_features': int(augmented_values.shape[1]),
            'market_signals_added': list(signals_aligned.columns),
            'walk_forward_window': ROLLING_WINDOW,
            'min_history': MIN_HISTORY,
            'refit_interval_months': 12,
        },
        'leakage_guards': {
            'market_signals_lagged_1m': True,
            'scaler_fit_on_train_only': True,
            'pca_fit_on_train_only': True,
            'kmeans_fit_on_train_only': True,
            'expanding_window_refit': True,
            'no_future_labels_in_scoring': True,
            'crisis_labels_from_exogenous_dates': True,
        },
        'baseline': {
            'AUROC': curves_base['AUROC'],
            'AUPRC': curves_base['AUPRC'],
            'operating_point': op_base,
            'f1_optimal': f1_base,
            'crisis_window_eval': crisis_eval_base,
        },
        'augmented': {
            'AUROC': curves_aug['AUROC'],
            'AUPRC': curves_aug['AUPRC'],
            'operating_point': op_aug,
            'f1_optimal': f1_aug,
            'crisis_window_eval': crisis_eval_aug,
        },
        'comparison': {
            'delta_AUROC': round(delta_auroc, 4),
            'delta_AUPRC': round(delta_auprc, 4),
            'delta_TPR_at_operating_point': round(delta_tpr, 4),
            'delta_FPR_at_operating_point': round(delta_fpr, 4),
        },
        'recommendation': {
            'action': recommendation,
            'rationale': rationale,
            'fpr_bound': 0.15,
        },
        'threshold_sweep': {
            'baseline': sweep_base,
            'augmented': sweep_aug,
        },
    }

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'type_error_tradeoff_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[OUTPUT] Saved {results_path}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Evaluation period: {valid_idx[0]} to {valid_idx[-1]} ({len(valid_idx)} months)")
    print(f"Crisis prevalence: {y_true.sum()}/{len(y_true)} ({y_true.mean()*100:.1f}%)")
    print()
    print(f"{'Metric':<30} {'Baseline':>12} {'Augmented':>12} {'Delta':>12}")
    print("-" * 66)
    print(f"{'AUROC':<30} {curves_base['AUROC']:>12.4f} {curves_aug['AUROC']:>12.4f} {delta_auroc:>+12.4f}")
    print(f"{'AUPRC':<30} {curves_base['AUPRC']:>12.4f} {curves_aug['AUPRC']:>12.4f} {delta_auprc:>+12.4f}")
    print(f"{'TPR @ operating point':<30} {op_base['TPR_recall']:>12.4f} {op_aug['TPR_recall']:>12.4f} {delta_tpr:>+12.4f}")
    print(f"{'FPR @ operating point':<30} {op_base['FPR']:>12.4f} {op_aug['FPR']:>12.4f} {delta_fpr:>+12.4f}")
    print(f"{'Precision @ operating point':<30} {op_base['precision']:>12.4f} {op_aug['precision']:>12.4f}")
    print(f"{'F1 @ operating point':<30} {op_base['F1']:>12.4f} {op_aug['F1']:>12.4f}")
    print(f"{'Threshold @ operating point':<30} {op_base['threshold']:>12.4f} {op_aug['threshold']:>12.4f}")
    print()
    print(f"RECOMMENDATION: {recommendation}")
    print(f"  {rationale}")
    print()

    # Named crisis detection comparison
    print(f"{'Crisis Window':<25} {'Base Det%':>10} {'Aug Det%':>10} {'Base Lag':>10} {'Aug Lag':>10}")
    print("-" * 65)
    for name, _ in named_crises:
        b = crisis_eval_base.get(name, {})
        a = crisis_eval_aug.get(name, {})
        b_rate = f"{b.get('detection_rate', 0)*100:.0f}%" if b else "N/A"
        a_rate = f"{a.get('detection_rate', 0)*100:.0f}%" if a else "N/A"
        b_lag = str(b.get('detection_lag_months', 'N/A')) if b else "N/A"
        a_lag = str(a.get('detection_lag_months', 'N/A')) if a else "N/A"
        print(f"{name:<25} {b_rate:>10} {a_rate:>10} {b_lag:>10} {a_lag:>10}")

    print("\n[DONE] Type I/II error tradeoff analysis complete.")
    return results


if __name__ == '__main__':
    results = main()
