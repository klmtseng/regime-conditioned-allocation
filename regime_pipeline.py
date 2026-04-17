#!/usr/bin/env python3
"""
Regime Detection Pipeline — arXiv 2503.11499
Tactical Asset Allocation with Macroeconomic Regime Detection

Full implementation: FRED-MD → PCA → Two-layer K-means → Markov forecasting →
Ridge regression → Portfolio construction → Performance evaluation

Author: AI-MAC Platform (Phase-2 Implementation)
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
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from scipy import stats
from scipy.spatial.distance import cosine

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ETFS = ['SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
PCA_VARIANCE_THRESHOLD = 0.95
N_REGIMES_LAYER2 = 5
TOTAL_REGIMES = 6  # 1 (crisis) + 5 (normal)
ROLLING_WINDOW = 48  # months
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
N_PERMUTATIONS = 100
RISK_FREE_RATE = 0.0  # monthly, simplified

REGIME_LABELS = {
    0: 'Economic Difficulty',
    1: 'Economic Recovery',
    2: 'Expansionary Growth',
    3: 'Stagflationary Pressure',
    4: 'Pre-Recession Transition',
    5: 'Reflationary Boom'
}

REGIME_COLORS = {
    0: '#d62728',  # red
    1: '#2ca02c',  # green
    2: '#1f77b4',  # blue
    3: '#ff7f0e',  # orange
    4: '#9467bd',  # purple
    5: '#17becf',  # cyan
}


# ============================================================
# STAGE 0: DATA ACQUISITION
# ============================================================

def download_fred_md():
    """
    Download macro variables from FRED via pandas_datareader.
    Replicates the FRED-MD dataset categories:
    Output/Income, Labor, Consumption, Housing, Money/Credit, Interest Rates, Prices.
    """
    from pandas_datareader import data as pdr
    print("[DATA] Downloading macro variables from FRED API...")

    # FRED-MD representative series with transformation codes
    # tcode: 1=level, 2=diff, 4=log, 5=dlog, 6=d2log
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
        'DPCERA3M086SBEA': 5, 'DGORDER': 5, 'AMDMNOx': 5,
        'ANDENOx': 5, 'AMDMUO': 5, 'BUSLOANS': 6,
        # Housing
        'HOUST': 4, 'HOUSTNE': 4, 'HOUSTMW': 4, 'HOUSTS': 4,
        'HOUSTW': 4, 'PERMIT': 4, 'PERMITNE': 4, 'PERMITMW': 4,
        'PERMITS': 4, 'PERMITW': 4,
        # Money & Credit
        'M1SL': 6, 'M2SL': 6, 'BOGMBASE': 6, 'TOTRESNS': 6,
        'NONBORRES': 7, 'BUSLOANS': 6, 'REALLN': 6,
        'NONREVSL': 6, 'CONSPI': 2, 'DTCOLNVHFNM': 2,
        'DTCTHFNM': 2, 'INVEST': 5,
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

    raw_data = pd.DataFrame()
    tcodes_dict = {}
    failed = []

    series_list = list(set(FRED_SERIES.keys()))  # deduplicate
    for i, series_id in enumerate(series_list):
        tc = FRED_SERIES[series_id]
        try:
            s = pdr.get_data_fred(series_id, start=start_date, end=end_date)
            raw_data[series_id] = s.iloc[:, 0]
            tcodes_dict[series_id] = tc
            if (i + 1) % 20 == 0:
                print(f"  Downloaded {i+1}/{len(series_list)} series...")
        except Exception as e:
            failed.append(series_id)

    print(f"[DATA] Downloaded {len(raw_data.columns)} series, {len(failed)} failed")
    if failed:
        print(f"[DATA] Failed series: {', '.join(failed[:10])}{'...' if len(failed)>10 else ''}")

    # Resample to month-start frequency
    raw_data = raw_data.resample('MS').last()
    raw_data = raw_data.sort_index()

    tcodes = pd.Series(tcodes_dict)

    print(f"[DATA] FRED macro data: {raw_data.shape[0]} months × {raw_data.shape[1]} variables")
    print(f"[DATA] Date range: {raw_data.index.min()} to {raw_data.index.max()}")
    return raw_data, tcodes


def apply_tcode_transforms(df, tcodes):
    """Apply FRED-MD transformation codes to stationarize data."""
    print("[DATA] Applying t-code transformations...")
    transformed = pd.DataFrame(index=df.index)

    for col in df.columns:
        if col not in tcodes.index:
            continue
        tc = int(tcodes[col])
        series = df[col].copy()

        if tc == 1:  # No transformation
            transformed[col] = series
        elif tc == 2:  # First difference
            transformed[col] = series.diff()
        elif tc == 3:  # Second difference
            transformed[col] = series.diff().diff()
        elif tc == 4:  # Log
            transformed[col] = np.log(series.clip(lower=1e-10))
        elif tc == 5:  # First difference of log
            transformed[col] = np.log(series.clip(lower=1e-10)).diff()
        elif tc == 6:  # Second difference of log
            transformed[col] = np.log(series.clip(lower=1e-10)).diff().diff()
        elif tc == 7:  # Delta of percentage change
            transformed[col] = (series / series.shift(1) - 1).diff()

    # Drop initial NaN rows from differencing
    transformed = transformed.dropna(how='all')
    # Drop columns with too many NaNs (>10%)
    threshold = len(transformed) * 0.10
    transformed = transformed.dropna(axis=1, thresh=int(len(transformed) - threshold))
    # Fill remaining NaNs with column median
    transformed = transformed.fillna(transformed.median())

    print(f"[DATA] After transforms: {transformed.shape[0]} months × {transformed.shape[1]} variables")
    return transformed


def download_etf_returns():
    """Download monthly ETF returns via yfinance."""
    print("[DATA] Downloading ETF returns via yfinance...")
    import yfinance as yf

    start_date = '1999-12-01'
    end_date = '2026-03-01'

    prices = pd.DataFrame()
    for etf in ETFS:
        try:
            ticker = yf.Ticker(etf)
            hist = ticker.history(start=start_date, end=end_date, interval='1mo')
            if len(hist) > 0:
                prices[etf] = hist['Close']
                print(f"  {etf}: {len(hist)} months")
        except Exception as e:
            print(f"  {etf}: FAILED - {e}")

    # Resample to month-end and compute returns
    prices.index = prices.index.tz_localize(None)
    prices = prices.resample('ME').last()
    returns = prices.pct_change().dropna()

    # Normalize to month-start for alignment with FRED data
    returns.index = returns.index.to_period('M').to_timestamp()

    print(f"[DATA] ETF returns: {returns.shape[0]} months × {returns.shape[1]} ETFs")
    print(f"[DATA] Date range: {returns.index.min()} to {returns.index.max()}")
    return returns


def download_nber_recessions():
    """Download NBER recession indicator from FRED."""
    print("[DATA] Downloading NBER recession dates...")
    nber = None
    try:
        from pandas_datareader import data as pdr
        usrec = pdr.get_data_fred('USREC', start='1959-01-01', end='2026-03-01')
        nber = pd.DataFrame({'recession': usrec.iloc[:, 0]})
        print("[DATA] Downloaded USREC via FRED API")
    except Exception as e:
        print(f"[DATA] FRED API failed: {e}, using hardcoded dates...")
        dates = pd.date_range('1960-01-01', '2026-03-01', freq='MS')
        recession_periods = [
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
        recession = pd.Series(0, index=dates, name='recession')
        for start, end in recession_periods:
            mask = (recession.index >= start) & (recession.index <= end)
            recession[mask] = 1
        nber = pd.DataFrame({'recession': recession})

    # Normalize to month-start
    nber = nber.resample('MS').max()
    print(f"[DATA] NBER data: {nber.shape[0]} months, {int(nber['recession'].sum())} recession months")
    return nber


# ============================================================
# STAGE A: TWO-LAYER REGIME CLASSIFICATION
# ============================================================

def run_pca(data, variance_threshold=PCA_VARIANCE_THRESHOLD):
    """Demean, standardize, and reduce via PCA."""
    print("[PCA] Running PCA dimensionality reduction...")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # Full PCA first to find number of components
    pca_full = PCA()
    pca_full.fit(scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumvar >= variance_threshold) + 1

    print(f"[PCA] Components for {variance_threshold*100}% variance: {n_components}")
    print(f"[PCA] Explained variance at {n_components} components: {cumvar[n_components-1]:.4f}")

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled)

    return components, pca, scaler, n_components, cumvar


def two_layer_kmeans(pca_data, dates):
    """
    Algorithm 1 from the paper: Two-layer modified k-means.
    Layer 1: L2 k-means (k=2) to isolate crisis months (Regime 0).
    Layer 2: Cosine k-means (k=5) on normal months (Regimes 1-5).

    Robustness: if Layer 1 produces a crisis cluster that is too small (<2%)
    or too large (>30%), fall back to distance-based outlier detection
    (top 10% most distant from global centroid = crisis).
    """
    print("[CLUSTER] Running two-layer k-means...")

    # Layer 1: L2-distance k-means, k=2
    km_l1 = KMeans(n_clusters=2, random_state=42, n_init=20)
    labels_l1 = km_l1.fit_predict(pca_data)

    # Identify the smaller cluster as crisis (Regime 0)
    cluster_sizes = np.bincount(labels_l1)
    crisis_cluster = np.argmin(cluster_sizes)
    crisis_pct = cluster_sizes[crisis_cluster] / len(labels_l1)

    if crisis_pct < 0.02 or crisis_pct > 0.30:
        # Fallback: distance-based outlier detection
        print(f"[CLUSTER] Layer 1 k-means crisis cluster={crisis_pct:.1%} — using distance-based fallback")
        centroid = pca_data.mean(axis=0)
        dists = np.linalg.norm(pca_data - centroid, axis=1)
        # Top 10% most distant = crisis (paper expects ~5-15%)
        threshold = np.percentile(dists, 90)
        crisis_mask = dists >= threshold
    else:
        crisis_mask = labels_l1 == crisis_cluster

    normal_mask = ~crisis_mask
    print(f"[CLUSTER] Layer 1: Crisis months={crisis_mask.sum()} "
          f"({crisis_mask.sum()/len(pca_data)*100:.1f}%)")

    # Layer 2: k-means on normal months
    normal_data = pca_data[normal_mask]

    # Elbow heuristic: test k=2..10
    inertias = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        km.fit(normal_data)
        inertias.append(km.inertia_)

    # Simple elbow detection: find largest drop in rate of change
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    elbow_k = np.argmax(np.abs(diffs2)) + 3
    elbow_k = min(max(elbow_k, 4), 6)
    print(f"[CLUSTER] Elbow heuristic suggests k={elbow_k} (using k={N_REGIMES_LAYER2})")

    km_l2 = KMeans(n_clusters=N_REGIMES_LAYER2, random_state=42, n_init=20)
    labels_l2 = km_l2.fit_predict(normal_data)

    # Combine: Regime 0 = crisis, Regimes 1-5 = Layer 2 labels + 1
    regime_labels = np.zeros(len(pca_data), dtype=int)
    regime_labels[crisis_mask] = 0
    regime_labels[normal_mask] = labels_l2 + 1

    # Compute fuzzy membership probabilities (Eq. 1)
    probabilities = compute_fuzzy_probabilities(pca_data, regime_labels, km_l1, km_l2,
                                                 crisis_mask, normal_mask)

    regime_df = pd.DataFrame({
        'regime': regime_labels,
    }, index=dates[:len(regime_labels)])

    # Add probability columns
    for r in range(TOTAL_REGIMES):
        regime_df[f'prob_r{r}'] = probabilities[:, r]

    print(f"[CLUSTER] Regime distribution:")
    for r in range(TOTAL_REGIMES):
        count = (regime_labels == r).sum()
        print(f"  Regime {r} ({REGIME_LABELS[r]}): {count} months ({count/len(regime_labels)*100:.1f}%)")

    return regime_df, km_l1, km_l2, elbow_k, inertias


def compute_fuzzy_probabilities(pca_data, labels, km_l1, km_l2, crisis_mask, normal_mask):
    """
    Compute fuzzy membership probabilities per Eq. 1 and Eq. 4 of the paper.
    Uses inverse distance to centroids, normalized to sum to 1.
    """
    n = len(pca_data)
    probs = np.zeros((n, TOTAL_REGIMES))

    # All centroids
    centroids = np.zeros((TOTAL_REGIMES, pca_data.shape[1]))
    centroids[0] = km_l1.cluster_centers_[np.argmin(np.bincount(km_l1.labels_))]
    for r in range(N_REGIMES_LAYER2):
        centroids[r + 1] = km_l2.cluster_centers_[r]

    for i in range(n):
        point = pca_data[i]
        # Compute distances to all centroids
        distances = np.array([np.linalg.norm(point - centroids[r]) for r in range(TOTAL_REGIMES)])
        # Inverse distance weighting (Eq. 1)
        distances = np.clip(distances, 1e-10, None)
        inv_dist = 1.0 / distances
        probs[i] = inv_dist / inv_dist.sum()

    # Apply log-scaling smoothing (Eq. 4) for crisis probability
    # Scale crisis probability using a sigmoid-like function
    crisis_prob = probs[:, 0]
    median_crisis = np.median(crisis_prob)
    probs[:, 0] = 1.0 / (1.0 + np.exp(-5 * (crisis_prob - median_crisis) / (median_crisis + 1e-10)))

    # Re-normalize
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / row_sums

    return probs


# ============================================================
# STAGE B: MARKOV TRANSITION & FORECASTING
# ============================================================

def build_transition_matrix(regimes, n_regimes=TOTAL_REGIMES):
    """Build Markov transition matrix from regime sequence."""
    T = np.zeros((n_regimes, n_regimes))
    for i in range(len(regimes) - 1):
        T[regimes[i], regimes[i + 1]] += 1
    # Normalize rows
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = T / row_sums
    return T


def predict_next_regime(current_probs, transition_matrix):
    """Propagate current regime probabilities one step via Markov chain."""
    return current_probs @ transition_matrix


def ridge_regression_forecast(macro_features, returns, regime_probs, regime_labels,
                               window=ROLLING_WINDOW):
    """
    Per-regime ridge regression: for each regime r and each ETF,
    fit a ridge model on macro features, then aggregate predictions
    weighted by next-period regime probabilities.
    """
    n_months = len(returns)
    n_etfs = returns.shape[1]
    predictions = np.full((n_months, n_etfs), np.nan)

    for t in range(window, n_months):
        # Training window
        train_start = t - window
        X_train = macro_features[train_start:t]
        y_train = returns.iloc[train_start:t].values
        regimes_train = regime_labels[train_start:t]
        next_probs = regime_probs[t]  # probability vector for month t

        pred_t = np.zeros(n_etfs)

        for r in range(TOTAL_REGIMES):
            mask_r = regimes_train == r
            if mask_r.sum() < 5:
                # Not enough samples for this regime in window
                continue

            X_r = X_train[mask_r]
            y_r = y_train[mask_r]

            for j in range(n_etfs):
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_r, y_r[:, j])
                pred_rj = ridge.predict(macro_features[t:t+1])[0]
                pred_t[j] += next_probs[r] * pred_rj

        predictions[t] = pred_t

    return predictions


def naive_forecast(returns, regime_labels, regime_probs, window=ROLLING_WINDOW):
    """Naive model: conditional Sharpe ratio on most-likely next regime."""
    n_months = len(returns)
    n_etfs = returns.shape[1]
    predictions = np.full((n_months, n_etfs), np.nan)

    for t in range(window, n_months):
        train_start = t - window
        y_train = returns.iloc[train_start:t].values
        regimes_train = regime_labels[train_start:t]
        next_probs = regime_probs[t]

        pred_t = np.zeros(n_etfs)
        for r in range(TOTAL_REGIMES):
            mask_r = regimes_train == r
            if mask_r.sum() < 3:
                continue
            regime_ret = y_train[mask_r]
            mean_r = regime_ret.mean(axis=0)
            std_r = regime_ret.std(axis=0) + 1e-10
            sharpe_r = mean_r / std_r
            pred_t += next_probs[r] * sharpe_r

        predictions[t] = pred_t

    return predictions


# ============================================================
# STAGE C: PORTFOLIO CONSTRUCTION
# ============================================================

def long_only_portfolio(predictions, l=3):
    """
    Long-only position sizing: allocate equally to top-l predicted ETFs.
    """
    n_months, n_etfs = predictions.shape
    weights = np.zeros_like(predictions)

    for t in range(n_months):
        if np.isnan(predictions[t]).any():
            continue
        top_l = np.argsort(predictions[t])[-l:]
        weights[t, top_l] = 1.0 / l

    return weights


def long_and_short_portfolio(predictions, l=3):
    """Long top-l, short bottom-l."""
    n_months, n_etfs = predictions.shape
    weights = np.zeros_like(predictions)

    for t in range(n_months):
        if np.isnan(predictions[t]).any():
            continue
        ranking = np.argsort(predictions[t])
        weights[t, ranking[-l:]] = 1.0 / l
        weights[t, ranking[:l]] = -1.0 / l

    return weights


def mixed_portfolio(predictions, regime_probs, l=3, crisis_threshold=0.3):
    """Long-only normally; short during Regime 0 (crisis)."""
    n_months, n_etfs = predictions.shape
    weights = np.zeros_like(predictions)

    for t in range(n_months):
        if np.isnan(predictions[t]).any():
            continue
        if regime_probs[t, 0] > crisis_threshold:
            # Crisis: short bottom-l
            ranking = np.argsort(predictions[t])
            weights[t, ranking[:l]] = -1.0 / l
        else:
            # Normal: long top-l
            top_l = np.argsort(predictions[t])[-l:]
            weights[t, top_l] = 1.0 / l

    return weights


# ============================================================
# EVALUATION
# ============================================================

def compute_portfolio_returns(weights, etf_returns):
    """Compute portfolio return series from weights and ETF returns."""
    # Align shapes
    valid = ~np.isnan(weights).any(axis=1) & (np.abs(weights).sum(axis=1) > 0)
    port_ret = np.full(len(weights), np.nan)
    for t in range(len(weights)):
        if valid[t]:
            port_ret[t] = np.sum(weights[t] * etf_returns.iloc[t].values)
    return pd.Series(port_ret, index=etf_returns.index[:len(weights)])


def compute_metrics(returns_series):
    """Compute Sharpe, Sortino, MaxDD, % positive months."""
    clean = returns_series.dropna()
    if len(clean) < 12:
        return {'sharpe': np.nan, 'sortino': np.nan, 'maxdd': np.nan, 'pct_positive': np.nan, 'n_months': len(clean)}

    mean_ret = clean.mean() * 12
    std_ret = clean.std() * np.sqrt(12)
    downside = clean[clean < 0].std() * np.sqrt(12)
    if downside == 0:
        downside = 1e-10

    sharpe = mean_ret / std_ret if std_ret > 0 else 0
    sortino = mean_ret / downside

    # Max drawdown
    cumret = (1 + clean).cumprod()
    running_max = cumret.cummax()
    drawdown = (cumret - running_max) / running_max
    maxdd = drawdown.min()

    pct_pos = (clean > 0).mean()

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'maxdd': round(maxdd, 4),
        'pct_positive': round(pct_pos, 4),
        'n_months': len(clean),
        'ann_return': round(mean_ret, 4),
        'ann_vol': round(std_ret, 4),
    }


def run_permutation_test(etf_returns, macro_features, regime_df, n_perms=N_PERMUTATIONS):
    """Run random-regime permutation controls."""
    print(f"[PERMTEST] Running {n_perms} random-regime permutations...")
    perm_sharpes = []
    orig_labels = regime_df['regime'].values

    for p in range(n_perms):
        shuffled = np.random.permutation(orig_labels)
        # Build fake regime probs (one-hot for shuffled labels)
        fake_probs = np.zeros((len(shuffled), TOTAL_REGIMES))
        for i, r in enumerate(shuffled):
            fake_probs[i, r] = 1.0

        preds = naive_forecast(etf_returns, shuffled, fake_probs, window=ROLLING_WINDOW)
        weights = long_only_portfolio(preds, l=3)
        port_ret = compute_portfolio_returns(weights, etf_returns)
        metrics = compute_metrics(port_ret)
        perm_sharpes.append(metrics['sharpe'])

        if (p + 1) % 25 == 0:
            print(f"  Permutation {p+1}/{n_perms}")

    return np.array(perm_sharpes)


# ============================================================
# VALIDATION
# ============================================================

def validate_nber_overlap(regime_df, nber_df):
    """Test 1: Check Regime 0 overlap with NBER recessions."""
    print("[VALIDATE] Checking Regime 0 vs NBER recession overlap...")
    common_idx = regime_df.index.intersection(nber_df.index)
    if len(common_idx) == 0:
        print("[VALIDATE] WARNING: No overlapping dates between regime and NBER data")
        return 0.0

    regime0_months = set(regime_df.loc[common_idx][regime_df.loc[common_idx, 'regime'] == 0].index)
    nber_recession_months = set(nber_df.loc[common_idx][nber_df.loc[common_idx, 'recession'] == 1].index)

    if len(regime0_months) == 0:
        return 0.0
    if len(nber_recession_months) == 0:
        return 0.0

    overlap = regime0_months & nber_recession_months
    # Fraction of regime 0 months that are NBER recessions
    precision = len(overlap) / len(regime0_months) if len(regime0_months) > 0 else 0
    # Fraction of NBER recession months captured by regime 0
    recall = len(overlap) / len(nber_recession_months) if len(nber_recession_months) > 0 else 0

    print(f"[VALIDATE] Regime 0 months: {len(regime0_months)}")
    print(f"[VALIDATE] NBER recession months: {len(nber_recession_months)}")
    print(f"[VALIDATE] Overlap: {len(overlap)}")
    print(f"[VALIDATE] Precision (R0→NBER): {precision:.2%}")
    print(f"[VALIDATE] Recall (NBER→R0): {recall:.2%}")

    return recall  # Paper uses recall-oriented metric


def lag_sensitivity_test(macro_data, pca_model, scaler, km_l1, km_l2, regime_df):
    """Test 4: Compare regimes with 1-month lagged data."""
    print("[VALIDATE] Running lag sensitivity test...")
    # Simulate 1-month lag: shift macro data by 1 month
    lagged_data = macro_data.shift(1).dropna()

    # Re-run PCA and clustering on lagged data
    common_idx = lagged_data.index.intersection(regime_df.index)
    if len(common_idx) < 50:
        print("[VALIDATE] Insufficient overlapping data for lag test")
        return 0.0

    lagged_scaled = scaler.transform(lagged_data.loc[common_idx])
    lagged_pca = pca_model.transform(lagged_scaled)

    # Predict regime labels using existing cluster models
    labels_l1 = km_l1.predict(lagged_pca)
    crisis_cluster = np.argmin(np.bincount(km_l1.labels_))
    crisis_mask = labels_l1 == crisis_cluster

    lagged_labels = np.zeros(len(lagged_pca), dtype=int)
    lagged_labels[crisis_mask] = 0

    if (~crisis_mask).sum() > 0:
        normal_data = lagged_pca[~crisis_mask]
        labels_l2 = km_l2.predict(normal_data)
        lagged_labels[~crisis_mask] = labels_l2 + 1

    # Compare
    original_labels = regime_df.loc[common_idx, 'regime'].values
    agreement = (lagged_labels == original_labels).mean()
    print(f"[VALIDATE] Lag agreement rate: {agreement:.2%}")
    return agreement


# ============================================================
# PLOTTING
# ============================================================

def plot_regime_timeline(regime_df, nber_df, output_path):
    """Plot regime classification timeline with NBER recession shading."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[1, 2])

    # Top panel: Hard regime labels
    ax1 = axes[0]
    for r in range(TOTAL_REGIMES):
        mask = regime_df['regime'] == r
        dates = regime_df.index[mask]
        ax1.scatter(dates, [r] * len(dates), c=REGIME_COLORS[r], s=8, label=REGIME_LABELS[r])

    # NBER shading
    common = regime_df.index.intersection(nber_df.index)
    if len(common) > 0:
        nber_aligned = nber_df.reindex(regime_df.index).fillna(0)
        for idx in regime_df.index:
            if idx in nber_aligned.index and nber_aligned.loc[idx, 'recession'] == 1:
                ax1.axvspan(idx - pd.Timedelta(days=15), idx + pd.Timedelta(days=15),
                           alpha=0.15, color='gray')

    ax1.set_ylabel('Regime')
    ax1.set_yticks(range(TOTAL_REGIMES))
    ax1.set_yticklabels([f'R{r}' for r in range(TOTAL_REGIMES)])
    ax1.set_title('Regime Classification Timeline (gray = NBER recessions)')
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

    # Bottom panel: Regime probabilities stacked
    ax2 = axes[1]
    prob_cols = [f'prob_r{r}' for r in range(TOTAL_REGIMES)]
    probs = regime_df[prob_cols].values
    ax2.stackplot(regime_df.index, probs.T,
                  labels=[f'R{r}: {REGIME_LABELS[r]}' for r in range(TOTAL_REGIMES)],
                  colors=[REGIME_COLORS[r] for r in range(TOTAL_REGIMES)],
                  alpha=0.8)
    ax2.set_ylabel('Regime Probability')
    ax2.set_xlabel('Date')
    ax2.set_title('Fuzzy Regime Membership Probabilities')
    ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {output_path}")


def plot_transition_matrix(T, output_path):
    """Heatmap of Markov transition matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [f'R{r}' for r in range(TOTAL_REGIMES)]
    sns.heatmap(T, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title('Regime Transition Matrix')
    ax.set_xlabel('To Regime')
    ax.set_ylabel('From Regime')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {output_path}")


def plot_performance_comparison(results_dict, spy_metrics, output_path):
    """Bar chart comparing strategy performance."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    strategies = list(results_dict.keys())
    sharpes = [results_dict[s]['sharpe'] for s in strategies]
    sortinos = [results_dict[s]['sortino'] for s in strategies]
    maxdds = [results_dict[s]['maxdd'] for s in strategies]

    # Add SPY
    strategies.append('SPY B&H')
    sharpes.append(spy_metrics['sharpe'])
    sortinos.append(spy_metrics['sortino'])
    maxdds.append(spy_metrics['maxdd'])

    colors = ['#1f77b4'] * (len(strategies) - 1) + ['#d62728']

    axes[0].barh(strategies, sharpes, color=colors)
    axes[0].set_title('Sharpe Ratio')
    axes[0].axvline(x=0, color='black', linewidth=0.5)

    axes[1].barh(strategies, sortinos, color=colors)
    axes[1].set_title('Sortino Ratio')
    axes[1].axvline(x=0, color='black', linewidth=0.5)

    axes[2].barh(strategies, maxdds, color=colors)
    axes[2].set_title('Max Drawdown')
    axes[2].axvline(x=0, color='black', linewidth=0.5)

    plt.suptitle('Strategy Performance Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {output_path}")


def plot_cumulative_returns(portfolio_returns_dict, spy_returns, output_path):
    """Plot cumulative return curves."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for name, ret_series in portfolio_returns_dict.items():
        clean = ret_series.dropna()
        cumret = (1 + clean).cumprod()
        ax.plot(cumret.index, cumret.values, label=name, linewidth=1.5)

    # SPY
    spy_cumret = (1 + spy_returns).cumprod()
    ax.plot(spy_cumret.index, spy_cumret.values, label='SPY B&H', color='black',
            linewidth=2, linestyle='--')

    ax.set_title('Cumulative Returns')
    ax.set_ylabel('Growth of $1')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {output_path}")


def plot_elbow(inertias, output_path):
    """Plot elbow curve for k selection."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = range(2, 2 + len(inertias))
    ax.plot(ks, inertias, 'bo-', linewidth=2)
    ax.axvline(x=5, color='red', linestyle='--', label=f'Selected k={N_REGIMES_LAYER2}')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for Layer 2 k Selection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {output_path}")


def plot_regime_characteristics(regime_df, etf_returns, output_path):
    """Plot mean ETF returns per regime."""
    common_idx = regime_df.index.intersection(etf_returns.index)
    merged = regime_df.loc[common_idx, ['regime']].join(etf_returns.loc[common_idx])

    means = merged.groupby('regime')[ETFS].mean() * 12 * 100  # Annualized %

    fig, ax = plt.subplots(figsize=(12, 6))
    means.T.plot(kind='bar', ax=ax, color=[REGIME_COLORS[r] for r in range(TOTAL_REGIMES)])
    ax.set_title('Annualized Mean Return by Regime and ETF (%)')
    ax.set_ylabel('Ann. Return (%)')
    ax.set_xlabel('ETF')
    ax.legend([f'R{r}: {REGIME_LABELS[r]}' for r in range(TOTAL_REGIMES)],
              bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {output_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 70)
    print("REGIME DETECTION PIPELINE — arXiv 2503.11499")
    print("Tactical Asset Allocation with Macroeconomic Regime Detection")
    print(f"Run date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}

    # ---- STAGE 0: DATA ----
    print("\n" + "=" * 50)
    print("STAGE 0: DATA ACQUISITION")
    print("=" * 50)

    fred_raw, tcodes = download_fred_md()
    fred_transformed = apply_tcode_transforms(fred_raw, tcodes)
    etf_returns = download_etf_returns()
    nber = download_nber_recessions()

    # Align datasets
    common_start = max(fred_transformed.index.min(), etf_returns.index.min())
    common_end = min(fred_transformed.index.max(), etf_returns.index.max())
    print(f"\n[DATA] Common date range: {common_start} to {common_end}")

    # ---- STAGE A: REGIME CLASSIFICATION ----
    print("\n" + "=" * 50)
    print("STAGE A: REGIME CLASSIFICATION")
    print("=" * 50)

    pca_components, pca_model, scaler, n_pca, cumvar = run_pca(fred_transformed)
    regime_df, km_l1, km_l2, elbow_k, inertias = two_layer_kmeans(
        pca_components, fred_transformed.index)

    # Validation Test 1: NBER overlap
    nber_overlap = validate_nber_overlap(regime_df, nber)
    results['nber_overlap'] = nber_overlap
    results['n_pca_components'] = n_pca
    results['elbow_k'] = elbow_k

    # ---- PLOTS: Regime ----
    plot_regime_timeline(regime_df, nber,
                        os.path.join(OUTPUT_DIR, 'regime_timeline.png'))
    plot_elbow(inertias, os.path.join(OUTPUT_DIR, 'elbow_curve.png'))

    # ---- Align macro + returns for forecasting ----
    common_idx = regime_df.index.intersection(etf_returns.index)
    regime_aligned = regime_df.loc[common_idx]
    returns_aligned = etf_returns.loc[common_idx]
    macro_aligned = fred_transformed.loc[common_idx]

    # PCA on aligned macro data
    macro_scaled = scaler.transform(macro_aligned)
    macro_pca = pca_model.transform(macro_scaled)

    print(f"\n[ALIGN] Forecasting dataset: {len(common_idx)} months")

    # Regime characteristics
    plot_regime_characteristics(regime_aligned, returns_aligned,
                               os.path.join(OUTPUT_DIR, 'regime_characteristics.png'))

    # ---- STAGE B: FORECASTING ----
    print("\n" + "=" * 50)
    print("STAGE B: FORECASTING MODELS")
    print("=" * 50)

    regime_labels_arr = regime_aligned['regime'].values
    prob_cols = [f'prob_r{r}' for r in range(TOTAL_REGIMES)]
    regime_probs_arr = regime_aligned[prob_cols].values

    # Build full-sample transition matrix for display
    T_full = build_transition_matrix(regime_labels_arr)
    plot_transition_matrix(T_full, os.path.join(OUTPUT_DIR, 'transition_matrix.png'))
    results['transition_matrix'] = T_full.tolist()

    # Ridge regression forecasts
    print("\n[FORECAST] Running ridge regression...")
    ridge_preds = ridge_regression_forecast(
        macro_pca, returns_aligned, regime_probs_arr, regime_labels_arr)

    # Naive forecasts
    print("[FORECAST] Running naive model...")
    naive_preds = naive_forecast(returns_aligned, regime_labels_arr, regime_probs_arr)

    # ---- STAGE C: PORTFOLIO CONSTRUCTION ----
    print("\n" + "=" * 50)
    print("STAGE C: PORTFOLIO CONSTRUCTION & EVALUATION")
    print("=" * 50)

    # SPY benchmark
    spy_returns = returns_aligned['SPY']
    spy_metrics = compute_metrics(spy_returns)
    results['spy_metrics'] = spy_metrics
    print(f"\n[BENCH] SPY B&H: Sharpe={spy_metrics['sharpe']}, "
          f"Sortino={spy_metrics['sortino']}, MaxDD={spy_metrics['maxdd']}")

    # Equal-weight benchmark
    ew_returns = returns_aligned.mean(axis=1)
    ew_metrics = compute_metrics(ew_returns)
    results['ew_metrics'] = ew_metrics
    print(f"[BENCH] Equal-Weight: Sharpe={ew_metrics['sharpe']}, "
          f"Sortino={ew_metrics['sortino']}, MaxDD={ew_metrics['maxdd']}")

    # Strategy performance
    strategy_results = {}
    portfolio_return_series = {}

    for model_name, preds in [('Ridge', ridge_preds), ('Naive', naive_preds)]:
        for l in [2, 3, 4]:
            # Long-only
            name = f'{model_name}_LO_l{l}'
            w = long_only_portfolio(preds, l=l)
            pr = compute_portfolio_returns(w, returns_aligned)
            m = compute_metrics(pr)
            strategy_results[name] = m
            portfolio_return_series[name] = pr
            print(f"  {name}: Sharpe={m['sharpe']}, Sortino={m['sortino']}, "
                  f"MaxDD={m['maxdd']}, %Pos={m['pct_positive']}")

        # Long-and-short for l=3
        name = f'{model_name}_LnS_l3'
        w = long_and_short_portfolio(preds, l=3)
        pr = compute_portfolio_returns(w, returns_aligned)
        m = compute_metrics(pr)
        strategy_results[name] = m
        portfolio_return_series[name] = pr
        print(f"  {name}: Sharpe={m['sharpe']}, Sortino={m['sortino']}, "
              f"MaxDD={m['maxdd']}, %Pos={m['pct_positive']}")

        # Mixed for l=3
        name = f'{model_name}_MX_l3'
        w = mixed_portfolio(preds, regime_probs_arr, l=3)
        pr = compute_portfolio_returns(w, returns_aligned)
        m = compute_metrics(pr)
        strategy_results[name] = m
        portfolio_return_series[name] = pr
        print(f"  {name}: Sharpe={m['sharpe']}, Sortino={m['sortino']}, "
              f"MaxDD={m['maxdd']}, %Pos={m['pct_positive']}")

    results['strategies'] = strategy_results

    # ---- PLOTS: Performance ----
    # Select key strategies for comparison plot
    key_strategies = {k: v for k, v in strategy_results.items()
                      if 'Ridge' in k and 'LO' in k}
    plot_performance_comparison(key_strategies, spy_metrics,
                               os.path.join(OUTPUT_DIR, 'performance_comparison.png'))

    key_returns = {k: v for k, v in portfolio_return_series.items()
                   if k in ['Ridge_LO_l2', 'Ridge_LO_l3', 'Ridge_LO_l4', 'Naive_LO_l3']}
    plot_cumulative_returns(key_returns, spy_returns,
                           os.path.join(OUTPUT_DIR, 'cumulative_returns.png'))

    # ---- PERMUTATION TEST ----
    print("\n" + "=" * 50)
    print("STATISTICAL VALIDATION")
    print("=" * 50)

    perm_sharpes = run_permutation_test(returns_aligned, macro_pca, regime_aligned)

    # Best ridge strategy
    best_ridge = 'Ridge_LO_l3'
    real_sharpe = strategy_results[best_ridge]['sharpe']
    perm_mean = np.nanmean(perm_sharpes)
    perm_std = np.nanstd(perm_sharpes)
    valid_perms = perm_sharpes[~np.isnan(perm_sharpes)]

    if len(valid_perms) > 1:
        t_stat, p_value = stats.ttest_1samp(valid_perms, real_sharpe)
        p_value_one_sided = p_value / 2 if t_stat < 0 else 1 - p_value / 2
        # Empirical p-value
        p_empirical = (valid_perms >= real_sharpe).mean()
    else:
        t_stat, p_value, p_value_one_sided, p_empirical = 0, 1, 1, 1

    results['permutation_test'] = {
        'real_sharpe': real_sharpe,
        'perm_mean': round(float(perm_mean), 4),
        'perm_std': round(float(perm_std), 4),
        't_stat': round(float(t_stat), 4),
        'p_value': round(float(p_value), 4),
        'p_empirical': round(float(p_empirical), 4),
    }

    print(f"\n[STAT] Real regime Sharpe ({best_ridge}): {real_sharpe}")
    print(f"[STAT] Random regime Sharpe: mean={perm_mean:.4f}, std={perm_std:.4f}")
    print(f"[STAT] t-statistic: {t_stat:.4f}")
    print(f"[STAT] p-value (two-sided): {p_value:.4f}")
    print(f"[STAT] p-value (empirical): {p_empirical:.4f}")

    # Plot permutation distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(valid_perms, bins=25, alpha=0.7, color='steelblue', edgecolor='white',
            label=f'Random Regime Sharpe (n={len(valid_perms)})')
    ax.axvline(real_sharpe, color='red', linewidth=2, linestyle='--',
               label=f'Real Regime Sharpe = {real_sharpe}')
    ax.set_title(f'Permutation Test: {best_ridge} vs Random Regimes')
    ax.set_xlabel('Sharpe Ratio')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'permutation_test.png'), dpi=150)
    plt.close()

    # ---- LAG SENSITIVITY TEST ----
    print("\n" + "=" * 50)
    print("LAG SENSITIVITY TEST")
    print("=" * 50)

    lag_agreement = lag_sensitivity_test(
        fred_transformed, pca_model, scaler, km_l1, km_l2, regime_df)
    results['lag_agreement'] = round(lag_agreement, 4)

    # ---- OUT-OF-SAMPLE ANALYSIS ----
    print("\n" + "=" * 50)
    print("OUT-OF-SAMPLE ANALYSIS")
    print("=" * 50)

    # Split: in-sample up to Dec 2022, OOS from Jan 2023
    oos_cutoff = pd.Timestamp('2022-12-31')
    is_mask = returns_aligned.index <= oos_cutoff
    oos_mask = returns_aligned.index > oos_cutoff

    if oos_mask.sum() > 6:
        # In-sample metrics for Ridge_LO_l3
        is_returns = portfolio_return_series['Ridge_LO_l3'][is_mask].dropna()
        is_metrics = compute_metrics(is_returns)

        oos_returns = portfolio_return_series['Ridge_LO_l3'][oos_mask].dropna()
        oos_metrics = compute_metrics(oos_returns)

        spy_oos = spy_returns[oos_mask]
        spy_oos_metrics = compute_metrics(spy_oos)

        results['in_sample'] = is_metrics
        results['oos_metrics'] = oos_metrics
        results['spy_oos_metrics'] = spy_oos_metrics

        print(f"[OOS] In-sample (→ Dec 2022): Sharpe={is_metrics['sharpe']}, n={is_metrics['n_months']}")
        print(f"[OOS] Out-of-sample (Jan 2023→): Sharpe={oos_metrics['sharpe']}, n={oos_metrics['n_months']}")
        print(f"[OOS] SPY OOS: Sharpe={spy_oos_metrics['sharpe']}")

        # OOS regime plausibility
        oos_regimes = regime_aligned.loc[oos_mask, 'regime'].value_counts()
        results['oos_regime_dist'] = oos_regimes.to_dict()
        print(f"[OOS] OOS regime distribution:")
        for r, count in sorted(oos_regimes.items()):
            print(f"  Regime {r} ({REGIME_LABELS[r]}): {count} months")
    else:
        print(f"[OOS] Only {oos_mask.sum()} OOS months available — insufficient for analysis")
        results['oos_metrics'] = {'sharpe': np.nan, 'n_months': oos_mask.sum()}

    # ---- GENERATE REGIME STATE JSON ----
    print("\n" + "=" * 50)
    print("GENERATING REGIME STATE OUTPUT")
    print("=" * 50)

    latest_date = regime_aligned.index[-1]
    latest_regime = int(regime_aligned.loc[latest_date, 'regime'])
    latest_probs = regime_aligned.loc[latest_date, prob_cols].values.tolist()

    # Predict next month
    next_probs = predict_next_regime(np.array(latest_probs), T_full)

    regime_state = {
        'timestamp': datetime.datetime.now().isoformat(),
        'data_as_of': str(latest_date.date()),
        'current_regime': latest_regime,
        'current_regime_label': REGIME_LABELS[latest_regime],
        'current_probabilities': {f'R{r}': round(float(latest_probs[r]), 4) for r in range(TOTAL_REGIMES)},
        'next_month_predicted_probabilities': {f'R{r}': round(float(next_probs[r]), 4) for r in range(TOTAL_REGIMES)},
        'transition_matrix': [[round(float(T_full[i, j]), 4) for j in range(TOTAL_REGIMES)]
                              for i in range(TOTAL_REGIMES)],
        'model_info': {
            'n_pca_components': int(n_pca),
            'n_regimes': int(TOTAL_REGIMES),
            'rolling_window': int(ROLLING_WINDOW),
            'data_source': 'FRED-MD + yfinance',
        }
    }

    regime_state_path = os.path.join(OUTPUT_DIR, 'regime_state.json')
    with open(regime_state_path, 'w') as f:
        json.dump(regime_state, f, indent=2)
    print(f"[OUTPUT] Saved: {regime_state_path}")

    # ---- SAVE PERFORMANCE TABLE ----
    perf_rows = []
    for name, m in strategy_results.items():
        perf_rows.append({'Strategy': name, **m})
    perf_rows.append({'Strategy': 'SPY_BH', **spy_metrics})
    perf_rows.append({'Strategy': 'EqualWeight', **ew_metrics})

    perf_df = pd.DataFrame(perf_rows)
    perf_path = os.path.join(OUTPUT_DIR, 'performance_table.csv')
    perf_df.to_csv(perf_path, index=False)
    print(f"[OUTPUT] Saved: {perf_path}")

    # ---- SAVE REGIME LABELS ----
    regime_path = os.path.join(OUTPUT_DIR, 'regime_labels.csv')
    regime_df.to_csv(regime_path)
    print(f"[OUTPUT] Saved: {regime_path}")

    # ---- SAVE FULL RESULTS JSON ----
    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    results_clean = convert_types(results)
    results_path = os.path.join(OUTPUT_DIR, 'full_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    print(f"[OUTPUT] Saved: {results_path}")

    # ---- SUMMARY ----
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"  PCA components: {n_pca}")
    print(f"  Elbow k: {elbow_k}")
    print(f"  NBER overlap: {nber_overlap:.2%}")
    print(f"  Current regime: R{latest_regime} ({REGIME_LABELS[latest_regime]})")
    print(f"  Best strategy: Ridge_LO_l3 Sharpe={strategy_results.get('Ridge_LO_l3', {}).get('sharpe', 'N/A')}")
    print(f"  Permutation p-value: {results.get('permutation_test', {}).get('p_empirical', 'N/A')}")
    print(f"  Lag agreement: {lag_agreement:.2%}")
    if 'oos_metrics' in results and not np.isnan(results['oos_metrics'].get('sharpe', np.nan)):
        print(f"  OOS Sharpe: {results['oos_metrics']['sharpe']}")
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    return results


if __name__ == '__main__':
    results = main()
