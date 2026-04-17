#!/usr/bin/env python3
"""
Gap-Closing Script — arXiv 2503.11499
Addresses remaining implementation quality gaps:
1. Maximize FRED data coverage toward paper's 127 variables
2. Stress-period validation (crisis windows)
3. Lag-sensitivity stabilization attempts
4. Generate final_status_2503.md

Author: AI-MAC Platform
Date: 2026-03-13
"""

import os
import sys
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
from scipy import stats
from scipy.spatial.distance import cdist

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ETFS = ['SPY', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
PCA_VARIANCE_THRESHOLD = 0.95
N_REGIMES_LAYER2 = 5
TOTAL_REGIMES = 6
ROLLING_WINDOW = 48
REGIME_LABELS = {
    0: 'Economic Difficulty', 1: 'Economic Recovery',
    2: 'Expansionary Growth', 3: 'Stagflationary Pressure',
    4: 'Pre-Recession Transition', 5: 'Reflationary Boom'
}
REGIME_COLORS = {
    0: '#d62728', 1: '#2ca02c', 2: '#1f77b4',
    3: '#ff7f0e', 4: '#9467bd', 5: '#17becf',
}

# ============================================================
# FULL FRED-MD 127 VARIABLE MAPPING
# Paper spec: 127 monthly macro series across 8 categories
# Each maps FRED-MD name -> (FRED API series ID, tcode)
# tcode: 1=level, 2=diff, 4=log, 5=dlog, 6=d2log, 7=dpct
# ============================================================

FRED_MD_FULL = {
    # ---- Group 1: Output and Income (17 series) ----
    'RPI': ('RPI', 5),
    'W875RX1': ('W875RX1', 5),
    'INDPRO': ('INDPRO', 5),
    'IPFPNSS': ('IPFPNSS', 5),
    'IPFINAL': ('IPFINAL', 5),
    'IPCONGD': ('IPCONGD', 5),
    'IPDCONGD': ('IPDCONGD', 5),
    'IPNCONGD': ('IPNCONGD', 5),
    'IPBUSEQ': ('IPBUSEQ', 5),
    'IPMAT': ('IPMAT', 5),
    'IPDMAT': ('IPDMAT', 5),
    'IPNMAT': ('IPNMAT', 5),
    'IPMANSICS': ('IPMANSICS', 5),
    'IPB51222S': ('IPB51222S', 5),
    'IPFUELS': ('IPFUELS', 5),
    'CUMFNS': ('CUMFNS', 2),
    'HWI': ('HELPWANT', 2),  # Help-Wanted Index (discontinued, fallback)

    # ---- Group 2: Labor Market (32 series) ----
    'HWIURATIO': ('HWIURATIO', 2),  # fallback available
    'CLF16OV': ('CLF16OV', 5),
    'CE16OV': ('CE16OV', 5),
    'UNRATE': ('UNRATE', 2),
    'UEMPMEAN': ('UEMPMEAN', 2),
    'UEMPLT5': ('UEMPLT5', 5),
    'UEMP5TO14': ('UEMP5TO14', 5),
    'UEMP15OV': ('UEMP15OV', 5),
    'UEMP15T26': ('UEMP15T26', 5),
    'UEMP27OV': ('UEMP27OV', 5),
    'CLAIMS': ('ICSA', 5),  # Initial claims
    'PAYEMS': ('PAYEMS', 5),
    'USGOOD': ('USGOOD', 5),
    'CES1021000001': ('CES1021000001', 5),
    'USCONS': ('USCONS', 5),
    'MANEMP': ('MANEMP', 5),
    'DMANEMP': ('DMANEMP', 5),
    'NDMANEMP': ('NDMANEMP', 5),
    'SRVPRD': ('SRVPRD', 5),
    'USTPU': ('USTPU', 5),
    'USWTRADE': ('USWTRADE', 5),
    'USTRADE': ('USTRADE', 5),
    'USFIRE': ('USFIRE', 5),
    'USGOVT': ('USGOVT', 5),
    'CES0600000007': ('CES0600000007', 1),
    'CES0600000008': ('CES0600000008', 5),
    'CES2000000008': ('CES2000000008', 5),
    'CES3000000008': ('CES3000000008', 5),
    'CIVPART': ('CIVPART', 2),
    'AWOTMAN': ('AWOTMAN', 2),  # Avg weekly overtime hours, manufacturing
    'AWHMAN': ('AWHMAN', 1),   # Avg weekly hours, manufacturing

    # ---- Group 3: Housing (10 series) ----
    'HOUST': ('HOUST', 4),
    'HOUSTNE': ('HOUSTNE', 4),
    'HOUSTMW': ('HOUSTMW', 4),
    'HOUSTS': ('HOUSTS', 4),
    'HOUSTW': ('HOUSTW', 4),
    'PERMIT': ('PERMIT', 4),
    'PERMITNE': ('PERMITNE', 4),
    'PERMITMW': ('PERMITMW', 4),
    'PERMITS': ('PERMITS', 4),
    'PERMITW': ('PERMITW', 4),

    # ---- Group 4: Consumption, Orders, Inventories (11 series) ----
    'DPCERA3M086SBEA': ('DPCERA3M086SBEA', 5),
    'CMRMTSPLx': ('CMRMTSPL', 5),   # Real M&T sales
    'RETAILx': ('RETAILMPCSMSA', 5), # Retail sales
    'ACOGNO': ('ACOGNO', 5),  # New orders consumer goods
    'AMDMNOx': ('AMDMNO', 5), # New orders durable mfg
    'ANDENOx': ('ANDENO', 5), # New orders nondefense capital goods
    'AMDMUOx': ('AMDMUO', 5), # Unfilled orders durable mfg
    'BUSINVx': ('BUSINV', 5), # Total business inventories
    'ISRATIOx': ('ISRATIO', 2), # Inventory/sales ratio
    'UMCSENTx': ('UMCSENT', 2), # U Michigan consumer sentiment

    # ---- Group 5: Money and Credit (14 series) ----
    'M1SL': ('M1SL', 6),
    'M2SL': ('M2SL', 6),
    'M2REAL': ('M2REAL', 5),  # Real M2
    'AMBSL': ('AMBSL', 6),    # St. Louis Adjusted Monetary Base
    'TOTRESNS': ('TOTRESNS', 6),
    'NONBORRES': ('NONBORRES', 7),
    'BUSLOANS': ('BUSLOANS', 6),
    'REALLN': ('REALLN', 6),
    'NONREVSL': ('NONREVSL', 6),
    'CONSPI': ('CONSPI', 2),
    'MZMSL': ('MZMSL', 6),    # MZM money stock (may be discontinued)
    'DTCOLNVHFNM': ('DTCOLNVHFNM', 2),
    'DTCTHFNM': ('DTCTHFNM', 2),
    'INVEST': ('INVEST', 5),

    # ---- Group 6: Interest Rates and Spreads (22 series) ----
    'FEDFUNDS': ('FEDFUNDS', 2),
    'CP3Mx': ('CP3M', 2),         # 3-month commercial paper
    'TB3MS': ('TB3MS', 2),
    'TB6MS': ('TB6MS', 2),
    'GS1': ('GS1', 2),
    'GS5': ('GS5', 2),
    'GS10': ('GS10', 2),
    'AAA': ('AAA', 2),
    'BAA': ('BAA', 2),
    'COMPAPFFx': ('COMPAPFF', 1),  # CP-FF spread
    'TB3SMFFM': ('TB3SMFFM', 1),
    'TB6SMFFM': ('TB6SMFFM', 1),
    'T1YFFM': ('T1YFFM', 1),
    'T5YFFM': ('T5YFFM', 1),
    'T10YFFM': ('T10YFFM', 1),
    'AAA_FF': ('AAAFFM', 1),      # AAA-FF spread
    'BAA_FF': ('BAAFFM', 1),      # BAA-FF spread
    'TWEXAFEGSMTH': ('TWEXAFEGSMTH', 5),  # Trade-weighted USD
    'EXSZUSx': ('EXSZUS', 5),
    'EXJPUSx': ('EXJPUS', 5),
    'EXUSUKx': ('EXUSUK', 5),
    'EXCAUSx': ('EXCAUS', 5),

    # ---- Group 7: Prices (21 series) ----
    'WPSFD49207': ('WPSFD49207', 6),  # PPI: Finished goods
    'WPSFD49502': ('WPSFD49502', 6),  # PPI: Finished consumer goods
    'WPSID61': ('WPSID61', 6),        # PPI: Intermediate materials
    'WPSID62': ('WPSID62', 6),        # PPI: Crude materials
    'OILPRICEx': ('MCOILWTICO', 5),   # WTI crude oil
    'PPICMM': ('PPICMM', 5),          # PPI: Metals & metal products
    'CPIAUCSL': ('CPIAUCSL', 6),
    'CPIAPPSL': ('CPIAPPSL', 6),      # CPI: Apparel
    'CPITRNSL': ('CPITRNSL', 6),      # CPI: Transportation
    'CPIMEDSL': ('CPIMEDSL', 6),
    'CUSR0000SAC': ('CUSR0000SAC', 6),  # CPI: Commodities
    'CUSR0000SAD': ('CUSR0000SAD', 6),  # CPI: Durables
    'CUSR0000SAS': ('CUSR0000SAS', 6),  # CPI: Services
    'CPIULFSL': ('CPIULFSL', 6),
    'CUSR0000SA0L2': ('CUSR0000SA0L2', 6),
    'CUSR0000SA0L5': ('CUSR0000SA0L5', 6),
    'PCEPI': ('PCEPI', 6),
    'DDURRG3M086SBEA': ('DDURRG3M086SBEA', 6),   # PCE: Durable goods
    'DNDGRG3M086SBEA': ('DNDGRG3M086SBEA', 6),   # PCE: Nondurable goods
    'DSERRG3M086SBEA': ('DSERRG3M086SBEA', 6),   # PCE: Services
    'CPILFESL': ('CPILFESL', 6),                   # Core CPI

    # ---- Group 8: Stock Market (5 series) ----
    'S_P500': ('SP500', 5),
    'S_P_div_yield': ('SP500', 5),    # Will derive from SP500
    'S_P_PE_ratio': ('SP500', 5),     # Will derive
    'VXOCLSx': ('VXOCLS', 1),        # VXO (old VIX)
    'VIXCLSx': ('VIXCLS', 1),        # VIX
}

# Alternative / fallback series for discontinued ones
FALLBACK_SERIES = {
    'HELPWANT': 'JTSJOL',      # JOLTS job openings replaces help-wanted
    'HWIURATIO': None,          # No direct replacement
    'MZMSL': 'M2SL',           # MZM discontinued, use M2
    'COMPAPFF': None,           # May not be available
    'TWEXAFEGSMTH': 'DTWEXBGS', # Trade-weighted USD alternative
    'EXSZUS': 'DEXSZUS',       # Daily -> monthly
    'EXJPUS': 'DEXJPUS',
    'EXUSUK': 'DEXUSUK',
    'EXCAUS': 'DEXCAUS',
    'MCOILWTICO': 'WTISPLC',   # WTI spot alternative
    'VXOCLS': 'VIXCLS',        # VXO discontinued, use VIX
    'RETAILMPCSMSA': 'RSAFS',  # Retail sales alternative
    'CMRMTSPL': 'RETAILIMSA',  # Real M&T sales alternative
    'CP3M': 'TB3MS',           # 3M commercial paper -> 3M T-bill
}


def download_expanded_fred():
    """Download expanded FRED series set targeting 127 variables."""
    from pandas_datareader import data as pdr

    print("=" * 60)
    print("TASK 1: MAXIMIZING FRED DATA COVERAGE")
    print("=" * 60)

    start_date = '1959-01-01'
    end_date = '2026-03-01'

    raw_data = pd.DataFrame()
    tcodes_dict = {}
    succeeded = []
    failed = []
    fallback_used = []

    # Deduplicate by API series ID
    seen_api_ids = set()
    series_to_download = []
    for fred_md_name, (api_id, tcode) in FRED_MD_FULL.items():
        if api_id not in seen_api_ids:
            series_to_download.append((fred_md_name, api_id, tcode))
            seen_api_ids.add(api_id)

    print(f"[DATA] Attempting {len(series_to_download)} unique FRED series (paper target: 127)...")

    for i, (name, api_id, tcode) in enumerate(series_to_download):
        try:
            s = pdr.get_data_fred(api_id, start=start_date, end=end_date)
            if len(s) > 100:  # Require reasonable history
                raw_data[name] = s.iloc[:, 0]
                tcodes_dict[name] = tcode
                succeeded.append((name, api_id, len(s)))
            else:
                raise ValueError(f"Only {len(s)} rows")
        except Exception as e:
            # Try fallback
            fallback_id = FALLBACK_SERIES.get(api_id)
            if fallback_id and fallback_id not in seen_api_ids:
                try:
                    s = pdr.get_data_fred(fallback_id, start=start_date, end=end_date)
                    if len(s) > 100:
                        raw_data[name] = s.iloc[:, 0]
                        tcodes_dict[name] = tcode
                        seen_api_ids.add(fallback_id)
                        fallback_used.append((name, api_id, fallback_id))
                        succeeded.append((name, fallback_id, len(s)))
                        continue
                except:
                    pass
            failed.append((name, api_id, str(e)[:60]))

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(series_to_download)} ({len(succeeded)} OK, {len(failed)} failed)")

    # Resample to month-start
    raw_data = raw_data.resample('MS').last().sort_index()
    tcodes = pd.Series(tcodes_dict)

    print(f"\n[DATA] COVERAGE REPORT:")
    print(f"  Paper target: 127 variables")
    print(f"  Unique series attempted: {len(series_to_download)}")
    print(f"  Successfully downloaded: {len(succeeded)}")
    print(f"  Fallbacks used: {len(fallback_used)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Coverage ratio: {len(succeeded)/127:.1%}")

    if failed:
        print(f"\n  Failed series (first 15):")
        for name, api_id, err in failed[:15]:
            print(f"    {name} ({api_id}): {err}")

    if fallback_used:
        print(f"\n  Fallback substitutions:")
        for name, orig, fallback in fallback_used:
            print(f"    {name}: {orig} -> {fallback}")

    return raw_data, tcodes, {
        'target': 127,
        'attempted': len(series_to_download),
        'succeeded': len(succeeded),
        'failed_count': len(failed),
        'fallbacks': len(fallback_used),
        'coverage_pct': round(len(succeeded)/127*100, 1),
        'failed_list': [(n, a) for n, a, _ in failed],
        'fallback_list': fallback_used,
        'succeeded_list': [(n, a) for n, a, _ in succeeded],
    }


def apply_tcode_transforms(df, tcodes):
    """Apply FRED-MD transformation codes."""
    transformed = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col not in tcodes.index:
            continue
        tc = int(tcodes[col])
        series = df[col].copy()
        if tc == 1:
            transformed[col] = series
        elif tc == 2:
            transformed[col] = series.diff()
        elif tc == 3:
            transformed[col] = series.diff().diff()
        elif tc == 4:
            transformed[col] = np.log(series.clip(lower=1e-10))
        elif tc == 5:
            transformed[col] = np.log(series.clip(lower=1e-10)).diff()
        elif tc == 6:
            transformed[col] = np.log(series.clip(lower=1e-10)).diff().diff()
        elif tc == 7:
            transformed[col] = (series / series.shift(1) - 1).diff()

    transformed = transformed.dropna(how='all')
    threshold = len(transformed) * 0.10
    transformed = transformed.dropna(axis=1, thresh=int(len(transformed) - threshold))
    transformed = transformed.fillna(transformed.median())
    print(f"[DATA] After transforms: {transformed.shape[0]} months x {transformed.shape[1]} variables")
    return transformed


def download_etf_returns():
    """Download monthly ETF returns."""
    import yfinance as yf
    print("[DATA] Downloading ETF returns...")
    prices = pd.DataFrame()
    for etf in ETFS:
        try:
            ticker = yf.Ticker(etf)
            hist = ticker.history(start='1999-12-01', end='2026-03-01', interval='1mo')
            if len(hist) > 0:
                prices[etf] = hist['Close']
        except:
            pass
    prices.index = prices.index.tz_localize(None)
    prices = prices.resample('ME').last()
    returns = prices.pct_change().dropna()
    returns.index = returns.index.to_period('M').to_timestamp()
    print(f"[DATA] ETF returns: {returns.shape}")
    return returns


def download_nber():
    """Download NBER recession indicator."""
    try:
        from pandas_datareader import data as pdr
        usrec = pdr.get_data_fred('USREC', start='1959-01-01', end='2026-03-01')
        return pd.DataFrame({'recession': usrec.iloc[:, 0]})
    except:
        dates = pd.date_range('1960-01-01', '2026-03-01', freq='MS')
        recession_periods = [
            ('1960-04-01', '1961-02-01'), ('1969-12-01', '1970-11-01'),
            ('1973-11-01', '1975-03-01'), ('1980-01-01', '1980-07-01'),
            ('1981-07-01', '1982-11-01'), ('1990-07-01', '1991-03-01'),
            ('2001-03-01', '2001-11-01'), ('2007-12-01', '2009-06-01'),
            ('2020-02-01', '2020-04-01'),
        ]
        recession = pd.Series(0, index=dates, name='recession')
        for start, end in recession_periods:
            mask = (recession.index >= start) & (recession.index <= end)
            recession[mask] = 1
        return pd.DataFrame({'recession': recession})


def run_pca(transformed_data):
    """PCA dimensionality reduction."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(transformed_data)
    pca_full = PCA()
    pca_full.fit(scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.searchsorted(cumvar, PCA_VARIANCE_THRESHOLD) + 1
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(scaled)
    print(f"[PCA] {n_components} components explain {cumvar[n_components-1]:.1%} variance")
    return components, pca, scaler, n_components, cumvar


def two_layer_kmeans(pca_data, dates):
    """Two-layer k-means: crisis isolation + normal clustering."""
    # Layer 1: k=2 to separate crisis
    km_l1 = KMeans(n_clusters=2, n_init=50, random_state=42)
    labels_l1 = km_l1.fit_predict(pca_data)
    crisis_cluster = np.argmin(np.bincount(labels_l1))
    crisis_mask = labels_l1 == crisis_cluster

    # Safety check: if crisis cluster is too small (<2% of data) or too large (>50%),
    # use distance-based outlier detection as fallback
    crisis_frac = crisis_mask.sum() / len(pca_data)
    if crisis_frac < 0.05 or crisis_frac > 0.50:
        print(f"[CLUSTER] Layer 1 k=2 gave {crisis_mask.sum()} crisis months ({crisis_frac:.1%}) — using distance-based fallback")
        # Use Mahalanobis-like distance: points beyond 2 std from centroid
        centroid = pca_data.mean(axis=0)
        dists = np.linalg.norm(pca_data - centroid, axis=1)
        threshold = np.percentile(dists, 88)  # ~12% crisis months (paper typical: 5-15%)
        crisis_mask = dists > threshold
        # Re-fit km_l1 on this split for downstream compatibility
        pseudo_labels = np.zeros(len(pca_data), dtype=int)
        pseudo_labels[crisis_mask] = 1
        km_l1 = KMeans(n_clusters=2, n_init=10, random_state=42)
        km_l1.fit(pca_data)
        # Override: set crisis_cluster to the one matching our mask
        c0_crisis = np.mean(crisis_mask[km_l1.labels_ == 0])
        c1_crisis = np.mean(crisis_mask[km_l1.labels_ == 1])
        crisis_cluster = 0 if c0_crisis > c1_crisis else 1

    print(f"[CLUSTER] Layer 1: {crisis_mask.sum()} crisis months, {(~crisis_mask).sum()} normal months")

    # Elbow for layer 2
    normal_data = pca_data[~crisis_mask]
    inertias = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        km.fit(normal_data)
        inertias.append(km.inertia_)

    # Layer 2: k=5 on normal months
    km_l2 = KMeans(n_clusters=N_REGIMES_LAYER2, n_init=50, random_state=42)
    labels_l2 = km_l2.fit_predict(normal_data)

    # Combine
    final_labels = np.zeros(len(pca_data), dtype=int)
    final_labels[crisis_mask] = 0
    final_labels[~crisis_mask] = labels_l2 + 1

    # Fuzzy probabilities via inverse distance
    regime_probs = np.zeros((len(pca_data), TOTAL_REGIMES))
    all_centers = np.vstack([km_l1.cluster_centers_[crisis_cluster:crisis_cluster+1],
                             km_l2.cluster_centers_])
    # Truncate/pad centers to match PCA dimensions
    n_dim = pca_data.shape[1]
    if all_centers.shape[1] < n_dim:
        all_centers = np.pad(all_centers, ((0,0),(0,n_dim-all_centers.shape[1])))
    elif all_centers.shape[1] > n_dim:
        all_centers = all_centers[:, :n_dim]

    for i in range(len(pca_data)):
        dists = np.linalg.norm(pca_data[i] - all_centers, axis=1) + 1e-10
        inv_dists = 1.0 / dists
        regime_probs[i] = inv_dists / inv_dists.sum()

    regime_df = pd.DataFrame(index=dates[:len(pca_data)])
    regime_df['regime'] = final_labels
    for r in range(TOTAL_REGIMES):
        regime_df[f'prob_r{r}'] = regime_probs[:, r]

    elbow_diffs = np.diff(inertias)
    elbow_k = np.argmin(elbow_diffs) + 3 if len(elbow_diffs) > 0 else 5

    return regime_df, km_l1, km_l2, elbow_k, inertias


def compute_metrics(returns_series):
    """Compute Sharpe, Sortino, MaxDD, % positive."""
    clean = returns_series.dropna()
    if len(clean) < 12:
        return {'sharpe': np.nan, 'sortino': np.nan, 'maxdd': np.nan,
                'pct_positive': np.nan, 'n_months': len(clean)}
    mean_ret = clean.mean() * 12
    std_ret = clean.std() * np.sqrt(12)
    downside = clean[clean < 0].std() * np.sqrt(12)
    if downside == 0:
        downside = 1e-10
    sharpe = mean_ret / std_ret if std_ret > 0 else 0
    sortino = mean_ret / downside
    cumret = (1 + clean).cumprod()
    maxdd = ((cumret - cumret.cummax()) / cumret.cummax()).min()
    return {
        'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'maxdd': round(maxdd, 4), 'pct_positive': round((clean > 0).mean(), 4),
        'n_months': len(clean), 'ann_return': round(mean_ret, 4),
        'ann_vol': round(std_ret, 4),
    }


# ============================================================
# TASK 2: STRESS-PERIOD VALIDATION
# ============================================================

CRISIS_WINDOWS = {
    'Volcker_Recession_1981-82': ('1981-07-01', '1982-11-01'),
    'Black_Monday_1987': ('1987-08-01', '1988-03-01'),
    'Gulf_War_Recession_1990-91': ('1990-07-01', '1991-06-01'),
    'LTCM_Asian_1997-98': ('1997-07-01', '1998-12-01'),
    'Dot-Com_Bust_2000-02': ('2000-03-01', '2002-12-01'),
    'GFC_2007-09': ('2007-12-01', '2009-06-01'),
    'Euro_Crisis_2011-12': ('2011-07-01', '2012-06-01'),
    'COVID_2020': ('2020-02-01', '2020-06-01'),
    'Inflation_Tightening_2022': ('2022-01-01', '2022-12-01'),
}

def stress_period_validation(regime_df, nber_df, etf_returns):
    """Validate regime detection across known crisis windows."""
    print("\n" + "=" * 60)
    print("TASK 2: STRESS-PERIOD VALIDATION")
    print("=" * 60)

    results = {}
    for crisis_name, (start, end) in CRISIS_WINDOWS.items():
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        # Get regime assignments for this window
        mask = (regime_df.index >= start_dt) & (regime_df.index <= end_dt)
        window_regimes = regime_df.loc[mask]

        if len(window_regimes) == 0:
            print(f"\n  {crisis_name}: NO DATA (before ETF coverage)")
            results[crisis_name] = {'status': 'no_data', 'reason': 'before_coverage'}
            continue

        regime_counts = window_regimes['regime'].value_counts().to_dict()
        dominant_regime = window_regimes['regime'].mode().iloc[0] if len(window_regimes) > 0 else -1
        r0_fraction = (window_regimes['regime'] == 0).mean()
        mean_r0_prob = window_regimes['prob_r0'].mean()

        # Check if ETF data available for performance
        etf_mask = (etf_returns.index >= start_dt) & (etf_returns.index <= end_dt)
        spy_crisis = etf_returns.loc[etf_mask, 'SPY'] if 'SPY' in etf_returns.columns else pd.Series()
        spy_drawdown = np.nan
        if len(spy_crisis) > 1:
            cum = (1 + spy_crisis).cumprod()
            spy_drawdown = ((cum - cum.cummax()) / cum.cummax()).min()

        # NBER overlap for this window
        nber_mask = (nber_df.index >= start_dt) & (nber_df.index <= end_dt)
        nber_window = nber_df.loc[nber_mask]
        nber_recession_months = (nber_window['recession'] == 1).sum() if len(nber_window) > 0 else 0
        overlap_with_r0 = 0
        if len(window_regimes) > 0 and len(nber_window) > 0:
            common = window_regimes.index.intersection(nber_window.index)
            if len(common) > 0:
                overlap_with_r0 = (
                    (regime_df.loc[common, 'regime'] == 0) &
                    (nber_df.loc[common, 'recession'] == 1)
                ).sum()

        detected = r0_fraction >= 0.3 or mean_r0_prob >= 0.25

        result = {
            'n_months': len(window_regimes),
            'regime_distribution': {int(k): int(v) for k, v in regime_counts.items()},
            'dominant_regime': int(dominant_regime),
            'dominant_label': REGIME_LABELS.get(int(dominant_regime), 'Unknown'),
            'r0_fraction': round(float(r0_fraction), 3),
            'mean_r0_prob': round(float(mean_r0_prob), 3),
            'spy_drawdown': round(float(spy_drawdown), 4) if not np.isnan(spy_drawdown) else None,
            'nber_recession_months': int(nber_recession_months),
            'r0_nber_overlap': int(overlap_with_r0),
            'crisis_detected': detected,
        }
        results[crisis_name] = result

        status = "DETECTED" if detected else "MISSED"
        print(f"\n  {crisis_name}:")
        print(f"    Months: {len(window_regimes)}, Dominant: R{dominant_regime} ({REGIME_LABELS.get(int(dominant_regime), '?')})")
        print(f"    R0 fraction: {r0_fraction:.1%}, Mean P(R0): {mean_r0_prob:.3f}")
        print(f"    SPY drawdown: {spy_drawdown:.1%}" if not np.isnan(spy_drawdown) else "    SPY: N/A")
        print(f"    Crisis detection: {status}")

    # Summary stats
    detected_count = sum(1 for v in results.values() if isinstance(v, dict) and v.get('crisis_detected'))
    total_with_data = sum(1 for v in results.values() if isinstance(v, dict) and v.get('n_months', 0) > 0)
    print(f"\n  SUMMARY: {detected_count}/{total_with_data} crises detected via R0 signal")
    results['_summary'] = {
        'detected': detected_count,
        'total_with_data': total_with_data,
        'detection_rate': round(detected_count / max(total_with_data, 1), 3),
    }
    return results


def plot_stress_validation(regime_df, crisis_results, output_path):
    """Plot stress-period regime detection results."""
    crises_with_data = [(k, v) for k, v in crisis_results.items()
                        if k != '_summary' and isinstance(v, dict) and v.get('n_months', 0) > 0]
    if not crises_with_data:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    names = [k.replace('_', ' ') for k, _ in crises_with_data]
    r0_fracs = [v['r0_fraction'] for _, v in crises_with_data]
    r0_probs = [v['mean_r0_prob'] for _, v in crises_with_data]
    detected = [v.get('crisis_detected', False) for _, v in crises_with_data]

    x = np.arange(len(names))
    width = 0.35
    colors_frac = ['#d62728' if d else '#aaaaaa' for d in detected]
    colors_prob = ['#ff7f0e' if d else '#cccccc' for d in detected]

    bars1 = ax.bar(x - width/2, r0_fracs, width, label='R0 Fraction', color=colors_frac, edgecolor='white')
    bars2 = ax.bar(x + width/2, r0_probs, width, label='Mean P(R0)', color=colors_prob, edgecolor='white')

    ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='R0 Fraction Threshold (30%)')
    ax.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='P(R0) Threshold (25%)')

    ax.set_ylabel('Value')
    ax.set_title('Stress-Period Regime Detection: R0 (Crisis) Signal Strength')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {output_path}")


# ============================================================
# TASK 3: LAG-SENSITIVITY STABILIZATION
# ============================================================

def lag_sensitivity_stabilization(fred_transformed, pca_model, scaler,
                                   km_l1, km_l2, regime_df):
    """
    Attempt multiple strategies to improve lag agreement:
    1. Baseline: direct predict on lagged data (original method)
    2. 3-month rolling average smoothing before PCA
    3. Higher PCA variance threshold (0.99)
    4. Ensemble: majority vote over 1-3 month lags
    5. Soft label comparison (fuzzy probability correlation)
    """
    print("\n" + "=" * 60)
    print("TASK 3: LAG-SENSITIVITY STABILIZATION")
    print("=" * 60)

    results = {}
    crisis_cluster = np.argmin(np.bincount(km_l1.labels_))

    def predict_regimes(data_subset, common_idx):
        """Predict regime labels for a data subset."""
        scaled = scaler.transform(data_subset.loc[common_idx])
        pca_out = pca_model.transform(scaled)
        labels_l1 = km_l1.predict(pca_out)
        labels = np.zeros(len(pca_out), dtype=int)
        crisis_mask = labels_l1 == crisis_cluster
        labels[crisis_mask] = 0
        if (~crisis_mask).sum() > 0:
            labels_l2 = km_l2.predict(pca_out[~crisis_mask])
            labels[~crisis_mask] = labels_l2 + 1
        return labels

    def predict_probs(data_subset, common_idx):
        """Predict fuzzy regime probabilities."""
        scaled = scaler.transform(data_subset.loc[common_idx])
        pca_out = pca_model.transform(scaled)

        # Reconstruct centers
        all_centers = np.vstack([
            km_l1.cluster_centers_[crisis_cluster:crisis_cluster+1],
            km_l2.cluster_centers_
        ])
        n_dim = pca_out.shape[1]
        if all_centers.shape[1] < n_dim:
            all_centers = np.pad(all_centers, ((0,0),(0,n_dim-all_centers.shape[1])))
        elif all_centers.shape[1] > n_dim:
            all_centers = all_centers[:, :n_dim]

        probs = np.zeros((len(pca_out), TOTAL_REGIMES))
        for i in range(len(pca_out)):
            dists = np.linalg.norm(pca_out[i] - all_centers, axis=1) + 1e-10
            inv = 1.0 / dists
            probs[i] = inv / inv.sum()
        return probs

    # ---- Method 1: Baseline (original lag test) ----
    print("\n  [1/5] Baseline: 1-month lag, direct predict...")
    lagged_1 = fred_transformed.shift(1).dropna()
    common_1 = lagged_1.index.intersection(regime_df.index)
    if len(common_1) > 50:
        labels_1 = predict_regimes(lagged_1, common_1)
        orig_1 = regime_df.loc[common_1, 'regime'].values
        baseline_agreement = (labels_1 == orig_1).mean()
        results['baseline_1m_lag'] = round(float(baseline_agreement), 4)
        print(f"    Agreement: {baseline_agreement:.2%}")
    else:
        results['baseline_1m_lag'] = 0.0

    # ---- Method 2: 3-month rolling mean smoothing ----
    print("  [2/5] 3-month rolling mean smoothing before PCA...")
    smoothed = fred_transformed.rolling(3, min_periods=2).mean().dropna()
    smoothed_lagged = smoothed.shift(1).dropna()
    common_2 = smoothed_lagged.index.intersection(regime_df.index)
    if len(common_2) > 50:
        labels_2 = predict_regimes(smoothed_lagged, common_2)
        orig_2 = regime_df.loc[common_2, 'regime'].values
        smooth_agreement = (labels_2 == orig_2).mean()
        results['smoothed_3m_lag'] = round(float(smooth_agreement), 4)
        print(f"    Agreement: {smooth_agreement:.2%}")
    else:
        results['smoothed_3m_lag'] = 0.0

    # ---- Method 3: Re-fit PCA with 0.99 variance ----
    print("  [3/5] Higher PCA threshold (0.99 variance)...")
    scaler_99 = StandardScaler()
    scaled_99 = scaler_99.fit_transform(fred_transformed)
    pca_99 = PCA(n_components=0.99)
    pca_99.fit(scaled_99)
    n_comp_99 = pca_99.n_components_
    print(f"    PCA components at 0.99: {n_comp_99}")

    lagged_scaled_99 = scaler_99.transform(lagged_1.loc[common_1])
    pca_orig_99 = pca_99.transform(scaler_99.transform(fred_transformed.loc[common_1]))
    pca_lag_99 = pca_99.transform(lagged_scaled_99)

    # Re-cluster with higher-dim PCA
    km_99_l1 = KMeans(n_clusters=2, n_init=50, random_state=42)
    labels_99_l1 = km_99_l1.fit_predict(pca_orig_99)
    crisis_99 = np.argmin(np.bincount(labels_99_l1))
    crisis_mask_99 = labels_99_l1 == crisis_99
    normal_99 = pca_orig_99[~crisis_mask_99]
    km_99_l2 = KMeans(n_clusters=N_REGIMES_LAYER2, n_init=50, random_state=42)
    if len(normal_99) > N_REGIMES_LAYER2:
        km_99_l2.fit(normal_99)

    # Predict on lagged data
    labels_lag_99_l1 = km_99_l1.predict(pca_lag_99)
    labels_lag_99 = np.zeros(len(pca_lag_99), dtype=int)
    crisis_lag_99 = labels_lag_99_l1 == crisis_99
    labels_lag_99[crisis_lag_99] = 0
    if (~crisis_lag_99).sum() > 0 and len(normal_99) > N_REGIMES_LAYER2:
        labels_lag_99[~crisis_lag_99] = km_99_l2.predict(pca_lag_99[~crisis_lag_99]) + 1

    # Original labels from 0.99 PCA
    orig_labels_99 = np.zeros(len(pca_orig_99), dtype=int)
    orig_labels_99[crisis_mask_99] = 0
    if (~crisis_mask_99).sum() > 0 and len(normal_99) > N_REGIMES_LAYER2:
        orig_labels_99[~crisis_mask_99] = km_99_l2.predict(normal_99) + 1

    pca99_agreement = (labels_lag_99 == orig_labels_99).mean()
    results['pca_099_lag'] = round(float(pca99_agreement), 4)
    print(f"    Agreement: {pca99_agreement:.2%}")

    # ---- Method 4: Ensemble majority vote (1-3 month lags) ----
    print("  [4/5] Ensemble: majority vote over 1-3 month lags...")
    lagged_2m = fred_transformed.shift(2).dropna()
    lagged_3m = fred_transformed.shift(3).dropna()
    common_ens = lagged_3m.index.intersection(regime_df.index)
    if len(common_ens) > 50:
        l1 = predict_regimes(lagged_1, common_ens)
        l2 = predict_regimes(lagged_2m, common_ens)
        l3 = predict_regimes(lagged_3m, common_ens)
        ensemble = np.zeros(len(common_ens), dtype=int)
        for i in range(len(common_ens)):
            votes = [l1[i], l2[i], l3[i]]
            ensemble[i] = max(set(votes), key=votes.count)
        orig_ens = regime_df.loc[common_ens, 'regime'].values
        ens_agreement = (ensemble == orig_ens).mean()
        results['ensemble_1to3m'] = round(float(ens_agreement), 4)
        print(f"    Agreement: {ens_agreement:.2%}")
    else:
        results['ensemble_1to3m'] = 0.0

    # ---- Method 5: Soft label comparison (cosine similarity) ----
    print("  [5/5] Soft label comparison (probability correlation)...")
    if len(common_1) > 50:
        probs_orig = regime_df.loc[common_1, [f'prob_r{r}' for r in range(TOTAL_REGIMES)]].values
        probs_lag = predict_probs(lagged_1, common_1)
        # Per-month cosine similarity
        cosine_sims = []
        for i in range(len(common_1)):
            dot = np.dot(probs_orig[i], probs_lag[i])
            norm = np.linalg.norm(probs_orig[i]) * np.linalg.norm(probs_lag[i])
            cosine_sims.append(dot / max(norm, 1e-10))
        mean_cosine = np.mean(cosine_sims)
        results['soft_cosine_sim'] = round(float(mean_cosine), 4)
        print(f"    Mean cosine similarity: {mean_cosine:.4f}")

        # Also rank correlation of probability vectors
        from scipy.stats import spearmanr
        flat_orig = probs_orig.flatten()
        flat_lag = probs_lag.flatten()
        rho, _ = spearmanr(flat_orig, flat_lag)
        results['soft_spearman_rho'] = round(float(rho), 4)
        print(f"    Spearman rho (flattened probs): {rho:.4f}")
    else:
        results['soft_cosine_sim'] = 0.0
        results['soft_spearman_rho'] = 0.0

    # ---- Summary ----
    best_method = max(
        [(k, v) for k, v in results.items() if k.startswith(('baseline', 'smoothed', 'pca', 'ensemble'))],
        key=lambda x: x[1]
    )
    print(f"\n  LAG SENSITIVITY SUMMARY:")
    for method, val in sorted(results.items()):
        print(f"    {method}: {val}")
    print(f"  Best hard-label method: {best_method[0]} ({best_method[1]:.2%})")
    results['best_method'] = best_method[0]
    results['best_agreement'] = best_method[1]

    return results


def plot_lag_sensitivity(lag_results, output_path):
    """Plot lag sensitivity comparison."""
    hard_methods = {k: v for k, v in lag_results.items()
                    if k in ('baseline_1m_lag', 'smoothed_3m_lag', 'pca_099_lag', 'ensemble_1to3m')}
    soft_methods = {k: v for k, v in lag_results.items()
                    if k in ('soft_cosine_sim', 'soft_spearman_rho')}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Hard label agreement
    names = list(hard_methods.keys())
    vals = list(hard_methods.values())
    colors = ['#1f77b4' if v < 0.85 else '#2ca02c' for v in vals]
    ax1.barh(names, vals, color=colors, edgecolor='white')
    ax1.axvline(x=0.85, color='red', linestyle='--', label='Target (85%)')
    ax1.axvline(x=0.65, color='orange', linestyle='--', alpha=0.5, label='Previous (65%)')
    ax1.set_xlabel('Agreement Rate')
    ax1.set_title('Lag Sensitivity: Hard Label Agreement')
    ax1.legend(fontsize=8)
    ax1.set_xlim(0, 1)

    # Soft similarity
    snames = list(soft_methods.keys())
    svals = list(soft_methods.values())
    ax2.barh(snames, svals, color='#ff7f0e', edgecolor='white')
    ax2.set_xlabel('Similarity Score')
    ax2.set_title('Lag Sensitivity: Soft Label Similarity')
    ax2.set_xlim(0, 1)

    plt.suptitle('Lag Sensitivity Stabilization Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved: {output_path}")


# ============================================================
# TASK 2b: FULL BACKTEST ON EXPANDED DATA
# ============================================================

def build_transition_matrix(regimes, n_regimes=TOTAL_REGIMES):
    T = np.zeros((n_regimes, n_regimes))
    for i in range(len(regimes) - 1):
        T[regimes[i], regimes[i + 1]] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return T / row_sums


def ridge_regression_forecast(macro_features, returns, regime_probs, regime_labels,
                               window=ROLLING_WINDOW):
    n_months = len(returns)
    n_etfs = returns.shape[1]
    predictions = np.full((n_months, n_etfs), np.nan)
    for t in range(window, n_months):
        train_start = t - window
        X_train = macro_features[train_start:t]
        y_train = returns.iloc[train_start:t].values
        regimes_train = regime_labels[train_start:t]
        next_probs = regime_probs[t]
        pred_t = np.zeros(n_etfs)
        for r in range(TOTAL_REGIMES):
            mask_r = regimes_train == r
            if mask_r.sum() < 5:
                continue
            X_r = X_train[mask_r]
            y_r = y_train[mask_r]
            for j in range(n_etfs):
                ridge = Ridge(alpha=1.0)
                ridge.fit(X_r, y_r[:, j])
                pred_t[j] += next_probs[r] * ridge.predict(macro_features[t:t+1])[0]
        predictions[t] = pred_t
    return predictions


def naive_forecast(returns, regime_labels, regime_probs, window=ROLLING_WINDOW):
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
            mean_r = y_train[mask_r].mean(axis=0)
            std_r = y_train[mask_r].std(axis=0) + 1e-10
            pred_t += next_probs[r] * (mean_r / std_r)
        predictions[t] = pred_t
    return predictions


def long_only_portfolio(predictions, l=3):
    n_months, n_etfs = predictions.shape
    weights = np.zeros_like(predictions)
    for t in range(n_months):
        if np.isnan(predictions[t]).any():
            continue
        top_l = np.argsort(predictions[t])[-l:]
        weights[t, top_l] = 1.0 / l
    return weights


def compute_portfolio_returns(weights, etf_returns):
    valid = ~np.isnan(weights).any(axis=1) & (np.abs(weights).sum(axis=1) > 0)
    port_ret = np.full(len(weights), np.nan)
    for t in range(len(weights)):
        if valid[t]:
            port_ret[t] = np.sum(weights[t] * etf_returns.iloc[t].values)
    return pd.Series(port_ret, index=etf_returns.index[:len(weights)])


def validate_nber_overlap(regime_df, nber_df):
    common_idx = regime_df.index.intersection(nber_df.index)
    if len(common_idx) == 0:
        return 0.0
    regime0_months = set(regime_df.loc[common_idx][regime_df.loc[common_idx, 'regime'] == 0].index)
    nber_months = set(nber_df.loc[common_idx][nber_df.loc[common_idx, 'recession'] == 1].index)
    if not regime0_months or not nber_months:
        return 0.0
    overlap = regime0_months & nber_months
    recall = len(overlap) / len(nber_months)
    precision = len(overlap) / len(regime0_months)
    print(f"  R0 months: {len(regime0_months)}, NBER months: {len(nber_months)}, Overlap: {len(overlap)}")
    print(f"  Precision: {precision:.2%}, Recall: {recall:.2%}")
    return recall, precision


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("GAP-CLOSING PIPELINE — arXiv 2503.11499")
    print(f"Run date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = {}

    # ---- TASK 1: Expanded data download ----
    fred_raw, tcodes, coverage_report = download_expanded_fred()
    all_results['coverage'] = coverage_report

    fred_transformed = apply_tcode_transforms(fred_raw, tcodes)
    etf_returns = download_etf_returns()
    nber = download_nber()

    # ---- Regime classification on expanded data ----
    print("\n" + "=" * 60)
    print("REGIME CLASSIFICATION (Expanded Dataset)")
    print("=" * 60)

    pca_components, pca_model, scaler, n_pca, cumvar = run_pca(fred_transformed)
    regime_df, km_l1, km_l2, elbow_k, inertias = two_layer_kmeans(
        pca_components, fred_transformed.index)

    all_results['expanded_pca_components'] = int(n_pca)
    all_results['expanded_variables'] = int(fred_transformed.shape[1])

    # NBER overlap
    print("\n[VALIDATE] NBER Overlap (expanded data)...")
    nber_result = validate_nber_overlap(regime_df, nber)
    if isinstance(nber_result, tuple):
        nber_recall, nber_precision = nber_result
    else:
        nber_recall, nber_precision = nber_result, 0.0
    all_results['nber_recall'] = round(float(nber_recall), 4)
    all_results['nber_precision'] = round(float(nber_precision), 4)
    print(f"  NBER Recall: {nber_recall:.2%} (target: >= 80%)")

    # ---- Align for backtesting ----
    common_idx = regime_df.index.intersection(etf_returns.index)
    regime_aligned = regime_df.loc[common_idx]
    returns_aligned = etf_returns.loc[common_idx]
    macro_aligned = fred_transformed.loc[common_idx]

    macro_scaled = scaler.transform(macro_aligned)
    macro_pca = pca_model.transform(macro_scaled)

    regime_labels_arr = regime_aligned['regime'].values
    prob_cols = [f'prob_r{r}' for r in range(TOTAL_REGIMES)]
    regime_probs_arr = regime_aligned[prob_cols].values

    # Full-sample backtest
    print("\n[BACKTEST] Ridge regression on expanded data...")
    ridge_preds = ridge_regression_forecast(
        macro_pca, returns_aligned, regime_probs_arr, regime_labels_arr)
    weights = long_only_portfolio(ridge_preds, l=3)
    port_ret = compute_portfolio_returns(weights, returns_aligned)
    expanded_metrics = compute_metrics(port_ret)
    all_results['expanded_ridge_lo_l3'] = expanded_metrics
    print(f"  Ridge_LO_l3 Sharpe: {expanded_metrics['sharpe']} (prev: 0.850)")

    spy_metrics = compute_metrics(returns_aligned['SPY'])
    all_results['expanded_spy'] = spy_metrics

    # OOS split
    oos_cutoff = pd.Timestamp('2022-12-31')
    oos_mask = returns_aligned.index > oos_cutoff
    if oos_mask.sum() > 6:
        oos_ret = port_ret[oos_mask].dropna()
        oos_metrics = compute_metrics(oos_ret)
        all_results['expanded_oos'] = oos_metrics
        print(f"  OOS Sharpe: {oos_metrics['sharpe']} (prev: 1.550)")

    # ---- TASK 2: Stress validation ----
    crisis_results = stress_period_validation(regime_df, nber, etf_returns)
    all_results['stress_validation'] = crisis_results

    plot_stress_validation(regime_df, crisis_results,
                          os.path.join(OUTPUT_DIR, 'stress_validation.png'))

    # ---- TASK 3: Lag sensitivity ----
    lag_results = lag_sensitivity_stabilization(
        fred_transformed, pca_model, scaler, km_l1, km_l2, regime_df)
    all_results['lag_sensitivity'] = lag_results

    plot_lag_sensitivity(lag_results, os.path.join(OUTPUT_DIR, 'lag_sensitivity.png'))

    # ---- Transition matrix on expanded data ----
    T_full = build_transition_matrix(regime_labels_arr)
    all_results['transition_matrix'] = T_full.tolist()

    # ---- Save all results ----
    def convert(obj):
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    results_path = os.path.join(OUTPUT_DIR, 'gap_closing_results.json')
    with open(results_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\n[OUTPUT] Saved: {results_path}")

    # ---- Save expanded regime labels ----
    regime_path = os.path.join(OUTPUT_DIR, 'regime_labels_expanded.csv')
    regime_df.to_csv(regime_path)
    print(f"[OUTPUT] Saved: {regime_path}")

    # ---- Updated regime_state.json ----
    latest_date = regime_aligned.index[-1]
    latest_regime = int(regime_aligned.loc[latest_date, 'regime'])
    latest_probs = regime_aligned.loc[latest_date, prob_cols].values.tolist()
    next_probs = (np.array(latest_probs) @ T_full).tolist()

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
            'n_fred_variables': int(fred_transformed.shape[1]),
            'n_pca_components': int(n_pca),
            'n_regimes': int(TOTAL_REGIMES),
            'rolling_window': int(ROLLING_WINDOW),
            'data_source': 'FRED API (expanded 127-target) + yfinance',
            'coverage': coverage_report['coverage_pct'],
        }
    }
    state_path = os.path.join(OUTPUT_DIR, 'regime_state.json')
    with open(state_path, 'w') as f:
        json.dump(regime_state, f, indent=2)
    print(f"[OUTPUT] Saved: {state_path}")

    print("\n" + "=" * 70)
    print("GAP-CLOSING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Variables: {fred_transformed.shape[1]} (target: 127, coverage: {coverage_report['coverage_pct']}%)")
    print(f"  PCA components: {n_pca}")
    print(f"  NBER recall: {nber_recall:.2%}")
    print(f"  Ridge_LO_l3 Sharpe: {expanded_metrics['sharpe']}")
    print(f"  Lag best agreement: {lag_results.get('best_agreement', 'N/A')}")
    print(f"  Stress detection: {crisis_results.get('_summary', {}).get('detected', '?')}/{crisis_results.get('_summary', {}).get('total_with_data', '?')}")
    print("=" * 70)

    return all_results


if __name__ == '__main__':
    results = main()
