# Regime-Conditioned Sector Allocation

Independent replication of a macro regime-conditioned sector-ETF allocation framework — and a decomposition showing *where* the signal actually lives (and where it doesn't).

📄 **Full write-up:** [Where Macro Regime Signals Actually Live](https://aarontsengquant.substack.com/p/where-macro-regime-signals-actually)
📚 **Paper replicated:** Oliveira et al., *Tactical Asset Allocation with Macroeconomic Regime Detection* ([arXiv:2503.11499](https://arxiv.org/abs/2503.11499))

---

## TL;DR

I replicated a regime-conditioned sector-ETF allocation pipeline, ran it through three independent validation milestones (IC decomposition, permutation testing, benchmark distance), and found that the signal is real but narrower than headline numbers suggest:

- It is **cross-sectional**, not time-series — the model ranks ETFs well but barely beats a random walk on point forecasts.
- It lives in **cyclical sectors** (Materials XLB, Consumer Discretionary XLY) and **transitional regimes** (Reflationary Boom, Economic Difficulty).
- It **actively misranks defensive sectors** (Consumer Staples XLP, Utilities XLU) — a diagnostic for a missing signal dimension, not just noise.

See the write-up linked above for the full narrative. Headline numbers and their provenance below.

---

## Key Findings

The numbers below come from **four independent milestone runs** executed between March 13 and March 23, 2026. Each milestone extended a specific aspect of the replication; they are **not** a single-command reproducible snapshot of one pipeline execution. The provenance table below shows which script produced each figure — this matters, because running `python regime_pipeline.py` on its own will *not* regenerate all of them.

### Provenance of headline numbers

| Finding | Script | Data coverage | Run date |
|---|---|---|---|
| In-sample Sharpe 0.966, OOS Sharpe 1.645 | `gap_closing.py` | 104 FRED-transformed variables (from 128 attempted, 124 successful) | 2026-03-13 |
| XLB IC +0.102, XLY IC +0.091, XLP IC −0.082, XLU IC −0.081 | `ic_validation.py` | 76 FRED series, ~69 after transforms | 2026-03-23 |
| Cross-sectional permutation p ≈ 0.055 (200 runs, seeded) | `negative_control.py` | same as M1 | 2026-03-23 |
| Aggregate OOS RMSFE ratio ≈ 0.99 vs. random walk | `benchmark_distance.py` | same as M1 | 2026-03-23 |

### Narrative summary

- **In-sample Sharpe 0.966** — consistent with a real but modest signal (gap-closing configuration).
- **Out-of-sample Sharpe 1.645** — approximately 2-year window from Jan 2023; wide confidence interval, treat as a data point rather than a conclusion.
- **Information Coefficient by ETF:** XLB **+0.102**, XLY **+0.091** (strongest positive); XLP **−0.082**, XLU **−0.081** (structural negative — defensive misranking).
- **Per-regime IC:** strongest during Reflationary Boom (**+0.118**, n=32), weakest during sentiment-driven recoveries.
- **Permutation test (n=200, seeded, cross-sectional):** observed Sharpe 1.424 vs. shuffled 95th percentile 1.426, **p ≈ 0.055** — borderline at the conventional threshold; the IC decomposition provides complementary economic-coherence evidence.
- **Benchmark distance (OOS):** RMSFE ratio vs. random walk ≈ **0.99** — the edge is in *ranking*, not in point forecasts.

Running `python regime_pipeline.py` alone reproduces the **baseline pipeline** (76-series coverage, Ridge_LO_l3 Sharpe ≈ 0.85), which is distinct from the gap-closing run that produced the 0.966 / 1.645 headline. To regenerate the full set of headline numbers, run the scripts in the order shown under **Full Validation Suite** below, and see [`repro_manifest.json`](./repro_manifest.json) for the machine-readable provenance.

---

## Methodology

**Pipeline:** FRED-MD → PCA → two-layer K-means → Markov regime forecasting → Ridge regression → top-k long-only portfolio → performance evaluation.

**Core design choices:**

- **Ridge regression fitting:** 48-month rolling window, walk-forward, strict causal (uses only data up to t−1).
- **Regime detection layer (PCA + K-means):** fit on the **full sample**. This follows convention in the regime-detection literature (including the original paper), where stable cluster centroids require a longer view than a rolling window provides. A fully walk-forward variant is implemented in `stress_validation.py::walk_forward_backtest` and `type_error_tradeoff.py::walk_forward_crisis_scores` for crisis-window stress testing.
- **Regimes:** 5 latent states identified via two-layer K-means (elbow method confirms k=5).
- **Universe:** 10 sector ETFs (SPY, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY).
- **Macro features:** 76 FRED series in the baseline pipeline (expanded to 104 in the gap-closing run), spanning output, labor, consumption, housing, money/credit, rates, prices, sentiment, and FX, with transformation codes following FRED-MD conventions.
- **Portfolio rule:** deliberately naive — top-k ranking by forecasted return, equal weight, long only. This isolates signal quality from optimizer cleverness.

**Validation milestones:**

| Milestone | Script | Purpose |
|---|---|---|
| M1 | `ic_validation.py` | Per-ETF, per-regime, rolling Spearman IC |
| M2 | `negative_control.py` | Cross-sectional permutation test (200 runs, seeded) |
| M3 | `benchmark_distance.py` | RMSFE vs. Random Walk and AR(1) baselines |
| Stress | `stress_validation.py` | Walk-forward across GFC, Euro Debt, COVID, 2022 Tightening |
| Type I/II | `type_error_tradeoff.py` | Crisis detection confusion matrix across thresholds |
| Gap closing | `gap_closing.py` | Extends FRED coverage from 76 to 104 transformed variables |

---

## Data Sources (Public Only)

| Source | Coverage |
|---|---|
| FRED (via `pandas_datareader`) | 76 macroeconomic series in baseline (up to 104 in gap-closing run), from 1959 onward |
| Yahoo Finance (via `yfinance`) | Monthly returns for 10 sector ETFs |
| NBER | Recession dates (via FRED `USREC` series) |

**No API key required.** `pandas_datareader` uses FRED's public endpoint.

No proprietary data, client information, or employer strategies used at any stage.

---

## Quick Start

**Requires Python 3.11 or newer.** Verified on Python 3.11 and 3.12.

```bash
git clone https://github.com/klmtseng/regime-conditioned-allocation.git
cd regime-conditioned-allocation

python -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\activate
                              # on some Linux hosts: use python3.11 instead of python
pip install -r requirements.txt

# Run the baseline pipeline
python regime_pipeline.py
```

Expected runtime: ~2–10 minutes on a laptop (FRED downloads dominate).

All scripts write to `outputs/` in the repo root. The committed `outputs/` folder contains reference results from the four milestone runs listed in the Provenance table — re-running will overwrite these with your freshly-computed results.

### Dependency strategy

- **`requirements.txt`** — 8 direct dependencies pinned to versions verified end-to-end. Transitive dependencies resolve automatically, which keeps installs working across platforms and minor CA/SSL bundle updates.
- **`requirements-lock.txt`** — full `pip freeze` snapshot from the verification environment. Use this if you want byte-for-byte reproducibility of the environment rather than just a working install.

---

## Full Validation Suite

To regenerate all headline numbers, run the scripts in this order (each reads the shared state produced by `regime_pipeline.py` via the import `from regime_pipeline import ...`):

```bash
python regime_pipeline.py        # baseline pipeline + shared state for validation scripts
python gap_closing.py            # extended FRED coverage (produces IS/OOS Sharpe 0.966 / 1.645)
python ic_validation.py          # M1: per-ETF and per-regime IC decomposition
python negative_control.py       # M2: 200-run cross-sectional permutation test
python benchmark_distance.py     # M3: RMSFE vs. Random Walk / AR(1)
python stress_validation.py      # historical crisis windows (optional)
python type_error_tradeoff.py    # Type I / Type II error analysis (optional)
```

See [`repro_manifest.json`](./repro_manifest.json) for the machine-readable mapping of each headline number to its producing script, data coverage, and seed policy.

---

## Reproducibility Notes

Stated up front because overstated claims help nobody:

- **Not a single-shot reproducible snapshot.** Headline numbers are assembled from four milestone runs (see the Provenance table above). A fresh `python regime_pipeline.py` alone will *not* regenerate all of them — it will produce the baseline pipeline's Ridge_LO_l3 Sharpe ≈ 0.85, which is distinct from the gap-closing 0.966 / 1.645 headline.

- **Public data drift.** FRED series revisions, yfinance adjustments, and the availability of individual FRED tickers (currently ~8 fail: the x-suffix exchange rate series, `OILPRICEx`, `AMDMNOx`, `ANDENOx`, `CONSPI`) mean exact numeric reproduction is not guaranteed across different dates. External fresh-clone runs have observed IC magnitudes differing by up to ±50% and permutation p-values ranging across 0.05–0.40. **Qualitative findings — cyclical sectors positive, defensives negative, cross-sectional ranking alpha greater than point-forecast alpha — are stable across reruns.**

- **Regime detection layer uses full-sample K-means and PCA.** The regime label assigned to any historical month t depends on the cluster centroids, which are fit on the full available data. This follows the convention in the regime-detection literature; strictly walk-forward variants exist in `stress_validation.py` and `type_error_tradeoff.py`. This design choice is the primary driver of the IC drift observed across reruns — when FRED revises historical data or adds new months, the centroids shift, and historical regime labels shift with them.

- **One known non-determinism in the core pipeline.** The permutation test inside `regime_pipeline.py` (around line 604) uses `np.random.permutation` without a fixed seed. The `p ≈ 0.055` figure in the Provenance table comes from `negative_control.py`, which *does* fix a seed (`BASE_SEED=20260323`, 200 permutations). The pipeline-internal permutation test is a legacy/exploratory path and its p-value should not be relied upon.

---

## Repository Layout

```
.
├── README.md
├── LICENSE                       MIT
├── .gitignore
├── requirements.txt              8 direct dependencies, pinned
├── requirements-lock.txt         Full frozen environment (optional)
├── repro_manifest.json           Machine-readable provenance for headline numbers
├── regime_pipeline.py            Baseline pipeline (data → regimes → ridge → portfolio)
├── gap_closing.py                Extended FRED coverage (produces IS/OOS Sharpe headline)
├── ic_validation.py              M1: Information Coefficient decomposition
├── negative_control.py           M2: Cross-sectional permutation test (seeded)
├── benchmark_distance.py         M3: RMSFE vs. baselines
├── stress_validation.py          Walk-forward historical crisis windows
├── type_error_tradeoff.py        Crisis detection Type I / II analysis
└── outputs/                      Reference results from the four milestone runs
    ├── *.csv                     Tables (IC per ETF, per regime, benchmark distances, ...)
    ├── *_report.md               Methodology + findings reports per milestone
    └── *.png                     Figures (cumulative returns, regime timeline, permutation, stress, ...)
```

---

## Limitations

- **Short OOS window.** Roughly 24 monthly observations; Sharpe confidence interval is wide (plausibly 0.8 to 2.4). The OOS > IS pattern deserves scrutiny rather than celebration.
- **Five regimes spread the data thin** in some states; rare regimes have small n and their IC estimates are not reliable. A coarser 2–3 regime model may trade descriptive richness for statistical robustness.
- **Defensive sector negative IC is a diagnostic, not a failure.** It indicates the growth/inflation-oriented feature set does not capture flight-to-safety dynamics. A separate signal source is needed for defensives.
- **Permutation p = 0.055** is borderline at the conventional 0.05 threshold. With more permutations (1,000+) and a longer OOS window, the statistical picture should sharpen.
- **Baseline covers 76 FRED series vs. the paper's 127.** The `gap_closing.py` script closes most of this coverage gap (reaching 104 transformed variables); residual differences remain due to currently unavailable FRED tickers.
- **Backward-looking.** Results describe historical statistical properties. They do not constitute a forecast of future returns or a trading recommendation.

---

## Development Note

Implementation leveraged AI-assisted coding tools. Research design, methodology selection, statistical interpretation, limitation analysis, and error audit were directed independently.

---

## Disclaimer

All data is from public sources (FRED, Yahoo Finance). No proprietary data, client information, or employer strategies are used at any stage. Views expressed are solely my own and do not represent the views, opinions, or positions of my employer or any affiliated entity. This repository is for research and educational purposes only and does not constitute investment advice, a trading signal, or a recommendation to buy, sell, or hold any security.

---

## License

[MIT](LICENSE) © 2026 Tzu-Wei (Aaron) Tseng.

The research paper replicated here (Oliveira et al., arXiv:2503.11499) is the property of its authors — please consult arXiv for its distribution terms.

---

## Citation

If you reference this replication:

```bibtex
@misc{tseng2026regime,
  author = {Tseng, Tzu-Wei (Aaron)},
  title  = {Regime-Conditioned Sector Allocation: An Independent Replication of arXiv:2503.11499},
  year   = {2026},
  url    = {https://github.com/klmtseng/regime-conditioned-allocation}
}
```
