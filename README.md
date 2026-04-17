# Regime-Conditioned Sector Allocation

Independent replication of a macro regime-conditioned sector-ETF allocation framework — and a decomposition showing *where* the signal actually lives (and where it doesn't).

📄 **Full write-up:** [Where Macro Regime Signals Actually Live](https://aarontsengquant.substack.com/p/where-macro-regime-signals-actually)
📚 **Paper replicated:** Oliveira et al., *Tactical Asset Allocation with Macroeconomic Regime Detection* ([arXiv:2503.11499](https://arxiv.org/abs/2503.11499))

---

## TL;DR

I replicated a regime-conditioned sector-ETF allocation pipeline, ran it through three independent validation milestones (IC decomposition, permutation testing, benchmark distance), and found that the signal is real but narrower than headline numbers suggest:

- It is **cross-sectional**, not time-series — the model ranks ETFs well but barely beats a random walk on point forecasts (OOS RMSFE ratio ≈ 0.99).
- It lives in **cyclical sectors** (Materials XLB, Consumer Discretionary XLY) and **transitional regimes** (Reflationary Boom, Economic Difficulty).
- It **actively misranks defensive sectors** (Consumer Staples XLP, Utilities XLU) — a diagnostic for a missing signal dimension, not just noise.

See the write-up linked above for the full narrative. Headline numbers below.

---

## Key Findings

- **In-sample Sharpe:** 0.966 — consistent with a real but modest signal.
- **Out-of-sample Sharpe:** 1.645 (≈ 2-year window from Jan 2023; wide confidence interval, treat as a data point rather than a conclusion).
- **Information Coefficient by ETF:** XLB **+0.102**, XLY **+0.091** (strongest positive); XLP **−0.082**, XLU **−0.081** (structural negative — defensive misranking).
- **Per-regime IC:** strongest during Reflationary Boom (**+0.118**, n=32), weakest during sentiment-driven recoveries.
- **Permutation test (n=200):** observed Sharpe 1.424 vs. shuffled 95th percentile 1.426, **p ≈ 0.055** — borderline at the conventional threshold; the IC decomposition provides complementary economic-coherence evidence.
- **Benchmark distance (OOS):** RMSFE ratio vs. random walk ≈ **0.99** — the edge is in *ranking*, not in point forecasts.

---

## Methodology

**Pipeline:** FRED-MD → PCA → two-layer K-means → Markov regime forecasting → Ridge regression → top-k long-only portfolio → performance evaluation.

**Core design choices:**

- **Rolling window:** 48 months, walk-forward, strict causal — all fitting (scaler, PCA, K-means, ridge coefficients) uses only data up to t−1.
- **Regimes:** 5 latent states identified via two-layer K-means (elbow method confirms k=5).
- **Universe:** 10 sector ETFs (SPY, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY).
- **Macro features:** ~76 FRED series spanning output, labor, consumption, housing, money/credit, rates, prices, sentiment, and FX, with transformation codes following FRED-MD conventions.
- **Portfolio rule:** deliberately naive — top-k ranking by forecasted return, equal weight, long only. This isolates signal quality from optimizer cleverness.

**Validation milestones:**

| Milestone | Script | Purpose |
|---|---|---|
| M1 | `ic_validation.py` | Per-ETF, per-regime, rolling Spearman IC |
| M2 | `negative_control.py` | Cross-sectional permutation test (200 runs, seeded) |
| M3 | `benchmark_distance.py` | RMSFE vs. Random Walk and AR(1) baselines |
| Stress | `stress_validation.py` | Walk-forward across GFC, Euro Debt, COVID, 2022 Tightening |
| Type I/II | `type_error_tradeoff.py` | Crisis detection confusion matrix across thresholds |
| Gap closing | `gap_closing.py` | Extends FRED coverage toward the paper's 127 variables |

---

## Data Sources (Public Only)

| Source | Coverage |
|---|---|
| FRED (via `pandas_datareader`) | ~76 macroeconomic series from 1959 onward |
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
pip install -r requirements.txt

# Run the core pipeline (downloads data, fits regime model, generates outputs)
python regime_pipeline.py
```

Expected runtime: ~5–10 minutes on a laptop (FRED downloads dominate).

All scripts write to `outputs/` in the repo root. The committed `outputs/` folder contains the reference results from the write-up — re-running will overwrite with your freshly-computed results.

### Dependency strategy

- **`requirements.txt`** — 8 direct dependencies pinned to versions verified end-to-end. Transitive dependencies resolve automatically, which keeps installs working across platforms and minor CA/SSL bundle updates.
- **`requirements-lock.txt`** — full `pip freeze` snapshot from the verification environment. Use this if you want byte-for-byte reproducibility rather than just a working install.

## Full Validation Suite

After the core pipeline has completed (so the shared state is populated), the validation scripts can be run in any order:

```bash
python ic_validation.py          # per-ETF, per-regime IC decomposition
python negative_control.py       # 200-run permutation test
python benchmark_distance.py     # RMSFE vs. Random Walk / AR(1)
python stress_validation.py      # historical crisis windows
python type_error_tradeoff.py    # Type I / Type II error analysis
python gap_closing.py            # extended FRED coverage run
```

Each script loads FRED and ETF data through `regime_pipeline`'s helper functions, so no additional configuration is needed.

---

## Repository Layout

```
.
├── README.md
├── LICENSE                       MIT
├── .gitignore
├── requirements.txt              8 direct dependencies, pinned
├── requirements-lock.txt         Full frozen environment (optional)
├── regime_pipeline.py            Core pipeline (data → regimes → ridge → portfolio)
├── ic_validation.py              M1: Information Coefficient decomposition
├── negative_control.py           M2: Cross-sectional permutation test
├── benchmark_distance.py         M3: RMSFE vs. baselines
├── stress_validation.py          Walk-forward historical crisis windows
├── type_error_tradeoff.py        Crisis detection Type I / II analysis
├── gap_closing.py                Extended FRED variable coverage
└── outputs/                      Reference results from the write-up
    ├── *.csv                     Tables (IC per ETF, per regime, benchmark distances, ...)
    ├── *_report.md               Methodology + findings reports per milestone
    └── *.png                     Figures (cumulative returns, regime timeline, permutation, stress, ...)
```

---

## Limitations

Stated explicitly because overstated results help nobody:

- **Short OOS window.** Roughly 24 monthly observations; Sharpe confidence interval is wide (plausibly 0.8 to 2.4). The OOS > IS pattern deserves scrutiny rather than celebration.
- **Five regimes spread the data thin** in some states; rare regimes have small n and their IC estimates are not reliable. A coarser 2–3 regime model may trade descriptive richness for statistical robustness.
- **Defensive sector negative IC is a diagnostic, not a failure.** It indicates the growth/inflation-oriented feature set does not capture flight-to-safety dynamics. A separate signal source is needed for defensives.
- **Permutation p = 0.055** is borderline at the conventional 0.05 threshold. With more permutations (1,000+) and a longer OOS window, the statistical picture should sharpen.
- **Covers ~76 FRED series vs. the paper's 127.** The `gap_closing.py` script attempts to close this coverage gap; residual differences remain.
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
