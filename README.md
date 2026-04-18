# Regime-Conditioned Sector Allocation

Replication of a macro regime-conditioned sector-ETF allocation framework — and a decomposition showing where the signal actually lives (and where it doesn't).

📄 **Full write-up:** [Where Macro Regime Signals Actually Live](https://aarontsengquant.substack.com/p/where-macro-regime-signals-actually)
📚 **Paper replicated:** Oliveira et al., [arXiv:2503.11499](https://arxiv.org/abs/2503.11499)

---

## TL;DR

We replicated a regime-conditioned sector-ETF allocation framework, ran it through three independent validation milestones, and found that **the signal is real but narrower than headline numbers suggest**:

- It is **cross-sectional**, not time-series. The model ranks assets well but barely beats a random walk on point forecasts.
- It lives in **cyclical sectors** (Materials, Consumer Discretionary) and **transitional regimes** (Reflationary Boom, Economic Difficulty).
- It **actively misranks defensive sectors** (Staples, Utilities) — a diagnostic for a missing signal dimension.

This tells us where to trust the system, where not to, and exactly what to fix next.

---

## Key Findings

- **Headline performance.** In-sample Sharpe 0.966; out-of-sample Sharpe 1.645 (Jan 2023 onward). The OOS > IS gap is almost certainly a short-window artifact (~24 observations) and is treated as a data point, not a conclusion.
- **Cross-sectional alpha confirmed, point-forecast edge minimal.** OOS RMSFE ratio ≈ 0.99 vs. random walk — the model's value is in *ranking*, not in absolute return prediction.
- **Per-ETF Information Coefficient decomposition:**
  - Positive: XLB **IC 0.102**, XLY **IC 0.091** (cyclicals, where sector dispersion is widest)
  - Negative: XLP **IC −0.082**, XLU **IC −0.081** (defensives — structural miss, not noise)
- **Per-regime IC:** strongest during **Reflationary Boom (IC 0.118, n=32)**, weakest during sentiment-driven recoveries where slow-moving macro features lag.
- **Stress validation across GFC, European Debt Crisis, COVID, and 2022 Tightening:** regime-conditioned allocation behaves as a slow-burn drawdown protection, not a tail-risk hedge. Best on slow-building crises (GFC), weakest on exogenous shocks (COVID).
- **Permutation test:** observed Sharpe 1.424 vs. 95th-percentile shuffled 1.426, **p = 0.055** — borderline at the conventional threshold, complemented by the economically coherent IC decomposition.

---

## Methodology

**Five-stage pipeline:**

```
Stage A: Macro + market feature ingestion (vintage-aligned, no look-ahead)
Stage B: 6-regime classification (latent macroeconomic states)
Stage C: Ridge regression (closed-form, implicit shrinkage)
Stage D: Top-k ranking (deliberately naive — isolates signal quality)
Stage E: Governance gate (validation hooks for halt / modify)
```

**Data reduction:** 127 macroeconomic variables → 61 principal components, 48-month rolling estimation window, monthly frequency.

**Six regimes:** Economic Difficulty, Economic Recovery, Expansionary Growth, Stagflationary Pressure, Pre-Recession Transition, Reflationary Boom (descriptive labels for latent states, not forecasts).

**Validation:**

- 200-run permutation test (regime-label shuffling)
- Per-ETF and per-regime IC decomposition
- Stress windows across four historical crises
- Benchmark distance vs. random walk (IS and OOS)

---

## Data Sources (Public Only)

| Source | Coverage |
|---|---|
| FRED | 127 macroeconomic variables (industrial production, unemployment, CPI, yield curve, credit spreads, etc.) |
| Yahoo Finance | Sector ETF returns (XLB, XLY, XLK, XLF, XLE, XLI, XLP, XLU, XLV, XLRE, XLC) and realized volatility |

No proprietary data, client information, or employer strategies used.

---

## Reproduce

```bash
git clone https://github.com/klmtseng/regime-conditioned-allocation.git
cd regime-conditioned-allocation
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Five-stage pipeline
python scripts/01_ingest_macro.py       # FRED vintage-aligned pull
python scripts/02_regime_classify.py    # 6-regime latent states
python scripts/03_ridge_forecast.py     # per-regime rolling ridge
python scripts/04_topk_portfolio.py     # long-only sector allocation
python scripts/05_validation.py         # IC decomp + permutation + stress
```

Expected runtime: ~20–30 minutes on a laptop.

---

## Repo Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   └── main_analysis.ipynb
├── scripts/
│   ├── 01_ingest_macro.py
│   ├── 02_regime_classify.py
│   ├── 03_ridge_forecast.py
│   ├── 04_topk_portfolio.py
│   └── 05_validation.py
├── src/
│   ├── regimes.py            # 6-state classification
│   ├── forecasting.py        # rolling ridge, closed-form
│   ├── portfolio.py          # top-k long-only construction
│   ├── validation.py         # IC, permutation, stress windows
│   └── data_loaders.py
├── results/
│   ├── figures/
│   └── ic_tables/
└── docs/
    └── limitations.md
```

---

## Limitations

See [`docs/limitations.md`](docs/limitations.md) for the full list. Key points:

- **Short OOS window.** ~24 observations (Jan 2023 onward); confidence interval on Sharpe is wide (plausibly 0.8 to 2.4). The OOS > IS pattern deserves scrutiny rather than celebration.
- **Six regimes spread the data thin.** Pre-Recession Transition has only 2 observations; its high IC cannot be relied upon. A coarser 2–3 regime model may trade descriptive richness for statistical robustness.
- **Defensive sector negative IC is a diagnostic, not a failure.** It indicates the growth/inflation feature set does not capture the "flight to safety" dynamic. Defensive allocation needs its own signal source.
- **Permutation p = 0.055 is borderline** at the conventional 0.05 threshold. With 1,000+ permutations and a longer OOS window, the statistical picture should sharpen.
- **Status: CONDITIONAL GO** — usable as an overlay alongside other inputs, not as a standalone allocation engine.
- **Backward-looking.** Results describe historical statistical properties; not a prediction of future returns.

---

## Development Note

Implementation leveraged AI-assisted coding tools. Research design, methodology selection, statistical interpretation, limitation analysis, and error audit were directed independently.

---

## Disclaimer

All data is from public sources (FRED, Yahoo Finance). No proprietary data, client information, or employer strategies are used at any stage. Views expressed are solely my own and do not represent the views, opinions, or positions of my employer or any affiliated entity. This repository is for research and educational purposes only and does not constitute investment advice, a trading signal, or a recommendation to buy, sell, or hold any security.

---

## License

[MIT](LICENSE)

---

## Citation

If you reference this work:

```bibtex
@misc{tseng2026regime,
  author = {Tseng, Tzu-Wei (Aaron)},
  title = {Regime-Conditioned Sector Allocation: Where the Signal Actually Lives},
  year = {2026},
  url = {https://github.com/klmtseng/regime-conditioned-allocation}
}
```
