# M3 — Benchmark Distance Report

**Generated:** 2026-03-23 21:14
**Pipeline:** arXiv 2503.11499 Regime Detection
**Window:** 48 months rolling | **Sample:** 314 months

## 1. Design

For each of 10 sector ETFs, compute the Root Mean Squared Forecast Error (RMSFE) of:
- **Ridge**: Per-regime ridge regression on PCA macro features (main pipeline)
- **Naive**: Conditional regime Sharpe forecast (pipeline alternative)
- **RW**: Random walk (predict return = 0, martingale)
- **AR(1)**: Rolling autoregressive(1) on ETF returns

Relative RMSFE < 1.0 means the model improves on the baseline.

## 2. Full-Sample Results

| ETF | RMSFE Ridge | RMSFE RW | RMSFE AR1 | Ridge/RW | Ridge/AR1 |
|-----|------------|---------|----------|---------|----------|
| SPY | 0.042365 | 0.043328 | 0.043261 | **0.9778** | **0.9793** |
| XLB | 0.057397 | 0.057936 | 0.059270 | **0.9907** | **0.9684** |
| XLE | 0.076070 | 0.076184 | 0.079859 | **0.9985** | **0.9526** |
| XLF | 0.061027 | 0.061813 | 0.062190 | **0.9873** | **0.9813** |
| XLI | 0.053436 | 0.053784 | 0.054165 | **0.9935** | **0.9865** |
| XLK | 0.052183 | 0.053312 | 0.052851 | **0.9788** | **0.9874** |
| XLP | 0.035433 | 0.035450 | 0.035406 | **0.9995** | 1.0008 |
| XLU | 0.043583 | 0.042926 | 0.042371 | 1.0153 | 1.0286 |
| XLV | 0.039451 | 0.040379 | 0.040400 | **0.9770** | **0.9765** |
| XLY | 0.054418 | 0.054810 | 0.055315 | **0.9929** | **0.9838** |
| AGGREGATE | 0.051536 | 0.051992 | 0.052509 | **0.9912** | **0.9815** |

**Ridge beats RW:** 9/10 ETFs
**Ridge beats AR1:** 8/10 ETFs

## 3. Diebold-Mariano Test (Ridge vs RW)

Tests whether forecast accuracy difference is statistically significant.

| ETF | DM stat | p-value | Direction |
|-----|---------|---------|-----------|
| SPY | +1.5376 | 0.1241 | Ridge better |
| XLB | +0.5724 | 0.5670 | Ridge better |
| XLE | +0.1199 | 0.9046 | Ridge better |
| XLF | +1.0044 | 0.3152 | Ridge better |
| XLI | +0.5259 | 0.5990 | Ridge better |
| XLK | +1.3393 | 0.1805 | Ridge better |
| XLP | +0.0363 | 0.9711 | Ridge better |
| XLU | -1.2517 | 0.2107 | RW better |
| XLV | +1.0625 | 0.2880 | Ridge better |
| XLY | +0.5309 | 0.5955 | Ridge better |

Significant Ridge wins (p<0.10): 0/10

## 4. Out-of-Sample Results (Jan 2023 →)

| ETF | RMSFE Ridge | RMSFE RW | Ridge/RW | Ridge/AR1 |
|-----|------------|---------|---------|----------|
| SPY | 0.035409 | 0.037532 | **0.9434** | 1.0140 |
| XLB | 0.050520 | 0.051254 | **0.9857** | **0.9991** |
| XLE | 0.059159 | 0.056480 | 1.0474 | **0.9833** |
| XLF | 0.047027 | 0.047796 | **0.9839** | 1.0315 |
| XLI | 0.045912 | 0.046970 | **0.9775** | 1.0317 |
| XLK | 0.052043 | 0.056123 | **0.9273** | **0.9871** |
| XLP | 0.035617 | 0.035425 | 1.0054 | **0.9993** |
| XLU | 0.046794 | 0.046553 | 1.0052 | 1.0329 |
| XLV | 0.039276 | 0.038292 | 1.0257 | **0.9919** |
| XLY | 0.055882 | 0.056989 | **0.9806** | 1.0093 |

## 5. Verdict

**PASS**

Pass criteria:
- Aggregate Ridge/RW < 1.0: PASS (0.9912)
- Ridge beats RW on ≥5/10 ETFs: PASS (9/10)

## 6. Interpretation

The benchmark distance report measures whether the regime-conditional Ridge model
produces more accurate point forecasts than simple baselines. This complements the
Sharpe-based evaluation (which measures portfolio-level value) by examining the
forecast accuracy channel directly.

Key insights:
- RMSFE captures point forecast accuracy, not portfolio construction quality.
- A model can have higher RMSFE but still produce better portfolios if its ranking
  of ETFs is correct (IC > 0) even when magnitude is noisy.
- The Diebold-Mariano test accounts for serial correlation in forecast errors.

---
*Report generated 2026-03-23 21:14 by benchmark_distance_2503.py*