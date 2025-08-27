Below is an expanded, “action-oriented” rewrite of every major finding that appears in your EDA summary, the follow-on meta-analysis, the trading-video transcript review, and the project README.
For each point I first restate the key takeaway (“What we learned”), then spell out **why it matters** for Minotaur’s CNN-Transformer pipeline, and finally give one or more **concrete next steps** you can slot into Phases 3-4 (feature engineering → model training).

---

## 1.  Data integrity & structure

**What we learned** – Columns and dtypes are correct; NaNs cluster at the front (indicator look-backs) and the tail (`target_long` unresolved).&#x20;

**Why it matters** – NaNs are predictable and mostly benign, but any learner that shuffles rows or is sensitive to missing values (e.g., tree models without native NaN support) will misbehave.

**Next steps**

* Keep your current “start-training-after-lookback” rule; explicitly `dropna()` rows where every resolution has NaNs.
* For the trailing unresolved rows, either drop or label with special class `-1` and mask them from the loss if you decide to keep them for online inference tests.

---

## 2.  Target variable imbalance (`≈67 % SL vs 33 % TP`)

**Why it matters** – Skews threshold-based metrics, and class-weight heuristics may be unstable because imbalance changes across market regimes.&#x20;

**Next steps**

* **Stick with focal-loss** (already default True in `minotaur_v1.py`) but tune `(α, γ)` during Optuna rather than hand-setting 0.25/2.0.
* Log and monitor per-epoch **precision-recall curves**; use the custom `SafeF1Score` to early-stop on minority-class recall, not just AUC.
* Consider *balanced bagging* for the baseline tree model you’ll build in 4.4, to double-check neural results.

---

## 3.  Numerical feature distributions

| Feature family                                                         | Takeaway                                                                                                     | Action                                                                                       | Rationale |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- | --------- |
| **Log returns** – fat-tailed, slight positive skew                     | Keep raw for tree models; for NN branch continue Robust scaling (already in CLI).                            | Scaling protects gradients; trees like the raw signal.                                       |           |
| **Dollar volume / BTC volume** – extreme right tails; 7 % bars ≥ \$3 M | Clip at **99.5 %** before scaling (flagged by `--clip-features`).                                            | Preserves genuine spikes but stops a few monster bars from dominating batch-norm statistics. |           |
| **ATR, ADX, intra-bar vol/kurtosis** – heavy skew                      | Add **log1p transform** before Robust scaling; keep original as extra channel so nets can learn both scales. | Log stabilises variance; dual-channel sometimes outperforms single transform.                |           |
| **Trade imbalance** – centred, leptokurtic                             | Good as-is; but add **rolling mean(10)** & **rolling std(10)** per resolution.                               | Captures persistence of buy/sell pressure that single-bar view misses.                       |           |

(The detailed stats for each are in the meta-analysis)&#x20;

---

## 4.  Categorical / time features

* **Hour-of-day** shows classic liquidity hump (12-16 UTC).
* **Day-of-week** weaker on weekends.&#x20;

**Next steps**

1. Convert both to *sin/cos cyclical* pairs **and** keep the raw integer – the engine can concatenate all three.
2. Create a boolean **`is_peak_session`** (12-16 UTC) and **`is_weekend`** flags; these can interact with volatility to allow regime-aware attention.

---

## 5.  Correlations & multicollinearity

**What we learned** – Very strong within-family correlations; linear correlations with `target_long` are all |ρ| < 0.015 (typical for crypto).&#x20;

**Actions already taken** – You pruned 25 >0.95-ρ features in `phase4a_prune_features.py`.

**Additional next steps**

* For NN: leave remaining redundancy; attention can learn which lag of RSI matters.
* For baseline logistic/XGBoost: run a second Variance Inflation Factor (VIF) pass after scaling.
* Add **interaction features** flagged by o3 (e.g., `trade_imbalance × intra_bar_vol`). Because raw correlations are tiny, nonlinear combos often unlock signal.

---

## 6.  Bivariate vs Target

Distributions for TP vs SL mostly overlap; `trade_imbalance` shows the *only* visible shift (TP slightly less negative).&#x20;

**Implication** – Single features won’t separate outcomes; model must exploit *temporal context* + *cross-feature interactions*.

**Next steps**

* Increase **sequence\_length** hyper-range in Optuna to {60, 90, 120}. Longer look-back lets Transformer pick up multi-bar build-ups before the decision bar.
* Keep **causal masking** (already on) so leakage doesn’t creep in.

---

## 7.  Outliers

Log-returns \~1 % per tail; volume 7 % right-tail; intra-bar vol 6 % “outliers” above tiny bound; MACD has 1.2 % extreme bars.&#x20;

**Next steps**

* Activate `--clip-features` with `(0.5 %, 99.5 %)` quantiles – values beyond are clipped *and* a “was\_clipped” boolean is appended, giving the model a flag that “something extreme just happened”.
* For trees, use un-clipped data but set `min_child_weight` high to dampen micro-splits on outliers.

---

## 8.  Insights from trading-video transcripts (feature inspiration)

The transcript digest surfaces concepts not yet in your feature set: VWAP traps, psychological levels, supply-demand zones, liquidity sweeps, session windows, etc.

### High-leverage feature additions

| Theme                           | Concrete feature                                                                                | Implementation hint                                                       |         |                             |
| ------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------- | --------------------------- |
| **VWAP interactions**           | Distance to rolling intraday VWAP (short res); boolean “reclaimed VWAP in last N bars”          | Compute rolling dollar-volume weighted price inside the short-bar stream. |         |                             |
| **Volume anomaly**              | `bar_volume / SMA(bar_volume,20)`; encode spike if > 3                                          | Already have volume; just add ratio column.                               |         |                             |
| **Liquidity sweep / stop-hunt** | Boolean if current low < previous N-bar swing low by \<X ATR *and* bar closes back inside range | Use rolling min & ATR.                                                    |         |                             |
| **Psychological round numbers** | Distance to nearest `$1000` increment; flag if                                                  | dist                                                                      | < 0.2 % | Easy integer math on price. |
| **Session regime**              | `session_id` categorical: Asia/London/NY based on UTC hour; plus “inside-Asian-range” boolean   | Hour mapping table.                                                       |         |                             |

These can be produced inside `minotaur_feature_engine.py` so they are available at every resolution.

---

## 9.  README pipeline alignment

Your three-phase dollar-bar pipeline is sound and already memory-safe. Two places to weave in new insights:

1. **Phase 3 Feature Engine** – Add the items in §8 and the transforms in §§3-4 by expanding the engine’s config dictionary.
2. **Phase 4 model training**

   * Use **walk-forward splits** (e.g., train 2018-22 → val 2023Q1-Q2 → test 2023Q3-2024Q1) to keep distribution shift realistic.
   * Start with a **LightGBM** baseline (handles categorical via one-hot, NaNs natively) before the CNN-Transformer, to gauge marginal NN lift.

(Full pipeline description is in the README)

---

## 10.  Quick win checklist (put into your project board)

| ✅ / ⏳ | Task                                                       | Owner           | Phase |
| ----- | ---------------------------------------------------------- | --------------- | ----- |
| ⏳     | Add log1p + dual channel for skewed features               | feature\_engine | 3     |
| ⏳     | Implement VWAP-distance & reclaim flag                     | feature\_engine | 3     |
| ⏳     | Implement “was\_clipped” indicator after quantile clipping | feature\_engine | 3     |
| ⏳     | Add focal-loss α, γ to Optuna search space                 | model\_training | 4     |
| ⏳     | Extend sequence\_length choices to 90/120                  | model\_training | 4     |
| ⏳     | Build LightGBM baseline with walk-forward split            | experiments     | 4     |

---

### Final thoughts

Most individual features show little linear signal – exactly what we expect in high-frequency crypto. The game now is **temporal context + cleverly crafted interaction flags** that embody the market-structure traps highlighted by professional traders. Folding the new VWAP/volume/psych-level detectors into your multi-resolution dollar-bar engine, while keeping robust scaling and careful outlier handling, gives the CNN-Transformer the best chance to surface those subtle patterns.
