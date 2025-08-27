# Project Minotaur: Multi-Timeframe Crypto Signal Model (5 m / 15 m / 4 h)

## 1 · Project Overview

Minotaur is an end-to-end research pipeline for BTC/USDT that transforms raw OHLCV data on **5-minute, 15-minute, and 4-hour** intervals into more than 600 engineered features, then trains a **CNN-Transformer** classifier to predict if price will reach a 2 : 1 take-profit before an adaptive stop-loss.

*Historical benchmark*: a single-stack CNN-Transformer (Run 038) achieved **AUC ≈ 0.68** and **precision ≈ 0.60** (calibrated) on unseen test data. The current code base reproduces and extends that pipeline with a fully modular feature engine, Optuna hyper-parameter search, and isotonic probability calibration.

> **Note**  Earlier experiments with tick-level *dollar bars* are not part of the public workflow and are intentionally omitted from this README.

---

## 2 · Data Flow at a Glance

```
Binance BTC/USDT 5 m CSV  ─┐
                          │   ┌────────────┐
Binance BTC/USDT 15 m CSV ─┼──▶│ Feature    │──┐  5 m sequences  │
                          │   │ Engine V2  │  ├─ 15 m sequences │
Binance BTC/USDT 4 h CSV  ─┘   └────────────┘  └─ 4 h sequences  ▼
                                                 CNN-Transformer → Calibrated probs → Threshold → Signal
```

1. **Raw OHLCV ingestion** (`data_preparation_pipeline/`)
2. **Feature generation** (`NN_trading_pipeline/feature_engine_v2.py`)
3. **Dataset assembly & scaling** (`chimera_5_19_optuna.py`)
4. **Model training / Optuna search** (`model_training/minotaur_v1.py`)
5. **Isotonic regression calibration + threshold optimisation**

---

## 3 · Feature Engineering (FeatureEngine V2)

The engine is fully streaming-compatible; it can calculate features live bar-by-bar once a warm-up history is buffered.

### 3.1 Core OHLCV Inputs

| Timeframe | Columns |
|-----------|---------|
| 5 m       | Open, High, Low, Close, Volume |
| 15 m (3×) | aggregated from 5 m |
| 4 h (48×) | aggregated |

### 3.2 Technical Indicator Families

* **Traditional TA-Lib** (per timeframe)
  * SMA/EMA (5→500 periods)
  * RSI, ATR/NATR, ADX, MACD, StochK/D, BBands, OBV, MFI, CCI
* **Rolling Stats**
  * Rolling mean / stdDev of log-returns
  * Percent rank / z-score
* **Volatility Metrics**
  * `volatility_N_tf` = stdDev(log-returns, _N_)
  * Multi-time-frame (MTF) volatility ratios
* **Divergence Detection** (`divergence_calculator.py`)
  * Bull / bear **RSI-14 divergences** on 15 m & 4 h (RSI 30/70 classic OR 40/60 "robust" thresholds)
* **Trend Regimes**
  * `Trend4H_State`: price vs SMA-200 on 4 h
  * Generic `TrendX_State` for arbitrary X (e.g., 15 m, 1 h)
* **Volatility & Volume Regimes**
  * 3-state quantile bucketing of ATR & Volume (`Vol5m_State`, `Volume5m_State`)
* **Candlestick Patterns**
  * TA-Lib CDL\, one-hot encoded per 5 m bar
* **Time Features**
  * Hour of day, day of week (cyclic sine/cosine pairs)

### 3.3 Feature Selection

A separate RF-importance script ranks features; the top-N list (100 by default) is read by the training script for rapid experimentation. Zero-variance & high-VIF columns are pruned automatically.

---

## 4 · Target Definition

Adaptive stop-loss / take-profit computed **per entry bar**:

```python
sl_dist = max(min_sl_pct * close, atr_multiplier * atr(close, atr_period))
tp_dist = reward_risk_ratio * sl_dist  # default 2.0

label = 1 if future_high ≥ close + tp_dist before future_low ≤ close − sl_dist else 0
```

Typical parameters: `min_sl_pct=1%`, `atr_multiplier=1.5`, `reward_risk_ratio=2.0`.
Class 1 frequency ≈ 33 % for BTC 2017-2024.

---

## 5 · Model Architecture

```
Input  (B, 60, F) ───────── Conv1D×3 (GLU-GELU, residual) ─┐
                                                           ↓
                                     Strided Conv (learned down-sampling)
                                                           ↓
                                 Feature-wise gating (Dense sigmoid ⊙)
                                                           ↓
                Positional Encoding + d_model projection (Dense  → 128-256)
                                                           ↓
            Transformer Encoder × {1-3} (heads 4-8, ff_dim 2-4×d_model)
                                                           ↓
                    AttentionPooling (learn weights across sequence)
                                                           ↓
                       MLP Head (Dense-GELU-Dropout) × {1-2}
                                                           ↓
                               Output sigmoid (1-unit)
```

* **Optimizer**: AdamW + linear warm-up (3 epochs) → cosine decay.
* **Loss**: Binary Cross-Entropy **or** Focal-Loss (Optuna-tunable α, γ).
* **Regularisation**: L2, dropout (CNN, Transformer, MLP), gradient clipping (`clipnorm=1`).
* **Probability Calibration**: Isotonic regression on validation predictions.

---

## 6 · Hyper-Parameter Optimisation (Optuna)

| Group          | Search Space (examples) |
|----------------|-------------------------|
| `learning_rate`| 1e-6 – 6e-5 (log-uniform)|
| `weight_decay` | 1e-5 – 8e-4             |
| CNN layers     | 1 – 3 layers × filters{16-128}, kernel{3,5,7}, dilation{1-4}|
| Transformer    | blocks{1-3}, heads{4-8}, head_size{32-64}, ff_dim_factor{2-4}|
| Dropouts       | 0.05 – 0.30             |
| Batch size     | 32 or 64               |
| Focal α/γ      | 0.3 – 0.7 / 1.0 – 3.0    |

Trials are stopped early via `val_auc` plateau & `best_val_f1` stagnation.

---

## 7 · Performance Snapshot

| Run ID | Dataset | Features | SeqLen | Val AUC | Calibrated Test AUC | Precision | Recall | F1 |
|--------|---------|----------|--------|---------|---------------------|-----------|--------|----|
| **038**| 5 m base (legacy) | 88 | 60 | 0.71 | **0.68** | 0.60 | 0.28 | 0.38 |
| 2025-05 Optuna best | 5m+15m+4h top-100 | 100 | 60 | 0.56 | 0.54 | 0.47 | 0.31 | 0.37 |

> *Performance varies with market regimes; above table is indicative.*

---

## 8 · Quick Start

```bash
pip install -r requirements.txt  # TensorFlow 2.15+, TA-Lib, Optuna, Dask …

# 1. Create parquet features (offline batch example)
python NN_trading_pipeline/main_pipeline.py \
  --ohlcv-5m data/binance/BTCUSDT_5m.csv \
  --ohlcv-15m data/binance/BTCUSDT_15m.csv \
  --ohlcv-4h data/binance/BTCUSDT_4h.csv \
  --out-parquet data/features/BTCUSDT_full.parquet

# 2. Train a single model (no Optuna)
python model_training/minotaur_v1.py \
  --no-optuna --epochs 50 --feature-parquet data/features/BTCUSDT_full.parquet

# 3. Hyper-parameter sweep (50 trials)
python model_training/minotaur_v1.py \
  --n-trials 50 --epochs 100 --feature-parquet data/features/BTCUSDT_full.parquet \
  --optuna-study-name minotaur_sweep_May25
```

---

## 9 · Research Directions

* **Mamba state-space models** for ultra-long sequences
* **Cross-asset feature transfer** (ETH, SOL)
* **On-chain metrics** integration
* **Reinforcement-learning overlay** for position sizing

---

## 10 · License & Disclaimer

This repository is released for educational and research purposes only. It is **not** financial advice. Trading cryptocurrencies is highly speculative. Use at your own risk.
