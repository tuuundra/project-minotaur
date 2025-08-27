# Project Minotaur: High-Frequency BTC Trading with Dollar Bars & CNN-Transformer Architecture

## Overview

Project Minotaur is an end-to-end machine learning pipeline for high-frequency cryptocurrency trading, specifically designed for BTC/USDT. The system processes tick-level data through a sophisticated multi-resolution feature engineering pipeline, culminating in a CNN-Transformer deep learning model optimized for predicting profitable trading opportunities.

**Key Achievement**: Historical runs have demonstrated promising performance with test AUC scores reaching **~0.68** and precision of **~0.60** on 2:1 reward-to-risk targets, indicating potential profitability for systematic trading strategies.

## Architecture & Innovation

### 1. Data Processing Pipeline

The system employs a novel **Dollar Bar** approach rather than traditional time-based bars, addressing the fundamental problem of information loss in fixed-time aggregation:

#### Phase 1: Tick Data Cleaning
- Converts raw CSV tick data to optimized Parquet format
- Standardizes timestamps to UTC with nanosecond precision
- Ensures consistent schema: `timestamp`, `price`, `quantity`, `quote_quantity`, `isBuyerMaker`

#### Phase 2: Dollar Bar Generation
- **Innovation**: Aggregates ticks based on fixed dollar volume ($2M USDT threshold) rather than time
- **Benefits**: Each bar represents consistent economic activity, better handling BTC's volatility
- **Microstructure Features**: Calculates advanced intra-bar metrics:
  - Trade imbalance (buy vs sell pressure)
  - Tick price volatility within bars
  - Taker buy/sell ratios
  - Price skewness and kurtosis
  - Directional change counts

#### Phase 3: Multi-Resolution Feature Engineering
The `MinotaurFeatureEngine` creates features across three temporal resolutions:
- **Short bars**: Base $2M dollar bars (~2-5 minutes typical duration)
- **Medium bars**: Aggregated from 10x short bars (~20-50M volume)
- **Long bars**: Aggregated from 16x medium bars (~320M+ volume)

**Feature Categories**:
- **Technical Indicators**: 15+ TA-Lib indicators (RSI, MACD, ATR, Bollinger Bands, etc.) across all resolutions
- **Price Dynamics**: Log returns, volatility measures, price momentum
- **Volume Profile**: Point of Control (POC), Value Area High/Low for multiple timeframes
- **Market Microstructure**: Order flow imbalance, trade intensity metrics
- **Temporal Features**: Cyclical time encoding (hour, day of week)
- **Regime Detection**: Bull/bear/choppy market classification

### 2. Time-Based Bars Integration
Parallel processing of traditional time bars (1-minute, 15-minute, 4-hour) provides complementary perspective:
- **Volume Profile Analysis**: POC, VAH, VAL calculations across multiple windows
- **Stochastic Oscillator Flags**: Overbought/oversold regime detection
- **Divergence Analysis**: Price-momentum divergences across timeframes

### 3. Feature Consolidation
The final dataset merges multi-resolution dollar bars with time-based features, resulting in:
- **Scale**: ~5.4M samples, 600+ features (before selection)
- **Coverage**: August 2017 - April 2024 (7+ years of BTC history)
- **Alignment**: All features timestamp-aligned using sophisticated merge strategies

## Model Architecture

### CNN-Transformer Hybrid Design

The model combines the pattern recognition strengths of CNNs with the sequence modeling capabilities of Transformers:

```
Input: (batch_size, sequence_length=60, features=~100-600)
    ↓
[CNN Stack - 3 Layers]
    Conv1D → BatchNorm → SpatialDropout1D
    Filters: [64, 64, 64] | Kernels: [3, 5, 7] | Dilations: [1, 2, 3]
    Activation: GELU with Gated Linear Units (GLUs)
    Residual Connections: ResNet-style skip connections
    ↓
[Learned Downsampling]
    Strided Conv1D (kernel=2, stride=2) replaces MaxPooling
    ↓
[Feature Gating Mechanism]
    Dense(sigmoid) → Element-wise gating of feature maps
    ↓
[Projection Layer]
    Dense layer to match Transformer d_model dimension
    ↓
[Positional Encoding]
    Sinusoidal position embeddings
    ↓
[Transformer Encoder Blocks - 1-3 layers]
    Pre-Layer Normalization
    Multi-Head Attention (4-8 heads, causal masking)
    Feed-Forward Networks (GELU activation)
    Residual connections & dropout
    ↓
[Attention Pooling]
    Learned attention weights for sequence aggregation
    (replaces GlobalAveragePooling1D)
    ↓
[MLP Head - 1-2 layers]
    Dense(32-128) → GELU → Dropout → Dense(1, sigmoid)
```

### Advanced Training Techniques

#### Hyperparameter Optimization with Optuna
- **Search Space**: 15+ hyperparameters optimized simultaneously
- **Key Parameters**:
  - Learning rate: 1e-6 to 6e-5 (log scale)
  - CNN filters, kernels, dilations per layer
  - Transformer heads (1-8), head size (32-64)
  - Dropout rates: CNN (0.05-0.3), Transformer (0.1-0.25)
  - Focal Loss parameters (α: 0.3-0.7, γ: 1.0-3.0)

#### Loss Function & Optimization
- **Focal Loss**: Addresses class imbalance (~33% positive class)
- **AdamW Optimizer**: Weight decay regularization
- **Learning Rate Schedule**: Linear warmup + Cosine decay
- **Regularization**: L2 regularization, multiple dropout layers

#### Advanced Callbacks
- **F1EvalCallback**: Real-time F1 optimization with threshold tuning
- **Early Stopping**: Prevents overfitting based on validation F1
- **Model Checkpointing**: Saves best models based on validation AUC

#### Probability Calibration
- **Isotonic Regression**: Post-hoc calibration for reliable probability estimates
- **Threshold Optimization**: Fine-tuned decision boundaries on calibrated probabilities
- **Performance**: Critical for real-world trading applications

## Target Definition & Strategy

### Adaptive Stop-Loss/Take-Profit System
The model predicts binary outcomes based on sophisticated target calculation:

```python
# Adaptive target calculation
stop_loss = max(min_sl_pct * entry_price, atr_multiplier * atr_period)
take_profit = 2.0 * stop_loss  # 2:1 reward-to-risk ratio

target = 1 if price_hits_tp_before_sl else 0
```

**Parameters**:
- `min_sl_pct`: 1.0% (minimum stop-loss)
- `atr_period`: 14 (volatility calculation window)
- `atr_multiplier`: 1.5 (volatility-based stop scaling)
- `reward_risk_ratio`: 2.0 (risk-adjusted returns)

## Performance Results

### Historical Benchmark (Run 038)
- **Test AUC**: 0.682
- **Test Precision**: 0.596 (at optimized threshold)
- **Test Recall**: 0.280
- **Test F1**: 0.383

### Recent Minotaur Results
Recent experiments with the full pipeline have shown:
- **Validation AUC**: 0.55-0.56 range
- **F1 Scores**: 0.35-0.50 range
- **Feature Count**: Successfully reduced from 611 to 100 top features using RandomForest importance

### Performance Analysis
Based on 2:1 reward-to-risk ratio and historical metrics:
- **Expected Value per Signal**: ~$81 (on $100 risk per trade)
- **Estimated Annual Signals**: ~280 (on test data)
- **Theoretical Annual Profit**: ~$22,700 (before costs/slippage)

*Note: These are theoretical estimates for model comparison, not guaranteed trading results.*

## Repository Structure

```
minotaur/
├── scripts/
│   ├── phase1_clean_ticks.py          # Tick data preprocessing
│   ├── phase2_generate_dollar_bars.py  # Dollar bar generation
│   ├── phase3_process_historical_features.py  # Feature engineering
│   ├── minotaur_feature_engine.py     # Core feature engine
│   ├── consolidate_features.py        # Multi-resolution merging
│   └── time_based_processing/         # Time bar processing
│       ├── generate_time_bars.py
│       ├── calculate_time_bar_features.py
│       ├── add_volume_profile.py
│       └── add_stochastic_flags.py
├── model_training/
│   └── minotaur_v1.py                 # Main training script with Optuna
├── research/                          # Technical documentation
└── research2/                         # Advanced architecture research
```

## Quick Start

### Prerequisites
```bash
# Python 3.8+
pip install tensorflow pandas numpy scikit-learn
pip install optuna dask pyarrow
pip install TA-Lib  # See platform-specific instructions
pip install volprofile  # For volume profile features
```

### Basic Usage

1. **Process Tick Data**:
```bash
python scripts/phase1_clean_ticks.py --input-dir raw_ticks/ --output-dir data/cleaned_tick_data/
```

2. **Generate Dollar Bars**:
```bash
python scripts/phase2_generate_dollar_bars.py --input-dir data/cleaned_tick_data/ --threshold 2000000
```

3. **Calculate Features**:
```bash
python scripts/phase3_process_historical_features.py --dollar-bars-dir data/dollar_bars/2M/
```

4. **Train Model (Single Run)**:
```bash
python model_training/minotaur_v1.py --no-optuna --epochs 50 \
  --feature-parquet data/consolidated_features_targets_all.parquet
```

5. **Hyperparameter Optimization**:
```bash
python model_training/minotaur_v1.py --n-trials 50 --epochs 100 \
  --feature-parquet data/consolidated_features_targets_all.parquet \
  --optuna-study-name "minotaur_optimization"
```

## Advanced Features

### Feature Selection
- **RandomForest Importance**: Automated feature ranking and selection
- **Multicollinearity Removal**: VIF-based redundancy elimination
- **Domain Knowledge**: Manual curation of financially meaningful features

### Model Variants
- **Single-stack CNN-Transformer**: Current production architecture
- **Multi-branch CNN**: Experimental separate processing of feature groups
- **Attention Mechanisms**: Grouped Query Attention, standard Multi-Head Attention

### Integration Options
- **Weights & Biases**: Optional experiment tracking (set `--use-wandb`)
- **TensorBoard**: Built-in training visualization
- **Custom Callbacks**: Extensible training pipeline

## Data Requirements

### Input Format
Tick data should follow this schema:
```
timestamp,price,quantity,quote_quantity,isBuyerMaker
2023-01-01 00:00:00.123456,16500.50,0.1,1650.05,true
```

### Storage Recommendations
- **Tick Data**: ~50GB for 7 years BTC/USDT
- **Dollar Bars**: ~500MB for processed bars
- **Features**: ~1-5GB for full feature set
- **Models**: ~10-50MB per trained model

## Research & Development

### Current Investigations
- **Mamba Architecture**: State-space models for sequence processing
- **Volume Profile Integration**: Enhanced market microstructure features
- **Multi-timeframe Fusion**: Optimal combination of dollar and time bars

### Performance Optimization
- **Dask Integration**: Distributed feature processing
- **Memory Management**: Chunked processing for large datasets
- **GPU Acceleration**: TensorFlow GPU support for training

## License & Disclaimer

This project is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred from using this software.

## Citation

If you use this work in your research, please cite:
```bibtex
@misc{minotaur2024,
  title={Project Minotaur: High-Frequency Cryptocurrency Trading with Dollar Bars and CNN-Transformer Architecture},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[username]/project-minotaur}
}
```

## Contributing

This is a research project. For questions or collaboration opportunities, please open an issue or submit a pull request.

---

*Built with TensorFlow, Optuna, and a passion for quantitative finance.*
