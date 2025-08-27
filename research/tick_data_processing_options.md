# Tick Data Processing & Representation Strategies

This document outlines alternative strategies for processing raw tick-by-tick trade data, moving beyond or supplementing traditional fixed-time OHLCV bars. The goal is to explore data representations that may better capture market dynamics and provide richer inputs for advanced trading models like CNN-Transformers.

## The Fundamental Question: To Bar or Not To Bar?

Before diving into different *types* of bars, it's crucial to address a more fundamental question: **Why use bars at all?**

The traditional use of bars (OHLCV, whether time-based or event-driven) stems from several motivations:

*   **Noise Reduction & Dimensionality Reduction:** Raw tick data is extremely granular and can be noisy. Bars aggregate these events, smoothing idiosyncrasies and providing a condensed summary. This reduces data volume and complexity for traditional models.
*   **Standardized Feature Engineering:** A vast body of financial technical analysis is built upon bar data (Open, High, Low, Close, Volume). Bars allow straightforward application of these established indicators.
*   **Defining a "Market State":** Bars offer discrete snapshots, historically convenient for defining a market state upon which trading decisions are made.

However, the primary drawback of *any* aggregation (including all bar types) is **potential information loss**. The precise sequence, timing, and path of trades *within* a bar are summarized, and nuances can be lost.

This leads to two main paths for processing tick data:

1.  **Aggregation into Bars (Event-Driven or Fixed-Time):** Aims to create meaningful, information-rich summaries of market activity over defined "quanta" (of time, trades, volume, etc.). The challenge is to choose/design bars that retain the most relevant information for the modeling task.
2.  **Direct Sequential Modeling of Ticks:** Aims to feed raw or minimally processed tick data directly to sequence-aware models, allowing the model itself to learn optimal features and patterns from the high-frequency stream, thereby minimizing pre-processing assumptions and information loss.

The choice between these approaches (or a hybrid) is a core research question, depending on model capabilities, computational resources, and the nature of the signals being sought.

## 1. Event-Driven Bar Types

Event-driven bars are constructed based on market activity rather than fixed time intervals. This allows them to adapt to changing market conditions. They represent a sophisticated form of aggregation.

### 1.1. Tick Bars

*   **Concept:** A new bar is formed after a predetermined number of trades (ticks) have occurred.
*   **Example:** A new bar is created every 500 trades.
*   **Behavior:** 
    *   During high market activity (many trades), tick bars will have a shorter time duration.
    *   During low market activity, tick bars will span a longer time duration.
*   **Potential Benefit:** Each bar represents a consistent amount of "transactional intensity" or "market participation events." This can help normalize for periods of high vs. low trading frequency.
*   **Considerations/Trade-offs:**
    *   Still an aggregation; intra-bar tick sequence information is summarized.
    *   The optimal number of ticks per bar is a hyperparameter that may vary by asset or market regime.

### 1.2. Volume Bars

*   **Concept:** A new bar is formed once a predetermined amount of the asset (e.g., BTC) has been traded.
*   **Example:** A new bar is created every 100 BTC traded.
*   **Behavior:** Bars complete more quickly when large trades occur or when a high volume of smaller trades accumulates to the target threshold.
*   **Potential Benefit:** Each bar represents a consistent amount of "market commitment" in terms of the actual quantity of the asset changing hands. This can highlight periods of significant accumulation or distribution.
*   **Considerations/Trade-offs:**
    *   Still an aggregation.
    *   A single very large trade can complete a bar quickly, potentially masking the behavior of smaller trades within that volume quantum.
    *   Optimal volume threshold is a hyperparameter.

### 1.3. Dollar (Quote Volume) Bars

*   **Concept:** A new bar is formed once a predetermined monetary value of the asset (e.g., USDT) has been traded.
*   **Example:** A new bar is created every $1,000,000 USDT worth of BTC traded.
*   **Behavior:** These bars normalize for price fluctuations. In a high-price environment, a smaller quantity of BTC would constitute $1M, whereas in a low-price environment, a larger quantity would be needed.
*   **Potential Benefit:** Each bar represents a consistent level of "economic significance" or "capital flow." This can be more stable than volume bars during periods of high price volatility.
*   **Considerations/Trade-offs:**
    *   Still an aggregation.
    *   Normalizes for price, which is good, but the "meaning" of a fixed dollar amount can change with overall market capitalization or inflation over very long periods.
    *   Optimal dollar value is a hyperparameter.

### 1.4. Imbalance Bars (Future Consideration)

*   **Concept:** More advanced bars formed when a certain cumulative imbalance is reached (e.g., in buy vs. sell volume, number of buyer-initiated ticks vs. seller-initiated ticks, or even order book imbalance if L2 data is incorporated).
*   **Potential Benefit:** Can directly capture shifts in order flow pressure and potentially signal directional moves more effectively.
*   **Considerations/Trade-offs:**
    *   More complex to define and calculate.
    *   The definition of "imbalance" and the threshold are key hyperparameters.
    *   May require more granular data (e.g., knowing buyer/seller for each tick).

### General Benefits of Event-Driven Bars:

*   **Adaptivity:** They naturally adjust to market volatility and activity levels.
*   **Information Consistency:** Each bar aims to contain a more uniform amount of "market information" or "eventfulness."
*   **Improved Statistical Properties:** Time series data constructed from event-driven bars (e.g., returns) may exhibit more desirable statistical properties (e.g., closer to normality, less autocorrelation) compared to fixed-time bars, which can be beneficial for modeling.

### General Considerations for All Event-Driven Bars:
*   They are still a form of aggregation, leading to some information loss compared to raw ticks.
*   The choice of bar type and its parameters (e.g., ticks per bar, volume per bar) are critical and may require careful tuning and validation.
*   While adaptive, they impose a specific structure on the data flow.

## 2. Direct Sequential Modeling of Ticks

This approach bypasses bar aggregation entirely, aiming to leverage the full granularity of the tick stream.

*   **Concept:** Instead of aggregating ticks into bars, feed sequences of raw or minimally processed tick data directly into a sequence-aware model (like a CNN-Transformer).
*   **Example Input Features per Tick:** A tuple or vector such as `(price_change_since_last_tick, volume_of_tick, time_delta_since_last_tick, is_buyer_maker_flag)`. Other tick-specific features could include rolling micro-features over the last N ticks.
*   **Model Responsibility:** The neural network itself would be responsible for learning relevant patterns, dependencies, and effective "features" from the high-frequency sequence of tick events.
*   **Potential Benefit:** 
    *   Minimizes pre-processing assumptions and theoretically preserves the maximum amount of information.
    *   Allows the model to capture very fine-grained, short-term dynamics that might be averaged out or missed by any form of bar aggregation.
    *   Potentially discovers novel patterns not captured by traditional bar-based feature engineering.
*   **Considerations/Trade-offs:**
    *   **Computational Intensity:** Processing, storing, and training on raw tick sequences can be significantly more demanding in terms of memory and compute resources.
    *   **Model Complexity:** Requires sophisticated sequence models capable of handling long, noisy, high-frequency series and learning meaningful representations.
    *   **Feature Engineering at Tick Level:** While the model learns high-level features, some basic feature engineering at the tick level (e.g., price changes, time deltas, simple moving averages over N ticks) might still be beneficial to provide the model with more structured inputs.
    *   **Long-Range Dependencies:** Capturing very long-range dependencies directly from ticks can be challenging for some architectures without specific mechanisms (like attention over very long sequences or hybrid approaches).
    *   **Noise Sensitivity:** Models might be more sensitive to noise in raw tick data if not robustly designed.

## 3. Hybrid Approaches

*   **Concept:** Combine different data representations. For instance, use event-driven bars as the primary input but augment them with features derived from fixed-time bars (e.g., a 1-hour moving average to provide longer-term context).
*   **Example:** Use 500-tick bars as input, but for each tick bar, also include the current 5-minute ATR or the state of a 4-hour trend filter.
*   **Potential Benefit:** Leverage the adaptivity of event-driven bars or the detail of tick sequences while still incorporating valuable signals from different time horizons or more stable, lower-frequency features. This can provide a balanced approach.
*   **Considerations/Trade-offs:**
    *   Increases complexity in data pipeline and model architecture.
    *   Careful alignment of different data sources (e.g., tick data and 1-hour features) is crucial.

## 4. Adapting Existing Features for Event-Driven Bars

Many features previously developed for fixed-time bars (e.g., in `feature_engine_v2.py` or from the `5_24_data_expansion.md` research) can be adapted:

*   **Basic OHLCV & Microstructure:** Once an event-driven bar is formed, it has an Open, High, Low, Close, Volume, VWAP, Start/End Timestamps, Duration, and Trade Count. These can be calculated from the constituent ticks.
*   **Technical Indicators (TA-Lib based):** Indicators like SMAs, EMAs, RSI, ATR, MACD, Stochastics can be calculated on the sequence of these event-driven bars. The "period" of these indicators will refer to the number of *event bars* (e.g., a 20-period SMA is the average close of the last 20 tick/volume/dollar bars).
*   **Candlestick Patterns:** Can be identified on the OHLC of event-driven bars.
*   **Price-Derived Features:** Metrics like bar bias, wick ratios, etc., can be computed.
*   **Regime and Contextual Features:** 
    *   Time-based features (hour of day, day of week) can be derived from the *closing timestamp* of each event-driven bar.
    *   Longer-term regime flags (e.g., based on 4-hour SMAs) would still be calculated using their original fixed-time logic but then *sampled* at the closing time of each event-driven bar to provide context.

## Next Steps

1.  **Deep Dive into Raw Tick Properties:** Before generating bars, further analyze raw tick data characteristics (inter-tick timings, price changes, volume distributions) to better inform bar construction parameters or direct modeling strategies.
2.  Develop exploratory scripts (e.g., in `minotaur/one_off_scripts/`) to generate samples of Tick Bars, Volume Bars, and Dollar Bars from raw tick data.
3.  Analyze the statistical properties and characteristics of these alternative bar types (e.g., distribution of durations, volatility, information content).
4.  Compare these properties against traditional fixed-time bars and also against the characteristics of the raw tick stream.
5.  Prototype data loading and preprocessing pipelines for both event-driven bars and direct sequential tick modeling to assess feasibility and computational requirements.
6.  Based on this exploration, refine the strategy for data representation and feature engineering for the trading model, potentially pursuing multiple paths initially. 