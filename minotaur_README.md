# Project Minotaur: Advanced Tick Data Processing & Featurization

This directory serves as the primary workspace for research, development, and implementation of advanced data processing and feature engineering strategies using high-frequency (tick-by-tick) BTC/USDT trade data. Our goal is to extract maximal predictive power from raw tick data to enhance a sophisticated trading model.

## Revised Core Strategy: Transition to Dollar Bars

Based on recent research (`minotaur/research/information_loss_in_data_aggregation.md`), our primary focus has shifted from general event-driven bars to **Dollar Bars** as the foundational data representation.

*   **Motivation:** To overcome significant information loss from fixed-time aggregation, better adapt to varying market activity levels, improve the statistical properties of input data for the model, and handle BTC's volatility more robustly.
*   **Core Questions Addressed by Dollar Bars:**
    *   How can we structure tick data for a CNN-Transformer model to maximize signal and minimize noise?
    *   How can we ensure each data input ("bar") to the model represents a more consistent level of economic activity?
*   **Chosen Approach:**
    1.  **Dollar Bars:** Aggregate tick data based on a fixed amount of quote currency (USDT) traded. This will be the primary bar type for feature engineering and model input. The threshold (e.g., $2,000,000 USDT per bar) will be a parameter for experimentation.

## Phased Data Processing Pipeline

To implement this strategy, we will adopt a three-phase data processing pipeline:

**Phase 1: Raw Tick Data -> Cleaned Tick Parquet Files (COMPLETED)**
*   **Input:** Raw tick data CSVs (e.g., from Binance).
*   **Process:**
    *   Convert CSVs to Parquet format.
    *   Perform basic cleaning: standardize timestamps to UTC (nanosecond or microsecond precision), ensure consistent column names (e.g., `timestamp`, `price`, `quantity`, `quote_quantity`, `isBuyerMaker`), handle obvious errors.
*   **Output & Storage:** Cleaned tick data stored in Parquet files, partitioned by year and month.
    *   **Directory:** `minotaur/data/cleaned_tick_data/` (Actual directory may vary based on local setup, assumed from previous context)
    *   **Filename Convention:** `{SYMBOL}_{YEAR}_{MONTH}.parquet` (e.g., `BTCUSDT_2023_01.parquet`)
*   **Status:** This phase is considered complete. Raw data has been processed into cleaned Parquet files.

**Phase 2: Cleaned Tick Parquet -> Enriched Dollar Bar Parquet Files (COMPLETED)**
*   **Input:** Cleaned tick Parquet files from Phase 1 (containing `isBuyerMaker` column).
*   **Process:**
    *   Iterate through ticks, accumulating the dollar value traded (`quote_quantity`).
    *   When the accumulated dollar value reaches the defined threshold (currently configured as $2,000,000 USDT), a new dollar bar is formed.
    *   Record: Open, High, Low, Close prices for the bar, total base asset Volume traded during the bar, total Quote Asset Volume (Dollar Volume), start (`open_timestamp`) and end (`close_timestamp`) timestamps of the bar, and the number of ticks (`num_ticks`) that formed the bar.
    *   **Key Features (as per inspection of `BTCUSDT_2024_04_dollar_bars_2M.parquet`):**
        *   `open_timestamp` (datetime64[ns])
        *   `open` (float64)
        *   `high` (float64)
        *   `low` (float64)
        *   `close` (float64)
        *   `close_timestamp` (datetime64[ns])
        *   `volume` (float64) - Base asset volume.
        *   `dollar_volume` (float64) - Quote asset volume traded, corresponds to "dollar_value_traded".
        *   `num_ticks` (int64)
    *   **Additional Microstructural Features Found (also in `BTCUSDT_2024_04_dollar_bars_2M.parquet`):**
        *   `trade_imbalance` (float64)
        *   `intra_bar_tick_price_volatility` (float64)
        *   `taker_buy_sell_ratio` (float64)
        *   `tick_price_skewness` (float64)
        *   `tick_price_kurtosis` (float64)
        *   `num_price_changes` (int64)
        *   `num_directional_changes` (int64)
    *   **Note:** `vwap` (Volume Weighted Average Price), previously listed as a key feature, was *not* present in the inspected `BTCUSDT_2024_04_dollar_bars_2M.parquet` file. Its generation source needs to be confirmed if it's expected.
*   **Implementation Notes:**
    *   The script `minotaur/scripts/phase2_generate_dollar_bars.py` implements this logic.
    *   Initial versions faced memory issues when processing large datasets. The script was revised to process input files one by one, generate dollar bars for that file, and save the resulting bars immediately to disk before processing the next file. This significantly reduces peak memory consumption.
    *   A `DollarBarGenerator` class encapsulates the bar generation logic, including a `reset()` method to clear state between files, calculation of the new intra-bar features, and a `flush_remaining_ticks_as_bar()` method to handle partial bars at the end of files.
    *   Runtime warnings for `skewness` and `kurtosis` on bars with (near) constant prices have been addressed by setting these features to 0.0 in such cases.
*   **Output & Storage:** Enriched dollar bar data stored in Parquet files, partitioned by the dollar bar threshold used and then by original file (year/month).
    *   **Directory Example:** `minotaur/data/dollar_bars/2M/` (where `2M` corresponds to the `$2,000,000` threshold)
    *   **Filename Convention (Observed):** `{SYMBOL}_{YEAR}_{MONTH}_dollar_bars_{THRESHOLD_STR}.parquet` (e.g., `BTCUSDT_2023_01_dollar_bars_2M.parquet`)
*   **Status:** Script implementation is complete and has successfully processed the full dataset. Output files for the 2M threshold are available in `minotaur/data/dollar_bars/2M/`. The structure of these files, including the features listed above (based on inspection of `BTCUSDT_2024_04_dollar_bars_2M.parquet`), has been verified.

**Phase 3: Multi-Resolution Feature Generation & Target Labeling (NEW STRATEGY)**
*   **Previous Challenge:** The original Phase 3 approach (concatenating all monthly dollar bars into a single Pandas DataFrame for feature calculation) was projected to exceed available RAM for the full dataset.
*   **New Strategy: Unified Feature Engine & Multi-Resolution Dollar Bars**
    1.  **`minotaur_feature_engine.py` (New File):** A dedicated Python module will be created to serve as the single source of truth for feature calculations, usable for both historical data generation and live trading.
    2.  **Input to Engine:** The engine will process "short" dollar bars (e.g., the $2M bars from Phase 2, which already include tick-derived intra-bar features for these short bars).
    3.  **Internal Aggregation:** `minotaur_feature_engine.py` will internally aggregate "short" dollar bars into "medium" and "long" dollar bars. The aggregation logic will be based on cumulative dollar value or a fixed number of shorter bars.
    4.  **Multi-Resolution Feature Calculation:** The engine will calculate a comprehensive set of features (TA-Lib indicators, simpler OHLCV-derived ratios, etc.) for *each* of the three dollar bar resolutions (short, medium, long). This allows the model to get a multi-faceted view of the market.
    5.  **Output Alignment:** For historical generation, the engine will output a unified DataFrame where features from all three resolutions are aligned to the timestamps of the "short" dollar bars (e.g., medium and long bar features will be forward-filled).
*   **Revised Phase 3 Script (e.g., `minotaur/scripts/phase3_process_features_targets.py`):**
    *   This script will instantiate and use `minotaur_feature_engine.py`.
    *   It will feed the "short" dollar bar data (output from Phase 2) into the engine.
    *   It will receive the final, multi-resolution feature DataFrame from the engine.
    *   Its primary remaining roles will be target calculation (as previously defined) and saving the final feature-rich, labeled dataset.
*   **Output & Storage:** The final dataset for model training, containing aligned features from short, medium, and long dollar bars, plus the target variable(s).
    *   **Directory Example:** `minotaur/data/multi_res_dollar_bar_features/S2M_M10M_L50M/` (Illustrative: S=Short, M=Medium, L=Long thresholds)
    *   **Filename Convention:** `{SYMBOL}_all_multi_res_features_targets.parquet`
*   **Status:**
    *   The `minotaur_feature_engine.py` module has been developed.
    *   The core structure for processing historical batches of short dollar bars (`process_historical_batch` method) is complete and tested.
    *   Aggregation logic to create medium and long dollar bars from short bars is implemented and verified. This includes researched strategies for aggregating intra-bar features (`trade_imbalance`, `intra_bar_tick_price_volatility`, etc.) from constituent short bars into the larger medium/long bars.
    *   Calculation for a comprehensive suite of features for each resolution (short, medium, long) is implemented, covering:
        *   Standard TA-Lib indicators (SMAs, EMAs, RSI, MACD, ADX, ATR, BBands, MFI, OBV, CCI, ROC, ULTOSC, WILLR, Volume SMA).
        *   Price/OHLCV-derived features (log returns, price changes, body/wick ratios).
        *   Time-based features (hour of day, day of week, etc.).
        *   Lagged features (for log returns, volume, and close price).
        *   Rolling window statistics (min, max, skew, kurtosis for log returns).
        *   Volatility measures.
    *   The engine correctly outputs three separate DataFrames (for short, medium, and long resolutions), each enriched with their respective features, which are then merged by the processing script.
    *   The `dollar_volume` column name has been standardized for consistency across all bar resolutions produced by the engine.
    *   Thorough smoke testing (`minotaur/scripts/smoke_test_feature_engine.py`) using a small dataset and a comprehensive feature configuration has been successful, validating the batch processing functionality.
    *   **COMPLETED: The script `minotaur/scripts/phase3_process_historical_features.py` has successfully processed all historical 2M dollar bars (approx. 1.96 million short bars from Aug 2017 - Apr 2024) using the `MinotaurFeatureEngine`.**
    *   **COMPLETED: The final multi-resolution feature dataset has been saved to `minotaur/data/multi_res_features/2M_x10_x16/BTCUSDT_all_multi_res_features.parquet` (approx. 928MB, 1.96M rows, 109 columns) and its structure has been verified.**
    *   **COMPLETED: The script `minotaur/scripts/phase3b_add_targets.py` has successfully loaded the multi-resolution features and added `target_long` labels, saving the output to `minotaur/data/multi_res_features/2M_x10_x16/BTCUSDT_all_multi_res_features_targets.parquet` (1.96M rows, 110 columns).**
    *   **COMPLETED: The script `minotaur/scripts/phase3c_consolidate_features.py` (new) has successfully combined features from multi-resolution dollar bars AND multi-resolution time-based bars (from Phase T2) and aligned them with the pre-calculated target `target_tp2.0_sl1.0`. The final consolidated dataset for model training is saved to `minotaur/data/consolidated_features_targets_all.parquet`. This file serves as the primary input for Phase 4 model training.**
    *   **NOTE ON `consolidated_features_targets_all.parquet` (Original Pandas-Consolidated Training Dataset):**
        *   **Path:** `minotaur/data/consolidated_features_targets_all.parquet`
        *   **Size:** 5,432,162 rows, 611 columns.
        *   **Contents:** Contains features from multi-resolution dollar bars (suffixes `_db_s`, `_db_m`, `_db_l`), multi-resolution time bars (suffixes `_tb_1min`, `_tb_15min`, `_tb_4hour` - without Volume Profile from `add_volume_profile.py` or specialized Stochastic Flags from `add_stochastic_flags.py` on the 15-min bars), and the `target_tp2.0_sl1.0` target column.
        *   This file was the primary input for model training before the recent successful addition of Volume Profile features to the 15-minute time bars.
    *   **NEWLY GENERATED (As of $(date +%Y-%m-%d)): `consolidated_features_targets_all_vp_updated.parquet`**
        *   **Path:** `minotaur/data/consolidated_features_targets_all_vp_updated.parquet`
        *   **Generation:** Created by merging the original `consolidated_features_targets_all.parquet` with `minotaur/data/time_bars_features_with_vp_debug_run2/15min/BTCUSDT_time_bars_features_15min_v2.parquet`.
        *   **Size:** 5,432,162 rows, 623 columns.
        *   **Contents:** Includes all features from `consolidated_features_targets_all.parquet` PLUS 12 new Volume Profile features (e.g., `vp_poc_50w_50b_tb_15min`) derived from the 15-minute time bars.
        *   **Important Note:** This version successfully incorporates the 15-minute Volume Profile features. However, the specialized stochastic *flags* (e.g., `is_stoch_ob_80_tb_15min`) that would be generated by `minotaur/scripts/time_based_processing/add_stochastic_flags.py` (intended to run on the VP-enriched 15-minute file) were **not** added to the 15-minute source file before this consolidation. Therefore, `consolidated_features_targets_all_vp_updated.parquet` **does not contain these specific stochastic flags.**
        *   **This file is the current dataset intended for immediate model training efforts.**
    *   **NOTE ON Dask-Processed `consolidated_features_all_v2.parquet/` (Future Primary Dataset):**
        *   **Status:** This phase is complete. The script `calculate_time_bar_features.py` has successfully processed all time-based bars and generated the feature-enriched files with the `_v2` suffix (indicating inclusion of log-transformed volume, among other updates). **These `_v2` files serve as the base input for subsequent enrichment scripts like `add_volume_profile.py` and `add_stochastic_flags.py`.**
        *   **Status:** This is the most up-to-date, comprehensive dataset available for Phase 4 model training.

2.  **Older Consolidated Dataset (Pandas Processed - Pre-VP/Stoch in Time Bars, Pre-Log Dollar Volume):**
    *   **Purpose:** An earlier consolidated dataset, primarily created using Pandas, before the full integration of Volume Profile and Stochastic features for time bars, and before log-transformed volumes were consistently added to dollar bars.
    *   **Script:** `minotaur/scripts/phase3c_consolidate_features.py` (original Pandas-based version or an earlier Dask iteration without full VP/Stoch).
    *   **Inputs:**
        *   Multi-Resolution Dollar Bar Features with targets (from II.3, potentially an older version without log dollar volumes).
        *   Time Bar Features from III.2 (e.g., `minotaur/data/time_bars_features/{RESOLUTION}/..._v2.parquet`), which did *not* yet have VP/Stoch successfully integrated.
    *   **Output Format:** Single Parquet file.
    *   **Verified Location:** `minotaur/data/consolidated_features_targets_all.parquet`
    *   **Verified Structure (Partial Inspection $(date +%Y-%m-%d)):** 5,432,162 rows, 611 columns.
    *   **Key Contents Confirmed:** Multi-resolution dollar bars (without log dollar volumes consistently), multi-resolution time bars (with log time bar volumes, but *without* VP/Stoch), and `target_tp2.0_sl1.0`.
    *   **Status:** Superseded by `minotaur/data/consolidated_features_all_v2.parquet/` for current training but useful as a reference for an earlier data state. **Further superseded for immediate use by `minotaur/data/consolidated_features_targets_all_vp_updated.parquet` which contains corrected 15-min VP features but not the specialized 15-min stochastic flags.**

3.  **Other Potentially Superseded or Intermediate Consolidated Dask Directories:**
    *   The `minotaur/data/` directory may contain other Dask-partitioned directories from various stages of development and debugging of the consolidation process. These include:
        *   `minotaur/data/consolidated_features_all_vp_stoch_consolidated.parquet/`: Likely an earlier Dask attempt to consolidate features including VP and Stochastics, possibly before all refinements in `..._all_v2.parquet/` were finalized. The README previously referred to this as a "new primary consolidated dataset" in an end-note, but `..._all_v2.parquet/` appears to have superseded it in the main narrative of latest developments.
        *   `minotaur/data/consolidated_features_all_vp_v2_consolidated.parquet/`: Naming suggests another iteration, possibly related to `..._all_v2.parquet/` but its exact state relative to the primary `_all_v2` would require deeper inspection if it were to be used.
        *   `minotaur/data/consolidated_features_all_v2_log_dask_fix_attempt2.parquet/`: As named, likely an intermediate output during Dask debugging.
    *   **Recommendation:** For current model training and development, `minotaur/data/consolidated_features_all_v2.parquet/` should be considered the primary and most reliable consolidated dataset. The others are likely historical artifacts unless specific investigation proves otherwise.

---
**Key to Understanding Suffixes in Consolidated Files (General Intent):**
*   `_db_s`, `_db_m`, `_db_l`: Features derived from Short, Medium, Long dollar bars respectively.
*   `_tb_1min`, `_tb_15min`, `_tb_4hour`: Features derived from 1-minute, 15-minute, 4-hour time bars respectively.
*   **Note on Doubled Suffixes:** Be aware that the current `consolidate_features.py` script might add *another* resolution suffix to time-bar features that already have one from `add_volume_profile.py` or `add_stochastic_flags.py` (e.g., `vp_poc_..._tb_15min` might become `vp_poc_..._tb_15min_tb_15min`). This is a known cosmetic issue to be addressed in `consolidate_features.py` eventually. The model training script (`minotaur_v1.py`) needs to handle these potentially doubled suffixes when selecting features.

---

## Overall Feature Generation Pipeline for `consolidated_features_targets_all.parquet`

The primary dataset used for model training, `minotaur/data/consolidated_features_targets_all.parquet`, is the result of a multi-stage pipeline that generates, processes, and consolidates features from both dollar bars and time-based bars. This section outlines the flow and the key scripts involved.

**High-Level Flow:**

1.  **Tick Data Cleaning:** Raw tick data is cleaned and standardized.
2.  **Dollar Bar Path:**
    *   Base dollar bars (e.g., 2M USDT) are generated from cleaned ticks.
    *   These base dollar bars are then used to create multiple resolutions of dollar bars (e.g., Short, Medium, Long) by aggregation.
    *   A comprehensive set of features is calculated for each dollar bar resolution.
    *   Target variables (e.g., `target_long`) are calculated based on the short-resolution dollar bar features.
3.  **Time Bar Path:**
    *   Base dollar bars (or cleaned ticks) are resampled into standard time-based intervals (e.g., 1-minute, 15-minute, 4-hour).
    *   A comprehensive set of features is calculated for each time bar resolution.
4.  **Consolidation:**
    *   The feature-rich, target-labeled multi-resolution dollar bars are combined with the feature-rich multi-resolution time bars.
    *   Timestamps are aligned, and columns are renamed for clarity and uniqueness, resulting in the final `consolidated_features_targets_all.parquet` file.

**Detailed Script Breakdown:**

**1. Tick Data Preparation:**
*   **Script:** `minotaur/scripts/phase1_clean_ticks.py`
*   **Input:** Raw tick data CSVs.
*   **Process:** Converts CSVs to Parquet, standardizes timestamps (UTC), ensures consistent column names, and handles basic errors.
*   **Output:** Cleaned tick data in Parquet files (e.g., `minotaur/data/cleaned_tick_data/BTCUSDT_YYYY_MM.parquet`).

**2. Dollar Bar Feature & Target Generation Pipeline:**
    *   **Step 2.1: Generate Base Dollar Bars (e.g., 2M USDT)**
        *   **Script:** `minotaur/scripts/phase2_generate_dollar_bars.py`
        *   **Input:** Cleaned tick Parquet files from Step 1.
        *   **Process:** Aggregates ticks into dollar bars based on a defined `quote_quantity` threshold. Calculates OHLCV, timestamps, tick count, and intra-bar features (trade imbalance, price volatility, etc.) for these base dollar bars.
        *   **Output:** Base dollar bars (e.g., `minotaur/data/dollar_bars/2M/BTCUSDT_YYYY_MM_dollar_bars_2M.parquet`).
    *   **Step 2.2: Generate Multi-Resolution Dollar Bar Features**
        *   **Script:** `minotaur/scripts/phase3_process_historical_features.py`
        *   **Input:** Base dollar bars from Step 2.1 (e.g., all `..._dollar_bars_2M.parquet` files).
        *   **Process:**
            *   Uses `minotaur.scripts.minotaur_feature_engine.MinotaurFeatureEngine`.
            *   Aggregates the input short dollar bars into medium and long dollar bar resolutions (e.g., 2M -> 20M -> 320M).
            *   Calculates a wide array of features (TA-Lib, price-derived, time-based, lagged, rolling stats) for each of the three dollar bar resolutions (Short, Medium, Long).
        *   **Output:** A single Parquet file containing aligned features for all three dollar bar resolutions (e.g., `minotaur/data/multi_res_features/2M_x10_x16/BTCUSDT_all_multi_res_features.parquet`). Column names are prefixed (e.g., `s_`, `m_`, `l_`).
    *   **Step 2.3: Add Targets to Multi-Resolution Dollar Bar Features**
        *   **Script:** `minotaur/scripts/phase3b_add_targets.py`
        *   **Input:** The multi-resolution dollar bar feature file from Step 2.2.
        *   **Process:** Calculates target variables (e.g., `target_long`) based on specified criteria (e.g., ATR of short dollar bars for stop-loss/take-profit levels).
        *   **Output:** The multi-resolution dollar bar features now including target columns (e.g., `minotaur/data/multi_res_features/2M_x10_x16/BTCUSDT_all_multi_res_features_targets.parquet`). **This is a key input for the final consolidation.**

**3. Time-Based Bar Feature Generation Pipeline:**
    *   **Step 3.1: Generate Raw Time-Based Bars**
        *   **Script:** `minotaur/scripts/time_based_processing/generate_time_bars.py`
        *   **Input:** Cleaned tick Parquet files from Step 1 (or potentially base dollar bars from Step 2.1, though current implementation uses ticks).
        *   **Process:** Aggregates input data into standard time intervals (e.g., 1-minute, 15-minute, 4-hour). Calculates OHLCV, VWAP, Order Flow Imbalance (OFI), and other basic bar statistics.
        *   **Output:** Raw time-based bars, typically saved per resolution and original file period (e.g., `minotaur/data/time_bars/1min/BTCUSDT_YYYY_MM_time_bars_1min.parquet`).
    *   **Step 3.2: Calculate Features for Time-Based Bars**
        *   **Script:** `minotaur/scripts/time_based_processing/calculate_time_bar_features.py`
        *   **Input:** Raw time-based bars from Step 3.1 (concatenated per resolution).
        *   **Process:** Calculates a comprehensive set of features for each time resolution. This includes TA-Lib indicators, price/return features, volume features, cyclical time features, volatility measures, candlestick patterns, lagged features, rolling statistics, divergence features, and market regime features.
        *   **Output:** Feature-rich time bars for each resolution, typically saved as a single file per resolution (e.g., `minotaur/data/time_bars_features/1min/BTCUSDT_time_bars_features_1min.parquet`). **This is a key input for the final consolidation.**

**4. Final Consolidation of All Features and Targets:**
*   **Script:** `minotaur/scripts/consolidate_features.py`
*   **Inputs:**
    *   The multi-resolution dollar bar features with targets from Step 2.3 (e.g., `.../BTCUSDT_all_multi_res_features_targets.parquet`).
    *   The feature-rich time bars for each resolution from Step 3.2 (e.g., `.../1min/BTCUSDT_time_bars_features_1min.parquet`, `.../15min/...`, etc.).
*   **Process:**
    *   Loads all input DataFrames (typically using Dask for memory efficiency).
    *   Renames columns from each source to ensure uniqueness and clarity (e.g., dollar bar features get `_db_s`, `_db_m`, `_db_l` suffixes; time bar features get `_tb_1min`, `_tb_15min` suffixes).
    *   Merges all DataFrames based on their timestamp index, using an as-of merge (`merge_asof`) to align less frequent data to the most frequent timeline (typically the dollar bar timeline or 1-min time bar timeline, depending on which is primary).
*   **Output:** The final consolidated dataset for model training: `minotaur/data/consolidated_features_targets_all.parquet`. This file contains all dollar bar features, all time bar features, and the primary target variables, all aligned to a common timestamp index. **Subsequently, `minotaur/data/consolidated_features_targets_all_vp_updated.parquet` was created by merging this file with 15-minute Volume Profile features.**

This consolidated pipeline provides a rich, multi-faceted dataset for the Minotaur trading model.

**Update (Post-Volume Profile Feature Addition):**
A new version of the consolidated dataset has been generated which includes Volume Profile features (POC, VAH, VAL) for **15-minute** time-bar resolutions: `minotaur/data/consolidated_features_targets_all_vp_updated.parquet`.
*   **Generation Process:**
    1. Volume Profile features were calculated and added to the `_v2` 15-minute time bar feature file using `minotaur/scripts/time_based_processing/add_volume_profile.py`, with output in `minotaur/data/time_bars_features_with_vp_debug_run2/15min/`.
    2. Stochastic Oscillator flags (from `minotaur/scripts/time_based_processing/add_stochastic_flags.py`) were **not** run on this updated 15-minute file prior to the next step.
    3. The `temp_merge_script.py` (a Pandas-based script) was run to combine `minotaur/data/consolidated_features_targets_all.parquet` (which has no VP features) with the 15-minute VP-enriched file from step 1.
*   **This `consolidated_features_targets_all_vp_updated.parquet` dataset is now the dataset for immediate model training. It contains 15-minute VP features but lacks the specialized 15-minute stochastic flags.**
*   **Note on `consolidated_features_all_vp_stoch_consolidated.parquet/`:** This Dask-processed directory is a more advanced target dataset that *aims* to include VP for all time resolutions and stochastic flags. Its generation is dependent on completing the VP and stochastic flag processing for all relevant time bar files.
*   **Note on Column Naming (Doubled Suffixes):** Due to the current suffixing logic in the feature addition scripts (e.g., `add_volume_profile.py`, `add_stochastic_flags.py` adding suffixes like `_tb_15min`) and the `consolidate_features.py` script also adding a resolution-specific suffix, some features derived from the time-bar pipeline (Volume Profile and Stochastic Oscillator features) will have a repeated suffix in the final consolidated file. 

---

## Future Action Items & Iteration Strategy

To guide ongoing development and systematically improve model performance, the following key areas will be addressed:

1.  **Verify Real-time Feature Processing Alignment:**
    *   **Goal:** Ensure that the feature calculation and processing pipeline used during historical training (`minotaur_v1.py` and preceding scripts like `minotaur_feature_engine.py`) precisely matches how features would be generated and processed in a live trading environment.
    *   **Rationale:** Even minor discrepancies between historical/offline feature generation and real-time/online feature generation can lead to significant performance degradation or unexpected model behavior when deployed. Consistency is paramount.
    *   **Tasks:**
        *   Thoroughly review and document the exact sequence of operations for feature calculation, aggregation, transformation, and scaling as implemented for training.
        *   Design and implement the `add_short_bar` method (and any related live processing logic in `minotaur_feature_engine.py` or a dedicated live engine) to mirror these historical processes with high fidelity.
        *   Pay close attention to lookback periods, state management (e.g., for rolling statistics or EMA calculations), and data alignment, especially for multi-resolution features.
        *   Develop a test suite or a comparative analysis framework to validate that features generated from a live tick stream (or a simulated one) match those generated historically for the same period.

2.  **Deep Dive into Multicollinearity and Advanced Feature Selection:**
    *   **Goal:** Optimize the input feature set by systematically addressing multicollinearity and employing intelligent feature selection techniques, in conjunction with model architecture experimentation.
    *   **Rationale:** Financial time-series data is often characterized by high multicollinearity among derived features (e.g., different moving averages of the same price, various momentum indicators). Redundant or highly correlated features can make model interpretation difficult, increase model variance, and sometimes hinder performance. Effective feature selection can lead to simpler, more robust, and better-performing models.
    *   **Tasks:**
        *   **Revisit Multicollinearity Analysis:** Extend the initial multicollinearity pruning (currently done on dollar bar features only) to the full `consolidated_features_targets_all.parquet` dataset. Use techniques like Variance Inflation Factor (VIF) in addition to correlation matrices.
        *   **Systematic Feature Selection:**
            *   Explore advanced feature selection methods beyond basic pruning:
                *   Recursive Feature Elimination (RFE) with different base models.
                *   Mutual Information scores.
                *   Model-based importance (e.g., from LightGBM, RandomForest, or SHAP values from the current CNN-Transformer).
                *   Consider dimensionality reduction techniques like PCA (Principal Component Analysis) if appropriate, though with caution regarding interpretability.
            *   Document the rationale for selecting or deselecting features rigorously.
        *   **Interaction Terms & Transformations:** Investigate the creation of meaningful interaction terms or further non-linear transformations of existing features, as suggested by EDA or domain knowledge.
        *   **Iterate with Model Architecture:** Conduct feature selection experiments in tandem with trying different model architectures or hyperparameter configurations, as feature relevance can be model-dependent.

    *   **Sub-Strategy: Enhanced Feature Engineering and Normalization:**
        *   **Goal:** Maximize signal extraction from the available data by refining existing features, introducing new conceptually valuable features (like volume profile levels), and applying appropriate data shaping techniques.
        *   **Rationale:** Beyond simply removing redundant features, actively engineering new features and carefully considering normalization/transformation can significantly improve model performance by making underlying patterns more accessible to the model.
        *   **Tasks:**
            *   **Advanced Normalization/Transformation Exploration:**
                *   Re-evaluate current normalization strategies (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `PowerTransformer` via Yeo-Johnson).
                *   Investigate other advanced techniques: Quantile transformation (to map features to a uniform or normal distribution), or custom non-linear transformations based on feature distributions observed in EDA.
                *   Consider feature-specific transformations where appropriate (e.g., differencing for non-stationary series, ratio creation).
            *   **New Feature Brainstorming & Implementation - Volume Profile:**
                *   **Introduce Conceptually Valuable New Features:**
                    *   **Volume Profile Levels:** Research and implement features derived from Volume Profile analysis (e.g., Point of Control (POC), Value Area High (VAH), Value Area Low (VAL)) for various lookback periods. These can act as dynamic support/resistance or indicate areas of high/low liquidity.
                    *   *(Initial investigation documented below)*
                    *   **Status (As of $(date +%Y-%m-%d)): Implemented and Consolidated.**
                        *   **Library Investigation:**
                            *   `volprofile` (PyPI: `volprofile` 1.0.1) was selected.
                        *   **Implementation Script (`add_volume_profile.py`):**
                            *   Located at `minotaur/scripts/time_based_processing/add_volume_profile.py`.
                            *   Adds POC, VAH, VAL for configurable windows/bins to existing time-bar feature Parquets.
                        *   **Data Processed & Output (VP Addition Stage):**
                            *   Processed `_v2.parquet` versions for `1min`, `15min`, and `4hour` resolutions.
                            *   Outputs saved to `minotaur/data/time_bars_features_with_vp/{RESOLUTION}/`.
                        *   **Consolidation:**
                            *   The `minotaur/scripts/consolidate_features.py` script was updated to use these VP-enriched time bar files.
                            *   The new primary consolidated dataset is `minotaur/data/consolidated_features_all_vp_stoch_consolidated.parquet/`.
                    *   **Order Flow Imbalance (OFI) Delta/Cumulative Delta:** Explore more sophisticated OFI metrics if the current ones prove insufficient.
                *   **Iterate with Model Architecture:** Conduct feature selection experiments in tandem with trying different model architectures or hyperparameter configurations, as feature relevance can be model-dependent.

            *   **Iterate with Model Architecture:** Conduct feature selection experiments in tandem with trying different model architectures or hyperparameter configurations, as feature relevance can be model-dependent.

    *   **Sub-Strategy: Multi-objective Optimization for Feature Selection & Model Tuning (Future Consideration):**
        *   **Goal:** To balance multiple objectives (e.g., maximizing signal extraction, minimizing model complexity, addressing multicollinearity) in the feature selection process.
        *   **Rationale:** Effective feature selection is crucial for building robust and interpretable models. A multi-objective approach can lead to a more comprehensive and balanced feature set.
        *   **Tasks:**
            *   **Define Multiple Objective Functions:**
                *   **Objective 1: Maximize Signal Extraction:**
                    *   **Goal:** To extract the most relevant and informative features from the data.
                    *   **Method:** Use feature importance scores (e.g., from SHAP values, model-based importance) to rank features based on their contribution to the model's performance.
                *   **Objective 2: Minimize Model Complexity:**
                    *   **Goal:** To keep the model simple and interpretable.
                    *   **Method:** Use feature selection methods that reduce the number of features (e.g., Recursive Feature Elimination, Mutual Information).
                *   **Objective 3: Address Multicollinearity:**
                    *   **Goal:** To reduce feature redundancy and improve model stability.
                    *   **Method:** Use techniques like Principal Component Analysis (PCA) to identify and remove correlated features.
            *   **Multi-objective Optimization Framework:**
                *   Implement a framework for combining these objectives into a single optimization problem. This could be done using a weighted sum approach, where each objective is assigned a weight based on its importance.
            *   **Iterate with Model Architecture:** Conduct feature selection experiments in tandem with trying different model architectures or hyperparameter configurations, as feature relevance can be model-dependent.

3.  **Analyze and Interpret Current Model Run Outputs:**
    *   **Goal:** Extract actionable insights from the ongoing Optuna hyperparameter search and model training runs to guide further improvements.
    *   **Rationale:** The current runs are producing potentially promising results ("better than legacy implementations"). A thorough analysis of these outputs can reveal which hyperparameter ranges are effective, identify common failure modes, understand feature importance (if derivable), and guide the next set of experiments.
    *   **Tasks:**
        *   **Systematic Log Review:** Analyze the detailed logs, `history_trial_*.json`, `f1_history_trial_*.jsonl`, and `test_metrics_eval_trial_*.json` files from the current Optuna study.
        *   **Hyperparameter Landscape:** Visualize the relationship between Optuna-selected hyperparameters and performance metrics (e.g., `val_auc`, `best_val_f1`) to understand sensitivities and promising regions in the search space.
        *   **Metric Deep Dive:** Go beyond aggregate metrics. Examine confusion matrices, per-class precision/recall, and training/validation loss curves for individual trials to understand model behavior.
        *   **Error Analysis (if feasible):** If possible, investigate specific instances where the model makes incorrect predictions on the validation/test sets to identify patterns or problematic data segments.
        *   **Feature Importance (Preliminary):** If the model architecture or available tools allow (e.g., SHAP for certain model types, or analyzing attention weights in Transformers), attempt a preliminary assessment of feature importance from the best-performing trials.
        *   **Compare with Baselines/Legacy:** Quantify the improvements over legacy implementations and establish clear benchmarks for future iterations.

3.  **Other Potentially Superseded or Intermediate Consolidated Dask Directories:**
    *   The `minotaur/data/` directory may contain other Dask-partitioned directories from various stages of development and debugging of the consolidation process. These include:
        *   `minotaur/data/consolidated_features_all_vp_stoch_consolidated.parquet/`: Likely an earlier Dask attempt to consolidate features including VP and Stochastics, possibly before all refinements in `..._all_v2.parquet/` were finalized. The README previously referred to this as a "new primary consolidated dataset" in an end-note, but `..._all_v2.parquet/` appears to have superseded it in the main narrative of latest developments.
        *   `minotaur/data/consolidated_features_all_vp_v2_consolidated.parquet/`: Naming suggests another iteration, possibly related to `..._all_v2.parquet/` but its exact state relative to the primary `_all_v2` would require deeper inspection if it were to be used.
        *   `minotaur/data/consolidated_features_all_v2_log_dask_fix_attempt2.parquet/`: As named, likely an intermediate output during Dask debugging.
    *   **Recommendation:** For current model training and development, `minotaur/data/consolidated_features_all_v2.parquet/` should be considered the primary and most reliable consolidated dataset. The others are likely historical artifacts unless specific investigation proves otherwise.

## Running Optuna Hyperparameter Search

This section describes how to run an Optuna hyperparameter search for the Minotaur models using the `minotaur/model_training/minotaur_v1.py` script.

### Command Structure

Here is an example command to launch an Optuna study:

```bash
python minotaur/model_training/minotaur_v1.py \\
    --n-trials 50 \\
    --epochs 100 \\
    --feature-parquet "minotaur/data/consolidated_features_targets_all_vp_updated.parquet" \\
    --optuna-study-name "minotaur_cnn_transformer_study_1" \\
    --optuna-db-filename "optuna_study.db" \\
    --output-dir "minotaur/optuna_studies/minotaur_study_run_$(date +%Y%m%d_%H%M%S)/" \\
    --symbol "BTCUSDT" \\
    --save-results \\
    --keras-verbose 1
```

### Key Arguments Explained

*   `--n-trials <integer>`: Specifies the total number of trials Optuna should run.
*   `--epochs <integer>`: Specifies the number of epochs each individual trial (model training instance) should run.
*   `--feature-parquet <path_to_file>`: Path to the consolidated Parquet file containing features and targets.
    *   Example: `minotaur/data/consolidated_features_targets_all_vp_updated.parquet` (current recommended dataset).
*   `--optuna-study-name <string>`: A descriptive name for the Optuna study. This name is stored within the Optuna database and helps identify the study across multiple script executions if they use the same database.
    *   Example: `"minotaur_cnn_transformer_study_1"`
*   `--optuna-db-filename <filename>`: The **filename** for the Optuna SQLite database.
    *   Example: `"optuna_study.db"`
    *   **Important:** This should only be the filename (e.g., `study.db`). The script will automatically place this database file inside a unique, numbered subdirectory created within the path specified by `--output-dir`.
*   `--output-dir <path_to_directory>`: The base directory where artifacts for this specific *execution group* of the Optuna study will be stored.
    *   Example: `"minotaur/optuna_studies/minotaur_study_run_$(date +%Y%m%d_%H%M%S)/"`
    *   The script will create a new numbered subdirectory (e.g., `study_001/`, `study_002/`, etc.) *inside* this `--output-dir` *each time* `minotaur_v1.py` is run.
    *   This numbered subdirectory (`study_00X/`) will contain:
        *   The Optuna database file (specified by `--optuna-db-filename`).
        *   Subdirectories for each trial's artifacts (e.g., `trial_000_.../`, `trial_001_.../`), which include saved models, logs, plots, and configuration files.
*   `--symbol <string>`: The trading symbol (e.g., `BTCUSDT`). Used for naming trial artifact directories.
*   `--save-results`: A flag to enable saving of models, plots, metrics, and configuration for each trial. Use `--no-save-results` to disable.
*   `--keras-verbose <0|1|2>`: Keras training verbosity level.
    *   `0`: Silent.
    *   `1`: Progress bar.
    *   `2`: One line per epoch.

### Directory Structure Example

If you run the example command above, the directory structure might look like this:

```
minotaur/
├── optuna_studies/
│   └── minotaur_study_run_20250530_190000/  # <-- Corresponds to --output-dir
│       └── study_001/                       # <-- Auto-created by the script on first run
│           ├── optuna_study.db              # <-- --optuna-db-filename
│           ├── trial_000_BTCUSDT_.../       # <-- Trial artifacts
│           │   ├── best_auc_model.keras
│           │   ├── history_trial_000.json
│           │   └── ...
│           ├── trial_001_BTCUSDT_.../
│           │   └── ...
│           └── ... (up to trial_049_...)
└── ... (other project files)
```
If you execute `minotaur_v1.py` again with the *same* `--output-dir` but potentially different `--n-trials` or other settings, it would create `study_002/` inside `minotaur_study_run_20250530_190000/`, containing a new database and new trial artifacts for that execution.

This setup allows for organizing different sets of Optuna experiments (e.g., by date or feature set via different `--output-dir` values) while keeping the results of each individual script execution distinct (via the `study_00X` subdirectories).

## Current Status & Next Steps (As of $(date +%Y-%m-%d_%H%M))

**1. Model Training Framework Enhancements:**
    *   Successfully modified `minotaur/model_training/minotaur_v1.py` to include a `--no-optuna` flag. This allows for running single training sessions with fixed hyperparameters, significantly speeding up initial tests and debugging compared to full Optuna hyperparameter optimization studies.
    *   Conducted a quick "smoke test" run using the `--no-optuna` flag with a small subset of data (`--smoke-test-nrows 50000`), limited epochs (`--epochs 10`), and a small number of steps per epoch (`--steps-per-epoch 10`) to verify the new functionality. The test completed successfully.

**2. Next Immediate Action: Full Training Run with Hypothesized Architecture:**
    *   **Objective:** Conduct a full training run using the `consolidated_features_targets_all_vp_updated.parquet` dataset.
    *   **Architecture:** The run will employ a model architecture based on insights from `5_21_research` notes, specifically focusing on:
        *   Using the `CNNDualStreamTransformer` model type (`--model-type CNNTransformer` with dual stream implied by feature config or to be explicitly set if an argument exists).
        *   Configuring two Transformer blocks (`--num-transformer-blocks-fixed 2` when using `--no-optuna`, or as an Optuna search space).
        *   Employing Grouped Query Attention (`--attention-mechanism grouped_query_attention`).
        *   Utilizing a ReGLU MLP head (`--mlp-activation reglu` and potentially `--mlp-head-type reglu_mlp` if distinct).
        *   Using `RMSprop` as the optimizer (`--optimizer rmsprop`).
        *   Applying instance normalization if available (`--instance-norm-affine` or similar).
    *   **Dataset:** `minotaur/data/consolidated_features_targets_all_vp_updated.parquet`.
    *   **Tracking:** Results will be saved to a new directory under `minotaur/outputs/` or `minotaur/optuna_studies/` depending on whether Optuna is used for this full run.

**3. Future Considerations (Post Full Run):**
    *   Analyze results from the full run, focusing on performance metrics and loss curves.
    *   Re-evaluate the need for Optuna hyperparameter tuning for this architecture or proceed with further targeted experiments.
    *   Continue refining feature selection and addressing multicollinearity based on model performance.

**Update ($(date +%Y-%m-%d_%H%M) - Addressing Overfitting and Feature Reduction Strategy):**

*   **Problem Identified:** Analysis of `minotaur_5_31_outputs/001_20250601_010103/trial_000_BTCUSDT_cnntransformer_single_stack_20250601_010104/history_0.json` (a `--no-optuna` run with 611 features) revealed significant overfitting (training metrics improving while validation metrics degraded).
*   **Initial Corrective Actions:**
    *   Increased default dropout rates in `minotaur_v1.py` to provide stronger regularization for fixed runs:
        *   General transformer dropout (`--dropout-rate`) default changed from 0.20 to 0.30.
        *   MLP dropout (`--mlp-dropout-rate`) default changed from 0.25 to 0.35.
        *   Fixed CNN dropout (`--cnn-dropout-rate-fixed`) default changed from 0.10 to 0.20.
*   **Feature Set Review:**
    *   Recognized that the 611 features used in the overfitting run (derived from `consolidated_features_targets_all.parquet`) is a very large set.
    *   Generated `all_features_list.txt` using `list_parquet_features.py` to provide a complete inventory of all columns in the source Parquet.
    *   Confirmed that while `target_tp2.0_sl1.0` appears in `all_features_list.txt`, it is correctly excluded as an input feature by `minotaur_v1.py` during training.
*   **Next Experimental Step - Focused Feature Reduction:**
    *   **Objective:** To assess if reducing the number of input features, combined with increased dropout, mitigates overfitting.
    *   **Plan:** Conduct a new `--no-optuna` training run using `minotaur_v1.py` with the following configuration:
        *   **Features:** Only short-term features will be enabled (e.g., by setting `--use-short-term-features` to true and `--use-medium-term-features`, `--use-long-term-features`, `--use-regime-features` to false).
        *   **Regularization:** Utilize the recently increased default dropout rates.
        *   **Dataset:** `minotaur/data/consolidated_features_targets_all_vp_updated.parquet` (or the standard `consolidated_features_targets_all.parquet` if VP features are not the immediate focus for this test).
*   **Future Strategic Feature Selection (if simpler reduction is insufficient):**
    *   If overfitting persists, or if performance with reduced features is unsatisfactory, a more systematic feature selection process will be initiated.
    *   Methods to explore include:
        1.  Manual review of `all_features_list.txt` for obvious redundancies.
        2.  Model-based feature importance ranking using RandomForest or XGBoost.
        3.  Statistical methods like correlation analysis and mutual information.

*   **Update ($(date +%Y-%m-%d_%H%M) - Feature Selection and New Training Run):**
    *   **Problem:** Continued observation of overfitting with a large feature set (611 features).
    *   **Action - Feature Selection:**
        *   Implemented a RandomForest-based feature importance ranking script (`select_features_rf.py`).
        *   Generated `top_100_rf_features.txt` containing the 100 most important features relative to `target_tp2.0_sl1.0` from the `consolidated_features_targets_all_vp_updated.parquet` dataset.
    *   **Action - Model Script Update:**
        *   Modified `minotaur/model_training/minotaur_v1.py` to accept a `--feature-list-file` argument, allowing it to use a specified subset of features.
    *   **Action - Smoke Test & Full Training Run:**
        *   Conducted a successful short smoke test (3 epochs, 10 steps) using the 100 features from `top_100_rf_features.txt` and fixed hyperparameters from a previous configuration (`trial_000_full_config.json`). This validated the new feature loading mechanism.
        *   **Initiated a full training run** (50 epochs, 25,000 steps/epoch) using the `--no-optuna` mode with the same 100 features and hyperparameters. This run is currently in progress. Output will be in a new subdirectory under `minotaur_5_31_outputs/`.