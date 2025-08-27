import pandas as pd
import glob
import os
import logging
from minotaur.scripts.minotaur_feature_engine import MinotaurFeatureEngine
from typing import Optional # Added for type hinting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_historical_feature_generation(limit_files: Optional[int] = None): # Added limit_files parameter
    """
    Loads all historical short dollar bars, generates multi-resolution features
    using MinotaurFeatureEngine, merges them, and saves the result.

    Args:
        limit_files (Optional[int]): If provided, limits the number of input parquet files to process.
    """
    logging.info("Starting historical feature generation...")
    if limit_files:
        logging.info(f"--- LIMITED RUN: Processing only the first {limit_files} files. ---")

    # Define paths
    # Assuming script is run from the root of trading-agent-v3
    base_data_dir = os.path.join("minotaur", "data")
    short_bars_input_dir = os.path.join(base_data_dir, "dollar_bars", "2M")
    
    # Define output path using factors for clarity
    # Short bars are 2M. Medium = 2M * 10. Long = (2M * 10) * 16.
    output_dir_name = "2M_x10_x16" # Base short bar threshold, medium factor, long factor
    output_features_dir = os.path.join(base_data_dir, "multi_res_features", output_dir_name)
    os.makedirs(output_features_dir, exist_ok=True)
    
    output_filename_base = "BTCUSDT_all_multi_res_features"
    if limit_files:
        output_filename = f"{output_filename_base}_LIMITED_{limit_files}_files.parquet"
    else:
        output_filename = f"{output_filename_base}.parquet"
    output_file_path = os.path.join(output_features_dir, output_filename)

    logging.info(f"Input short dollar bar directory: {short_bars_input_dir}")
    logging.info(f"Output features directory: {output_features_dir}")
    logging.info(f"Output file: {output_file_path}")

    # 1. Load and concatenate all short dollar bar files
    all_short_bar_files = sorted(glob.glob(os.path.join(short_bars_input_dir, "BTCUSDT_*_dollar_bars_2M.parquet")))
    
    if not all_short_bar_files:
        logging.error(f"No short dollar bar files found in {short_bars_input_dir}. Exiting.")
        return

    if limit_files:
        logging.info(f"Limiting to the first {limit_files} files out of {len(all_short_bar_files)} found.")
        all_short_bar_files = all_short_bar_files[:limit_files]

    logging.info(f"Found {len(all_short_bar_files)} short dollar bar files to process.")

    list_of_dfs = []
    for f in all_short_bar_files:
        try:
            df = pd.read_parquet(f)
            list_of_dfs.append(df)
            logging.debug(f"Successfully loaded and appended {f}, shape: {df.shape}")
        except Exception as e:
            logging.error(f"Error loading file {f}: {e}")
            # Decide if you want to skip or halt on error
            # For now, let's halt
            return 
            
    if not list_of_dfs:
        logging.error("No dataframes were loaded. Exiting.")
        return

    logging.info("Concatenating all loaded short dollar bar DataFrames...")
    short_bars_df = pd.concat(list_of_dfs, ignore_index=True)
    logging.info(f"Concatenated short bars DataFrame shape: {short_bars_df.shape}")
    
    # Ensure timestamps are sorted - critical for feature calculation and merging
    short_bars_df.sort_values(by='open_timestamp', inplace=True)
    short_bars_df.reset_index(drop=True, inplace=True)
    logging.info("Short bars DataFrame sorted by open_timestamp.")

    # 2. Define MinotaurFeatureEngine configurations (as from smoke test)
    medium_factor = 10
    long_factor = 16 # This is applied to medium bars to get long bars

    short_bar_config = {
        'name': 'short_2M_dollar_bars'
    }
    medium_bar_config = {
        'agg_type': 'bar_count',
        'bars_per_agg': medium_factor,
        'agg_col': 'dollar_volume' 
    }
    long_bar_config = {
        'agg_type': 'bar_count',
        'bars_per_agg': long_factor, # Number of medium bars to make one long bar
        'agg_col': 'dollar_volume'
    }

    feature_configs = {
        'short': [
            # TA-Lib
            {'name': 'sma', 'params': {'timeperiod': 10, 'price_col': 'close'}},
            {'name': 'sma', 'params': {'timeperiod': 20, 'price_col': 'close'}},
            {'name': 'ema', 'params': {'timeperiod': 10, 'price_col': 'close'}},
            {'name': 'rsi', 'params': {'timeperiod': 14, 'price_col': 'close'}},
            {'name': 'macd', 'params': {'price_col': 'close'}}, # Default fast/slow/signal
            {'name': 'adx', 'params': {'timeperiod': 14}},
            {'name': 'plus_di', 'params': {'timeperiod': 14}},
            {'name': 'minus_di', 'params': {'timeperiod': 14}},
            {'name': 'stoch', 'params': {}}, # Default k/d/slowing
            {'name': 'obv', 'params': {}},
            {'name': 'atr', 'params': {'timeperiod': 14}},
            {'name': 'bbands', 'params': {'timeperiod': 20, 'price_col': 'close'}},
            {'name': 'cci', 'params': {'timeperiod': 14}},
            {'name': 'mfi', 'params': {'timeperiod': 14}},
            {'name': 'roc', 'params': {'timeperiod': 10, 'price_col': 'close'}},
            {'name': 'ultosc', 'params': {}},
            {'name': 'willr', 'params': {'timeperiod': 14}},
            {'name': 'volume_sma', 'params': {'timeperiod': 20}},
            # Price/OHLCV-derived
            {'name': 'log_return', 'params': {'price_col': 'close', 'type': 'close'}},
            {'name': 'log_return_high_low', 'params': {'type': 'high_low'}},
            {'name': 'price_change_pct', 'params': {'price_col': 'close'}},
            {'name': 'high_low_pct_range', 'params': {}},
            {'name': 'close_vs_high_pct', 'params': {}},
            {'name': 'close_vs_low_pct', 'params': {}},
            {'name': 'wick_body_vs_range', 'params': {}},
            # Volatility
            {'name': 'volatility_log_returns', 'params': {'window': 20, 'price_col': 'close'}},
            {'name': 'volatility_hl_log_returns', 'params': {'window': 20}},
            # Time-based (applied to short bars, but relevant for all resolutions)
            {'name': 'time_hour_of_day', 'params': {'timestamp_col': 'close_timestamp'}},
            {'name': 'time_day_of_week', 'params': {'timestamp_col': 'close_timestamp'}},
            # Lagged (source_col_name must match either raw df column or a previously calculated feature name for THIS resolution)
            {'name': 'lagged_feature', 'params': {'source_col_name': 'log_return_close', 'lags': [1, 2, 3, 5]}},
            {'name': 'lagged_feature', 'params': {'source_col_name': 'volume', 'lags': [1, 2, 3]}},
            # Rolling (source_col_name must match a previously calculated feature name for THIS resolution)
            {'name': 'rolling_stat', 'params': {'source_col_name': 'log_return_close', 'window': 10, 'stat': 'mean'}},
            {'name': 'rolling_stat', 'params': {'source_col_name': 'log_return_close', 'window': 20, 'stat': 'std'}},
            {'name': 'rolling_stat', 'params': {'source_col_name': 'log_return_close', 'window': 20, 'stat': 'skew'}},
            {'name': 'rolling_stat', 'params': {'source_col_name': 'log_return_close', 'window': 20, 'stat': 'kurt'}},
        ],
        'medium': [ # Same features, but calculated on medium bars
            {'name': 'sma', 'params': {'timeperiod': 10, 'price_col': 'close'}},
            {'name': 'sma', 'params': {'timeperiod': 20, 'price_col': 'close'}},
            {'name': 'ema', 'params': {'timeperiod': 10, 'price_col': 'close'}},
            {'name': 'rsi', 'params': {'timeperiod': 14, 'price_col': 'close'}},
            {'name': 'macd', 'params': {'price_col': 'close'}},
            {'name': 'adx', 'params': {'timeperiod': 14}},
            {'name': 'atr', 'params': {'timeperiod': 14}},
            {'name': 'obv', 'params': {}},
            {'name': 'log_return', 'params': {'price_col': 'close', 'type': 'close'}},
            {'name': 'volatility_log_returns', 'params': {'window': 20, 'price_col': 'close'}},
            {'name': 'lagged_feature', 'params': {'source_col_name': 'log_return_close', 'lags': [1, 2, 3]}},
        ],
        'long': [ # And long bars
            {'name': 'sma', 'params': {'timeperiod': 10, 'price_col': 'close'}},
            {'name': 'sma', 'params': {'timeperiod': 20, 'price_col': 'close'}},
            {'name': 'ema', 'params': {'timeperiod': 10, 'price_col': 'close'}},
            {'name': 'rsi', 'params': {'timeperiod': 14, 'price_col': 'close'}},
            {'name': 'macd', 'params': {'price_col': 'close'}},
            {'name': 'adx', 'params': {'timeperiod': 14}},
            {'name': 'atr', 'params': {'timeperiod': 14}},
            {'name': 'obv', 'params': {}},
            {'name': 'log_return', 'params': {'price_col': 'close', 'type': 'close'}},
            {'name': 'volatility_log_returns', 'params': {'window': 20, 'price_col': 'close'}},
            {'name': 'lagged_feature', 'params': {'source_col_name': 'log_return_close', 'lags': [1, 2, 3]}},
        ]
    }

    # 3. Instantiate and run MinotaurFeatureEngine
    logging.info("Instantiating MinotaurFeatureEngine...")
    feature_engine = MinotaurFeatureEngine(
        short_bar_agg_config=short_bar_config,
        medium_bar_agg_config=medium_bar_config,
        long_bar_agg_config=long_bar_config,
        feature_configs=feature_configs,
        # Assuming default values for other engine params like intra-bar aggregation methods
    )

    logging.info("Calling process_historical_batch on the full dataset...")
    # Pass a copy to avoid SettingWithCopyWarning if the engine modifies the input df
    df_s, df_m, df_l = feature_engine.process_historical_batch(short_bars_df.copy())
    logging.info("process_historical_batch completed.")

    # 4. Merge the feature DataFrames
    logging.info("Merging short, medium, and long feature DataFrames...")

    # df_s is our base. It has the most frequent timestamps.
    # All features in df_s, df_m, df_l are already prefixed (s_, m_, l_) by the engine.
    # We need to merge df_m and df_l into df_s based on 'close_timestamp'.
    # Since medium and long bars are less frequent, we use merge_asof to forward-fill their features.

    # Columns to keep from the original short_bars_df (these are not prefixed by the engine's feature calculation)
    # These are the raw inputs to the feature engine for the short resolution.
    # We need to decide which of these to keep in the final merged output.
    # For now, let's keep all original short bar columns and then add medium/long features.
    # The engine prefixes its calculated features, so there shouldn't be direct name collisions for *features*.
    # However, 'open', 'high', 'low', 'close', 'volume', 'dollar_volume', 'num_ticks', 
    # 'open_timestamp', 'close_timestamp' exist in df_s, df_m, and df_l *before* feature calculation.
    # The engine adds prefixed features. Let's ensure the final df has one set of primary OHLCV and timestamps.
    
    # df_s already contains the features prefixed with 's_' AND the original short bar columns.
    # df_m contains features prefixed with 'm_' AND the original medium bar columns (open, high, etc. for medium bars).
    # df_l contains features prefixed with 'l_' AND the original long bar columns.

    # Select only the prefixed feature columns from df_m and df_l to avoid redundant OHLCV etc.
    # Also keep 'close_timestamp' for merging.
    m_features_to_merge = [col for col in df_m.columns if col.startswith('m_')] + ['close_timestamp']
    df_m_subset = df_m[m_features_to_merge].copy()
    
    l_features_to_merge = [col for col in df_l.columns if col.startswith('l_')] + ['close_timestamp']
    df_l_subset = df_l[l_features_to_merge].copy()

    # Ensure timestamps are sorted for merge_asof
    df_s.sort_values('close_timestamp', inplace=True)
    df_m_subset.sort_values('close_timestamp', inplace=True)
    df_l_subset.sort_values('close_timestamp', inplace=True)

    # Merge medium features into short
    # direction='backward' will take the next available medium bar's features if current short bar is past the medium bar
    # direction='forward' will take the previous
    # direction='nearest'
    # We want features from the most RECENTLY COMPLETED medium/long bar.
    # So, if short bar closes at T_s, we want features from medium bar that closed at or before T_s.
    # `merge_asof` default direction is 'backward' which means for each left row, we find the right row 
    # with the largest on_key value less than or equal to the left's on_key value. This is what we want.
    
    merged_df = pd.merge_asof(
        df_s, 
        df_m_subset, 
        on='close_timestamp',
        direction='backward' # find largest m_timestamp <= s_timestamp
    )
    logging.info(f"Shape after merging medium features: {merged_df.shape}")

    # Merge long features into the result
    merged_df = pd.merge_asof(
        merged_df,
        df_l_subset,
        on='close_timestamp',
        direction='backward'
    )
    logging.info(f"Shape after merging long features: {merged_df.shape}")
    
    # After merge_asof, there might be NaNs at the beginning where no prior medium/long bar exists.
    # This is expected.

    # 5. Save the final DataFrame
    logging.info(f"Saving final merged DataFrame to {output_file_path}...")
    try:
        merged_df.to_parquet(output_file_path, index=False)
        logging.info("Successfully saved the final DataFrame.")
    except Exception as e:
        logging.error(f"Error saving final DataFrame: {e}")
        return

    logging.info("Historical feature generation completed.")
    logging.info(f"Final DataFrame shape: {merged_df.shape}")
    logging.info(f"Final DataFrame columns: {merged_df.columns.tolist()}")
    
    # Print some info about NaNs introduced by merge_asof for medium/long features
    # This is expected at the start of the series
    for res_prefix in ['m_', 'l_']:
        cols_to_check = [col for col in merged_df.columns if col.startswith(res_prefix)]
        if cols_to_check:
            nan_counts = merged_df[cols_to_check].isnull().sum().sum()
            first_valid_indices = {}
            for col in cols_to_check:
                first_valid_idx = merged_df[col].first_valid_index()
                if first_valid_idx is not None:
                    first_valid_indices[col] = first_valid_idx
            
            min_first_valid = min(first_valid_indices.values()) if first_valid_indices else 'N/A'
            logging.info(f"Total NaNs in '{res_prefix}' prefixed columns: {nan_counts}. First valid index for any '{res_prefix}' column: {min_first_valid}")


if __name__ == "__main__":
    # For a smoke test with limited files:
    # run_historical_feature_generation(limit_files=3)
    # To run on the full dataset, call:
    run_historical_feature_generation() 