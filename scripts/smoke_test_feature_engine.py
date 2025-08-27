import pandas as pd
from minotaur.scripts.minotaur_feature_engine import MinotaurFeatureEngine
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_smoke_test():
    """
    Runs a smoke test for the MinotaurFeatureEngine.
    """
    logging.info("Starting feature engine smoke test...")

    # Define parameters
    # short_bar_threshold = 2_000_000 # Original parameter, now part of config
    medium_factor = 10
    long_factor = 16 # This is applied to medium bars to get long bars

    # Define aggregation configurations
    short_bar_config = {
        'name': 'short_2M_dollar_bars' # Identifier for the input short bars
    }
    medium_bar_config = {
        'agg_type': 'bar_count',       # Aggregate based on number of constituent bars
        'bars_per_agg': medium_factor, # Number of short bars to make one medium bar
        'agg_col': 'dollar_volume'     # Column to monitor for threshold if agg_type was 'dollar_value' (not used for 'bar_count')
                                       # but good to have as a placeholder or if logic evolves
    }
    # Long bars are aggregated from medium bars.
    # The 'long_factor' here means 1 long bar = 'long_factor' medium bars.
    long_bar_config = {
        'agg_type': 'bar_count',
        'bars_per_agg': long_factor,  # Number of medium bars to make one long bar
        'agg_col': 'dollar_volume'    # Placeholder, same as above
    }

    # Define feature configurations (minimal for smoke test)
    feature_configs = {
        'short': [
            # TA-Lib
            {'name': 'sma', 'params': {'timeperiod': 10, 'price_col': 'close'}},
            {'name': 'sma', 'params': {'timeperiod': 20, 'price_col': 'close'}},
            {'name': 'ema', 'params': {'timeperiod': 10, 'price_col': 'close'}},
            {'name': 'rsi', 'params': {'timeperiod': 14, 'price_col': 'close'}},
            {'name': 'macd', 'params': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9, 'price_col': 'close'}},
            {'name': 'adx', 'params': {'timeperiod': 14}}, # Uses HLC
            {'name': 'plus_di', 'params': {'timeperiod': 14}}, # Uses HLC
            {'name': 'minus_di', 'params': {'timeperiod': 14}}, # Uses HLC
            {'name': 'stoch', 'params': {'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3}}, # Uses HLC
            {'name': 'atr', 'params': {'timeperiod': 14}}, # Uses HLC
            {'name': 'bbands', 'params': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2, 'price_col': 'close'}},
            {'name': 'cci', 'params': {'timeperiod': 14}}, # Uses HLC
            {'name': 'mfi', 'params': {'timeperiod': 14}}, # Uses HLCV
            {'name': 'roc', 'params': {'timeperiod': 10, 'price_col': 'close'}},
            {'name': 'obv', 'params': {}}, # Uses Close, Volume, ensure params exists
            {'name': 'ultosc', 'params': {'timeperiod1': 7, 'timeperiod2': 14, 'timeperiod3': 28}}, # Uses HLC
            {'name': 'willr', 'params': {'timeperiod': 14}}, # Uses HLC
            {'name': 'volume_sma', 'params': {'timeperiod': 20}}, # Uses Volume

            # Price-derived (log_return_close is already a base feature, this is to ensure mapping works)
            {'name': 'log_return', 'params': {'price_col': 'close', 'type': 'close'}},
            # Volatility (log_return_close is used internally)
            {'name': 'volatility_close', 'params': {'timeperiod': 20}},
            {'name': 'volatility_high_low', 'params': {'timeperiod': 20}},
        ],
        'medium': [
            # TA-Lib
            {'name': 'sma', 'params': {'timeperiod': 10, 'price_col': 'close'}}, # Medium-term SMA
            {'name': 'sma', 'params': {'timeperiod': 30, 'price_col': 'close'}},
            {'name': 'ema', 'params': {'timeperiod': 20, 'price_col': 'close'}},
            {'name': 'rsi', 'params': {'timeperiod': 14, 'price_col': 'close'}},
            {'name': 'macd', 'params': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9, 'price_col': 'close'}},
            {'name': 'adx', 'params': {'timeperiod': 14}},
            {'name': 'atr', 'params': {'timeperiod': 14}},
            {'name': 'bbands', 'params': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2, 'price_col': 'close'}},
            {'name': 'mfi', 'params': {'timeperiod': 14}},
            {'name': 'obv', 'params': {}},

            # Volatility
            {'name': 'volatility_close', 'params': {'timeperiod': 20}},
            {'name': 'volatility_high_low', 'params': {'timeperiod': 20}},
        ],
        'long': [
            # TA-Lib
            {'name': 'sma', 'params': {'timeperiod': 20, 'price_col': 'close'}}, # Long-term SMA
            {'name': 'sma', 'params': {'timeperiod': 50, 'price_col': 'close'}},
            {'name': 'ema', 'params': {'timeperiod': 50, 'price_col': 'close'}},
            {'name': 'rsi', 'params': {'timeperiod': 14, 'price_col': 'close'}},
            {'name': 'macd', 'params': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9, 'price_col': 'close'}},
            {'name': 'adx', 'params': {'timeperiod': 14}},
            {'name': 'atr', 'params': {'timeperiod': 14}},
            {'name': 'mfi', 'params': {'timeperiod': 14}},
            {'name': 'obv', 'params': {}},

            # Volatility
            {'name': 'volatility_close', 'params': {'timeperiod': 20}},
            {'name': 'volatility_high_low', 'params': {'timeperiod': 20}},
        ]
    }
    
    # Construct the path to the test file
    # Assuming the script is run from the root of the trading-agent-v3 directory
    base_path = os.path.join("minotaur", "data", "dollar_bars", "2M")
    test_file = os.path.join(base_path, "BTCUSDT_2017_08_dollar_bars_2M.parquet")
    
    logging.info(f"Attempting to load data from: {test_file}")

    if not os.path.exists(test_file):
        logging.error(f"Test file not found: {test_file}")
        logging.error(f"Current working directory: {os.getcwd()}")
        # List contents of relevant directories to help debug
        logging.error(f"Contents of 'minotaur/data/': {os.listdir(os.path.join('minotaur', 'data')) if os.path.exists(os.path.join('minotaur', 'data')) else 'Not found'}")
        logging.error(f"Contents of 'minotaur/data/dollar_bars/': {os.listdir(os.path.join('minotaur', 'data', 'dollar_bars')) if os.path.exists(os.path.join('minotaur', 'data', 'dollar_bars')) else 'Not found'}")
        logging.error(f"Contents of 'minotaur/data/dollar_bars/2M/': {os.listdir(base_path) if os.path.exists(base_path) else 'Not found'}")
        return

    try:
        # 1. Load a small sample of data
        logging.info(f"Loading data from {test_file}...")
        short_bars_df = pd.read_parquet(test_file)
        logging.info(f"Loaded {len(short_bars_df)} rows from {test_file}.")
        logging.info(f"Columns in loaded data: {short_bars_df.columns.tolist()}")
        
        # Ensure necessary columns for aggregation are present
        required_cols = ['open_timestamp', 'open', 'high', 'low', 'close', 'volume', 
                         'dollar_volume', 'num_ticks', 'close_timestamp', 
                         'trade_imbalance', 'intra_bar_tick_price_volatility', 
                         'taker_buy_sell_ratio', 'tick_price_skewness', 
                         'tick_price_kurtosis', 'num_directional_changes']
        
        missing_cols = [col for col in required_cols if col not in short_bars_df.columns]
        if missing_cols:
            logging.error(f"Missing required columns in the input Parquet file: {missing_cols}")
            # Log available columns for easier debugging
            logging.error(f"Available columns: {short_bars_df.columns.tolist()}")
            return

        # Ensure 'close_timestamp' and 'open_timestamp' are datetime objects
        if not pd.api.types.is_datetime64_any_dtype(short_bars_df['close_timestamp']):
            short_bars_df['close_timestamp'] = pd.to_datetime(short_bars_df['close_timestamp'])
        if not pd.api.types.is_datetime64_any_dtype(short_bars_df['open_timestamp']):
            short_bars_df['open_timestamp'] = pd.to_datetime(short_bars_df['open_timestamp'])


        # 2. Instantiate the MinotaurFeatureEngine
        logging.info("Instantiating MinotaurFeatureEngine...")
        feature_engine = MinotaurFeatureEngine(
            short_bar_agg_config=short_bar_config,
            medium_bar_agg_config=medium_bar_config,
            long_bar_agg_config=long_bar_config,
            feature_configs=feature_configs
        )
        logging.info("MinotaurFeatureEngine instantiated.")

        # 3. Call the process_historical_batch method
        logging.info("Calling process_historical_batch...")
        # Pass a copy to avoid SettingWithCopyWarning if the engine modifies the input df
        # featured_df = feature_engine.process_historical_batch(short_bars_df.copy())
        df_s, df_m, df_l = feature_engine.process_historical_batch(short_bars_df.copy())
        logging.info("process_historical_batch completed.")

        # 4. Print the shape and column names of the resulting DataFrames
        logging.info("--- Short Bar Features ---")
        if df_s is not None and not df_s.empty:
            logging.info(f"Shape of the short featured DataFrame: {df_s.shape}")
            logging.info(f"Columns in the short featured DataFrame: {df_s.columns.tolist()}")
            nan_counts_s = df_s.isnull().sum()
            nan_summary_s = nan_counts_s[nan_counts_s > 0]
            if not nan_summary_s.empty:
                logging.warning("NaN values found in the short featured DataFrame:")
                logging.warning(nan_summary_s)
            else:
                logging.info("No NaN values found in the short featured DataFrame.")
        else:
            logging.info("Short featured DataFrame is None or empty.")

        logging.info("--- Medium Bar Features ---")
        if df_m is not None and not df_m.empty:
            logging.info(f"Shape of the medium featured DataFrame: {df_m.shape}")
            logging.info(f"Columns in the medium featured DataFrame: {df_m.columns.tolist()}")
            nan_counts_m = df_m.isnull().sum()
            nan_summary_m = nan_counts_m[nan_counts_m > 0]
            if not nan_summary_m.empty:
                logging.warning("NaN values found in the medium featured DataFrame:")
                logging.warning(nan_summary_m)
            else:
                logging.info("No NaN values found in the medium featured DataFrame.")
        else:
            logging.info("Medium featured DataFrame is None or empty.")

        logging.info("--- Long Bar Features ---")
        if df_l is not None and not df_l.empty:
            logging.info(f"Shape of the long featured DataFrame: {df_l.shape}")
            logging.info(f"Columns in the long featured DataFrame: {df_l.columns.tolist()}")
            nan_counts_l = df_l.isnull().sum()
            nan_summary_l = nan_counts_l[nan_counts_l > 0]
            if not nan_summary_l.empty:
                logging.warning("NaN values found in the long featured DataFrame:")
                logging.warning(nan_summary_l)
            else:
                logging.info("No NaN values found in the long featured DataFrame.")
        else:
            logging.info("Long featured DataFrame is None or empty (as expected for small input).")

        logging.info("Smoke test completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during the smoke test: {e}", exc_info=True)

if __name__ == "__main__":
    run_smoke_test() 