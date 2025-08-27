import pandas as pd
import numpy as np
import logging
import talib # Ensure TA-Lib is imported

# Configure the logger
# You might want to make this configurable or inherit from a project-level logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

class MinotaurFeatureEngine:
    def __init__(self, short_bar_agg_config, medium_bar_agg_config, long_bar_agg_config, feature_configs):
        """
        Initializes the MinotaurFeatureEngine.

        Args:
            short_bar_agg_config (dict): Config for how short bars are defined (e.g., their input name if pre-aggregated).
                                       Not used for aggregation FROM short bars, but for identifying them.
            medium_bar_agg_config (dict): Configuration for aggregating to medium dollar bars.
                                      Example: {'agg_from_col': 'quote_quantity', 'threshold': 20000000, 'bars_per_agg': 10, 'agg_type': 'bar_count' or 'dollar_value'}
                                      'threshold' is for dollar_value type, 'bars_per_agg' for bar_count type.
            long_bar_agg_config (dict): Configuration for aggregating to long dollar bars.
                                     Example: {'agg_from_col': 'quote_quantity', 'threshold': 160000000, 'bars_per_agg': 16, 'agg_type': 'bar_count' or 'dollar_value'}
            feature_configs (dict): Dictionary specifying features to calculate for each bar type ('short', 'medium', 'long').
                                    Features are dictionaries with 'name' and other params.
                                    Example: {
                                        'short': [{'name': 'sma', 'period': 10}, {'name': 'rsi', 'period': 14}],
                                        'medium': [{'name': 'sma', 'period': 20}],
                                        'long': [{'name': 'atr', 'period': 14}]
                                    }
        """
        self.logger = logger
        self.logger.info("Initializing MinotaurFeatureEngine...")

        self.short_bar_agg_config = short_bar_agg_config # May not be strictly needed if short bars are always the direct input
        self.medium_bar_agg_config = medium_bar_agg_config
        self.long_bar_agg_config = long_bar_agg_config
        self.feature_configs = feature_configs

        # Buffers for live processing (can be adapted for batch)
        self.short_bars_buffer = pd.DataFrame()
        self.medium_bars_buffer = pd.DataFrame()
        self.long_bars_buffer = pd.DataFrame()

        self.current_medium_bar_accumulator = []
        self.current_medium_bar_value_sum = 0.0
        self.current_long_bar_accumulator = []
        self.current_long_bar_value_sum = 0.0
        
        self.latest_features_short = pd.Series(dtype='float64')
        self.latest_features_medium = pd.Series(dtype='float64')
        self.latest_features_long = pd.Series(dtype='float64')

        self.logger.info("MinotaurFeatureEngine initialized.")

    def _aggregate_bars(self, new_constituent_bar_df, bar_type_to_form, config):
        """
        Aggregates constituent bars to form a new, coarser bar (medium or long).
        This version is designed for batch processing primarily.

        Args:
            new_constituent_bar_df (pd.DataFrame): DataFrame of constituent bars (e.g., all short bars to form medium bars).
            bar_type_to_form (str): 'medium' or 'long'.
            config (dict): The aggregation configuration for the target bar type.
                           Example: {'agg_col': 'quote_quantity', 'threshold': 20000000, 'agg_type': 'dollar_value'}
                           OR {'bars_per_agg': 10, 'agg_type': 'bar_count'}

        Returns:
            pd.DataFrame: DataFrame of newly formed aggregated bars.
        """
        self.logger.info(f"Aggregating to form {bar_type_to_form} bars...")
        if new_constituent_bar_df.empty:
            return pd.DataFrame()

        aggregated_bars_list = []
        accumulator = []
        current_value_sum = 0.0 # For dollar_value aggregation

        # Ensure timestamps are datetime objects for calculations
        if 'open_timestamp' in new_constituent_bar_df.columns and not pd.api.types.is_datetime64_any_dtype(new_constituent_bar_df['open_timestamp']):
            new_constituent_bar_df['open_timestamp'] = pd.to_datetime(new_constituent_bar_df['open_timestamp'])
        if 'close_timestamp' in new_constituent_bar_df.columns and not pd.api.types.is_datetime64_any_dtype(new_constituent_bar_df['close_timestamp']):
            new_constituent_bar_df['close_timestamp'] = pd.to_datetime(new_constituent_bar_df['close_timestamp'])


        for _, row in new_constituent_bar_df.iterrows():
            accumulator.append(row)
            
            should_form_bar = False
            if config['agg_type'] == 'bar_count':
                if len(accumulator) >= config['bars_per_agg']:
                    should_form_bar = True
            elif config['agg_type'] == 'dollar_value':
                current_value_sum += row[config['agg_col']]
                if current_value_sum >= config['threshold']:
                    should_form_bar = True
            
            if should_form_bar:
                agg_df = pd.DataFrame(accumulator)
                
                open_price = agg_df['open'].iloc[0]
                high_price = agg_df['high'].max()
                low_price = agg_df['low'].min()
                close_price = agg_df['close'].iloc[-1]
                total_volume = agg_df['volume'].sum() # Base asset volume
                total_quote_quantity = agg_df['dollar_volume'].sum() # This is our dollar_volume for the bar
                num_ticks_sum = agg_df['num_ticks'].sum()
                
                start_ts = agg_df['open_timestamp'].iloc[0]
                end_ts = agg_df['close_timestamp'].iloc[-1]
                duration = (end_ts - start_ts).total_seconds() if pd.notnull(start_ts) and pd.notnull(end_ts) else np.nan

                # --- Aggregate Intra-Bar Features (Based on o3's research) ---
                aggregated_intra_bar_features = {}

                # 1. trade_imbalance: Sum
                if 'trade_imbalance' in agg_df.columns:
                    aggregated_intra_bar_features['trade_imbalance'] = agg_df['trade_imbalance'].sum()

                # 2. intra_bar_tick_price_volatility: Volume-weighted pooling of variance
                if 'intra_bar_tick_price_volatility' in agg_df.columns and 'dollar_volume' in agg_df.columns:
                    # Ensure dollar_volume is not zero to avoid division by zero if a constituent bar had no volume (unlikely for dollar bars but good practice)
                    valid_vols = agg_df['dollar_volume'][agg_df['dollar_volume'] > 0]
                    if not valid_vols.empty:
                        sum_weighted_variance = (agg_df['intra_bar_tick_price_volatility']**2 * agg_df['dollar_volume']).sum()
                        total_dollar_volume_for_weighting = agg_df['dollar_volume'].sum() # Use the sum of dollar volumes of constituents
                        if total_dollar_volume_for_weighting > 0:
                            pooled_variance = sum_weighted_variance / total_dollar_volume_for_weighting
                            aggregated_intra_bar_features['intra_bar_tick_price_volatility'] = np.sqrt(pooled_variance)
                        else:
                            aggregated_intra_bar_features['intra_bar_tick_price_volatility'] = np.nan # Or 0, if sum of volumes is 0
                    else:
                        aggregated_intra_bar_features['intra_bar_tick_price_volatility'] = np.nan
                
                # 3. taker_buy_sell_ratio: Recompute from aggregated taker volumes
                if 'taker_buy_sell_ratio' in agg_df.columns and 'dollar_volume' in agg_df.columns:
                    # r = B / S, V = B + S  => B = rV / (1+r), S = V / (1+r)
                    # Handle cases where r = -1 (if S=0 and B > 0, not typical from exchange but good to be safe for ratio math)
                    # or r is very large (S is near zero)
                    # For simplicity, if 1+r is zero (r=-1), implies S=0, B=V. If r is inf, S=0, B=V.
                    # A ratio of 0 means B=0, S=V.
                    r_plus_1 = 1 + agg_df['taker_buy_sell_ratio']
                    # Avoid division by zero if r_plus_1 is 0 (i.e., ratio is -1, which is unusual)
                    # Replace 0 in r_plus_1 with a very small number or handle as a special case.
                    # For now, let's assume taker_buy_sell_ratio is non-negative as is typical.
                    
                    taker_buy_dollar_volume = (agg_df['taker_buy_sell_ratio'] * agg_df['dollar_volume']) / r_plus_1
                    taker_sell_dollar_volume = agg_df['dollar_volume'] / r_plus_1
                    
                    # Handle potential NaNs if ratio was NaN or inf resulting in NaN components
                    total_tb_dv = taker_buy_dollar_volume.sum()
                    total_ts_dv = taker_sell_dollar_volume.sum()

                    if pd.notna(total_tb_dv) and pd.notna(total_ts_dv):
                        if total_ts_dv > 1e-9: # Avoid division by zero or extremely small denominator
                            aggregated_intra_bar_features['taker_buy_sell_ratio'] = total_tb_dv / total_ts_dv
                        elif total_tb_dv > 1e-9: # Sells are zero/tiny, buys are not
                            aggregated_intra_bar_features['taker_buy_sell_ratio'] = np.finfo(float).max # Represent as very large
                        else: # Both are zero/tiny
                            aggregated_intra_bar_features['taker_buy_sell_ratio'] = 1.0 # Or np.nan, debatable what 0/0 should be here, 1.0 implies balance
                    else:
                        aggregated_intra_bar_features['taker_buy_sell_ratio'] = np.nan

                # 4. tick_price_skewness: Dollar volume-weighted average
                if 'tick_price_skewness' in agg_df.columns and 'dollar_volume' in agg_df.columns:
                    weighted_skewness_sum = (agg_df['tick_price_skewness'] * agg_df['dollar_volume']).sum()
                    total_dv_for_weighting = agg_df['dollar_volume'].sum()
                    if total_dv_for_weighting > 0:
                        aggregated_intra_bar_features['tick_price_skewness'] = weighted_skewness_sum / total_dv_for_weighting
                    else:
                        aggregated_intra_bar_features['tick_price_skewness'] = np.nan

                # 5. tick_price_kurtosis: Dollar volume-weighted average
                if 'tick_price_kurtosis' in agg_df.columns and 'dollar_volume' in agg_df.columns:
                    weighted_kurtosis_sum = (agg_df['tick_price_kurtosis'] * agg_df['dollar_volume']).sum()
                    total_dv_for_weighting = agg_df['dollar_volume'].sum()
                    if total_dv_for_weighting > 0:
                        aggregated_intra_bar_features['tick_price_kurtosis'] = weighted_kurtosis_sum / total_dv_for_weighting
                    else:
                        aggregated_intra_bar_features['tick_price_kurtosis'] = np.nan
                
                # 6. num_price_changes: Omitted for now as per o3's advice on difficulty of accurate aggregation

                # 7. num_directional_changes: Sum + adjustments for cross-bar transitions
                if 'num_directional_changes' in agg_df.columns:
                    sum_internal_changes = agg_df['num_directional_changes'].sum()
                    boundary_changes = 0
                    # Iterate through adjacent pairs in the accumulator (agg_df)
                    for i in range(len(agg_df) - 1):
                        bar1_close = agg_df['close'].iloc[i]
                        bar1_open = agg_df['open'].iloc[i]
                        bar2_close = agg_df['close'].iloc[i+1]
                        bar2_open = agg_df['open'].iloc[i+1]
                        
                        # Heuristic: use net change of bar1 and net change of bar2 at the boundary.
                        # More advanced: use last tick direction of bar1 and first tick direction of bar2 if available.
                        # Using sign of (close - open) as proxy for bar's net direction.
                        # A change in sign of net direction implies a boundary directional change.
                        sign_bar1_net_change = np.sign(bar1_close - bar1_open)
                        # For bar2, consider its direction relative to bar1's close, or its own net change.
                        # Let's use bar2's own net change relative to its open.
                        sign_bar2_net_change = np.sign(bar2_close - bar2_open)
                        
                        # Avoid counting a change if one of the bars is flat (sign=0)
                        if sign_bar1_net_change != 0 and sign_bar2_net_change != 0 and sign_bar1_net_change != sign_bar2_net_change:
                            boundary_changes += 1
                    aggregated_intra_bar_features['num_directional_changes'] = sum_internal_changes + boundary_changes

                aggregated_bars_list.append({
                    'open_timestamp': start_ts,
                    'close_timestamp': end_ts,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': total_volume,
                    'dollar_volume': total_quote_quantity, # Changed from 'quote_quantity' to 'dollar_volume'
                    'num_ticks': num_ticks_sum,
                    'bar_duration_seconds': duration,
                    'constituent_bar_count': len(accumulator),
                    **aggregated_intra_bar_features # Unpack the newly aggregated features here
                })
                accumulator = []
                current_value_sum = 0.0 # Reset for dollar_value

        if not aggregated_bars_list:
             self.logger.warning(f"No {bar_type_to_form} bars were formed during aggregation. Check input data and config.")
             return pd.DataFrame()
             
        return pd.DataFrame(aggregated_bars_list)

    def _calculate_features_for_resolution(self, df_bars, prefix, feature_config_list):
        """
        Calculates a defined set of features for a given DataFrame of bars.

        Args:
            df_bars (pd.DataFrame): DataFrame with OHLCV and other base data.
                                   Must have 'open', 'high', 'low', 'close', 'volume'.
                                   'close_timestamp' required for time-based features.
            prefix (str): 's_', 'm_', or 'l_' to prepend to feature names.
            feature_config_list (list): List of feature configuration dicts.
                                      e.g., [{'name': 'sma', 'period': 10, 'col': 'close'}, ...]

        Returns:
            pd.DataFrame: DataFrame with original columns plus new feature columns.
        """
        if df_bars.empty:
            self.logger.warning(f"Input DataFrame for prefix '{prefix}' is empty. Skipping feature calculation.")
            return df_bars
        
        self.logger.info(f"Calculating features for prefix '{prefix}' on {len(df_bars)} bars...")
        out_df = df_bars.copy()

        # Ensure necessary columns exist for basic operations and TA-Lib
        required_cols = ['open', 'high', 'low', 'close', 'volume'] # dollar_volume handled separately for log
        for col in required_cols:
            if col not in out_df.columns:
                self.logger.error(f"Missing required column '{col}' for prefix '{prefix}'. Cannot calculate many features.")
                # Depending on strictness, could return df_bars or raise error.
                # For now, proceed and let individual features fail if they need it.
        
        # --- Log Transform Volume and Dollar Volume (NEW) ---
        # Using np.log1p for log(1+x) to handle potential zeros gracefully.
        if 'volume' in out_df.columns:
            out_df[f'{prefix}volume_log'] = np.log1p(out_df['volume'])
            self.logger.info(f"Calculated {prefix}volume_log for {prefix} bars.")
        else:
            self.logger.warning(f"'volume' column not found in {prefix} bars. Cannot calculate {prefix}volume_log.")

        if 'dollar_volume' in out_df.columns: # Assuming 'dollar_volume' is the name in the input df_bars
            out_df[f'{prefix}dollar_volume_log'] = np.log1p(out_df['dollar_volume'])
            self.logger.info(f"Calculated {prefix}dollar_volume_log for {prefix} bars.")
        else:
            self.logger.warning(f"'dollar_volume' column not found in {prefix} bars. Cannot calculate {prefix}dollar_volume_log.")
        # --- End Log Transform ---

        # Base Features (Section I from plan) & Global Price-Based (Section IV items 1-6)
        # These are fundamental and calculated upfront.
        if 'close' in out_df.columns:
            out_df[f'{prefix}log_return_close'] = np.log(out_df['close'] / out_df['close'].shift(1))
        if 'high' in out_df.columns:
            out_df[f'{prefix}log_return_high'] = np.log(out_df['high'] / out_df['high'].shift(1))
        if 'low' in out_df.columns:
            out_df[f'{prefix}log_return_low'] = np.log(out_df['low'] / out_df['low'].shift(1))
        
        if 'open' in out_df.columns and 'close' in out_df.columns:
            out_df[f'{prefix}price_change_pct'] = (out_df['close'] - out_df['open']) / out_df['open']
        if 'high' in out_df.columns and 'low' in out_df.columns:
            out_df[f'{prefix}high_low_pct'] = (out_df['high'] - out_df['low']) / out_df['low']
            # Handle potential division by zero for wick/body vs range
            range_val = (out_df['high'] - out_df['low'])
            if 'open' in out_df.columns and 'close' in out_df.columns:
                out_df[f'{prefix}wick_vs_range'] = np.where(range_val > 0, ((out_df['high'] - out_df['low']) - abs(out_df['open'] - out_df['close'])) / range_val, 0)
                out_df[f'{prefix}body_vs_range'] = np.where(range_val > 0, abs(out_df['open'] - out_df['close']) / range_val, 0)
        
        if 'close' in out_df.columns and 'high' in out_df.columns:
             out_df[f'{prefix}close_vs_high_pct'] = (out_df['close'] - out_df['high']) / out_df['high']
        if 'close' in out_df.columns and 'low' in out_df.columns:
             out_df[f'{prefix}close_vs_low_pct'] = (out_df['close'] - out_df['low']) / out_df['low']
        
        out_df.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle divisions by zero from pct calculations

        # --- Feature Calculation Loop (for TA-Lib and other configured features) ---
        for feature_conf in feature_config_list:
            feature_name = feature_conf['name']
            params = feature_conf.get('params', {}) 
            
            # Default input column for many indicators
            input_col_name = params.get('price_col', 'close') # some features use 'close', others 'volume', etc.
                                                          # TA-Lib functions often take specific columns directly (high, low, close, volume)

            # Construct base column name (e.g., s_sma, m_rsi)
            base_col_name = f"{prefix}{feature_name}"
            
            # Specific naming logic for different feature types
            final_col_name = base_col_name # Default
            if feature_name in ['sma', 'ema', 'rsi', 'adx', 'plus_di', 'minus_di', 'atr', 'cci', 'mfi', 'roc', 'willr', 'volume_sma'] and 'timeperiod' in params:
                final_col_name = f"{base_col_name}_{params['timeperiod']}"
            elif feature_name == 'bbands' and 'timeperiod' in params: # BBands creates multiple output columns
                bb_period = params['timeperiod']
                # Names are handled inside the bbands block
            elif feature_name == 'stoch': # STOCH creates multiple output columns
                 # Names are handled inside the STOCH block
                 pass
            elif feature_name == 'ultosc': # ULTOSC uses multiple timeperiods but one output
                 # Name uses base_col_name, periods are params
                 pass
            elif feature_name in ['volatility_log_returns', 'volatility_hl_log_returns'] and 'window' in params:
                 final_col_name = f"{base_col_name}_{params['window']}"
            # lagged_feature and rolling_stat will construct their own names within their blocks.
            # Features like 'macd', 'obv', 'time_features', 'log_return' that don't fit simple period naming are handled by base_col_name or inside their blocks.

            try:
                # Check for necessary input columns for the specific feature
                # This is a basic check; TA-Lib functions have their own internal length requirements.
                if input_col_name not in out_df.columns and feature_name not in ['macd', 'stoch', 'bbands', 'adx', 'plus_di', 'minus_di', 'atr', 'cci', 'mfi', 'ultosc', 'willr', 'obv', 'time_features', 'lagged_feature', 'rolling_stat', 'volatility_hl_log_returns']:
                    if not (feature_name == 'volume_sma' and 'volume' in out_df.columns): # volume_sma uses 'volume' directly
                        self.logger.warning(f"Input column '{input_col_name}' for feature '{feature_name}' not found. Skipping '{final_col_name}'.")
                        continue
                
                # TA-Lib Features
                if feature_name == 'sma':
                    if len(out_df) >= params['timeperiod'] and input_col_name in out_df:
                        out_df[final_col_name] = talib.SMA(out_df[input_col_name], timeperiod=params['timeperiod'])
                elif feature_name == 'ema':
                    if len(out_df) >= params['timeperiod'] and input_col_name in out_df:
                        out_df[final_col_name] = talib.EMA(out_df[input_col_name], timeperiod=params['timeperiod'])
                elif feature_name == 'rsi':
                    if len(out_df) >= params['timeperiod'] and input_col_name in out_df:
                        out_df[final_col_name] = talib.RSI(out_df[input_col_name], timeperiod=params['timeperiod'])
                elif feature_name == 'macd':
                    fastperiod = params.get('fastperiod', 12)
                    slowperiod = params.get('slowperiod', 26)
                    signalperiod = params.get('signalperiod', 9)
                    if len(out_df) >= slowperiod and input_col_name in out_df:
                        macd_val, macdsignal_val, macdhist_val = talib.MACD(out_df[input_col_name], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
                        out_df[f'{prefix}macd'] = macd_val
                        out_df[f'{prefix}macd_signal'] = macdsignal_val
                        out_df[f'{prefix}macd_hist'] = macdhist_val
                elif feature_name == 'adx':
                    if len(out_df) >= params['timeperiod'] * 2 and all(c in out_df for c in ['high', 'low', 'close']): # ADX needs HLC
                        out_df[final_col_name] = talib.ADX(out_df['high'], out_df['low'], out_df['close'], timeperiod=params['timeperiod'])
                elif feature_name == 'plus_di':
                    if len(out_df) >= params['timeperiod'] * 2 and all(c in out_df for c in ['high', 'low', 'close']):
                        out_df[final_col_name] = talib.PLUS_DI(out_df['high'], out_df['low'], out_df['close'], timeperiod=params['timeperiod'])
                elif feature_name == 'minus_di':
                    if len(out_df) >= params['timeperiod'] * 2 and all(c in out_df for c in ['high', 'low', 'close']):
                        out_df[final_col_name] = talib.MINUS_DI(out_df['high'], out_df['low'], out_df['close'], timeperiod=params['timeperiod'])
                elif feature_name == 'stoch':
                    fastk_period = params.get('fastk_period', 5)
                    slowk_period = params.get('slowk_period', 3)
                    slowd_period = params.get('slowd_period', 3)
                    if len(out_df) >= fastk_period + slowk_period + slowd_period - 2 and all(c in out_df for c in ['high', 'low', 'close']): # Approx
                        slowk, slowd = talib.STOCH(out_df['high'], out_df['low'], out_df['close'],
                                                   fastk_period=fastk_period, slowk_period=slowk_period, slowd_period=slowd_period)
                        out_df[f'{prefix}stoch_k'] = slowk
                        out_df[f'{prefix}stoch_d'] = slowd
                elif feature_name == 'volume_sma': # Uses 'volume' directly
                    if len(out_df) >= params['timeperiod'] and 'volume' in out_df:
                        out_df[final_col_name] = talib.SMA(out_df['volume'], timeperiod=params['timeperiod'])
                elif feature_name == 'obv': # Uses 'close' and 'volume'
                    if all(c in out_df for c in ['close', 'volume']):
                        out_df[f'{prefix}obv'] = talib.OBV(out_df['close'], out_df['volume'])
                elif feature_name == 'atr':
                    if len(out_df) >= params['timeperiod'] and all(c in out_df for c in ['high', 'low', 'close']):
                        out_df[final_col_name] = talib.ATR(out_df['high'], out_df['low'], out_df['close'], timeperiod=params['timeperiod'])
                elif feature_name == 'bbands':
                    bb_period = params.get('timeperiod', 20)
                    nbdevup = params.get('nbdevup', 2)
                    nbdevdn = params.get('nbdevdn', 2)
                    if len(out_df) >= bb_period and input_col_name in out_df:
                        upper, middle, lower = talib.BBANDS(out_df[input_col_name], timeperiod=bb_period, nbdevup=nbdevup, nbdevdn=nbdevdn)
                        out_df[f'{prefix}bbands_upper_{bb_period}'] = upper
                        out_df[f'{prefix}bbands_middle_{bb_period}'] = middle
                        out_df[f'{prefix}bbands_lower_{bb_period}'] = lower
                elif feature_name == 'cci':
                     if len(out_df) >= params['timeperiod'] and all(c in out_df for c in ['high', 'low', 'close']):
                        out_df[final_col_name] = talib.CCI(out_df['high'], out_df['low'], out_df['close'], timeperiod=params['timeperiod'])
                elif feature_name == 'mfi':
                    if len(out_df) >= params['timeperiod'] and all(c in out_df for c in ['high', 'low', 'close', 'volume']):
                        out_df[final_col_name] = talib.MFI(out_df['high'], out_df['low'], out_df['close'], out_df['volume'], timeperiod=params['timeperiod'])
                elif feature_name == 'roc':
                    if len(out_df) > params['timeperiod'] and input_col_name in out_df:
                        out_df[final_col_name] = talib.ROC(out_df[input_col_name], timeperiod=params['timeperiod'])
                elif feature_name == 'ultosc':
                    timeperiod1 = params.get('timeperiod1', 7)
                    timeperiod2 = params.get('timeperiod2', 14)
                    timeperiod3 = params.get('timeperiod3', 28)
                    if len(out_df) >= timeperiod3 and all(c in out_df for c in ['high', 'low', 'close']):
                         out_df[f'{prefix}ultosc'] = talib.ULTOSC(out_df['high'], out_df['low'], out_df['close'],
                                                                timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)
                elif feature_name == 'willr':
                    if len(out_df) >= params['timeperiod'] and all(c in out_df for c in ['high', 'low', 'close']):
                        out_df[final_col_name] = talib.WILLR(out_df['high'], out_df['low'], out_df['close'], timeperiod=params['timeperiod'])
                
                # Custom/Other Features (already calculated base features are skipped here by not having explicit handling)
                elif feature_name == 'volatility_log_returns': 
                     window = params.get('window', 20)
                     price_col_for_lr = params.get('price_col', 'close') # Source for log return
                     log_return_col_to_use = f'{prefix}log_return_{price_col_for_lr}' # e.g. s_log_return_close
                     
                     if log_return_col_to_use not in out_df: # If the specific log return wasn't pre-calculated
                         if price_col_for_lr in out_df: # Check if the base price col (e.g. 'open') exists
                             out_df[log_return_col_to_use] = np.log(out_df[price_col_for_lr] / out_df[price_col_for_lr].shift(1))
                             self.logger.info(f"Dynamically calculated {log_return_col_to_use} for volatility.")
                         else:
                             self.logger.warning(f"Cannot calculate {final_col_name}: Source column '{price_col_for_lr}' for log return not found.")
                             out_df[final_col_name] = np.nan
                             continue # Skip this feature
                     
                     if len(out_df) >= window and log_return_col_to_use in out_df:
                        out_df[final_col_name] = out_df[log_return_col_to_use].rolling(window=window).std()
                     else:
                        out_df[final_col_name] = np.nan
                elif feature_name == 'volatility_hl_log_returns': # Volatility of log(high/low)
                    window = params.get('window', 20)
                    if len(out_df) >= window and 'high' in out_df and 'low' in out_df:
                        # Calculate log(high/low) directly, ensure low is not zero
                        valid_low = out_df['low'][out_df['low'] > 1e-9] # Avoid log(0) or log(negative)
                        if not valid_low.empty:
                            log_hl_range = np.log(out_df['high'].loc[valid_low.index] / valid_low)
                            out_df[final_col_name] = log_hl_range.rolling(window=window).std()
                        else:
                            out_df[final_col_name] = np.nan
                    else:
                        out_df[final_col_name] = np.nan
                
                # 'log_return', 'price_change_pct', etc. are calculated upfront if they are standard (e.g. from 'close')
                # If feature_conf specifies a 'log_return' for a different price_col (e.g. 'open'), it would be handled here.
                elif feature_name == 'log_return':
                    # This handles cases where log_return is requested for a column other than 'close', 'high', 'low'
                    # or if specific params are given. Standard ones are pre-calculated.
                    lr_price_col = params.get('price_col', 'close')
                    # Standard log_return_close is already f'{prefix}log_return_close'
                    # If this config is for that, it's already done.
                    # This block is more for log_return on other columns e.g. log_return_vwap if vwap was a column
                    if f"{prefix}log_return_{lr_price_col}" not in out_df.columns and lr_price_col in out_df:
                         out_df[f"{prefix}log_return_{lr_price_col}"] = np.log(out_df[lr_price_col] / out_df[lr_price_col].shift(1))
                         self.logger.info(f"Calculated non-standard {prefix}log_return_{lr_price_col}")
                    # else, it was either pre-calculated or source column doesn't exist.

                # No specific 'else' here to warn about unimplemented features, as many are pre-calculated
                # or handled by dedicated blocks below (time, lag, rolling).

            except Exception as e:
                self.logger.error(f"Error calculating feature {final_col_name} (base name {feature_name}) for prefix {prefix}: {e}", exc_info=True)
                if final_col_name not in out_df: # Ensure column exists even if error
                    out_df[final_col_name] = np.nan
                elif feature_name == 'macd': # Ensure all macd cols exist
                    out_df[f'{prefix}macd'] = out_df.get(f'{prefix}macd', pd.Series(np.nan, index=out_df.index))
                    out_df[f'{prefix}macd_signal'] = out_df.get(f'{prefix}macd_signal', pd.Series(np.nan, index=out_df.index))
                    out_df[f'{prefix}macd_hist'] = out_df.get(f'{prefix}macd_hist', pd.Series(np.nan, index=out_df.index))
                # Add similar for bbands, stoch if necessary
        
        # --- Time-based features (Cyclical) ---
        time_feature_configs = [cfg for cfg in feature_config_list if cfg['name'] == 'time_features']
        if time_feature_configs and 'close_timestamp' in out_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(out_df['close_timestamp']):
                out_df['close_timestamp'] = pd.to_datetime(out_df['close_timestamp'], errors='coerce')
            
            out_df.dropna(subset=['close_timestamp'], inplace=True) # Drop if conversion failed
            if not out_df.empty:
                dt_accessor = out_df['close_timestamp'].dt
                
                # Basic time features (can be made configurable from time_feature_configs[0]['params'] if needed)
                out_df[f'{prefix}time_hour_of_day'] = dt_accessor.hour
                out_df[f'{prefix}time_day_of_week'] = dt_accessor.dayofweek 
                out_df[f'{prefix}time_day_of_month'] = dt_accessor.day
                out_df[f'{prefix}time_week_of_year'] = dt_accessor.isocalendar().week.astype(int)
                out_df[f'{prefix}time_month_of_year'] = dt_accessor.month
                out_df[f'{prefix}time_is_weekend'] = (dt_accessor.dayofweek >= 5).astype(int)

                # Sin/Cos transformations for cyclical features (hour and day_of_week)
                # Hour
                hours_in_day = 24
                out_df[f'{prefix}time_hour_sin'] = np.sin(2 * np.pi * out_df[f'{prefix}time_hour_of_day'] / hours_in_day)
                out_df[f'{prefix}time_hour_cos'] = np.cos(2 * np.pi * out_df[f'{prefix}time_hour_of_day'] / hours_in_day)
                # Day of week
                days_in_week = 7
                out_df[f'{prefix}time_day_of_week_sin'] = np.sin(2 * np.pi * out_df[f'{prefix}time_day_of_week'] / days_in_week)
                out_df[f'{prefix}time_day_of_week_cos'] = np.cos(2 * np.pi * out_df[f'{prefix}time_day_of_week'] / days_in_week)
            else:
                self.logger.warning(f"DataFrame empty after NaT drop in close_timestamp for '{prefix}'. Skipping time features.")
        elif time_feature_configs:
             self.logger.warning(f"'close_timestamp' not found for '{prefix}'. Skipping time features.")

        # --- Lagged features ---
        lag_requests = [cfg for cfg in feature_config_list if cfg['name'] == 'lagged_feature']
        if lag_requests:
            for lag_cfg in lag_requests:
                params = lag_cfg.get('params', {})
                source_col_name_from_config = params.get('source_col_name')
                lags = params.get('lags')
                
                if not source_col_name_from_config or not isinstance(lags, list):
                    self.logger.warning(f"Skipping lagged_feature for {prefix}: 'source_col_name' or 'lags' (list) missing/invalid in params: {params}")
                    continue

                # Determine actual source column (e.g. 's_log_return_close' or 'volume')
                actual_source_col = ""
                if f"{prefix}{source_col_name_from_config}" in out_df.columns:
                    actual_source_col = f"{prefix}{source_col_name_from_config}"
                elif source_col_name_from_config in out_df.columns: # For non-prefixed base columns like 'volume'
                    actual_source_col = source_col_name_from_config
                else:
                    self.logger.warning(f"Lag source '{source_col_name_from_config}' (or prefixed) not found for {prefix}. Skipping.")
                    continue
                
                for lag in lags:
                    if isinstance(lag, int) and lag > 0:
                        lagged_col_name = f"{actual_source_col}_lag_{lag}"
                        out_df[lagged_col_name] = out_df[actual_source_col].shift(lag)
                    else:
                        self.logger.warning(f"Invalid lag value '{lag}' for {actual_source_col}. Must be positive int.")
        
        # --- Rolling Window Features ---
        rolling_stat_requests = [cfg for cfg in feature_config_list if cfg['name'] == 'rolling_stat']
        if rolling_stat_requests:
            for roll_cfg in rolling_stat_requests:
                params = roll_cfg.get('params', {})
                source_col_name_from_config = params.get('source_col_name')
                window = params.get('window')
                stat_fn_name = params.get('stat') # 'mean', 'std', 'min', 'max', 'skew', 'kurt'
                
                if not source_col_name_from_config or not isinstance(window, int) or window <=0 or not stat_fn_name:
                    self.logger.warning(f"Skipping rolling_stat for {prefix}: params invalid: {params}")
                    continue

                actual_source_col = ""
                if f"{prefix}{source_col_name_from_config}" in out_df.columns:
                    actual_source_col = f"{prefix}{source_col_name_from_config}"
                elif source_col_name_from_config in out_df.columns:
                    actual_source_col = source_col_name_from_config
                else:
                    self.logger.warning(f"Rolling stat source '{source_col_name_from_config}' not found for {prefix}. Skipping.")
                    continue
                
                rolling_col_name = f"{actual_source_col}_rolling_{stat_fn_name}_{window}"
                
                if len(out_df) >= window:
                    try:
                        # getattr on Series.rolling(window) object
                        rolling_series = out_df[actual_source_col].rolling(window=window)
                        out_df[rolling_col_name] = getattr(rolling_series, stat_fn_name)()
                    except AttributeError:
                        self.logger.error(f"Invalid stat function '{stat_fn_name}' for rolling window on {actual_source_col}.")
                        out_df[rolling_col_name] = np.nan
                    except Exception as e_roll:
                        self.logger.error(f"Error rolling {stat_fn_name} on {actual_source_col}: {e_roll}")
                        out_df[rolling_col_name] = np.nan
                else:
                    out_df[rolling_col_name] = np.nan # Not enough data for window
        
        self.logger.info(f"Finished calculating features for prefix '{prefix}'. Shape: {out_df.shape}")
        return out_df

    def process_historical_batch(self, historical_short_bars_df):
        """
        Processes a historical batch of short dollar bars to generate features
        for short, medium, and long resolutions.

        Args:
            historical_short_bars_df (pd.DataFrame): DataFrame of short dollar bars.
                                                     Must contain OHLCV, timestamps, quote_quantity, num_ticks.
        Returns:
            tuple: (df_short_with_features, df_medium_with_features, df_long_with_features)
        """
        self.logger.info(f"Starting historical batch processing on {len(historical_short_bars_df)} short bars.")
        if historical_short_bars_df.empty:
            self.logger.warning("Historical short bars DataFrame is empty.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Ensure 'close_timestamp' is datetime for indexing and time features later
        if 'close_timestamp' in historical_short_bars_df.columns and \
           not pd.api.types.is_datetime64_any_dtype(historical_short_bars_df['close_timestamp']):
            historical_short_bars_df['close_timestamp'] = pd.to_datetime(historical_short_bars_df['close_timestamp'])
        
        # Set index for potential resampling/alignment later, though direct calculation is primary
        # df_short = historical_short_bars_df.set_index('close_timestamp', drop=False)
        df_short = historical_short_bars_df.copy()


        # 1. Calculate features for Short Bars
        df_short_with_features = self._calculate_features_for_resolution(
            df_short, 
            's_', 
            self.feature_configs.get('short', [])
        )
        self.logger.info(f"Short bar features calculated. Shape: {df_short_with_features.shape}")

        # 2. Aggregate to Medium Bars
        df_medium_bars = self._aggregate_bars(
            df_short_with_features, # Pass short bars (can be without features if agg doesn't need them)
            'medium',
            self.medium_bar_agg_config
        )
        if df_medium_bars.empty:
            self.logger.warning("No medium bars were formed. Returning empty DataFrames for medium and long.")
            return df_short_with_features, pd.DataFrame(), pd.DataFrame()

        # 3. Calculate features for Medium Bars
        df_medium_with_features = self._calculate_features_for_resolution(
            df_medium_bars,
            'm_',
            self.feature_configs.get('medium', [])
        )
        self.logger.info(f"Medium bar features calculated. Shape: {df_medium_with_features.shape}")

        # 4. Aggregate to Long Bars
        df_long_bars = self._aggregate_bars(
            df_medium_with_features, # Pass medium bars
            'long',
            self.long_bar_agg_config
        )
        if df_long_bars.empty:
            self.logger.warning("No long bars were formed. Returning empty DataFrame for long.")
            return df_short_with_features, df_medium_with_features, pd.DataFrame()

        # 5. Calculate features for Long Bars
        df_long_with_features = self._calculate_features_for_resolution(
            df_long_bars,
            'l_',
            self.feature_configs.get('long', [])
        )
        self.logger.info(f"Long bar features calculated. Shape: {df_long_with_features.shape}")

        self.logger.info("Historical batch processing completed.")
        return df_short_with_features, df_medium_with_features, df_long_with_features
    
    def add_short_bar(self, short_bar_series):
        """
        Processes a new incoming short dollar bar for live/incremental feature generation.
        This method needs significant adaptation to work with the new batch-style feature calculation.
        For now, it's a placeholder and will likely need to maintain its own buffers and
        trigger mini-batch calculations or more complex state management.
        """
        self.logger.warning("Live 'add_short_bar' method is not fully implemented for incremental feature updates yet.")
        # Placeholder:
        # 1. Add to short_bars_buffer
        # 2. If enough new short bars, (re)calculate short features on the buffer.
        # 3. Attempt to aggregate to form a new medium bar.
        # 4. If medium bar formed, add to medium_bars_buffer, (re)calculate medium features.
        # 5. Attempt to aggregate to form a new long bar.
        # 6. If long bar formed, add to long_bars_buffer, (re)calculate long features.
        # 7. Align and return latest features.
        return pd.Series(dtype='float64') # Placeholder return

# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    logger.info("Running MinotaurFeatureEngine example...")

    # Define configurations
    cfg_short_agg = {} # Short bars are input, no specific "aggregation to make them" config here.
    
    # Medium bars: aggregate every 10 short bars
    cfg_medium_agg = {'agg_type': 'bar_count', 'bars_per_agg': 10, 'agg_col': 'quote_quantity'} # agg_col needed for dollar_value type
    
    # Long bars: aggregate every 16 medium bars
    cfg_long_agg = {'agg_type': 'bar_count', 'bars_per_agg': 16, 'agg_col': 'quote_quantity'}

    # Feature configurations (example)
    feature_cfg = {
        'short': [
            {'name': 'sma', 'period': 10, 'col': 'close'}, {'name': 'sma', 'period': 20, 'col': 'close'},
            {'name': 'rsi', 'period': 14, 'col': 'close'},
            {'name': 'macd'}, # Will use default periods
            {'name': 'volatility_close', 'period': 20},
            {'name': 'volatility_high_low', 'period': 20},
            {'name': 'time_features'},
            # Placeholder to indicate we want lags for default features (log_return_close, volume, close)
            {'name': 'lag_features'},
            # Placeholder for rolling features
            {'name': 'rolling_features'} 
        ],
        'medium': [
            {'name': 'sma', 'period': 10, 'col': 'close'}, {'name': 'sma', 'period': 50, 'col': 'close'},
            {'name': 'rsi', 'period': 14, 'col': 'close'},
            {'name': 'atr', 'period': 14, 'col': 'close'},
            {'name': 'time_features'},
            {'name': 'lag_features'},
            {'name': 'rolling_features'}
        ],
        'long': [
            {'name': 'sma', 'period': 10, 'col': 'close'}, {'name': 'sma', 'period': 200, 'col': 'close'},
            {'name': 'rsi', 'period': 14, 'col': 'close'},
            {'name': 'bbands', 'period': 20, 'col': 'close'},
            {'name': 'time_features'},
            {'name': 'lag_features'},
            {'name': 'rolling_features'}
        ]
    }

    engine = MinotaurFeatureEngine(
        short_bar_agg_config=cfg_short_agg,
        medium_bar_agg_config=cfg_medium_agg,
        long_bar_agg_config=cfg_long_agg,
        feature_configs=feature_cfg
    )

    # Create dummy historical short bar data
    n_short_bars = 500
    data = {
        'open_timestamp': pd.to_datetime(pd.date_range('2023-01-01', periods=n_short_bars, freq='1min')),
        'close_timestamp': pd.to_datetime(pd.date_range('2023-01-01', periods=n_short_bars, freq='1min') + pd.Timedelta(minutes=0.9)),
        'open': np.random.rand(n_short_bars) * 100 + 20000,
        'high': 0, # Will be derived
        'low': 0,  # Will be derived
        'close': np.random.rand(n_short_bars) * 100 + 20000,
        'volume': np.random.rand(n_short_bars) * 10 + 1,
        'quote_quantity': np.random.rand(n_short_bars) * 200000 + 100000, # Approx $2M bars
        'num_ticks': np.random.randint(50, 200, size=n_short_bars)
    }
    dummy_short_df = pd.DataFrame(data)
    dummy_short_df['high'] = dummy_short_df[['open', 'close']].max(axis=1) + np.random.rand(n_short_bars) * 10
    dummy_short_df['low'] = dummy_short_df[['open', 'close']].min(axis=1) - np.random.rand(n_short_bars) * 10
    # Ensure high >= close and open, low <= close and open
    dummy_short_df['high'] = np.maximum(dummy_short_df['high'], dummy_short_df['close'])
    dummy_short_df['high'] = np.maximum(dummy_short_df['high'], dummy_short_df['open'])
    dummy_short_df['low'] = np.minimum(dummy_short_df['low'], dummy_short_df['close'])
    dummy_short_df['low'] = np.minimum(dummy_short_df['low'], dummy_short_df['open'])


    logger.info(f"Created dummy short bar data with shape: {dummy_short_df.shape}")
    logger.info(f"Dummy short bar columns: {dummy_short_df.columns.tolist()}")
    logger.info(f"Dummy short bar head:\\n{dummy_short_df.head()}")


    # Process the historical batch
    df_s, df_m, df_l = engine.process_historical_batch(dummy_short_df)

    logger.info(f"--- Short Features DataFrame (Top 5) --- ({df_s.shape})")
    if not df_s.empty: logger.info(f"Columns: {df_s.columns.tolist()}\n{df_s.head()}")
    
    logger.info(f"--- Medium Features DataFrame (Top 5) --- ({df_m.shape})")
    if not df_m.empty: logger.info(f"Columns: {df_m.columns.tolist()}\n{df_m.head()}")

    logger.info(f"--- Long Features DataFrame (Top 5) --- ({df_l.shape})")
    if not df_l.empty: logger.info(f"Columns: {df_l.columns.tolist()}\n{df_l.head()}")

    # Example of live processing (conceptual, add_short_bar needs full implementation)
    # if not dummy_short_df.empty:
    #     logger.info("\\n--- Conceptual Live Processing Example ---")
    #     for _, row in dummy_short_df.iloc[:5].iterrows(): # Process first 5 bars
    #         logger.info(f"Adding live short bar: {row['close_timestamp']}")
    #         # combined_features = engine.add_short_bar(row)
    #         # logger.info(f"Combined features: {combined_features.to_dict() if not combined_features.empty else 'No features yet'}")
    # else:
    #     logger.info("Skipping live processing example as dummy_short_df is empty.")
    logger.info("MinotaurFeatureEngine example finished.") 