import pandas as pd
from pathlib import Path
import logging
import re
import numpy as np
import talib
from itertools import product
from scipy.signal import find_peaks
import argparse # Added for command-line arguments
from volprofile import getVPWithOHLC # Added for Volume Profile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # minotaur/scripts/time_based_processing -> trading-agent-v3
INPUT_TIME_BAR_BASE_DIR = PROJECT_ROOT / 'data' / 'time_bars'
OUTPUT_FEATURE_BASE_DIR = PROJECT_ROOT / 'data' / 'time_bars_features'
SYMBOL = 'BTCUSDT'

TIME_RESOLUTIONS_TO_PROCESS = ['1min', '15min', '4hour'] # Matches dir names from generate_time_bars.py

# --- Feature Configuration ---
FEATURE_CONFIG = {
    'price_features': {
        'log_return_close': True,
        'price_change_pct': True, # (close - open) / open
        'high_low_pct': True,     # (high - low) / low
        'close_vs_high_pct': True,# (close - high) / high
        'close_vs_low_pct': True, # (close - low) / low
        'wick_vs_range': True,
        'body_vs_range': True,
    },
    'talib_indicators': {
        'SMA': {'periods': [10, 20, 50, 100, 200]},
        'EMA': {'periods': [10, 12, 20, 26, 50, 100, 200]},
        'RSI': {'periods': [7, 14, 21], 'ob_level': 70, 'os_level': 30},
        'MACD': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
        'ADX': {'periods': [14, 20]}, # Also calculates PLUS_DI, MINUS_DI
        'ATR': {'periods': [14, 20]},
        'BBANDS': {'periods': [20], 'nbdevup': 2, 'nbdevdn': 2}, # Also calculates width, pct_b
        'CCI': {'periods': [14, 20]},
        'MFI': {'periods': [14, 20]},
        'ROC': {'periods': [1, 5, 10, 20]},
        'STOCH': {'fastk_periods': [14], 'slowk_periods': [3], 'slowd_periods': [3]}, # slowk_matype=0, slowd_matype=0 (SMA)
        'ULTOSC': {'timeperiod1': 7, 'timeperiod2': 14, 'timeperiod3': 28},
        'WILLR': {'periods': [14, 28]},
        'OBV': True,
        'HT_TRENDLINE': True, # Hilbert Transform - Instantaneous Trendline
        'HT_SINE': True,      # Hilbert Transform - SineWave
    },
    'volume_features': {
        'volume_sma': {'periods': [20, 50]},
        'volume_log': True, # Log transform of raw volume
        # Add other volume-based features like VROC, VOSC later if needed
    },
    'cyclical_time_features': {
        'hour_sin_cos': True,
        'day_of_week_sin_cos': True,
        'month_sin_cos': True,
    },
    'volatility_features': {
        # Std dev of log returns
        'log_return_std_dev': {'windows': [10, 20, 30]},
        # Normalized ATR (NATR)
        'NATR': {'periods': [14, 20]}, # Uses same periods as ATR for consistency
    },
    'candlestick_patterns': {
        'enabled': True,
        'patterns_to_calculate': 'all', # or a list of specific CDL names
    },
    'lagged_features': {
        'columns_to_lag': {
            'log_return_close': [1, 2, 3, 5, 8],
            'Volume': [1, 2, 3, 5, 8], # Lag raw volume
            'ofi': [1, 2, 3], # Lag Order Flow Imbalance
            # Add other columns from intra-bar features if desired
        }
    },
    'rolling_stats': { # Skewness and Kurtosis
        'columns_for_rolling_stats': {
            'log_return_close': {'windows': [20, 50], 'stats': ['skew', 'kurt']},
            # 'price_change_pct': {'windows': [20, 50], 'stats': ['skew', 'kurt']},
        }
    },
    'divergence_features': {
        'enabled': True,
        'indicator_name': 'rsi', # Can be 'rsi', 'macd_hist', etc. later
        'indicator_period': 14, # Must match a calculated RSI period
        'rsi_ob_level_div': 60, # RSI overbought for divergence context (feature_engine_v2 uses 60)
        'rsi_os_level_div': 40, # RSI oversold for divergence context (feature_engine_v2 uses 40)
        'price_lookback': 30,    # Window to find preceding price peak/trough
        'peak_prominence_price': 0.005, # Prominence for price peaks (as fraction of price)
        'peak_prominence_indicator': 1.0, # Prominence for indicator peaks (absolute value)
        'peak_distance': 5, # Min distance between peaks
    },
    'regime_features': {
        'enabled': True,
        'adx_period': 14,  # Must match a calculated ADX period
        'adx_threshold': 25, 
        'ema_fast_period': 12, # Must match a calculated EMA period
        'ema_slow_period': 26, # Must match a calculated EMA period
    },
    'moving_averages': {
        'sma_periods': [10, 20, 50, 100, 200],
        'ema_periods': [10, 12, 20, 26, 50, 100, 200], # Added 12 and 26 for regime features
    },
    'rsi': {
        'periods': [7, 14, 21],
    },
    'market_structure': {
        'enabled': False, # Placeholder, not implemented yet
        'fractal_period': 5 
    },
    'volume_profile_features': {
        'enabled': True,
        'window_sizes': [50, 100], # Lookback periods for VP calculation
        'n_bins': [50, 100]       # Number of bins for profile histogram
    },
    # More advanced features to adapt from feature_engine_v2 later:
    # 'volume_profile_features': { ... }
}

# Get all TA-Lib candlestick pattern function names
ALL_TALIB_PATTERNS = [
    func for func in dir(talib) if func.startswith('CDL')
]

def get_sorted_input_files_for_resolution(input_dir_path, symbol_str, resolution_name_str):
    logger.info(f"Searching for input Time Bar Parquet files in: {input_dir_path}")
    file_pattern = re.compile(rf"^{symbol_str}_(\d{{4}})_(\d{{2}})_time_bars_{resolution_name_str}\.parquet$")
    
    matched_files = []
    for f_path in input_dir_path.iterdir():
        if f_path.is_file():
            match = file_pattern.match(f_path.name)
            if match:
                year = int(match.group(1))
                month = int(match.group(2))
                matched_files.append({'path': f_path, 'year': year, 'month': month, 'name': f_path.name})
    
    matched_files.sort(key=lambda x: (x['year'], x['month']))
    
    if not matched_files:
        logger.warning(f"No input files found matching pattern in {input_dir_path} for resolution {resolution_name_str}")
        return []
        
    logger.info(f"Found and sorted {len(matched_files)} input files for {resolution_name_str}.")
    return [mf['path'] for mf in matched_files]

def calculate_vp_metrics(window_df, n_bins=20):
    # Expects lowercase columns: open, high, low, close, volume
    if window_df.empty or len(window_df) < 2: # Need at least 2 rows for OHLC, and some data
        return np.nan, np.nan, np.nan

    try:
        vp_df = getVPWithOHLC(window_df, nBins=n_bins)
        
        if vp_df.empty or 'aggregateVolume' not in vp_df.columns or vp_df['aggregateVolume'].sum() == 0:
            return np.nan, np.nan, np.nan

        # POC
        poc_row = vp_df.loc[vp_df['aggregateVolume'].idxmax()]
        poc_price = (poc_row['minPrice'] + poc_row['maxPrice']) / 2

        # VAH/VAL
        total_volume_in_profile = vp_df['aggregateVolume'].sum()
        target_va_volume = total_volume_in_profile * 0.70
        
        vp_df_sorted_by_price = vp_df.sort_values(by='minPrice').reset_index(drop=True)
        
        poc_bin_candidates = vp_df_sorted_by_price[
            (vp_df_sorted_by_price['minPrice'] == poc_row['minPrice']) &
            (vp_df_sorted_by_price['maxPrice'] == poc_row['maxPrice'])
        ]
        if poc_bin_candidates.empty:
            return np.nan, np.nan, np.nan 
        poc_bin_index_in_sorted = poc_bin_candidates.index[0]
        
        current_va_volume = vp_df_sorted_by_price.loc[poc_bin_index_in_sorted, 'aggregateVolume']
        val_idx, vah_idx = poc_bin_index_in_sorted, poc_bin_index_in_sorted
        
        while current_va_volume < target_va_volume and not (val_idx == 0 and vah_idx == len(vp_df_sorted_by_price) - 1):
            vol_above = 0
            vol_below = 0
            
            if vah_idx + 1 < len(vp_df_sorted_by_price):
                vol_above = vp_df_sorted_by_price.loc[vah_idx + 1, 'aggregateVolume']
            
            if val_idx - 1 >= 0:
                vol_below = vp_df_sorted_by_price.loc[val_idx - 1, 'aggregateVolume']
            
            if vol_above == 0 and vol_below == 0: break
                
            if vol_above > vol_below:
                vah_idx += 1
                current_va_volume += vol_above
            elif vol_below > vol_above:
                val_idx -= 1
                current_va_volume += vol_below
            else: 
                can_expand_up = vah_idx + 1 < len(vp_df_sorted_by_price)
                can_expand_down = val_idx - 1 >= 0
                if can_expand_up and can_expand_down and vol_above > 0: 
                    vah_idx +=1
                    val_idx -=1
                    current_va_volume += vol_above + vol_below 
                elif can_expand_up and vol_above > 0:
                    vah_idx += 1
                    current_va_volume += vol_above
                elif can_expand_down and vol_below > 0:
                    val_idx -=1
                    current_va_volume += vol_below
                else:
                    break 
        
        val_price = vp_df_sorted_by_price.loc[val_idx, 'minPrice']
        vah_price = vp_df_sorted_by_price.loc[vah_idx, 'maxPrice']
        
        return poc_price, val_price, vah_price

    except Exception as e:
        # logger.warning(f"Error in calculate_vp_metrics for window ({len(window_df)} rows, n_bins={n_bins}): {e}")
        # if not window_df.empty:
        #     logger.warning(f"Window head:\n{window_df.head(2)}")
        #     logger.warning(f"Window tail:\n{window_df.tail(2)}")
        return np.nan, np.nan, np.nan

def calculate_features_for_df(df, resolution_name, feature_config):
    logger.info(f"Calculating features for {resolution_name} DataFrame with {len(df)} bars using new config...")
    
    # Standardize column names and set index
    column_rename_map = {
        'open_timestamp': 'datetime',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        # Keep other intra-bar features as they are (e.g., vwap, ofi, price_skew etc.)
    }
    out_df = df.rename(columns=column_rename_map)

    if 'datetime' not in out_df.columns:
        logger.error(f"Missing 'datetime' (renamed from 'open_timestamp') column for {resolution_name}. Cannot proceed.")
        return pd.DataFrame()
    
    out_df['datetime'] = pd.to_datetime(out_df['datetime'])
    out_df = out_df.set_index('datetime').sort_index()

    # Verify essential OHLCV columns exist after renaming
    required_ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_ohlcv_cols:
        if col not in out_df.columns:
            logger.error(f"Missing required column '{col}' (after rename) for {resolution_name}. Cannot calculate features.")
            return pd.DataFrame()

    # Ensure correct dtypes for calculations
    for col in required_ohlcv_cols:
        out_df[col] = pd.to_numeric(out_df[col], errors='coerce')
    
    # Drop rows if essential OHLCV data is NaN after conversion (e.g. if a bar had no trades)
    out_df.dropna(subset=required_ohlcv_cols, inplace=True)

    if out_df.empty:
        logger.warning(f"DataFrame for {resolution_name} became empty after dtype conversion and NaN drop on OHLCV. Skipping feature calculation.")
        return pd.DataFrame()

    # --- Log Transform Volume (early in the process) ---
    if 'Volume' in out_df.columns and feature_config.get('volume_features', {}).get('volume_log'):
        out_df['Volume_log'] = np.log1p(out_df['Volume'])
        logger.info(f"Calculated Volume_log for {resolution_name} bars.")

    # --- 0. Carry over existing intra-bar features ---
    # These are already calculated, just ensure they are part of out_df
    # Example: vwap, ofi, price_skew, price_kurtosis, volume_skew, volume_kurtosis, avg_trade_size, num_ticks
    # No specific action needed here if they are already columns, just be aware of them.

    # --- 1. Basic Price/Return Features ---
    if feature_config.get('price_features', {}).get('log_return_close'):
        out_df['log_return_close'] = np.log(out_df['Close'] / out_df['Close'].shift(1))
    if feature_config.get('price_features', {}).get('price_change_pct'):
        out_df['price_change_pct'] = (out_df['Close'] - out_df['Open']) / out_df['Open']
    if feature_config.get('price_features', {}).get('high_low_pct'):
        out_df['high_low_pct'] = (out_df['High'] - out_df['Low']) / out_df['Low']
    if feature_config.get('price_features', {}).get('close_vs_high_pct'):
        out_df['close_vs_high_pct'] = (out_df['Close'] - out_df['High']) / out_df['High']
    if feature_config.get('price_features', {}).get('close_vs_low_pct'):
        out_df['close_vs_low_pct'] = (out_df['Close'] - out_df['Low']) / out_df['Low']
    
    if feature_config.get('price_features', {}).get('wick_vs_range') or feature_config.get('price_features', {}).get('body_vs_range'):
        range_val = (out_df['High'] - out_df['Low'])
        if feature_config.get('price_features', {}).get('wick_vs_range'):
            out_df['wick_vs_range'] = np.where(range_val > 0, ((out_df['High'] - out_df['Low']) - abs(out_df['Open'] - out_df['Close'])) / range_val, 0)
        if feature_config.get('price_features', {}).get('body_vs_range'):
            out_df['body_vs_range'] = np.where(range_val > 0, abs(out_df['Open'] - out_df['Close']) / range_val, 0)
    
    out_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- 2. TA-Lib Indicators ---
    # Prepare numpy arrays for TA-Lib
    open_arr = out_df['Open'].values
    high_arr = out_df['High'].values
    low_arr = out_df['Low'].values
    close_arr = out_df['Close'].values
    volume_arr = out_df['Volume'].values # Standardized to 'Volume'

    talib_config = feature_config.get('talib_indicators', {})

    if 'SMA' in talib_config:
        for period in talib_config['SMA'].get('periods', []):
            if len(close_arr) >= period: out_df[f'sma_{period}'] = talib.SMA(close_arr, timeperiod=period)
    if 'EMA' in talib_config:
        for period in talib_config['EMA'].get('periods', []):
            if len(close_arr) >= period: out_df[f'ema_{period}'] = talib.EMA(close_arr, timeperiod=period)
    
    if 'RSI' in talib_config:
        rsi_params = talib_config['RSI']
        for period in rsi_params.get('periods', []):
            if len(close_arr) >= period:
                rsi_values = talib.RSI(close_arr, timeperiod=period)
                out_df[f'rsi_{period}'] = rsi_values
                if rsi_params.get('ob_level') is not None:
                    out_df[f'rsi_{period}_ob'] = (rsi_values > rsi_params['ob_level']).astype(int)
                if rsi_params.get('os_level') is not None:
                    out_df[f'rsi_{period}_os'] = (rsi_values < rsi_params['os_level']).astype(int)
                if rsi_params.get('ob_level') is not None and rsi_params.get('os_level') is not None:
                     out_df[f'rsi_{period}_neutral'] = ((rsi_values >= rsi_params['os_level']) & (rsi_values <= rsi_params['ob_level'])).astype(int)

    if 'MACD' in talib_config:
        macd_params = talib_config['MACD']
        if len(close_arr) >= macd_params.get('slowperiod', 26): # Check against slow period
            macd, macdsignal, macdhist = talib.MACD(close_arr, 
                                                    fastperiod=macd_params.get('fastperiod', 12), 
                                                    slowperiod=macd_params.get('slowperiod', 26), 
                                                    signalperiod=macd_params.get('signalperiod', 9))
            out_df['macd'] = macd
            out_df['macd_signal'] = macdsignal
            out_df['macd_hist'] = macdhist

    if 'ADX' in talib_config:
        for period in talib_config['ADX'].get('periods', []):
            if len(high_arr) >= period * 2: # ADX typically needs more data
                out_df[f'adx_{period}'] = talib.ADX(high_arr, low_arr, close_arr, timeperiod=period)
                out_df[f'plus_di_{period}'] = talib.PLUS_DI(high_arr, low_arr, close_arr, timeperiod=period)
                out_df[f'minus_di_{period}'] = talib.MINUS_DI(high_arr, low_arr, close_arr, timeperiod=period)

    if 'ATR' in talib_config:
        for period in talib_config['ATR'].get('periods', []):
            if len(high_arr) >= period: out_df[f'atr_{period}'] = talib.ATR(high_arr, low_arr, close_arr, timeperiod=period)

    if 'BBANDS' in talib_config:
        bb_params = talib_config['BBANDS']
        for period in bb_params.get('periods', []):
            if len(close_arr) >= period:
                upper, middle, lower = talib.BBANDS(close_arr, 
                                                    timeperiod=period, 
                                                    nbdevup=bb_params.get('nbdevup', 2), 
                                                    nbdevdn=bb_params.get('nbdevdn', 2))
                out_df[f'bb_upper_{period}'] = upper
                out_df[f'bb_middle_{period}'] = middle
                out_df[f'bb_lower_{period}'] = lower
                if middle_val_not_zero := (middle != 0).all(): # Check if middle is not zero for all values to avoid division by zero warnings
                    out_df[f'bb_width_{period}'] = (upper - lower) / middle
                    out_df[f'bb_pct_b_{period}'] = np.where((upper - lower) != 0, (close_arr - lower) / (upper - lower), np.nan)


    if 'CCI' in talib_config:
        for period in talib_config['CCI'].get('periods', []):
            if len(high_arr) >= period: out_df[f'cci_{period}'] = talib.CCI(high_arr, low_arr, close_arr, timeperiod=period)
    if 'MFI' in talib_config:
        for period in talib_config['MFI'].get('periods', []):
            # MFI needs open_arr as well, but TA-Lib's MFI takes H,L,C,V
            if len(high_arr) >= period and len(volume_arr) >= period:
                out_df[f'mfi_{period}'] = talib.MFI(high_arr, low_arr, close_arr, volume_arr, timeperiod=period)
    if 'ROC' in talib_config:
        for period in talib_config['ROC'].get('periods', []):
            if len(close_arr) > period: out_df[f'roc_{period}'] = talib.ROC(close_arr, timeperiod=period)
    
    if 'STOCH' in talib_config:
        stoch_params = talib_config['STOCH']
        # Using product to combine all period settings if multiple are provided, though typically one set.
        for k_p in stoch_params.get('fastk_periods', [14]):
            for sk_p in stoch_params.get('slowk_periods', [3]):
                for sd_p in stoch_params.get('slowd_periods', [3]):
                    # Ensure sufficient data for the combined effect of fastk and smoothing periods
                    if len(high_arr) >= k_p + sk_p + sd_p - 2 : # Approximation
                        slowk, slowd = talib.STOCH(high_arr, low_arr, close_arr, 
                                                   fastk_period=k_p, 
                                                   slowk_period=sk_p, slowk_matype=0, # 0 for SMA
                                                   slowd_period=sd_p, slowd_matype=0) # 0 for SMA
                        out_df[f'stoch_k_{k_p}_{sk_p}_{sd_p}'] = slowk
                        out_df[f'stoch_d_{k_p}_{sk_p}_{sd_p}'] = slowd

    if 'ULTOSC' in talib_config:
        ultosc_params = talib_config['ULTOSC']
        if len(high_arr) >= ultosc_params.get('timeperiod3', 28): # Longest period for ULTOSC
            out_df['ultosc'] = talib.ULTOSC(high_arr, low_arr, close_arr, 
                                            timeperiod1=ultosc_params.get('timeperiod1', 7), 
                                            timeperiod2=ultosc_params.get('timeperiod2', 14), 
                                            timeperiod3=ultosc_params.get('timeperiod3', 28))
    if 'WILLR' in talib_config:
        for period in talib_config['WILLR'].get('periods', []):
            if len(high_arr) >= period: out_df[f'willr_{period}'] = talib.WILLR(high_arr, low_arr, close_arr, timeperiod=period)
    
    if talib_config.get('OBV'):
        if len(close_arr) > 0 and len(volume_arr) > 0 : out_df['obv'] = talib.OBV(close_arr, volume_arr)

    if talib_config.get('HT_TRENDLINE'):
        if len(close_arr) > 6: # Hilbert functions need some data
            out_df['ht_trendline'] = talib.HT_TRENDLINE(close_arr)
    if talib_config.get('HT_SINE'):
        if len(close_arr) > 63: # Needs more data
             sine, leadsine = talib.HT_SINE(close_arr)
             out_df['ht_sine'] = sine
             out_df['ht_leadsine'] = leadsine
             
    out_df.replace([np.inf, -np.inf], np.nan, inplace=True) # Final catch-all for infs

    # --- 3. Volume Features ---
    vol_conf = feature_config.get('volume_features', {})
    if 'volume_sma' in vol_conf:
        for period in vol_conf['volume_sma'].get('periods', []):
            if len(volume_arr) >= period: out_df[f'volume_sma_{period}'] = talib.SMA(volume_arr, timeperiod=period)
    if vol_conf.get('volume_log'):
        # Apply to the original 'Volume' column and also other existing volume features if needed
        # For now, just the main 'Volume'
        if 'Volume' in out_df.columns:
             out_df['Volume_log'] = np.log1p(out_df['Volume'])
        if 'quote_volume' in out_df.columns: # also log transform quote_volume if it exists
             out_df['quote_volume_log'] = np.log1p(out_df['quote_volume'])


    # --- 4. Cyclical Time Features ---
    time_conf = feature_config.get('cyclical_time_features', {})
    # dt_accessor is based on the index, which is already datetime
    dt_accessor = out_df.index.to_series().dt
    if time_conf.get('hour_sin_cos'):
        out_df['hour_sin'] = np.sin(2 * np.pi * dt_accessor.hour / 24.0)
        out_df['hour_cos'] = np.cos(2 * np.pi * dt_accessor.hour / 24.0)
    if time_conf.get('day_of_week_sin_cos'):
        out_df['day_of_week_sin'] = np.sin(2 * np.pi * dt_accessor.dayofweek / 7.0)
        out_df['day_of_week_cos'] = np.cos(2 * np.pi * dt_accessor.dayofweek / 7.0)
    if time_conf.get('month_sin_cos'):
        out_df['month_sin'] = np.sin(2 * np.pi * dt_accessor.month / 12.0)
        out_df['month_cos'] = np.cos(2 * np.pi * dt_accessor.month / 12.0)

    # --- 5. Volatility Features ---
    volatility_conf = feature_config.get('volatility_features', {})
    if 'log_return_std_dev' in volatility_conf and 'log_return_close' in out_df.columns:
        for window in volatility_conf['log_return_std_dev'].get('windows', []):
            if len(out_df['log_return_close']) >= window:
                 out_df[f'volatility_logret_{window}'] = out_df['log_return_close'].rolling(window=window).std()
    
    if 'NATR' in volatility_conf: # Normalized ATR
        natr_params = volatility_conf['NATR']
        for period in natr_params.get('periods', []):
             # Check if ATR for this period was calculated and exists (dependency)
             atr_col_name = f'atr_{period}'
             if atr_col_name in out_df.columns and len(high_arr) >= period :
                 out_df[f'natr_{period}'] = talib.NATR(high_arr, low_arr, close_arr, timeperiod=period)


    # --- 6. Candlestick Patterns ---
    candle_conf = feature_config.get('candlestick_patterns', {})
    if candle_conf.get('enabled'):
        patterns_to_calc = candle_conf.get('patterns_to_calculate', 'all')
        target_patterns = ALL_TALIB_PATTERNS if patterns_to_calc == 'all' else patterns_to_calc
        
        for pattern_name in target_patterns:
            if pattern_name in ALL_TALIB_PATTERNS: # Ensure it's a valid TA-Lib pattern
                try:
                    pattern_func = getattr(talib, pattern_name)
                    if len(open_arr) > 1: # Basic check
                        out_df[pattern_name.lower()] = pattern_func(open_arr, high_arr, low_arr, close_arr)
                except Exception as e:
                    logger.debug(f"Could not calculate pattern {pattern_name} for {resolution_name}: {e}")

    # --- 7. Lagged Features ---
    lag_conf = feature_config.get('lagged_features', {})
    cols_to_lag_map = lag_conf.get('columns_to_lag', {})
    for col_name, periods in cols_to_lag_map.items():
        if col_name in out_df.columns:
            for p in periods:
                out_df[f'{col_name}_lag_{p}'] = out_df[col_name].shift(p)
        # else: # Column not found, maybe log a warning
            # logger.warning(f"Lagged feature: Column '{col_name}' not found in DataFrame for {resolution_name}.")


    # --- 8. Rolling Statistics (Skew, Kurtosis) ---
    rolling_conf = feature_config.get('rolling_stats', {})
    cols_for_rolling_map = rolling_conf.get('columns_for_rolling_stats', {})
    for col_name, params in cols_for_rolling_map.items():
        if col_name in out_df.columns:
            for window in params.get('windows', []):
                if len(out_df[col_name]) >= window:
                    if 'skew' in params.get('stats', []):
                        out_df[f'{col_name}_roll_skew_{window}'] = out_df[col_name].rolling(window=window).skew()
                    if 'kurt' in params.get('stats', []):
                        out_df[f'{col_name}_roll_kurt_{window}'] = out_df[col_name].rolling(window=window).kurt()
        # else: # Column not found
            # logger.warning(f"Rolling stats: Column '{col_name}' not found for {resolution_name}.")


    # --- 9. Divergence Features ---
    div_conf = feature_config.get('divergence_features', {})
    if div_conf.get('enabled'):
        indicator_col = f"{div_conf.get('indicator_name', 'rsi')}_{div_conf.get('indicator_period', 14)}"
        if indicator_col in out_df.columns:
            logger.info(f"Calculating divergence features based on {indicator_col}...")
            # Ensure Close, High, Low, and the indicator series are numpy arrays for efficiency
            close_prices = out_df['Close'].values
            high_prices = out_df['High'].values
            low_prices = out_df['Low'].values
            indicator_series = out_df[indicator_col].values

            bull_div_col_name = f"{indicator_col}_bull_div"
            bear_div_col_name = f"{indicator_col}_bear_div"

            # Initialize divergence columns with 0
            out_df[bull_div_col_name] = 0
            out_df[bear_div_col_name] = 0

            # Parameters from config
            price_lookback = div_conf.get('price_lookback', 30)
            peak_distance = div_conf.get('peak_distance', 5)
            # Prominence for price should be relative, for indicator can be absolute
            # For price, calculate prominence based on a fraction of the mean price in the window, or use a fixed small value
            # For indicator (like RSI), prominence can be an absolute value (e.g., 1-5 RSI points)
            prominence_price_frac = div_conf.get('peak_prominence_price', 0.005) # e.g. 0.5% of local mean price
            prominence_indicator_abs = div_conf.get('peak_prominence_indicator', 2.0) # e.g. 2 RSI points

            # Find peaks (for highs/bearish div) and troughs (for lows/bullish div)
            # For price highs (bearish divergence)
            # Prominence for price peaks: use a fraction of the current price or a rolling mean
            # For simplicity, using a fixed fraction of the global mean price for now - can be refined to rolling
            mean_price_for_prom = np.mean(high_prices[~np.isnan(high_prices)])
            price_high_peaks, _ = find_peaks(high_prices, distance=peak_distance, prominence=mean_price_for_prom * prominence_price_frac)
            
            # For price lows (bullish divergence) - find peaks in -low_prices
            mean_low_price_for_prom = np.mean(low_prices[~np.isnan(low_prices)]) # For prominence calc
            price_low_troughs, _ = find_peaks(-low_prices, distance=peak_distance, prominence=abs(mean_low_price_for_prom * prominence_price_frac)) # abs for prominence

            # For indicator peaks (bearish divergence)
            indicator_peaks, _ = find_peaks(indicator_series, distance=peak_distance, prominence=prominence_indicator_abs)
            
            # For indicator troughs (bullish divergence)
            indicator_troughs, _ = find_peaks(-indicator_series, distance=peak_distance, prominence=prominence_indicator_abs)

            # --- Bearish Divergence Check (Higher Highs in Price, Lower Highs in Indicator) ---
            # Iterate through price peaks. For each price peak, look for a preceding price peak.
            # Then find corresponding indicator peaks around these price peaks.
            for i in range(1, len(price_high_peaks)):
                current_price_peak_idx = price_high_peaks[i]
                # Search for a preceding price peak within the lookback window
                for j in range(i - 1, -1, -1):
                    prev_price_peak_idx = price_high_peaks[j]
                    if current_price_peak_idx - prev_price_peak_idx > price_lookback:
                        break # Preceding peak is too far back

                    # Condition 1: Price makes a higher high
                    if high_prices[current_price_peak_idx] > high_prices[prev_price_peak_idx]:
                        # Find indicator peaks near these price peaks
                        # This is a simplified way: find closest indicator peak within a small window
                        # A more robust way would be to ensure indicator peak falls between two price troughs etc.
                        current_indicator_peak_idx = indicator_peaks[(np.abs(indicator_peaks - current_price_peak_idx)).argmin()]
                        prev_indicator_peak_idx = indicator_peaks[(np.abs(indicator_peaks - prev_price_peak_idx)).argmin()]

                        # Ensure these are distinct and somewhat aligned with price peaks
                        if current_indicator_peak_idx != prev_indicator_peak_idx and \
                           abs(current_indicator_peak_idx - current_price_peak_idx) < price_lookback / 2 and \
                           abs(prev_indicator_peak_idx - prev_price_peak_idx) < price_lookback / 2:
                           
                            # Condition 2: Indicator makes a lower high
                            if indicator_series[current_indicator_peak_idx] < indicator_series[prev_indicator_peak_idx]:
                                # Bearish divergence found at current_price_peak_idx
                                # Mark the bar of the current (second) peak
                                out_df.loc[out_df.index[current_price_peak_idx], bear_div_col_name] = 1
                                # Potentially break inner loops if one divergence is enough for this peak
                                break 
           
            # --- Bullish Divergence Check (Lower Lows in Price, Higher Lows in Indicator) ---
            # Iterate through price troughs
            for i in range(1, len(price_low_troughs)):
                current_price_trough_idx = price_low_troughs[i]
                for j in range(i - 1, -1, -1):
                    prev_price_trough_idx = price_low_troughs[j]
                    if current_price_trough_idx - prev_price_trough_idx > price_lookback:
                        break

                    # Condition 1: Price makes a lower low
                    if low_prices[current_price_trough_idx] < low_prices[prev_price_trough_idx]:
                        current_indicator_trough_idx = indicator_troughs[(np.abs(indicator_troughs - current_price_trough_idx)).argmin()]
                        prev_indicator_trough_idx = indicator_troughs[(np.abs(indicator_troughs - prev_price_trough_idx)).argmin()]

                        if current_indicator_trough_idx != prev_indicator_trough_idx and \
                           abs(current_indicator_trough_idx - current_price_trough_idx) < price_lookback / 2 and \
                           abs(prev_indicator_trough_idx - prev_price_trough_idx) < price_lookback / 2:                           
                            # Condition 2: Indicator makes a higher low
                            if indicator_series[current_indicator_trough_idx] > indicator_series[prev_indicator_trough_idx]:
                                # Bullish divergence found at current_price_trough_idx
                                out_df.loc[out_df.index[current_price_trough_idx], bull_div_col_name] = 1
                                break
            logger.info(f"Calculated divergence features. Bullish: {out_df[bull_div_col_name].sum()}, Bearish: {out_df[bear_div_col_name].sum()}")       
        else:
            logger.warning(f"Cannot calculate divergence: indicator column '{indicator_col}' not found.")

    # --- 10. Regime Features ---
    regime_conf = feature_config.get('regime_features', {})
    if regime_conf.get('enabled'):
        # Ensure ADX and EMAs are calculated
        adx_period = regime_conf.get('adx_period', 14)
        adx_col = f'adx_{adx_period}'
        ema_fast_col = f'ema_{regime_conf.get("ema_fast_period", 12)}'
        ema_slow_col = f'ema_{regime_conf.get("ema_slow_period", 26)}'

        if adx_col in out_df.columns and ema_fast_col in out_df.columns and ema_slow_col in out_df.columns:
            adx_threshold = regime_conf.get('adx_threshold', 25)
            
            # Trend direction based on EMAs
            uptrend_ema = out_df[ema_fast_col] > out_df[ema_slow_col]
            downtrend_ema = out_df[ema_fast_col] < out_df[ema_slow_col]
            
            # ADX indicates trend strength
            trending_adx = out_df[adx_col] > adx_threshold
            ranging_adx = out_df[adx_col] <= adx_threshold
            
            out_df['regime_uptrend_strong'] = (uptrend_ema & trending_adx).astype(int)
            out_df['regime_downtrend_strong'] = (downtrend_ema & trending_adx).astype(int)
            out_df['regime_uptrend_weak'] = (uptrend_ema & ranging_adx).astype(int)
            out_df['regime_downtrend_weak'] = (downtrend_ema & ranging_adx).astype(int)
            out_df['regime_ranging'] = ranging_adx.astype(int) # Simplified: ranging if ADX is low, regardless of EMAs for this flag
            logger.info(f"Calculated regime features for {resolution_name} bars.")
        else:
            logger.warning(f"Skipping regime features for {resolution_name}: Required ADX/EMA columns not found.")
            
    # --- Volume Profile Features ---
    vp_config = feature_config.get('volume_profile_features', {})
    if vp_config.get('enabled'):
        logger.info(f"Calculating Volume Profile features for {resolution_name}...")
        # Ensure df_for_vp has lowercase column names expected by calculate_vp_metrics
        # and uses the original OHLCV data before other features might modify/drop them
        df_for_vp = out_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df_for_vp.columns = [col.lower() for col in df_for_vp.columns]

        for window in vp_config.get('window_sizes', [50]): 
            for n_bins_vp in vp_config.get('n_bins', [100]): 
                rolling_poc = []
                rolling_val = []
                rolling_vah = []
                
                # Pad with NaNs for initial window period
                for _ in range(window - 1):
                    rolling_poc.append(np.nan)
                    rolling_val.append(np.nan)
                    rolling_vah.append(np.nan)
                
                # Calculate for full windows
                for i in range(window - 1, len(df_for_vp)):
                    current_window_df = df_for_vp.iloc[i - window + 1 : i + 1]
                    poc_w, val_w, vah_w = calculate_vp_metrics(current_window_df, n_bins=n_bins_vp)
                    rolling_poc.append(poc_w)
                    rolling_val.append(val_w)
                    rolling_vah.append(vah_w)
                
                # Align the length of rolling results with the out_df index
                # This is crucial if df_for_vp was shorter due to earlier NaNs in OHLCV
                # However, df_for_vp is created from out_df AFTER basic OHLCV NaN handling,
                # so their lengths should typically correspond from the start of valid OHLCV data.
                # If out_df has leading NaNs that were dropped, iloc based indexing might be off.
                # A safer way is to assign to a Series with out_df.index
                
                if len(rolling_poc) == len(out_df):
                    out_df[f'vp_poc_{window}w_{n_bins_vp}b'] = rolling_poc
                    out_df[f'vp_val_{window}w_{n_bins_vp}b'] = rolling_val
                    out_df[f'vp_vah_{window}w_{n_bins_vp}b'] = rolling_vah
                else:
                    # This case might occur if out_df had leading NaNs removed before this stage,
                    # making its index shorter than a simple range(len(df_for_vp)).
                    # We need to align based on the actual index of out_df.
                    # This assumes that df_for_vp starts at the same point as out_df's valid data.
                    nan_padding_count = len(out_df) - len(rolling_poc)
                    if nan_padding_count > 0 : # Should not happen if window > len(df_for_vp)
                         #This should actually be an error or warning.
                         logger.warning(f"Mismatch in length for VP features for {resolution_name}. Padding: {nan_padding_count}. Rolling_poc len: {len(rolling_poc)}, out_df len: {len(out_df)}")
                         final_poc = [np.nan] * nan_padding_count + rolling_poc
                         final_val = [np.nan] * nan_padding_count + rolling_val
                         final_vah = [np.nan] * nan_padding_count + rolling_vah
                    else: # rolling_poc is longer, this is an error.
                        logger.error(f"VP feature list longer than target DataFrame for {resolution_name}. VP len: {len(rolling_poc)}, out_df len: {len(out_df)}. Skipping assignment.")
                        continue # Skip assigning these specific features

                    # If lengths still don't match, something is wrong, log and skip.
                    if len(final_poc) == len(out_df):
                         out_df[f'vp_poc_{window}w_{n_bins_vp}b'] = final_poc
                         out_df[f'vp_val_{window}w_{n_bins_vp}b'] = final_val
                         out_df[f'vp_vah_{window}w_{n_bins_vp}b'] = final_vah
                    else:
                        logger.error(f"Corrected VP feature list still not matching target DataFrame for {resolution_name}. Corrected len: {len(final_poc)}, out_df len: {len(out_df)}. Skipping assignment.")
                        continue


                logger.info(f"Calculated Volume Profile (POC, VAH, VAL) for {resolution_name} with window={window}, n_bins={n_bins_vp}")

    out_df.dropna(axis=1, how='all', inplace=True) # Drop columns that are ALL NaN
    logger.info(f"Finished calculating all features for {resolution_name}. Total columns: {len(out_df.columns)}")
    return out_df

# --- Main Processing Logic ---
def main(output_suffix=None):
    logger.info("Starting feature calculation for all time resolutions...")

    for res_name in TIME_RESOLUTIONS_TO_PROCESS:
        logger.info(f"--- Processing resolution: {res_name} ---")
        input_dir = INPUT_TIME_BAR_BASE_DIR / res_name
        output_dir = OUTPUT_FEATURE_BASE_DIR / res_name
        output_dir.mkdir(parents=True, exist_ok=True)

        input_files = get_sorted_input_files_for_resolution(input_dir, SYMBOL, res_name)

        if not input_files:
            logger.warning(f"No input files found for resolution {res_name}. Skipping.")
            continue

        all_dfs_for_resolution = []
        for i, file_path in enumerate(input_files):
            logger.info(f"Processing file {i+1}/{len(input_files)}: {file_path.name} for resolution {res_name}")
            try:
                df = pd.read_parquet(file_path)
                if df.empty:
                    logger.warning(f"File {file_path.name} is empty. Skipping.")
                    continue
                
                # Critical: Ensure 'open_timestamp' is present before feature calculation
                if 'open_timestamp' not in df.columns:
                    logger.error(f"File {file_path.name} is missing 'open_timestamp'. Cannot process. Columns: {df.columns.tolist()}")
                    continue
                
                features_df = calculate_features_for_df(df, res_name, FEATURE_CONFIG)
                if not features_df.empty:
                    all_dfs_for_resolution.append(features_df)
                else:
                    logger.warning(f"No features calculated for file {file_path.name} (resolution {res_name}).")

            except Exception as e:
                logger.error(f"Error processing file {file_path.name} for resolution {res_name}: {e}", exc_info=True)
                continue
        
        if not all_dfs_for_resolution:
            logger.warning(f"No DataFrames with features to concatenate for resolution {res_name}. Skipping output.")
            continue

        final_res_df = pd.concat(all_dfs_for_resolution).sort_index() # Sort by datetime index
        
        # Construct output filename with suffix
        base_filename = f"{SYMBOL}_time_bars_features_{res_name}"
        if output_suffix:
            output_filename = f"{base_filename}_{output_suffix}.parquet"
        else:
            output_filename = f"{base_filename}.parquet"
        output_file_path = output_dir / output_filename

        logger.info(f"Successfully processed all files for resolution {res_name}. Final DF shape: {final_res_df.shape}")
        try:
            final_res_df.to_parquet(output_file_path, index=True)
            logger.info(f"Successfully saved features for {res_name} to {output_file_path}")
        except Exception as e:
            logger.error(f"Error saving final DataFrame for {res_name} to {output_file_path}: {e}", exc_info=True)

    logger.info("--- Time Bar Feature Calculation Script Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate features for time bars and save them.")
    parser.add_argument(
        "--output-suffix", 
        type=str, 
        default=None, 
        help="Optional suffix to append to the output Parquet filenames (e.g., 'v2', 'logvol')."
    )
    args = parser.parse_args()

    main(output_suffix=args.output_suffix) # Pass the suffix to main 