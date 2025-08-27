import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import logging
import re
import numpy as np

# Attempt to import TA-Lib
try:
    import talib
    TALIB_AVAILABLE = True
    # Import all pattern functions from talib.abstract
    from talib.abstract import (
        CDL2CROWS, CDL3BLACKCROWS, CDL3INSIDE, CDL3LINESTRIKE, CDL3OUTSIDE, 
        CDL3STARSINSOUTH, CDL3WHITESOLDIERS, CDLABANDONEDBABY, 
        CDLADVANCEBLOCK, CDLBELTHOLD, CDLBREAKAWAY, CDLCLOSINGMARUBOZU, 
        CDLCONCEALBABYSWALL, CDLCOUNTERATTACK, CDLDARKCLOUDCOVER, CDLDOJI, 
        CDLDOJISTAR, CDLDRAGONFLYDOJI, CDLENGULFING, CDLEVENINGDOJISTAR, 
        CDLEVENINGSTAR, CDLGAPSIDESIDEWHITE, CDLGRAVESTONEDOJI, CDLHAMMER, 
        CDLHANGINGMAN, CDLHARAMI, CDLHARAMICROSS, CDLHIGHWAVE, 
        CDLHIKKAKE, CDLHIKKAKEMOD, CDLHOMINGPIGEON, CDLIDENTICAL3CROWS, 
        CDLINNECK, CDLINVERTEDHAMMER, CDLKICKING, CDLKICKINGBYLENGTH,
        CDLLADDERBOTTOM, CDLLONGLEGGEDDOJI, CDLLONGLINE, CDLMARUBOZU, 
        CDLMATCHINGLOW, CDLMATHOLD, CDLMORNINGDOJISTAR, CDLMORNINGSTAR, 
        CDLONNECK, CDLPIERCING, CDLRICKSHAWMAN, CDLRISEFALL3METHODS, 
        CDLSEPARATINGLINES, CDLSHOOTINGSTAR, CDLSHORTLINE, CDLSPINNINGTOP, 
        CDLSTALLEDPATTERN, CDLSTICKSANDWICH, CDLTAKURI, CDLTASUKIGAP, 
        CDLTHRUSTING, CDLTRISTAR, CDLUNIQUE3RIVER, CDLUPSIDEGAP2CROWS, 
        CDLXSIDEGAP3METHODS
    )
    ALL_TALIB_PATTERNS = {
        'CDL2CROWS': CDL2CROWS, 'CDL3BLACKCROWS': CDL3BLACKCROWS, 'CDL3INSIDE': CDL3INSIDE,
        'CDL3LINESTRIKE': CDL3LINESTRIKE, 'CDL3OUTSIDE': CDL3OUTSIDE, 'CDL3STARSINSOUTH': CDL3STARSINSOUTH,
        'CDL3WHITESOLDIERS': CDL3WHITESOLDIERS, 'CDLABANDONEDBABY': CDLABANDONEDBABY,
        'CDLADVANCEBLOCK': CDLADVANCEBLOCK, 'CDLBELTHOLD': CDLBELTHOLD, 'CDLBREAKAWAY': CDLBREAKAWAY,
        'CDLCLOSINGMARUBOZU': CDLCLOSINGMARUBOZU, 'CDLCONCEALBABYSWALL': CDLCONCEALBABYSWALL,
        'CDLCOUNTERATTACK': CDLCOUNTERATTACK, 'CDLDARKCLOUDCOVER': CDLDARKCLOUDCOVER, 'CDLDOJI': CDLDOJI,
        'CDLDOJISTAR': CDLDOJISTAR, 'CDLDRAGONFLYDOJI': CDLDRAGONFLYDOJI, 'CDLENGULFING': CDLENGULFING,
        'CDLEVENINGDOJISTAR': CDLEVENINGDOJISTAR, 'CDLEVENINGSTAR': CDLEVENINGSTAR,
        'CDLGAPSIDESIDEWHITE': CDLGAPSIDESIDEWHITE, 'CDLGRAVESTONEDOJI': CDLGRAVESTONEDOJI,
        'CDLHAMMER': CDLHAMMER, 'CDLHANGINGMAN': CDLHANGINGMAN, 'CDLHARAMI': CDLHARAMI,
        'CDLHARAMICROSS': CDLHARAMICROSS, 'CDLHIGHWAVE': CDLHIGHWAVE, 'CDLHIKKAKE': CDLHIKKAKE,
        'CDLHIKKAKEMOD': CDLHIKKAKEMOD, 'CDLHOMINGPIGEON': CDLHOMINGPIGEON,
        'CDLIDENTICAL3CROWS': CDLIDENTICAL3CROWS, 'CDLINNECK': CDLINNECK,
        'CDLINVERTEDHAMMER': CDLINVERTEDHAMMER, 'CDLKICKING': CDLKICKING,
        'CDLKICKINGBYLENGTH': CDLKICKINGBYLENGTH, 'CDLLADDERBOTTOM': CDLLADDERBOTTOM,
        'CDLLONGLEGGEDDOJI': CDLLONGLEGGEDDOJI, 'CDLLONGLINE': CDLLONGLINE, 'CDLMARUBOZU': CDLMARUBOZU,
        'CDLMATCHINGLOW': CDLMATCHINGLOW, 'CDLMATHOLD': CDLMATHOLD,
        'CDLMORNINGDOJISTAR': CDLMORNINGDOJISTAR, 'CDLMORNINGSTAR': CDLMORNINGSTAR,
        'CDLONNECK': CDLONNECK, 'CDLPIERCING': CDLPIERCING, 'CDLRICKSHAWMAN': CDLRICKSHAWMAN,
        'CDLRISEFALL3METHODS': CDLRISEFALL3METHODS, 'CDLSEPARATINGLINES': CDLSEPARATINGLINES,
        'CDLSHOOTINGSTAR': CDLSHOOTINGSTAR, 'CDLSHORTLINE': CDLSHORTLINE,
        'CDLSPINNINGTOP': CDLSPINNINGTOP, 'CDLSTALLEDPATTERN': CDLSTALLEDPATTERN,
        'CDLSTICKSANDWICH': CDLSTICKSANDWICH, 'CDLTAKURI': CDLTAKURI, 'CDLTASUKIGAP': CDLTASUKIGAP,
        'CDLTHRUSTING': CDLTHRUSTING, 'CDLTRISTAR': CDLTRISTAR, 'CDLUNIQUE3RIVER': CDLUNIQUE3RIVER,
        'CDLUPSIDEGAP2CROWS': CDLUPSIDEGAP2CROWS, 'CDLXSIDEGAP3METHODS': CDLXSIDEGAP3METHODS
    }
except ImportError:
    TALIB_AVAILABLE = False
    ALL_TALIB_PATTERNS = {}
    logging.warning("TA-Lib not found. Some features (e.g., advanced indicators, candlestick patterns) will not be available. Please install TA-Lib.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOLLAR_THRESHOLD_STR = "2M" # Should match the one used in Phase 2

INPUT_DOLLAR_BAR_DIR = PROJECT_ROOT / 'data' / 'dollar_bars' / DOLLAR_THRESHOLD_STR
OUTPUT_FEATURES_DIR = PROJECT_ROOT / 'data' / 'dollar_bars_features' / DOLLAR_THRESHOLD_STR
SYMBOL = 'BTCUSDT'

# Feature Engineering Parameters
SMA_PERIODS = [10, 20, 50]
EMA_PERIODS = [10, 20, 50]
RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
BBANDS_PERIOD = 20
BBANDS_STD_DEV = 2
VOLATILITY_PERIODS = [10, 20] # For std dev of log returns
VOLUME_SMA_PERIOD = 20
VOLUME_OSC_SHORT_PERIOD = 5
VOLUME_OSC_LONG_PERIOD = 20
ROC_PERIODS = [1, 5, 10]

# Target Calculation Parameters
RISK_REWARD_RATIO = 2.0
ATR_SL_MULTIPLIER = 1.5 # e.g., SL is 1.5 * ATR of entry bar
# MAX_LOOKAHEAD_BARS = 100 # Max bars to look for TP/SL hit (optional, README says no vertical barrier)
# Let's make MAX_LOOKAHEAD_BARS an actual parameter for the function, defaulting to a large number if we want "no barrier"
# For now, let's stick to the "no vertical barrier initially" as per README, effectively a very large lookahead.
# We can add a concrete MAX_LOOKAHEAD_BARS later if needed.

def get_sorted_input_files(input_dir_path, symbol_str, threshold_str):
    logger.info(f"Searching for input Dollar Bar Parquet files in: {input_dir_path}")
    file_pattern = re.compile(rf"^{symbol_str}_(\d{{4}})_(\d{{2}})_dollar_bars_{threshold_str}\.parquet$")
    
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
        logger.warning(f"No input files found matching pattern in {input_dir_path}")
        return []
        
    logger.info(f"Found and sorted {len(matched_files)} input dollar bar files.")
    return [mf['path'] for mf in matched_files]

def calculate_features(df):
    logger.info(f"Calculating features for DataFrame with {len(df)} bars...")
    
    # Prepare inputs for TA-Lib (expects numpy arrays, and specific keys like 'open', 'high', 'low', 'close', 'volume')
    inputs = {
        'open': df['open'].astype(float).values,
        'high': df['high'].astype(float).values,
        'low': df['low'].astype(float).values,
        'close': df['close'].astype(float).values
    }
    if 'volume' in df.columns:
        inputs['volume'] = df['volume'].astype(float).values
    if 'dollar_volume' in df.columns:
        df['dollar_volume_log'] = np.log1p(df['dollar_volume'].astype(float))

    # 1. Dollar Bar Specific Features (Keep existing, add log transform for volume if present)
    df['bar_duration_seconds'] = (df['close_timestamp'] - df['open_timestamp']).dt.total_seconds()
    if 'dollar_volume' in df.columns and 'volume' in df.columns and df['volume'].min() > 0:
        df['vwap_bar'] = df['dollar_volume'] / df['volume']
        df['volume_log'] = np.log1p(df['volume'].astype(float)) # Log transform for raw volume
    else:
        df['vwap_bar'] = np.nan
        if 'volume' in df.columns:
            df['volume_log'] = np.log1p(df['volume'].astype(float))
        else:
            df['volume_log'] = np.nan

    if TALIB_AVAILABLE:
        # 2. Moving Averages (TA-Lib)
        for period in SMA_PERIODS:
            df[f'sma_{period}'] = talib.SMA(inputs['close'], timeperiod=period)
        for period in EMA_PERIODS:
            df[f'ema_{period}'] = talib.EMA(inputs['close'], timeperiod=period)

        # 3. RSI (TA-Lib)
        df['rsi'] = talib.RSI(inputs['close'], timeperiod=RSI_PERIOD)
        
        # 4. ATR (Average True Range - TA-Lib)
        df['atr'] = talib.ATR(inputs['high'], inputs['low'], inputs['close'], timeperiod=ATR_PERIOD)

        # 4b. NATR (Normalized Average True Range - TA-Lib)
        df['natr'] = talib.NATR(inputs['high'], inputs['low'], inputs['close'], timeperiod=ATR_PERIOD)

        # 6. Volatility (Standard Deviation of Log Returns) - Log returns calculated first
        df['log_return'] = np.log(df['close'] / df['close'].shift()) # Keep pandas for this basic one
        for period in VOLATILITY_PERIODS:
            df[f'volatility_{period}'] = talib.STDDEV(df['log_return'], timeperiod=period, nbdev=1)

        # 7. MACD (Moving Average Convergence Divergence - TA-Lib)
        macd, macdsignal, macdhist = talib.MACD(inputs['close'], 
                                                fastperiod=MACD_FAST_PERIOD, 
                                                slowperiod=MACD_SLOW_PERIOD, 
                                                signalperiod=MACD_SIGNAL_PERIOD)
        df['macd'] = macd
        df['macd_signal'] = macdsignal
        df['macd_hist'] = macdhist

        # 8. Bollinger Bands (TA-Lib)
        upperband, middleband, lowerband = talib.BBANDS(inputs['close'], 
                                                        timeperiod=BBANDS_PERIOD, 
                                                        nbdevup=BBANDS_STD_DEV, 
                                                        nbdevdn=BBANDS_STD_DEV, 
                                                        matype=0) # MA_Type: 0 for SMA
        df[f'bb_upper_{BBANDS_PERIOD}'] = upperband
        df[f'bb_middle_{BBANDS_PERIOD}'] = middleband
        df[f'bb_lower_{BBANDS_PERIOD}'] = lowerband
        df[f'bb_width_{BBANDS_PERIOD}'] = (upperband - lowerband) / middleband
        df[f'bb_percent_b_{BBANDS_PERIOD}'] = (df['close'] - lowerband) / (upperband - lowerband)
        
        # 9. Volume-based features (TA-Lib if applicable, else pandas)
        if 'volume' in inputs:
            df[f'volume_sma_{VOLUME_SMA_PERIOD}'] = talib.SMA(inputs['volume'], timeperiod=VOLUME_SMA_PERIOD)
            
            # PVO (Percentage Volume Oscillator) - TA-Lib doesn't have direct PVO like TradingView
            # We'll keep the manual SMA-based oscillator
            vol_sma_short = df['volume'].rolling(window=VOLUME_OSC_SHORT_PERIOD).mean()
            vol_sma_long = df['volume'].rolling(window=VOLUME_OSC_LONG_PERIOD).mean()
            df['volume_oscillator'] = ((vol_sma_short - vol_sma_long) / vol_sma_long) * 100
        else:
            df[f'volume_sma_{VOLUME_SMA_PERIOD}'] = np.nan
            df['volume_oscillator'] = np.nan

        # 10. Price Rate of Change (ROC - TA-Lib)
        for period in ROC_PERIODS:
            df[f'roc_{period}'] = talib.ROC(inputs['close'], timeperiod=period)

        # 11. Stochastic Oscillator (TA-Lib)
        slowk, slowd = talib.STOCH(inputs['high'], inputs['low'], inputs['close'], 
                                   fastk_period=14, slowk_period=3, slowk_matype=0, 
                                   slowd_period=3, slowd_matype=0)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd

        # 12. Candlestick Patterns (TA-Lib)
        logger.info(f"Calculating {len(ALL_TALIB_PATTERNS)} candlestick patterns...")
        # Prepare DataFrame for talib.abstract, which expects columns named 'open', 'high', 'low', 'close', 'volume'
        # Our `inputs` dict already holds these as numpy arrays. Need to pass to each pattern func.
        for pattern_name, pattern_func in ALL_TALIB_PATTERNS.items():
            try:
                # Pattern functions from talib.abstract typically expect Series/DataFrames 
                # or can work with numpy arrays if passed correctly via talib.func directly.
                # Here, we use the functions directly from talib.abstract which should accept numpy arrays.
                pattern_result = pattern_func(inputs['open'], inputs['high'], inputs['low'], inputs['close'])
                df[pattern_name] = pattern_result
            except Exception as e:
                logger.warning(f"Could not calculate pattern {pattern_name}: {e}")
                df[pattern_name] = np.nan
        logger.info("Finished candlestick patterns.")

    else: # TALIB_AVAILABLE is False
        logger.warning("TA-Lib not available. Falling back to pandas for some features, others will be skipped.")
        # Original pandas calculations (subset)
        for period in SMA_PERIODS:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        for period in EMA_PERIODS:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift())
        low_close_prev = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=ATR_PERIOD).mean()
        df['log_return'] = np.log(df['close'] / df['close'].shift())
        for period in VOLATILITY_PERIODS:
            df[f'volatility_{period}'] = df['log_return'].rolling(window=period).std()
        # ... (other pandas fallbacks can be added here if desired)
        # Mark TA-Lib specific features as NaN
        df['natr'] = np.nan
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_hist'] = np.nan
        df[f'bb_upper_{BBANDS_PERIOD}'] = np.nan
        df[f'bb_middle_{BBANDS_PERIOD}'] = np.nan
        df[f'bb_lower_{BBANDS_PERIOD}'] = np.nan
        df[f'bb_width_{BBANDS_PERIOD}'] = np.nan
        df[f'bb_percent_b_{BBANDS_PERIOD}'] = np.nan
        df['stoch_k'] = np.nan
        df['stoch_d'] = np.nan
        for pattern_name in ALL_TALIB_PATTERNS.keys():
            df[pattern_name] = np.nan

    # 13. Time-based Categorical & Cyclical Features
    if 'close_timestamp' in df.columns:
        df['close_timestamp'] = pd.to_datetime(df['close_timestamp'])
        df['hour_of_day'] = df['close_timestamp'].dt.hour
        df['day_of_week'] = df['close_timestamp'].dt.dayofweek # Monday=0, Sunday=6
        
        # Sin/Cos transformations
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    else:
        logger.warning("'close_timestamp' column not found. Skipping time-based categorical & cyclical features.")
        df['hour_of_day'] = np.nan
        df['day_of_week'] = np.nan
        df['hour_sin'] = np.nan
        df['hour_cos'] = np.nan
        df['day_sin'] = np.nan
        df['day_cos'] = np.nan
        
    # 14. Relative Features (Example)
    if 'atr' in df.columns and 'close_open_diff' not in df.columns:
        # Calculate close_open_diff if not present but ATR is
        df['close_open_diff'] = df['close'] - df['open']
        
    if 'atr' in df.columns and 'close_open_diff' in df.columns and df['atr'].notna().any():
        df['close_open_diff_rel_atr'] = df['close_open_diff'] / (df['atr'] + 1e-9) # Add epsilon to avoid division by zero
    else:
        df['close_open_diff_rel_atr'] = np.nan
        
    if 'atr' in df.columns and 'high_low_diff' not in df.columns:
        # Calculate high_low_diff if not present but ATR is
        df['high_low_diff'] = df['high'] - df['low']

    if 'atr' in df.columns and 'high_low_diff' in df.columns and df['atr'].notna().any():
        df['high_low_diff_rel_atr'] = df['high_low_diff'] / (df['atr'] + 1e-9)
    else:
        df['high_low_diff_rel_atr'] = np.nan

    # --- Final Data Cleaning/Adjustments for Intra-bar Features ---
    # Handle potential infinity values in taker_buy_sell_ratio (from Phase 2)
    if 'taker_buy_sell_ratio' in df.columns:
        df['taker_buy_sell_ratio'] = df['taker_buy_sell_ratio'].replace([np.inf, -np.inf], np.nan)
        logger.info("Replaced np.inf and -np.inf in 'taker_buy_sell_ratio' with np.nan.")

    logger.info("Finished calculating features.")
    return df

def calculate_targets(df, atr_col_name='atr', sl_atr_multiplier=1.5, risk_reward_ratio=2.0, max_lookahead_bars=None):
    logger.info(f"Calculating targets with R:R={risk_reward_ratio}, SL_ATR_mult={sl_atr_multiplier}, ATR_col='{atr_col_name}'")
    n = len(df)
    target = np.full(n, np.nan) # Initialize target column with NaNs

    # Loop from the first bar up to the second-to-last bar (need at least one subsequent bar for entry)
    # And if max_lookahead_bars is used, ensure we don't try to look beyond the dataframe for the *first* check.
    # The loop for 'i' should go up to n - 2 if we need at least one bar i+1 for entry.
    # If max_lookahead_bars is, say, 1, we need i+1 and i+2.
    # So, loop up to n - (1 + (1 if max_lookahead_bars is None else max_lookahead_bars_effective_min_1) )
    # Simpler: Loop up to n-2, and inside the lookahead loop, cap `j` by `min(i + 1 + max_lookahead_bars, n)` if max_lookahead_bars is set.

    for i in range(n - 1): # Signal bar is 'i', entry is on 'i+1'
        entry_price = df['open'].iloc[i+1]
        atr_value = df[atr_col_name].iloc[i]

        # Skip if ATR is not valid for setting risk
        if pd.isna(atr_value) or atr_value <= 0:
            # target[i] remains np.nan
            continue

        risk_amount = sl_atr_multiplier * atr_value
        
        # For Long trade (can add logic for short later or a separate target column)
        sl_target_price_long = entry_price - risk_amount
        tp_target_price_long = entry_price + (risk_reward_ratio * risk_amount)

        outcome_found = False
        # Look ahead for outcome
        # If max_lookahead_bars is None or 0, look until the end of the dataframe
        lookahead_limit = n
        if max_lookahead_bars is not None and max_lookahead_bars > 0:
            lookahead_limit = min(i + 1 + max_lookahead_bars, n) # i+1 is entry bar, so look max_lookahead_bars *from* entry

        for j in range(i + 1, lookahead_limit): # Start from entry bar (i+1)
            # Check for TP hit (Long)
            if df['high'].iloc[j] >= tp_target_price_long:
                target[i] = 1 # TP hit
                outcome_found = True
                break
            # Check for SL hit (Long)
            elif df['low'].iloc[j] <= sl_target_price_long:
                target[i] = 0 # SL hit
                outcome_found = True
                break
        
        # If loop finishes and no outcome (and max_lookahead_bars was hit, if specified), target remains NaN
        # This implicitly handles the "no vertical barrier" if max_lookahead_bars is None.

    df['target_long'] = target # For now, explicitly naming it target_long
    logger.info(f"Finished calculating targets. Distribution: TP (1): {np.sum(target == 1)}, SL (0): {np.sum(target == 0)}, Unresolved (NaN): {np.sum(np.isnan(target))}")
    return df

def main():
    logger.info(f"Starting Phase 3: Feature Engineering & Target Calculation for {DOLLAR_THRESHOLD_STR} dollar bars.")
    OUTPUT_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory for features: {OUTPUT_FEATURES_DIR}")

    input_files = get_sorted_input_files(INPUT_DOLLAR_BAR_DIR, SYMBOL, DOLLAR_THRESHOLD_STR)
    if not input_files:
        logger.error("No input dollar bar files found. Aborting.")
        return

    all_raw_bars = [] # Changed variable name

    for i, file_path in enumerate(input_files):
        logger.info(f"Processing input dollar bar file {i+1}/{len(input_files)}: {file_path.name}")
        try:
            df_bars = pd.read_parquet(file_path)
            if df_bars.empty:
                logger.info(f"File {file_path.name} is empty. Skipping.")
                continue
            
            all_raw_bars.append(df_bars) # Append raw df_bars
            
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}", exc_info=True)

    if not all_raw_bars: # Check the new list name
        logger.error("No data was loaded from input files. Aborting.")
        return

    logger.info(f"Concatenating data from {len(all_raw_bars)} files...") # Use new list name
    master_df = pd.concat(all_raw_bars, ignore_index=True)
    
    # Sort by timestamp just in case, critical for feature and target calculation
    master_df.sort_values(by='close_timestamp', inplace=True)
    master_df.reset_index(drop=True, inplace=True) # Reset index after sort

    # Calculate features on the full, sorted, combined DataFrame to ensure continuity of rolling features
    logger.info("Calculating features on the combined DataFrame...")
    master_df_with_features = calculate_features(master_df.copy()) # Use .copy()

    # Now calculate targets on the full, sorted, feature-rich dataset
    # Using global constants for parameters, and explicitly setting max_lookahead_bars=None for no vertical barrier
    logger.info("Calculating targets on the feature-rich DataFrame...")
    master_df_with_targets = calculate_targets(
        master_df_with_features.copy(), # Use .copy()
        atr_col_name='atr', # Assuming 'atr' is the column name from calculate_features
        sl_atr_multiplier=ATR_SL_MULTIPLIER,
        risk_reward_ratio=RISK_REWARD_RATIO,
        max_lookahead_bars=None # Explicitly no vertical time barrier for now
    )

    # Save the final feature-rich data
    # For now, saving as one large file for simplicity, but partitioning is better for large datasets.
    output_filename = f"{SYMBOL}_all_features_targets_{DOLLAR_THRESHOLD_STR}.parquet"
    output_file_path = OUTPUT_FEATURES_DIR / output_filename
    
    try:
        master_df_with_targets.to_parquet(output_file_path, index=False, engine='pyarrow')
        logger.info(f"Saved final dataset with {len(master_df_with_targets)} rows to {output_file_path}")
    except Exception as e:
        logger.error(f"Error saving final dataset to {output_file_path}: {e}", exc_info=True)

    logger.info("Phase 3: Feature Engineering & Target Calculation finished.")

if __name__ == "__main__":
    main() 