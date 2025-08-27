import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Assuming the script is run from the root of trading-agent-v3 directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
BASE_FEATURES_DIR = PROJECT_ROOT / 'data' / 'multi_res_features' / '2M_x10_x16'
INPUT_FEATURES_FILE = BASE_FEATURES_DIR / 'BTCUSDT_all_multi_res_features.parquet' # Full dataset
OUTPUT_FEATURES_TARGETS_FILE = BASE_FEATURES_DIR / 'BTCUSDT_all_multi_res_features_targets.parquet' # Full dataset output

# Target Calculation Parameters (from phase3_calculate_features_targets.py and README)
# These could be made configurable or command-line arguments if needed
ATR_COL_NAME_FOR_TARGETS = 's_atr_14' # Using ATR from short-term bars
RISK_REWARD_RATIO = 2.0
ATR_SL_MULTIPLIER = 1.5
MAX_LOOKAHEAD_BARS = None # Set to an integer for a vertical barrier, None for no limit (until end of data)

def calculate_long_targets(df, 
                           atr_col_name='s_atr_14', 
                           sl_atr_multiplier=1.5, 
                           risk_reward_ratio=2.0, 
                           max_lookahead_bars=None):
    """
    Calculates target labels for long trades based on ATR for stop-loss and a risk/reward ratio for take-profit.

    Args:
        df (pd.DataFrame): Input DataFrame with OHLC data and an ATR column.
                           Must contain 'open', 'high', 'low', 'close', and the atr_col_name.
        atr_col_name (str): The name of the ATR column to use for calculations.
        sl_atr_multiplier (float): Multiplier for ATR to set the stop-loss distance.
        risk_reward_ratio (float): Desired risk/reward ratio for the take-profit.
        max_lookahead_bars (Optional[int]): Maximum number of bars to look ahead for TP/SL.
                                           If None, looks until the end of the DataFrame.

    Returns:
        pd.Series: A series with target labels (1 for TP hit, 0 for SL hit, NaN otherwise).
    """
    logger.info(f"Calculating long targets using ATR column: {atr_col_name}, SL mult: {sl_atr_multiplier}, R/R: {risk_reward_ratio}")
    
    n = len(df)
    target_long = pd.Series(np.nan, index=df.index)

    if atr_col_name not in df.columns:
        logger.error(f"ATR column '{atr_col_name}' not found in DataFrame. Cannot calculate targets.")
        return target_long
    if 'open' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
        logger.error("Required columns 'open', 'high', 'low' not found. Cannot calculate targets.")
        return target_long

    for i in range(n - 1): # Iterate up to n-2 because we look at bar i+1
        entry_price = df['open'].iloc[i+1]
        atr_value_signal_bar = df[atr_col_name].iloc[i]

        if pd.isna(entry_price) or pd.isna(atr_value_signal_bar) or atr_value_signal_bar <= 0:
            # Cannot determine entry or risk, or ATR is invalid
            continue

        stop_loss_price = entry_price - (sl_atr_multiplier * atr_value_signal_bar)
        risk_per_share = entry_price - stop_loss_price
        take_profit_price = entry_price + (risk_reward_ratio * risk_per_share)

        # Determine the end point for lookahead
        lookahead_end_index = n
        if max_lookahead_bars is not None:
            lookahead_end_index = min(n, i + 1 + max_lookahead_bars)
            
        # logger.debug(f"Bar {i}: Entry={entry_price:.2f}, ATR={atr_value_signal_bar:.2f}, SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}, Lookahead: {i+1} to {lookahead_end_index-1}")


        for j in range(i + 1, lookahead_end_index): # Look from bar i+1 onwards
            bar_high = df['high'].iloc[j]
            bar_low = df['low'].iloc[j]

            if pd.isna(bar_high) or pd.isna(bar_low):
                continue # Skip bars with NaN high/low

            # Check for TP hit first (important if a bar could hit both)
            if bar_high >= take_profit_price:
                target_long.iloc[i] = 1 # TP hit
                # logger.debug(f"  TP hit at bar {j} (high: {bar_high})")
                break 
            
            # Check for SL hit
            if bar_low <= stop_loss_price:
                target_long.iloc[i] = 0 # SL hit
                # logger.debug(f"  SL hit at bar {j} (low: {bar_low})")
                break
        # If loop finishes without break, target_long.iloc[i] remains NaN (outcome unresolved within lookahead)

    logger.info(f"Target calculation complete. TP hits: {target_long.eq(1).sum()}, SL hits: {target_long.eq(0).sum()}, Unresolved: {target_long.isna().sum()}")
    return target_long

def run_target_labeling():
    """
    Loads the multi-resolution features, calculates target labels, and saves the combined DataFrame.
    """
    logger.info(f"Starting target labeling process...")
    logger.info(f"Loading features from: {INPUT_FEATURES_FILE}")

    if not INPUT_FEATURES_FILE.exists():
        logger.error(f"Input feature file not found: {INPUT_FEATURES_FILE}")
        return

    try:
        features_df = pd.read_parquet(INPUT_FEATURES_FILE)
        logger.info(f"Successfully loaded features DataFrame with shape: {features_df.shape}")
    except Exception as e:
        logger.error(f"Error loading features Parquet file: {e}")
        return

    # Calculate long targets
    # The target function uses 'open', 'high', 'low' from the original (short) bars for entry and TP/SL checks.
    # The ATR for risk calculation is taken from the specified atr_col_name (e.g., s_atr_14).
    features_df['target_long'] = calculate_long_targets(
        df=features_df,
        atr_col_name=ATR_COL_NAME_FOR_TARGETS,
        sl_atr_multiplier=ATR_SL_MULTIPLIER,
        risk_reward_ratio=RISK_REWARD_RATIO,
        max_lookahead_bars=MAX_LOOKAHEAD_BARS
    )

    # --- Future: Calculate short targets here and add as 'target_short' ---
    # features_df['target_short'] = calculate_short_targets(df=features_df, ...) 
    # logger.info("Short targets would be calculated here.")

    # Ensure output directory exists
    OUTPUT_FEATURES_TARGETS_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving final DataFrame with targets to: {OUTPUT_FEATURES_TARGETS_FILE}")
    try:
        features_df.to_parquet(OUTPUT_FEATURES_TARGETS_FILE, index=False)
        logger.info(f"Successfully saved DataFrame with targets. Final shape: {features_df.shape}")
    except Exception as e:
        logger.error(f"Error saving final Parquet file: {e}")

if __name__ == "__main__":
    run_target_labeling() 