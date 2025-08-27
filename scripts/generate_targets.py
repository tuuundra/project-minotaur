import pandas as pd
import numpy as np
from pathlib import Path
import glob
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to sys.path to allow importing from other Minotaur modules if necessary
# (though for this script, we'll copy the target function directly)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Configuration ---
CONSOLIDATED_FEATURES_FILE = PROJECT_ROOT / "data" / "consolidated_features_all.parquet"
RAW_1MIN_BARS_DIR = PROJECT_ROOT / "data" / "time_bars" / "1min" # Directory containing monthly 1-min raw bars
OUTPUT_FILE = PROJECT_ROOT / "data" / "consolidated_features_targets_all.parquet"

# Target calculation parameters
TP_PCT = 2.0  # Take Profit percentage
SL_PCT = 1.0  # Stop Loss percentage

# --- Target Calculation Function (copied and adapted from minotaur_v1.py) ---
def create_fixed_pct_sl_tp_target(data_ohlcv, tp_pct=0.5, sl_pct=0.25):
    """
    Calculates a binary target variable based on whether a fixed percentage
    take profit (TP) is hit before a fixed percentage stop loss (SL).

    The input DataFrame `data_ohlcv` MUST have 'Open', 'High', 'Low', 'Close' columns
    and a DatetimeIndex.

    Args:
        data_ohlcv (pd.DataFrame): DataFrame with OHLCV data, DatetimeIndexed.
        tp_pct (float): Take profit percentage (e.g., 0.5 for 0.5%).
        sl_pct (float): Stop loss percentage (e.g., 0.25 for 0.25%).

    Returns:
        pd.Series: Binary target (1 for TP hit first, 0 for SL hit first, NaN if neither within future bars).
                   Index will match `data_ohlcv`.
    """
    logger.info(f"Starting fixed percentage SL/TP target calculation with TP={tp_pct}%, SL={sl_pct}%")
    if not isinstance(data_ohlcv.index, pd.DatetimeIndex):
        raise ValueError("Input data_ohlcv must have a DatetimeIndex.")
    if not all(col in data_ohlcv.columns for col in ['Open', 'High', 'Low', 'Close']):
        raise ValueError("Input data_ohlcv must contain 'Open', 'High', 'Low', 'Close' columns.")

    n_rows = len(data_ohlcv)
    target = pd.Series(np.nan, index=data_ohlcv.index, name=f'target_tp{tp_pct}_sl{sl_pct}')

    # Convert percentages to factors
    tp_factor = tp_pct / 100.0
    sl_factor = sl_pct / 100.0

    # Using .values for faster access in the loop
    high_prices = data_ohlcv['High'].values
    low_prices = data_ohlcv['Low'].values
    open_prices = data_ohlcv['Open'].values # Using Open as the entry price reference

    for i in range(n_rows):
        if i % 100000 == 0 and i > 0:
            logger.info(f"Processed {i}/{n_rows} rows for target calculation...")

        entry_price = open_prices[i] # Assuming entry at the open of the current bar i
        if pd.isna(entry_price):
            continue

        tp_price = entry_price * (1 + tp_factor)
        sl_price = entry_price * (1 - sl_factor)

        # Iterate through subsequent bars to see if TP or SL is hit
        for j in range(i + 1, n_rows): # Look from the next bar onwards
            # TP hit if high of a subsequent bar crosses tp_price
            if high_prices[j] >= tp_price:
                # Check if SL was also hit in the same bar or before TP
                if low_prices[j] <= sl_price:
                    # If both hit in the same bar, conservative approach might be SL
                    # or based on intra-bar movement (not available here).
                    # For simplicity, if low is <= SL, and high >= TP in the same bar,
                    # we need a rule. Let's assume SL hit if low_prices[j] <= sl_price
                    # *before* or *at the same time* as high_prices[j] >= tp_price.
                    # Without tick data, it's an approximation.
                    # A common convention: if a bar touches both, assume SL is hit first.
                    target.iloc[i] = 0 # SL hit
                    break
                else:
                    target.iloc[i] = 1 # TP hit
                    break
            # SL hit if low of a subsequent bar crosses sl_price
            elif low_prices[j] <= sl_price:
                target.iloc[i] = 0 # SL hit
                break
        # If loop finishes without break, neither was hit (or end of data) -> NaN (already default)

    logger.info(f"Finished target calculation. Target distribution:")
    logger.info(target.value_counts(dropna=False))
    return target

# --- Main Script Logic ---
def main():
    logger.info("--- Starting Target Generation Script ---")

    # 1. Load Consolidated Features
    logger.info(f"Loading consolidated features from: {CONSOLIDATED_FEATURES_FILE}")
    if not CONSOLIDATED_FEATURES_FILE.exists():
        logger.error(f"Consolidated features file not found: {CONSOLIDATED_FEATURES_FILE}")
        return
    try:
        features_df = pd.read_parquet(CONSOLIDATED_FEATURES_FILE)
        logger.info(f"Loaded consolidated features. Shape: {features_df.shape}")
        if not isinstance(features_df.index, pd.DatetimeIndex):
            logger.warning("Consolidated features DataFrame does not have a DatetimeIndex. Attempting to set it.")
            # Assuming 'timestamp' or 'datetime' column exists if index is not already datetime
            if 'timestamp' in features_df.columns:
                features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
                features_df = features_df.set_index('timestamp')
            elif 'datetime' in features_df.columns: # Common alternative name
                 features_df['datetime'] = pd.to_datetime(features_df['datetime'])
                 features_df = features_df.set_index('datetime')
            else:
                logger.error("Could not find a suitable timestamp column to set as DatetimeIndex.")
                return
            if not isinstance(features_df.index, pd.DatetimeIndex):
                 logger.error("Failed to convert index to DatetimeIndex.")
                 return
            logger.info("Successfully set DatetimeIndex for consolidated features.")
        features_df = features_df.sort_index() # Ensure it's sorted for reliable merges
        logger.info(f"Consolidated features index type: {type(features_df.index)}, is_monotonic_increasing: {features_df.index.is_monotonic_increasing}")

    except Exception as e:
        logger.error(f"Error loading consolidated features: {e}")
        return

    # 2. Load and Prepare 1-minute Raw OHLCV Data
    logger.info(f"Loading 1-minute raw OHLCV bars from: {RAW_1MIN_BARS_DIR}")
    if not RAW_1MIN_BARS_DIR.exists() or not RAW_1MIN_BARS_DIR.is_dir():
        logger.error(f"Raw 1-minute bars directory not found: {RAW_1MIN_BARS_DIR}")
        return

    raw_1min_files = sorted(list(RAW_1MIN_BARS_DIR.glob("*.parquet")))
    if not raw_1min_files:
        logger.error(f"No Parquet files found in {RAW_1MIN_BARS_DIR}")
        return
    logger.info(f"Found {len(raw_1min_files)} raw 1-minute Parquet files.")

    all_1min_ohlcv_dfs = []
    for f_path in raw_1min_files:
        try:
            monthly_df = pd.read_parquet(f_path)
            
            # Ensure datetime index from 'open_timestamp'
            if 'open_timestamp' in monthly_df.columns:
                monthly_df['open_timestamp'] = pd.to_datetime(monthly_df['open_timestamp'])
                monthly_df = monthly_df.set_index('open_timestamp', drop=True)
                monthly_df.index.name = 'timestamp' # Standardize index name
            else:
                logger.warning(f"File {f_path.name} is missing the 'open_timestamp' column. Skipping.")
                continue
            
            if not isinstance(monthly_df.index, pd.DatetimeIndex):
                logger.warning(f"Failed to set DatetimeIndex for {f_path.name} from 'open_timestamp'. Skipping.")
                continue

            # Standardize column names to what create_fixed_pct_sl_tp_target expects (Open, High, Low, Close)
            # The raw files use lowercase: open, high, low, close.
            rename_map = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume' # Also standardize volume if present and needed later
            }
            # Only rename columns that exist to avoid errors if some are missing (though OHLC are critical)
            existing_rename_map = {k: v for k, v in rename_map.items() if k in monthly_df.columns}
            monthly_df = monthly_df.rename(columns=existing_rename_map)
            
            # Select only necessary columns (now using standardized names)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] 
            if not all(col in monthly_df.columns for col in required_cols[:4]): # Check for OHLC
                logger.warning(f"File {f_path.name} is missing one or more required OHLC columns after rename attempt. Skipping. Columns found: {monthly_df.columns.tolist()}")
                continue
            
            all_1min_ohlcv_dfs.append(monthly_df[required_cols])
        except Exception as e:
            logger.error(f"Error processing raw 1-minute file {f_path.name}: {e}")
            continue
    
    if not all_1min_ohlcv_dfs:
        logger.error("No valid 1-minute OHLCV data could be loaded.")
        return

    raw_ohlcv_df = pd.concat(all_1min_ohlcv_dfs)
    raw_ohlcv_df = raw_ohlcv_df.sort_index() # Ensure chronological order
    raw_ohlcv_df = raw_ohlcv_df[~raw_ohlcv_df.index.duplicated(keep='first')] # Remove any duplicate timestamps

    logger.info(f"Loaded and concatenated all 1-minute raw OHLCV data. Shape: {raw_ohlcv_df.shape}")
    logger.info(f"Raw OHLCV index type: {type(raw_ohlcv_df.index)}, is_monotonic_increasing: {raw_ohlcv_df.index.is_monotonic_increasing}")
    logger.info(f"Raw OHLCV head:")
    logger.info(raw_ohlcv_df.head())
    logger.info(f"Raw OHLCV tail:")
    logger.info(raw_ohlcv_df.tail())


    # 3. Calculate Targets
    # Align raw_ohlcv_df to the feature_df's index to ensure we are calculating targets
    # for the same period and at the same frequency.
    # This is important if features_df has a more restricted date range than raw_ohlcv_df.
    # We calculate targets on the full raw_ohlcv_df, then align.
    
    # First, ensure both are timezone-naive or consistently timezone-aware to avoid issues.
    # The generate_time_bars script should produce timezone-naive UTC timestamps.
    if features_df.index.tz is not None:
        logger.info("Consolidated features index is timezone-aware. Converting to UTC then naive.")
        features_df.index = features_df.index.tz_convert('UTC').tz_localize(None)
    if raw_ohlcv_df.index.tz is not None:
        logger.info("Raw OHLCV index is timezone-aware. Converting to UTC then naive.")
        raw_ohlcv_df.index = raw_ohlcv_df.index.tz_convert('UTC').tz_localize(None)

    # Reindex raw_ohlcv_df to match the feature set's timeline *before* target calculation
    # This ensures that the target calculation loop operates on data relevant to the features.
    # We need Open, High, Low, Close for target calc.
    # We need to be careful here: the `create_fixed_pct_sl_tp_target` looks into the *future*
    # of the dataframe it's given. So, the `ohlcv_for_targets` should span the range of
    # `features_df` PLUS enough future data for targets to resolve.

    # Find common start and ensure ohlcv_for_targets extends beyond features_df end
    common_start_time = max(features_df.index.min(), raw_ohlcv_df.index.min())
    # For target calculation, we need raw OHLCV data that aligns with feature timestamps
    # AND extends sufficiently into the future.
    # Let's use the raw_ohlcv_df as is, since it should cover the whole period.
    # The target series will then be aligned to the features_df.

    logger.info(f"Calculating targets using TP={TP_PCT}%, SL={SL_PCT}% on the full 1-min OHLCV data.")
    target_series = create_fixed_pct_sl_tp_target(raw_ohlcv_df.copy(), tp_pct=TP_PCT, sl_pct=SL_PCT)
    
    # 4. Merge Targets
    logger.info(f"Merging targets with consolidated features. Target series length: {len(target_series)}")
    # Align target_series to features_df.index. This handles cases where
    # raw_ohlcv_df might be longer or have slightly different timestamps.
    # We only want targets for timestamps present in features_df.
    final_df = features_df.join(target_series, how='left')

    logger.info(f"Shape of final DataFrame after merging targets: {final_df.shape}")
    logger.info(f"Target column name: {target_series.name}")
    logger.info(f"Value counts for target column in final_df (after merge):")
    logger.info(final_df[target_series.name].value_counts(dropna=False))

    # 5. Save Final Dataset
    logger.info(f"Saving final dataset with features and targets to: {OUTPUT_FILE}")
    try:
        final_df.to_parquet(OUTPUT_FILE)
        logger.info("Successfully saved final dataset.")
    except Exception as e:
        logger.error(f"Error saving final dataset: {e}")

    logger.info("--- Target Generation Script Finished ---")

if __name__ == "__main__":
    main() 