import pandas as pd
import talib
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # Should point to minotaur/
INPUT_FILE_PATH = PROJECT_ROOT / "data" / "time_bars_features_with_vp" / "15min" / "BTCUSDT_time_bars_features_15min_v2.parquet"
OUTPUT_FILE_PATH = INPUT_FILE_PATH # Overwrite the existing file

# Stochastic Parameters
FASTK_PERIOD = 14
SLOWK_PERIOD = 3
SLOWD_PERIOD = 3
STOCH_OB_THRESHOLD = 80
STOCH_OS_THRESHOLD = 20

# Suffix for new columns
SUFFIX = "_tb_15min" 

def add_stochastic_features(df):
    """
    Calculates and adds Stochastic Oscillator features and OB/OS flags to the DataFrame.
    """
    logger.info("Starting Stochastic feature calculation...")

    # Identify actual High, Low, Close column names (they might have suffixes)
    # Common column names from calculate_time_bar_features.py are 'High', 'Low', 'Close', 'Volume'
    # After consolidation script processing (which isn't run yet on these files), they might get suffixes.
    # For now, let's assume they are 'High', 'Low', 'Close' or we find them.
    
    # Attempt to find columns that are likely high, low, close.
    # The source file from `calculate_time_bar_features.py` should have 'High', 'Low', 'Close'
    # before any further suffixing by other scripts.
    high_col = None
    low_col = None
    close_col = None

    if 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
        high_col = 'High'
        low_col = 'Low'
        close_col = 'Close'
    else:
        # Try to find them with common suffixes if direct names aren't present
        potential_high = [col for col in df.columns if 'high' in col.lower() and SUFFIX in col.lower()]
        potential_low = [col for col in df.columns if 'low' in col.lower() and SUFFIX in col.lower()]
        potential_close = [col for col in df.columns if 'close' in col.lower() and SUFFIX in col.lower()]
        
        if potential_high: high_col = potential_high[0]
        if potential_low: low_col = potential_low[0]
        if potential_close: close_col = potential_close[0]

    if not all([high_col, low_col, close_col]):
        logger.error(f"Could not reliably identify High, Low, or Close columns. Found: H={high_col}, L={low_col}, C={close_col}. Aborting.")
        raise ValueError("Missing HLC columns for Stochastic calculation")

    logger.info(f"Using columns for STOCH: High='{high_col}', Low='{low_col}', Close='{close_col}'")

    # Calculate Stochastic
    slowk, slowd = talib.STOCH(
        df[high_col],
        df[low_col],
        df[close_col],
        fastk_period=FASTK_PERIOD,
        slowk_period=SLOWK_PERIOD,
        slowk_matype=0, # SMA
        slowd_period=SLOWD_PERIOD,
        slowd_matype=0  # SMA
    )

    stoch_k_col_name = f'stoch_slowk_{FASTK_PERIOD}_{SLOWK_PERIOD}_{SLOWD_PERIOD}{SUFFIX}'
    stoch_d_col_name = f'stoch_slowd_{FASTK_PERIOD}_{SLOWK_PERIOD}_{SLOWD_PERIOD}{SUFFIX}'
    
    df[stoch_k_col_name] = slowk
    df[stoch_d_col_name] = slowd

    # Add Overbought/Oversold flags based on slowk
    os_flag_col_name = f'is_stoch_os_{STOCH_OS_THRESHOLD}{SUFFIX}'
    ob_flag_col_name = f'is_stoch_ob_{STOCH_OB_THRESHOLD}{SUFFIX}'

    df[os_flag_col_name] = (df[stoch_k_col_name] < STOCH_OS_THRESHOLD).astype(int)
    df[ob_flag_col_name] = (df[stoch_k_col_name] > STOCH_OB_THRESHOLD).astype(int)

    logger.info(f"Added Stochastic columns: {stoch_k_col_name}, {stoch_d_col_name}, {os_flag_col_name}, {ob_flag_col_name}")
    return df

def main():
    logger.info(f"Reading Parquet file: {INPUT_FILE_PATH}")
    if not INPUT_FILE_PATH.exists():
        logger.error(f"Input file not found: {INPUT_FILE_PATH}")
        return

    try:
        df = pd.read_parquet(INPUT_FILE_PATH)
        logger.info(f"Successfully loaded DataFrame. Shape: {df.shape}")

        df = add_stochastic_features(df)

        logger.info(f"Writing updated DataFrame to {OUTPUT_FILE_PATH}...")
        df.to_parquet(OUTPUT_FILE_PATH, index=True)
        logger.info("Successfully saved updated DataFrame.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 