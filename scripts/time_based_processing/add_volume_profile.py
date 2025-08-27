import pandas as pd
from pathlib import Path
import logging
import argparse
import glob
from volprofile import getVPWithOHLC # Ensure volprofile is installed
import numpy as np # Make sure numpy is imported

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Volume Profile Feature Configuration (consistent with calculate_time_bar_features.py)
VOLUME_PROFILE_CONFIG = {
    'enabled': True,
    'window_sizes': [50, 100],  # Lookback periods for VP calculation
    'n_bins': [50, 100]         # Number of bins for profile histogram
}

def _calculate_vp_metrics_for_window(window_df, n_bins_vp, tick_size=0.01):
    """
    Helper to calculate POC, VAH, VAL for a given window of data (which is a DataFrame).
    """
    # window_df is expected to be a DataFrame with columns: open, high, low, close, volume
    if not isinstance(window_df, pd.DataFrame):
        logger.warning(f"VP calc: received type {type(window_df)} instead of DataFrame for window. Skipping.")
        return pd.Series({'poc': np.nan, 'val': np.nan, 'vah': np.nan})

    if window_df.empty or len(window_df) < 2: 
        logger.debug(f"VP calc: Window is empty or too short (len: {len(window_df)}). Returning NaNs.")
        return pd.Series({'poc': np.nan, 'val': np.nan, 'vah': np.nan})

    # Columns are already 'open', 'high', 'low', 'close', 'volume' from temp_df_for_vp_calc
    # No need to re-check/rename here if temp_df_for_vp_calc is correctly prepared.
    
    try:
        vp_df = getVPWithOHLC(window_df, nBins=n_bins_vp)

        if vp_df.empty:
            logger.debug("VP calc: getVPWithOHLC returned empty DataFrame. Returning NaNs.")
            return pd.Series({'poc': np.nan, 'val': np.nan, 'vah': np.nan})

        poc_price = vp_df.loc[vp_df['aggregateVolume'].idxmax(), 'minPrice'] + (vp_df['maxPrice'].iloc[0] - vp_df['minPrice'].iloc[0]) / 2
        
        total_volume = vp_df['aggregateVolume'].sum()
        value_area_volume_target = total_volume * 0.70
        
        profile_sorted_by_volume = vp_df.sort_values(by='aggregateVolume', ascending=False)
        
        cumulative_volume = 0
        value_area_prices = []
        for _, row in profile_sorted_by_volume.iterrows():
            cumulative_volume += row['aggregateVolume']
            value_area_prices.append(row['minPrice'])
            value_area_prices.append(row['maxPrice'])
            if cumulative_volume >= value_area_volume_target:
                break
        
        val_price = min(value_area_prices) if value_area_prices else np.nan
        vah_price = max(value_area_prices) if value_area_prices else np.nan
        
        return pd.Series({'poc': poc_price, 'val': val_price, 'vah': vah_price})

    except Exception as e:
        logger.warning(f"VP calc: Error in _calculate_vp_metrics_for_window: {e}. Window head:\n{window_df.head(2)}")
        return pd.Series({'poc': np.nan, 'val': np.nan, 'vah': np.nan})


def add_volume_profile_features(df: pd.DataFrame, resolution_suffix: str):
    """
    Adds Volume Profile features (POC, VAH, VAL) to the DataFrame.
    resolution_suffix is like 'tb_1min', 'tb_15min', etc.
    """
    if not VOLUME_PROFILE_CONFIG['enabled']:
        logger.info("Volume Profile features are disabled in the config.")
        return df

    logger.info(f"Calculating Volume Profile features for resolution suffix: {resolution_suffix}...")
    
    base_ohlcv_cols_map = {
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    
    missing_base_cols = [orig_col for orig_col in base_ohlcv_cols_map.keys() if orig_col not in df.columns]
    if missing_base_cols:
        logger.error(f"VP calc: Missing base OHLCV columns in DataFrame for suffix {resolution_suffix}: {missing_base_cols}. Skipping VP calculation.")
        return df

    temp_df_for_vp_calc = df[list(base_ohlcv_cols_map.keys())].rename(columns=base_ohlcv_cols_map)

    for window in VOLUME_PROFILE_CONFIG['window_sizes']:
        for n_bins_vp in VOLUME_PROFILE_CONFIG['n_bins']:
            logger.info(f"  Calculating VP for window={window}, n_bins={n_bins_vp}...")
            
            poc_col_name = f'vp_poc_{window}w_{n_bins_vp}b_{resolution_suffix}'
            val_col_name = f'vp_val_{window}w_{n_bins_vp}b_{resolution_suffix}'
            vah_col_name = f'vp_vah_{window}w_{n_bins_vp}b_{resolution_suffix}'
            
            # Manual rolling loop
            results_poc = [np.nan] * len(temp_df_for_vp_calc)
            results_val = [np.nan] * len(temp_df_for_vp_calc)
            results_vah = [np.nan] * len(temp_df_for_vp_calc)

            for i in range(len(temp_df_for_vp_calc)):
                min_data_for_calc = max(2, int(window * 0.5)) # Define min_periods for this window
                
                if i < window -1: # Not enough data for a full window yet
                    current_window_size = i + 1
                    if current_window_size >= min_data_for_calc:
                        window_data = temp_df_for_vp_calc.iloc[0:i+1] # Expanding window up to min_periods
                    else:
                        logger.debug(f"idx {i}: Not enough data for expanding window (size {current_window_size} < {min_data_for_calc}). Skipping.")
                        continue # Skip if not enough for min_periods
                else:
                     window_data = temp_df_for_vp_calc.iloc[i - window + 1 : i + 1]
                
                # Check against min_periods again for the slice, to be absolutely sure
                # This check might be redundant if the logic above is correct, but safe
                if len(window_data) < min_data_for_calc:
                    logger.debug(f"idx {i}: Final check, window_data too short (len {len(window_data)} < {min_data_for_calc}). Skipping.")
                    continue

                # Added logging for window_data
                logger.debug(f"idx {i}: Processing window_data (shape {window_data.shape}):\n{window_data.head(2)}")
                
                metrics = _calculate_vp_metrics_for_window(window_data, n_bins_vp=n_bins_vp, tick_size=0.01)
                
                # Added logging for metrics
                if i % (len(temp_df_for_vp_calc) // 1000 + 1) == 0: # Log roughly 1000 times or so
                    if pd.notna(metrics['poc']) or pd.notna(metrics['val']) or pd.notna(metrics['vah']):
                         logger.debug(f"idx {i}: Metrics for window={window}, n_bins={n_bins_vp}: POC={metrics['poc']} VAL={metrics['val']} VAH={metrics['vah']}")
                    else:
                         logger.debug(f"idx {i}: Metrics for window={window}, n_bins={n_bins_vp}: All NaN")
                
                results_poc[i] = metrics['poc']
                results_val[i] = metrics['val']
                results_vah[i] = metrics['vah']

            df[poc_col_name] = results_poc
            df[val_col_name] = results_val
            df[vah_col_name] = results_vah
            logger.info(f"  Finished VP for window={window}, n_bins={n_bins_vp}. NaNs: POC={df[poc_col_name].isnull().sum()}, VAL={df[val_col_name].isnull().sum()}, VAH={df[vah_col_name].isnull().sum()}")

    logger.info(f"Finished all Volume Profile calculations for {resolution_suffix}.")
    return df


def main():
    parser = argparse.ArgumentParser(description="Add Volume Profile features to existing time-bar feature Parquet files.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input Parquet files (e.g., minotaur/data/time_bars_features_v2/1min/).")
    parser.add_argument("--base-output-dir", type=str, default="minotaur/data/", help="Base directory for output (e.g., minotaur/data/).")
    parser.add_argument("--output-suffix", type=str, default="time_bars_features_with_vp_debug", help="Suffix for the output subdirectory (e.g., 'time_bars_features_with_vp_v2_test').")
    parser.add_argument("--resolution", type=str, required=True, choices=['1min', '5min', '15min', '4hour'], help="Time resolution of the bars (e.g., '1min', '15min').")
    
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_resolution_dir = Path(args.base_output_dir) / args.output_suffix / args.resolution
    
    resolution = args.resolution

    if not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    output_resolution_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be saved to: {output_resolution_dir}")

    resolution_suffix_map = {
        "1min": "tb_1min",
        "5min": "tb_5min", 
        "15min": "tb_15min",
        "4hour": "tb_4hour"
    }
    if resolution not in resolution_suffix_map:
        logger.error(f"Unsupported resolution: {resolution}. Supported: {list(resolution_suffix_map.keys())}")
        return
    
    res_suffix_for_cols = resolution_suffix_map[resolution]

    search_pattern = str(input_dir / f"*_time_bars_features_{resolution}*_v2.parquet") 
    
    input_files = sorted(glob.glob(search_pattern))

    if not input_files:
        logger.warning(f"No files found matching pattern '{search_pattern}' in {input_dir} for resolution {resolution}")
        return

    logger.info(f"Found {len(input_files)} files to process in {input_dir} for resolution {resolution}.")

    for i, file_path_str in enumerate(input_files):
        file_path = Path(file_path_str)
        logger.info(f"Processing file {i+1}/{len(input_files)}: {file_path.name}")
        
        try:
            df = pd.read_parquet(file_path)
            df_with_vp = add_volume_profile_features(df.copy(), res_suffix_for_cols) 
            
            output_file_path = output_resolution_dir / file_path.name # Save with original name in new dir
            df_with_vp.to_parquet(output_file_path, index=True)
            logger.info(f"Successfully processed and saved: {output_file_path}")

        except Exception as e:
            logger.error(f"Failed to process file {file_path.name}: {e}", exc_info=True)

    logger.info("All files processed.")

if __name__ == "__main__":
    # Set debug level for more verbose output if needed
    logger.setLevel(logging.DEBUG) 
    main() 