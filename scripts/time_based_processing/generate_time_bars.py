import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import logging
import re
import numpy as np
from scipy.stats import skew, kurtosis # For skewness and kurtosis
import gc # Import garbage collector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # Adjusted for subdir minotaur/scripts/time_based_processing
INPUT_PARQUET_DIR = PROJECT_ROOT / 'data' / 'historical_btc_trades_parquet'
SYMBOL = 'BTCUSDT'

TIME_AGGREGATIONS = {
    '1min': '1min',  # Pandas offset alias : Output directory name
    '15min': '15min',
    '4h': '4hour' # Changed H to h
}

OUTPUT_TIME_BAR_BASE_DIR = PROJECT_ROOT / 'data' / 'time_bars'

def get_sorted_input_files(input_dir_path, symbol_str):
    logger.info(f"Searching for input Parquet files in: {input_dir_path}")
    file_pattern = re.compile(rf"^{symbol_str}_(\d{{4}})_(\d{{2}})\.parquet$")
    
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
        
    logger.info(f"Found and sorted {len(matched_files)} input files.")
    return matched_files # Return list of dicts to access name and path

def _calculate_intrabar_features_from_list(ticks_in_bar_list_of_dicts): # Renamed to avoid conflict, helper
    if not ticks_in_bar_list_of_dicts:
        return pd.Series({
            'vwap': np.nan, 'ofi': np.nan, 'price_skew': np.nan, 'price_kurtosis': np.nan,
            'volume_skew': np.nan, 'volume_kurtosis': np.nan, 'avg_trade_size': np.nan
        })
    
    ticks_df = pd.DataFrame(ticks_in_bar_list_of_dicts)
    if ticks_df.empty:
        return pd.Series({
            'vwap': np.nan, 'ofi': np.nan, 'price_skew': np.nan, 'price_kurtosis': np.nan,
            'volume_skew': np.nan, 'volume_kurtosis': np.nan, 'avg_trade_size': np.nan
        })

    price = pd.to_numeric(ticks_df['price'], errors='coerce')
    quantity = pd.to_numeric(ticks_df['quantity'], errors='coerce')
    
    valid_trades = quantity.notna() & price.notna() & (quantity > 0)
    price = price[valid_trades]
    quantity = quantity[valid_trades]

    if quantity.empty or price.empty:
         return pd.Series({
            'vwap': np.nan, 'ofi': np.nan, 'price_skew': np.nan, 'price_kurtosis': np.nan,
            'volume_skew': np.nan, 'volume_kurtosis': np.nan, 'avg_trade_size': np.nan
        })

    vwap = np.sum(price * quantity) / np.sum(quantity) if np.sum(quantity) > 0 else np.nan
    
    ofi = np.nan
    if 'side' in ticks_df.columns: # side should have been derived before calling this
        side = pd.to_numeric(ticks_df['side'], errors='coerce')[valid_trades]
        if not side.empty: # Ensure side is not empty after filtering
            signed_quantity = quantity * side
            ofi = np.sum(signed_quantity.dropna()) / np.sum(quantity) if np.sum(quantity) > 0 else np.nan
        
    price_skew_val = skew(price) if len(price) > 2 else np.nan
    price_kurt_val = kurtosis(price) if len(price) > 3 else np.nan
    volume_skew_val = skew(quantity) if len(quantity) > 2 else np.nan
    volume_kurt_val = kurtosis(quantity) if len(quantity) > 3 else np.nan
    avg_trade_size_val = quantity.mean()

    return pd.Series({
        'vwap': vwap, 'ofi': ofi, 'price_skew': price_skew_val, 'price_kurtosis': price_kurt_val,
        'volume_skew': volume_skew_val, 'volume_kurtosis': volume_kurt_val, 'avg_trade_size': avg_trade_size_val
    })

class TimeBarGenerator:
    def __init__(self, time_freq_alias, current_month_start_dt, current_month_end_dt, output_dir, tf_name_for_file, symbol, current_month_year_str):
        self.time_freq_alias = time_freq_alias
        self.current_month_start_dt = current_month_start_dt
        self.current_month_end_dt = current_month_end_dt # Exclusive end for the month
        self.output_dir = output_dir
        self.tf_name_for_file = tf_name_for_file
        self.symbol = symbol
        self.current_month_year_str = current_month_year_str

        self.ticks_for_current_bar = []
        self.completed_bars_for_month = []
        self.current_bar_start_time = None
        self.current_bar_end_time = None
        try:
            self.period_timedelta = pd.to_timedelta(time_freq_alias)
        except Exception as e:
            logger.error(f"Invalid time_freq_alias '{time_freq_alias}' for TimeBarGenerator: {e}")
            raise # Re-raise to stop initialization

    def _derive_side(self, tick_dict):
        """Helper to ensure 'side' is present in a tick dictionary for intrabar calculations."""
        if 'side' in tick_dict:
            return tick_dict # Side already exists
        
        new_tick_dict = tick_dict.copy()
        if 'isBuyerMaker' in new_tick_dict:
            # Ensure isBuyerMaker is not NaN before boolean evaluation
            if pd.notna(new_tick_dict['isBuyerMaker']):
                new_tick_dict['side'] = 1 if not new_tick_dict['isBuyerMaker'] else -1
            else:
                new_tick_dict['side'] = np.nan # isBuyerMaker is NaN
        else:
            new_tick_dict['side'] = np.nan # No basis to derive side
        return new_tick_dict

    def process_tick(self, tick_dict):
        # tick_dict is expected to have 'timestamp', 'price', 'quantity', 'quote_quantity'
        # and optionally 'isBuyerMaker' or 'side'
        tick_timestamp = tick_dict['timestamp'] # Assuming timestamp is already datetime object

        if self.current_bar_start_time is None:
            self.current_bar_start_time = tick_timestamp.floor(self.time_freq_alias)
            self.current_bar_end_time = self.current_bar_start_time + self.period_timedelta
        
        while tick_timestamp >= self.current_bar_end_time:
            if self.ticks_for_current_bar:
                # Form the bar
                open_price = self.ticks_for_current_bar[0]['price']
                close_price = self.ticks_for_current_bar[-1]['price']
                prices = [t['price'] for t in self.ticks_for_current_bar]
                high_price = max(prices) if prices else np.nan
                low_price = min(prices) if prices else np.nan
                
                total_quantity = sum(t['quantity'] for t in self.ticks_for_current_bar)
                total_quote_quantity = sum(t['quote_quantity'] for t in self.ticks_for_current_bar)
                num_ticks = len(self.ticks_for_current_bar)

                # Prepare ticks with 'side' for intrabar calculation
                prepared_ticks_for_calc = [self._derive_side(t) for t in self.ticks_for_current_bar]
                intrabar_stats = _calculate_intrabar_features_from_list(prepared_ticks_for_calc)
                
                bar_data = {
                    'open_timestamp': self.current_bar_start_time,
                    'open': open_price, 'high': high_price, 'low': low_price, 'close': close_price,
                    'close_timestamp': self.current_bar_end_time - pd.Timedelta(nanoseconds=1),
                    'volume': total_quantity, 'dollar_volume': total_quote_quantity, 'num_ticks': num_ticks,
                    **intrabar_stats
                }
                
                # Only add if bar's open_timestamp is within the target month
                if self.current_bar_start_time >= self.current_month_start_dt and \
                   self.current_bar_start_time < self.current_month_end_dt:
                    self.completed_bars_for_month.append(bar_data)
            
            self.ticks_for_current_bar = [] # Reset for next bar
            self.current_bar_start_time = self.current_bar_end_time
            self.current_bar_end_time = self.current_bar_start_time + self.period_timedelta

        # Add current tick to the (potentially new) bar if its timestamp is within the current bar window
        if tick_timestamp >= self.current_bar_start_time: # Ensure tick is not from before the bar started (e.g. carry-over alignment)
             self.ticks_for_current_bar.append(tick_dict)


    def finalize_month_processing(self):
        # Save completed bars for the month
        if self.completed_bars_for_month:
            bars_df = pd.DataFrame(self.completed_bars_for_month)
            if not bars_df.empty:
                output_file = self.output_dir / f"{self.symbol}_{self.current_month_year_str}_time_bars_{self.tf_name_for_file}.parquet"
                try:
                    bars_df.to_parquet(output_file, index=False)
                    logger.info(f"Saved {len(bars_df)} {self.time_freq_alias} bars for {self.current_month_year_str} to {output_file.name}")
                except Exception as e:
                    logger.error(f"Error saving {self.time_freq_alias} bars to parquet {output_file.name}: {e}")
        else:
            logger.info(f"No completed bars for {self.time_freq_alias} in {self.current_month_year_str} to save.")

        # Return current ticks_for_current_bar as the next carry-over (list of dicts)
        carry_over_list = self.ticks_for_current_bar
        logger.debug(f"Finalizing month for {self.time_freq_alias}. Carry-over size: {len(carry_over_list)} ticks.")
        
        # Reset state for potential reuse (though typically a new generator is created per month/tf)
        self.ticks_for_current_bar = []
        self.completed_bars_for_month = []
        self.current_bar_start_time = None
        self.current_bar_end_time = None
        
        return carry_over_list

def main():
    logger.info(f"Starting Time Bar Generation for intervals: {list(TIME_AGGREGATIONS.keys())}")

    OUTPUT_TIME_BAR_BASE_DIR.mkdir(parents=True, exist_ok=True)
    for tf_dir_name in TIME_AGGREGATIONS.values(): # tf_dir_name is '1min', '15min', '4hour'
        (OUTPUT_TIME_BAR_BASE_DIR / tf_dir_name).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory for {tf_dir_name} bars: {OUTPUT_TIME_BAR_BASE_DIR / tf_dir_name}")

    all_input_file_metas = get_sorted_input_files(INPUT_PARQUET_DIR, SYMBOL)
    if not all_input_file_metas:
        logger.error("No input tick data files found. Aborting.")
        return

    # resume_from_filename = "BTCUSDT_2023_02.parquet" # Comment out or set to None for full run
    resume_from_filename = None 
    start_processing_idx = 0
    if resume_from_filename:
        try:
            start_processing_idx = next(i for i, f_meta in enumerate(all_input_file_metas) if f_meta['name'] == resume_from_filename)
            logger.info(f"Attempting to resume from specified file: {resume_from_filename} at index {start_processing_idx}")
        except StopIteration:
            logger.error(f"Resume file {resume_from_filename} not found. Starting from the beginning (index 0).")
            start_processing_idx = 0
    
    files_to_process_meta = all_input_file_metas[start_processing_idx:]
    if start_processing_idx > 0 and files_to_process_meta: 
        logger.info(f"Resuming process from file: {files_to_process_meta[0]['name']}. Total files to process: {len(files_to_process_meta)}")
    elif start_processing_idx > 0 and not files_to_process_meta:
        logger.info(f"Resume index {start_processing_idx} is beyond the number of available files. No files to process.")
        return
    
    SMOKE_TEST = False # Set to False for a full run
    SMOKE_TEST_FILE_LIMIT = 1
    if SMOKE_TEST: 
        files_to_process_meta = files_to_process_meta[:SMOKE_TEST_FILE_LIMIT]
        logger.info(f"--- SMOKE TEST MODE ENABLED --- Processing: {[f['name'] for f in files_to_process_meta]}")

    # Store carry-over as list of dicts
    carry_over_tick_dicts_by_freq = {tf_alias: [] for tf_alias in TIME_AGGREGATIONS.keys()}
    
    ideal_columns_to_read = [
        'timestamp', 'trade_time', 'time',
        'price', 'quantity', 'quote_quantity',
        'isBuyerMaker'
    ]
    # Define dict_cols_for_iteration here, at the correct scope within main(), before the main file loop.
    dict_cols_for_iteration = ['timestamp', 'price', 'quantity', 'quote_quantity', 'side'] 

    for i, file_meta in enumerate(files_to_process_meta):
        file_path = file_meta['path']
        current_month_year_str = f"{file_meta['year']}_{file_meta['month']:02d}"
        logger.info(f"Processing file {i+1+start_processing_idx}/{len(all_input_file_metas)}: {file_path.name} ({current_month_year_str})")

        raw_current_month_ticks_df = None
        raw_current_month_ticks_list = [] # Initialize as empty list

        try:
            parquet_file_schema = pq.read_schema(file_path)
            available_columns_in_file = parquet_file_schema.names
            logger.debug(f"Available columns in {file_path.name}: {available_columns_in_file}")

            columns_to_actually_read = [col for col in ideal_columns_to_read if col in available_columns_in_file]
            
            if not columns_to_actually_read:
                 logger.error(f"No usable columns found in {file_path.name} from the ideal set. Skipping file.")
                 continue
            if not any(c in columns_to_actually_read for c in ['price', 'quantity', 'quote_quantity']):
                 logger.error(f"Core data columns (price, quantity, quote_quantity) not found in {file_path.name}. Skipping file.")
                 continue
            if not any(c in columns_to_actually_read for c in ['timestamp', 'trade_time', 'time']):
                logger.error(f"No potential timestamp column found in {file_path.name}. Skipping file.")
                continue

            logger.info(f"Reading existing columns from {file_path.name}: {columns_to_actually_read}")
            # Use pyarrow.parquet.read_table for potentially better memory on selective read
            table = pq.read_table(file_path, columns=columns_to_actually_read)
            raw_current_month_ticks_df = table.to_pandas(split_blocks=True, self_destruct=True)
            del table # Release Arrow table memory
            gc.collect()

            # --- Start: DataFrame Preprocessing (mimicking phase2_generate_dollar_bars) ---
            # 1. Standardize timestamp column name
            if 'timestamp' not in raw_current_month_ticks_df.columns:
                if 'trade_time' in raw_current_month_ticks_df.columns:
                    raw_current_month_ticks_df.rename(columns={'trade_time': 'timestamp'}, inplace=True)
                elif 'time' in raw_current_month_ticks_df.columns:
                    raw_current_month_ticks_df.rename(columns={'time': 'timestamp'}, inplace=True)
                else:
                    logger.error(f"Logic error: No standardized timestamp column after read for {file_path.name}. Should have been caught. Skipping.")
                    continue 
            
            # 2. Convert timestamp to datetime
            raw_current_month_ticks_df['timestamp'] = pd.to_datetime(raw_current_month_ticks_df['timestamp'])

            # 3. Convert core numeric columns and handle NaNs
            for col in ['price', 'quantity', 'quote_quantity']:
                raw_current_month_ticks_df[col] = pd.to_numeric(raw_current_month_ticks_df[col], errors='coerce')
            raw_current_month_ticks_df.dropna(subset=['price', 'quantity', 'quote_quantity'], inplace=True)

            if raw_current_month_ticks_df.empty:
                logger.warning(f"DataFrame empty after preprocessing for {file_path.name}. Skipping.")
                # Ensure df is deleted if we skip here
                if raw_current_month_ticks_df is not None: del raw_current_month_ticks_df; raw_current_month_ticks_df = None
                gc.collect()
                continue
            
            # 4. Derive 'side' column
            if 'isBuyerMaker' in raw_current_month_ticks_df.columns and raw_current_month_ticks_df['isBuyerMaker'].notna().any():
                raw_current_month_ticks_df['side'] = np.where(raw_current_month_ticks_df['isBuyerMaker'] == False, 1, -1)
            else:
                raw_current_month_ticks_df['side'] = np.nan # Ensure 'side' column exists even if isBuyerMaker is missing/all NaN
            
            # 5. Sort DataFrame by timestamp (ONCE per month)
            raw_current_month_ticks_df.sort_values(by='timestamp', inplace=True)
            # --- End: DataFrame Preprocessing ---
            # No conversion to list of dicts here; raw_current_month_ticks_df is used directly with itertuples

        except Exception as e:
            logger.error(f"Error reading or preprocessing Parquet file {file_path.name}: {e}. Skipping file.")
            if raw_current_month_ticks_df is not None: del raw_current_month_ticks_df; raw_current_month_ticks_df = None
            gc.collect()
            continue
            
        # Explicit check for None before entering the time frequency loop
        if raw_current_month_ticks_df is None:
            logger.warning(f"Skipping time frequency processing for {file_path.name} as raw_current_month_ticks_df is None (due to error in try-except block).")
            continue # Skip to the next file in the outer loop
        
        current_period = pd.Period(year=file_meta['year'], month=file_meta['month'], freq='M')
        current_month_start_dt = current_period.start_time
        current_month_end_dt = current_period.end_time + pd.Timedelta(nanoseconds=1)

        for tf_alias, tf_dir_name in TIME_AGGREGATIONS.items():
            logger.info(f"  Generating {tf_alias} bars for {current_month_year_str}...")
            output_dir_for_res = OUTPUT_TIME_BAR_BASE_DIR / tf_dir_name
            try:
                time_bar_generator = TimeBarGenerator(
                    tf_alias, current_month_start_dt, current_month_end_dt, 
                    output_dir_for_res, tf_dir_name, SYMBOL, current_month_year_str
                )
            except Exception as e_gen:
                logger.error(f"Failed to initialize TimeBarGenerator for {tf_alias}: {e_gen}. Skipping this frequency.")
                continue

            # Process carry-over ticks (already list of dicts)
            # These should be processed first as they are chronologically earlier than current month's main ticks
            if carry_over_tick_dicts_by_freq[tf_alias]:
                logger.debug(f"Processing {len(carry_over_tick_dicts_by_freq[tf_alias])} carry-over ticks for {tf_alias}")
                # Carry-over ticks are already sorted from their previous generation
                # However, it might be safer to sort them again if there's any doubt
                # sorted_carry_over = sorted(carry_over_tick_dicts_by_freq[tf_alias], key=lambda x: x['timestamp'])
                for tick_dict in carry_over_tick_dicts_by_freq[tf_alias]: # Assuming they are already appropriately sorted
                    time_bar_generator.process_tick(tick_dict)
            
            # Process current month's ticks using itertuples()
            logger.debug(f"Processing {len(raw_current_month_ticks_df) if raw_current_month_ticks_df is not None else 0} main ticks for {tf_alias} from {file_path.name}")
            processed_main_ticks_count = 0
            
            if raw_current_month_ticks_df is not None and not raw_current_month_ticks_df.empty:
                # Use dict_cols_for_iteration which is defined in the correct scope
                missing_dict_cols = [col for col in dict_cols_for_iteration if col not in raw_current_month_ticks_df.columns]
                if missing_dict_cols:
                    logger.error(f"Critical columns for tick dictionary {missing_dict_cols} missing in raw_current_month_ticks_df. Skipping main tick processing for {tf_alias}.")
                else:
                    # Create df_for_iteration only with the necessary columns
                    df_for_iteration = raw_current_month_ticks_df[dict_cols_for_iteration].copy()
                    for row_tuple in df_for_iteration.itertuples(index=False):
                        tick_dict = row_tuple._asdict() 
                        time_bar_generator.process_tick(tick_dict)
                        processed_main_ticks_count += 1
                    
                    del df_for_iteration 
                    gc.collect()

            logger.debug(f"Processed {processed_main_ticks_count} main ticks for {tf_alias}.")

            next_carry_over_list = time_bar_generator.finalize_month_processing()
            carry_over_tick_dicts_by_freq[tf_alias] = next_carry_over_list
            
            del time_bar_generator
            gc.collect()
        
        # Delete the main monthly DataFrame after all frequencies are processed
        if raw_current_month_ticks_df is not None:
            del raw_current_month_ticks_df
            raw_current_month_ticks_df = None
        gc.collect()

    logger.info("Finished processing all files.")
    for tf_alias, final_carry_list in carry_over_tick_dicts_by_freq.items():
        if final_carry_list:
            logger.info(f"Final carry-over for {tf_alias}: {len(final_carry_list)} ticks. (These were not processed into bars)")

if __name__ == "__main__":
    main() 