import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import logging
import re # For sorting files
import numpy as np # Added for numerical operations
from scipy import stats # Added for skewness and kurtosis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PARQUET_DIR = PROJECT_ROOT / 'data' / 'historical_btc_trades_parquet'
SYMBOL = 'BTCUSDT'

# Dollar Bar Configuration (User will choose one to start, e.g., 2M)
DOLLAR_THRESHOLD_VALUE = 2_000_000 
DOLLAR_THRESHOLD_STR = "2M" # For naming output directory and files

OUTPUT_DOLLAR_BAR_BASE_DIR = PROJECT_ROOT / 'data' / 'dollar_bars'
OUTPUT_DOLLAR_BAR_DIR = OUTPUT_DOLLAR_BAR_BASE_DIR / DOLLAR_THRESHOLD_STR


class DollarBarGenerator:
    def __init__(self, dollar_threshold):
        self.threshold = float(dollar_threshold) # Ensure it's float for comparisons
        self.current_bar_trades = [] # Stores dicts of {'timestamp', 'price', 'quantity', 'quote_quantity', 'isBuyerMaker'}
        self.current_bar_cum_dollar_val = 0.0
        self._generated_bars_accumulator = [] # Temp storage before converting to DataFrame

    def process_tick(self, timestamp, price, quantity, quote_quantity, is_buyer_maker): # Added is_buyer_maker
        self.current_bar_trades.append({
            'timestamp': timestamp, 
            'price': float(price), 
            'quantity': float(quantity), 
            'quote_quantity': float(quote_quantity),
            'isBuyerMaker': is_buyer_maker # Store this
        })
        self.current_bar_cum_dollar_val += float(quote_quantity)

        if self.current_bar_cum_dollar_val >= self.threshold:
            self._form_bar()
            # Reset for the next bar (current_bar_trades and current_bar_cum_dollar_val are reset in _form_bar or after its call)
            # No, _form_bar uses them, so reset should be here or after the call in process_tick.
            # Let's ensure reset happens cleanly *after* _form_bar has used the data.
            # The original logic seems to do this correctly by re-initializing them.

    def _form_bar(self):
        if not self.current_bar_trades:
            self.current_bar_trades = [] # Ensure it's reset even if no bar formed
            self.current_bar_cum_dollar_val = 0.0
            return

        # --- Standard OHLCV and basic aggregations ---
        prices_in_bar = [float(t['price']) for t in self.current_bar_trades]
        quantities_in_bar = [float(t['quantity']) for t in self.current_bar_trades]
        quote_quantities_in_bar = [float(t['quote_quantity']) for t in self.current_bar_trades]
        # is_buyer_maker_flags = [t['isBuyerMaker'] for t in self.current_bar_trades] # Corrected key

        open_price = self.current_bar_trades[0]['price']
        open_timestamp = self.current_bar_trades[0]['timestamp']
        close_price = self.current_bar_trades[-1]['price']
        close_timestamp = self.current_bar_trades[-1]['timestamp']
        
        high_price = max(prices_in_bar) if prices_in_bar else np.nan
        low_price = min(prices_in_bar) if prices_in_bar else np.nan
        
        total_quantity = sum(quantities_in_bar)
        actual_dollar_volume_for_bar = sum(quote_quantities_in_bar) # This is self.current_bar_cum_dollar_val
        num_ticks_in_bar = len(self.current_bar_trades)

        # --- New Intra-bar Aggregate Features ---
        prices_in_bar_np = np.array([]) # Initialize to empty array
        trade_imbalance = 0.0
        intra_bar_tick_price_volatility = 0.0
        taker_buy_sell_ratio = 0.0 # Initialize to 0 to handle cases with no taker sells
        tick_price_skewness = 0.0
        tick_price_kurtosis = 0.0
        num_price_changes = 0
        num_directional_changes = 0

        if num_ticks_in_bar > 0:
            prices_in_bar_np = np.array(prices_in_bar) # Moved definition here

            # Trade Imbalance & Taker Buy/Sell Ratio
            buy_volume = sum(t['quantity'] for t in self.current_bar_trades if not t['isBuyerMaker']) # isBuyerMaker=False means buyer is taker
            sell_volume = sum(t['quantity'] for t in self.current_bar_trades if t['isBuyerMaker'])  # isBuyerMaker=True means seller is taker
            
            total_taker_volume = buy_volume + sell_volume
            if total_taker_volume > 0:
                trade_imbalance = (buy_volume - sell_volume) / total_taker_volume
            
            if sell_volume > 0:
                taker_buy_sell_ratio = buy_volume / sell_volume
            elif buy_volume > 0: # sell_volume is 0, but buy_volume is not
                taker_buy_sell_ratio = np.inf 
            # else both are 0, ratio remains np.nan

            # Volatility of Tick Prices within the Bar
            if len(prices_in_bar) > 1: # prices_in_bar_np is already defined if num_ticks_in_bar > 0
                log_returns = np.log(prices_in_bar_np[1:] / prices_in_bar_np[:-1])
                if len(log_returns) > 0:
                    intra_bar_tick_price_volatility = np.std(log_returns)
            
            # Skewness and Kurtosis - check for near-constant prices
            # Define a small epsilon for checking near-zero standard deviation
            epsilon = 1e-9  # A small number to define "close to zero"
            # Ensure prices_in_bar_np has data points to calculate std, though if num_ticks_in_bar = 1, std is 0 or nan.
            # The check num_ticks_in_bar > 0 already ensures prices_in_bar_np exists.
            # If num_ticks_in_bar is 1, std() might be 0 or NaN depending on numpy version & ddof.
            # We want to set skew/kurt to 0 if there's effectively no variation.
            if num_ticks_in_bar <= 1 or prices_in_bar_np.std(ddof=0) < epsilon: # Use ddof=0 for population std for this check
                tick_price_skewness = 0.0
                tick_price_kurtosis = 0.0
            else:
                tick_price_skewness = stats.skew(prices_in_bar_np)
                tick_price_kurtosis = stats.kurtosis(prices_in_bar_np) # Fisher kurtosis (normal is 0)

            # Number of Price Changes / Directional Changes
            if len(prices_in_bar) > 1:
                price_diffs = np.diff(np.array(prices_in_bar))
                num_price_changes = np.count_nonzero(price_diffs)
                
                # For directional changes, consider sign of differences
                # A change occurs if sign(diff[i]) != sign(diff[i-1]) and both are non-zero
                signed_diffs = np.sign(price_diffs[price_diffs != 0]) # Get signs of non-zero diffs
                if len(signed_diffs) > 1:
                    directional_changes_arr = np.diff(signed_diffs)
                    num_directional_changes = np.count_nonzero(directional_changes_arr)

        self._generated_bars_accumulator.append({
            'open_timestamp': open_timestamp,
            'open': open_price, 
            'high': high_price, 
            'low': low_price, 
            'close': close_price,
            'close_timestamp': close_timestamp,
            'volume': total_quantity, 
            'dollar_volume': actual_dollar_volume_for_bar,
            'num_ticks': num_ticks_in_bar,
            # New features
            'trade_imbalance': trade_imbalance,
            'intra_bar_tick_price_volatility': intra_bar_tick_price_volatility,
            'taker_buy_sell_ratio': taker_buy_sell_ratio,
            'tick_price_skewness': tick_price_skewness,
            'tick_price_kurtosis': tick_price_kurtosis,
            'num_price_changes': num_price_changes,
            'num_directional_changes': num_directional_changes
        })
        
        # Reset for the next bar after current bar's data has been used
        self.current_bar_trades = []
        self.current_bar_cum_dollar_val = 0.0

    def flush_remaining_ticks_as_bar(self):
        if self.current_bar_trades and self.current_bar_cum_dollar_val > 0:
            logger.info(f"Flushing remaining {len(self.current_bar_trades)} ticks into a final partial bar (dollar volume: {self.current_bar_cum_dollar_val:.2f}).")
            self._form_bar()
            self.current_bar_trades = []
            self.current_bar_cum_dollar_val = 0.0

    def get_all_bars(self):
        if not self._generated_bars_accumulator:
            return pd.DataFrame() # Return empty DataFrame if no bars
            
        bars_df = pd.DataFrame(self._generated_bars_accumulator)
        self._generated_bars_accumulator = [] 
        
        bars_df['open_timestamp'] = pd.to_datetime(bars_df['open_timestamp'])
        bars_df['close_timestamp'] = pd.to_datetime(bars_df['close_timestamp'])
        bars_df['open'] = bars_df['open'].astype(float)
        bars_df['high'] = bars_df['high'].astype(float)
        bars_df['low'] = bars_df['low'].astype(float)
        bars_df['close'] = bars_df['close'].astype(float)
        bars_df['volume'] = bars_df['volume'].astype(float)
        bars_df['dollar_volume'] = bars_df['dollar_volume'].astype(float)
        bars_df['num_ticks'] = bars_df['num_ticks'].astype(int)
        # Add dtypes for new features
        new_float_cols = [
            'trade_imbalance', 'intra_bar_tick_price_volatility', 
            'taker_buy_sell_ratio', 'tick_price_skewness', 'tick_price_kurtosis'
        ]
        new_int_cols = ['num_price_changes', 'num_directional_changes']
        for col in new_float_cols:
            if col in bars_df.columns: bars_df[col] = bars_df[col].astype(float)
        for col in new_int_cols:
            if col in bars_df.columns: bars_df[col] = bars_df[col].astype(int)
            
        return bars_df

    def reset(self):
        self.current_bar_trades = []
        self.current_bar_cum_dollar_val = 0.0
        self._generated_bars_accumulator = []

def get_sorted_input_files(input_dir_path, symbol_str):
    logger.info(f"Searching for input Parquet files in: {input_dir_path}")
    
    # Regex to match and capture year and month for sorting
    # e.g. BTCUSDT_2023_05.parquet
    file_pattern = re.compile(rf"^{symbol_str}_(\d{{4}})_(\d{{2}})\.parquet$")
    
    matched_files = []
    for f_path in input_dir_path.iterdir():
        if f_path.is_file():
            match = file_pattern.match(f_path.name)
            if match:
                year = int(match.group(1))
                month = int(match.group(2))
                matched_files.append({'path': f_path, 'year': year, 'month': month, 'name': f_path.name})
    
    # Sort files chronologically by year, then month
    matched_files.sort(key=lambda x: (x['year'], x['month']))
    
    if not matched_files:
        logger.warning(f"No input files found matching pattern in {input_dir_path}")
        return []
        
    logger.info(f"Found and sorted {len(matched_files)} input files.")
    return [mf['path'] for mf in matched_files]


def main():
    logger.info(f"Starting Phase 2: Dollar Bar Generation ({DOLLAR_THRESHOLD_STR} threshold: ${DOLLAR_THRESHOLD_VALUE:,.0f})")
    
    SMOKE_TEST = False  # Set to False to run on all files
    SMOKE_TEST_FILE_LIMIT = 2 # Number of files to process for smoke test

    # Create output directory if it doesn't exist
    OUTPUT_DOLLAR_BAR_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory for dollar bars: {OUTPUT_DOLLAR_BAR_DIR}")

    logger.info(f"Searching for input Parquet files in: {INPUT_PARQUET_DIR}")
    input_files = get_sorted_input_files(INPUT_PARQUET_DIR, SYMBOL)
    if not input_files:
        logger.error("No input tick data files found. Aborting.")
        return
    logger.info(f"Found and sorted {len(input_files)} input files.")

    if SMOKE_TEST:
        logger.warning(f"--- SMOKE TEST MODE ENABLED: Processing only the first {SMOKE_TEST_FILE_LIMIT} files. ---")
        input_files = input_files[:SMOKE_TEST_FILE_LIMIT]
        if not input_files: # Handle case where there are fewer files than SMOKE_TEST_FILE_LIMIT
             logger.warning("Smoke test enabled, but not enough files to meet SMOKE_TEST_FILE_LIMIT. Processing all available files.")
        logger.info(f"Smoke test will process: {[f.name for f in input_files]}")


    generator = DollarBarGenerator(dollar_threshold=DOLLAR_THRESHOLD_VALUE)
    
    total_ticks_processed_overall = 0
    total_bars_generated_overall = 0
    processed_files_count = 0

    for i, file_path in enumerate(input_files):
        logger.info(f"Processing input file {i+1}/{len(input_files)}: {file_path.name}")
        try:
            # Read necessary columns, including quote_quantity AND isBuyerMaker
            # Assuming 'isBuyerMaker' is the column name in your Parquet files. Adjust if different.
            required_columns = ['timestamp', 'price', 'quantity', 'quote_quantity', 'isBuyerMaker']
            
            # Check available columns in the parquet file first to avoid error if 'isBuyerMaker' is missing
            try:
                parquet_file_schema = pq.read_schema(file_path)
                available_columns = [name for name in parquet_file_schema.names]
            except Exception as e_schema:
                logger.warning(f"Could not read schema for {file_path.name}. Proceeding with default required columns. Error: {e_schema}")
                available_columns = required_columns # Assume they are there if schema read fails

            columns_to_read = [col for col in required_columns if col in available_columns]
            missing_cols = set(required_columns) - set(columns_to_read)
            if 'isBuyerMaker' in missing_cols:
                logger.warning(f"'isBuyerMaker' column not found in {file_path.name}. Trade imbalance features will be NaN.")
                # Add a dummy isBuyerMaker column if missing, so the rest of the code doesn't break
                # This allows processing files that might not have this specific column, imbalance features will be NaN.
                # No, this is bad. The generator expects it. Better to pass None or a flag.

            if not all(col in columns_to_read for col in ['timestamp', 'price', 'quantity', 'quote_quantity']):
                 logger.error(f"Core columns missing in {file_path.name} ({missing_cols}). Skipping file.")
                 continue

            df_ticks = pq.read_table(file_path, columns=columns_to_read).to_pandas()
            
            # If 'isBuyerMaker' was not available and thus not read, add it as a column of Nones or a default (e.g. False)
            # This ensures the itertuples call below doesn't fail.
            if 'isBuyerMaker' not in df_ticks.columns:
                df_ticks['isBuyerMaker'] = None # Or False, or some other indicator of missing data
                logger.warning(f"'isBuyerMaker' column was not in {file_path.name}, added as None. Imbalance features will be NaN.")


            df_ticks['timestamp'] = pd.to_datetime(df_ticks['timestamp'], unit='ms')
            df_ticks.sort_values(by='timestamp', inplace=True) # Ensure sorted by timestamp

            if df_ticks.empty:
                logger.info(f"Input file {file_path.name} is empty or contains no usable data. Skipping.")
                continue
            
            ticks_in_file = len(df_ticks)
            # Use itertuples for efficiency and ensure all fields are passed
            for tick_row in df_ticks.itertuples(index=False):
                # Get is_buyer_maker, defaulting to None if the column somehow still wasn't there (should be handled above)
                is_buyer_maker_val = getattr(tick_row, 'isBuyerMaker', None) 
                generator.process_tick(
                    timestamp=tick_row.timestamp,
                    price=tick_row.price,
                    quantity=tick_row.quantity,
                    quote_quantity=tick_row.quote_quantity,
                    is_buyer_maker=is_buyer_maker_val # Pass it here
                )
            
            logger.info(f"Finished reading {ticks_in_file} ticks from {file_path.name}.")
            total_ticks_processed_overall += ticks_in_file

            # Flush any remaining ticks for this specific file into a final bar
            generator.flush_remaining_ticks_as_bar()
            
            # Get all bars generated from this file
            current_file_bars_df = generator.get_all_bars()

            if not current_file_bars_df.empty:
                output_filename_stem = file_path.stem
                output_bar_file = OUTPUT_DOLLAR_BAR_DIR / f"{output_filename_stem}_dollar_bars_{DOLLAR_THRESHOLD_STR}.parquet"
                
                current_file_bars_df.to_parquet(output_bar_file, index=False, engine='pyarrow')
                logger.info(f"Saved {len(current_file_bars_df)} dollar bars to {output_bar_file}")
                total_bars_generated_overall += len(current_file_bars_df)
            else:
                logger.info(f"No dollar bars generated from {file_path.name}.")

            # Reset generator state for the next file (clears its internal accumulators)
            generator.reset() 
            processed_files_count += 1
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}. Skipping.")
            continue
        except pd.errors.EmptyDataError:
            logger.error(f"No data or empty file: {file_path}. Skipping.")
            continue
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}", exc_info=True)
            # Decide whether to continue or break here. For robustness, let's continue.
            logger.warning(f"Skipping file {file_path.name} due to error. Continuing with next file.")
            continue

    logger.info(f"--- Phase 2: Dollar Bar Generation ({DOLLAR_THRESHOLD_STR}) Finished ---")
    logger.info(f"Processed {processed_files_count}/{len(input_files) if input_files else 0} input files.")
    logger.info(f"Total ticks processed across all files: {total_ticks_processed_overall}")
    logger.info(f"Total dollar bars generated and saved: {total_bars_generated_overall}")


if __name__ == '__main__':
    main() 