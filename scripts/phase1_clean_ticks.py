import pandas as pd
import os
from pathlib import Path
import logging
import glob
import re # For extracting year and month from filename
import zipfile # Added for unzipping
import tempfile # Added for temporary extraction
import pyarrow as pa # Added for explicit Parquet writer control
import pyarrow.parquet as pq # Added for explicit Parquet writer control

# Configure logging
# More verbose logging for this specific test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Assumes the script is in minotaur/scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
# Updated RAW_CSV_DIR to point to the parent directory of year_data folders
RAW_DATA_PARENT_DIR = PROJECT_ROOT / 'data' / 'raw_historical_btc_trades' 
OUTPUT_PARQUET_DIR = PROJECT_ROOT / 'data' / 'historical_btc_trades_parquet'
TEMP_EXTRACTION_DIR = PROJECT_ROOT / 'temp_extraction' # Define a specific temp dir within the project
SYMBOL = 'BTCUSDT' # Define symbol for output filenames
CHUNK_SIZE = 1_000_000 # Process 1 million rows at a time

# Expected columns in raw CSV and their desired names.
# These names are based on the discovery from tick_data_explorer.py
# The order in this list defines the column order IF the CSV has no header.
RAW_CSV_COLUMN_ORDER = [
    'trade_id', 
    'price', 
    'quantity', 
    'quote_quantity', 
    'timestamp', 
    'is_buyer_maker', 
    'is_best_match' # We'll read it but might not keep it in FINAL_COLUMNS
]

# This mapping helps standardize column names for the Parquet files.
# KEY: What your CSV column is currently NAMED (from RAW_CSV_COLUMN_ORDER)
# VALUE: The standardized name we want to use in the Parquet files
RAW_COLUMN_MAP = {
    'trade_id': 'trade_id',
    'price': 'price',
    'quantity': 'quantity',         # Changed from 'qty'
    'quote_quantity': 'quote_quantity', # Changed from 'quote_qty'
    'timestamp': 'timestamp',          # Changed from 'time'
    'is_buyer_maker': 'isBuyerMaker',
    'is_best_match': 'isBestMatch'    # Added, though might not be in FINAL_COLUMNS
}
# Define the columns we absolutely want to keep in the final Parquet.
# We'll use names from the VALUES of RAW_COLUMN_MAP.
# Keeping isBestMatch out for now to focus on dollar bar core features.
FINAL_COLUMNS = ['timestamp', 'price', 'quantity', 'quote_quantity', 'isBuyerMaker', 'trade_id']


def process_single_csv(csv_file_path, output_dir_path, year, month):
    logger.info(f"Inside process_single_csv for: {csv_file_path.name} ({year}-{month})")
    parquet_writer = None # Initialize Parquet writer outside the loop
    first_chunk_processed_schema = None # To store schema from the first chunk
    try:
        logger.info(f"Processing file in chunks: {csv_file_path.name} (Original Zip for Year: {year}, Month: {month})")
        
        output_filename = f"{SYMBOL}_{year}_{month}.parquet"
        output_file_path = output_dir_path / output_filename
        output_dir_path.mkdir(parents=True, exist_ok=True)

        for i, chunk_df in enumerate(pd.read_csv(csv_file_path, header=None, names=RAW_CSV_COLUMN_ORDER, chunksize=CHUNK_SIZE)):
            logger.debug(f"Processing chunk {i+1} of size {len(chunk_df)} for {csv_file_path.name}")
            
            df_renamed = chunk_df[[raw_col for raw_col in RAW_COLUMN_MAP.keys() if raw_col in chunk_df.columns]].copy()
            df_renamed.rename(columns=RAW_COLUMN_MAP, inplace=True)

            for col in FINAL_COLUMNS:
                if col not in df_renamed.columns:
                    logger.warning(f"Column '{col}' not found after renaming for chunk in {csv_file_path.name}.")
            
            df_final_chunk = df_renamed[[col for col in FINAL_COLUMNS if col in df_renamed.columns]].copy()

            if 'timestamp' not in df_final_chunk.columns:
                logger.error(f"'timestamp' column not found in chunk from {csv_file_path.name}. Skipping this chunk.")
                continue
            
            df_final_chunk['timestamp'] = pd.to_datetime(df_final_chunk['timestamp'], unit='ms', utc=True)
            df_final_chunk['timestamp'] = df_final_chunk['timestamp'].dt.tz_localize(None)

            df_final_chunk['price'] = pd.to_numeric(df_final_chunk['price'], errors='coerce')
            df_final_chunk['quantity'] = pd.to_numeric(df_final_chunk['quantity'], errors='coerce')
            df_final_chunk['quote_quantity'] = pd.to_numeric(df_final_chunk['quote_quantity'], errors='coerce')
            
            if 'isBuyerMaker' in df_final_chunk.columns:
                if df_final_chunk['isBuyerMaker'].dtype == 'object':
                    df_final_chunk['isBuyerMaker'] = df_final_chunk['isBuyerMaker'].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})
                df_final_chunk['isBuyerMaker'] = df_final_chunk['isBuyerMaker'].astype(bool)

            cols_to_check_for_nans = ['price', 'quantity', 'quote_quantity']
            initial_rows = len(df_final_chunk)
            df_final_chunk.dropna(subset=[col for col in cols_to_check_for_nans if col in df_final_chunk.columns], inplace=True)
            if len(df_final_chunk) < initial_rows:
                logger.warning(f"Dropped {initial_rows - len(df_final_chunk)} rows with NaNs from chunk in {csv_file_path.name}")

            if df_final_chunk.empty:
                logger.info(f"Chunk is empty after processing for {csv_file_path.name}. Skipping write.")
                continue

            # Convert pandas DataFrame chunk to PyArrow Table
            arrow_table = pa.Table.from_pandas(df_final_chunk, preserve_index=False)

            if parquet_writer is None: # First non-empty chunk
                first_chunk_processed_schema = arrow_table.schema
                parquet_writer = pq.ParquetWriter(output_file_path, first_chunk_processed_schema)
                logger.info(f"Created Parquet file and writer: {output_file_path} with schema from first chunk.")
            
            # Ensure subsequent chunks conform to the schema of the first chunk if necessary, though types should be consistent
            # For simplicity, we assume schema consistency after the cleaning steps. Robust code might add checks here.
            parquet_writer.write_table(arrow_table)
            logger.debug(f"Wrote chunk {i+1} to {output_file_path}")
        
        if parquet_writer is None: # Means no non-empty chunks were processed
            logger.warning(f"No data processed for {csv_file_path.name}. Parquet file might be empty or not created.")
            # Create an empty Parquet file with a schema if needed, or handle as an error.
            # For now, if no writer was created, it means no data.
            if first_chunk_processed_schema: # Should not happen if writer is None, but defensive
                 #This case means we had a schema but never wrote, which is odd.
                 pass # Or create an empty file with schema: pq.write_table(pa.Table.from_pandas(pd.DataFrame(columns=FINAL_COLUMNS), schema=first_chunk_processed_schema), output_file_path)
            else: # No data at all, not even a schema from a first chunk
                 logger.info(f"No data at all for {csv_file_path.name}, no Parquet file created.")
            return False 

        logger.info(f"Successfully processed all chunks and saved to: {output_file_path}")
        return True

    except Exception as e:
        logger.error(f"Error processing file {csv_file_path.name} by chunks: {e}", exc_info=True)
        return False
    finally:
        if parquet_writer:
            parquet_writer.close()
            logger.debug(f"Closed Parquet writer for {output_file_path}")

def main():
    logger.info("Starting Phase 1: Raw Tick ZIP/CSV to Cleaned Parquet Conversion (Full Run).")
    
    # Reverted from targeted file processing to full directory scan
    # TARGET_YEAR_DIR_NAME = "2024_data"
    # TARGET_ZIP_FILENAME = "BTCUSDT-trades-2024-03.zip"

    success_count = 0
    failure_count = 0
    skipped_count = 0 # To count already processed files

    OUTPUT_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Searching for year directories in: {RAW_DATA_PARENT_DIR}")
    year_dirs = sorted([d for d in RAW_DATA_PARENT_DIR.iterdir() if d.is_dir() and d.name.endswith('_data')])
    if not year_dirs:
        logger.warning(f"No year_data directories found in {RAW_DATA_PARENT_DIR}. Please check RAW_DATA_PARENT_DIR path.")
        return
    logger.info(f"Found year directories: {[yd.name for yd in year_dirs]}")

    for year_dir in year_dirs:
        logger.info(f"Processing year directory: {year_dir.name}")
        zip_files = sorted(list(year_dir.glob(f'{SYMBOL}-trades-*.zip')))
        if not zip_files:
            logger.info(f"No ZIP files found in {year_dir.name} for symbol {SYMBOL}. Skipping.")
            continue
        logger.info(f"Found {len(zip_files)} ZIP files in {year_dir.name}.")

        for zip_file_path in zip_files:
            zip_match = re.search(r'-(\d{4})-(\d{2})\.zip$', zip_file_path.name, re.IGNORECASE)
            if not zip_match:
                logger.warning(f"Could not extract year/month from ZIP filename: {zip_file_path.name} for pre-check. Skipping.")
                failure_count += 1 
                continue
            file_year, file_month = zip_match.groups()

            expected_output_filename = f"{SYMBOL}_{file_year}_{file_month}.parquet"
            expected_output_file_path = OUTPUT_PARQUET_DIR / expected_output_filename

            if expected_output_file_path.exists():
                logger.info(f"Output file {expected_output_file_path.name} already exists. Skipping.")
                skipped_count += 1
                continue 

            logger.info(f"Processing ZIP file: {zip_file_path.name}")
            try:
                with tempfile.TemporaryDirectory(dir=TEMP_EXTRACTION_DIR) as tmpdir:
                    tmpdir_path = Path(tmpdir)
                    logger.debug(f"Using temporary directory: {tmpdir_path} for {zip_file_path.name}")
                    with zipfile.ZipFile(zip_file_path, 'r') as zf:
                        csv_members = [m for m in zf.namelist() if m.lower().endswith('.csv')]
                        if not csv_members:
                            logger.warning(f"No CSV file found inside ZIP: {zip_file_path.name}")
                            failure_count +=1
                        else:
                            csv_to_extract = csv_members[0]
                            logger.info(f"Attempting to extract {csv_to_extract} from {zip_file_path.name} to {tmpdir_path}")
                            zf.extract(csv_to_extract, path=tmpdir_path)
                            logger.info(f"Successfully extracted {csv_to_extract} to {tmpdir_path}")
                            extracted_csv_path = tmpdir_path / csv_to_extract
                            
                            if process_single_csv(extracted_csv_path, OUTPUT_PARQUET_DIR, file_year, file_month):
                                success_count += 1
                            else:
                                failure_count += 1
            except zipfile.BadZipFile:
                logger.error(f"Bad ZIP file encountered: {zip_file_path.name}. Skipping.")
                failure_count += 1                    
            except Exception as e:
                logger.error(f"Error processing ZIP file {zip_file_path.name}: {e}", exc_info=True)
                failure_count += 1
            
    logger.info("--- Full Processing Summary ---")
    logger.info(f"Successfully converted {success_count} new files.")
    logger.info(f"Skipped {skipped_count} already existing files.")
    logger.info(f"Failed to process {failure_count} files.")
    logger.info("Full Parquet conversion process finished.")

if __name__ == "__main__":
    main() 