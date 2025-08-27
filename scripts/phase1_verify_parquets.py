import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd # For timestamp checks and NaN checks
from pathlib import Path
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = PROJECT_ROOT / 'data' / 'historical_btc_trades_parquet'
SYMBOL = 'BTCUSDT' # Used to match filenames
SAMPLE_SIZE = 5 # Number of rows to check from head and tail for content verification

# Define the expected schema based on phase1_clean_ticks.py
# FINAL_COLUMNS = ['timestamp', 'price', 'quantity', 'quote_quantity', 'isBuyerMaker', 'trade_id']
EXPECTED_SCHEMA = pa.schema([
    ('timestamp', pa.timestamp('ns')),
    ('price', pa.float64()),
    ('quantity', pa.float64()),
    ('quote_quantity', pa.float64()),
    ('isBuyerMaker', pa.bool_()),
    ('trade_id', pa.int64()) 
    # Note: trade_id was read as is, if it was string in CSV it might be string here.
    # However, phase1_clean_ticks.py does not explicitly convert trade_id to numeric.
    # Pandas read_csv without explicit dtype for trade_id might infer it as int64 if all values are integers.
    # If trade_ids can be non-numeric or very large, this might need adjustment or explicit conversion in phase1.
    # For now, assuming int64 is the most likely inferred type for numerical trade IDs.
])

# Expected column order as a list of names for easier comparison if needed, though schema field order matters more
EXPECTED_COLUMN_ORDER = [field.name for field in EXPECTED_SCHEMA]
COLUMNS_FOR_CONTENT_CHECK = ['timestamp', 'price', 'quantity', 'quote_quantity', 'isBuyerMaker']
COLUMNS_FOR_NAN_CHECK = ['price', 'quantity', 'quote_quantity']

def verify_parquet_file(file_path, file_year_str, file_month_str):
    """
    Verifies schema, non-emptiness, and data content of a single Parquet file.
    Returns True if all checks pass, False otherwise.
    """
    passed_all_checks = True
    try:
        # --- 1. Schema Verification ---
        actual_schema = pq.read_schema(file_path)
        actual_column_names = [field.name for field in actual_schema]
        
        missing_expected_cols = [name for name in EXPECTED_COLUMN_ORDER if name not in actual_column_names]
        if missing_expected_cols:
            logger.error(f"File: {file_path.name} - Check: SCHEMA - FAIL - Missing expected columns: {missing_expected_cols}")
            passed_all_checks = False

        extra_actual_cols = [name for name in actual_column_names if name not in EXPECTED_COLUMN_ORDER]
        if extra_actual_cols:
            logger.warning(f"File: {file_path.name} - Check: SCHEMA - WARN - Found extra columns: {extra_actual_cols}")
            # Not failing for extra columns, but good to note.

        for expected_field in EXPECTED_SCHEMA:
            expected_name = expected_field.name
            expected_type = expected_field.type
            if expected_name in actual_column_names:
                actual_field = actual_schema.field(expected_name)
                actual_type = actual_field.type
                if not actual_type.equals(expected_type):
                    logger.error(f"File: {file_path.name} - Check: SCHEMA - FAIL - Column '{expected_name}': Expected type {expected_type}, got {actual_type}")
                    passed_all_checks = False
            elif passed_all_checks: # Only log if not already failed due to missing column
                 logger.error(f"File: {file_path.name} - Check: SCHEMA - FAIL - Expected column '{expected_name}' not found for type check.")
                 passed_all_checks = False # Should have been caught by missing_expected_cols

        if not passed_all_checks:
            logger.info(f"File: {file_path.name} - Schema verification FAILED. Skipping content checks.")
            return False
        logger.info(f"File: {file_path.name} - Check: SCHEMA - PASS")

        # --- 2. Non-Empty Check ---
        metadata = pq.read_metadata(file_path)
        if metadata.num_rows == 0:
            logger.error(f"File: {file_path.name} - Check: NON-EMPTY - FAIL - File is empty (0 rows).")
            return False # Cannot do content checks on empty file
        logger.info(f"File: {file_path.name} - Check: NON-EMPTY - PASS ({metadata.num_rows} rows)")

        # --- 3. Data Content Checks (on a sample) ---
        # Read only necessary columns for content check to save memory
        table_sample_cols = pq.read_table(file_path, columns=COLUMNS_FOR_CONTENT_CHECK)
        
        sample_dfs = []
        if table_sample_cols.num_rows <= 2 * SAMPLE_SIZE:
            sample_dfs.append(table_sample_cols.to_pandas())
        else:
            sample_dfs.append(table_sample_cols.slice(0, SAMPLE_SIZE).to_pandas())
            sample_dfs.append(table_sample_cols.slice(table_sample_cols.num_rows - SAMPLE_SIZE, SAMPLE_SIZE).to_pandas())
        
        df_sample = pd.concat(sample_dfs).reset_index(drop=True)

        # 3a. NaN Check
        nan_check_passed = True
        for col in COLUMNS_FOR_NAN_CHECK:
            if col in df_sample.columns and df_sample[col].isnull().any():
                logger.error(f"File: {file_path.name} - Check: NaN - FAIL - NaN values found in column '{col}' in sample.")
                nan_check_passed = False
                passed_all_checks = False
        if nan_check_passed:
            logger.info(f"File: {file_path.name} - Check: NaN - PASS (for sample)")
        
        # 3b. Timestamp Range Check
        timestamp_check_passed = True
        if 'timestamp' in df_sample.columns:
            try:
                file_year = int(file_year_str)
                file_month = int(file_month_str)
                # Timestamps are 'ns', ensure they are timezone-naive for comparison as in phase1
                timestamps_to_check = pd.to_datetime(df_sample['timestamp']).dt.tz_localize(None)
                
                for ts in timestamps_to_check:
                    if not (ts.year == file_year and ts.month == file_month):
                        logger.error(f"File: {file_path.name} - Check: TIMESTAMP RANGE - FAIL - Timestamp {ts} outside expected year/month {file_year}-{file_month:02d}.")
                        timestamp_check_passed = False
                        passed_all_checks = False
                        break # One failure is enough for this check
            except ValueError:
                 logger.error(f"File: {file_path.name} - Check: TIMESTAMP RANGE - FAIL - Could not parse year/month from filename for check: {file_year_str}-{file_month_str}")
                 timestamp_check_passed = False
                 passed_all_checks = False
                 
            if timestamp_check_passed:
                logger.info(f"File: {file_path.name} - Check: TIMESTAMP RANGE - PASS (for sample)")
        else:
            logger.error(f"File: {file_path.name} - Check: TIMESTAMP RANGE - FAIL - 'timestamp' column not in sample for check.")
            passed_all_checks = False

        # 3c. Boolean Check for isBuyerMaker
        boolean_check_passed = True
        if 'isBuyerMaker' in df_sample.columns:
            # Schema check already confirmed bool type for the column.
            # This check ensures actual values are interpretable as bool by pandas, though pyarrow bool should be fine.
            if not pd.api.types.is_bool_dtype(df_sample['isBuyerMaker']):
                 logger.error(f"File: {file_path.name} - Check: BOOLEAN TYPE (isBuyerMaker) - FAIL - Column 'isBuyerMaker' in sample is not boolean. Actual type: {df_sample['isBuyerMaker'].dtype}")
                 boolean_check_passed = False
                 passed_all_checks = False
            # Additionally, ensure no NaNs if column is present, as bools shouldn't be NaN after cleaning
            elif df_sample['isBuyerMaker'].isnull().any():
                 logger.error(f"File: {file_path.name} - Check: BOOLEAN NaN (isBuyerMaker) - FAIL - NaN values found in 'isBuyerMaker' column in sample.")
                 boolean_check_passed = False
                 passed_all_checks = False
        else:
            # This should be caught by schema check if 'isBuyerMaker' is expected but missing
            logger.warning(f"File: {file_path.name} - Check: BOOLEAN TYPE (isBuyerMaker) - WARN - 'isBuyerMaker' column not in sample for check.")

        if boolean_check_passed and 'isBuyerMaker' in df_sample.columns:
             logger.info(f"File: {file_path.name} - Check: BOOLEAN (isBuyerMaker) - PASS (for sample)")

        if passed_all_checks:
            logger.info(f"File: {file_path.name} - ALL CHECKS PASSED.")
        return passed_all_checks

    except Exception as e:
        logger.error(f"File: {file_path.name} - Error during verification: {e}", exc_info=True)
        return False

def main():
    logger.info("Starting Parquet File Verification Process (Schema, Non-Empty, Content Sample)...")
    
    if not PARQUET_DIR.exists() or not PARQUET_DIR.is_dir():
        logger.error(f"Parquet directory does not exist or is not a directory: {PARQUET_DIR}")
        return

    filename_pattern = re.compile(rf"^{SYMBOL}_(\d{{4}})_(\d{{2}})\.parquet$")
    
    parquet_files_info = []
    for f in PARQUET_DIR.iterdir():
        if f.is_file():
            match = filename_pattern.match(f.name)
            if match:
                year, month = match.groups()
                parquet_files_info.append({'path': f, 'year': year, 'month': month})

    if not parquet_files_info:
        logger.warning(f"No Parquet files matching pattern '{filename_pattern.pattern}' found in {PARQUET_DIR}.")
        return

    logger.info(f"Found {len(parquet_files_info)} Parquet files to verify in {PARQUET_DIR}.")

    passed_files_count = 0
    failed_files_details = [] # Store name and reason

    # Sort by path for consistent processing order
    for file_info in sorted(parquet_files_info, key=lambda x: x['path']):
        pf_path = file_info['path']
        file_year = file_info['year']
        file_month = file_info['month']
        
        logger.info(f"--- Verifying: {pf_path.name} (Expected: {file_year}-{file_month}) ---")
        if verify_parquet_file(pf_path, file_year, file_month):
            passed_files_count += 1
        else:
            failed_files_details.append(pf_path.name)
        logger.info(f"--- Finished Verifying {pf_path.name} ---")

    logger.info("--- Overall Verification Summary ---")
    total_files = len(parquet_files_info)
    logger.info(f"Total files checked: {total_files}")
    logger.info(f"Files PASSED all checks: {passed_files_count}")
    logger.info(f"Files FAILED one or more checks: {len(failed_files_details)}")

    if failed_files_details:
        logger.error("List of files that FAILED verification:")
        for f_name in failed_files_details:
            logger.error(f"  - {f_name}")
    elif total_files > 0:
        logger.info("All checked files passed verification successfully!")
    else:
        logger.info("No files were checked.")

    logger.info("Parquet verification process finished.")

if __name__ == "__main__":
    main() 