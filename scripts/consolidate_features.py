import pandas as pd
import numpy as np
from pathlib import Path
import os
import glob
import logging
import argparse # Added for command-line arguments
import dask
import dask.dataframe as dd # Added for Dask
import dask.utils # Potentially useful for Dask, e.g. MBytes
import shutil # For managing spill directory
import sys # Added for sys.exit()

# --- Dask Configuration for Spilling ---
PROJECT_ROOT_FOR_SPILL = Path(__file__).resolve().parent.parent # minotaur/
SPILL_TEMP_DIR = PROJECT_ROOT_FOR_SPILL / "data" / "dask_spill_temp"

# Ensure the spill directory exists and is empty
if SPILL_TEMP_DIR.exists():
    shutil.rmtree(SPILL_TEMP_DIR) # Remove if exists to start clean
SPILL_TEMP_DIR.mkdir(parents=True, exist_ok=True)
dask.config.set({'temporary_directory': str(SPILL_TEMP_DIR)})
logging.info(f"Dask temporary directory for spilling set to: {SPILL_TEMP_DIR}")
# --- End Dask Spill Configuration ---

# Attempt to disable dask-expr query planner if it's causing issues
try:
    dask.config.set({"dataframe.query-planning": False})
    logging.info("Attempted to disable Dask experimental query planner (dask-expr).")
except Exception as e:
    logging.warning(f"Could not disable dask-expr, proceeding with default Dask behavior: {e}")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent # This should point to minotaur/

TIME_BARS_BASE_DIR = PROJECT_ROOT / "data" / "time_bars_features_with_vp" # New directory with VP features
TIME_RESOLUTIONS = ["1min", "15min", "4hour"] # Corresponds to directory names

# Source for S, M, L dollar bar features
DOLLAR_BAR_FEATURES_FILE = PROJECT_ROOT / "data" / "multi_res_features" / "2M_x10_x16" / "BTCUSDT_all_multi_res_features.parquet"

# --- Source for Target Column ---
# The original consolidated file that contains the properly aligned target variable.
TARGET_SOURCE_FILE = PROJECT_ROOT / "data" / "consolidated_features_targets_all.parquet"
TARGET_COLUMN_NAME_IN_SOURCE = "target_tp2.0_sl1.0"
# Assuming the timestamp column in TARGET_SOURCE_FILE that corresponds to the index is 'timestamp' or 'open_timestamp'
# Let's try to determine this or ensure the merge key is consistent.
# The previous check showed index name as None, but it has the target.
# The dollar bar features use 'open_timestamp' as their index after processing.
# The time bar features also use 'datetime' which is renamed to 'timestamp' and then becomes the index.
# We need a common index name for the final merge. Let's aim for 'open_timestamp' or 'timestamp'.
# The final merged DataFrame from features has its index named 'timestamp'.
# So, we need to ensure the target DataFrame is also indexed by 'timestamp'.

OUTPUT_DIR = PROJECT_ROOT / "data"
# CONSOLIDATED_FILENAME = "consolidated_features_all.parquet" # Will be dynamic based on suffix

# This map is for documentation or if string market_regime columns were encountered.
# Time bar features already have integer market_regime (0:Bear, 1:Bull, 2:Range)
MARKET_REGIME_DEFINITION = {
    0: 'Bearish Trend',
    1: 'Bullish Trend',
    2: 'Ranging Market'
}
# Original string mapping if needed for other sources, but time bars are int.
# MARKET_REGIME_MAP_FROM_STRING = { 
#     'bearish': 0, 
#     'bullish': 1,
#     'ranging': 2,
# }


# Columns that should NOT have suffixes added (typically the index or common merge keys)
COLS_TO_IGNORE_SUFFIXING_TB = [] # For time bars
# For the dollar bar file, the 'open_timestamp' is the index, others get transformed.

def get_time_resolution_suffix_map():
    """Returns the mapping from time resolution string to model feature suffix."""
    return {
        "1min": "tb_1min",
        "15min": "tb_15min",
        "4hour": "tb_4hour"
    }

def load_and_prepare_time_bar_df(file_path, resolution_key, expected_suffix):
    # print("DEBUG_PRINT: Entered load_and_prepare_time_bar_df", flush=True) # REMOVED
    logger.info(f"Processing Time Bar file for {resolution_key} from {file_path} using Dask...")
    try:
        # Use Dask to read Parquet. Dask will infer divisions if possible or operate without them.
        # For Parquet, Dask typically handles partitioning well if the Parquet file was written efficiently.
        df = dd.read_parquet(file_path, engine='pyarrow') # Specify engine for clarity
        logger.info(f"Dask DataFrame loaded for {resolution_key}. Partitions: {df.npartitions}")
    except Exception as e:
        logger.error(f"Dask error reading {file_path}: {e}")
        return None

    # Dask DataFrames are lazy, operations build a graph.
    # Ensure datetime index
    # We need to know the name of the original datetime column or if it's already the index.
    # Assuming the original Parquet files for time bars might have 'datetime' as a column
    # or the index is already convertible to datetime.
    current_index_name = df.index.name
    if 'datetime' in df.columns:
        try:
            df['datetime'] = dd.to_datetime(df['datetime'])
            df = df.set_index('datetime', sort=True) # Let set_index handle sorting
            logger.info(f"Set AND sorted index to 'datetime' for Dask DataFrame from {file_path}.")
        except Exception as e:
            logger.error(f"Error during set_index(sort=True) on 'datetime' for {file_path}: {e}")
            return None
    elif current_index_name is not None and current_index_name != 'datetime':
        logger.info(f"Existing index \"{current_index_name}\" found for {file_path}. Attempting to sort it if not already sorted.")
        # If the existing index is not named 'datetime', we should still try to sort it if it needs to be sorted.
        # However, dask-expr DataFrames might not have .sort_index(). 
        # This path might need re-evaluation if it's common and causes issues.
        # For now, if set_index(sort=True) is the primary way, this path is less critical if data is usually indexed by 'datetime'.
        # Let's assume for now that if an index exists and is not 'datetime', it might already be sorted or is not the primary path.
        # If sorting is essential here, we would need a dask-expr compatible way to sort an existing index.
        # This might involve resetting and then setting with sort=True if that's the only mechanism.
        # For now, we will log and proceed, relying on the `datetime` column path to be the main one.
        pass # Pass for now, as we are focusing on set_index(sort=True)
    elif df.index.name == 'datetime': # Index is already 'datetime'
        logger.info(f"Index is already 'datetime' for Dask DataFrame from {file_path}. Ensuring it is sorted.")
        # If it's already named datetime, it might not be sorted. 
        # The previous logic attempted df.sort_index() here. 
        # If dask-expr df doesn't have .sort_index(), we need an alternative or to rely on it being sorted by creation.
        # This is tricky. One way is to reset and set_index with sort=True:
        try:
            df = df.reset_index().set_index('datetime', sort=True)
            logger.info(f"Reset and re-set index to 'datetime' with sort=True for {file_path}.")
        except Exception as e:
            logger.error(f"Error resetting and re-setting index 'datetime' with sort=True for {file_path}: {e}")
            return None
    else: # Index is None or some other name, and 'datetime' column not found
        logger.warning(f"Index for {file_path} is '{(df.index.name)}'. It's not 'datetime' and 'datetime' column not found. Proceeding with existing index. Sorting may not occur.")

    # Persist the DataFrame state after index normalization and sorting (via set_index)
    df = df.persist()
    logger.info(f"Persisted Dask DataFrame for {file_path} after index setup/sort. Index name: {df.index.name}")

    # CRITICAL: Sort index BEFORE renaming columns -- This is now handled by set_index(sort=True)
    # try:
    #     df = df.sort_index()
    #     logger.info(f"Successfully sorted index for Dask DataFrame from {file_path}. Index name: {df.index.name}")
    # except Exception as e:
    #     logger.error(f"Error sorting index for {file_path} (Index: {df.index.name}, Type: {type(df.index)}): {e}. Returning None.")
    #     return None
    
    cols_to_rename = {}
    for col in df.columns:
        if col not in COLS_TO_IGNORE_SUFFIXING_TB:
            new_col_name = f"{col}_{expected_suffix}"
            cols_to_rename[col] = new_col_name
    
    df = df.rename(columns=cols_to_rename)
    # Example: logging computed columns might be too slow here.
    # logger.info(f"Renamed columns for {file_path}. Example new columns (Dask): {list(df.columns)[:5]}")
    logger.info(f"Prepared Dask DataFrame for {resolution_key} from {file_path}. Column examples: {list(df.columns)[:5]}. Index name: {df.index.name}")
    df = df.persist() # Persist after renaming
    return df

def load_and_prepare_dollar_bar_df(file_path):
    # print("DEBUG_PRINT: Entered load_and_prepare_dollar_bar_df", flush=True) # REMOVED
    logger.info(f"Loading dollar bar features from: {file_path} using Dask...")
    if not file_path.exists(): # This check is fine as Path object exists
        logger.error(f"Dollar bar features file not found: {file_path}")
        return None
    try:
        db_features_df = dd.read_parquet(file_path, engine='pyarrow')
        logger.info(f"Dask DataFrame loaded for dollar bars. Partitions: {db_features_df.npartitions}. Initial columns: {list(db_features_df.columns)[:10]}")

        if 'open_timestamp' not in db_features_df.columns:
            logger.error("'open_timestamp' column not found in dollar bar features file. Cannot set DatetimeIndex.")
            # Try to compute to see if it's an issue with lazy loading column names, though unlikely for Parquet
            # actual_cols = list(db_features_df.columns)
            # if 'open_timestamp' not in actual_cols:
            #     logger.error(f"Confirmed 'open_timestamp' not in actual columns: {actual_cols}")
            return None
        
        try:
            db_features_df['open_timestamp'] = dd.to_datetime(db_features_df['open_timestamp'])
            db_features_df = db_features_df.set_index('open_timestamp', sort=True) # Let set_index handle sorting
            logger.info(f"Set AND sorted index to 'open_timestamp' for dollar bar Dask DataFrame.")
        except Exception as e:
            logger.error(f"Error during set_index(sort=True) on 'open_timestamp' for dollar bar Dask DataFrame: {e}")
            return None

        # Persist the DataFrame state after index normalization and sorting (via set_index)
        db_features_df = db_features_df.persist()
        logger.info(f"Persisted dollar bar Dask DataFrame after index setup/sort. Index name: {db_features_df.index.name}")

        # CRITICAL: Sort index BEFORE renaming columns or renaming the axis -- This is now handled by set_index(sort=True)
        # try:
        #     db_features_df = db_features_df.sort_index()
        #     logger.info(f"Successfully sorted index for dollar bar Dask DataFrame. Index name: {db_features_df.index.name}")
        # except Exception as e:
        #     logger.error(f"Error sorting index for dollar bar Dask DataFrame (Index: {db_features_df.index.name}, Type: {type(db_features_df.index)}): {e}. Returning None.")
        #     return None
        
        cols_to_rename = {}
        # This loop iterates over column names, which Dask provides without computation.
        for col in db_features_df.columns:
            if col.startswith("s_"):
                new_col_name = f"{col[2:]}_db_s"
            elif col.startswith("m_"):
                new_col_name = f"{col[2:]}_db_m"
            elif col.startswith("l_"):
                new_col_name = f"{col[2:]}_db_l"
            else:
                new_col_name = col # Keep original name if no prefix matched

            if new_col_name != col:
                cols_to_rename[col] = new_col_name
        
        # Apply rename. Dask will build this into its task graph.
        df_processed = db_features_df.rename(columns=cols_to_rename)
        
        # For Dask, selecting columns this way is fine and lazy.
        # However, the original logic for `final_dollar_columns` and `df_processed` was a bit convoluted.
        # The rename above should handle keeping only relevant columns if the unmapped ones had no s_/m_/l_ prefix.
        # If we only want columns that were actually part of the s_/m_/l_ structure, we'd filter by new_col_name patterns.
        # The current db_features_df.rename(columns=cols_to_rename) will keep all columns, renaming those in the map.
        # This is usually fine. If we strictly need to ONLY keep renamed s/m/l columns, we'd do:
        # df_processed = db_features_df[list(cols_to_rename.keys())].rename(columns=cols_to_rename)
        # For now, let's assume keeping other original columns is fine, and they'll get dropped if not selected by model training script.

        logger.info(f"Renamed Dollar Bar Dask DataFrame columns. Resulting column examples: {list(df_processed.columns)[:10]}. Index name: {df_processed.index.name}")
        df_processed = df_processed.persist() # Persist after renaming
        return df_processed
    except Exception as e:
        logger.error(f"Dask error reading or processing dollar bar file {file_path}: {e}")
        return None

def load_and_prepare_target_df(target_file_path, target_column_name_in_source):
    logger.info(f"TARGET_LOAD: Loading and preparing target from: {target_file_path}")
    
    ACTUAL_TIMESTAMP_COL_IN_FILE = 'close_timestamp_tb_1min' # Fallback column

    try:
        pdf_full_source = pd.read_parquet(target_file_path, engine='pyarrow')
        logger.info(f"TARGET_LOAD: Pandas successfully loaded {target_file_path}. Shape: {pdf_full_source.shape}, Index type: {type(pdf_full_source.index)}, Index name: {pdf_full_source.index.name}")

        # --- Determine and set the correct DatetimeIndex ---
        if isinstance(pdf_full_source.index, pd.DatetimeIndex):
            logger.info(f"TARGET_LOAD: Source file's existing index is a DatetimeIndex. Name: '{pdf_full_source.index.name}'. Using it.")
            # If index is already DatetimeIndex, no need to set_index from column initially
        elif ACTUAL_TIMESTAMP_COL_IN_FILE in pdf_full_source.columns:
            logger.info(f"TARGET_LOAD: Source file's index is not DatetimeIndex OR is None. Attempting to set index from column: '{ACTUAL_TIMESTAMP_COL_IN_FILE}'.")
            pdf_full_source[ACTUAL_TIMESTAMP_COL_IN_FILE] = pd.to_datetime(pdf_full_source[ACTUAL_TIMESTAMP_COL_IN_FILE])
            pdf_full_source = pdf_full_source.set_index(ACTUAL_TIMESTAMP_COL_IN_FILE, drop=True)
            logger.info(f"TARGET_LOAD: Set index from '{ACTUAL_TIMESTAMP_COL_IN_FILE}'. New index type: {type(pdf_full_source.index)}")
        else:
            logger.error(f"TARGET_LOAD: Critical - Cannot establish DatetimeIndex. Index is not DatetimeIndex AND fallback column '{ACTUAL_TIMESTAMP_COL_IN_FILE}' not found. Available columns: {list(pdf_full_source.columns)}")
            return None

        # --- Ensure index is sorted and named 'timestamp' ---
        if not isinstance(pdf_full_source.index, pd.DatetimeIndex):
             logger.error(f"TARGET_LOAD: Critical - Index is still not a DatetimeIndex after attempts. Type: {type(pdf_full_source.index)}. Cannot proceed.")
             return None

        pdf_full_source = pdf_full_source.sort_index()
        pdf_full_source = pdf_full_source.rename_axis('timestamp') # Standardize index name
        logger.info(f"TARGET_LOAD: Index sorted and named 'timestamp'. Index head: {pdf_full_source.index[:3]}")

        # --- Select target column ---
        if target_column_name_in_source not in pdf_full_source.columns:
            logger.error(f"TARGET_LOAD: Critical - Target column '{target_column_name_in_source}' not found in DataFrame after index processing. Available: {list(pdf_full_source.columns)}")
            return None
        
        # Select only the target column and the (now processed) index
        pdf_target_selected = pdf_full_source[[target_column_name_in_source]]
        logger.info(f"TARGET_LOAD: Selected target column '{target_column_name_in_source}'. Shape: {pdf_target_selected.shape}")

        # --- Handle NaNs in index ---
        initial_rows_target = len(pdf_target_selected)
        pdf_target_selected = pdf_target_selected[pdf_target_selected.index.notna()]
        rows_after_idx_dropna = len(pdf_target_selected)
        if initial_rows_target - rows_after_idx_dropna > 0:
            logger.info(f"TARGET_LOAD: Dropped {initial_rows_target - rows_after_idx_dropna} rows due to NaN in the 'timestamp' index.")
        
        if pdf_target_selected.empty:
            logger.error("TARGET_LOAD: Target DataFrame is empty after dropping rows with NaN in the timestamp index. Cannot proceed.")
            return None
            
        logger.info(f"TARGET_LOAD: Prepared Pandas target DataFrame. Index name: {pdf_target_selected.index.name}, Example Index: {pdf_target_selected.index[:3]}, Shape: {pdf_target_selected.shape}, Non-null targets: {pdf_target_selected[target_column_name_in_source].notna().sum()}")

        # --- Convert to Dask DataFrame ---
        # For a single column + index, npartitions can be smaller.
        # Consider npartitions strategy, maybe related to expected size or feature partitions.
        # Using fewer partitions for target as it's usually smaller than all features combined.
        ddf_target = dd.from_pandas(pdf_target_selected, npartitions=max(2, os.cpu_count() // 2)) 
        ddf_target = ddf_target.persist() # Persist to make it more robust for merge
        
        logger.info(f"TARGET_LOAD: Converted to Dask DataFrame and persisted. Partitions: {ddf_target.npartitions}, Known Divisions: {ddf_target.known_divisions}, Index name: {ddf_target.index.name}. Target Column: {list(ddf_target.columns)}")
        return ddf_target

    except Exception as e:
        logger.error(f"TARGET_LOAD: Error loading/preparing target from {target_file_path}: {e}", exc_info=True)
        return None

def main(time_bar_suffix=None, output_suffix=None):
    # Removed DEBUG_PRINT statements and inspection-only call
    logger.info("Starting Dask feature consolidation process...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    time_suffix_map = get_time_resolution_suffix_map()
    
    # --- Output Path Configuration ---
    base_output_dirname = "consolidated_features_all"
    final_output_dirname = f"{base_output_dirname}{'_' + output_suffix if output_suffix else ''}.parquet"
    final_output_dir_path = OUTPUT_DIR / final_output_dirname

    # --- Load Dollar Bar Features (Dask) ---
    logger.info(f"Attempting to load dollar bar features from: {DOLLAR_BAR_FEATURES_FILE}")
    ddf_dollar_all_res = None
    if DOLLAR_BAR_FEATURES_FILE.exists():
        ddf_dollar_all_res = load_and_prepare_dollar_bar_df(DOLLAR_BAR_FEATURES_FILE)
        if ddf_dollar_all_res is not None and len(ddf_dollar_all_res.columns) > 0:
            if ddf_dollar_all_res.index.name != 'timestamp':
                 ddf_dollar_all_res = ddf_dollar_all_res.rename_axis('timestamp')
                 logger.info(f"Renamed dollar bar feature DataFrame index to 'timestamp'. Persisting change.")
                 ddf_dollar_all_res = ddf_dollar_all_res.persist()
            logger.info(f"Dollar bar features prepared. Index: {ddf_dollar_all_res.index.name}, Columns: {len(ddf_dollar_all_res.columns)}")
        else:
            logger.warning(f"Dollar bar Dask DataFrame from {DOLLAR_BAR_FEATURES_FILE} is None or empty. Skipping dollar bars.")
            ddf_dollar_all_res = None
    else:
        logger.warning(f"Consolidated Dollar Bar feature file not found at {DOLLAR_BAR_FEATURES_FILE}. Skipping dollar bars.")
        ddf_dollar_all_res = None

    # --- Load and Prepare Time Bar Features (Dask) ---
    all_feature_dfs_to_merge = []
    if ddf_dollar_all_res is not None:
        all_feature_dfs_to_merge.append(ddf_dollar_all_res)

    for resolution in TIME_RESOLUTIONS:
        logger.info(f"--- Preparing Time Resolution: {resolution} for merge ---")
        expected_suffix = time_suffix_map.get(resolution)
        if not expected_suffix:
            logger.warning(f"No suffix mapping found for resolution: {resolution}. Skipping.")
            continue

        specific_time_bar_dir = TIME_BARS_BASE_DIR / resolution
        # Assuming the pattern is like BTCUSDT_time_bars_features_1min_v2.parquet
        # If time_bar_suffix is 'v2', pattern becomes: f"BTCUSDT_time_bars_features_{resolution}_{time_bar_suffix}.parquet"
        # If time_bar_suffix is None, it was using a hardcoded _v2. Let's make it dynamic or check original intent.
        # The original snippet had `glob_pattern = f"BTCUSDT_time_bars_features_{resolution}_v2.parquet"`
        # This implies v2 was fixed. Let's stick to that unless time_bar_suffix is meant to override this.
        # For now, assume `_v2` is part of the filename structure we're looking for.
        # If the `time_bar_suffix` arg is provided, it should ideally be used here.
        # Let's make it use the time_bar_suffix if provided, otherwise default to _v2 (or no suffix if that's intended)
        
        file_pattern_suffix = f"_{time_bar_suffix}" if time_bar_suffix else "_v2" # Default to _v2 if no suffix provided
        glob_pattern = f"BTCUSDT_time_bars_features_{resolution}{file_pattern_suffix}.parquet"
        logger.info(f"Using glob pattern for {resolution}: {glob_pattern}")

        file_paths = sorted(glob.glob(str(specific_time_bar_dir / glob_pattern)))

        if not file_paths:
            logger.warning(f"No files found in {specific_time_bar_dir} for pattern '{glob_pattern}'. Skipping {resolution}.")
            continue
        
        file_path_to_load = Path(file_paths[0]) # Takes the first file if multiple match (e.g. different date ranges)
        logger.info(f"Selected time bar file for {resolution}: {file_path_to_load}")
        
        ddf_time_res = load_and_prepare_time_bar_df(file_path_to_load, resolution, expected_suffix)

        if ddf_time_res is not None and len(ddf_time_res.columns) > 0:
            if ddf_time_res.index.name != 'timestamp': # Ensure index is named 'timestamp'
                ddf_time_res = ddf_time_res.rename_axis('timestamp')
                logger.info(f"Renamed {resolution} time bar DataFrame index to 'timestamp'. Persisting change.")
                ddf_time_res = ddf_time_res.persist()
            all_feature_dfs_to_merge.append(ddf_time_res)
            logger.info(f"Added {resolution} time bar Dask DataFrame to merge list. Index: {ddf_time_res.index.name}, Columns: {len(ddf_time_res.columns)}")
        else:
            logger.warning(f"Dask DataFrame for {resolution} is None or empty. Skipping.")

    if not all_feature_dfs_to_merge:
        logger.error("No feature DataFrames (dollar or time bars) were loaded. Cannot proceed.")
        sys.exit("Exiting: No feature data loaded.") # Exit if no features

    # --- Merge All Feature DataFrames ---
    logger.info(f"Starting merge of {len(all_feature_dfs_to_merge)} Dask feature DataFrames...")
    features_merged_ddf = all_feature_dfs_to_merge[0]
    if len(all_feature_dfs_to_merge) > 1:
        for i in range(1, len(all_feature_dfs_to_merge)):
            logger.info(f"Merging feature DataFrame {i+1}/{len(all_feature_dfs_to_merge)} (Index: {all_feature_dfs_to_merge[i].index.name}) onto current (Index: {features_merged_ddf.index.name})")
            # Ensure both DFs have known divisions before merge if possible
            if not features_merged_ddf.known_divisions:
                logger.warning(f"Left side of merge (running total) has no known divisions. Repartitioning.")
                features_merged_ddf = features_merged_ddf.repartition(partition_size="128MB").persist()
            if not all_feature_dfs_to_merge[i].known_divisions:
                logger.warning(f"Right side of merge (DF {i+1}) has no known divisions. Repartitioning.")
                all_feature_dfs_to_merge[i] = all_feature_dfs_to_merge[i].repartition(partition_size="128MB").persist()

            features_merged_ddf = dd.merge(features_merged_ddf, all_feature_dfs_to_merge[i], how='outer', left_index=True, right_index=True)
            features_merged_ddf = features_merged_ddf.persist() 
            logger.info(f"Persisted after merging feature DataFrame {i+1}. Current merged columns: {len(features_merged_ddf.columns)}")
    else:
        logger.info("Only one feature DataFrame to merge (or start with). No iterative merge needed.")
    
    logger.info(f"All feature DataFrames merged. Resulting features_merged_ddf columns: {len(features_merged_ddf.columns)}. Index: {features_merged_ddf.index.name}")

    # --- Save and reload features_merged_ddf to simplify its graph and ensure divisions ---
    TEMP_FEATURES_PATH = SPILL_TEMP_DIR / "intermediate_merged_features.parquet"
    if TEMP_FEATURES_PATH.exists():
        logger.info(f"Removing existing temporary features path: {TEMP_FEATURES_PATH}")
        if TEMP_FEATURES_PATH.is_dir():
            shutil.rmtree(TEMP_FEATURES_PATH)
        else:
            os.remove(TEMP_FEATURES_PATH)

    logger.info(f"Saving merged features to temporary path for graph simplification: {TEMP_FEATURES_PATH}")
    try:
        features_merged_ddf.to_parquet(TEMP_FEATURES_PATH, engine='pyarrow', overwrite=True) 
    except Exception as e:
        logger.error(f"Failed to save intermediate merged features to {TEMP_FEATURES_PATH}: {e}", exc_info=True)
        sys.exit("Exiting: Failed to save intermediate features.")
    
    logger.info(f"Reloading merged features from temporary path: {TEMP_FEATURES_PATH}")
    try:
        solid_features_ddf = dd.read_parquet(TEMP_FEATURES_PATH, engine='pyarrow')
    except Exception as e:
        logger.error(f"Failed to reload intermediate merged features from {TEMP_FEATURES_PATH}: {e}", exc_info=True)
        sys.exit("Exiting: Failed to reload intermediate features.")
        
    if solid_features_ddf.index.name != 'timestamp':
        solid_features_ddf = solid_features_ddf.rename_axis('timestamp')
        logger.info(f"Renamed reloaded index to 'timestamp'.")
    
    solid_features_ddf = solid_features_ddf.persist()
    logger.info(f"Persisted reloaded features. Index: {solid_features_ddf.index.name}, Known Divisions: {solid_features_ddf.known_divisions}, Partitions: {solid_features_ddf.npartitions}")

    if not solid_features_ddf.known_divisions:
        logger.warning(f"Divisions for reloaded features (solid_features_ddf) are not known. Attempting repartition.")
        solid_features_ddf = solid_features_ddf.repartition(partition_size="128MB").persist() 
        logger.info(f"Repartitioned solid_features_ddf. New Known Divisions: {solid_features_ddf.known_divisions}, New Partitions: {solid_features_ddf.npartitions}")
        if not solid_features_ddf.known_divisions:
            logger.error("CRITICAL: Divisions for solid_features_ddf are STILL unknown after repartition. Merge with target might fail or be very slow.")
    
    logger.info(f"DEBUG_PROGRESS: Completed processing solid_features_ddf. Index: {solid_features_ddf.index.name}, Known Divisions: {solid_features_ddf.known_divisions}, Columns: {len(solid_features_ddf.columns)}")
    
    # --- Load and Merge Target Column (using the new Pandas-first approach) ---
    logger.info("DEBUG_PROGRESS: About to call load_and_prepare_target_df for the actual target processing.")
    ddf_target = load_and_prepare_target_df(TARGET_SOURCE_FILE, TARGET_COLUMN_NAME_IN_SOURCE)

    final_ddf_to_save = solid_features_ddf 
    if ddf_target is None:
        logger.error("Failed to load or prepare target DataFrame. Final output will NOT include targets.")
    else:
        logger.info(f"TARGET_LOAD_COMPLETE: ddf_target index: {ddf_target.index.name}, Known Divisions: {ddf_target.known_divisions}, Partitions: {ddf_target.npartitions}, Columns: {list(ddf_target.columns)}")
        
        if not ddf_target.known_divisions: # This should ideally be true from from_pandas with sorted input
            logger.warning(f"Divisions for ddf_target are not known after loading. Attempting repartition (this should be rare).")
            # Repartition strategy if divisions are lost. For a 2-col DF, npartitions from load_and_prepare should be fine.
            # If it got here, it means from_pandas didn't set divisions correctly.
            ddf_target = ddf_target.repartition(npartitions=ddf_target.npartitions).persist() 
            logger.info(f"Repartitioned ddf_target. New Known Divisions: {ddf_target.known_divisions}, New Partitions: {ddf_target.npartitions}")
            if not ddf_target.known_divisions:
                 logger.error("CRITICAL: Divisions for ddf_target are STILL unknown after explicit repartition. Merge is likely to fail or be slow.")

        logger.info(f"TARGET_MERGE: Merging target DataFrame (Index: {ddf_target.index.name}, Columns: {list(ddf_target.columns)}) with reloaded features (Index: {solid_features_ddf.index.name}, Columns: {len(solid_features_ddf.columns)})")
        logger.info(f"TARGET_MERGE: solid_features_ddf - Partitions: {solid_features_ddf.npartitions}, Divisions known: {solid_features_ddf.known_divisions}")
        logger.info(f"TARGET_MERGE: ddf_target - Partitions: {ddf_target.npartitions}, Divisions known: {ddf_target.known_divisions}")
        
        if solid_features_ddf.index.name != 'timestamp': # Should be 'timestamp'
             solid_features_ddf = solid_features_ddf.rename_axis('timestamp').persist()
             logger.warning("TARGET_MERGE: solid_features_ddf index was not 'timestamp' just before target merge; renamed.")
        if ddf_target.index.name != 'timestamp': # Should be 'timestamp'
             ddf_target = ddf_target.rename_axis('timestamp').persist()
             logger.warning("TARGET_MERGE: ddf_target index was not 'timestamp' just before target merge; renamed.")
        
        final_ddf_to_save = dd.merge(solid_features_ddf, ddf_target, how='left', left_index=True, right_index=True)
        final_ddf_to_save = final_ddf_to_save.persist()
        logger.info(f"TARGET_MERGE: Successfully merged targets. Final columns count: {len(final_ddf_to_save.columns)}. Example last 10: {list(final_ddf_to_save.columns)[-10:]}")
        
        if TARGET_COLUMN_NAME_IN_SOURCE in final_ddf_to_save.columns:
            logger.info(f"TARGET_MERGE: Confirmed target column '{TARGET_COLUMN_NAME_IN_SOURCE}' is in the final DataFrame.")
        else:
            logger.error(f"TARGET_MERGE: CRITICAL - Target column '{TARGET_COLUMN_NAME_IN_SOURCE}' MISSING after merge. Columns: {list(final_ddf_to_save.columns)}")

    # --- Save the Final DataFrame ---
    if final_ddf_to_save is not None:
        logger.info(f"Attempting to save final consolidated Dask DataFrame to: {final_output_dir_path}")
        if final_output_dir_path.exists():
            if final_output_dir_path.is_dir(): # Parquet output is a directory
                logger.warning(f"Final output directory {final_output_dir_path} exists. Removing it.")
                shutil.rmtree(final_output_dir_path)
            else: 
                logger.warning(f"Final output path {final_output_dir_path} exists as a file (should be dir for Parquet). Removing it.")
                os.remove(final_output_dir_path)
        
        try:
            final_ddf_to_save.to_parquet(final_output_dir_path, engine='pyarrow', overwrite=False) # Overwrite False as we handled dir removal
            logger.info(f"Successfully saved final consolidated Dask DataFrame to {final_output_dir_path}")
        except Exception as e:
            logger.error(f"Failed to save final DataFrame to {final_output_dir_path}: {e}", exc_info=True)
            sys.exit("Exiting: Failed to save final output.")
    else:
        logger.error("No final Dask DataFrame to save.")
        sys.exit("Exiting: No final DataFrame generated.")

    logger.info("Consolidation process finished.")

if __name__ == "__main__":
    # Removed DEBUG_PRINT statements for cleaner logs now
    parser = argparse.ArgumentParser(description="Consolidates time bar and dollar bar features into a single Parquet file.")
    parser.add_argument(
        "--time-bar-suffix",
        type=str,
        default=None, # Defaulting to None. Logic in main() now handles defaulting to _v2 for glob pattern if this is None.
        help="Suffix for the input time bar feature files (e.g., 'v2'). Excludes the leading underscore and .parquet."
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Suffix for the output consolidated Parquet file (e.g., 'v2'). Excludes the leading underscore and .parquet."
    )
    args = parser.parse_args()

    main(time_bar_suffix=args.time_bar_suffix, output_suffix=args.output_suffix) 