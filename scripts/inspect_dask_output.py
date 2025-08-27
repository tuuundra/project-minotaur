import dask.dataframe as dd
import pandas as pd
import sys # For exiting cleanly

# Path to the Dask Parquet directory
# Note: This path is relative to the workspace root where the script will be run from.
parquet_dir_path = "minotaur/data/consolidated_features_all_v2_log_dask_fix_attempt2.parquet/"

print(f"Attempting to inspect Dask Parquet directory: {parquet_dir_path}")

try:
    # Read the Dask DataFrame from the directory of Parquet files
    ddf = dd.read_parquet(parquet_dir_path, engine='pyarrow')
    
    print(f"Successfully loaded Dask DataFrame from {parquet_dir_path}")
    
    # Get the (approximate) number of rows and exact number of columns
    num_rows = len(ddf) # This triggers a computation for the length
    num_cols = len(ddf.columns)
    
    print(f"Approximate number of rows: {num_rows}")
    print(f"Number of columns: {num_cols}")
    
    # Get the head of the DataFrame (computes the first few rows)
    print("\nFirst 5 rows (head):")
    # Compute the head and convert to pandas for display
    head_df = ddf.head(5) # ddf.head() itself computes and returns a pandas DataFrame
    print(head_df)

    print("\nData types (dtypes) of the first 30 columns (if available):")
    all_dtypes = ddf.dtypes # ddf.dtypes is a pandas Series
    if len(all_dtypes) > 30:
        print(all_dtypes.head(30))
    else:
        print(all_dtypes)
    
    print("\nData types (dtypes) of the last 30 columns (if available):")
    if len(all_dtypes) > 30:
        print(all_dtypes.tail(30))
    elif len(all_dtypes) > 0 : # if less than 30, but some columns exist
        print(all_dtypes)


    if 'target_tp2.0_sl1.0' in ddf.columns:
        print("\nTarget column 'target_tp2.0_sl1.0' is present.")
        print(f"Data type of target column: {ddf['target_tp2.0_sl1.0'].dtype}")
        # For a more thorough check, you might compute NaN counts or value counts on target:
        # nan_in_target = ddf['target_tp2.0_sl1.0'].isna().sum().compute()
        # print(f"Number of NaNs in target column: {nan_in_target}")
        # value_counts_target = ddf['target_tp2.0_sl1.0'].value_counts().compute()
        # print(f"Value counts for target column:\n{value_counts_target}")
    else:
        print("\nWARNING: Target column 'target_tp2.0_sl1.0' NOT found!")
        
except FileNotFoundError:
    print(f"ERROR: Parquet directory not found at {parquet_dir_path}. Please ensure the path is correct and the consolidation script ran successfully.")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: An error occurred while inspecting Parquet files in {parquet_dir_path}:")
    print(e)
    import traceback
    print(traceback.format_exc())
    sys.exit(1)

print("\nInspection script completed.") 