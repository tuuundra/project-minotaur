import pandas as pd
import os

# --- Configuration ---
FULL_DATA_PATH = '../data/multi_res_features/2M_x10_x16/BTCUSDT_all_multi_res_features_targets.parquet'
PRUNED_DATA_OUTPUT_DIR = '../data/selected_features/'
PRUNED_DATA_FILENAME = 'BTCUSDT_selected_features_v1.parquet'

# List of 29 features to drop based on multicollinearity analysis
COLUMNS_TO_DROP = [
    # Round 1 (near-perfect correlations)
    's_wick_vs_range', 'm_wick_vs_range', 'l_wick_vs_range',
    's_bbands_middle_20', 'm_bbands_middle_20', 'l_bbands_middle_20',
    's_sma_10', 'm_sma_10', 'l_sma_10',
    's_sma_20', 'm_sma_20', 'l_sma_20',
    's_volatility_log_returns_20', 'm_volatility_log_returns_20', 'l_volatility_log_returns_20',
    # Round 2 (still very high correlations & redundancy)
    's_roc_10', 'm_roc_10', 'l_roc_10',
    'volume_lag_2', 'volume_lag_3', # keeping volume, volume_lag_1, and s/m/l_volume_sma_20
    's_price_change_pct', 'm_price_change_pct', 'l_price_change_pct',
    's_macd', 'm_macd', 'l_macd',
    's_macdsignal', 'm_macdsignal', 'l_macdsignal' # keeping s/m/l_macdhist
]

def prune_features():
    """
    Loads the full feature dataset, drops specified columns, 
    and saves the pruned dataset.
    """
    # Ensure output directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_output_dir = os.path.join(script_dir, PRUNED_DATA_OUTPUT_DIR)
    if not os.path.exists(absolute_output_dir):
        os.makedirs(absolute_output_dir)
        print(f"Created directory: {absolute_output_dir}")

    absolute_input_path = os.path.join(script_dir, FULL_DATA_PATH)
    absolute_output_path = os.path.join(absolute_output_dir, PRUNED_DATA_FILENAME)

    print(f"Loading full feature dataset from: {absolute_input_path}")
    try:
        df = pd.read_parquet(absolute_input_path)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {absolute_input_path}. Exiting.")
        return

    print(f"Original DataFrame shape: {df.shape}")
    
    # Verify that all columns to drop actually exist in the DataFrame
    existing_columns_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
    missing_columns = [col for col in COLUMNS_TO_DROP if col not in df.columns]

    if missing_columns:
        print(f"Warning: The following columns intended for dropping were not found in the DataFrame and will be ignored:")
        for col in missing_columns:
            print(f"  - {col}")
    
    if not existing_columns_to_drop:
        print("No columns to drop from the provided list were found in the DataFrame. Exiting.")
        return

    print(f"\nDropping {len(existing_columns_to_drop)} columns:")
    for col in existing_columns_to_drop:
        print(f"  - {col}")

    df_pruned = df.drop(columns=existing_columns_to_drop)

    print(f"\nPruned DataFrame shape: {df_pruned.shape}")

    print(f"\nSaving pruned dataset to: {absolute_output_path}")
    try:
        df_pruned.to_parquet(absolute_output_path, index=False)
        print("Successfully saved pruned dataset.")
    except Exception as e:
        print(f"ERROR saving pruned dataset: {e}")

if __name__ == '__main__':
    prune_features() 