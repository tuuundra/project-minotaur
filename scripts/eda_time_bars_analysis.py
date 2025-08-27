import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path # Add Path import

# Custom Logger to redirect print statements
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8') # Added encoding

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- Configuration ---
RESOLUTION_NAME = '15min' # Added for clarity and future use

SCRIPT_DIR = Path(__file__).resolve().parent
# Assuming script is in minotaur/scripts/...
MINOTAUR_ROOT = SCRIPT_DIR.parent # This should be 'minotaur/scripts_folder_name_incorrectly_assumed_here/../minotaur'

# Correcting the MINOTAUR_ROOT to be the actual 'minotaur' directory if script is in minotaur/scripts/
MINOTAUR_ROOT_CORRECTED = SCRIPT_DIR.parent # This is minotaur/scripts, so parent is minotaur

# Build paths from the corrected minotaur root, relative to the workspace root which is parent of minotaur_root
WORKSPACE_ROOT = MINOTAUR_ROOT_CORRECTED.parent

DATA_PATH = WORKSPACE_ROOT / 'minotaur' / 'data' / 'time_bars_features' / RESOLUTION_NAME / f'BTCUSDT_time_bars_features_{RESOLUTION_NAME}.parquet'
PLOT_OUTPUT_BASE_DIR = WORKSPACE_ROOT / 'minotaur' / 'outputs' / 'time_bar_eda_outputs' / RESOLUTION_NAME 
PLOT_OUTPUT_DIR = PLOT_OUTPUT_BASE_DIR / 'plots'


# --- Helper Functions ---
def ensure_dir(directory_path):
    """Ensures that the directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

# --- EDA Functions ---
def load_data(path):
    """Loads the dataset."""
    print(f"Loading data from: {path} for {RESOLUTION_NAME} EDA")
    try:
        df = pd.read_parquet(path)
        print(f"Successfully loaded data. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"ERROR: File not found at {path}. Please check the path.")
        return pd.DataFrame()

def display_basic_info(df):
    """Displays basic information about the DataFrame."""
    if df.empty:
        print("DataFrame is empty. Cannot display info.")
        return
    print("\n")
    print(f"--- First 5 rows ({RESOLUTION_NAME}) ---")
    print(df.head())
    print("\n")
    print(f"--- Last 5 rows ({RESOLUTION_NAME}) ---")
    print(df.tail())
    print("\n")
    print(f"--- DataFrame Info ({RESOLUTION_NAME}) ---")
    df.info(verbose=True, show_counts=True)

def display_descriptive_stats(df, plot_dir): # plot_dir is not used here for desc_stats saving
    """Displays and saves descriptive statistics."""
    if df.empty:
        print("DataFrame is empty. Cannot display descriptive statistics.")
        return
    print("\n")
    print(f"--- Descriptive Statistics (Overall - {RESOLUTION_NAME}) ---")
    desc_stats = df.describe(include='all', percentiles=[.01, .05, .25, .5, .75, .95, .99]).transpose()
    print(desc_stats)
    # Optionally save to CSV
    # ensure_dir(plot_dir) # ensure base plot_dir exists if we save here
    # desc_stats_path = os.path.join(plot_dir, f'descriptive_statistics_{RESOLUTION_NAME}.csv')
    # desc_stats.to_csv(desc_stats_path)
    # print(f"Descriptive statistics saved to {desc_stats_path}")

# Commented out as target variable is not loaded in this version
# def analyze_target_variable(df, target_col='target_long', plot_dir=None):
#     """Analyzes and plots the target variable distribution."""
#     if df.empty:
#         print("DataFrame is empty. Cannot analyze target variable.")
#         return
#     if target_col not in df.columns:
#         print(f"ERROR: Target column '{target_col}' not found in DataFrame.")
#         return

#     print(f"\n--- Target Variable ({target_col}) Value Counts ({RESOLUTION_NAME}) ---")
#     target_counts = df[target_col].value_counts(dropna=False)
#     print(target_counts)
#     print(f"\n--- Target Variable ({target_col}) Proportions ({RESOLUTION_NAME}) ---")
#     target_proportions = df[target_col].value_counts(normalize=True, dropna=False)
#     print(target_proportions)

#     # Plotting
#     plt.figure(figsize=(8, 6))
#     ax = sns.countplot(x=target_col, data=df, order=target_counts.index, palette="viridis")
#     plt.title(f'Distribution of Target Variable ({target_col}) - {RESOLUTION_NAME}')
#     plt.xlabel('Target Label')
#     plt.ylabel('Count')

#     for i, count in enumerate(target_counts):
#         percentage = target_proportions.iloc[i] * 100
#         ax.text(i, count + (0.01 * df.shape[0]), f'{count}\n({percentage:.2f}%)', ha='center', va='bottom')
    
#     if plot_dir:
#         plot_path = os.path.join(plot_dir, f'{target_col}_distribution_{RESOLUTION_NAME}.png')
#         ensure_dir(os.path.dirname(plot_path))
#         plt.savefig(plot_path)
#         print(f"Target variable distribution plot saved to {plot_path}")
#     plt.close()

#     nan_in_target = df[target_col].isnull().sum()
#     print(f"\nNumber of NaN values in '{target_col}': {nan_in_target}")
#     if nan_in_target > 0:
#         print(f"Percentage of NaN values in '{target_col}': {(nan_in_target / len(df) * 100):.2f}%")


def analyze_missing_values(df, plot_dir=None):
    """Analyzes and plots missing values."""
    if df.empty:
        print("DataFrame is empty. Cannot perform NaN analysis.")
        return
        
    print("\n")
    print(f"--- Missing Values (NaN) Analysis ({RESOLUTION_NAME}) ---")
    nan_counts = df.isnull().sum()
    nan_percentages = (df.isnull().sum() / len(df)) * 100
    
    nan_summary = pd.DataFrame({
        'NaN Count': nan_counts,
        'NaN Percentage': nan_percentages
    })
    
    nan_summary_filtered = nan_summary[nan_summary['NaN Count'] > 0].sort_values(by='NaN Percentage', ascending=False)
    
    if not nan_summary_filtered.empty:
        print("Columns with Missing Values:")
        print(nan_summary_filtered)
        
        if plot_dir:
            plt.figure(figsize=(15, 10)) # Increased figure size & height
            sns.barplot(x=nan_summary_filtered.index, y='NaN Percentage', data=nan_summary_filtered, palette="rocket")
            plt.xticks(rotation=90)
            plt.title(f'Percentage of Missing Values by Feature ({RESOLUTION_NAME})')
            plt.ylabel('Percentage (%)')
            plt.tight_layout() # Ensure labels fit
            plot_path = os.path.join(plot_dir, f'missing_values_percentage_{RESOLUTION_NAME}.png')
            ensure_dir(os.path.dirname(plot_path))
            plt.savefig(plot_path)
            print(f"Missing values plot saved to {plot_path}")
            plt.close()
    else:
        print("No missing values found in the dataset. Well done!")


def analyze_numerical_feature_distributions(df, features_to_plot, plot_dir=None):
    """Analyzes and plots distributions for a list of numerical features."""
    if df.empty:
        print("DataFrame is empty. Cannot analyze numerical features.")
        return

    print("\n")
    print(f"--- Numerical Feature Distribution Analysis ({RESOLUTION_NAME}) ---")
    dist_plot_dir = os.path.join(plot_dir, 'numerical_distributions')
    ensure_dir(dist_plot_dir)

    for feature in features_to_plot:
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in DataFrame. Skipping.")
            continue

        print("\n")
        print(f"Analyzing feature: {feature}")
        
        feature_data = df[feature].dropna()

        if feature_data.empty:
            print(f"No data available for '{feature}' after dropping NaNs. Skipping.")
            continue

        print("Descriptive Statistics:")
        print(f"  Mean:   {feature_data.mean():.4f}")
        print(f"  Median: {feature_data.median():.4f}")
        print(f"  Std Dev: {feature_data.std():.4f}")
        print(f"  Skew:   {feature_data.skew():.4f}")
        print(f"  Kurtosis: {feature_data.kurtosis():.4f}")
        print(f"  Min:    {feature_data.min():.4f}")
        print(f"  Max:    {feature_data.max():.4f}")

        plt.figure(figsize=(12, 6))
        sns.histplot(feature_data, kde=True, bins=50)
        plt.title(f'Distribution of {feature} ({RESOLUTION_NAME})')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        if plot_dir:
            hist_path = os.path.join(dist_plot_dir, f'{feature}_histogram_{RESOLUTION_NAME}.png')
            plt.savefig(hist_path)
            print(f"Histogram for {feature} saved to {hist_path}")
        plt.close()

        plt.figure(figsize=(10, 4))
        sns.boxplot(x=feature_data)
        plt.title(f'Box Plot of {feature} ({RESOLUTION_NAME})')
        plt.xlabel(feature)
        if plot_dir:
            box_path = os.path.join(dist_plot_dir, f'{feature}_boxplot_{RESOLUTION_NAME}.png')
            plt.savefig(box_path)
            print(f"Box plot for {feature} saved to {box_path}")
        plt.close()

def analyze_categorical_feature_distributions(df, features_to_plot, plot_dir=None):
    """Analyzes and plots distributions for a list of categorical features."""
    if df.empty:
        print("DataFrame is empty. Cannot analyze categorical features.")
        return

    print("\n")
    print(f"--- Categorical Feature Distribution Analysis ({RESOLUTION_NAME}) ---")
    cat_plot_dir = os.path.join(plot_dir, 'categorical_distributions')
    ensure_dir(cat_plot_dir)

    for feature in features_to_plot:
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in DataFrame. Skipping.")
            continue
        
        print("\n")
        print(f"Analyzing categorical feature: {feature}")
        feature_counts = df[feature].value_counts(dropna=False)
        print(feature_counts)
        feature_proportions = df[feature].value_counts(normalize=True, dropna=False)
        print(feature_proportions)

        plt.figure(figsize=(10, 6))
        # Order by index (category value) if it makes sense, otherwise by count
        order = feature_counts.index #.sort_values() if feature_counts.index.is_numeric() else feature_counts.index
        ax = sns.barplot(x=feature_counts.index, y=feature_counts.values, order=order, palette="Set2")
        plt.title(f'Distribution of {feature} ({RESOLUTION_NAME})')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right') # Rotate for readability if many categories
        
        for i, count in enumerate(feature_counts):
            percentage = feature_proportions.iloc[i] * 100
            # Adjust text position if needed
            ax.text(i, count + (0.01 * df.shape[0] if df.shape[0] > 0 else 1), f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')

        if plot_dir:
            plot_path = os.path.join(cat_plot_dir, f'{feature}_barplot_{RESOLUTION_NAME}.png')
            plt.tight_layout()
            plt.savefig(plot_path)
            print(f"Bar plot for {feature} saved to {plot_path}")
        plt.close()


def analyze_correlations(df, target_col='target_long', plot_dir=None): # target_col is not used here
    """Analyzes and plots feature correlations."""
    if df.empty:
        print("DataFrame is empty. Cannot analyze correlations.")
        return

    print("\n")
    print(f"--- Correlation Analysis ({RESOLUTION_NAME}) ---")
    
    # Select only numerical columns for correlation matrix
    numerical_df = df.select_dtypes(include=np.number)
    if numerical_df.empty:
        print("No numerical columns found for correlation analysis.")
        return

    correlation_matrix = numerical_df.corr()
    
    print("\nFull Numerical Feature Correlation Matrix (Head):")
    print(correlation_matrix.head())

    if plot_dir:
        corr_plot_dir = os.path.join(plot_dir, 'correlation_plots')
        ensure_dir(corr_plot_dir)

        plt.figure(figsize=(20, 18)) # Adjust size as needed
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1) # annot=False for large matrices
        plt.title(f'Full Numerical Feature Correlation Matrix ({RESOLUTION_NAME})')
        plt.tight_layout()
        plot_path = os.path.join(corr_plot_dir, f'full_correlation_heatmap_{RESOLUTION_NAME}.png')
        plt.savefig(plot_path)
        print(f"Full correlation heatmap saved to {plot_path}")
        plt.close()

    # Target correlation part is removed as target_col is not guaranteed to be present
    # if target_col in numerical_df.columns:
    #     target_correlations = numerical_df.corr()[target_col].sort_values(ascending=False)
    #     print(f"\n--- Correlations with Target Variable ({target_col}) ---")
    #     print(target_correlations)

    #     if plot_dir:
    #         plt.figure(figsize=(10, max(12, len(target_correlations)//4))) # Adjusted figsize
    #         sns.barplot(x=target_correlations.values, y=target_correlations.index, palette="coolwarm_r", orient='h')
    #         plt.title(f'Feature Correlations with {target_col} ({RESOLUTION_NAME})')
    #         plt.xlabel('Correlation Coefficient')
    #         plt.axvline(0, color='grey', lw=1, linestyle='--')
    #         plt.tight_layout()
    #         target_corr_path = os.path.join(corr_plot_dir, f'target_correlations_barplot_{RESOLUTION_NAME}.png')
    #         plt.savefig(target_corr_path)
    #         print(f"Target correlations bar plot saved to {target_corr_path}")
    #         plt.close()
    # else:
    #     print(f"Target column '{target_col}' not found in numerical features for correlation analysis.")
    

def identify_highly_correlated_features(df, threshold=0.95): # Increased threshold slightly
    """Identifies pairs of features with correlation above the threshold."""
    if df.empty:
        print("DataFrame is empty. Cannot identify highly correlated features.")
        return
    print("\n")
    print(f"--- Highly Correlated Feature Pairs (Threshold > {threshold}) ({RESOLUTION_NAME}) ---")
    
    numerical_df = df.select_dtypes(include=np.number)
    if numerical_df.empty:
        print("No numerical columns found for identifying highly correlated features.")
        return
        
    corr_matrix = numerical_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)) # Use astype(bool) instead of deprecated np.bool
    
    highly_correlated = []
    for column in upper.columns:
        for index in upper.index:
            if upper.loc[index, column] > threshold:
                highly_correlated.append((index, column, upper.loc[index, column]))
                
    if highly_correlated:
        print(f"Found {len(highly_correlated)} pairs with absolute correlation > {threshold}:")
        for feat1, feat2, corr_val in sorted(highly_correlated, key=lambda x: x[2], reverse=True):
            print(f"  {feat1} and {feat2}: {corr_val:.4f}")
    else:
        print(f"No feature pairs found with absolute correlation > {threshold}.")
    return highly_correlated


# Bivariate analysis relies on target, so commenting out for now
# def analyze_bivariate_distributions(df, features_to_analyze, target_col='target_long', plot_dir=None):
#     """Analyzes and plots bivariate distributions of features against the target."""
#     if df.empty:
#         print("DataFrame is empty. Cannot perform bivariate analysis.")
#         return
#     if target_col not in df.columns:
#         print(f"ERROR: Target column '{target_col}' not found. Skipping bivariate analysis.")
#         return

#     print(f"\n--- Bivariate Analysis (Features vs. {target_col}) ({RESOLUTION_NAME}) ---")
#     bivar_plot_dir = os.path.join(plot_dir, 'bivariate_analysis')
#     ensure_dir(bivar_plot_dir)

#     for feature in features_to_analyze:
#         if feature not in df.columns:
#             print(f"Feature '{feature}' not found in DataFrame. Skipping bivariate for this feature.")
#             continue

#         print(f"\nAnalyzing {feature} vs. {target_col}:")
        
#         # Descriptive stats grouped by target
#         grouped_stats = df.groupby(target_col)[feature].agg(['mean', 'median', 'std', 'count'])
#         print("Grouped Descriptive Statistics:")
#         print(grouped_stats)

#         # Box Plot
#         plt.figure(figsize=(8, 6))
#         sns.boxplot(x=target_col, y=feature, data=df, palette="pastel")
#         plt.title(f'{feature} by {target_col} ({RESOLUTION_NAME})')
#         if plot_dir:
#             box_path = os.path.join(bivar_plot_dir, f'{feature}_vs_{target_col}_boxplot_{RESOLUTION_NAME}.png')
#             plt.savefig(box_path)
#             print(f"Bivariate boxplot for {feature} saved to {box_path}")
#         plt.close()

#         # KDE Plot
#         plt.figure(figsize=(10, 6))
#         sns.kdeplot(data=df, x=feature, hue=target_col, fill=True, alpha=.5, palette="crest", common_norm=False)
#         plt.title(f'KDE of {feature} by {target_col} ({RESOLUTION_NAME})')
#         if plot_dir:
#             kde_path = os.path.join(bivar_plot_dir, f'{feature}_vs_{target_col}_kdeplot_{RESOLUTION_NAME}.png')
#             plt.savefig(kde_path)
#             print(f"Bivariate KDE plot for {feature} saved to {kde_path}")
#         plt.close()


def analyze_outliers(df, features_to_check, plot_dir, method='iqr', k=1.5): # Defaulted k to 1.5 for standard IQR
    """Analyzes outliers for specified features using IQR or Z-score method."""
    if df.empty:
        print("DataFrame is empty. Cannot analyze outliers.")
        return
    print("\n")
    print(f"--- Outlier Analysis (Method: {method.upper()}, k={k}) ({RESOLUTION_NAME}) ---")
    outlier_plot_dir = os.path.join(plot_dir, 'outlier_analysis_plots') # New distinct subdir for these plots
    ensure_dir(outlier_plot_dir)

    for feature in features_to_check:
        if feature not in df.columns:
            print(f"Feature '{feature}' not found. Skipping outlier analysis.")
            continue
        
        feature_data = df[feature].dropna()
        if feature_data.empty:
            print(f"No data for '{feature}' after dropping NaNs. Skipping outlier analysis.")
            continue

        print("\n")
        print(f"Outlier analysis for feature: {feature}")
        
        if method == 'iqr':
            Q1 = feature_data.quantile(0.25)
            Q3 = feature_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR
            outliers = feature_data[(feature_data < lower_bound) | (feature_data > upper_bound)]
        elif method == 'zscore':
            mean = feature_data.mean()
            std = feature_data.std()
            if std == 0: # Avoid division by zero for constant series
                print(f"  Standard deviation for {feature} is 0. Cannot calculate Z-scores.")
                outliers = pd.Series(dtype=feature_data.dtype) # Empty series
            else:
                z_scores = (feature_data - mean) / std
                outliers = feature_data[np.abs(z_scores) > k]
        else:
            print(f"Unsupported outlier detection method: {method}. Skipping {feature}.")
            continue
            
        num_outliers = len(outliers)
        percentage_outliers = (num_outliers / len(feature_data)) * 100 if len(feature_data) > 0 else 0
        
        print(f"  Number of outliers: {num_outliers}")
        print(f"  Percentage of outliers: {percentage_outliers:.2f}%")
        if num_outliers > 0:
            print(f"  Outlier values (first 5 if many): {outliers.head().tolist()}")
            print(f"  Min outlier: {outliers.min():.4f}, Max outlier: {outliers.max():.4f}")
        if method == 'iqr':
            print(f"  Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}")
            print(f"  Lower Bound: {lower_bound:.4f}, Upper Bound: {upper_bound:.4f}")

        # Plotting outliers - e.g., a histogram showing outliers
        if num_outliers > 0 and plot_dir: # Only plot if outliers exist and plot_dir is specified
            plt.figure(figsize=(12,6))
            sns.histplot(feature_data, label='Data', kde=False, bins=100)
            sns.histplot(outliers, label='Outliers', color='red', kde=False, bins=50)
            plt.axvline(lower_bound, color='orange', linestyle='--', label=f'Lower Bound ({lower_bound:.2f})')
            plt.axvline(upper_bound, color='purple', linestyle='--', label=f'Upper Bound ({upper_bound:.2f})')
            plt.title(f'Outlier Detection for {feature} ({RESOLUTION_NAME}) using {method.upper()}')
            plt.legend()
            outlier_hist_path = os.path.join(outlier_plot_dir, f'{feature}_outliers_{method}_{RESOLUTION_NAME}.png')
            plt.savefig(outlier_hist_path)
            print(f"Outlier visualization for {feature} saved to {outlier_hist_path}")
            plt.close()


def main():
    # Setup output directory and logger
    ensure_dir(PLOT_OUTPUT_BASE_DIR) # Ensure base output dir exists
    ensure_dir(PLOT_OUTPUT_DIR)     # Ensure plots subdir exists
    
    log_file_path = os.path.join(PLOT_OUTPUT_BASE_DIR, f'eda_{RESOLUTION_NAME}_console_output.txt')
    sys.stdout = Logger(log_file_path) # Redirect print to log file and console
    
    print(f"--- Starting EDA for {RESOLUTION_NAME} Time Bars ---")
    print(f"Script execution started at {pd.Timestamp.now()}")
    print(f"Data Source: {DATA_PATH}")
    print(f"Plot Output Directory: {PLOT_OUTPUT_DIR}")
    print(f"Console Log: {log_file_path}")

    df = load_data(DATA_PATH)

    if df.empty:
        print("Exiting script due to data loading failure.")
        return

    display_basic_info(df)
    display_descriptive_stats(df, PLOT_OUTPUT_BASE_DIR) # Pass base dir for potential CSV save
    
    # analyze_target_variable(df, plot_dir=PLOT_OUTPUT_DIR) # Commented out for time-bar EDA
    
    analyze_missing_values(df, plot_dir=PLOT_OUTPUT_DIR)

    # Define features for detailed distribution analysis - ADJUST THESE AS NEEDED for time bars
    NUMERICAL_FEATURES_FOR_DETAILED_DIST_ANALYSIS = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'log_return_close', 
        'price_change_pct', 'rsi_14', 'adx_14', 'atr_14', 'macd_hist',
        'market_regime', 'rsi_14_bull_div', 'rsi_14_bear_div',
        'ofi', 'vwap', # These are intra-bar features carried over
        'bb_width_20', 'cci_14', 'mfi_14', 'stoch_k_14_3_3', 'willr_14', 'volatility_logret_20'
    ]
    # Filter list to only include columns present in the DataFrame
    actual_numerical_features_to_plot = [f for f in NUMERICAL_FEATURES_FOR_DETAILED_DIST_ANALYSIS if f in df.columns]
    if not actual_numerical_features_to_plot:
         print("Warning: None of the specified NUMERICAL_FEATURES_FOR_DETAILED_DIST_ANALYSIS were found in the dataframe.")
    else:
        analyze_numerical_feature_distributions(df, actual_numerical_features_to_plot, plot_dir=PLOT_OUTPUT_DIR)

    CATEGORICAL_FEATURES_FOR_DIST_ANALYSIS = [
        # 'market_regime' is numerical (0,1,-1) but could be treated as categorical too.
        # Time-bar specific categorical features might be less common unless we bin some numerical ones.
        # For now, let's rely on the numerical analysis for market_regime.
        # Add any true categorical columns if they exist, e.g., from candlestick patterns (though they are int flags)
    ]
    actual_categorical_features_to_plot = [f for f in CATEGORICAL_FEATURES_FOR_DIST_ANALYSIS if f in df.columns]
    if actual_categorical_features_to_plot: # Only call if list is not empty
        analyze_categorical_feature_distributions(df, actual_categorical_features_to_plot, plot_dir=PLOT_OUTPUT_DIR)
    else:
        print("\nNo categorical features specified for detailed distribution analysis or none found in DataFrame.")

    analyze_correlations(df, plot_dir=PLOT_OUTPUT_DIR) # Target correlation part is commented out internally
    identify_highly_correlated_features(df, threshold=0.95) 

    # analyze_bivariate_distributions(df, actual_numerical_features_to_plot, plot_dir=PLOT_OUTPUT_DIR) # Commented out

    OUTLIER_FEATURES_TO_CHECK = actual_numerical_features_to_plot # Use the same list as for distributions
    analyze_outliers(df, OUTLIER_FEATURES_TO_CHECK, plot_dir=PLOT_OUTPUT_DIR, method='iqr', k=3) # Using k=3 as per original

    print("\n")
    print(f"--- EDA for {RESOLUTION_NAME} Time Bars Finished ---")
    print(f"Script execution finished at {pd.Timestamp.now()}")

if __name__ == "__main__":
    main() 