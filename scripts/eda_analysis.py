import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Custom Logger to redirect print statements
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command, which for example
        # sys.stdout.flush() is Terminal            self.terminal.flush()
        self.log.flush()

# --- Configuration ---
DATA_PATH = '../data/multi_res_features/2M_x10_x16/BTCUSDT_all_multi_res_features_targets.parquet'
PLOT_OUTPUT_DIR = '../outputs/eda_plots/' # Relative to the script's location

# --- Helper Functions ---
def ensure_dir(directory_path):
    """Ensures that the directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

# --- EDA Functions ---
def load_data(path):
    """Loads the dataset."""
    print(f"Loading data from: {path}")
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
    print("\n--- First 5 rows ---")
    print(df.head())
    print("\n--- Last 5 rows ---")
    print(df.tail())
    print("\n--- DataFrame Info ---")
    df.info(verbose=True, show_counts=True)

def display_descriptive_stats(df, plot_dir):
    """Displays and saves descriptive statistics."""
    if df.empty:
        print("DataFrame is empty. Cannot display descriptive statistics.")
        return
    print("\n--- Descriptive Statistics (Overall) ---")
    desc_stats = df.describe(include='all', percentiles=[.01, .05, .25, .5, .75, .95, .99]).transpose()
    print(desc_stats)
    # Optionally save to CSV
    # desc_stats.to_csv(os.path.join(plot_dir, 'descriptive_statistics.csv'))
    # print(f"Descriptive statistics saved to {os.path.join(plot_dir, 'descriptive_statistics.csv')}")


def analyze_target_variable(df, target_col='target_long', plot_dir=None):
    """Analyzes and plots the target variable distribution."""
    if df.empty:
        print("DataFrame is empty. Cannot analyze target variable.")
        return
    if target_col not in df.columns:
        print(f"ERROR: Target column '{target_col}' not found in DataFrame.")
        return

    print(f"\n--- Target Variable ({target_col}) Value Counts ---")
    target_counts = df[target_col].value_counts(dropna=False)
    print(target_counts)
    print(f"\n--- Target Variable ({target_col}) Proportions ---")
    target_proportions = df[target_col].value_counts(normalize=True, dropna=False)
    print(target_proportions)

    # Plotting
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=target_col, data=df, order=target_counts.index, palette="viridis")
    plt.title(f'Distribution of Target Variable ({target_col})')
    plt.xlabel('Target Label')
    plt.ylabel('Count')

    for i, count in enumerate(target_counts):
        percentage = target_proportions.iloc[i] * 100
        ax.text(i, count + (0.01 * df.shape[0]), f'{count}\n({percentage:.2f}%)', ha='center', va='bottom')
    
    if plot_dir:
        plot_path = os.path.join(plot_dir, f'{target_col}_distribution.png')
        plt.savefig(plot_path)
        print(f"Target variable distribution plot saved to {plot_path}")
    plt.close()

    nan_in_target = df[target_col].isnull().sum()
    print(f"\nNumber of NaN values in '{target_col}': {nan_in_target}")
    if nan_in_target > 0:
        print(f"Percentage of NaN values in '{target_col}': {(nan_in_target / len(df) * 100):.2f}%")

def analyze_missing_values(df, plot_dir=None):
    """Analyzes and plots missing values."""
    if df.empty:
        print("DataFrame is empty. Cannot perform NaN analysis.")
        return
        
    print("\n--- Missing Values (NaN) Analysis ---")
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
            plt.figure(figsize=(15, 8)) # Increased figure size
            sns.barplot(x=nan_summary_filtered.index, y='NaN Percentage', data=nan_summary_filtered, palette="rocket")
            plt.xticks(rotation=90)
            plt.title('Percentage of Missing Values by Feature')
            plt.ylabel('Percentage (%)')
            plt.tight_layout()
            plot_path = os.path.join(plot_dir, 'missing_values_percentage.png')
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

    print("\n--- Numerical Feature Distribution Analysis ---")
    ensure_dir(os.path.join(plot_dir, 'numerical_distributions')) # Subdirectory for these plots

    for feature in features_to_plot:
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in DataFrame. Skipping.")
            continue

        print(f"\nAnalyzing feature: {feature}")
        
        # Drop NaNs for this specific feature analysis
        feature_data = df[feature].dropna()

        if feature_data.empty:
            print(f"No data available for '{feature}' after dropping NaNs. Skipping.")
            continue

        # Descriptive Stats
        print("Descriptive Statistics:")
        print(f"  Mean:   {feature_data.mean():.4f}")
        print(f"  Median: {feature_data.median():.4f}")
        print(f"  Std Dev: {feature_data.std():.4f}")
        print(f"  Skew:   {feature_data.skew():.4f}")
        print(f"  Kurtosis: {feature_data.kurtosis():.4f}")
        print(f"  Min:    {feature_data.min():.4f}")
        print(f"  Max:    {feature_data.max():.4f}")

        # Plot Histogram & KDE
        plt.figure(figsize=(12, 6))
        sns.histplot(feature_data, kde=True, bins=50)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        if plot_dir:
            hist_path = os.path.join(plot_dir, 'numerical_distributions', f'{feature}_histogram.png')
            plt.savefig(hist_path)
            print(f"Histogram for {feature} saved to {hist_path}")
        plt.close()

        # Plot Box Plot
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=feature_data)
        plt.title(f'Box Plot of {feature}')
        plt.xlabel(feature)
        if plot_dir:
            box_path = os.path.join(plot_dir, 'numerical_distributions', f'{feature}_boxplot.png')
            plt.savefig(box_path)
            print(f"Box plot for {feature} saved to {box_path}")
        plt.close()

def analyze_categorical_features(df, features_to_plot, plot_dir=None):
    """Analyzes and plots distributions for a list of categorical features."""
    if df.empty:
        print("DataFrame is empty. Cannot analyze categorical features.")
        return

    print("\n--- Categorical Feature Distribution Analysis ---")
    cat_plot_dir = os.path.join(plot_dir, 'categorical_distributions')
    ensure_dir(cat_plot_dir)

    for feature in features_to_plot:
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in DataFrame. Skipping.")
            continue
        
        feature_data = df[feature].dropna()
        if feature_data.empty:
            print(f"No data available for '{feature}' after dropping NaNs. Skipping.")
            continue

        print(f"\nAnalyzing feature: {feature}")
        counts = feature_data.value_counts().sort_index()
        proportions = feature_data.value_counts(normalize=True).sort_index()
        
        print("Value Counts:")
        print(counts)
        print("\nProportions:")
        print(proportions)

        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature_data, order=counts.index, palette='viridis')
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if plot_dir:
            plot_path = os.path.join(cat_plot_dir, f'{feature}_distribution.png')
            plt.savefig(plot_path)
            print(f"Plot for {feature} saved to {plot_path}")
        plt.close()

def analyze_correlations(df, target_col='target_long', plot_dir=None):
    """Analyzes and plots feature correlations."""
    if df.empty:
        print("DataFrame is empty. Cannot analyze correlations.")
        return

    print("\n--- Correlation Analysis ---")
    corr_plot_dir = os.path.join(plot_dir, 'correlation_plots')
    ensure_dir(corr_plot_dir)

    # Select numerical features for correlation matrix
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Exclude the target column itself from the list used for the main heatmap if it's purely for features
    # However, for a comprehensive view, it's often included. Let's include it.
    # If there are too many features, the heatmap can become hard to read.
    
    if not numerical_cols:
        print("No numerical columns found for correlation analysis.")
        return

    print(f"Calculating correlation matrix for {len(numerical_cols)} numerical features...")
    corr_matrix = df[numerical_cols].corr()

    # Plotting the full correlation matrix heatmap
    plt.figure(figsize=(40, 35)) # Large figure size for many features
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5) # annot=False for large matrices
    plt.title('Full Correlation Matrix of Numerical Features', fontsize=20)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    if plot_dir:
        heatmap_path = os.path.join(corr_plot_dir, 'full_correlation_heatmap.png')
        plt.savefig(heatmap_path, dpi=150) # Adjust DPI if needed for clarity/size
        print(f"Full correlation heatmap saved to {heatmap_path}")
    plt.close()

    # Correlation with the target variable
    if target_col in numerical_cols:
        print(f"\n--- Correlation with Target Variable ({target_col}) ---")
        target_correlations = corr_matrix[target_col].sort_values(ascending=False)
        
        # Remove self-correlation if target_col is in the matrix
        if target_col in target_correlations.index:
            target_correlations = target_correlations.drop(target_col) 
            
        print(target_correlations)

        # Plotting correlations with target
        plt.figure(figsize=(12, 20)) # Adjusted for many features
        target_correlations.plot(kind='barh', colormap='viridis') # Using colormap for better visual
        plt.title(f'Feature Correlation with {target_col}')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Feature')
        plt.grid(axis='x', linestyle='--')
        plt.tight_layout()
        if plot_dir:
            target_corr_path = os.path.join(corr_plot_dir, 'target_correlations_barplot.png')
            plt.savefig(target_corr_path)
            print(f"Bar plot of correlations with {target_col} saved to {target_corr_path}")
        plt.close()
    else:
        print(f"Target column '{target_col}' not found among numerical features for detailed correlation analysis.")

def identify_highly_correlated_features(df, threshold=0.9):
    """Identifies and prints pairs of features with absolute correlation above a threshold."""
    if df.empty:
        print("DataFrame is empty. Cannot identify highly correlated features.")
        return

    print(f"\n--- Highly Correlated Feature Pairs (Threshold > {threshold}) ---")
    numerical_df = df.select_dtypes(include=np.number)
    # Exclude the target column if it's present, as we are interested in feature-feature correlations
    if 'target_long' in numerical_df.columns:
        numerical_df = numerical_df.drop(columns=['target_long'])
        
    corr_matrix = numerical_df.corr().abs()
    
    # Create a mask for the upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    highly_correlated_pairs = []
    for column in upper.columns:
        for index in upper.index:
            if upper.loc[index, column] > threshold:
                highly_correlated_pairs.append((index, column, upper.loc[index, column]))
    
    if highly_correlated_pairs:
        print(f"Found {len(highly_correlated_pairs)} pairs with absolute correlation > {threshold}:")
        # Sort by correlation value, descending
        highly_correlated_pairs.sort(key=lambda x: x[2], reverse=True)
        for pair in highly_correlated_pairs:
            print(f"  {pair[0]} --- {pair[1]}: {pair[2]:.4f}")
    else:
        print(f"No feature pairs found with absolute correlation > {threshold}.")

def analyze_bivariate_distributions(df, features_to_analyze, plot_dir):
    logger.write("\\n--- Bivariate Feature Distributions vs. Target Analysis ---\\n")
    bivariate_plot_dir = os.path.join(plot_dir, 'bivariate_analysis')
    os.makedirs(bivariate_plot_dir, exist_ok=True)

    # Ensure target_long is not all NaN, and has both 0 and 1
    if df['target_long'].isna().all():
        logger.write("Target variable 'target_long' is all NaN. Skipping bivariate analysis.\\n")
        return
    if not (df['target_long'].isin([0, 1]).any()):
         logger.write("Target variable 'target_long' does not contain 0 or 1 after dropping NaNs. Skipping bivariate analysis.\\n")
         return


    df_filtered = df.dropna(subset=['target_long']) # Drop rows where target is NaN for this analysis
    df_filtered['target_long'] = df_filtered['target_long'].astype(int)


    for feature in features_to_analyze:
        if feature not in df_filtered.columns:
            logger.write(f"Feature '{feature}' not found in DataFrame. Skipping bivariate analysis for this feature.\\n")
            continue
        
        logger.write(f"\\nAnalyzing bivariate distribution for: {feature} vs. target_long\\n")

        # Descriptive stats grouped by target
        grouped_stats = df_filtered.groupby('target_long')[feature].agg(['mean', 'median', 'std'])
        logger.write("Descriptive Statistics by Target Outcome:\\n")
        logger.write(grouped_stats.to_string() + '\\n')

        # Boxplot
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='target_long', y=feature, data=df_filtered)
        plt.title(f'Box Plot of {feature} by Target Outcome (0: SL, 1: TP)')
        plt.xlabel('Target Outcome')
        plt.ylabel(feature)
        plt.tight_layout()
        plt.savefig(os.path.join(bivariate_plot_dir, f'{feature}_bivariate_boxplot.png'))
        plt.close()
        logger.write(f"Bivariate box plot for {feature} saved to {bivariate_plot_dir}\\n")

        # KDE Plot
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df_filtered[df_filtered['target_long'] == 0][feature], label='Target = 0 (SL)', fill=True)
        sns.kdeplot(df_filtered[df_filtered['target_long'] == 1][feature], label='Target = 1 (TP)', fill=True)
        plt.title(f'KDE Plot of {feature} by Target Outcome')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(bivariate_plot_dir, f'{feature}_bivariate_kde.png'))
        plt.close()
        logger.write(f"Bivariate KDE plot for {feature} saved to {bivariate_plot_dir}\\n")

def analyze_outliers(df, features_to_check, plot_dir, method='iqr', k=3):
    logger.write("\\n--- Outlier Analysis ---\\n")
    # plot_dir is passed for consistency, though not used for plots in this initial version

    for feature in features_to_check:
        if feature not in df.columns:
            logger.write(f"Feature '{feature}' not found in DataFrame. Skipping outlier analysis for this feature.\\n")
            continue
        
        logger.write(f"\\nAnalyzing outliers for: {feature}\\n")
        feature_data = df[feature].dropna()

        if feature_data.empty:
            logger.write(f"No data available for '{feature}' after dropping NaNs. Skipping.\\n")
            continue

        if method == 'iqr':
            Q1 = feature_data.quantile(0.25)
            Q3 = feature_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR
            
            logger.write(f"  Method: IQR (k={k})\n")
            logger.write(f"  Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}\\n")
            logger.write(f"  Lower Bound: {lower_bound:.4f}, Upper Bound: {upper_bound:.4f}\\n")
            
            outliers_lower = feature_data[feature_data < lower_bound]
            outliers_upper = feature_data[feature_data > upper_bound]
            
            num_outliers_lower = len(outliers_lower)
            num_outliers_upper = len(outliers_upper)
            total_data_points = len(feature_data)
            
            perc_outliers_lower = (num_outliers_lower / total_data_points) * 100 if total_data_points > 0 else 0
            perc_outliers_upper = (num_outliers_upper / total_data_points) * 100 if total_data_points > 0 else 0
            
            logger.write(f"  Outliers below lower bound: {num_outliers_lower} ({perc_outliers_lower:.4f}%)\\n")
            if num_outliers_lower > 0:
                logger.write(f"    Min value among lower outliers: {outliers_lower.min():.4f}\\n")
            logger.write(f"  Outliers above upper bound: {num_outliers_upper} ({perc_outliers_upper:.4f}%)\\n")
            if num_outliers_upper > 0:
                logger.write(f"    Max value among upper outliers: {outliers_upper.max():.4f}\\n")
        
        elif method == 'percentile':
            # Example: 1st and 99th percentile
            lower_perc = 1
            upper_perc = 99
            lower_bound = feature_data.quantile(lower_perc / 100.0)
            upper_bound = feature_data.quantile(upper_perc / 100.0)

            logger.write(f"  Method: Percentile ({lower_perc}th & {upper_perc}th)\\n")
            logger.write(f"  Lower Bound ({lower_perc}th percentile): {lower_bound:.4f}\\n")
            logger.write(f"  Upper Bound ({upper_perc}th percentile): {upper_bound:.4f}\\n")

            outliers_lower = feature_data[feature_data < lower_bound]
            outliers_upper = feature_data[feature_data > upper_bound]
            
            num_outliers_lower = len(outliers_lower)
            num_outliers_upper = len(outliers_upper)
            total_data_points = len(feature_data)
            
            # For percentile method, these percentages should ideally be close to lower_perc and (100-upper_perc)
            perc_outliers_lower = (num_outliers_lower / total_data_points) * 100 if total_data_points > 0 else 0
            perc_outliers_upper = (num_outliers_upper / total_data_points) * 100 if total_data_points > 0 else 0

            logger.write(f"  Values below {lower_perc}th percentile: {num_outliers_lower} ({perc_outliers_lower:.4f}%)\\n")
            logger.write(f"  Values above {upper_perc}th percentile: {num_outliers_upper} ({perc_outliers_upper:.4f}%)\\n")
        else:
            logger.write(f"  Outlier detection method '{method}' not recognized. Skipping for {feature}.\\n")

# --- Main Execution ---
def main():
    """Main function to run the EDA script."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_plot_dir = os.path.join(script_dir, PLOT_OUTPUT_DIR)
    ensure_dir(absolute_plot_dir) # Ensure plot dir exists before trying to write log file there
    
    log_file_path = os.path.join(absolute_plot_dir, 'eda_script_console_output.txt')
    
    # Clear the log file at the beginning of each run if it exists
    if os.path.exists(log_file_path):
        open(log_file_path, 'w').close()
        
    global logger # Make logger global so other functions can use it
    logger = Logger(log_file_path)
    # Redirect stdout and stderr
    sys.stdout = logger
    sys.stderr = logger # Redirect stderr as well for robustness

    try:
        # Set display options for pandas
        pd.set_option('display.max_columns', 200)
        pd.set_option('display.max_rows', 200)
        pd.set_option('display.width', 120) # Adjusted for typical console width
        pd.set_option('display.float_format', '{:.4f}'.format)

        # For data path, it's also relative to the script.
        absolute_data_path = os.path.join(script_dir, DATA_PATH)

        # Matplotlib style
        plt.style.use('ggplot')

        df = load_data(absolute_data_path)
        if df.empty:
            print("Exiting script as data loading failed.")
            return

        display_basic_info(df)
        display_descriptive_stats(df, absolute_plot_dir)
        analyze_target_variable(df, plot_dir=absolute_plot_dir)
        analyze_missing_values(df, plot_dir=absolute_plot_dir)

        # Define numerical features to plot distributions for
        numerical_features_to_analyze = [
            's_log_return_close', 'm_log_return_close', 'l_log_return_close',
            'dollar_volume', 'volume', # Using the short-bar versions as representative
            's_rsi_14', 'm_atr_14', 'l_adx_14',
            'trade_imbalance', 'intra_bar_tick_price_volatility'
        ]
        analyze_numerical_feature_distributions(df, numerical_features_to_analyze, absolute_plot_dir)

        categorical_features_to_analyze = [
            's_time_hour_of_day', 's_time_day_of_week'
        ]
        analyze_categorical_features(df, categorical_features_to_analyze, absolute_plot_dir)

        analyze_correlations(df, target_col='target_long', plot_dir=absolute_plot_dir)
        identify_highly_correlated_features(df, threshold=0.9)

        bivariate_features_to_analyze = [
            'trade_imbalance', 'l_rsi_14', 'tick_price_skewness', 's_macd', 'm_atr_14'
        ]
        analyze_bivariate_distributions(df, bivariate_features_to_analyze, absolute_plot_dir)

        outlier_features_to_analyze = [
            's_log_return_close', 'm_log_return_close', 'l_log_return_close',
            'dollar_volume', 'volume', 
            'intra_bar_tick_price_volatility', 'tick_price_skewness', 'tick_price_kurtosis',
            'm_atr_14', 's_macd', 'l_adx_14'
        ]
        analyze_outliers(df, outlier_features_to_analyze, absolute_plot_dir, method='iqr', k=3)

        print("\nEDA script execution finished.")

    finally: # Ensure stdout is restored and log is flushed (file will close on script exit)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if 'logger' in globals() and hasattr(logger, 'log') and not logger.log.closed:
            logger.log.flush() # Ensure everything is written
        # logger.close() # Not needed as file closes on exit or GC
        print(f"Console output also saved to: {log_file_path}")

if __name__ == '__main__':
    main() 