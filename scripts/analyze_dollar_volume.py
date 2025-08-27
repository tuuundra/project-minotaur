import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = PROJECT_ROOT / 'data' / 'historical_btc_trades_parquet'

# Sample files for analysis (diverse periods)
SAMPLE_FILES = [
    "BTCUSDT_2018_01.parquet", # Early period
    "BTCUSDT_2020_03.parquet", # High volatility (COVID)
    "BTCUSDT_2021_05.parquet", # Bull run high activity
    "BTCUSDT_2023_03.parquet", # More recent high volume
    "BTCUSDT_2024_04.parquet"  # Latest full month
]

def analyze_file_volume(file_path):
    """
    Analyzes a single Parquet file to calculate average daily and hourly dollar volume.
    Returns (avg_daily_volume, avg_hourly_volume, num_days_in_sample) or (None, None, 0) if an error occurs.
    """
    try:
        logger.info(f"Analyzing file: {file_path.name}")
        table = pq.read_table(file_path, columns=['timestamp', 'quote_quantity'])
        df = table.to_pandas()

        if df.empty:
            logger.warning(f"File {file_path.name} is empty or contains no relevant data.")
            return None, None, 0

        # Ensure timestamp is datetime and no NaNs in quote_quantity
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.dropna(subset=['quote_quantity'], inplace=True)
        df['quote_quantity'] = pd.to_numeric(df['quote_quantity'], errors='coerce')
        df.dropna(subset=['quote_quantity'], inplace=True)

        if df.empty:
            logger.warning(f"File {file_path.name} has no valid data after cleaning.")
            return None, None, 0
            
        # Calculate daily volume
        df['date'] = df['timestamp'].dt.date
        daily_volume = df.groupby('date')['quote_quantity'].sum()
        avg_daily_vol = daily_volume.mean()
        num_days = daily_volume.count()

        # Calculate hourly volume
        df['hour_of_day'] = df['timestamp'].dt.floor('H') # Group by hour
        hourly_volume = df.groupby('hour_of_day')['quote_quantity'].sum()
        avg_hourly_vol = hourly_volume.mean()
        
        logger.info(f"File: {file_path.name} - Avg Daily Vol: ${avg_daily_vol:,.2f} (over {num_days} days), Avg Hourly Vol: ${avg_hourly_vol:,.2f}")
        return avg_daily_vol, avg_hourly_vol, num_days

    except Exception as e:
        logger.error(f"Error analyzing file {file_path.name}: {e}", exc_info=True)
        return None, None, 0

def main():
    logger.info("Starting Dollar Volume Analysis for selected Parquet files...")

    all_daily_volumes = []
    all_hourly_volumes = []
    total_days_sampled = 0

    for file_name in SAMPLE_FILES:
        file_path = PARQUET_DIR / file_name
        if not file_path.exists():
            logger.warning(f"Sample file {file_name} not found. Skipping.")
            continue
        
        avg_daily, avg_hourly, num_days = analyze_file_volume(file_path)
        if avg_daily is not None and num_days > 0: # Weight by number of days in month
            all_daily_volumes.extend([avg_daily] * num_days) # Simple way to weight by days for overall average
        if avg_hourly is not None:
             # For hourly, a simple average of averages is probably fine, or we could weight by num_hours.
             # Let's keep it simple for now.
            all_hourly_volumes.append(avg_hourly)
        total_days_sampled += num_days

    if not all_daily_volumes:
        logger.error("No data collected from sample files. Cannot proceed with analysis.")
        return

    overall_avg_daily_volume = np.mean(all_daily_volumes) if all_daily_volumes else 0
    overall_avg_hourly_volume = np.mean(all_hourly_volumes) if all_hourly_volumes else 0

    logger.info("--- Overall Volume Summary (from samples) ---")
    logger.info(f"Overall Average Daily Dollar Volume: ${overall_avg_daily_volume:,.2f} (based on {total_days_sampled} days from sampled months)")
    logger.info(f"Overall Average Hourly Dollar Volume: ${overall_avg_hourly_volume:,.2f} (simple average of monthly hourly averages)")

    if overall_avg_daily_volume == 0:
        logger.error("Overall average daily volume is zero. Cannot suggest bar sizes.")
        return

    logger.info("\n--- Suggested Dollar Bar Sizes & Implications ---")
    
    target_bars_per_day_options = [150, 200, 250, 300, 400, 500, 600]

    for target_bars in target_bars_per_day_options:
        dollar_bar_size = overall_avg_daily_volume / target_bars
        logger.info(f"\nTargeting ~{target_bars} bars/day:")
        logger.info(f"  Suggested Dollar Bar Size: ${dollar_bar_size:,.0f}")
        
        if overall_avg_hourly_volume > 0 and dollar_bar_size > 0 :
            avg_bars_per_avg_hour = overall_avg_hourly_volume / dollar_bar_size
            logger.info(f"  This would yield ~{avg_bars_per_avg_hour:.2f} bars in an AVERAGE activity hour.")
            
            # Estimate for "active" hour (e.g., 2x average hourly volume)
            active_hourly_volume_estimate = overall_avg_hourly_volume * 2 
            avg_bars_per_active_hour = active_hourly_volume_estimate / dollar_bar_size
            logger.info(f"  This might yield ~{avg_bars_per_active_hour:.2f} bars in a MORE ACTIVE hour (e.g., 2x avg hourly vol).")
        else:
            logger.info("  Cannot estimate hourly bar formation due to zero average hourly volume or bar size.")

    logger.info("\n--- Considerations ---")
    logger.info("1. 'Average' can be skewed by extreme days/months in the sample.")
    logger.info("2. 'Active hour' is a rough estimate (2x average). Real peak activity can be much higher.")
    logger.info("3. Your goal of 'up to 5 bars an hour' suggests looking at the 'MORE ACTIVE hour' estimates.")
    logger.info("   Choose a dollar bar size where that estimate is around 5 or more, but also consider the average.")
    logger.info("4. This is a starting point. You may need to iterate based on feature generation and model performance.")
    
    logger.info("\nDollar volume analysis finished.")

if __name__ == "__main__":
    main() 