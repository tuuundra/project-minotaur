#!/usr/bin/env python3
"""
Tick Data Processor
==================

Converts historical BTC/USDT tick data (2020-2024) into aggregated OHLCV bars 
and advanced microstructure features. Uses streaming processing to minimize 
memory footprint and storage requirements.

Usage:
    python tick_data_processor.py
"""

import os
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TickDataProcessor:
    def __init__(self, base_path="historical_btc_trades", output_path="tick_data_processed"):
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.bars_1m_path = self.output_path / "bars_1m"
        self.bars_5m_path = self.output_path / "bars_5m" 
        self.features_path = self.output_path / "features"
        
        for path in [self.bars_1m_path, self.bars_5m_path, self.features_path]:
            path.mkdir(exist_ok=True)
            
        self.processing_stats = {}
        
    def get_target_years(self):
        """Get list of years to process (2020-2024)."""
        target_years = ['2020', '2021', '2022', '2023', '2024']
        available_years = []
        
        for year in target_years:
            year_dir = self.base_path / f"{year}_data"
            if year_dir.exists():
                available_years.append(year)
                
        print(f"📅 Target years: {target_years}")
        print(f"✅ Available years: {available_years}")
        return available_years
    
    def process_tick_data_to_bars(self, df_ticks, timeframe='1min'):
        """Convert tick data to OHLCV bars with microstructure features."""
        
        # Ensure timestamp is datetime
        if 'datetime' not in df_ticks.columns:
            df_ticks['datetime'] = pd.to_datetime(df_ticks['timestamp'], unit='ms')
        
        # Set datetime as index for resampling
        df_ticks = df_ticks.set_index('datetime').sort_index()
        
        # Basic OHLCV aggregation
        ohlcv = df_ticks['price'].resample(timeframe).agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last'
        }).dropna()
        
        # Volume aggregations
        volume_agg = df_ticks.resample(timeframe).agg({
            'quantity': 'sum',           # Total BTC volume
            'quote_quantity': 'sum',     # Total USDT volume
            'trade_id': 'count'          # Number of trades
        })
        volume_agg.columns = ['volume_btc', 'volume_usdt', 'trade_count']
        
        # Microstructure features
        micro_features = self._calculate_microstructure_features(df_ticks, timeframe)
        
        # Combine all features
        bars = pd.concat([ohlcv, volume_agg, micro_features], axis=1)
        
        # Fill forward any missing OHLC values
        bars[['open', 'high', 'low', 'close']] = bars[['open', 'high', 'low', 'close']].ffill()
        
        # Fill zeros for volume/count features
        bars[['volume_btc', 'volume_usdt', 'trade_count']] = bars[['volume_btc', 'volume_usdt', 'trade_count']].fillna(0)
        
        return bars.dropna()
    
    def _calculate_microstructure_features(self, df_ticks, timeframe):
        """Calculate advanced microstructure features from tick data."""
        
        features = {}
        
        # Resample for aggregations
        resampled = df_ticks.resample(timeframe)
        
        # 1. Order Flow Features
        features['buyer_initiated_ratio'] = resampled['is_buyer_maker'].apply(lambda x: (~x).mean())
        features['avg_trade_size_btc'] = resampled['quantity'].mean()
        features['avg_trade_size_usdt'] = resampled['quote_quantity'].mean()
        
        # 2. Trade Size Distribution
        def calc_large_trade_ratio(group):
            if len(group) == 0:
                return 0
            q80 = group['quote_quantity'].quantile(0.8)
            return (group['quote_quantity'] > q80).mean()
        
        def calc_small_trade_ratio(group):
            if len(group) == 0:
                return 0
            q20 = group['quote_quantity'].quantile(0.2)
            return (group['quote_quantity'] < q20).mean()
            
        features['large_trade_ratio'] = resampled.apply(calc_large_trade_ratio)
        features['small_trade_ratio'] = resampled.apply(calc_small_trade_ratio)
        
        # 3. Price Impact Proxies
        def calc_price_range_norm(group):
            if len(group) <= 1:
                return 0
            price_range = group['price'].max() - group['price'].min()
            price_mean = group['price'].mean()
            return price_range / price_mean if price_mean > 0 else 0
            
        features['price_range_norm'] = resampled.apply(calc_price_range_norm)
        
        # 4. Volume-Weighted Features
        def calc_vwap(group):
            if len(group) == 0:
                return np.nan
            return np.average(group['price'], weights=group['quantity'])
            
        features['vwap'] = resampled.apply(calc_vwap)
        
        # 5. Trade Frequency Features
        features['trades_per_minute'] = resampled['trade_id'].count()
        
        # 6. Bid-Ask Spread Proxy (using consecutive price differences)
        def calc_avg_price_step(group):
            if len(group) <= 1:
                return 0
            return group['price'].diff().abs().mean()
            
        features['avg_price_step'] = resampled.apply(calc_avg_price_step)
        
        # 7. Market Urgency (time between trades)
        def calc_avg_trade_interval(group):
            if len(group) <= 1:
                return np.nan
            time_diffs = group.index.to_series().diff().dt.total_seconds() * 1000
            return time_diffs.mean()
            
        features['avg_trade_interval_ms'] = resampled.apply(calc_avg_trade_interval)
        
        # 8. Volume Imbalance
        def calc_volume_imbalance(group):
            if len(group) == 0:
                return 0
            buyer_vol = group[~group['is_buyer_maker']]['quote_quantity'].sum()
            seller_vol = group[group['is_buyer_maker']]['quote_quantity'].sum()
            total_vol = buyer_vol + seller_vol
            if total_vol == 0:
                return 0
            return (buyer_vol - seller_vol) / total_vol
            
        features['volume_imbalance'] = resampled.apply(calc_volume_imbalance)
        
        return pd.DataFrame(features)
    
    def process_month_file(self, year, month_file):
        """Process a single month file and return aggregated bars."""
        
        print(f"  📝 Processing {month_file}...")
        
        year_dir = self.base_path / f"{year}_data"
        file_path = year_dir / month_file
        
        try:
            # Read tick data from zip
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                csv_name = month_file.replace('.zip', '.csv')
                
                with zip_ref.open(csv_name) as csv_file:
                    # Read all data for the month
                    df_ticks = pd.read_csv(
                        csv_file,
                        names=['trade_id', 'price', 'quantity', 'quote_quantity', 
                              'timestamp', 'is_buyer_maker', 'is_best_match']
                    )
            
            print(f"    📊 Loaded {len(df_ticks):,} trades")
            
            # Convert to bars
            bars_1m = self.process_tick_data_to_bars(df_ticks, '1min')
            bars_5m = self.process_tick_data_to_bars(df_ticks, '5min')
            
            print(f"    ✅ Generated {len(bars_1m)} x 1min bars, {len(bars_5m)} x 5min bars")
            
            # Calculate basic stats
            stats = {
                'month_file': month_file,
                'total_trades': len(df_ticks),
                'bars_1m': len(bars_1m),
                'bars_5m': len(bars_5m),
                'date_range': f"{df_ticks['timestamp'].min()} to {df_ticks['timestamp'].max()}",
                'price_range': f"${df_ticks['price'].min():.2f} - ${df_ticks['price'].max():.2f}",
                'processing_timestamp': datetime.now().isoformat()
            }
            
            return bars_1m, bars_5m, stats
            
        except Exception as e:
            print(f"    ❌ Error processing {month_file}: {e}")
            return None, None, {'error': str(e)}
    
    def process_month_file_chunked(self, year, month_file, chunk_size=5000000):
        """Process a large month file in chunks to avoid memory issues."""
        
        print(f"  📝 Processing {month_file} in chunks...")
        
        year_dir = self.base_path / f"{year}_data"
        file_path = year_dir / month_file
        
        try:
            all_bars_1m = []
            all_bars_5m = []
            total_trades = 0
            
            # Read and process in chunks
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                csv_name = month_file.replace('.zip', '.csv')
                
                with zip_ref.open(csv_name) as csv_file:
                    # Read in chunks
                    chunk_num = 0
                    
                    for chunk in pd.read_csv(
                        csv_file,
                        names=['trade_id', 'price', 'quantity', 'quote_quantity', 
                              'timestamp', 'is_buyer_maker', 'is_best_match'],
                        chunksize=chunk_size
                    ):
                        chunk_num += 1
                        total_trades += len(chunk)
                        print(f"    📊 Processing chunk {chunk_num}: {len(chunk):,} trades")
                        
                        # Convert chunk to bars
                        bars_1m = self.process_tick_data_to_bars(chunk, '1min')
                        bars_5m = self.process_tick_data_to_bars(chunk, '5min')
                        
                        all_bars_1m.append(bars_1m)
                        all_bars_5m.append(bars_5m)
            
            # Combine all chunks and re-aggregate overlapping periods
            if all_bars_1m:
                combined_1m = pd.concat(all_bars_1m, axis=0).sort_index()
                combined_5m = pd.concat(all_bars_5m, axis=0).sort_index()
                
                # Re-aggregate to handle overlapping periods at chunk boundaries
                final_1m = self._reaggregate_bars(combined_1m, '1min')
                final_5m = self._reaggregate_bars(combined_5m, '5min')
                
                print(f"    ✅ Generated {len(final_1m)} x 1min bars, {len(final_5m)} x 5min bars")
                
                # Calculate basic stats
                stats = {
                    'month_file': month_file,
                    'total_trades': total_trades,
                    'chunks_processed': chunk_num,
                    'bars_1m': len(final_1m),
                    'bars_5m': len(final_5m),
                    'processing_timestamp': datetime.now().isoformat()
                }
                
                return final_1m, final_5m, stats
            else:
                return None, None, {'error': 'No data processed'}
                
        except Exception as e:
            print(f"    ❌ Error processing {month_file}: {e}")
            return None, None, {'error': str(e)}
    
    def _reaggregate_bars(self, bars_df, timeframe):
        """Re-aggregate bars to handle overlapping periods from chunked processing."""
        
        # Group by timeframe and aggregate properly
        if 'open' in bars_df.columns:
            # OHLCV aggregation
            agg_dict = {
                'open': 'first',
                'high': 'max',
                'low': 'min', 
                'close': 'last',
                'volume_btc': 'sum',
                'volume_usdt': 'sum',
                'trade_count': 'sum'
            }
            
            # Add microstructure features with appropriate aggregations
            micro_cols = [col for col in bars_df.columns if col not in agg_dict.keys()]
            for col in micro_cols:
                if 'ratio' in col or 'imbalance' in col:
                    agg_dict[col] = 'mean'  # Average ratios/imbalances
                elif 'count' in col or 'trades' in col:
                    agg_dict[col] = 'sum'   # Sum counts
                else:
                    agg_dict[col] = 'mean'  # Average other features
            
            reaggregated = bars_df.resample(timeframe).agg(agg_dict)
            return reaggregated.dropna()
        else:
            return bars_df
    
    def process_year(self, year):
        """Process all months for a given year."""
        
        print(f"\n📅 Processing year {year}...")
        
        year_dir = self.base_path / f"{year}_data"
        month_files = sorted([f for f in os.listdir(year_dir) if f.endswith('.zip')])
        
        year_bars_1m = []
        year_bars_5m = []
        year_stats = []
        
        for month_file in month_files:
            # Use chunked processing for large files
            bars_1m, bars_5m, stats = self.process_month_file_chunked(year, month_file)
            
            if bars_1m is not None:
                year_bars_1m.append(bars_1m)
                year_bars_5m.append(bars_5m)
            
            year_stats.append(stats)
        
        # Combine all months for the year
        if year_bars_1m:
            combined_1m = pd.concat(year_bars_1m, axis=0).sort_index()
            combined_5m = pd.concat(year_bars_5m, axis=0).sort_index()
            
            # Save to parquet
            output_1m = self.bars_1m_path / f"btcusdt_1m_{year}.parquet"
            output_5m = self.bars_5m_path / f"btcusdt_5m_{year}.parquet"
            
            combined_1m.to_parquet(output_1m)
            combined_5m.to_parquet(output_5m)
            
            print(f"✅ Saved {year}: {len(combined_1m)} x 1min bars, {len(combined_5m)} x 5min bars")
            print(f"   📁 Files: {output_1m.name}, {output_5m.name}")
            
            # Calculate file sizes
            size_1m = output_1m.stat().st_size / (1024**2)
            size_5m = output_5m.stat().st_size / (1024**2)
            print(f"   💾 Sizes: {size_1m:.1f} MB (1m), {size_5m:.1f} MB (5m)")
            
        self.processing_stats[year] = year_stats
        return len(combined_1m) if year_bars_1m else 0, len(combined_5m) if year_bars_1m else 0
    
    def generate_summary_report(self):
        """Generate processing summary and recommendations."""
        
        # Calculate total statistics
        total_1m_bars = 0
        total_5m_bars = 0
        total_size_mb = 0
        
        for parquet_file in self.bars_1m_path.glob("*.parquet"):
            total_size_mb += parquet_file.stat().st_size / (1024**2)
            
        for parquet_file in self.bars_5m_path.glob("*.parquet"):
            total_size_mb += parquet_file.stat().st_size / (1024**2)
        
        # Count total bars
        all_years = self.get_target_years()
        for year in all_years:
            file_1m = self.bars_1m_path / f"btcusdt_1m_{year}.parquet"
            file_5m = self.bars_5m_path / f"btcusdt_5m_{year}.parquet"
            
            if file_1m.exists():
                df_1m = pd.read_parquet(file_1m)
                total_1m_bars += len(df_1m)
                
            if file_5m.exists():
                df_5m = pd.read_parquet(file_5m)
                total_5m_bars += len(df_5m)
        
        # Generate report
        report = {
            'processing_summary': {
                'years_processed': all_years,
                'total_1m_bars': total_1m_bars,
                'total_5m_bars': total_5m_bars,
                'total_storage_mb': total_size_mb,
                'processing_timestamp': datetime.now().isoformat()
            },
            'year_stats': self.processing_stats,
            'next_steps': [
                "Data successfully converted to 1m and 5m bars with microstructure features",
                "Ready for feature engineering pipeline integration",
                "Consider creating combined dataset for model training",
                "Validate data quality and feature distributions"
            ]
        }
        
        report_path = self.output_path / "processing_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        return report
    
    def run_full_processing(self):
        """Run complete tick-to-bars processing pipeline."""
        
        print("🚀 Starting tick data processing (2020-2024)...\n")
        
        target_years = self.get_target_years()
        
        total_1m = 0
        total_5m = 0
        
        for year in target_years:
            bars_1m, bars_5m = self.process_year(year)
            total_1m += bars_1m
            total_5m += bars_5m
        
        # Generate summary
        report = self.generate_summary_report()
        
        print(f"\n" + "="*60)
        print("🎯 PROCESSING COMPLETE")
        print("="*60)
        print(f"📊 Total 1-minute bars: {total_1m:,}")
        print(f"📊 Total 5-minute bars: {total_5m:,}")
        print(f"💾 Total storage: {report['processing_summary']['total_storage_mb']:.1f} MB")
        print(f"📁 Output directory: {self.output_path}")
        
        print(f"\n🎉 Ready for feature engineering phase!")
        
        return report

if __name__ == "__main__":
    processor = TickDataProcessor()
    results = processor.run_full_processing() 