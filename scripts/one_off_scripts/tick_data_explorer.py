#!/usr/bin/env python3
"""
Tick Data Explorer
==================

Comprehensive exploration of historical BTC/USDT tick data from Binance.
Analyzes data consistency, quality, and characteristics across 7 years (2017-2024).

Usage:
    python tick_data_explorer.py
"""

import os
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class TickDataExplorer:
    def __init__(self, base_path="historical_btc_trades"):
        self.base_path = Path(base_path)
        self.summary_stats = {}
        
    def discover_data_structure(self):
        """Discover and catalog all available data files."""
        print("🔍 Discovering tick data structure...")
        
        data_dirs = []
        for item in self.base_path.iterdir():
            if item.is_dir() and "data" in item.name:
                data_dirs.append(item)
        
        data_dirs.sort()
        
        file_inventory = {}
        total_compressed_size = 0
        
        for data_dir in data_dirs:
            year = data_dir.name.split('_')[0]
            zip_files = list(data_dir.glob("*.zip"))
            
            dir_size = sum(f.stat().st_size for f in zip_files)
            total_compressed_size += dir_size
            
            file_inventory[year] = {
                'directory': data_dir,
                'zip_files': len(zip_files),
                'compressed_size_mb': dir_size / (1024**2),
                'files': [f.name for f in zip_files]
            }
            
            print(f"📁 {year}: {len(zip_files)} files, {dir_size/(1024**2):.1f} MB compressed")
        
        print(f"\n📊 Total compressed size: {total_compressed_size/(1024**3):.2f} GB")
        self.file_inventory = file_inventory
        return file_inventory
    
    def sample_data_quality(self, sample_files=3):
        """Sample data from different periods to check quality and consistency."""
        print(f"\n🧪 Sampling {sample_files} files for quality analysis...")
        
        # Select sample files from different years
        sample_targets = []
        years = sorted(self.file_inventory.keys())
        
        for i, year in enumerate(years[::max(1, len(years)//sample_files)]):
            if len(sample_targets) >= sample_files:
                break
            year_files = self.file_inventory[year]['files']
            if year_files:
                # Pick middle file of year for representativeness
                mid_file = year_files[len(year_files)//2]
                sample_targets.append((year, mid_file))
        
        sample_stats = {}
        
        for year, filename in sample_targets:
            print(f"\n📖 Analyzing {filename}...")
            
            file_path = self.file_inventory[year]['directory'] / filename
            
            try:
                # Read sample of data from zip
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    csv_name = filename.replace('.zip', '.csv')
                    
                    with zip_ref.open(csv_name) as csv_file:
                        # Read first 1000 lines for analysis
                        df_sample = pd.read_csv(
                            csv_file, 
                            nrows=1000,
                            names=['trade_id', 'price', 'quantity', 'quote_quantity', 
                                  'timestamp', 'is_buyer_maker', 'is_best_match']
                        )
                
                # Convert timestamp to datetime
                df_sample['datetime'] = pd.to_datetime(df_sample['timestamp'], unit='ms')
                
                # Analyze sample
                stats = {
                    'filename': filename,
                    'sample_rows': len(df_sample),
                    'date_range': f"{df_sample['datetime'].min()} to {df_sample['datetime'].max()}",
                    'price_range': f"${df_sample['price'].min():.2f} - ${df_sample['price'].max():.2f}",
                    'avg_trade_size': df_sample['quantity'].mean(),
                    'avg_quote_volume': df_sample['quote_quantity'].mean(),
                    'buyer_maker_ratio': df_sample['is_buyer_maker'].mean(),
                    'data_types': df_sample.dtypes.to_dict(),
                    'missing_values': df_sample.isnull().sum().to_dict()
                }
                
                sample_stats[year] = stats
                
                print(f"   ✅ Date range: {stats['date_range']}")
                print(f"   💰 Price range: {stats['price_range']}")
                print(f"   📊 Avg trade size: {stats['avg_trade_size']:.6f} BTC")
                print(f"   🔄 Buyer maker ratio: {stats['buyer_maker_ratio']:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error processing {filename}: {e}")
                sample_stats[year] = {'error': str(e)}
        
        self.sample_stats = sample_stats
        return sample_stats
    
    def estimate_full_data_characteristics(self):
        """Estimate characteristics of full dataset."""
        print(f"\n📈 Estimating full dataset characteristics...")
        
        # Use sample data to estimate full dataset
        total_files = sum(inv['zip_files'] for inv in self.file_inventory.values())
        total_compressed_gb = sum(inv['compressed_size_mb'] for inv in self.file_inventory.values()) / 1024
        
        # Estimate uncompressed size (typical compression ratio for CSV: 3-5x)
        estimated_uncompressed_gb = total_compressed_gb * 4
        
        # Estimate row count based on sample
        if self.sample_stats:
            sample_file_sizes = []
            sample_row_counts = []
            
            for year, stats in self.sample_stats.items():
                if 'error' not in stats:
                    # Estimate full file row count from sample
                    year_dir = self.file_inventory[year]['directory']
                    sample_filename = stats['filename']
                    file_path = year_dir / sample_filename
                    file_size_mb = file_path.stat().st_size / (1024**2)
                    
                    # Rows per MB estimate
                    rows_per_mb = stats['sample_rows'] / (file_size_mb / 4)  # Assuming 4x compression
                    sample_file_sizes.append(file_size_mb)
                    sample_row_counts.append(rows_per_mb)
            
            if sample_row_counts:
                avg_rows_per_mb = np.mean(sample_row_counts)
                estimated_total_rows = avg_rows_per_mb * total_compressed_gb * 1024 * 4  # 4x for compression
                
                print(f"   📁 Total files: {total_files}")
                print(f"   🗜️  Compressed size: {total_compressed_gb:.2f} GB")
                print(f"   📄 Estimated uncompressed: {estimated_uncompressed_gb:.2f} GB")
                print(f"   📊 Estimated total trades: {estimated_total_rows:,.0f}")
                print(f"   ⚡ Avg trades per file: {estimated_total_rows/total_files:,.0f}")
                
                self.total_estimates = {
                    'total_files': total_files,
                    'compressed_gb': total_compressed_gb,
                    'estimated_uncompressed_gb': estimated_uncompressed_gb,
                    'estimated_total_trades': estimated_total_rows
                }
    
    def check_data_consistency(self):
        """Check for data consistency issues across time periods."""
        print(f"\n🔍 Checking data consistency...")
        
        issues = []
        
        # Check for gaps in data coverage
        expected_months = []
        for year in range(2017, 2025):
            if year == 2017:
                expected_months.extend([f"{year}-{m:02d}" for m in range(8, 13)])  # Aug-Dec
            elif year == 2024:
                expected_months.extend([f"{year}-{m:02d}" for m in range(1, 5)])   # Jan-Apr
            else:
                expected_months.extend([f"{year}-{m:02d}" for m in range(1, 13)])  # Full year
        
        actual_months = []
        for year_data in self.file_inventory.values():
            for filename in year_data['files']:
                if filename.endswith('.zip'):
                    # Extract YYYY-MM from filename
                    parts = filename.replace('.zip', '').split('-')
                    if len(parts) >= 3:
                        month_str = f"{parts[2]}-{parts[3]}"
                        actual_months.append(month_str)
        
        missing_months = set(expected_months) - set(actual_months)
        if missing_months:
            issues.append(f"Missing data for months: {sorted(missing_months)}")
        
        # Check sample data consistency
        price_ranges = {}
        for year, stats in self.sample_stats.items():
            if 'error' not in stats and 'price_range' in stats:
                price_ranges[year] = stats['price_range']
        
        print(f"   ✅ Found data for {len(actual_months)} months")
        if missing_months:
            print(f"   ⚠️  Missing {len(missing_months)} expected months")
        else:
            print(f"   ✅ No missing months detected")
        
        if issues:
            print(f"   🚨 Issues found: {len(issues)}")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print(f"   ✅ No major consistency issues detected")
        
        self.consistency_issues = issues
        return issues
    
    def save_exploration_report(self):
        """Save comprehensive exploration report."""
        report = {
            'exploration_timestamp': datetime.now().isoformat(),
            'file_inventory': {k: {**v, 'directory': str(v['directory'])} 
                             for k, v in self.file_inventory.items()},
            'sample_stats': self.sample_stats,
            'total_estimates': getattr(self, 'total_estimates', {}),
            'consistency_issues': getattr(self, 'consistency_issues', [])
        }
        
        report_path = "tick_data_exploration_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Exploration report saved to: {report_path}")
        return report_path
    
    def run_full_exploration(self):
        """Run complete data exploration pipeline."""
        print("🚀 Starting comprehensive tick data exploration...\n")
        
        # Step 1: Discover structure
        self.discover_data_structure()
        
        # Step 2: Sample quality check
        self.sample_data_quality()
        
        # Step 3: Estimate full characteristics
        self.estimate_full_data_characteristics()
        
        # Step 4: Check consistency
        self.check_data_consistency()
        
        # Step 5: Save report
        report_path = self.save_exploration_report()
        
        print(f"\n✅ Exploration complete! Report saved to {report_path}")
        
        return {
            'inventory': self.file_inventory,
            'sample_stats': self.sample_stats,
            'estimates': getattr(self, 'total_estimates', {}),
            'issues': getattr(self, 'consistency_issues', [])
        }

if __name__ == "__main__":
    explorer = TickDataExplorer()
    results = explorer.run_full_exploration()
    
    print("\n" + "="*60)
    print("🎯 EXPLORATION SUMMARY")
    print("="*60)
    
    if hasattr(explorer, 'total_estimates'):
        est = explorer.total_estimates
        print(f"📊 Estimated total trades: {est['estimated_total_trades']:,.0f}")
        print(f"🗜️  Total compressed size: {est['compressed_gb']:.2f} GB")
        print(f"📄 Estimated uncompressed: {est['estimated_uncompressed_gb']:.2f} GB")
    
    print(f"\n🎉 Ready for next phase: Data processing & feature engineering!") 