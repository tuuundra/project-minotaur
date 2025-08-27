#!/usr/bin/env python3
"""
Find the Bug
============

Isolate which microstructure feature is causing the pandas error.
"""

import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

def test_each_feature():
    """Test each microstructure feature individually."""
    
    print("🔍 Testing each feature individually...")
    
    # Load small sample
    base_path = Path("historical_btc_trades")
    year = '2024'
    month_file = 'BTCUSDT-trades-2024-01.zip'
    
    year_dir = base_path / f"{year}_data"
    file_path = year_dir / month_file
    
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        csv_name = month_file.replace('.zip', '.csv')
        
        with zip_ref.open(csv_name) as csv_file:
            df_ticks = pd.read_csv(
                csv_file,
                nrows=100000,  # Use same chunk size
                names=['trade_id', 'price', 'quantity', 'quote_quantity', 
                      'timestamp', 'is_buyer_maker', 'is_best_match']
            )
    
    print(f"📊 Loaded {len(df_ticks)} trades")
    
    # Convert and set index
    df_ticks['datetime'] = pd.to_datetime(df_ticks['timestamp'], unit='ms')
    df_ticks = df_ticks.set_index('datetime').sort_index()
    
    # Create resampled object
    resampled = df_ticks.resample('1min')
    
    # Test features one by one
    features_to_test = [
        ('buyer_initiated_ratio', lambda: resampled['is_buyer_maker'].apply(lambda x: (~x).mean())),
        ('avg_trade_size_btc', lambda: resampled['quantity'].mean()),
        ('avg_trade_size_usdt', lambda: resampled['quote_quantity'].mean()),
        ('trades_per_minute', lambda: resampled['trade_id'].count()),
    ]
    
    def test_large_trade_ratio():
        def calc_large_trade_ratio(group):
            if len(group) == 0:
                return 0
            q80 = group['quote_quantity'].quantile(0.8)
            return (group['quote_quantity'] > q80).mean()
        return resampled.apply(calc_large_trade_ratio)
    
    def test_vwap():
        def calc_vwap(group):
            if len(group) == 0:
                return np.nan
            return np.average(group['price'], weights=group['quantity'])
        return resampled.apply(calc_vwap)
    
    def test_volume_imbalance():
        def calc_volume_imbalance(group):
            if len(group) == 0:
                return 0
            buyer_vol = group[~group['is_buyer_maker']]['quote_quantity'].sum()
            seller_vol = group[group['is_buyer_maker']]['quote_quantity'].sum()
            total_vol = buyer_vol + seller_vol
            if total_vol == 0:
                return 0
            return (buyer_vol - seller_vol) / total_vol
        return resampled.apply(calc_volume_imbalance)
    
    features_to_test.extend([
        ('large_trade_ratio', test_large_trade_ratio),
        ('vwap', test_vwap),
        ('volume_imbalance', test_volume_imbalance),
    ])
    
    results = {}
    
    for feature_name, feature_func in features_to_test:
        try:
            print(f"\n🧪 Testing {feature_name}...")
            result = feature_func()
            print(f"   ✅ Success: {len(result)} values")
            results[feature_name] = result
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            print(f"   Error type: {type(e).__name__}")
            return feature_name, e
    
    print(f"\n🎉 All individual features passed!")
    
    # Now test combining them
    print(f"\n🔗 Testing DataFrame creation...")
    try:
        combined_df = pd.DataFrame(results)
        print(f"   ✅ DataFrame creation successful: {combined_df.shape}")
        print(f"   Columns: {list(combined_df.columns)}")
        return None, None
    except Exception as e:
        print(f"   ❌ DataFrame creation FAILED: {e}")
        return "DataFrame creation", e

if __name__ == "__main__":
    failed_feature, error = test_each_feature()
    if failed_feature:
        print(f"\n🚨 Bug found in: {failed_feature}")
        print(f"Error: {error}")
    else:
        print(f"\n🎉 No bugs found - the issue might be elsewhere!") 