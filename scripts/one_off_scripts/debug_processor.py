#!/usr/bin/env python3
"""
Debug Tick Processor
==================

Simplified version to debug the pandas resampling issue.
"""

import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

def debug_tick_processing():
    """Debug the tick processing step by step."""
    
    print("🔍 Debugging tick processing...")
    
    # Load data
    base_path = Path("historical_btc_trades")
    year = '2024'
    month_file = 'BTCUSDT-trades-2024-01.zip'
    
    year_dir = base_path / f"{year}_data"
    file_path = year_dir / month_file
    
    print(f"📁 Loading {file_path}...")
    
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        csv_name = month_file.replace('.zip', '.csv')
        
        with zip_ref.open(csv_name) as csv_file:
            # Read sample data only
            df_ticks = pd.read_csv(
                csv_file,
                nrows=10000,  # Smaller sample for debugging
                names=['trade_id', 'price', 'quantity', 'quote_quantity', 
                      'timestamp', 'is_buyer_maker', 'is_best_match']
            )
    
    print(f"📊 Loaded {len(df_ticks)} trades")
    print(f"📋 Columns: {df_ticks.columns.tolist()}")
    print(f"📋 Dtypes: {df_ticks.dtypes}")
    
    # Convert timestamp
    print("\n🕐 Converting timestamps...")
    df_ticks['datetime'] = pd.to_datetime(df_ticks['timestamp'], unit='ms')
    df_ticks = df_ticks.set_index('datetime').sort_index()
    
    print(f"✅ Index type: {type(df_ticks.index)}")
    print(f"✅ Index range: {df_ticks.index.min()} to {df_ticks.index.max()}")
    
    # Test basic resampling
    print("\n📊 Testing basic OHLCV resampling...")
    try:
        ohlcv = df_ticks['price'].resample('1T').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last'
        }).dropna()
        print(f"✅ OHLCV resampling successful: {len(ohlcv)} bars")
    except Exception as e:
        print(f"❌ OHLCV resampling failed: {e}")
        return
    
    # Test volume resampling  
    print("\n📊 Testing volume resampling...")
    try:
        volume_agg = df_ticks.resample('1T').agg({
            'quantity': 'sum',
            'quote_quantity': 'sum',
            'trade_id': 'count'
        })
        volume_agg.columns = ['volume_btc', 'volume_usdt', 'trade_count']
        print(f"✅ Volume resampling successful: {len(volume_agg)} bars")
    except Exception as e:
        print(f"❌ Volume resampling failed: {e}")
        return
    
    # Test simple microstructure features
    print("\n🔬 Testing simple microstructure features...")
    try:
        resampled = df_ticks.resample('1T')
        
        # Try simple features first
        buyer_ratio = resampled['is_buyer_maker'].apply(lambda x: (~x).mean())
        print(f"✅ Buyer ratio calculation successful: {len(buyer_ratio)} values")
        
        avg_trade_size = resampled['quantity'].mean()
        print(f"✅ Avg trade size calculation successful: {len(avg_trade_size)} values")
        
    except Exception as e:
        print(f"❌ Simple microstructure failed: {e}")
        return
    
    # Test complex microstructure features
    print("\n🔬 Testing complex microstructure features...")
    try:
        def test_large_trade_ratio(group):
            if len(group) == 0:
                return 0
            q80 = group['quote_quantity'].quantile(0.8)
            return (group['quote_quantity'] > q80).mean()
        
        large_trade_ratio = resampled.apply(test_large_trade_ratio)
        print(f"✅ Large trade ratio calculation successful: {len(large_trade_ratio)} values")
        
    except Exception as e:
        print(f"❌ Complex microstructure failed: {e}")
        print(f"Error details: {type(e).__name__}: {e}")
        return
    
    print("\n🎉 All tests passed! The processor should work.")

if __name__ == "__main__":
    debug_tick_processing() 