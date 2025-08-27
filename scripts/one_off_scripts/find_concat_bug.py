#!/usr/bin/env python3
"""
Find Concat Bug
===============

Debug the pd.concat operation in the tick processor.
"""

import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

def debug_concat_operation():
    """Debug the exact pd.concat operation that's failing."""
    
    print("🔍 Debugging pd.concat operation...")
    
    # Load sample data
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
                nrows=100000,
                names=['trade_id', 'price', 'quantity', 'quote_quantity', 
                      'timestamp', 'is_buyer_maker', 'is_best_match']
            )
    
    print(f"📊 Loaded {len(df_ticks)} trades")
    
    # Process exactly like the main function
    df_ticks['datetime'] = pd.to_datetime(df_ticks['timestamp'], unit='ms')
    df_ticks = df_ticks.set_index('datetime').sort_index()
    
    print(f"✅ Index set: {type(df_ticks.index)}")
    
    # 1. OHLCV aggregation
    print(f"\n📊 Creating OHLCV...")
    try:
        ohlcv = df_ticks['price'].resample('1min').agg({
            'open': 'first',
            'high': 'max', 
            'low': 'min',
            'close': 'last'
        }).dropna()
        print(f"   ✅ OHLCV shape: {ohlcv.shape}")
        print(f"   ✅ OHLCV index type: {type(ohlcv.index)}")
        print(f"   ✅ OHLCV columns: {list(ohlcv.columns)}")
    except Exception as e:
        print(f"   ❌ OHLCV failed: {e}")
        return
    
    # 2. Volume aggregation
    print(f"\n📊 Creating volume aggregation...")
    try:
        volume_agg = df_ticks.resample('1min').agg({
            'quantity': 'sum',
            'quote_quantity': 'sum',
            'trade_id': 'count'
        })
        volume_agg.columns = ['volume_btc', 'volume_usdt', 'trade_count']
        print(f"   ✅ Volume shape: {volume_agg.shape}")
        print(f"   ✅ Volume index type: {type(volume_agg.index)}")
        print(f"   ✅ Volume columns: {list(volume_agg.columns)}")
    except Exception as e:
        print(f"   ❌ Volume failed: {e}")
        return
    
    # 3. Microstructure features
    print(f"\n🔬 Creating microstructure features...")
    try:
        resampled = df_ticks.resample('1min')
        
        features = {}
        features['buyer_initiated_ratio'] = resampled['is_buyer_maker'].apply(lambda x: (~x).mean())
        features['avg_trade_size_btc'] = resampled['quantity'].mean()
        
        micro_features = pd.DataFrame(features)
        print(f"   ✅ Micro shape: {micro_features.shape}")
        print(f"   ✅ Micro index type: {type(micro_features.index)}")
        print(f"   ✅ Micro columns: {list(micro_features.columns)}")
    except Exception as e:
        print(f"   ❌ Microstructure failed: {e}")
        return
    
    # 4. Check index compatibility
    print(f"\n🔗 Checking index compatibility...")
    print(f"   OHLCV index: {ohlcv.index.min()} to {ohlcv.index.max()} ({len(ohlcv)} entries)")
    print(f"   Volume index: {volume_agg.index.min()} to {volume_agg.index.max()} ({len(volume_agg)} entries)")
    print(f"   Micro index: {micro_features.index.min()} to {micro_features.index.max()} ({len(micro_features)} entries)")
    
    # 5. Try the concat operation
    print(f"\n🔗 Testing pd.concat...")
    try:
        # Test individual combinations first
        print("   Testing OHLCV + Volume...")
        test1 = pd.concat([ohlcv, volume_agg], axis=1)
        print(f"      ✅ Success: {test1.shape}")
        
        print("   Testing OHLCV + Micro...")
        test2 = pd.concat([ohlcv, micro_features], axis=1)
        print(f"      ✅ Success: {test2.shape}")
        
        print("   Testing Volume + Micro...")
        test3 = pd.concat([volume_agg, micro_features], axis=1)
        print(f"      ✅ Success: {test3.shape}")
        
        print("   Testing All Three...")
        bars = pd.concat([ohlcv, volume_agg, micro_features], axis=1)
        print(f"      ✅ Success: {bars.shape}")
        print(f"      ✅ Final columns: {list(bars.columns)}")
        
    except Exception as e:
        print(f"   ❌ pd.concat failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n🎉 All concat operations successful!")

if __name__ == "__main__":
    debug_concat_operation() 