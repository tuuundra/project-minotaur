#!/usr/bin/env python3
"""
Test Single Month Processing
===========================

Test the tick data processor on a single month to verify fixes.
"""

from tick_data_processor import TickDataProcessor
import pandas as pd

def test_single_month():
    """Test processing one month of data."""
    
    print("🧪 Testing single month processing...")
    
    processor = TickDataProcessor()
    
    # Test with 2024 data (smallest files)
    year = '2024'
    month_file = 'BTCUSDT-trades-2024-01.zip'
    
    try:
        bars_1m, bars_5m, stats = processor.process_month_file(year, month_file)
        
        if bars_1m is not None:
            print(f"✅ SUCCESS!")
            print(f"📊 1-minute bars: {len(bars_1m)}")
            print(f"📊 5-minute bars: {len(bars_5m)}")
            print(f"📊 Features in 1m: {list(bars_1m.columns)}")
            print(f"\n📈 Sample 1m data:")
            print(bars_1m.head())
            
            return True
        else:
            print(f"❌ FAILED: {stats}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_single_month()
    if success:
        print("\n🎉 Test passed! Ready to process full dataset.")
    else:
        print("\n🚨 Test failed! Need to debug further.") 