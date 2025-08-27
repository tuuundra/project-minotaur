#!/usr/bin/env python3
"""
Test Chunked Processing
======================

Test the chunked tick data processor.
"""

from tick_data_processor import TickDataProcessor

def test_chunked_processing():
    """Test chunked processing with a small chunk size."""
    
    print("🧪 Testing chunked processing...")
    
    processor = TickDataProcessor()
    
    # Test with 2024 data and small chunk size for testing
    year = '2024'
    month_file = 'BTCUSDT-trades-2024-01.zip'
    
    try:
        # Use very small chunk size for testing
        bars_1m, bars_5m, stats = processor.process_month_file_chunked(
            year, month_file, chunk_size=100000  # Much smaller chunks
        )
        
        if bars_1m is not None:
            print(f"✅ SUCCESS!")
            print(f"📊 Stats: {stats}")
            print(f"📊 1-minute bars: {len(bars_1m)}")
            print(f"📊 5-minute bars: {len(bars_5m)}")
            print(f"📊 Features in 1m: {list(bars_1m.columns)}")
            
            return True
        else:
            print(f"❌ FAILED: {stats}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chunked_processing()
    if success:
        print("\n🎉 Chunked processing works! Ready for full dataset.")
    else:
        print("\n🚨 Chunked processing failed! Need to debug further.") 