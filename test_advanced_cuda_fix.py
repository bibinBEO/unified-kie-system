#!/usr/bin/env python3

import asyncio
import json
import sys
import os
from PIL import Image
from extractors.nanonets_extractor import NanoNetsExtractor

async def test_advanced_cuda_fixes():
    """Test advanced CUDA IndexKernel fixes"""
    print("🚀 Testing Advanced CUDA IndexKernel Fixes...")
    
    # Set additional CUDA debugging environment variables
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initialize extractor with GPU mode
    config = {}
    extractor = NanoNetsExtractor(config)
    
    try:
        # Initialize the extractor
        print("🔧 Initializing NanoNets extractor with advanced fixes...")
        await extractor.initialize()
        
        if extractor.model is None:
            print("⚠️  Model failed to load - cannot test CUDA fixes")
            return
        
        # Load test image
        image_path = "/home/bibin.wilson/unified-kie-system/uploads/test_invoice_proper.png"
        print(f"📸 Loading test image: {image_path}")
        
        try:
            image = Image.open(image_path)
            print(f"✅ Image loaded: {image.size} ({image.mode})")
        except Exception as e:
            print(f"⚠️  Failed to load image: {e}")
            return
        
        # Test extraction with advanced CUDA fixes
        print("\n" + "="*60)
        print("🔧 TESTING ADVANCED CUDA INDEXKERNEL FIXES")
        print("="*60)
        
        result = await extractor.extract(image, language="auto")
        
        # Display results
        print("\n" + "="*50)
        print("📊 ADVANCED CUDA FIX RESULTS")
        print("="*50)
        
        print(f"Extraction Method: {result.get('extraction_method', 'unknown')}")
        print(f"Timestamp: {result.get('timestamp', 'unknown')}")
        
        # Check for errors
        key_values = result.get('key_values', {})
        if 'error' in key_values:
            print(f"\n⚠️  Error detected: {key_values['error']}")
            print(f"Error type: {key_values.get('error_type', 'unknown')}")
            
            # Check if this is a CUDA IndexKernel error
            error_msg = key_values['error'].lower()
            if "device-side assert" in error_msg or "indexkernel" in error_msg:
                print(f"❌ CUDA IndexKernel assertion still persists")
                print(f"🔧 Additional debugging needed")
            else:
                print(f"✅ Different error type - CUDA IndexKernel issue may be resolved")
        else:
            print(f"\n✅ No errors detected - CUDA IndexKernel fixes appear successful!")
            
            print("\n📝 Raw Response (first 300 chars):")
            print("-" * 30)
            raw_text = result.get('raw_text', '')
            if len(raw_text) > 300:
                print(raw_text[:300] + "...")
            else:
                print(raw_text)
        
        return result
        
    except Exception as e:
        print(f"⚠️  Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main test function"""
    result = await test_advanced_cuda_fixes()
    
    if result:
        # Save result to file
        output_file = "/home/bibin.wilson/unified-kie-system/test_advanced_cuda_result.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_file}")
        except Exception as e:
            print(f"⚠️  Failed to save results: {e}")
    else:
        print("\n❌ Test failed - no results to save")

if __name__ == "__main__":
    asyncio.run(main())