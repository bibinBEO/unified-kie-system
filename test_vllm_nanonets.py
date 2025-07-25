#!/usr/bin/env python3

import asyncio
import json
from PIL import Image
from extractors.nanonets_vllm_extractor import NanoNetsVLLMExtractor

async def test_vllm_extraction():
    """Test vLLM-based extraction"""
    print("🚀 Testing vLLM-based NanoNets Extraction...")
    
    config = {}
    extractor = NanoNetsVLLMExtractor(config)
    
    try:
        await extractor.initialize()
        
        if not extractor.model:
            print("⚠️  vLLM engine failed to load - cannot test extraction")
            return
        
        image_path = "uploads/test_invoice_proper.png"
        print(f"📸 Loading test image: {image_path}")
        
        try:
            image = Image.open(image_path)
            print(f"✅ Image loaded: {image.size} ({image.mode})")
        except Exception as e:
            print(f"⚠️  Failed to load image: {e}")
            return
        
        result = await extractor.extract(image, language="auto")
        
        print("\n" + "="*50)
        print("📊 vLLM EXTRACTION RESULTS")
        print("="*50)
        
        print(f"Extraction Method: {result.get('extraction_method', 'unknown')}")
        print(f"Timestamp: {result.get('timestamp', 'unknown')}")
        
        key_values = result.get('key_values', {})
        if 'error' in key_values:
            print(f"\n⚠️  Error detected: {key_values['error']}")
        else:
            print("\n✅ No errors detected - vLLM extraction appears successful!")
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
    import sys
    sys.stderr = sys.stdout
    result = await test_vllm_extraction()
    
    if result:
        output_file = "/home/bibin.wilson/unified-kie-system/test_vllm_result.json"
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
