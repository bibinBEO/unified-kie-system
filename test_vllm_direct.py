#!/usr/bin/env python3
"""
Direct test of vLLM processing
"""

import requests
import time
import io
from PIL import Image, ImageDraw

def create_simple_image():
    """Create simple test image"""
    img = Image.new('RGB', (200, 150), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Rechnung", fill='black')
    draw.text((10, 30), "Nr: 12345", fill='black')
    draw.text((10, 50), "100 EUR", fill='black')
    return img

def test_processing():
    print("🔍 Testing Processing Method Detection")
    print("=" * 40)
    
    # Create small test image
    img = create_simple_image()
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    try:
        files = {'file': ('test.png', img_buffer, 'image/png')}
        data = {'extraction_type': 'invoice', 'language': 'de'}
        
        print("📤 Sending request...")
        start = time.time()
        response = requests.post(
            "http://mira.beo-software.de/extract/file",
            files=files,
            data=data,
            timeout=120
        )
        
        duration = time.time() - start
        print(f"⏱️ Total time: {duration:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"📊 Method: {result.get('extraction_method', 'unknown')}")
            
            if 'processing_metadata' in result:
                meta = result['processing_metadata']
                print(f"📊 Server time: {meta.get('processing_time', 0):.2f}s")
                print(f"📊 Method used: {meta.get('method_used', 'unknown')}")
                print(f"📊 vLLM available: {meta.get('vllm_available', False)}")
                print(f"📊 Fallback used: {meta.get('fallback_used', False)}")
            
            # Check actual extracted content
            if 'raw_text' in result and result['raw_text']:
                if result['raw_text'].startswith('Extract all'):
                    print("❌ Still returning prompt instead of extraction")
                else:
                    print("✅ Returning actual extraction")
                    print(f"📝 Preview: {result['raw_text'][:100]}...")
        else:
            print(f"❌ Error {response.status_code}: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_processing()