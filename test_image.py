#!/usr/bin/env python3
"""
Test with actual image
"""

import requests
import time
import io
from PIL import Image, ImageDraw

def create_simple_image():
    """Create simple test image"""
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "TEST INVOICE", fill='black')
    draw.text((20, 50), "Invoice #: 12345", fill='black')
    draw.text((20, 80), "Amount: $100.00", fill='black')
    return img

def test_with_image():
    print("🖼️ Testing with Image")
    print("=" * 25)
    
    # Create test image
    img = create_simple_image()
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    try:
        files = {'file': ('test.png', img_buffer, 'image/png')}
        data = {'extraction_type': 'invoice', 'language': 'en'}
        
        start = time.time()
        response = requests.post(
            "http://mira.beo-software.de/extract/file",
            files=files,
            data=data,
            timeout=45
        )
        
        duration = time.time() - start
        print(f"⏱️ Response time: {duration:.2f}s")
        print(f"📡 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Method: {result.get('extraction_method', 'unknown')}")
            
            if 'processing_metadata' in result:
                meta = result['processing_metadata']
                print(f"📊 Server time: {meta.get('processing_time', 0):.2f}s")
                print(f"📊 Method used: {meta.get('method_used', 'unknown')}")
                print(f"📊 Fallback used: {meta.get('fallback_used', False)}")
            
            if result.get('raw_text'):
                preview = result['raw_text'][:100]
                print(f"📝 Text preview: {preview}...")
                
        else:
            print(f"❌ Error: {response.text[:300]}")
            
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_with_image()