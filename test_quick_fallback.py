#!/usr/bin/env python3
"""
Quick test to check fallback performance while vLLM times out
"""

import requests
import time
import io
from PIL import Image, ImageDraw

def test_fallback():
    print("🔍 Testing Fallback Performance")
    print("=" * 35)
    
    # Create simple image
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "Rechnung Nr: 2025-001", fill='black')
    draw.text((20, 50), "Betrag: 150,00 EUR", fill='black')
    draw.text((20, 80), "Firma: Test GmbH", fill='black')
    
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    try:
        files = {'file': ('test.png', img_buffer, 'image/png')}
        data = {'extraction_type': 'invoice', 'language': 'de'}
        
        start = time.time()
        response = requests.post(
            "http://mira.beo-software.de/extract/file",
            files=files,
            data=data,
            timeout=60  # Wait longer to see if it eventually falls back
        )
        
        duration = time.time() - start
        print(f"⏱️ Response time: {duration:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            
            if 'processing_metadata' in result:
                meta = result['processing_metadata']
                print(f"📊 Method used: {meta.get('method_used', 'unknown')}")
                print(f"📊 vLLM available: {meta.get('vllm_available', False)}")
                print(f"📊 Fallback used: {meta.get('fallback_used', False)}")
                print(f"📊 Server time: {meta.get('processing_time', 0):.2f}s")
            
            # Quick extraction check
            if 'raw_text' in result:
                text = result['raw_text']
                if 'Rechnung' in text or 'EUR' in text:
                    print("✅ German extraction working")
                else:
                    print(f"📝 Extracted: {text[:100]}")
        else:
            print(f"❌ Failed: {response.status_code}")
            print(response.text[:200])
            
    except requests.exceptions.Timeout:
        print("⏰ Still timing out after 60s")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_fallback()