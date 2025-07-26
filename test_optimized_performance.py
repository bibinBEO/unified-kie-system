#!/usr/bin/env python3
"""
Quick performance test for the optimized vLLM system
"""

import requests
import time
import io
from PIL import Image, ImageDraw, ImageFont

def create_test_invoice():
    """Create a simple test invoice image"""
    image = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw invoice content
    draw.text((50, 50), "INVOICE", fill='black', font=font)
    draw.text((50, 100), "Invoice #: INV-2025-001", fill='black', font=font)
    draw.text((50, 130), "Date: January 26, 2025", fill='black', font=font)
    draw.text((50, 180), "Bill To: Test Company Ltd.", fill='black', font=font)
    draw.text((50, 230), "Description: Document Processing Service", fill='black', font=font)
    draw.text((50, 260), "Amount: $100.00", fill='black', font=font)
    draw.text((50, 320), "Total: $100.00", fill='black', font=font)
    
    return image

def test_performance():
    """Test the optimized system performance"""
    
    print("🚀 Testing Optimized vLLM Performance")
    print("=" * 50)
    
    # Test health endpoint speed
    start = time.time()
    health_response = requests.get("http://mira.beo-software.de/health")
    health_time = time.time() - start
    
    print(f"Health check: {health_time:.3f}s")
    print(f"Status: {health_response.json().get('status')}")
    
    # Test document processing speed
    test_image = create_test_invoice()
    
    # Convert image to bytes
    img_buffer = io.BytesIO()
    test_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    print(f"\n📄 Testing document extraction...")
    start = time.time()
    
    try:
        files = {'file': ('test_invoice.png', img_buffer, 'image/png')}
        data = {
            'extraction_type': 'invoice',
            'language': 'en'
        }
        
        response = requests.post(
            "http://mira.beo-software.de/extract/file",
            files=files,
            data=data,
            timeout=60
        )
        
        processing_time = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Extraction completed in {processing_time:.2f}s")
            print(f"Method: {result.get('extraction_method', 'unknown')}")
            
            if 'processing_metadata' in result:
                metadata = result['processing_metadata']
                print(f"Server processing time: {metadata.get('processing_time', 0):.2f}s")
                print(f"File size: {metadata.get('file_size_mb', 0):.1f}MB")
            
            # Check for extracted content
            if 'raw_text' in result and result['raw_text']:
                text_preview = result['raw_text'][:200] + "..." if len(result['raw_text']) > 200 else result['raw_text']
                print(f"Extracted text preview: {text_preview}")
            
            if 'key_values' in result and result['key_values']:
                print(f"Key-value pairs extracted: {len(result['key_values'])}")
            
            print("\n🎯 Performance Assessment:")
            if processing_time < 10:
                print("🟢 EXCELLENT performance (< 10s)")
            elif processing_time < 30:
                print("🟡 GOOD performance (< 30s)")
            else:
                print("🔴 NEEDS IMPROVEMENT (> 30s)")
            
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text[:300]}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print(f"\n🏁 Total test time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    test_performance()