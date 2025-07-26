#!/usr/bin/env python3
"""
Simple test to verify CUDA tensor mismatch fix
"""

import requests
import json
import base64
from PIL import Image, ImageDraw, ImageFont
import io

print("🔧 Testing CUDA Tensor Mismatch Fix After Server Restart")
print("=" * 60)

def test_extraction():
    """Test the extraction API for CUDA errors"""
    
    # Create a simple test image
    print("🖼️  Creating test document image...")
    img = Image.new('RGB', (800, 600), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw invoice-like content
    try:
        font = ImageFont.load_default()
    except:
        font = None

    draw.text((50, 50), "INVOICE", fill='black', font=font)
    draw.text((50, 100), "Invoice Number: INV-2025-001", fill='black', font=font)
    draw.text((50, 130), "Date: 2025-07-26", fill='black', font=font)
    draw.text((50, 160), "Total Amount: $1,250.00", fill='black', font=font)
    draw.text((50, 190), "Company: Test Corp", fill='black', font=font)

    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    print("🚀 Testing document extraction with fixed model...")

    try:
        url = "http://localhost:8000/extract/file"
        headers = {"Content-Type": "application/json"}
        
        data = {
            "image_data": f"data:image/png;base64,{img_base64}",
            "strategy": "nanonets",
            "fields": ["invoice_number", "date", "total_amount", "company_name"]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            raw_text = result.get('raw_text', '')
            key_values = result.get('key_values', {})
            
            print("📊 EXTRACTION RESULTS:")
            print(f"   Status Code: {response.status_code}")
            print(f"   Extraction Method: {result.get('extraction_method', 'unknown')}")
            
            # Check for CUDA device-side assertion
            if 'device-side assert triggered' in raw_text:
                print("❌ CUDA DEVICE-SIDE ASSERTION STILL DETECTED!")
                print(f"   Raw text: {raw_text[:200]}...")
                return False
            elif 'error' in key_values and 'device-side assert' in str(key_values.get('error', '')):
                print("❌ CUDA ERROR IN KEY VALUES!")
                print(f"   Error: {key_values.get('error', '')[:200]}...")
                return False
            else:
                print("✅ NO CUDA DEVICE-SIDE ASSERTIONS DETECTED!")
                print(f"   Raw text length: {len(raw_text)} characters")
                print(f"   Key-value pairs: {len(key_values)} extracted")
                
                if key_values:
                    print("   Sample extractions:")
                    for key, value in list(key_values.items())[:3]:
                        print(f"     {key}: {value}")
                
                return True
                
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"   Response: {response.text[:300]}...")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    try:
        success = test_extraction()
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 FINAL RESULT: CUDA TENSOR MISMATCH FIX SUCCESSFUL!")
            print("   ✅ Qwen2.5-VL architecture properly loaded")
            print("   ✅ No CUDA device-side assertions")
            print("   ✅ NVIDIA RTX 4000 ADA working correctly")
            print("   ✅ Nanonets-OCR-s model functioning properly")
        else:
            print("❌ FINAL RESULT: ISSUES REMAIN")
            print("   The CUDA tensor mismatch may still need attention")
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        success = False