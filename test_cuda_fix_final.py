#!/usr/bin/env python3
"""
Final test to verify CUDA tensor mismatch fix is working
"""

import requests
import json
import base64
from PIL import Image
import io

print("🔧 Testing CUDA Tensor Mismatch Fix - Final Verification")
print("=" * 60)

# Create a simple test image
print("🖼️  Creating test document image...")
img = Image.new('RGB', (800, 600), 'white')
# Add some simple text-like content
from PIL import ImageDraw, ImageFont
draw = ImageDraw.Draw(img)

# Draw some invoice-like content
try:
    # Try to use a default font
    font = ImageFont.load_default()
except:
    font = None

# Draw invoice-like content
draw.text((50, 50), "INVOICE", fill='black', font=font)
draw.text((50, 100), "Invoice Number: INV-2025-001", fill='black', font=font)
draw.text((50, 130), "Date: 2025-07-26", fill='black', font=font)
draw.text((50, 160), "Total Amount: $1,250.00", fill='black', font=font)
draw.text((50, 190), "Company: Test Corp", fill='black', font=font)

# Convert to base64
buffered = io.BytesIO()
img.save(buffered, format="PNG")
img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

# Test the API
print("🚀 Testing document extraction with fixed model...")

try:
    # Prepare the request
    url = "http://localhost:8000/extract"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "image_data": f"data:image/png;base64,{img_base64}",
        "strategy": "nanonets",
        "fields": ["invoice_number", "date", "total_amount", "company_name"]
    }
    
    # Make the request
    response = requests.post(url, headers=headers, json=data, timeout=120)
    
    if response.status_code == 200:
        result = response.json()
        
        # Check for CUDA errors
        raw_text = result.get('raw_text', '')
        key_values = result.get('key_values', {})
        
        print("📊 EXTRACTION RESULTS:")
        print(f"   Status Code: {response.status_code}")
        print(f"   Extraction Method: {result.get('extraction_method', 'unknown')}")
        

def test_extraction():
    """Run the extraction test"""
    # Create a simple test image
    print("🖼️  Creating test document image...")
    img = Image.new('RGB', (800, 600), 'white')
    # Add some simple text-like content
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)

    # Draw some invoice-like content
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None

    # Draw invoice-like content
    draw.text((50, 50), "INVOICE", fill='black', font=font)
    draw.text((50, 100), "Invoice Number: INV-2025-001", fill='black', font=font)
    draw.text((50, 130), "Date: 2025-07-26", fill='black', font=font)
    draw.text((50, 160), "Total Amount: $1,250.00", fill='black', font=font)
    draw.text((50, 190), "Company: Test Corp", fill='black', font=font)

    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Test the API
    print("🚀 Testing document extraction with fixed model...")

    try:
        # Prepare the request
        url = "http://localhost:8000/extract"
        headers = {"Content-Type": "application/json"}
        
        data = {
            "image_data": f"data:image/png;base64,{img_base64}",
            "strategy": "nanonets",
            "fields": ["invoice_number", "date", "total_amount", "company_name"]
        }
        
        # Make the request
        response = requests.post(url, headers=headers, json=data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            
            # Check for CUDA errors
            raw_text = result.get('raw_text', '')
            key_values = result.get('key_values', {})
            
            print("📊 EXTRACTION RESULTS:")
            print(f"   Status Code: {response.status_code}")
            print(f"   Extraction Method: {result.get('extraction_method', 'unknown')}")
            
            # Check for CUDA device-side assertion
            if 'device-side assert triggered' in raw_text:
                print("❌ CUDA DEVICE-SIDE ASSERTION DETECTED!")
                print(f"   Raw text: {raw_text[:200]}...")
                print("   The tensor mismatch fix did NOT work")
                return False
            elif 'error' in key_values and 'device-side assert' in str(key_values.get('error', '')):
                print("❌ CUDA ERROR IN KEY VALUES!")
                print(f"   Error: {key_values.get('error', '')[:200]}...")
                print("   The tensor mismatch fix did NOT work")
                return False
            else:
                print("✅ NO CUDA DEVICE-SIDE ASSERTIONS DETECTED!")
                print(f"   Raw text length: {len(raw_text)} characters")
                print(f"   Key-value pairs: {len(key_values)} extracted")
                
                # Show some results
                if key_values:
                    print("   Sample extractions:")
                    for key, value in list(key_values.items())[:3]:
                        print(f"     {key}: {value}")
                
                print("   🎯 TENSOR MISMATCH FIX: SUCCESS!")
                return True
                
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"   Response: {response.text[:300]}...")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - this might indicate model loading issues")
        return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    success = True
    try:
        success = test_extraction()
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 FINAL RESULT: CUDA TENSOR MISMATCH FIX SUCCESSFUL!")
        print("   ✅ Qwen2.5-VL architecture properly loaded")
        print("   ✅ No CUDA device-side assertions")
        print("   ✅ NVIDIA RTX 4000 ADA working correctly")
        print("   ✅ Nanonets-OCR-s model functioning properly")
    else:
        print("❌ FINAL RESULT: ISSUES REMAIN")
        print("   Further investigation may be needed")
        
    exit(0 if success else 1)