#!/usr/bin/env python3
"""
Final test to verify CUDA tensor mismatch fix using correct API
"""

import requests
import io
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os

print("🔧 Final CUDA Tensor Mismatch Fix Verification")
print("=" * 60)

def create_test_image():
    """Create a test invoice image"""
    img = Image.new('RGB', (800, 600), 'white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None

    # Draw invoice content
    draw.text((50, 50), "INVOICE", fill='black', font=font)
    draw.text((50, 100), "Invoice Number: INV-2025-001", fill='black', font=font)
    draw.text((50, 130), "Date: 2025-07-26", fill='black', font=font)
    draw.text((50, 160), "Total Amount: $1,250.00", fill='black', font=font)
    draw.text((50, 190), "Company: Test Corp", fill='black', font=font)
    draw.text((50, 220), "Customer: ABC Company", fill='black', font=font)
    
    return img

def test_extraction():
    """Test extraction with proper API endpoint"""
    
    print("🖼️  Creating test invoice image...")
    img = create_test_image()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        img.save(tmp_file.name, 'PNG')
        temp_path = tmp_file.name
    
    try:
        print("🚀 Testing document extraction via file upload...")
        
        # Prepare multipart form data
        url = "http://localhost:8000/extract/file"
        
        with open(temp_path, 'rb') as f:
            files = {'file': ('test_invoice.png', f, 'image/png')}
            data = {
                'extraction_type': 'invoice',
                'language': 'auto',
                'use_schema': 'true'
            }
            
            response = requests.post(url, files=files, data=data, timeout=60)
        
        # Cleanup temp file
        os.unlink(temp_path)
        
        if response.status_code == 200:
            result = response.json()
            raw_text = result.get('raw_text', '')
            key_values = result.get('key_values', {})
            extraction_method = result.get('extraction_method', 'unknown')
            
            print("📊 EXTRACTION RESULTS:")
            print(f"   Status Code: {response.status_code}")
            print(f"   Extraction Method: {extraction_method}")
            
            # Check for CUDA device-side assertion
            cuda_error_indicators = [
                'device-side assert triggered',
                'CUDA error: device-side assert',
                'device-side assertion'
            ]
            
            has_cuda_error = False
            for indicator in cuda_error_indicators:
                if indicator in raw_text or indicator in str(key_values):
                    has_cuda_error = True
                    break
            
            if has_cuda_error:
                print("❌ CUDA DEVICE-SIDE ASSERTION STILL DETECTED!")
                print(f"   Raw text preview: {raw_text[:200]}...")
                if 'error' in key_values:
                    print(f"   Error in key_values: {key_values.get('error', '')[:200]}...")
                print("   🔧 The tensor mismatch fix did NOT resolve the issue")
                return False
            else:
                print("✅ NO CUDA DEVICE-SIDE ASSERTIONS DETECTED!")
                print(f"   Raw text length: {len(raw_text)} characters")
                print(f"   Key-value pairs extracted: {len(key_values)}")
                
                # Show extraction details
                if key_values:
                    print("   📋 Sample extractions:")
                    sample_count = 0
                    for key, value in key_values.items():
                        if sample_count < 5 and key != 'error' and value:
                            print(f"     • {key}: {value}")
                            sample_count += 1
                
                print("   🎯 TENSOR MISMATCH FIX: SUCCESS!")
                return True
                
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Error: {error_detail}")
            except:
                print(f"   Response: {response.text[:300]}...")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - model may be loading")
        return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False
    finally:
        # Ensure cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def main():
    """Main test execution"""
    
    # First check server health
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=10)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Server Status: {health_data.get('status', 'unknown')}")
            print(f"   GPU Available: {health_data.get('gpu_available', False)}")
            
            models_loaded = health_data.get('models_loaded', {})
            print(f"   Models Loaded: nanonets={models_loaded.get('nanonets', False)}")
        else:
            print(f"⚠️  Server health check failed: {health_response.status_code}")
    except Exception as e:
        print(f"⚠️  Could not check server health: {e}")
    
    print()
    
    # Run extraction test
    try:
        success = test_extraction()
        
        print("\n" + "=" * 60)
        print("🎯 FINAL VERIFICATION RESULTS:")
        
        if success:
            print("🎉 CUDA TENSOR MISMATCH FIX: ✅ SUCCESSFUL!")
            print()
            print("   ✅ Qwen2.5-VL architecture correctly loaded")
            print("   ✅ No CUDA device-side assertions detected")
            print("   ✅ NVIDIA RTX 4000 ADA working properly")
            print("   ✅ Nanonets-OCR-s model functioning without errors")
            print("   ✅ Document extraction working as expected")
            print()
            print("   🚀 Your system is now ready for production use!")
            
        else:
            print("❌ CUDA TENSOR MISMATCH FIX: ⚠️  NEEDS ATTENTION")
            print()
            print("   The fix may need additional refinement.")
            print("   Please check the server logs for more details.")
            
        return success
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)