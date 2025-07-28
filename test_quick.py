#!/usr/bin/env python3
"""
Quick test for the optimized system
"""

import requests
import time

def test_quick():
    print("🚀 Quick System Test")
    print("=" * 30)
    
    # Test health
    try:
        response = requests.get("http://mira.beo-software.de/health", timeout=5)
        print(f"✅ Health: {response.json()['status']}")
        print(f"📊 vLLM ready: {response.json().get('vllm_ready', False)}")
        print(f"📊 Fallback ready: {response.json().get('fallback_ready', False)}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test with minimal data
    print("\n📄 Testing minimal extraction...")
    try:
        # Create minimal test data
        test_data = "test content for extraction"
        files = {'file': ('test.txt', test_data.encode(), 'text/plain')}
        data = {'extraction_type': 'auto', 'language': 'en'}
        
        start = time.time()
        response = requests.post(
            "http://mira.beo-software.de/extract/file",
            files=files,
            data=data,
            timeout=30  # Shorter timeout
        )
        
        duration = time.time() - start
        print(f"⏱️ Response time: {duration:.2f}s")
        print(f"📡 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Method used: {result.get('extraction_method', 'unknown')}")
            if 'processing_metadata' in result:
                meta = result['processing_metadata']
                print(f"📊 Server time: {meta.get('processing_time', 0):.2f}s")
        else:
            print(f"❌ Error: {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out after 30s")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_quick()