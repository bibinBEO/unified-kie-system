#!/usr/bin/env python3
"""
Quick vLLM performance optimization for invoice processing
"""

import requests
import json

print("🚀 vLLM Performance Optimization Check")
print("=" * 50)

# Check server status
try:
    health = requests.get("http://localhost:8000/health", timeout=5)
    print(f"✅ Server responding: {health.status_code}")
    
    if health.status_code == 200:
        try:
            data = health.json()
            print(f"✅ GPU Available: {data.get('gpu_available', False)}")
            print(f"✅ RAM Usage: {data.get('memory_usage', {}).get('ram_usage_mb', 0):.1f} MB")
            print(f"✅ GPU Memory: {data.get('memory_usage', {}).get('gpu_memory_allocated_mb', 0):.1f} MB")
        except:
            print("⚠️  Health endpoint returned HTML instead of JSON")
            print("   This suggests an internal server error")
    
except Exception as e:
    print(f"❌ Server check failed: {e}")

print("\n🔧 Performance Issues Likely Causes:")
print("1. GPU Memory Saturation (vLLM + multiple models)")
print("2. Model loading every request (no model caching)")
print("3. Large input image processing") 
print("4. Multiple concurrent requests")

print("\n🎯 Quick Fixes Needed:")
print("• Optimize GPU memory allocation")
print("• Implement request queuing") 
print("• Add image preprocessing/resizing")
print("• Configure vLLM model caching")
print("• Set processing timeouts")

print(f"\n📋 Status: vLLM needs performance tuning for production speed")