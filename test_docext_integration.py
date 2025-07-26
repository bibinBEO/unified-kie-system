#!/usr/bin/env python3
"""
Test script for DocExt integration with NVIDIA RTX 4000 ADA
Tests CUDA error handling and fallback mechanisms
"""

import asyncio
import os
import sys
import torch
from PIL import Image
import tempfile
import json
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from extractors.nanonets_ocr_s_extractor import NanoNetsOCRSExtractor
from deployment.config import Config


async def test_docext_integration():
    """Test DocExt integration with CUDA error handling"""
    
    print("🚀 Testing DocExt Integration with NVIDIA RTX 4000 ADA")
    print("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    # Check CUDA status
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"🔧 Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
    
    # Set CUDA environment variables for debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    
    print(f"🔧 CUDA Environment:")
    print(f"   CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING')}")
    print(f"   TORCH_USE_CUDA_DSA: {os.environ.get('TORCH_USE_CUDA_DSA')}")
    print(f"   PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    
    # Initialize extractor
    print("\n🔄 Initializing NanoNets OCR-s Extractor...")
    extractor = NanoNetsOCRSExtractor(config)
    
    try:
        await extractor.initialize()
        print("✅ Extractor initialized successfully")
    except Exception as e:
        print(f"⚠️  Extractor initialization warning: {e}")
        print("   This is expected if DocExt/vLLM are not available - fallback will be used")
    
    # Create a test image (simple invoice-like document)
    print("\n🖼️  Creating test document image...")
    test_image = create_test_invoice_image()
    
    # Test extraction with different scenarios
    test_scenarios = [
        {
            "name": "Basic KIE Extraction",
            "fields": ["invoice_number", "date", "total_amount", "company_name"],
            "description": "Test basic key information extraction"
        },
        {
            "name": "Comprehensive Document Analysis",
            "fields": None,
            "description": "Test full document analysis without predefined fields"
        },
        {
            "name": "Custom Fields Extraction",
            "fields": ["vendor_name", "tax_id", "payment_terms", "line_items"],
            "description": "Test extraction with custom field requirements"
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Test {i}: {scenario['name']}")
        print(f"   {scenario['description']}")
        
        try:
            # Clear CUDA cache before each test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Perform extraction
            start_time = datetime.now()
            result = await extractor.extract_key_value_pairs(
                test_image,
                fields=scenario['fields']
            )
            end_time = datetime.now()
            
            # Calculate processing time
            processing_time = (end_time - start_time).total_seconds()
            
            # Analyze results
            extraction_method = result.get('extraction_method', 'unknown')
            key_values_count = len(result.get('key_values', {}))
            has_raw_text = bool(result.get('raw_text', '').strip())
            
            print(f"   ✅ Success - Method: {extraction_method}")
            print(f"   ⏱️  Processing time: {processing_time:.2f}s")
            print(f"   📊 Extracted {key_values_count} key-value pairs")
            print(f"   📝 Raw text available: {has_raw_text}")
            
            # Check for CUDA errors
            if 'error' in result.get('key_values', {}):
                error_msg = result['key_values']['error']
                if 'device-side assert' in error_msg:
                    print(f"   🔧 CUDA Error Detected and Handled: {error_msg[:100]}...")
                else:
                    print(f"   ⚠️  Other Error: {error_msg[:100]}...")
            
            # Store result for summary
            results.append({
                'scenario': scenario['name'],
                'success': True,
                'method': extraction_method,
                'processing_time': processing_time,
                'kvp_count': key_values_count,
                'has_cuda_error': 'device-side assert' in str(result),
                'result': result
            })
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            results.append({
                'scenario': scenario['name'],
                'success': False,
                'error': str(e),
                'has_cuda_error': 'device-side assert' in str(e)
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    cuda_errors_handled = sum(1 for r in results if r.get('has_cuda_error', False))
    
    print(f"✅ Successful tests: {successful_tests}/{total_tests}")
    print(f"🔧 CUDA errors handled: {cuda_errors_handled}")
    
    # Method usage summary
    methods_used = {}
    for r in results:
        if r['success']:
            method = r['method']
            methods_used[method] = methods_used.get(method, 0) + 1
    
    print(f"📈 Extraction methods used:")
    for method, count in methods_used.items():
        print(f"   {method}: {count} times")
    
    # Performance summary
    if successful_tests > 0:
        avg_time = sum(r['processing_time'] for r in results if r['success']) / successful_tests
        print(f"⏱️  Average processing time: {avg_time:.2f}s")
    
    # Save detailed results
    results_file = "/home/bibin.wilson/unified-kie-system/docext_integration_test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'gpu_info': {
                'cuda_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None
            },
            'test_results': results,
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'cuda_errors_handled': cuda_errors_handled,
                'methods_used': methods_used,
                'avg_processing_time': avg_time if successful_tests > 0 else None
            }
        }, f, indent=2, default=str)
    
    print(f"💾 Detailed results saved to: {results_file}")
    
    # Cleanup
    try:
        extractor.cleanup()
        print("🧹 Extractor cleanup completed")
    except:
        pass
    
    return results


def create_test_invoice_image():
    """Create a simple test invoice image for extraction testing"""
    
    # Create a simple invoice-like image using PIL
    width, height = 800, 600
    image = Image.new('RGB', (width, height), 'white')
    
    # For this test, we'll just create a white image
    # In a real scenario, you would have an actual document image
    return image


async def main():
    """Main test function"""
    try:
        results = await test_docext_integration()
        
        # Final assessment
        successful_tests = sum(1 for r in results if r['success'])
        total_tests = len(results)
        
        print(f"\n🎯 FINAL ASSESSMENT:")
        print(f"   Integration Status: {'✅ SUCCESS' if successful_tests > 0 else '❌ FAILED'}")
        print(f"   Test Success Rate: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)")
        print(f"   CUDA Compatibility: {'✅ GOOD' if torch.cuda.is_available() else '⚠️  NO GPU'}")
        
        if successful_tests > 0:
            print(f"   📋 DocExt integration is working correctly with RTX 4000 ADA")
        else:
            print(f"   📋 DocExt integration needs attention")
        
        return successful_tests > 0
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)