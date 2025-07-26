#!/usr/bin/env python3
"""
Test script to verify NanoNets OCR-s integration for KIE
"""
import asyncio
import json
from document_processor import UnifiedDocumentProcessor
from PIL import Image
import tempfile
import os

async def test_ocr_s_integration():
    """Test the OCR-s integration"""
    
    # Basic configuration
    config = {
        "api_key": "test_key",
        "model_name": "nanonets/Nanonets-OCR-s",
        "max_tokens": 2048
    }
    
    print("🚀 Testing NanoNets OCR-s Integration")
    print("=" * 50)
    
    # Initialize processor
    processor = UnifiedDocumentProcessor(config)
    
    try:
        print("📋 Initializing extractors...")
        await processor.initialize()
        
        print("\n📊 Model Status:")
        status = await processor.get_model_status()
        for model, loaded in status.items():
            print(f"  {model}: {'✅ Loaded' if loaded else '❌ Not loaded'}")
        
        print("\n💾 Memory Usage:")
        memory = await processor.get_memory_usage()
        print(f"  RAM: {memory['ram_usage_mb']:.1f} MB")
        if 'gpu_memory_allocated_mb' in memory:
            print(f"  GPU: {memory['gpu_memory_allocated_mb']:.1f} MB allocated")
        
        # Create a simple test image
        print("\n🖼️  Creating test image...")
        test_image = Image.new('RGB', (800, 600), color='white')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_image.save(tmp.name)
            test_file = tmp.name
        
        try:
            print(f"📄 Processing test document: {test_file}")
            result = await processor.process_document(
                test_file, 
                extraction_type="invoice",
                use_schema=True
            )
            
            print("\n📋 Extraction Result:")
            print(f"  Strategy used: {result['processing_info']['strategy_used']}")
            print(f"  Extraction type: {result['processing_info']['extraction_type']}")
            
            if 'key_values' in result:
                print(f"  Key-value pairs found: {len(result['key_values'])}")
            
            if 'extraction_method' in result:
                print(f"  Method: {result['extraction_method']}")
                
            print("\n✅ Integration test completed successfully!")
            
        finally:
            # Clean up temp file
            if os.path.exists(test_file):
                os.unlink(test_file)
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ocr_s_integration())