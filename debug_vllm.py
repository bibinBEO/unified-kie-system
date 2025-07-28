#!/usr/bin/env python3
"""
Debug vLLM issues step by step
"""

import torch
import sys

print("🔍 vLLM Debug Analysis")
print("=" * 30)

# Test 1: Basic imports
try:
    from vllm import LLM, SamplingParams
    print("✅ vLLM imports successful")
except Exception as e:
    print(f"❌ vLLM import failed: {e}")
    sys.exit(1)

# Test 2: CUDA availability
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"📊 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Test 3: Try to initialize vLLM with minimal settings
try:
    print("\n🔄 Testing minimal vLLM initialization...")
    model = LLM(
        model="nanonets/Nanonets-OCR-s",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,  # Very conservative
        max_model_len=512,  # Very small
        enforce_eager=True,
        max_num_seqs=1
    )
    print("✅ vLLM model loaded successfully!")
    
    # Test 4: Try text-only generation first
    try:
        print("\n🔄 Testing text-only generation...")
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=50
        )
        outputs = model.generate(["What is 2+2?"], sampling_params)
        print(f"✅ Text generation works: {outputs[0].outputs[0].text}")
        
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        
except Exception as e:
    print(f"❌ vLLM initialization failed: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

print("\n🏁 Debug complete")