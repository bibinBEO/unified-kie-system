import torch
import asyncio
from transformers import AutoProcessor, AutoTokenizer
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
try:
    # Import AutoConfig first to detect model architecture
    from transformers import AutoConfig, AutoModelForVision2Seq
    
    # Try to import the correct Qwen architecture based on model config
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
        QWEN25_AVAILABLE = True
        print("✅ Using Qwen2.5-VL architecture (correct for Nanonets-OCR-s)")
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as QwenVLModel
        QWEN25_AVAILABLE = False
        print("⚠️  Fallback to Qwen2-VL architecture (may cause tensor mismatches)")
    
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    QWEN25_AVAILABLE = False
    QwenVLModel = None
    print("Warning: Qwen VL models not available, using fallback")
    
try:
    from qwen_vl_utils import process_vision_info
    QWEN_UTILS_AVAILABLE = True
except ImportError:
    QWEN_UTILS_AVAILABLE = False
    print("Warning: qwen_vl_utils not available, using fallback")
from PIL import Image
import json
import re
from typing import Dict, Any, List
from datetime import datetime

class NanoNetsExtractor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
        self.tokenizer = None
        # GPU mode with proper CUDA fixes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 NanoNets initializing on {self.device.upper()}")
        self.max_memory = self._get_available_memory()
    
    def _get_available_memory(self):
        """Get available GPU memory in bytes"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
        return 0
    
    def _run_with_timeout(self, func, timeout_seconds=120):
        """Run a function with timeout to prevent hanging"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout_seconds)
            except FutureTimeoutError:
                print(f"⚠️  Operation timed out after {timeout_seconds} seconds")
                return None
        
    async def initialize(self):
        """Initialize the NanoNets OCR-s model"""
        if not QWEN_AVAILABLE:
            print("⚠️  NanoNets OCR-s not available - using fallback mode")
            self.model = None
            self.processor = None
            self.tokenizer = None
            return
            
        def load_model():
            model_name = "nanonets/Nanonets-OCR-s"
            
            try:
                # Enable comprehensive CUDA debugging for better error tracing
                import os
                os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
                os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enable device-side assertions for debugging
                print("🔧 CUDA debugging enabled: CUDA_LAUNCH_BLOCKING=1, TORCH_USE_CUDA_DSA=1")
                
                # First, check the model's actual architecture to avoid tensor mismatches
                print("🔍 Detecting model architecture...")
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                model_arch = config.architectures[0] if config.architectures else "unknown"
                print(f"📋 Model architecture: {model_arch}")
                
                # Verify we're using the correct model class
                if "Qwen2_5_VL" in model_arch and not QWEN25_AVAILABLE:
                    print("⚠️  WARNING: Model requires Qwen2.5-VL but only Qwen2-VL available!")
                    print("   This will likely cause CUDA tensor shape mismatches!")
                elif "Qwen2_5_VL" in model_arch and QWEN25_AVAILABLE:
                    print("✅ Correct architecture match: Qwen2.5-VL")
                elif "Qwen2VL" in model_arch and not QWEN25_AVAILABLE:
                    print("✅ Correct architecture match: Qwen2-VL")
                
                # Proper dtype configuration for Flash Attention
                torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
                
                # Load model WITHOUT Flash Attention to avoid CUDA issues
                if self.device == "cuda":
                    print("🔧 Loading GPU model WITHOUT Flash Attention for stability...")
                    model = QwenVLModel.from_pretrained(
                        model_name,
                        torch_dtype=torch_dtype,
                        device_map="cuda:0",
                        ignore_mismatched_sizes=True,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        attn_implementation="eager"  # Explicitly disable Flash Attention
                    )
                    print("✅ Loaded on GPU without Flash Attention")
                else:
                    # CPU loading
                    model = QwenVLModel.from_pretrained(
                        model_name,
                        torch_dtype=torch_dtype,
                        ignore_mismatched_sizes=True,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                
                processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                
                return model, processor, tokenizer
            except Exception as e:
                print(f"⚠️  Failed to load NanoNets OCR-s on GPU: {e}")
                print(f"⚠️  Error type: {type(e).__name__}")
                # Try CPU fallback only if we started with CUDA
                if self.device == "cuda":
                    print("🔄 Attempting CPU fallback due to GPU loading issues...")
                    self.device = "cpu"
                    try:
                        model = QwenVLModel.from_pretrained(
                            model_name,
                            torch_dtype=torch.float32,
                            ignore_mismatched_sizes=True,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True
                        )
                        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                        print("✅ CPU fallback loading successful")
                        return model, processor, tokenizer
                    except Exception as cpu_e:
                        print(f"⚠️  CPU fallback also failed: {cpu_e}")
                        print(f"⚠️  CPU Error type: {type(cpu_e).__name__}")
                else:
                    print(f"⚠️  Model loading failed on {self.device}: {e}")
                return None, None, None
        
        loop = asyncio.get_event_loop()
        self.model, self.processor, self.tokenizer = await loop.run_in_executor(None, load_model)
        
        if self.model is not None:
            print(f"✅ NanoNets OCR-s loaded on {self.device}")
        else:
            print("⚠️  NanoNets OCR-s fallback mode active")

    async def extract(self, image: Image.Image, language: str = "auto") -> Dict[str, Any]:
        """Extract key information from image"""
        try:
            # Convert image to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Validate image dimensions and size
            width, height = image.size
            if width < 32 or height < 32:
                raise ValueError(f"Image too small: {width}x{height}. Minimum size is 32x32")
            
            # ULTRA-AGGRESSIVE image resizing for maximum speed 
            max_dimension = 512  # Even smaller for maximum speed
            if width > max_dimension or height > max_dimension:
                # Calculate resize ratio to maintain aspect ratio
                ratio = min(max_dimension / width, max_dimension / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                
                # Ensure dimensions are divisible by common factors to prevent tensor issues
                new_width = (new_width // 32) * 32
                new_height = (new_height // 32) * 32
                
                # Minimum size check
                if new_width < 64:
                    new_width = 64
                if new_height < 64:
                    new_height = 64
                    
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"ULTRA-AGGRESSIVE resize: {width}x{height} -> {image.size} (IndexKernel fix)")
            
            # Fallback mode when model is not available  
            if not QWEN_AVAILABLE or self.model is None:
                return await self._fallback_extract(image, language)
        except Exception as img_error:
            print(f"Image preprocessing error: {img_error}")
            # Try fallback extraction for problematic images
            return await self._fallback_extract(image, language)
            
        prompt = self._create_extraction_prompt(language)
        
        def inference():
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                if QWEN_UTILS_AVAILABLE:
                    try:
                        image_inputs, video_inputs = process_vision_info(messages)
                    except Exception as vision_error:
                        print(f"Vision processing error: {vision_error}")
                        # Fallback to direct image processing
                        image_inputs = [image]
                        video_inputs = None
                else:
                    image_inputs = [image]
                    video_inputs = None
                
                try:
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    )
                    
                    # Proper device transfer with vision input fixes
                    inputs = inputs.to(self.device)
                    
                    # Fix for CUDA device assertion - ensure media inputs are on correct device
                    if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
                        inputs.image_grid_thw = inputs.image_grid_thw.to(self.device)
                        print("✅ Fixed image_grid_thw device placement")
                    
                    # Additional vision input device fixes
                    for key in ['pixel_values', 'image_sizes', 'video_grid_thw']:
                        if hasattr(inputs, key) and getattr(inputs, key) is not None:
                            setattr(inputs, key, getattr(inputs, key).to(self.device))
                    
                    # Fix for tensor indexing issue - validate and fix tensor dimensions
                    print(f"🔧 Tensor validation:")
                    print(f"   input_ids shape: {inputs.input_ids.shape}")
                    if hasattr(inputs, 'attention_mask'):
                        print(f"   attention_mask shape: {inputs.attention_mask.shape}")
                    if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
                        print(f"   pixel_values shape: {inputs.pixel_values.shape}")
                    
                    # ADVANCED TENSOR VALIDATION AND FIXES
                    
                    # Critical fix: Ensure pixel_values has proper batch dimension
                    if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
                        pixel_shape = inputs.pixel_values.shape
                        print(f"🔧 ADVANCED PIXEL_VALUES FIX:")
                        print(f"   Original shape: {pixel_shape}")
                        print(f"   Device: {inputs.pixel_values.device}")
                        print(f"   Dtype: {inputs.pixel_values.dtype}")
                        
                        # Validate pixel_values tensor bounds
                        if torch.any(torch.isnan(inputs.pixel_values)):
                            print(f"⚠️  NaN detected in pixel_values - replacing with zeros")
                            inputs.pixel_values = torch.nan_to_num(inputs.pixel_values, nan=0.0)
                        
                        if torch.any(torch.isinf(inputs.pixel_values)):
                            print(f"⚠️  Inf detected in pixel_values - clamping")
                            inputs.pixel_values = torch.clamp(inputs.pixel_values, -10.0, 10.0)
                        
                        # Ensure proper batch dimension structure
                        if len(pixel_shape) == 2:  # Missing batch dimension
                            print(f"🔧 Adding batch dimension")
                            inputs.pixel_values = inputs.pixel_values.unsqueeze(0)
                        elif len(pixel_shape) == 3:  # Likely (channels, height, width)
                            print(f"🔧 Adding batch dimension to 3D tensor")
                            inputs.pixel_values = inputs.pixel_values.unsqueeze(0)
                        elif len(pixel_shape) == 4 and pixel_shape[0] != 1:  # Wrong batch size
                            print(f"🔧 Fixing batch dimension")
                            inputs.pixel_values = inputs.pixel_values[:1]
                        
                        # Validate final pixel_values shape
                        final_shape = inputs.pixel_values.shape
                        print(f"   Final shape: {final_shape}")
                        
                        # Additional safety: ensure pixel_values are within expected bounds
                        if len(final_shape) >= 2:
                            h, w = final_shape[-2], final_shape[-1]
                            if h <= 0 or w <= 0 or h > 4096 or w > 4096:
                                raise ValueError(f"Invalid image dimensions: {h}x{w}")
                        
                        # Ensure pixel_values are contiguous for CUDA operations
                        if not inputs.pixel_values.is_contiguous():
                            print(f"🔧 Making pixel_values contiguous")
                            inputs.pixel_values = inputs.pixel_values.contiguous()
                    
                    # ADVANCED SEQUENCE AND TENSOR VALIDATION
                    print(f"🔧 ADVANCED SEQUENCE VALIDATION:")
                    
                    # Validate input_ids tensor
                    if torch.any(inputs.input_ids < 0):
                        print(f"⚠️  Negative token IDs detected - fixing")
                        inputs.input_ids = torch.clamp(inputs.input_ids, min=0)
                    
                    # Check for extremely large token IDs that could cause indexing issues
                    vocab_size = getattr(self.model.config, 'vocab_size', 151936)  # Qwen2VL vocab size
                    if torch.any(inputs.input_ids >= vocab_size):
                        print(f"⚠️  Token IDs >= vocab_size ({vocab_size}) detected - clamping")
                        inputs.input_ids = torch.clamp(inputs.input_ids, max=vocab_size-1)
                    
                    batch_size, seq_len = inputs.input_ids.shape
                    print(f"   Batch size: {batch_size}, Sequence length: {seq_len}")
                    
                    # Force batch size to 1 for stability
                    if batch_size > 1:
                        print(f"🔧 Reducing batch size from {batch_size} to 1")
                        inputs.input_ids = inputs.input_ids[:1]
                        if hasattr(inputs, 'attention_mask'):
                            inputs.attention_mask = inputs.attention_mask[:1]
                        if hasattr(inputs, 'position_ids'):
                            inputs.position_ids = inputs.position_ids[:1]
                        batch_size = 1
                    
                    # ULTRA-CONSERVATIVE sequence length limit to prevent masked_scatter errors
                    max_seq_len = 512  # Ultra-conservative limit for IndexKernel safety
                    if seq_len > max_seq_len:
                        print(f"🔧 ULTRA-AGGRESSIVE truncating: {seq_len} -> {max_seq_len} (IndexKernel fix)")
                        inputs.input_ids = inputs.input_ids[:, :max_seq_len]
                        if hasattr(inputs, 'attention_mask'):
                            inputs.attention_mask = inputs.attention_mask[:, :max_seq_len]
                        if hasattr(inputs, 'position_ids'):
                            inputs.position_ids = inputs.position_ids[:, :max_seq_len]
                        seq_len = max_seq_len
                    
                    # CRITICAL: Additional safety for image token processing
                    # Count image tokens and ensure they don't exceed limits
                    image_token_id = getattr(self.model.config, 'image_token_id', 151655)
                    image_token_count = torch.sum(inputs.input_ids == image_token_id).item()
                    
                    if image_token_count > 256:  # Conservative image token limit
                        print(f"⚠️  Too many image tokens ({image_token_count}), this may cause IndexKernel errors")
                        print(f"🔧 Consider using smaller images or different processing approach")
                    
                    print(f"   Image token count: {image_token_count}")
                    print(f"   Max sequence length: {max_seq_len}")
                    print(f"   Actual sequence length: {seq_len}")
                    
                    # Validate attention_mask if present
                    if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
                        if inputs.attention_mask.shape != inputs.input_ids.shape:
                            print(f"⚠️  Attention mask shape mismatch - fixing")
                            # Create new attention mask with correct shape
                            inputs.attention_mask = torch.ones_like(inputs.input_ids)
                        
                        # Ensure attention_mask is binary
                        if not torch.all((inputs.attention_mask == 0) | (inputs.attention_mask == 1)):
                            print(f"🔧 Converting attention_mask to binary")
                            inputs.attention_mask = (inputs.attention_mask > 0).long()
                    
                    # CRITICAL: Handle position_ids to prevent IndexKernel errors
                    if hasattr(inputs, 'position_ids') and inputs.position_ids is not None:
                        print(f"🔧 POSITION_IDS VALIDATION:")
                        print(f"   Shape: {inputs.position_ids.shape}")
                        print(f"   Max value: {torch.max(inputs.position_ids).item()}")
                        print(f"   Min value: {torch.min(inputs.position_ids).item()}")
                        
                        # Ensure position_ids are within valid range
                        max_position = getattr(self.model.config, 'max_position_embeddings', 32768)
                        if torch.any(inputs.position_ids >= max_position):
                            print(f"⚠️  Position IDs >= max_position ({max_position}) - clamping")
                            inputs.position_ids = torch.clamp(inputs.position_ids, max=max_position-1)
                        
                        if torch.any(inputs.position_ids < 0):
                            print(f"⚠️  Negative position IDs detected - fixing")
                            inputs.position_ids = torch.clamp(inputs.position_ids, min=0)
                        
                        # Ensure position_ids are contiguous and properly shaped
                        if not inputs.position_ids.is_contiguous():
                            inputs.position_ids = inputs.position_ids.contiguous()
                    
                    # Ensure all tensors are contiguous for CUDA operations
                    inputs.input_ids = inputs.input_ids.contiguous()
                    if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
                        inputs.attention_mask = inputs.attention_mask.contiguous()
                    
                    # Final validation
                    print(f"✅ Final tensor shapes:")
                    print(f"   input_ids: {inputs.input_ids.shape}")
                    if hasattr(inputs, 'attention_mask'):
                        print(f"   attention_mask: {inputs.attention_mask.shape}")
                    if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
                        print(f"   pixel_values: {inputs.pixel_values.shape}")
                    
                    # ULTRA-CONSERVATIVE generation parameters for IndexKernel safety
                    generation_kwargs = {
                        "max_new_tokens": 64,   # Extremely conservative to prevent IndexKernel errors
                        "do_sample": False,
                        "pad_token_id": self.processor.tokenizer.eos_token_id,
                        "eos_token_id": self.processor.tokenizer.eos_token_id,
                        "use_cache": False,     # Disable cache to avoid state issues
                        "output_attentions": False,
                        "output_hidden_states": False,
                        "return_dict_in_generate": False
                    }
                    
                    print(f"🔧 Generation config: max_new_tokens={generation_kwargs['max_new_tokens']}")
                    
                    # ADVANCED CUDA MEMORY MANAGEMENT AND SAFE INFERENCE
                    def safe_inference():
                        try:
                            with torch.no_grad():
                                # Advanced CUDA memory management
                                if self.device == "cuda":
                                    print(f"🔧 CUDA MEMORY MANAGEMENT:")
                                    
                                    # Clear all caches
                                    torch.cuda.empty_cache()
                                    torch.cuda.ipc_collect()
                                    
                                    # Check memory before inference
                                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                                    print(f"   Memory allocated: {memory_allocated:.2f}GB")
                                    print(f"   Memory reserved: {memory_reserved:.2f}GB")
                                    
                                    # Synchronize device to ensure all operations are complete
                                    torch.cuda.synchronize()
                                    
                                    # Set deterministic behavior for debugging
                                    torch.backends.cudnn.deterministic = True
                                    torch.backends.cudnn.benchmark = False
                                
                                # Final input validation before generation
                                print(f"🔧 FINAL INPUT VALIDATION:")
                                
                                # Validate input_ids
                                if not hasattr(inputs, 'input_ids') or inputs.input_ids is None:
                                    raise ValueError("Missing input_ids")
                                
                                if inputs.input_ids.numel() == 0:
                                    raise ValueError("Empty input_ids tensor")
                                
                                # Validate all tensor devices match
                                target_device = inputs.input_ids.device
                                for attr_name in ['attention_mask', 'pixel_values', 'position_ids', 'image_grid_thw']:
                                    if hasattr(inputs, attr_name):
                                        attr_tensor = getattr(inputs, attr_name)
                                        if attr_tensor is not None and attr_tensor.device != target_device:
                                            print(f"⚠️  Device mismatch for {attr_name}: {attr_tensor.device} != {target_device}")
                                            setattr(inputs, attr_name, attr_tensor.to(target_device))
                                
                                # Log final tensor information
                                print(f"   Input device: {inputs.input_ids.device}")
                                print(f"   Input dtype: {inputs.input_ids.dtype}")
                                print(f"   Input shape: {inputs.input_ids.shape}")
                                print(f"   Input range: [{torch.min(inputs.input_ids).item()}, {torch.max(inputs.input_ids).item()}]")
                                
                                # SAFE GENERATION WITH CUDA ERROR HANDLING
                                print(f"🔧 STARTING SAFE GENERATION...")
                                
                                try:
                                    # Enable CUDA error checking
                                    if self.device == "cuda":
                                        torch.cuda.set_sync_debug_mode("warn")
                                    
                                    # Generate with extra safety (avoid duplicate parameters)
                                    safe_generation_kwargs = generation_kwargs.copy()
                                    safe_generation_kwargs.update({
                                        "output_attentions": False,
                                        "output_hidden_states": False,
                                        "return_dict_in_generate": False,
                                        "use_cache": False,  # Disable KV cache to reduce memory issues
                                        "do_sample": False,  # Deterministic generation
                                        "num_beams": 1,      # No beam search
                                        "repetition_penalty": 1.0,  # No repetition penalty
                                        "length_penalty": 1.0       # No length penalty
                                    })
                                    
                                    generated_ids = self.model.generate(**inputs, **safe_generation_kwargs)
                                    
                                    if self.device == "cuda":
                                        torch.cuda.synchronize()  # Wait for generation to complete
                                        print(f"✅ Generation completed successfully")
                                    
                                    return generated_ids
                                    
                                except RuntimeError as cuda_error:
                                    error_msg = str(cuda_error).lower()
                                    
                                    if "device-side assert" in error_msg or "indexkernel" in error_msg:
                                        print(f"🔧 CUDA IndexKernel assertion detected - attempting recovery")
                                        
                                        # Clear CUDA state and try with even more conservative settings
                                        if self.device == "cuda":
                                            torch.cuda.empty_cache()
                                            torch.cuda.synchronize()
                                        
                                        # MINIMAL generation parameters for IndexKernel recovery
                                        recovery_kwargs = {
                                            "max_new_tokens": 16,   # Minimal to avoid IndexKernel
                                            "do_sample": False,
                                            "num_beams": 1,
                                            "pad_token_id": self.processor.tokenizer.eos_token_id,
                                            "eos_token_id": self.processor.tokenizer.eos_token_id,
                                            "use_cache": False,
                                            "output_attentions": False,
                                            "output_hidden_states": False,
                                            "return_dict_in_generate": False
                                        }
                                        
                                        print(f"🔧 IndexKernel recovery: max_new_tokens=16")
                                        return self.model.generate(**inputs, **recovery_kwargs)
                                    
                                    else:
                                        raise cuda_error
                                        
                        except Exception as gen_error:
                            print(f"⚠️  Generation error: {gen_error}")
                            print(f"⚠️  Error type: {type(gen_error).__name__}")
                            
                            # Additional CUDA error diagnostics
                            if self.device == "cuda":
                                try:
                                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                                    print(f"⚠️  CUDA Memory at error: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                                except:
                                    pass
                            
                            raise gen_error
                    
                    generated_ids = self._run_with_timeout(safe_inference, timeout_seconds=90)
                    
                    if generated_ids is None:
                        return {
                            "raw_text": f"Extraction failed: {self.device.upper()} inference timeout",
                            "key_values": {
                                "error": f"{self.device.upper()} inference timeout",
                                "status": "failed",
                                "device": self.device
                            },
                            "extraction_method": "nanonets_ocr_timeout",
                            "timestamp": datetime.now().isoformat()
                        }
                    
                    # Continue with successful processing
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    
                    response = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    
                    return response
                    
                except Exception as processing_error:
                    print(f"⚠️  Input processing failed: {processing_error}")
                    raise processing_error
                    
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                print(f"⚠️  NanoNets extraction failed: {error_msg}")
                print(f"⚠️  Error type: {error_type}")
                print(f"⚠️  Device: {self.device}")
                
                # Specific handling for CUDA errors
                if "device-side assert" in error_msg.lower():
                    print("🔧 CUDA device assertion detected - this indicates tensor indexing issues")
                    if self.device == "cuda":
                        print("💡 Suggestion: Check input tensor shapes and ranges")
                
                return {
                    "raw_text": f"Extraction failed: {error_msg}",
                    "key_values": {
                        "error": error_msg,
                        "error_type": error_type,
                        "device": self.device,
                        "status": "failed",
                        "debug_info": {
                            "cuda_available": torch.cuda.is_available(),
                            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
                        }
                    },
                    "extraction_method": "nanonets_ocr_error",
                    "timestamp": datetime.now().isoformat()
                }
        
        loop = asyncio.get_event_loop()
        raw_response = await loop.run_in_executor(None, inference)
        
        # Handle error responses that are already formatted as dicts
        if isinstance(raw_response, dict):
            return raw_response
        
        # Parse and structure the response
        structured_data = self._parse_response(raw_response)
        
        return {
            "raw_text": raw_response,
            "key_values": structured_data,
            "extraction_method": "nanonets_ocr_s",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _fallback_extract(self, image: Image.Image, language: str = "auto") -> Dict[str, Any]:
        """Fallback extraction method when NanoNets is not available"""
        return {
            "raw_text": "NanoNets OCR-s model not available - using fallback mode",
            "key_values": {
                "status": "fallback_mode",
                "message": "NanoNets OCR-s requires newer transformers version",
                "recommendation": "Please update transformers to use full functionality"
            },
            "extraction_method": "nanonets_fallback",
            "timestamp": datetime.now().isoformat()
        }

    def _create_extraction_prompt(self, language: str) -> str:
        """Create simplified extraction prompt for standard extractor"""
        
        if language == "de" or language == "auto":
            return """Extract key info as JSON:
{"company": "Firmenname", "invoice": "Rechnungsnummer", "date": "Datum", "amount": "Betrag", "currency": "EUR"}"""

        else:  # English or other languages
            return """Extract key info as JSON:
{"company": "Company name", "invoice": "Invoice number", "date": "Date", "amount": "Amount", "currency": "USD"}"""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the model response into structured data"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_data = json.loads(json_str)
                return self._enhance_extraction(parsed_data, response)
            else:
                return self._fallback_parse(response)
        except json.JSONDecodeError:
            return self._fallback_parse(response)

    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails"""
        lines = response.strip().split('\n')
        result = {"raw_response": response, "parsed_fields": {}}
        
        current_section = "general"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.endswith(':') and len(line.split()) <= 3:
                current_section = line.replace(':', '').lower().replace(' ', '_')
                result["parsed_fields"][current_section] = {}
                continue
            
            # Extract key-value pairs
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower().replace(' ', '_')
                    value = parts[1].strip()
                    
                    if value:
                        if current_section not in result["parsed_fields"]:
                            result["parsed_fields"][current_section] = {}
                        result["parsed_fields"][current_section][key] = value
        
        return result

    def _enhance_extraction(self, parsed_data: Dict[str, Any], raw_response: str) -> Dict[str, Any]:
        """Enhance parsed data with additional processing"""
        enhanced_data = parsed_data.copy()
        
        # Add extraction metadata
        enhanced_data["extraction_metadata"] = {
            "model_used": "nanonets-ocr-s",
            "extraction_timestamp": datetime.now().isoformat(),
            "confidence_indicators": self._assess_confidence(raw_response),
            "document_language": self._detect_language(raw_response)
        }
        
        # Normalize field names and values
        enhanced_data = self._normalize_fields(enhanced_data)
        
        return enhanced_data

    def _assess_confidence(self, response: str) -> Dict[str, Any]:
        """Assess extraction confidence based on response characteristics"""
        return {
            "response_length": len(response),
            "json_structure_found": '{' in response and '}' in response,
            "has_structured_data": ':' in response,
            "estimated_confidence": "high" if len(response) > 100 and '{' in response else "medium"
        }

    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        german_indicators = ['der', 'die', 'das', 'und', 'ist', 'von', 'zu', 'mit', 'auf']
        english_indicators = ['the', 'and', 'is', 'of', 'to', 'with', 'on', 'for']
        
        text_lower = text.lower()
        german_count = sum(1 for word in german_indicators if word in text_lower)
        english_count = sum(1 for word in english_indicators if word in text_lower)
        
        if german_count > english_count:
            return "german"
        elif english_count > german_count:
            return "english"
        else:
            return "unknown"

    def _normalize_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize field names and values for consistency"""
        if not isinstance(data, dict):
            return data
        
        normalized = {}
        for key, value in data.items():
            # Normalize key
            normalized_key = key.lower().replace(' ', '_').replace('-', '_')
            
            # Recursively normalize nested dictionaries
            if isinstance(value, dict):
                normalized[normalized_key] = self._normalize_fields(value)
            elif isinstance(value, list):
                normalized[normalized_key] = [
                    self._normalize_fields(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                normalized[normalized_key] = value
        
        return normalized