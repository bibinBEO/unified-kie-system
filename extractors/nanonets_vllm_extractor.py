import torch
import asyncio
from transformers import AutoProcessor
from PIL import Image
import json
import re
from typing import Dict, Any
from datetime import datetime

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError as e:
    VLLM_AVAILABLE = False
    import sys
    import traceback
    print("Warning: vLLM not available, using fallback", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print(f"ImportError details: {e}", file=sys.stderr)

class NanoNetsVLLMExtractor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 NanoNets vLLM initializing on {self.device.upper()}")

    async def initialize(self):
        """Initialize the vLLM engine and processor with container-friendly settings."""
        print(f"DEBUG: VLLM_AVAILABLE={VLLM_AVAILABLE}, self.device={self.device}")
        if not VLLM_AVAILABLE or self.device != "cuda":
            print("⚠️  vLLM is not available or not running on GPU. Fallback mode.")
            return

        model_name = "nanonets/Nanonets-OCR-s"
        print(f"DEBUG: Attempting to initialize vLLM with model: {model_name}")
        
        # Clear CUDA cache before initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 CUDA cache cleared. Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        try:
            # Import multiprocessing context fix for containers
            import multiprocessing
            multiprocessing.set_start_method('spawn', force=True)
            
            # Initialize vLLM engine with container-optimized settings (compatible parameters only)
            self.model = LLM(
                model=model_name, 
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.75,  # Balanced memory usage
                max_model_len=2048,  # Increase for better vision processing
                enforce_eager=True,  # Disable CUDA graphs for compatibility
                disable_custom_all_reduce=True,  # Better compatibility
                max_num_seqs=1,  # Single sequence for maximum stability
                disable_log_stats=True,  # Reduce logging overhead
                use_v2_block_manager=False,  # Use stable block manager
                swap_space=0  # Disable swap for performance
            )
            print("DEBUG: vLLM LLM initialized with conservative settings.")
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            print("✅ vLLM engine and processor initialized successfully.")
        except Exception as e:
            import traceback
            print(f"⚠️  Failed to initialize vLLM engine: {type(e).__name__}: {e}")
            traceback.print_exc()
            self.model = None
            # Clear cache on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def extract(self, image: Image.Image, language: str = "auto") -> Dict[str, Any]:
        """Extract key information from an image using vLLM."""
        if not self.model:
            return await self._fallback_extract(image, language)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        prompt = self._create_extraction_prompt(language)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic output for speed
            top_p=1.0, 
            max_tokens=256,  # Very small output for maximum speed
            skip_special_tokens=True  # Skip special tokens for efficiency
        )
        
        try:
            # Aggressive image preprocessing for vLLM speed
            if hasattr(image, 'size'):
                width, height = image.size
                # Very aggressive resizing for vLLM performance
                max_dim = 768  # Smaller for vLLM vision processing
                if width > max_dim or height > max_dim:
                    ratio = min(max_dim/width, max_dim/height)
                    new_size = (int(width * ratio), int(height * ratio))
                    # Ensure dimensions are multiples of 32 for better tensor handling
                    new_size = ((new_size[0] // 32) * 32, (new_size[1] // 32) * 32)
                    if new_size[0] < 32:
                        new_size = (32, new_size[1])
                    if new_size[1] < 32:
                        new_size = (new_size[0], 32)
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                    print(f"🖼️ vLLM optimized resize: {width}x{height} → {new_size[0]}x{new_size[1]}")
            
            # Add timeout mechanism for vLLM generation
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("vLLM generation timed out")
            
            # Set 45-second timeout for vLLM processing
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(45)
            
            try:
                # Generate with optimized settings and detailed logging
                print(f"🔄 Starting vLLM generation with text length: {len(text)}")
                outputs = self.model.generate([text], sampling_params)
                print(f"✅ vLLM generation completed successfully")
                raw_response = outputs[0].outputs[0].text
                print(f"📝 Generated response length: {len(raw_response)}")
                structured_data = self._parse_response(raw_response)
                
                signal.alarm(0)  # Cancel timeout
                
                return {
                    "raw_text": raw_response,
                    "key_values": structured_data,
                    "extraction_method": "nanonets_vllm",
                    "timestamp": datetime.now().isoformat()
                }
            except TimeoutError:
                signal.alarm(0)  # Cancel timeout
                print("⏰ vLLM processing timed out (60s), falling back...")
                raise Exception("vLLM timeout - fallback required")
        except RuntimeError as e:
            error_str = str(e)
            print(f"⚠️  CUDA RuntimeError in vLLM extraction: {error_str}")
            
            # Handle specific CUDA errors
            if "device-side assert" in error_str or "CUDA error" in error_str:
                print("🔧 CUDA device-side assert detected. Clearing cache and falling back...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                print("🔄 Attempting fallback to standard NanoNets extractor...")
                return await self._fallback_extract(image, language)
            else:
                raise e  # Re-raise if not a known CUDA issue
        except Exception as e:
            print(f"⚠️  General vLLM extraction failed: {e}")
            return {
                "raw_text": f"Extraction failed: {e}",
                "key_values": {"error": str(e), "error_type": type(e).__name__},
                "extraction_method": "nanonets_vllm_error",
                "timestamp": datetime.now().isoformat()
            }

    async def _fallback_extract(self, image: Image.Image, language: str = "auto") -> Dict[str, Any]:
        """Fallback extraction method using standard NanoNets extractor."""
        try:
            # Import and use the standard NanoNets extractor as fallback
            from .nanonets_extractor import NanoNetsExtractor
            fallback_extractor = NanoNetsExtractor(self.config)
            await fallback_extractor.initialize()
            result = await fallback_extractor.extract(image, language)
            
            # Mark as fallback method
            result["extraction_method"] = "nanonets_fallback_from_vllm"
            result["fallback_reason"] = "vLLM unavailable or CUDA error"
            
            return result
        except Exception as e:
            print(f"⚠️  Fallback extraction also failed: {e}")
            return {
                "raw_text": "Both vLLM and fallback extraction failed.",
                "key_values": {
                    "status": "fallback_failed", 
                    "vllm_error": "CUDA device-side assert or initialization failure",
                    "fallback_error": str(e)
                },
                "extraction_method": "nanonets_complete_failure",
                "timestamp": datetime.now().isoformat()
            }

    def _create_extraction_prompt(self, language: str) -> str:
        """Create ultra-simple extraction prompt for vLLM speed."""
        
        if language == "de" or language == "auto":
            return """Extract key info from this German document as JSON:
{"company": "company name", "invoice": "number", "date": "date", "amount": "total", "currency": "EUR/USD"}"""

        else:  # English or other languages
            return """Extract key info from this document as JSON:
{"company": "company name", "invoice": "number", "date": "date", "amount": "total", "currency": "USD/EUR"}"""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the model response into structured data."""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                return self._fallback_parse(response)
        except json.JSONDecodeError:
            return self._fallback_parse(response)

    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails."""
        return {"raw_response": response, "parsed_fields": {}}
