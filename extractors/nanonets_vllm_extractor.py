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
        """Initialize the vLLM engine and processor with CUDA error handling."""
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
            # Initialize vLLM engine with conservative settings
            self.model = LLM(
                model=model_name, 
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,  # Conservative memory usage
                max_model_len=4096,  # Reduced context length
                enforce_eager=True,  # Disable CUDA graphs for stability
                disable_custom_all_reduce=True  # Better compatibility
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
        
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)
        
        try:
            # Clear CUDA cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Generate with error recovery
            outputs = self.model.generate([text], sampling_params)
            raw_response = outputs[0].outputs[0].text
            structured_data = self._parse_response(raw_response)
            
            return {
                "raw_text": raw_response,
                "key_values": structured_data,
                "extraction_method": "nanonets_vllm",
                "timestamp": datetime.now().isoformat()
            }
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
        """Create extraction prompt based on language and document type."""
        # Using the same detailed prompts as the original extractor
        if language == "de" or language == "auto":
            return """Extract all key-value pairs from this document. This could be a German customs export declaration (Ausfuhranmeldung), invoice (Rechnung), or other business document.

EXTRACT ALL VISIBLE TEXT AND STRUCTURE AS JSON:

For German Customs Documents:
- LRN (Local Reference Number / Lokale Referenznummer)
- MRN (Movement Reference Number / Bearbeitungsnummer)  
- EORI-Nummer
- Dates: Anmeldedatum, Ausgangsdatum, Gültigkeitsdatum
- Companies: Anmelder, Ausführer, Empfänger (Name, Adresse, Kontakt)
- Customs offices: Gestellungszollstelle, Ausfuhrzollstelle
- Goods: Warenbezeichnung, Warennummer, Ursprungsland
- Quantities: Menge, Gewicht, Wert, Währung
- Transport: Verkehrszweig, Kennzeichen, Container
- Procedures: Verfahren, Bewilligung

For Invoices:
- Invoice number (Rechnungsnummer)
- Date (Datum)
- Vendor/Customer details (Lieferant/Kunde)
- Line items (Positionen)
- Amounts (Beträge)
- Tax information (Steuern)

For Other Documents:
- Extract all visible text fields
- Identify dates, numbers, names, addresses
- Capture table data with structure
- Note any stamps, signatures, or handwritten text

INSTRUCTIONS:
1. Preserve original language and formatting
2. Extract ALL text, even partial or unclear
3. Structure as comprehensive JSON
4. Include confidence indicators for uncertain text
5. Maintain relationships between related fields

Return complete JSON structure with all extracted information."""

        else:  # English or other languages
            return """Extract all key-value pairs from this document (invoice, customs declaration, business document, etc.).

EXTRACT ALL VISIBLE TEXT AND STRUCTURE AS JSON:

For Invoices:
- Invoice number, date, due date
- Vendor/supplier information (name, address, contact)
- Customer/buyer information  
- Line items (description, quantity, unit price, total)
- Subtotal, tax amount, grand total
- Payment terms, currency
- Any additional notes or terms

For Customs/Export Documents:
- Reference numbers (LRN, MRN, EORI)
- Declaration dates and validity
- Parties involved (declarant, exporter, consignee)
- Goods description and classification
- Origin and destination countries
- Quantities, weights, values
- Transport details
- Customs procedures and authorizations

For General Documents:
- All visible text fields and values
- Dates, numbers, names, addresses
- Table data with proper structure
- Company information and contacts
- Any stamps, signatures, or annotations

INSTRUCTIONS:
1. Extract ALL visible text, even if partially readable
2. Maintain original formatting and structure
3. Return comprehensive JSON with nested objects
4. Include metadata about extraction confidence
5. Preserve relationships between related fields

Return complete JSON structure with extracted information."""

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
