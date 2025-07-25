import torch
import asyncio
from transformers import AutoProcessor, AutoTokenizer
try:
    from transformers import Qwen2VLForConditionalGeneration
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    print("Warning: Qwen2VLForConditionalGeneration not available, using fallback")
    
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
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
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype="auto",
                    ignore_mismatched_sizes=True,
                    attn_implementation="flash_attention_2"
                )
                
                processor = AutoProcessor.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                return model, processor, tokenizer
            except Exception as e:
                print(f"⚠️  Failed to load NanoNets OCR-s: {e}")
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
            if width > 4096 or height > 4096:
                # Resize large images to prevent memory issues
                image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                print(f"Resized large image from {width}x{height} to {image.size}")
            
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
                    
                    inputs = inputs.to(self.device)
                    
                    # Clear CUDA cache before inference
                    torch.cuda.empty_cache()
                    
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=512,  # Reduced to prevent memory issues
                            do_sample=False,
                            temperature=0.1,
                            pad_token_id=self.processor.tokenizer.eos_token_id
                        )
                except Exception as cuda_error:
                    print(f"CUDA error in NanoNets inference: {cuda_error}")
                    # Fallback to CPU inference
                    try:
                        inputs = inputs.to('cpu')
                        with torch.no_grad():
                            generated_ids = self.model.to('cpu').generate(
                                **inputs,
                                max_new_tokens=256,
                                do_sample=False,
                                temperature=0.1,
                                pad_token_id=self.processor.tokenizer.eos_token_id
                            )
                        # Move model back to GPU
                        self.model = self.model.to(self.device)
                    except Exception as fallback_error:
                        print(f"CPU fallback also failed: {fallback_error}")
                        raise Exception(f"Both GPU and CPU inference failed: {cuda_error}")
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                response = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                return response
            except Exception as e:
                print(f"⚠️  NanoNets extraction failed: {e}")
                return f"Extraction failed: {str(e)}"
        
        loop = asyncio.get_event_loop()
        raw_response = await loop.run_in_executor(None, inference)
        
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
        """Create extraction prompt based on language and document type"""
        
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