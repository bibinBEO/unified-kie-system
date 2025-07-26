"""
Optimized NanoNets OCR-s extractor with vLLM performance enhancements
"""

import asyncio
import aiohttp
import json
import base64
import io
from PIL import Image
from typing import Dict, Any
from datetime import datetime
import concurrent.futures
import time

class NanoNetsOCRSExtractorOptimized:
    def __init__(self, config):
        self.config = config
        self.vllm_url = "http://localhost:8000"  
        self.session = None
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """Initialize async session with optimized settings"""
        timeout = aiohttp.ClientTimeout(total=120, connect=10)  # Increased timeout
        connector = aiohttp.TCPConnector(
            limit=100,  # Connection pool size
            limit_per_host=30,  # Connections per host
            keepalive_timeout=30,  # Keep connections alive
            enable_cleanup_closed=True
        )
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        print("✅ Optimized NanoNets OCR-s session initialized")

    async def extract(self, image: Image.Image, language: str = "auto") -> Dict[str, Any]:
        """Optimized extraction with image preprocessing and async processing"""
        start_time = time.time()
        
        try:
            # Step 1: Optimize image in thread pool
            optimized_image = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, self._optimize_image, image
            )
            
            # Step 2: Encode image efficiently
            image_data = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, self._encode_image, optimized_image
            )
            
            # Step 3: Create optimized payload
            payload = {
                "model": "nanonets/Nanonets-OCR-s",
                "messages": [
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                            },
                            {
                                "type": "text", 
                                "text": self._get_optimized_prompt(language)
                            }
                        ]
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.1,  # Lower for faster, more deterministic output
                "top_p": 0.9,
                "stream": False
            }
            
            # Step 4: Send request with async optimization
            async with self.session.post(
                f"{self.vllm_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content']
                    
                    # Step 5: Parse response in thread pool
                    structured_data = await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool, self._parse_response, content
                    )
                    
                    processing_time = time.time() - start_time
                    print(f"⚡ OCR-s extraction completed in {processing_time:.2f}s")
                    
                    return {
                        "raw_text": content,
                        "key_values": structured_data,
                        "extraction_method": "nanonets_ocr_s_optimized",
                        "processing_time_seconds": processing_time,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"vLLM API error {response.status}: {error_text}")
                    
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ OCR-s extraction failed after {processing_time:.2f}s: {e}")
            return {
                "raw_text": f"Extraction failed: {str(e)}",
                "key_values": {"error": str(e), "processing_time": processing_time},
                "extraction_method": "nanonets_ocr_s_optimized_error",
                "timestamp": datetime.now().isoformat()
            }

    def _optimize_image(self, image: Image.Image) -> Image.Image:
        """Optimize image for faster processing"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for optimal processing speed vs quality
        width, height = image.size
        max_dimension = 1536  # Optimal size for speed/quality balance
        
        if width > max_dimension or height > max_dimension:
            ratio = min(max_dimension/width, max_dimension/height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"🖼️ Optimized image: {width}x{height} -> {new_size[0]}x{new_size[1]}")
        
        return image

    def _encode_image(self, image: Image.Image) -> str:
        """Efficiently encode image to base64"""
        buffer = io.BytesIO()
        
        # Use JPEG with optimal quality/size balance
        image.save(buffer, format='JPEG', quality=85, optimize=True)
        image_bytes = buffer.getvalue()
        
        return base64.b64encode(image_bytes).decode('utf-8')

    def _get_optimized_prompt(self, language: str) -> str:
        """Get optimized extraction prompt for faster processing"""
        if language == "de" or language == "auto":
            return \"\"\"Extract key information from this document efficiently. Focus on:

PRIORITY FIELDS (extract first):
- Numbers: LRN, MRN, EORI, Invoice numbers, dates
- Companies: Names and addresses of key parties
- Amounts: Values, currencies, quantities, weights
- Critical dates: Anmeldedatum, invoice date, due dates

STRUCTURE AS COMPACT JSON:
{
  "document_type": "invoice|customs|other",
  "key_numbers": {"lrn": "", "mrn": "", "invoice_no": "", "eori": ""},
  "dates": {"date": "", "due_date": "", "anmeldedatum": ""},
  "parties": {"sender": "", "recipient": "", "anmelder": ""},
  "amounts": {"total": "", "currency": "", "weight": ""},
  "goods": {"description": "", "origin": "", "classification": ""},
  "other_fields": {}
}

Return ONLY the JSON. Be efficient and accurate.\"\"\"
        
        else:
            return \"\"\"Extract key information efficiently. Priority fields:

- Document type and numbers (invoice #, reference numbers)
- Key dates (invoice date, due date, processing date)  
- Party information (vendor, customer, addresses)
- Financial data (amounts, currency, totals, tax)
- Goods/services (description, quantities, classifications)

Return compact JSON structure:
{
  "document_type": "",
  "numbers": {"invoice": "", "reference": "", "po": ""},
  "dates": {"invoice_date": "", "due_date": ""},
  "parties": {"vendor": "", "customer": ""},
  "amounts": {"subtotal": "", "tax": "", "total": "", "currency": ""},
  "items": [],
  "other": {}
}

ONLY JSON response. Be fast and precise.\"\"\"

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Optimized response parsing"""
        try:
            # Try direct JSON parsing first
            if response.strip().startswith('{'):
                return json.loads(response.strip())
            
            # Find JSON block
            import re
            json_match = re.search(r'\\{[^{}]*(?:\\{[^{}]*\\}[^{}]*)*\\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            # Fallback parsing
            return {"raw_response": response, "parsed": False}
            
        except json.JSONDecodeError:
            return {"raw_response": response, "json_parse_error": True}

    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        self.thread_pool.shutdown(wait=True)