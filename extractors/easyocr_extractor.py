import easyocr
import asyncio
from PIL import Image
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import re

class EasyOCRExtractor:
    def __init__(self, config):
        self.config = config
        self.reader = None
        self.languages = config.EASYOCR_LANGUAGES
        
    async def initialize(self):
        """Initialize EasyOCR reader"""
        def load_reader():
            return easyocr.Reader(self.languages, gpu=not self.config.FORCE_CPU)
        
        loop = asyncio.get_event_loop()
        self.reader = await loop.run_in_executor(None, load_reader)
        print(f"✅ EasyOCR initialized with languages: {self.languages}")

    async def extract(self, image: Image.Image, language: str = "auto") -> Dict[str, Any]:
        """Extract text using EasyOCR"""
        def ocr_inference():
            # Convert PIL Image to numpy array
            image_array = np.array(image)
            
            # Perform OCR
            results = self.reader.readtext(image_array)
            
            return results
        
        loop = asyncio.get_event_loop()
        ocr_results = await loop.run_in_executor(None, ocr_inference)
        
        # Process OCR results
        full_text = ""
        text_blocks = []
        
        for (bbox, text, confidence) in ocr_results:
            if confidence > 0.3:  # Filter low-confidence results
                full_text += text + " "
                text_blocks.append({
                    "text": text,
                    "bbox": bbox,
                    "confidence": confidence
                })
        
        # Extract key-value pairs from text
        key_values = self._extract_key_values(full_text)
        
        return {
            "raw_text": full_text.strip(),
            "text_blocks": text_blocks,
            "key_values": key_values,
            "extraction_method": "easyocr",
            "timestamp": datetime.now().isoformat()
        }

    def _extract_key_values(self, text: str) -> Dict[str, Any]:
        """Extract key-value pairs from text using patterns"""
        key_values = {}
        
        # Common patterns for key-value extraction
        patterns = [
            # Pattern: "Key: Value"
            (r'([A-Za-z\s]+):\s*([^\n\r]+)', lambda m: (m.group(1).strip().lower().replace(' ', '_'), m.group(2).strip())),
            
            # Pattern: "Key Value" (for amounts, dates, etc.)
            (r'(total|amount|sum|summe|betrag)\s*:?\s*([0-9,.€$£]+)', lambda m: (m.group(1).lower(), m.group(2))),
            (r'(date|datum)\s*:?\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})', lambda m: (m.group(1).lower(), m.group(2))),
            (r'(invoice|rechnung)\s*(?:number|nummer|no\.?|nr\.?)\s*:?\s*([A-Z0-9-]+)', lambda m: ('invoice_number', m.group(2))),
            
            # German specific patterns
            (r'(rechnungsnummer)\s*:?\s*([A-Z0-9-]+)', lambda m: ('invoice_number', m.group(2))),
            (r'(rechnungsdatum)\s*:?\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})', lambda m: ('invoice_date', m.group(2))),
            (r'(gesamtbetrag|total)\s*:?\s*([0-9,.€]+)', lambda m: ('total_amount', m.group(2))),
            
            # Contact information
            (r'(email|e-mail)\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', lambda m: ('email', m.group(2))),
            (r'(phone|telefon|tel)\s*:?\s*([+0-9\s()-]+)', lambda m: ('phone', m.group(2).strip())),
        ]
        
        # Apply patterns
        for pattern, extractor in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                key, value = extractor(match)
                if key and value:
                    key_values[key] = value
        
        # Extract addresses (simplified)
        address_pattern = r'([A-Za-z\s]+)\s+([0-9]+[A-Za-z]?)\s*,?\s*([0-9]{4,5})\s+([A-Za-z\s]+)'
        address_matches = re.finditer(address_pattern, text)
        addresses = []
        for match in address_matches:
            addresses.append({
                "street": f"{match.group(1).strip()} {match.group(2)}",
                "postal_code": match.group(3),
                "city": match.group(4).strip()
            })
        
        if addresses:
            key_values['addresses'] = addresses
        
        # Extract all numbers (amounts, quantities, etc.)
        numbers = re.findall(r'[0-9,.€$£]+', text)
        if numbers:
            key_values['numbers_found'] = numbers
        
        # Extract all dates
        dates = re.findall(r'[0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4}', text)
        if dates:
            key_values['dates_found'] = dates
        
        return key_values