import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import re

class SchemaManager:
    def __init__(self):
        self.schemas_dir = Path("schemas")
        self.schemas_dir.mkdir(exist_ok=True)
        self.schemas = {}
        self._load_default_schemas()
    
    def _load_default_schemas(self):
        """Load default schemas for different document types"""
        
        # Invoice Schema (from DocReader project)
        self.schemas["invoice"] = {
            "name": "Invoice Processing",
            "description": "Schema for invoices and commercial documents",
            "fields": {
                "vendor_name": {"type": "string", "required": True},
                "invoice_number": {"type": "string", "required": True},
                "invoice_date": {"type": "date", "required": True},
                "due_date": {"type": "date", "required": False},
                "subtotal": {"type": "number", "required": True},
                "tax_total": {"type": "number", "required": True},
                "grand_total": {"type": "number", "required": True},
                "currency": {"type": "string", "required": True},
                "line_items": {
                    "type": "array",
                    "items": {
                        "description": {"type": "string", "required": True},
                        "qty": {"type": "number", "required": True},
                        "unit_price": {"type": "number", "required": True},
                        "total": {"type": "number", "required": True}
                    }
                },
                "customer_name": {"type": "string", "required": False},
                "customer_address": {"type": "string", "required": False},
                "payment_terms": {"type": "string", "required": False}
            }
        }
        
        # German Customs Declaration Schema
        self.schemas["customs_declaration"] = {
            "name": "German Customs Export Declaration",
            "description": "Schema for German customs export declarations (Ausfuhranmeldung)",
            "fields": {
                "lrn": {"type": "string", "required": True, "description": "Local Reference Number"},
                "mrn": {"type": "string", "required": False, "description": "Movement Reference Number"},
                "eori_nummer": {"type": "string", "required": True, "description": "EORI Number"},
                "anmeldedatum": {"type": "date", "required": True, "description": "Declaration date"},
                "ausgangsdatum": {"type": "date", "required": False, "description": "Exit date"},
                "anmelder": {
                    "type": "object",
                    "fields": {
                        "name": {"type": "string", "required": True},
                        "adresse": {
                            "type": "object",
                            "fields": {
                                "strasse": {"type": "string", "required": True},
                                "plz": {"type": "string", "required": True},
                                "ort": {"type": "string", "required": True},
                                "land": {"type": "string", "required": True}
                            }
                        },
                        "tin": {"type": "string", "required": False},
                        "kontakt": {
                            "type": "object",
                            "fields": {
                                "telefon": {"type": "string", "required": False},
                                "email": {"type": "string", "required": False}
                            }
                        }
                    }
                },
                "ausfuhrer": {
                    "type": "object",
                    "fields": {
                        "name": {"type": "string", "required": True},
                        "adresse": {"type": "object", "required": True}
                    }
                },
                "empfanger": {
                    "type": "object",
                    "fields": {
                        "name": {"type": "string", "required": False},
                        "adresse": {"type": "object", "required": False}
                    }
                },
                "position": {
                    "type": "array",
                    "items": {
                        "warenbezeichnung": {"type": "string", "required": True},
                        "warennummer": {"type": "string", "required": True},
                        "ursprungsland": {"type": "string", "required": True},
                        "bestimmungsland": {"type": "string", "required": False},
                        "menge": {"type": "number", "required": True},
                        "gewicht": {"type": "number", "required": False},
                        "wert": {"type": "number", "required": True},
                        "waehrung": {"type": "string", "required": True}
                    }
                },
                "gestellungszollstelle": {"type": "string", "required": False},
                "ausfuhrzollstelle": {"type": "string", "required": False},
                "verkehrszweig": {"type": "string", "required": False},
                "verfahren": {"type": "string", "required": False},
                "bewilligung": {"type": "string", "required": False}
            }
        }
        
        # Generic document schema
        self.schemas["generic"] = {
            "name": "Generic Document",
            "description": "Schema for general document processing",
            "fields": {
                "document_type": {"type": "string", "required": False},
                "title": {"type": "string", "required": False},
                "date": {"type": "date", "required": False},
                "sender": {"type": "string", "required": False},
                "recipient": {"type": "string", "required": False},
                "reference_number": {"type": "string", "required": False},
                "content_summary": {"type": "string", "required": False},
                "key_information": {
                    "type": "object",
                    "fields": {
                        "names": {"type": "array", "items": {"type": "string"}},
                        "dates": {"type": "array", "items": {"type": "date"}},
                        "amounts": {"type": "array", "items": {"type": "number"}},
                        "addresses": {"type": "array", "items": {"type": "string"}},
                        "phone_numbers": {"type": "array", "items": {"type": "string"}},
                        "email_addresses": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "extracted_tables": {"type": "array", "required": False},
                "metadata": {
                    "type": "object",
                    "fields": {
                        "language": {"type": "string", "required": False},
                        "confidence": {"type": "number", "required": False},
                        "page_count": {"type": "number", "required": False}
                    }
                }
            }
        }

    def get_available_schemas(self) -> List[str]:
        """Get list of available schema names"""
        return list(self.schemas.keys())

    def get_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Get schema by name"""
        return self.schemas.get(schema_name)

    def get_default_schema_name(self) -> str:
        """Get default schema name"""
        return "generic"

    def detect_document_type(self, text: str) -> str:
        """Detect document type from text content"""
        text_lower = text.lower()
        
        # Check for invoice indicators
        invoice_keywords = [
            'invoice', 'rechnung', 'bill', 'faktura', 'total', 'subtotal',
            'tax', 'steuer', 'amount', 'betrag', 'due date', 'fälligkeit'
        ]
        invoice_score = sum(1 for keyword in invoice_keywords if keyword in text_lower)
        
        # Check for customs indicators
        customs_keywords = [
            'ausfuhranmeldung', 'customs', 'zoll', 'eori', 'lrn', 'mrn',
            'export', 'ausfuhr', 'warennummer', 'ursprungsland', 'zollstelle'
        ]
        customs_score = sum(1 for keyword in customs_keywords if keyword in text_lower)
        
        # Determine type based on scores
        if customs_score >= 3:
            return "customs_declaration"
        elif invoice_score >= 3:
            return "invoice"
        else:
            return "generic"

    def apply_schema(self, extraction_result: Dict[str, Any], schema_name: str) -> Dict[str, Any]:
        """Apply schema to structure extracted data"""
        schema = self.get_schema(schema_name)
        if not schema:
            return extraction_result
        
        structured_data = {
            "schema_applied": schema_name,
            "schema_version": "1.0",
            "extraction_timestamp": extraction_result.get("timestamp"),
            "confidence_score": self._calculate_confidence(extraction_result, schema),
            "structured_fields": {}
        }
        
        # Extract fields based on schema
        raw_data = self._get_raw_data(extraction_result)
        structured_data["structured_fields"] = self._extract_schema_fields(
            raw_data, schema["fields"]
        )
        
        # Add validation results
        structured_data["validation"] = self._validate_against_schema(
            structured_data["structured_fields"], schema["fields"]
        )
        
        return structured_data

    def _get_raw_data(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get the raw extracted data from various sources"""
        raw_data = {}
        
        # Combine from different extraction sources
        if "key_values" in extraction_result:
            raw_data.update(extraction_result["key_values"])
        
        if "unified_extraction" in extraction_result:
            raw_data.update(extraction_result["unified_extraction"])
        
        if "all_key_values" in extraction_result:
            raw_data.update(extraction_result["all_key_values"])
        
        # Include text for pattern matching
        if "combined_text" in extraction_result:
            raw_data["_full_text"] = extraction_result["combined_text"]
        elif "raw_text" in extraction_result:
            raw_data["_full_text"] = extraction_result["raw_text"]
        
        return raw_data

    def _extract_schema_fields(self, raw_data: Dict[str, Any], schema_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fields according to schema definition"""
        structured = {}
        
        for field_name, field_config in schema_fields.items():
            value = self._extract_field_value(raw_data, field_name, field_config)
            if value is not None:
                structured[field_name] = value
        
        return structured

    def _extract_field_value(self, raw_data: Dict[str, Any], field_name: str, field_config: Dict[str, Any]):
        """Extract a specific field value"""
        # Direct match
        if field_name in raw_data:
            return self._convert_field_value(raw_data[field_name], field_config)
        
        # Try various field name variations
        variations = self._generate_field_variations(field_name)
        for variation in variations:
            if variation in raw_data:
                return self._convert_field_value(raw_data[variation], field_config)
        
        # Pattern matching in full text
        if "_full_text" in raw_data:
            pattern_value = self._extract_by_pattern(raw_data["_full_text"], field_name, field_config)
            if pattern_value:
                return pattern_value
        
        return None

    def _generate_field_variations(self, field_name: str) -> List[str]:
        """Generate variations of field names for matching"""
        variations = [field_name]
        
        # Add common variations
        variations.extend([
            field_name.lower(),
            field_name.upper(),
            field_name.replace('_', ' '),
            field_name.replace('_', '-'),
            field_name.replace(' ', '_'),
            field_name.replace('-', '_'),
        ])
        
        # German-English mappings for common fields
        mappings = {
            'invoice_number': ['rechnungsnummer', 'invoice_no', 'bill_number'],
            'invoice_date': ['rechnungsdatum', 'date'],
            'vendor_name': ['lieferant', 'verkäufer', 'supplier'],
            'customer_name': ['kunde', 'käufer', 'buyer'],
            'total': ['gesamtbetrag', 'summe', 'grand_total'],
            'subtotal': ['zwischensumme', 'nettobetrag'],
            'tax': ['steuer', 'mwst', 'vat'],
            'lrn': ['lokale_referenznummer', 'local_reference_number'],
            'eori_nummer': ['eori', 'eori_number'],
            'anmelder': ['declarant', 'deklarant'],
            'ausfuhrer': ['exporter', 'ausführer']
        }
        
        if field_name in mappings:
            variations.extend(mappings[field_name])
        
        return list(set(variations))  # Remove duplicates

    def _extract_by_pattern(self, text: str, field_name: str, field_config: Dict[str, Any]):
        """Extract field value using pattern matching"""
        field_type = field_config.get("type", "string")
        
        if field_type == "date":
            return self._extract_date_pattern(text, field_name)
        elif field_type == "number":
            return self._extract_number_pattern(text, field_name)
        elif field_type == "string":
            return self._extract_string_pattern(text, field_name)
        
        return None

    def _extract_date_pattern(self, text: str, field_name: str) -> Optional[str]:
        """Extract date using patterns"""
        date_patterns = [
            r'\d{1,2}[./]\d{1,2}[./]\d{4}',  # DD/MM/YYYY or DD.MM.YYYY
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD
            r'\d{1,2}\s+\w+\s+\d{4}',        # DD Month YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]  # Return first match
        
        return None

    def _extract_number_pattern(self, text: str, field_name: str) -> Optional[float]:
        """Extract number using patterns"""
        # Look for currency amounts
        currency_pattern = r'[\d,.]+(?:\s*(?:EUR|USD|GBP|€|\$|£))?'
        matches = re.findall(currency_pattern, text)
        
        if matches:
            # Clean and convert first match
            number_str = matches[0].replace(',', '.').replace(' ', '')
            try:
                return float(re.sub(r'[^\d.]', '', number_str))
            except ValueError:
                pass
        
        return None

    def _extract_string_pattern(self, text: str, field_name: str) -> Optional[str]:
        """Extract string value using patterns"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated NLP techniques
        
        lines = text.split('\n')
        for line in lines:
            if field_name.lower() in line.lower():
                # Extract value after colon or similar delimiter
                if ':' in line:
                    return line.split(':', 1)[1].strip()
        
        return None

    def _convert_field_value(self, value: Any, field_config: Dict[str, Any]) -> Any:
        """Convert field value to expected type"""
        field_type = field_config.get("type", "string")
        
        if value is None:
            return None
        
        try:
            if field_type == "string":
                return str(value)
            elif field_type == "number":
                if isinstance(value, str):
                    # Clean numeric strings
                    cleaned = re.sub(r'[^\d.,\-]', '', value)
                    cleaned = cleaned.replace(',', '.')
                    return float(cleaned)
                return float(value)
            elif field_type == "date":
                # Keep as string for now, could add date parsing
                return str(value)
            elif field_type == "array":
                if isinstance(value, list):
                    return value
                return [value]
            elif field_type == "object":
                if isinstance(value, dict):
                    return value
                return {"value": value}
        except (ValueError, TypeError):
            pass
        
        return value

    def _validate_against_schema(self, structured_fields: Dict[str, Any], schema_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Validate structured data against schema"""
        validation_result = {
            "is_valid": True,
            "missing_required_fields": [],
            "field_validations": {},
            "completeness_score": 0.0
        }
        
        total_fields = len(schema_fields)
        found_fields = 0
        
        for field_name, field_config in schema_fields.items():
            is_required = field_config.get("required", False)
            has_value = field_name in structured_fields and structured_fields[field_name] is not None
            
            if has_value:
                found_fields += 1
                validation_result["field_validations"][field_name] = {"status": "found", "valid": True}
            elif is_required:
                validation_result["missing_required_fields"].append(field_name)
                validation_result["is_valid"] = False
                validation_result["field_validations"][field_name] = {"status": "missing", "valid": False}
            else:
                validation_result["field_validations"][field_name] = {"status": "optional_missing", "valid": True}
        
        validation_result["completeness_score"] = found_fields / total_fields if total_fields > 0 else 0.0
        
        return validation_result

    def _calculate_confidence(self, extraction_result: Dict[str, Any], schema: Dict[str, Any]) -> float:
        """Calculate confidence score for extraction"""
        # Simple confidence calculation based on data availability
        base_confidence = 0.5
        
        # Boost confidence if structured data is present
        if "key_values" in extraction_result:
            base_confidence += 0.2
        
        if "unified_extraction" in extraction_result:
            base_confidence += 0.2
        
        # Boost based on extraction method
        if extraction_result.get("extraction_method") == "nanonets_ocr_s":
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)