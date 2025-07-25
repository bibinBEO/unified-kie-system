import re
from typing import Dict, Any, List
import nltk
from datetime import datetime

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class TextProcessor:
    def __init__(self):
        self.currency_symbols = {'€', '$', '£', '¥', '₹', 'EUR', 'USD', 'GBP', 'JPY', 'INR'}
        
    def extract_key_values(self, text: str) -> Dict[str, Any]:
        """Extract key-value pairs from text using advanced patterns"""
        key_values = {}
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Extract various types of information
        key_values.update(self._extract_dates(cleaned_text))
        key_values.update(self._extract_amounts(cleaned_text))
        key_values.update(self._extract_identifiers(cleaned_text))
        key_values.update(self._extract_contact_info(cleaned_text))
        key_values.update(self._extract_addresses(cleaned_text))
        key_values.update(self._extract_companies(cleaned_text))
        key_values.update(self._extract_line_items(cleaned_text))
        
        return key_values
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere with parsing
        text = re.sub(r'[^\w\s.,:\-/()@€$£¥₹]', ' ', text)
        return text.strip()
    
    def _extract_dates(self, text: str) -> Dict[str, Any]:
        """Extract various date formats"""
        dates = {}
        
        # Date patterns
        patterns = [
            (r'invoice\s+date\s*:?\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})', 'invoice_date'),
            (r'rechnungsdatum\s*:?\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})', 'invoice_date'),
            (r'due\s+date\s*:?\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})', 'due_date'),
            (r'fälligkeitsdatum\s*:?\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})', 'due_date'),
            (r'date\s*:?\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})', 'date'),
            (r'datum\s*:?\s*([0-9]{1,2}[./-][0-9]{1,2}[./-][0-9]{2,4})', 'date'),
        ]
        
        for pattern, key in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dates[key] = match.group(1)
                break  # Take first match
        
        return dates
    
    def _extract_amounts(self, text: str) -> Dict[str, Any]:
        """Extract monetary amounts"""
        amounts = {}
        
        # Amount patterns
        patterns = [
            (r'total\s*:?\s*([0-9,.]+)\s*([€$£¥₹]|EUR|USD|GBP)', 'total_amount'),
            (r'gesamtbetrag\s*:?\s*([0-9,.]+)\s*([€$£¥₹]|EUR|USD|GBP)', 'total_amount'),
            (r'subtotal\s*:?\s*([0-9,.]+)\s*([€$£¥₹]|EUR|USD|GBP)', 'subtotal'),
            (r'zwischensumme\s*:?\s*([0-9,.]+)\s*([€$£¥₹]|EUR|USD|GBP)', 'subtotal'),
            (r'tax\s*:?\s*([0-9,.]+)\s*([€$£¥₹]|EUR|USD|GBP)', 'tax_amount'),
            (r'steuer\s*:?\s*([0-9,.]+)\s*([€$£¥₹]|EUR|USD|GBP)', 'tax_amount'),
            (r'mwst\s*:?\s*([0-9,.]+)\s*([€$£¥₹]|EUR|USD|GBP)', 'tax_amount'),
        ]
        
        for pattern, key in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount = match.group(1).replace(',', '.')
                currency = match.group(2)
                amounts[key] = {
                    'amount': float(amount) if amount.replace('.', '').isdigit() else amount,
                    'currency': currency
                }
                break
        
        return amounts
    
    def _extract_identifiers(self, text: str) -> Dict[str, Any]:
        """Extract various identifiers"""
        identifiers = {}
        
        patterns = [
            (r'invoice\s+(?:number|no\.?|nr\.?)\s*:?\s*([A-Z0-9\-]+)', 'invoice_number'),
            (r'rechnungsnummer\s*:?\s*([A-Z0-9\-]+)', 'invoice_number'),
            (r'order\s+(?:number|no\.?)\s*:?\s*([A-Z0-9\-]+)', 'order_number'),
            (r'bestellnummer\s*:?\s*([A-Z0-9\-]+)', 'order_number'),
            (r'customer\s+(?:number|no\.?|id)\s*:?\s*([A-Z0-9\-]+)', 'customer_number'),
            (r'kundennummer\s*:?\s*([A-Z0-9\-]+)', 'customer_number'),
            (r'lrn\s*:?\s*([A-Z0-9\-]+)', 'lrn'),
            (r'mrn\s*:?\s*([A-Z0-9\-]+)', 'mrn'),
            (r'eori\s*:?\s*([A-Z0-9\-]+)', 'eori_number'),
        ]
        
        for pattern, key in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                identifiers[key] = match.group(1)
                break
        
        return identifiers
    
    def _extract_contact_info(self, text: str) -> Dict[str, Any]:
        """Extract contact information"""
        contact = {}
        
        # Email pattern
        email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        emails = re.findall(email_pattern, text)
        if emails:
            contact['emails'] = emails
        
        # Phone pattern
        phone_pattern = r'(?:phone|tel|telefon)\s*:?\s*([+0-9\s()-]{8,})'
        phones = re.findall(phone_pattern, text, re.IGNORECASE)
        if phones:
            contact['phones'] = [phone.strip() for phone in phones]
        
        return contact
    
    def _extract_addresses(self, text: str) -> Dict[str, Any]:
        """Extract address information"""
        addresses = {}
        
        # Simple address pattern (street number, postal code, city)
        address_pattern = r'([A-Za-z\s]+)\s+([0-9]+[A-Za-z]?)\s*,?\s*([0-9]{4,5})\s+([A-Za-z\s]+)'
        matches = re.finditer(address_pattern, text)
        
        address_list = []
        for match in matches:
            address = {
                'street': f"{match.group(1).strip()} {match.group(2)}",
                'postal_code': match.group(3),
                'city': match.group(4).strip()
            }
            address_list.append(address)
        
        if address_list:
            addresses['addresses'] = address_list
        
        return addresses
    
    def _extract_companies(self, text: str) -> Dict[str, Any]:
        """Extract company names and information"""
        companies = {}
        
        # Company patterns
        patterns = [
            (r'(?:von|from|vendor|supplier)\s*:?\s*([A-Za-z\s&.,-]+(?:GmbH|AG|Ltd|LLC|Inc|Corp))', 'vendor_name'),
            (r'(?:an|to|customer|buyer)\s*:?\s*([A-Za-z\s&.,-]+(?:GmbH|AG|Ltd|LLC|Inc|Corp))', 'customer_name'),
            (r'([A-Za-z\s&.,-]+(?:GmbH|AG|Ltd|LLC|Inc|Corp))', 'company_names'),
        ]
        
        company_names = set()
        for pattern, key in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                company_name = match.group(1).strip()
                if len(company_name) > 3:  # Filter very short matches
                    if key == 'company_names':
                        company_names.add(company_name)
                    else:
                        companies[key] = company_name
                        break
        
        if company_names:
            companies['all_companies'] = list(company_names)
        
        return companies
    
    def _extract_line_items(self, text: str) -> Dict[str, Any]:
        """Extract line items from tables or lists"""
        line_items = {}
        
        # Look for table-like structures
        lines = text.split('\n')
        items = []
        
        for line in lines:
            # Pattern: Description Quantity Price Total
            item_pattern = r'(.+?)\s+([0-9,.]+)\s+([0-9,.]+)\s+([0-9,.]+)'
            match = re.search(item_pattern, line)
            if match:
                items.append({
                    'description': match.group(1).strip(),
                    'quantity': match.group(2),
                    'unit_price': match.group(3),
                    'total': match.group(4)
                })
        
        if items:
            line_items['line_items'] = items
        
        return line_items
    
    def detect_language(self, text: str) -> str:
        """Simple language detection"""
        german_words = ['der', 'die', 'das', 'und', 'ist', 'von', 'zu', 'mit', 'auf', 'für', 
                       'rechnung', 'datum', 'betrag', 'steuer', 'mwst', 'kunde', 'lieferant']
        english_words = ['the', 'and', 'is', 'of', 'to', 'with', 'on', 'for', 'invoice', 
                        'date', 'amount', 'tax', 'customer', 'vendor', 'total']
        
        text_lower = text.lower()
        german_count = sum(1 for word in german_words if word in text_lower)
        english_count = sum(1 for word in english_words if word in text_lower)
        
        if german_count > english_count:
            return 'de'
        elif english_count > german_count:
            return 'en'
        else:
            return 'unknown'
    
    def extract_table_data(self, text: str) -> List[Dict[str, Any]]:
        """Extract structured table data"""
        lines = text.split('\n')
        tables = []
        current_table = []
        
        for line in lines:
            # Detect table rows (multiple columns separated by spaces)
            columns = re.split(r'\s{2,}', line.strip())
            if len(columns) >= 2:
                current_table.append(columns)
            else:
                if current_table and len(current_table) > 1:
                    tables.append(current_table)
                current_table = []
        
        # Add last table if exists
        if current_table and len(current_table) > 1:
            tables.append(current_table)
        
        return tables