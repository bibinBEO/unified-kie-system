#!/usr/bin/env python3
"""
Production deployment test for mira.beo-software.de
"""

import requests
import time
import io
from PIL import Image, ImageDraw, ImageFont

def create_german_invoice():
    """Create a German invoice for testing"""
    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
        font_bold = font
    
    # German invoice content
    draw.text((50, 30), "RECHNUNG", fill='black', font=font_bold)
    draw.text((50, 70), "Musterfirma GmbH", fill='black', font=font)
    draw.text((50, 90), "Musterstraße 123", fill='black', font=font)
    draw.text((50, 110), "12345 Musterstadt", fill='black', font=font)
    draw.text((50, 150), "Rechnungsnummer: RE-2025-001", fill='black', font=font)
    draw.text((50, 170), "Datum: 26.01.2025", fill='black', font=font)
    draw.text((50, 210), "Kunde: Testkunde AG", fill='black', font=font)
    draw.text((50, 250), "Leistung: Dokumentenverarbeitung", fill='black', font=font)
    draw.text((50, 290), "Betrag: 250,00 EUR", fill='black', font=font)
    draw.text((50, 330), "Gesamtsumme: 250,00 EUR", fill='black', font=font_bold)
    
    return img

def test_production():
    """Test production deployment"""
    print("🚀 Production Deployment Test")
    print("🌐 Server: mira.beo-software.de")
    print("=" * 50)
    
    # Test health endpoint
    try:
        start = time.time()
        health_response = requests.get("http://mira.beo-software.de/health", timeout=10)
        health_time = time.time() - start
        
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Health check: {health_time:.3f}s")
            print(f"📊 Status: {health_data.get('status')}")
            print(f"📊 vLLM ready: {health_data.get('vllm_ready', False)}")
            print(f"📊 Fallback ready: {health_data.get('fallback_ready', False)}")
            print(f"📊 Version: {health_data.get('version')}")
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
            return
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # Test German invoice processing
    print(f"\n📄 Testing German invoice processing...")
    test_image = create_german_invoice()
    
    img_buffer = io.BytesIO()
    test_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    try:
        start = time.time()
        files = {'file': ('german_invoice.png', img_buffer, 'image/png')}
        data = {
            'extraction_type': 'invoice',
            'language': 'de'
        }
        
        response = requests.post(
            "http://mira.beo-software.de/extract/file",
            files=files,
            data=data,
            timeout=30
        )
        
        processing_time = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Processing completed in {processing_time:.2f}s")
            print(f"📊 Method: {result.get('extraction_method', 'unknown')}")
            
            if 'processing_metadata' in result:
                meta = result['processing_metadata']
                print(f"📊 Server time: {meta.get('processing_time', 0):.2f}s")
                print(f"📊 Method used: {meta.get('method_used', 'unknown')}")
                print(f"📊 File size: {meta.get('file_size_mb', 0):.2f}MB")
                print(f"📊 vLLM available: {meta.get('vllm_available', False)}")
                print(f"📊 Fallback used: {meta.get('fallback_used', False)}")
            
            # Check extraction quality
            if 'raw_text' in result and result['raw_text']:
                raw_text = result['raw_text']
                print(f"\n📝 Extraction quality check:")
                
                # Check for German terms
                german_terms = ['Rechnung', 'Rechnungsnummer', 'Betrag', 'EUR', 'Datum']
                found_terms = [term for term in german_terms if term.lower() in raw_text.lower()]
                print(f"   German terms found: {len(found_terms)}/{len(german_terms)} - {found_terms}")
                
                # Show preview
                preview = raw_text[:200] + "..." if len(raw_text) > 200 else raw_text
                print(f"   Text preview: {preview}")
            
            # Performance assessment
            print(f"\n🎯 Performance Assessment:")
            if processing_time < 5:
                print("🟢 EXCELLENT performance (< 5s)")
            elif processing_time < 10:
                print("🟡 GOOD performance (< 10s)")
            elif processing_time < 30:
                print("🟠 ACCEPTABLE performance (< 30s)")
            else:
                print("🔴 NEEDS IMPROVEMENT (> 30s)")
                
        else:
            print(f"❌ Extraction failed: {response.status_code}")
            print(f"Response: {response.text[:300]}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print(f"\n🏁 Production test completed")

if __name__ == "__main__":
    test_production()