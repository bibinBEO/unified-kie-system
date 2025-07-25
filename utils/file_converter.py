import fitz  # PyMuPDF
import tempfile
import os
from PIL import Image
from typing import List, Dict, Any
import asyncio

class FileConverter:
    def __init__(self):
        self.supported_formats = {'.pdf', '.png', '.jpg', '.jpeg', '.docx', '.txt', '.csv'}
    
    async def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[Dict[str, Any]]:
        """Convert PDF pages to images"""
        def convert():
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Create transformation matrix for desired DPI
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                
                # Save as temporary image
                temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                pix.save(temp_img.name)
                
                images.append({
                    'path': temp_img.name,
                    'page_number': page_num + 1,
                    'temporary': True,
                    'width': pix.width,
                    'height': pix.height
                })
            
            doc.close()
            return images
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, convert)
    
    async def image_to_pil(self, image_path: str) -> Image.Image:
        """Convert image file to PIL Image"""
        def load_image():
            return Image.open(image_path).convert('RGB')
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, load_image)
    
    def cleanup_temp_files(self, file_list: List[Dict[str, Any]]):
        """Clean up temporary files"""
        for file_info in file_list:
            if file_info.get('temporary') and os.path.exists(file_info['path']):
                try:
                    os.unlink(file_info['path'])
                except OSError:
                    pass  # Ignore cleanup errors
    
    def validate_file_type(self, filename: str) -> bool:
        """Validate if file type is supported"""
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.supported_formats
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file metadata"""
        stat = os.stat(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        
        return {
            'filename': os.path.basename(file_path),
            'extension': ext,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'supported': ext in self.supported_formats
        }