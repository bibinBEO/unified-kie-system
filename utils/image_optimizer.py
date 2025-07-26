"""
Image optimization utilities for faster processing
"""

from PIL import Image, ImageFilter, ImageEnhance
import io
import asyncio
from typing import Tuple, Optional
import numpy as np

class ImageOptimizer:
    """Optimize images for faster OCR and ML processing"""
    
    def __init__(self):
        self.optimal_size = 1536  # Optimal dimension for speed/quality balance
        self.max_file_size_mb = 5  # Max file size in MB
    
    async def optimize_for_ocr(self, image: Image.Image) -> Image.Image:
        """Optimize image specifically for OCR processing"""
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for optimal processing
        image = await self._smart_resize(image)
        
        # Enhance for better OCR
        image = await self._enhance_for_ocr(image)
        
        return image
    
    async def optimize_for_vllm(self, image: Image.Image) -> Image.Image:
        """Optimize image specifically for vLLM processing"""
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize with aspect ratio preservation
        image = await self._smart_resize(image, max_dimension=1536)
        
        # Light enhancement for vision models
        image = await self._enhance_for_vision_model(image)
        
        return image
    
    async def _smart_resize(self, image: Image.Image, max_dimension: int = None) -> Image.Image:
        """Smart resizing that preserves aspect ratio and optimizes for processing"""
        max_dim = max_dimension or self.optimal_size
        width, height = image.size
        
        # Skip if already optimal size
        if width <= max_dim and height <= max_dim:
            return image
        
        # Calculate new dimensions
        ratio = min(max_dim / width, max_dim / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # Ensure dimensions are even (better for some ML models)
        new_width = new_width if new_width % 2 == 0 else new_width - 1
        new_height = new_height if new_height % 2 == 0 else new_height - 1
        
        # Use high-quality resampling
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        print(f"🖼️ Resized: {width}x{height} -> {new_width}x{new_height} (ratio: {ratio:.2f})")
        
        return resized
    
    async def _enhance_for_ocr(self, image: Image.Image) -> Image.Image:
        """Enhance image for better OCR accuracy"""
        loop = asyncio.get_event_loop()
        
        def enhance():
            # Increase contrast slightly
            enhancer = ImageEnhance.Contrast(image)
            enhanced = enhancer.enhance(1.1)
            
            # Increase sharpness slightly
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.1)
            
            return enhanced
        
        return await loop.run_in_executor(None, enhance)
    
    async def _enhance_for_vision_model(self, image: Image.Image) -> Image.Image:
        """Light enhancement for vision models (they prefer natural images)"""
        loop = asyncio.get_event_loop()
        
        def enhance():
            # Very light contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
            enhanced = enhancer.enhance(1.05)
            
            return enhanced
        
        return await loop.run_in_executor(None, enhance)
    
    async def compress_image(self, image: Image.Image, quality: int = 85) -> bytes:
        """Compress image to reduce transfer time"""
        loop = asyncio.get_event_loop()
        
        def compress():
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality, optimize=True)
            return buffer.getvalue()
        
        return await loop.run_in_executor(None, compress)
    
    def estimate_processing_time(self, image: Image.Image) -> float:
        """Estimate processing time based on image dimensions"""
        width, height = image.size
        pixels = width * height
        
        # Rough estimates based on typical processing (in seconds)
        if pixels < 500_000:  # Small images
            return 0.5
        elif pixels < 2_000_000:  # Medium images  
            return 1.5
        elif pixels < 5_000_000:  # Large images
            return 3.0
        else:  # Very large images
            return 6.0
    
    async def preprocess_batch(self, images: list) -> list:
        """Process multiple images concurrently"""
        tasks = [self.optimize_for_vllm(img) for img in images]
        return await asyncio.gather(*tasks)

class ImageQualityAnalyzer:
    """Analyze image quality for processing optimization"""
    
    @staticmethod
    def analyze_quality(image: Image.Image) -> dict:
        """Analyze image quality metrics"""
        
        # Convert to grayscale for analysis
        gray = image.convert('L')
        img_array = np.array(gray)
        
        # Calculate metrics
        mean_brightness = float(np.mean(img_array))
        contrast = float(np.std(img_array))
        
        # Estimate sharpness using Laplacian variance
        def estimate_sharpness():
            try:
                from scipy import ndimage
                laplacian = ndimage.laplace(img_array)
                return float(np.var(laplacian))
            except ImportError:
                # Fallback without scipy
                return contrast  # Use contrast as proxy
        
        sharpness = estimate_sharpness()
        
        # Quality assessment
        quality_score = min(100, (contrast / 128 * 50) + (sharpness / 1000 * 30) + 20)
        
        return {
            "brightness": mean_brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "quality_score": quality_score,
            "dimensions": image.size,
            "recommended_processing": "standard" if quality_score > 60 else "enhanced"
        }