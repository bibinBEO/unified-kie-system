import os
from pathlib import Path
from typing import Dict, Any

class Config:
    def __init__(self):
        # Base directories
        self.BASE_DIR = Path(__file__).parent.parent
        self.UPLOAD_DIR = self.BASE_DIR / "uploads"
        self.RESULTS_DIR = self.BASE_DIR / "results"
        self.MODELS_DIR = self.BASE_DIR / "models"
        self.LOGS_DIR = self.BASE_DIR / "logs"
        
        # Server configuration
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", 8000))
        self.WORKERS = int(os.getenv("WORKERS", 1))  # Single worker for GPU models
        
        # Model configuration
        self.NANONETS_MODEL = os.getenv("NANONETS_MODEL", "nanonets/Nanonets-OCR-s")
        self.LAYOUTLM_MODEL = os.getenv("LAYOUTLM_MODEL", "microsoft/layoutlmv3-base")
        self.EASYOCR_LANGUAGES = os.getenv("EASYOCR_LANGUAGES", "en,de").split(",")
        
        # Processing configuration
        self.MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024))  # 50MB
        self.MAX_PAGES = int(os.getenv("MAX_PAGES", 20))
        self.PROCESSING_TIMEOUT = int(os.getenv("PROCESSING_TIMEOUT", 300))  # 5 minutes
        
        # GPU configuration
        self.FORCE_CPU = os.getenv("FORCE_CPU", "false").lower() == "true"
        self.GPU_MEMORY_FRACTION = float(os.getenv("GPU_MEMORY_FRACTION", 0.8))
        
        # Storage configuration
        self.CLEANUP_UPLOADS = os.getenv("CLEANUP_UPLOADS", "true").lower() == "true"
        self.RESULTS_RETENTION_DAYS = int(os.getenv("RESULTS_RETENTION_DAYS", 7))
        
        # Security
        self.ALLOWED_EXTENSIONS = set(os.getenv("ALLOWED_EXTENSIONS", ".pdf,.png,.jpg,.jpeg,.docx,.txt,.csv").split(","))
        self.MAX_URL_SIZE = int(os.getenv("MAX_URL_SIZE", 10 * 1024 * 1024))  # 10MB
        
        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Create directories
        for directory in [self.UPLOAD_DIR, self.RESULTS_DIR, self.MODELS_DIR, self.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            "nanonets_model": self.NANONETS_MODEL,
            "layoutlm_model": self.LAYOUTLM_MODEL,
            "easyocr_languages": self.EASYOCR_LANGUAGES,
            "force_cpu": self.FORCE_CPU,
            "gpu_memory_fraction": self.GPU_MEMORY_FRACTION
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration"""
        return {
            "max_file_size": self.MAX_FILE_SIZE,
            "max_pages": self.MAX_PAGES,
            "processing_timeout": self.PROCESSING_TIMEOUT,
            "allowed_extensions": self.ALLOWED_EXTENSIONS,
            "max_url_size": self.MAX_URL_SIZE
        }
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        return {
            "host": self.HOST,
            "port": self.PORT,
            "workers": self.WORKERS,
            "log_level": self.LOG_LEVEL
        }