"""
GPU memory management and model caching optimizations
"""

import torch
import gc
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import psutil
import threading
import asyncio

class MemoryStrategy(Enum):
    AGGRESSIVE = "aggressive"  # Maximum performance, higher memory usage
    BALANCED = "balanced"     # Good balance of performance and memory
    CONSERVATIVE = "conservative"  # Lower memory usage, may impact performance

@dataclass
class ModelCacheEntry:
    model: Any
    last_used: float
    load_time: float
    memory_usage_mb: float
    access_count: int

class GPUMemoryManager:
    """Intelligent GPU memory management for optimal performance"""
    
    def __init__(self, strategy: MemoryStrategy = MemoryStrategy.BALANCED):
        self.strategy = strategy
        self.model_cache = {}
        self.memory_stats = {}
        self.cleanup_threshold = self._get_cleanup_threshold()
        self.max_cache_size = self._get_max_cache_size()
        self._lock = threading.Lock()
        
    def _get_cleanup_threshold(self) -> float:
        """Get memory cleanup threshold based on strategy"""
        if self.strategy == MemoryStrategy.AGGRESSIVE:
            return 0.95  # Use 95% of GPU memory before cleanup
        elif self.strategy == MemoryStrategy.BALANCED:
            return 0.85  # Use 85% of GPU memory before cleanup
        else:  # CONSERVATIVE
            return 0.75  # Use 75% of GPU memory before cleanup
    
    def _get_max_cache_size(self) -> int:
        """Get maximum number of models to cache"""
        if self.strategy == MemoryStrategy.AGGRESSIVE:
            return 5
        elif self.strategy == MemoryStrategy.BALANCED:
            return 3
        else:  # CONSERVATIVE
            return 2
    
    async def get_or_load_model(
        self, 
        model_key: str, 
        loader_func: Callable,
        *args, 
        **kwargs
    ) -> Any:
        """Get model from cache or load it"""
        
        with self._lock:
            # Check if model is in cache
            if model_key in self.model_cache:
                entry = self.model_cache[model_key]
                entry.last_used = time.time()
                entry.access_count += 1
                print(f"🎯 Model {model_key} retrieved from cache (used {entry.access_count} times)")
                return entry.model
            
            # Check if we need to free memory first
            await self._ensure_memory_available()
            
            # Load new model
            print(f"🔄 Loading model {model_key}...")
            start_time = time.time()
            
            try:
                model = await self._load_model_with_monitoring(loader_func, *args, **kwargs)
                load_time = time.time() - start_time
                
                # Calculate memory usage
                memory_usage = self._get_model_memory_usage()
                
                # Cache the model
                self.model_cache[model_key] = ModelCacheEntry(
                    model=model,
                    last_used=time.time(),
                    load_time=load_time,
                    memory_usage_mb=memory_usage,
                    access_count=1
                )
                
                print(f"✅ Model {model_key} loaded in {load_time:.2f}s, using {memory_usage:.1f}MB")
                
                # Cleanup old models if needed
                await self._cleanup_old_models()
                
                return model
                
            except Exception as e:
                print(f"❌ Failed to load model {model_key}: {e}")
                # Emergency cleanup and retry
                await self._emergency_cleanup()
                raise e
    
    async def _load_model_with_monitoring(self, loader_func: Callable, *args, **kwargs) -> Any:
        """Load model with memory monitoring"""
        
        # Clear cache before loading
        await self._smart_cache_clear()
        
        # Monitor memory during loading
        initial_memory = self._get_gpu_memory_usage()
        
        try:
            if asyncio.iscoroutinefunction(loader_func):
                model = await loader_func(*args, **kwargs)
            else:
                # Run in executor for non-async functions
                loop = asyncio.get_event_loop()
                model = await loop.run_in_executor(None, loader_func, *args, **kwargs)
            
            final_memory = self._get_gpu_memory_usage()
            print(f"📊 Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB")
            
            return model
            
        except torch.cuda.OutOfMemoryError:
            print("🚨 CUDA OOM during model loading - performing emergency cleanup")
            await self._emergency_cleanup()
            raise
    
    async def _ensure_memory_available(self):
        """Ensure sufficient GPU memory is available"""
        
        current_usage = self._get_gpu_memory_usage()
        total_memory = self._get_total_gpu_memory()
        usage_ratio = current_usage / total_memory
        
        if usage_ratio > self.cleanup_threshold:
            print(f"🧹 Memory usage at {usage_ratio:.1%}, performing cleanup...")
            await self._smart_cleanup()
    
    async def _smart_cleanup(self):
        """Intelligent memory cleanup"""
        
        # Strategy 1: Clear unused model cache entries
        await self._cleanup_unused_models()
        
        # Strategy 2: Smart CUDA cache clearing
        await self._smart_cache_clear()
        
        # Strategy 3: Garbage collection
        gc.collect()
        
        # Strategy 4: Force CUDA synchronization
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    async def _cleanup_unused_models(self):
        """Remove least recently used models from cache"""
        
        if len(self.model_cache) <= 1:
            return
        
        # Sort by last used time
        sorted_models = sorted(
            self.model_cache.items(),
            key=lambda x: x[1].last_used
        )
        
        # Remove oldest models if cache is too large
        while len(self.model_cache) > self.max_cache_size:
            model_key, entry = sorted_models.pop(0)
            
            print(f"🗑️ Removing cached model {model_key} (last used {time.time() - entry.last_used:.1f}s ago)")
            
            # Delete model and clear memory
            del entry.model
            del self.model_cache[model_key]
            
            # Force cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def _cleanup_old_models(self):
        """Remove models that haven't been used recently"""
        
        current_time = time.time()
        max_idle_time = 300  # 5 minutes
        
        to_remove = []
        for model_key, entry in self.model_cache.items():
            if current_time - entry.last_used > max_idle_time:
                to_remove.append(model_key)
        
        for model_key in to_remove:
            entry = self.model_cache[model_key]
            print(f"🗑️ Removing idle model {model_key} (idle for {current_time - entry.last_used:.1f}s)")
            
            del entry.model
            del self.model_cache[model_key]
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def _smart_cache_clear(self):
        """Smart CUDA cache clearing based on strategy"""
        
        if not torch.cuda.is_available():
            return
        
        if self.strategy == MemoryStrategy.AGGRESSIVE:
            # Only clear if really needed
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            
            if reserved > allocated * 1.5:  # Too much reserved vs allocated
                torch.cuda.empty_cache()
                
        elif self.strategy == MemoryStrategy.BALANCED:
            # Regular cache clearing
            torch.cuda.empty_cache()
            
        else:  # CONSERVATIVE
            # Aggressive cache clearing
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    async def _emergency_cleanup(self):
        """Emergency cleanup when CUDA OOM occurs"""
        
        print("🚨 Emergency GPU memory cleanup")
        
        # Clear all cached models
        for model_key in list(self.model_cache.keys()):
            entry = self.model_cache[model_key]
            del entry.model
            del self.model_cache[model_key]
        
        # Aggressive cleanup
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
        
        print("✅ Emergency cleanup completed")
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return 0.0
    
    def _get_total_gpu_memory(self) -> float:
        """Get total GPU memory in MB"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        return 0.0
    
    def _get_model_memory_usage(self) -> float:
        """Estimate memory usage of currently loaded models"""
        return self._get_gpu_memory_usage()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        
        stats = {
            "cached_models": len(self.model_cache),
            "strategy": self.strategy.value,
            "cleanup_threshold": self.cleanup_threshold,
            "max_cache_size": self.max_cache_size
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
                "gpu_memory_total_mb": torch.cuda.get_device_properties(0).total_memory / (1024 ** 2),
                "gpu_utilization_percent": torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
            })
        
        # Add system memory
        memory = psutil.virtual_memory()
        stats.update({
            "system_memory_used_mb": memory.used / (1024 ** 2),
            "system_memory_total_mb": memory.total / (1024 ** 2),
            "system_memory_percent": memory.percent
        })
        
        # Add cache info
        cache_info = []
        for model_key, entry in self.model_cache.items():
            cache_info.append({
                "model": model_key,
                "last_used_seconds_ago": time.time() - entry.last_used,
                "access_count": entry.access_count,
                "load_time": entry.load_time,
                "memory_usage_mb": entry.memory_usage_mb
            })
        
        stats["cache_details"] = cache_info
        
        return stats
    
    async def optimize_for_inference(self):
        """Optimize GPU settings for inference"""
        
        if torch.cuda.is_available():
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Set memory allocation strategy
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                if self.strategy == MemoryStrategy.AGGRESSIVE:
                    torch.cuda.set_per_process_memory_fraction(0.95)
                elif self.strategy == MemoryStrategy.BALANCED:
                    torch.cuda.set_per_process_memory_fraction(0.85)
                else:  # CONSERVATIVE
                    torch.cuda.set_per_process_memory_fraction(0.75)
            
            print(f"🚀 GPU optimized for {self.strategy.value} inference")

class ModelPreloader:
    """Preload and warm up models for faster inference"""
    
    def __init__(self, memory_manager: GPUMemoryManager):
        self.memory_manager = memory_manager
        self.preloaded_models = set()
    
    async def preload_models(self, model_configs: list):
        """Preload multiple models in optimal order"""
        
        print(f"🔄 Preloading {len(model_configs)} models...")
        
        # Sort by priority and estimated memory usage
        sorted_configs = sorted(model_configs, key=lambda x: (
            -x.get('priority', 0),  # Higher priority first
            x.get('estimated_memory_mb', 1000)  # Smaller models first
        ))
        
        for config in sorted_configs:
            try:
                model_key = config['key']
                loader_func = config['loader']
                args = config.get('args', ())
                kwargs = config.get('kwargs', {})
                
                await self.memory_manager.get_or_load_model(
                    model_key, loader_func, *args, **kwargs
                )
                
                self.preloaded_models.add(model_key)
                print(f"✅ Preloaded {model_key}")
                
            except Exception as e:
                print(f"❌ Failed to preload {config.get('key', 'unknown')}: {e}")
    
    async def warm_up_model(self, model_key: str, warm_up_func: Callable):
        """Warm up a specific model with dummy data"""
        
        try:
            print(f"🔥 Warming up model {model_key}...")
            
            start_time = time.time()
            await warm_up_func()
            warm_up_time = time.time() - start_time
            
            print(f"✅ Model {model_key} warmed up in {warm_up_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Failed to warm up {model_key}: {e}")