"""
Request queuing and async processing optimizations
"""

import asyncio
import time
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import uuid
import psutil
import torch

class RequestPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class ProcessingRequest:
    id: str
    priority: RequestPriority
    payload: Dict[str, Any]
    callback: Callable
    created_at: float
    timeout: float = 120.0  # Default 2 minutes
    
class SmartRequestQueue:
    """Intelligent request queue with load balancing and optimization"""
    
    def __init__(self, max_concurrent_requests: int = 4):
        self.max_concurrent = max_concurrent_requests
        self.queue = asyncio.PriorityQueue()
        self.active_requests = {}
        self.completed_requests = {}
        self.stats = {
            "total_processed": 0,
            "average_processing_time": 0.0,
            "queue_size": 0,
            "active_count": 0
        }
        self.running = False
        self.worker_tasks = []
        
    async def start(self):
        """Start the queue processing workers"""
        if self.running:
            return
            
        self.running = True
        
        # Start worker tasks based on system capabilities
        optimal_workers = self._calculate_optimal_workers()
        print(f"🚀 Starting {optimal_workers} queue workers")
        
        for i in range(optimal_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(worker)
    
    async def stop(self):
        """Stop all queue workers"""
        self.running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
    
    async def submit_request(
        self, 
        processing_func: Callable,
        payload: Dict[str, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 120.0
    ) -> str:
        """Submit a request for processing"""
        
        request_id = str(uuid.uuid4())
        
        # Wrap the processing function
        async def callback():
            try:
                start_time = time.time()
                result = await processing_func(**payload)
                processing_time = time.time() - start_time
                
                self.completed_requests[request_id] = {
                    "result": result,
                    "processing_time": processing_time,
                    "completed_at": time.time()
                }
                
                # Update stats
                self._update_stats(processing_time)
                
                return result
                
            except Exception as e:
                self.completed_requests[request_id] = {
                    "error": str(e),
                    "completed_at": time.time()
                }
                raise e
        
        request = ProcessingRequest(
            id=request_id,
            priority=priority,
            payload=payload,
            callback=callback,
            created_at=time.time(),
            timeout=timeout
        )
        
        # Add to queue with priority (negative for proper sorting)
        await self.queue.put((-priority.value, time.time(), request))
        self.stats["queue_size"] = self.queue.qsize()
        
        return request_id
    
    async def get_result(self, request_id: str, wait: bool = True) -> Optional[Dict[str, Any]]:
        """Get the result of a submitted request"""
        
        if request_id in self.completed_requests:
            return self.completed_requests[request_id]
        
        if not wait:
            return None
        
        # Wait for completion
        max_wait = 300  # 5 minutes max wait
        start_wait = time.time()
        
        while time.time() - start_wait < max_wait:
            if request_id in self.completed_requests:
                return self.completed_requests[request_id]
            
            await asyncio.sleep(0.1)
        
        return {"error": "Request timeout", "timeout": True}
    
    async def _worker(self, worker_id: str):
        """Queue worker that processes requests"""
        print(f"🔄 Queue worker {worker_id} started")
        
        while self.running:
            try:
                # Wait for next request with timeout
                try:
                    _, queued_time, request = await asyncio.wait_for(
                        self.queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check if request has expired
                if time.time() - request.created_at > request.timeout:
                    print(f"⏰ Request {request.id} expired")
                    self.completed_requests[request.id] = {
                        "error": "Request expired",
                        "expired": True
                    }
                    continue
                
                # Process the request
                self.active_requests[request.id] = {
                    "started_at": time.time(),
                    "worker": worker_id
                }
                self.stats["active_count"] = len(self.active_requests)
                
                wait_time = time.time() - queued_time
                print(f"🔨 Worker {worker_id} processing {request.id} (waited {wait_time:.2f}s)")
                
                try:
                    await request.callback()
                    print(f"✅ Worker {worker_id} completed {request.id}")
                    
                except Exception as e:
                    print(f"❌ Worker {worker_id} failed {request.id}: {e}")
                
                finally:
                    # Clean up
                    if request.id in self.active_requests:
                        del self.active_requests[request.id]
                    self.stats["active_count"] = len(self.active_requests)
                    self.stats["queue_size"] = self.queue.qsize()
                
            except Exception as e:
                print(f"❌ Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on system resources"""
        
        # Get system info
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # GPU info if available
        gpu_memory_gb = 0
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Calculate based on resources
        if gpu_memory_gb > 16:  # High-end GPU
            optimal = min(6, cpu_count)
        elif gpu_memory_gb > 8:  # Mid-range GPU
            optimal = min(4, cpu_count)
        elif gpu_memory_gb > 0:  # Any GPU
            optimal = min(2, cpu_count)
        else:  # CPU only
            optimal = min(2, cpu_count // 2)
        
        # Ensure minimum and maximum
        return max(1, min(optimal, self.max_concurrent))
    
    def _update_stats(self, processing_time: float):
        """Update processing statistics"""
        self.stats["total_processed"] += 1
        
        # Update running average
        current_avg = self.stats["average_processing_time"]
        count = self.stats["total_processed"]
        
        new_avg = ((current_avg * (count - 1)) + processing_time) / count
        self.stats["average_processing_time"] = new_avg
    
    def get_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            "running": self.running,
            "queue_size": self.queue.qsize(),
            "active_requests": len(self.active_requests),
            "workers": len(self.worker_tasks),
            "stats": self.stats.copy(),
            "system_load": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "gpu_memory_allocated": torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            }
        }

class LoadBalancer:
    """Load balancer for distributing requests across multiple processing endpoints"""
    
    def __init__(self):
        self.endpoints = []
        self.endpoint_stats = {}
        
    def add_endpoint(self, name: str, process_func: Callable, weight: float = 1.0):
        """Add a processing endpoint"""
        self.endpoints.append({
            "name": name,
            "process_func": process_func,
            "weight": weight,
            "active_requests": 0
        })
        self.endpoint_stats[name] = {
            "total_requests": 0,
            "success_count": 0,
            "error_count": 0,
            "average_time": 0.0
        }
    
    async def process_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process request using optimal endpoint"""
        
        if not self.endpoints:
            raise Exception("No processing endpoints available")
        
        # Select best endpoint
        endpoint = self._select_endpoint()
        endpoint["active_requests"] += 1
        
        try:
            start_time = time.time()
            result = await endpoint["process_func"](**payload)
            processing_time = time.time() - start_time
            
            # Update stats
            stats = self.endpoint_stats[endpoint["name"]]
            stats["total_requests"] += 1
            stats["success_count"] += 1
            
            # Update average time
            current_avg = stats["average_time"]
            count = stats["success_count"]
            stats["average_time"] = ((current_avg * (count - 1)) + processing_time) / count
            
            result["endpoint_used"] = endpoint["name"]
            result["processing_time"] = processing_time
            
            return result
            
        except Exception as e:
            self.endpoint_stats[endpoint["name"]]["error_count"] += 1
            raise e
            
        finally:
            endpoint["active_requests"] -= 1
    
    def _select_endpoint(self) -> Dict[str, Any]:
        """Select optimal endpoint based on load and performance"""
        
        # Score each endpoint
        best_endpoint = None
        best_score = float('inf')
        
        for endpoint in self.endpoints:
            stats = self.endpoint_stats[endpoint["name"]]
            
            # Calculate score (lower is better)
            load_factor = endpoint["active_requests"] / endpoint["weight"]
            error_rate = stats["error_count"] / max(1, stats["total_requests"])
            avg_time = stats["average_time"]
            
            score = load_factor + (error_rate * 10) + (avg_time / 10)
            
            if score < best_score:
                best_score = score
                best_endpoint = endpoint
        
        return best_endpoint or self.endpoints[0]