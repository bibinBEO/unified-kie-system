#!/usr/bin/env python3
"""
Performance benchmark test for optimized vLLM system
"""

import asyncio
import aiohttp
import time
import json
from concurrent.futures import ThreadPoolExecutor
import statistics
from PIL import Image, ImageDraw, ImageFont
import io
import base64

class PerformanceBenchmark:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        
    async def run_benchmark(self):
        """Run comprehensive performance benchmark"""
        
        print("🚀 Starting Performance Benchmark")
        print("=" * 50)
        
        # Test 1: Health check latency
        await self._test_health_check_latency()
        
        # Test 2: Single document processing speed
        await self._test_single_document_processing()
        
        # Test 3: Concurrent processing throughput
        await self._test_concurrent_processing()
        
        # Test 4: Different image sizes
        await self._test_image_size_impact()
        
        # Test 5: Memory usage monitoring
        await self._test_memory_efficiency()
        
        # Generate report
        self._generate_performance_report()
    
    async def _test_health_check_latency(self):
        """Test health check response time"""
        print("\n🔍 Testing Health Check Latency...")
        
        latencies = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(10):
                start = time.time()
                try:
                    async with session.get(f"{self.base_url}/health") as response:
                        await response.json()
                        latency = time.time() - start
                        latencies.append(latency)
                        print(f"  Health check {i+1}: {latency:.3f}s")
                except Exception as e:
                    print(f"  Health check {i+1}: FAILED - {e}")
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            print(f"✅ Average health check latency: {avg_latency:.3f}s")
            self.results.append({
                "test": "health_check_latency",
                "average_ms": avg_latency * 1000,
                "samples": len(latencies)
            })
        else:
            print("❌ Health check test failed")
    
    async def _test_single_document_processing(self):
        """Test single document processing speed"""
        print("\n📄 Testing Single Document Processing Speed...")
        
        # Create test invoice image
        test_image = self._create_test_invoice()
        
        processing_times = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(5):
                print(f"  Processing test document {i+1}/5...")
                
                start = time.time()
                try:
                    # Prepare form data
                    data = aiohttp.FormData()
                    
                    # Convert image to bytes
                    img_buffer = io.BytesIO()
                    test_image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    data.add_field('file', img_buffer, filename='test_invoice.png', content_type='image/png')
                    data.add_field('extraction_type', 'invoice')
                    data.add_field('language', 'en')
                    data.add_field('use_schema', 'true')
                    data.add_field('priority', 'high')
                    
                    async with session.post(f"{self.base_url}/extract/file", data=data, timeout=180) as response:
                        if response.status == 200:
                            result = await response.json()
                            processing_time = time.time() - start
                            processing_times.append(processing_time)
                            
                            print(f"    ✅ Completed in {processing_time:.2f}s")
                            
                            # Extract key metrics
                            if 'processing_info' in result:
                                method = result['processing_info'].get('strategy_used', 'unknown')
                                print(f"    Method: {method}")
                        else:
                            print(f"    ❌ HTTP {response.status}: {await response.text()}")
                            
                except asyncio.TimeoutError:
                    print(f"    ⏰ Request {i+1} timed out")
                except Exception as e:
                    print(f"    ❌ Request {i+1} failed: {e}")
        
        if processing_times:
            avg_time = statistics.mean(processing_times)
            min_time = min(processing_times)
            max_time = max(processing_times)
            
            print(f"\n📊 Single Document Processing Results:")
            print(f"  Average: {avg_time:.2f}s")
            print(f"  Fastest: {min_time:.2f}s")
            print(f"  Slowest: {max_time:.2f}s")
            
            self.results.append({
                "test": "single_document_processing",
                "average_seconds": avg_time,
                "min_seconds": min_time,
                "max_seconds": max_time,
                "samples": len(processing_times)
            })
        else:
            print("❌ Single document processing test failed")
    
    async def _test_concurrent_processing(self):
        """Test concurrent processing capabilities"""
        print("\n🔄 Testing Concurrent Processing Throughput...")
        
        test_image = self._create_test_invoice()
        
        async def process_single_request(session, request_id):
            """Process a single request"""
            try:
                start = time.time()
                
                # Prepare form data
                data = aiohttp.FormData()
                
                img_buffer = io.BytesIO()
                test_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                data.add_field('file', img_buffer, filename=f'test_{request_id}.png', content_type='image/png')
                data.add_field('extraction_type', 'invoice')
                data.add_field('language', 'en')
                data.add_field('use_schema', 'false')  # Faster without schema
                data.add_field('priority', 'normal')
                
                async with session.post(f"{self.base_url}/extract/file", data=data, timeout=300) as response:
                    if response.status == 200:
                        result = await response.json()
                        processing_time = time.time() - start
                        return {
                            "request_id": request_id,
                            "success": True,
                            "processing_time": processing_time
                        }
                    else:
                        return {
                            "request_id": request_id,
                            "success": False,
                            "error": f"HTTP {response.status}"
                        }
                        
            except Exception as e:
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": str(e)
                }
        
        # Test different concurrency levels
        for concurrent_requests in [2, 4, 6]:
            print(f"\n  Testing {concurrent_requests} concurrent requests...")
            
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                tasks = [
                    process_single_request(session, i) 
                    for i in range(concurrent_requests)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                total_time = time.time() - start_time
                successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
                
                print(f"    Total time: {total_time:.2f}s")
                print(f"    Successful: {len(successful_results)}/{concurrent_requests}")
                print(f"    Throughput: {len(successful_results)/total_time:.2f} docs/sec")
                
                if successful_results:
                    avg_processing_time = statistics.mean([r['processing_time'] for r in successful_results])
                    print(f"    Avg processing time: {avg_processing_time:.2f}s")
                    
                    self.results.append({
                        "test": f"concurrent_processing_{concurrent_requests}",
                        "total_time": total_time,
                        "successful_requests": len(successful_results),
                        "throughput_docs_per_sec": len(successful_results)/total_time,
                        "average_processing_time": avg_processing_time
                    })
    
    async def _test_image_size_impact(self):
        """Test impact of different image sizes on processing speed"""
        print("\n🖼️ Testing Image Size Impact on Performance...")
        
        sizes = [
            (800, 600, "Small"),
            (1920, 1080, "Medium"),
            (3840, 2160, "Large")
        ]
        
        for width, height, size_name in sizes:
            print(f"\n  Testing {size_name} image ({width}x{height})...")
            
            # Create test image of specific size
            test_image = self._create_test_invoice(size=(width, height))
            
            async with aiohttp.ClientSession() as session:
                try:
                    start = time.time()
                    
                    data = aiohttp.FormData()
                    
                    img_buffer = io.BytesIO()
                    test_image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    file_size_mb = img_buffer.getbuffer().nbytes / (1024 * 1024)
                    
                    data.add_field('file', img_buffer, filename=f'test_{size_name.lower()}.png', content_type='image/png')
                    data.add_field('extraction_type', 'invoice')
                    data.add_field('language', 'en')
                    data.add_field('use_schema', 'false')
                    
                    async with session.post(f"{self.base_url}/extract/file", data=data, timeout=300) as response:
                        if response.status == 200:
                            result = await response.json()
                            processing_time = time.time() - start
                            
                            print(f"    ✅ {size_name} ({file_size_mb:.1f}MB): {processing_time:.2f}s")
                            
                            self.results.append({
                                "test": f"image_size_{size_name.lower()}",
                                "dimensions": f"{width}x{height}",
                                "file_size_mb": file_size_mb,
                                "processing_time": processing_time
                            })
                        else:
                            print(f"    ❌ {size_name}: HTTP {response.status}")
                            
                except Exception as e:
                    print(f"    ❌ {size_name}: {e}")
    
    async def _test_memory_efficiency(self):
        """Test memory usage efficiency"""
        print("\n🧠 Testing Memory Efficiency...")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Get initial memory stats
                async with session.get(f"{self.base_url}/stats") as response:
                    if response.status == 200:
                        initial_stats = await response.json()
                        initial_gpu_memory = initial_stats.get('memory', {}).get('gpu_memory_allocated_mb', 0)
                        initial_ram = initial_stats.get('processing', {}).get('ram_usage_mb', 0)
                        
                        print(f"  Initial GPU memory: {initial_gpu_memory:.1f}MB")
                        print(f"  Initial RAM usage: {initial_ram:.1f}MB")
                        
                        self.results.append({
                            "test": "memory_efficiency",
                            "initial_gpu_memory_mb": initial_gpu_memory,
                            "initial_ram_mb": initial_ram
                        })
                        
            except Exception as e:
                print(f"  ❌ Memory efficiency test failed: {e}")
    
    def _create_test_invoice(self, size=(1200, 800)):
        """Create a test invoice image"""
        width, height = size
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        try:
            # Try to use a font, fallback to default if not available
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw invoice content
        y_pos = 50
        
        # Header
        draw.text((50, y_pos), "INVOICE", fill='black', font=font_large)
        y_pos += 60
        
        # Invoice details
        draw.text((50, y_pos), "Invoice #: INV-2025-001", fill='black', font=font_medium)
        y_pos += 30
        draw.text((50, y_pos), "Date: January 26, 2025", fill='black', font=font_medium)
        y_pos += 30
        draw.text((50, y_pos), "Due Date: February 26, 2025", fill='black', font=font_medium)
        y_pos += 50
        
        # Billing info
        draw.text((50, y_pos), "Bill To:", fill='black', font=font_medium)
        y_pos += 25
        draw.text((50, y_pos), "Test Company Ltd.", fill='black', font=font_small)
        y_pos += 20
        draw.text((50, y_pos), "123 Business Street", fill='black', font=font_small)
        y_pos += 20
        draw.text((50, y_pos), "Test City, TC 12345", fill='black', font=font_small)
        y_pos += 50
        
        # Items
        draw.text((50, y_pos), "Description", fill='black', font=font_medium)
        draw.text((300, y_pos), "Qty", fill='black', font=font_medium)
        draw.text((400, y_pos), "Price", fill='black', font=font_medium)
        draw.text((500, y_pos), "Total", fill='black', font=font_medium)
        y_pos += 30
        
        draw.text((50, y_pos), "Document Processing Service", fill='black', font=font_small)
        draw.text((300, y_pos), "1", fill='black', font=font_small)
        draw.text((400, y_pos), "$100.00", fill='black', font=font_small)
        draw.text((500, y_pos), "$100.00", fill='black', font=font_small)
        y_pos += 50
        
        # Total
        draw.text((400, y_pos), "Subtotal: $100.00", fill='black', font=font_medium)
        y_pos += 25
        draw.text((400, y_pos), "Tax (10%): $10.00", fill='black', font=font_medium)
        y_pos += 25
        draw.text((400, y_pos), "TOTAL: $110.00", fill='black', font=font_large)
        
        return image
    
    def _generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE BENCHMARK REPORT")
        print("=" * 60)
        
        if not self.results:
            print("❌ No benchmark results available")
            return
        
        # Categorize results
        latency_tests = [r for r in self.results if 'latency' in r['test']]
        processing_tests = [r for r in self.results if 'processing' in r['test']]
        concurrent_tests = [r for r in self.results if 'concurrent' in r['test']]
        image_size_tests = [r for r in self.results if 'image_size' in r['test']]
        
        # Print results
        if latency_tests:
            print("\n🚀 LATENCY RESULTS:")
            for test in latency_tests:
                print(f"  Health Check: {test['average_ms']:.1f}ms average")
        
        if processing_tests:
            print("\n📄 SINGLE DOCUMENT PROCESSING:")
            for test in processing_tests:
                if test['test'] == 'single_document_processing':
                    print(f"  Average: {test['average_seconds']:.2f}s")
                    print(f"  Best:    {test['min_seconds']:.2f}s")
                    print(f"  Worst:   {test['max_seconds']:.2f}s")
        
        if concurrent_tests:
            print("\n🔄 CONCURRENT PROCESSING:")
            for test in concurrent_tests:
                concurrent_level = test['test'].split('_')[-1]
                print(f"  {concurrent_level} concurrent: {test['throughput_docs_per_sec']:.2f} docs/sec")
        
        if image_size_tests:
            print("\n🖼️ IMAGE SIZE IMPACT:")
            for test in image_size_tests:
                size_name = test['test'].split('_')[-1].title()
                print(f"  {size_name} ({test['dimensions']}): {test['processing_time']:.2f}s")
        
        # Performance assessment
        print("\n🎯 PERFORMANCE ASSESSMENT:")
        
        # Assess single document speed
        single_doc_results = [r for r in processing_tests if r['test'] == 'single_document_processing']
        if single_doc_results:
            avg_time = single_doc_results[0]['average_seconds']
            if avg_time < 5:
                print("  ✅ Single document speed: EXCELLENT (< 5s)")
            elif avg_time < 15:
                print("  ⚡ Single document speed: GOOD (< 15s)")
            elif avg_time < 30:
                print("  ⚠️  Single document speed: ACCEPTABLE (< 30s)")
            else:
                print("  ❌ Single document speed: NEEDS IMPROVEMENT (> 30s)")
        
        # Assess throughput
        best_throughput = 0
        if concurrent_tests:
            best_throughput = max([t['throughput_docs_per_sec'] for t in concurrent_tests])
            if best_throughput > 2:
                print("  ✅ Throughput: EXCELLENT (> 2 docs/sec)")
            elif best_throughput > 1:
                print("  ⚡ Throughput: GOOD (> 1 docs/sec)")
            elif best_throughput > 0.5:
                print("  ⚠️  Throughput: ACCEPTABLE (> 0.5 docs/sec)")
            else:
                print("  ❌ Throughput: NEEDS IMPROVEMENT (< 0.5 docs/sec)")
        
        print(f"\n🏁 Benchmark completed with {len(self.results)} test results")
        
        # Save detailed results
        with open('performance_benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("📄 Detailed results saved to performance_benchmark_results.json")

async def main():
    """Run the performance benchmark"""
    benchmark = PerformanceBenchmark()
    await benchmark.run_benchmark()

if __name__ == "__main__":
    asyncio.run(main())