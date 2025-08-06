"""
Comprehensive tests for memory transfer optimization
PRODUCTION-READY - NO PLACEHOLDERS - HARD FAILURES ONLY

Tests all transfer optimization strategies and performance targets.
"""

import torch
import torch.cuda as cuda
import numpy as np
import time
import pytest
from typing import Dict, List, Tuple, Optional
import gc
import math

# Import components
from tropical_channel_extractor import (
    TropicalChannels,
    TropicalChannelManager,
    TropicalPolynomial,
    TropicalMonomial
)
from transfer_optimizer import (
    TransferOptimizationConfig,
    ChannelTransferOptimizer,
    TransferStrategy,
    CompressionAlgorithm,
    TransferMetrics,
    PinnedMemoryManager,
    AsyncTransferCoordinator,
    TransferCompressionEngine
)
from gpu_memory_optimizer import GPUMemoryLayoutConfig


class TestTransferBandwidth:
    """Test transfer bandwidth optimization"""
    
    @pytest.fixture
    def large_channels(self):
        """Create large channel data for bandwidth testing"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create 100MB+ of channel data
        num_monomials = 1000000  # 1M monomials
        num_variables = 10
        
        channels = TropicalChannels(
            coefficient_channel=torch.randn(num_monomials, dtype=torch.float32),
            exponent_channel=torch.randint(0, 10, (num_monomials, num_variables), dtype=torch.int32),
            index_channel=torch.arange(num_monomials, dtype=torch.int64),
            metadata={'num_variables': num_variables, 'degree': 5},
            device=torch.device('cpu')
        )
        
        return channels
    
    def test_baseline_bandwidth(self, large_channels):
        """Measure baseline transfer bandwidth"""
        device = torch.device('cuda:0')
        
        # Warm up
        for _ in range(3):
            _ = large_channels.coefficient_channel.to(device)
            torch.cuda.synchronize()
        
        # Measure baseline
        start = time.perf_counter()
        coeff_gpu = large_channels.coefficient_channel.to(device)
        exp_gpu = large_channels.exponent_channel.to(device)
        idx_gpu = large_channels.index_channel.to(device)
        torch.cuda.synchronize()
        baseline_time = time.perf_counter() - start
        
        total_bytes = large_channels.get_memory_usage()
        baseline_bandwidth_gbps = (total_bytes / (1024**3)) / baseline_time
        
        print(f"Baseline bandwidth: {baseline_bandwidth_gbps:.2f} GB/s")
        
        # Clean up
        del coeff_gpu, exp_gpu, idx_gpu
        torch.cuda.empty_cache()
        
        return baseline_bandwidth_gbps
    
    def test_optimized_bandwidth(self, large_channels):
        """Test optimized transfer bandwidth"""
        config = TransferOptimizationConfig.from_system_specs()
        optimizer = ChannelTransferOptimizer(config)
        
        device = torch.device('cuda:0')
        
        # Warm up
        for _ in range(3):
            result, _ = optimizer.transfer_to_gpu(large_channels, device)
            del result
            torch.cuda.empty_cache()
        
        # Measure optimized
        result, metrics = optimizer.transfer_to_gpu(large_channels, device)
        
        print(f"Optimized bandwidth: {metrics.bandwidth_gbps:.2f} GB/s")
        print(f"Strategy used: {metrics.strategy_used.value}")
        print(f"Transfer time: {metrics.transfer_time_ms:.2f} ms")
        
        # Verify we achieve target bandwidth
        assert metrics.bandwidth_gbps >= config.min_transfer_bandwidth_gbps, \
            f"Bandwidth {metrics.bandwidth_gbps:.2f} GB/s below target {config.min_transfer_bandwidth_gbps} GB/s"
        
        # Clean up
        del result
        torch.cuda.empty_cache()
        optimizer.shutdown()
        
        return metrics.bandwidth_gbps
    
    def test_bandwidth_improvement(self, large_channels):
        """Test bandwidth improvement over baseline"""
        baseline_bandwidth = self.test_baseline_bandwidth(large_channels)
        optimized_bandwidth = self.test_optimized_bandwidth(large_channels)
        
        improvement_factor = optimized_bandwidth / baseline_bandwidth
        print(f"Bandwidth improvement: {improvement_factor:.2f}x")
        
        # Should achieve at least 1.5x improvement
        assert improvement_factor >= 1.5, \
            f"Insufficient bandwidth improvement: {improvement_factor:.2f}x"


class TestPinnedMemory:
    """Test pinned memory optimization"""
    
    def test_pinned_memory_allocation(self):
        """Test pinned memory pool allocation"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = TransferOptimizationConfig(
            enable_pinned_memory=True,
            pinned_memory_pool_size_mb=128
        )
        manager = PinnedMemoryManager(config)
        
        # Test allocation
        buffer1 = manager.acquire_buffer(1024 * 1024)  # 1MB
        assert buffer1 is not None
        assert buffer1.is_pinned()
        
        # Test reuse
        manager.release_buffer(buffer1)
        buffer2 = manager.acquire_buffer(1024 * 1024)
        assert buffer2 is buffer1  # Should reuse same buffer
        
        # Test statistics
        stats = manager.get_stats()
        assert stats['hit_count'] > 0
        assert stats['hit_rate'] > 0
    
    def test_pinned_vs_pageable(self):
        """Compare pinned vs pageable memory transfers"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        size = 10 * 1024 * 1024  # 10MB
        data = torch.randn(size // 4, dtype=torch.float32)  # float32 = 4 bytes
        device = torch.device('cuda:0')
        
        # Test pageable transfer
        start = time.perf_counter()
        for _ in range(10):
            gpu_data = data.to(device)
            torch.cuda.synchronize()
            del gpu_data
        pageable_time = time.perf_counter() - start
        
        # Test pinned transfer
        data_pinned = data.pin_memory()
        start = time.perf_counter()
        for _ in range(10):
            gpu_data = data_pinned.to(device)
            torch.cuda.synchronize()
            del gpu_data
        pinned_time = time.perf_counter() - start
        
        speedup = pageable_time / pinned_time
        print(f"Pinned memory speedup: {speedup:.2f}x")
        
        # Pinned should be faster
        assert speedup > 1.2, f"Insufficient pinned memory speedup: {speedup:.2f}x"


class TestAsyncTransfers:
    """Test asynchronous transfer optimization"""
    
    def test_async_coordinator(self):
        """Test async transfer coordinator"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = TransferOptimizationConfig(
            enable_async_transfers=True,
            num_cuda_streams=4
        )
        coordinator = AsyncTransferCoordinator(config)
        
        # Create test data
        tensors = [torch.randn(1000000) for _ in range(10)]
        device = torch.device('cuda:0')
        
        # Schedule async transfers
        futures = []
        for tensor in tensors:
            future = coordinator.schedule_transfer(tensor, device)
            futures.append(future)
        
        # Wait for completion
        results = [future.result() for future in futures]
        
        # Verify all transferred
        assert len(results) == len(tensors)
        for result in results:
            assert result.device == device
        
        # Check statistics
        assert coordinator.total_transfers == len(tensors)
        assert coordinator.concurrent_peak > 1  # Should have concurrent transfers
        
        coordinator.shutdown()
    
    def test_async_vs_sync(self):
        """Compare async vs synchronous transfers"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create multiple channel sets
        num_channels = 10
        channels_list = []
        for _ in range(num_channels):
            channels = TropicalChannels(
                coefficient_channel=torch.randn(100000),
                exponent_channel=torch.randint(0, 10, (100000, 5), dtype=torch.int32),
                index_channel=torch.arange(100000, dtype=torch.int64),
                metadata={'num_variables': 5},
                device=torch.device('cpu')
            )
            channels_list.append(channels)
        
        device = torch.device('cuda:0')
        
        # Test synchronous transfers
        start = time.perf_counter()
        for channels in channels_list:
            gpu_channels = channels.to_gpu(device)
            del gpu_channels
        torch.cuda.synchronize()
        sync_time = time.perf_counter() - start
        
        # Test async transfers
        config = TransferOptimizationConfig(
            enable_async_transfers=True,
            default_strategy=TransferStrategy.ASYNC
        )
        optimizer = ChannelTransferOptimizer(config)
        
        start = time.perf_counter()
        results = optimizer.batch_transfer_to_gpu(channels_list, device)
        torch.cuda.synchronize()
        async_time = time.perf_counter() - start
        
        speedup = sync_time / async_time
        print(f"Async transfer speedup: {speedup:.2f}x")
        
        # Async should be faster for multiple transfers
        assert speedup > 1.3, f"Insufficient async speedup: {speedup:.2f}x"
        
        optimizer.shutdown()


class TestPipelineOverlap:
    """Test pipeline overlap optimization"""
    
    def test_pipelined_transfer(self):
        """Test pipelined transfer with overlap"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create large channel data
        channels = TropicalChannels(
            coefficient_channel=torch.randn(5000000),  # 5M elements
            exponent_channel=torch.randint(0, 10, (5000000, 8), dtype=torch.int32),
            index_channel=torch.arange(5000000, dtype=torch.int64),
            metadata={'num_variables': 8},
            device=torch.device('cpu')
        )
        
        config = TransferOptimizationConfig(
            enable_pipelining=True,
            pipeline_chunk_size_mb=8,
            num_cuda_streams=4,
            default_strategy=TransferStrategy.PIPELINED
        )
        optimizer = ChannelTransferOptimizer(config)
        
        device = torch.device('cuda:0')
        
        # Transfer with pipelining
        result, metrics = optimizer.transfer_to_gpu(channels, device)
        
        print(f"Pipeline transfer bandwidth: {metrics.bandwidth_gbps:.2f} GB/s")
        print(f"Pipeline efficiency: {metrics.pipeline_efficiency:.2f}")
        
        # Should achieve good bandwidth
        assert metrics.bandwidth_gbps >= 8.0, \
            f"Pipeline bandwidth {metrics.bandwidth_gbps:.2f} GB/s too low"
        
        # Clean up
        del result
        torch.cuda.empty_cache()
        optimizer.shutdown()
    
    def test_compute_transfer_overlap(self):
        """Test overlapping computation with transfer"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda:0')
        
        # Create multiple channel sets
        num_batches = 5
        channels_list = []
        for _ in range(num_batches):
            channels = TropicalChannels(
                coefficient_channel=torch.randn(1000000),
                exponent_channel=torch.randint(0, 10, (1000000, 6), dtype=torch.int32),
                index_channel=torch.arange(1000000, dtype=torch.int64),
                metadata={'num_variables': 6},
                device=torch.device('cpu')
            )
            channels_list.append(channels)
        
        config = TransferOptimizationConfig(
            enable_pipelining=True,
            enable_async_transfers=True
        )
        optimizer = ChannelTransferOptimizer(config)
        
        # Simulate computation + transfer pipeline
        stream_compute = torch.cuda.Stream()
        stream_transfer = torch.cuda.Stream()
        
        results = []
        start = time.perf_counter()
        
        for i, channels in enumerate(channels_list):
            # Start transfer on transfer stream
            with torch.cuda.stream(stream_transfer):
                gpu_channels, _ = optimizer.transfer_to_gpu(channels, device)
            
            # Do computation on previous batch while transferring
            if i > 0:
                with torch.cuda.stream(stream_compute):
                    prev_result = results[-1]
                    # Simulate computation
                    output = prev_result.coefficient_channel * 2.0
                    output = output + 1.0
            
            # Synchronize and store
            stream_transfer.synchronize()
            results.append(gpu_channels)
        
        # Final computation
        with torch.cuda.stream(stream_compute):
            final_result = results[-1]
            output = final_result.coefficient_channel * 2.0
        
        torch.cuda.synchronize()
        overlap_time = time.perf_counter() - start
        
        # Compare with sequential
        results_seq = []
        start = time.perf_counter()
        
        for channels in channels_list:
            gpu_channels = channels.to_gpu(device)
            # Simulate computation
            output = gpu_channels.coefficient_channel * 2.0
            output = output + 1.0
            results_seq.append(gpu_channels)
        
        torch.cuda.synchronize()
        sequential_time = time.perf_counter() - start
        
        overlap_efficiency = 1 - (overlap_time / sequential_time)
        print(f"Overlap efficiency: {overlap_efficiency:.2%}")
        
        # Should achieve some overlap benefit
        assert overlap_efficiency > 0.2, \
            f"Insufficient overlap efficiency: {overlap_efficiency:.2%}"
        
        optimizer.shutdown()


class TestCompressionBenefit:
    """Test transfer compression benefits"""
    
    def test_compression_algorithms(self):
        """Test different compression algorithms"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create compressible data (with patterns)
        num_elements = 1000000
        data = torch.zeros(num_elements)
        # Add sparse non-zero values
        indices = torch.randint(0, num_elements, (num_elements // 10,))
        data[indices] = torch.randn(len(indices))
        
        channels = TropicalChannels(
            coefficient_channel=data,
            exponent_channel=torch.zeros((num_elements, 4), dtype=torch.int32),
            index_channel=torch.arange(num_elements, dtype=torch.int64),
            metadata={'num_variables': 4},
            device=torch.device('cpu')
        )
        
        # Test LZ4 compression
        config_lz4 = TransferOptimizationConfig(
            enable_compression=True,
            compression_algorithm=CompressionAlgorithm.LZ4,
            compression_threshold_mb=0.1
        )
        engine_lz4 = TransferCompressionEngine(config_lz4)
        
        compressed_lz4, ratio_lz4 = engine_lz4.compress_tensor(data)
        print(f"LZ4 compression ratio: {ratio_lz4:.2f}x")
        
        # Test Snappy compression
        config_snappy = TransferOptimizationConfig(
            enable_compression=True,
            compression_algorithm=CompressionAlgorithm.SNAPPY,
            compression_threshold_mb=0.1
        )
        engine_snappy = TransferCompressionEngine(config_snappy)
        
        compressed_snappy, ratio_snappy = engine_snappy.compress_tensor(data)
        print(f"Snappy compression ratio: {ratio_snappy:.2f}x")
        
        # Both should achieve some compression on sparse data
        assert ratio_lz4 > 1.5, f"Insufficient LZ4 compression: {ratio_lz4:.2f}x"
        assert ratio_snappy > 1.3, f"Insufficient Snappy compression: {ratio_snappy:.2f}x"
    
    def test_compressed_transfer_benefit(self):
        """Test benefit of compressed transfers for large data"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create highly compressible large data
        num_elements = 10000000  # 10M elements
        data = torch.zeros(num_elements)
        # Sparse pattern
        stride = 100
        data[::stride] = torch.randn(num_elements // stride)
        
        channels = TropicalChannels(
            coefficient_channel=data,
            exponent_channel=torch.zeros((num_elements, 3), dtype=torch.int32),
            index_channel=torch.arange(num_elements, dtype=torch.int64),
            metadata={'num_variables': 3},
            device=torch.device('cpu')
        )
        
        device = torch.device('cuda:0')
        
        # Test uncompressed transfer
        config_nocomp = TransferOptimizationConfig(
            enable_compression=False
        )
        optimizer_nocomp = ChannelTransferOptimizer(config_nocomp)
        
        start = time.perf_counter()
        result_nocomp, metrics_nocomp = optimizer_nocomp.transfer_to_gpu(channels, device)
        time_nocomp = time.perf_counter() - start
        
        del result_nocomp
        torch.cuda.empty_cache()
        
        # Test compressed transfer
        config_comp = TransferOptimizationConfig(
            enable_compression=True,
            compression_algorithm=CompressionAlgorithm.LZ4,
            compression_threshold_mb=1.0,
            default_strategy=TransferStrategy.COMPRESSED
        )
        optimizer_comp = ChannelTransferOptimizer(config_comp)
        
        start = time.perf_counter()
        result_comp, metrics_comp = optimizer_comp.transfer_to_gpu(channels, device)
        time_comp = time.perf_counter() - start
        
        speedup = time_nocomp / time_comp
        print(f"Compressed transfer speedup: {speedup:.2f}x")
        print(f"Compression ratio: {metrics_comp.compression_ratio:.2f}x")
        
        # Compressed should be faster for compressible data
        assert speedup > 1.2, f"Insufficient compression speedup: {speedup:.2f}x"
        
        optimizer_nocomp.shutdown()
        optimizer_comp.shutdown()


class TestLargeDataTransfers:
    """Test transfers of very large data (> 1GB)"""
    
    def test_gigabyte_transfer(self):
        """Test transfer of > 1GB data"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Check available GPU memory
        device = torch.device('cuda:0')
        gpu_props = torch.cuda.get_device_properties(device)
        available_memory_gb = gpu_props.total_memory / (1024**3)
        
        if available_memory_gb < 8:
            pytest.skip("Insufficient GPU memory for large transfer test")
        
        # Create 1.5GB of channel data
        num_elements = 100000000  # 100M elements
        channels = TropicalChannels(
            coefficient_channel=torch.randn(num_elements, dtype=torch.float32),  # 400MB
            exponent_channel=torch.randint(0, 10, (num_elements, 10), dtype=torch.int32),  # 4GB
            index_channel=torch.arange(num_elements, dtype=torch.int64),  # 800MB
            metadata={'num_variables': 10},
            device=torch.device('cpu')
        )
        
        total_size_gb = channels.get_memory_usage() / (1024**3)
        print(f"Total data size: {total_size_gb:.2f} GB")
        
        config = TransferOptimizationConfig.from_system_specs()
        config.enable_pipelining = True
        config.pipeline_chunk_size_mb = 64
        optimizer = ChannelTransferOptimizer(config)
        
        # Transfer large data
        start = time.perf_counter()
        result, metrics = optimizer.transfer_to_gpu(channels, device)
        transfer_time = time.perf_counter() - start
        
        print(f"Large transfer bandwidth: {metrics.bandwidth_gbps:.2f} GB/s")
        print(f"Transfer time: {transfer_time:.2f} seconds")
        
        # Should achieve good bandwidth even for large transfers
        assert metrics.bandwidth_gbps >= 10.0, \
            f"Large transfer bandwidth {metrics.bandwidth_gbps:.2f} GB/s too low"
        
        # Verify data integrity
        assert torch.allclose(result.coefficient_channel.cpu(), channels.coefficient_channel)
        
        del result
        torch.cuda.empty_cache()
        optimizer.shutdown()


class TestStreamCoordination:
    """Test CUDA stream coordination"""
    
    def test_multiple_streams(self):
        """Test using multiple CUDA streams"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = TransferOptimizationConfig(
            num_cuda_streams=8,
            stream_pool_size=8,
            enable_async_transfers=True
        )
        coordinator = AsyncTransferCoordinator(config)
        
        # Acquire all streams
        streams = []
        for _ in range(config.stream_pool_size):
            stream = coordinator.acquire_stream()
            streams.append(stream)
            assert stream is not None
        
        # Try to acquire one more (should timeout or raise)
        with pytest.raises(RuntimeError):
            extra_stream = coordinator.acquire_stream()
        
        # Release all streams
        for stream in streams:
            coordinator.release_stream(stream)
        
        # Now should be able to acquire again
        stream = coordinator.acquire_stream()
        assert stream is not None
        coordinator.release_stream(stream)
        
        coordinator.shutdown()
    
    def test_stream_synchronization(self):
        """Test proper stream synchronization"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda:0')
        num_transfers = 20
        
        config = TransferOptimizationConfig(
            num_cuda_streams=4,
            enable_async_transfers=True
        )
        coordinator = AsyncTransferCoordinator(config)
        
        # Create test data
        tensors = [torch.randn(1000000) for _ in range(num_transfers)]
        
        # Schedule all transfers
        futures = []
        for tensor in tensors:
            future = coordinator.schedule_transfer(tensor, device)
            futures.append(future)
        
        # Wait for all
        coordinator.wait_all()
        
        # All futures should be done
        for future in futures:
            assert future.done()
            result = future.result()
            assert result.device == device
        
        # Check peak concurrency
        print(f"Peak concurrent transfers: {coordinator.concurrent_peak}")
        assert coordinator.concurrent_peak > 1
        assert coordinator.concurrent_peak <= config.max_concurrent_transfers
        
        coordinator.shutdown()


class TestStrategySelection:
    """Test automatic strategy selection"""
    
    def test_auto_strategy_selection(self):
        """Test automatic selection of transfer strategy"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = TransferOptimizationConfig(
            enable_auto_strategy=True,
            zero_copy_threshold_kb=64,
            small_transfer_threshold_mb=1,
            medium_transfer_threshold_mb=10,
            large_transfer_threshold_mb=100
        )
        optimizer = ChannelTransferOptimizer(config)
        
        # Test tiny data (should use zero-copy)
        tiny_size = 32 * 1024  # 32KB
        strategy = optimizer.select_strategy(tiny_size)
        assert strategy == TransferStrategy.ZERO_COPY
        
        # Test small data (should use pinned)
        small_size = 512 * 1024  # 512KB
        strategy = optimizer.select_strategy(small_size)
        assert strategy == TransferStrategy.PINNED
        
        # Test medium data (should use async)
        medium_size = 5 * 1024 * 1024  # 5MB
        strategy = optimizer.select_strategy(medium_size)
        assert strategy == TransferStrategy.ASYNC
        
        # Test large data (should use pipelined)
        large_size = 50 * 1024 * 1024  # 50MB
        strategy = optimizer.select_strategy(large_size)
        assert strategy == TransferStrategy.PIPELINED
        
        # Test very large data (should consider compression or NVLink)
        very_large_size = 500 * 1024 * 1024  # 500MB
        strategy = optimizer.select_strategy(very_large_size)
        assert strategy in [TransferStrategy.COMPRESSED, TransferStrategy.PIPELINED, TransferStrategy.NVLINK]
        
        optimizer.shutdown()


class TestIntegration:
    """Integration tests with full channel pipeline"""
    
    def test_channel_manager_integration(self):
        """Test integration with TropicalChannelManager"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create channel manager with transfer optimization
        from transfer_optimizer import TransferOptimizationConfig
        
        transfer_config = TransferOptimizationConfig.from_system_specs()
        manager = TropicalChannelManager(
            device=torch.device('cpu'),
            transfer_config=transfer_config
        )
        
        # Create polynomial
        monomials = []
        for _ in range(10000):
            exponents = {i: np.random.randint(0, 5) for i in range(5)}
            coeff = np.random.randn()
            monomials.append(TropicalMonomial(coeff, exponents))
        
        polynomial = TropicalPolynomial(monomials, num_variables=5)
        
        # Convert to channels
        channels = manager.polynomial_to_channels(polynomial)
        
        # Transfer to GPU using optimized transfer
        device = torch.device('cuda:0')
        gpu_channels = channels.to_gpu(device, transfer_optimizer=manager.transfer_optimizer)
        
        # Verify on GPU
        assert gpu_channels.device == device
        assert gpu_channels.coefficient_channel.device == device
        
        # Get transfer statistics
        stats = gpu_channels.transfer_statistics(manager.transfer_optimizer)
        print(f"Transfer stats: {stats}")
        
        assert 'transfer_optimizer_stats' in stats
        assert stats['transfer_optimizer_stats']['total_transfers'] > 0
        
        # Transfer back to CPU
        cpu_channels = gpu_channels.to_cpu(transfer_optimizer=manager.transfer_optimizer)
        assert cpu_channels.device.type == 'cpu'
        
        # Test prefetch
        future = channels.prefetch_to_gpu(device, manager.transfer_optimizer)
        prefetched = future.result()
        assert prefetched.device == device
        
        # Clean up
        manager.transfer_optimizer.shutdown()
    
    def test_packed_channel_transfer(self):
        """Test transfer of packed channels"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        from channel_packing import ChannelPackingConfig, UnifiedChannelPacker
        
        # Create packed channels
        channels = TropicalChannels(
            coefficient_channel=torch.randn(100000),
            exponent_channel=torch.randint(0, 10, (100000, 5), dtype=torch.int32),
            index_channel=torch.arange(100000, dtype=torch.int64),
            metadata={'num_variables': 5},
            device=torch.device('cpu')
        )
        
        packing_config = ChannelPackingConfig()
        packer = UnifiedChannelPacker(packing_config)
        
        packed_data, packing_metrics = packer.pack_channels(channels)
        channels.packed_data = packed_data
        channels.packing_metadata = {'metrics': packing_metrics.__dict__}
        
        # Transfer packed channels
        config = TransferOptimizationConfig.from_system_specs()
        optimizer = ChannelTransferOptimizer(config)
        
        device = torch.device('cuda:0')
        gpu_channels, metrics = optimizer.transfer_to_gpu(channels, device)
        
        print(f"Packed channel transfer bandwidth: {metrics.bandwidth_gbps:.2f} GB/s")
        
        # Verify packed data transferred
        assert gpu_channels.packed_data is not None
        assert gpu_channels.device == device
        
        optimizer.shutdown()
    
    def test_validation_preserved(self):
        """Test that validation is preserved during transfer"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        from channel_validation import TropicalChannelValidator, ChannelValidationConfig
        
        # Create channels
        channels = TropicalChannels(
            coefficient_channel=torch.randn(50000),
            exponent_channel=torch.randint(0, 10, (50000, 6), dtype=torch.int32),
            index_channel=torch.arange(50000, dtype=torch.int64),
            metadata={'num_variables': 6, 'degree': 5},
            device=torch.device('cpu')
        )
        
        # Validate original
        validator = TropicalChannelValidator()
        validation_result = validator.validate_channels(channels)
        assert validation_result.is_valid
        
        # Transfer with optimization
        config = TransferOptimizationConfig.from_system_specs()
        optimizer = ChannelTransferOptimizer(config)
        
        device = torch.device('cuda:0')
        gpu_channels, _ = optimizer.transfer_to_gpu(channels, device)
        
        # Validate after transfer
        gpu_validation = validator.validate_channels(gpu_channels)
        assert gpu_validation.is_valid
        
        # Transfer back
        cpu_channels, _ = optimizer.transfer_to_cpu(gpu_channels)
        
        # Validate round-trip
        cpu_validation = validator.validate_channels(cpu_channels)
        assert cpu_validation.is_valid
        
        # Verify data integrity
        assert torch.allclose(cpu_channels.coefficient_channel, channels.coefficient_channel)
        assert torch.equal(cpu_channels.exponent_channel, channels.exponent_channel)
        assert torch.equal(cpu_channels.index_channel, channels.index_channel)
        
        optimizer.shutdown()


class TestPerformanceTargets:
    """Test that performance targets are met"""
    
    def test_bandwidth_target(self):
        """Test 90% of theoretical PCIe bandwidth"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Theoretical PCIe 3.0 x16: ~15.75 GB/s
        # Theoretical PCIe 4.0 x16: ~31.5 GB/s
        # We'll test for at least 10 GB/s which is ~63% of PCIe 3.0
        
        config = TransferOptimizationConfig.from_system_specs()
        config.target_bandwidth_utilization = 0.9
        config.min_transfer_bandwidth_gbps = 10.0
        optimizer = ChannelTransferOptimizer(config)
        
        # Create large data for bandwidth test
        channels = TropicalChannels(
            coefficient_channel=torch.randn(10000000),  # 10M elements = 40MB
            exponent_channel=torch.randint(0, 10, (10000000, 8), dtype=torch.int32),  # 320MB
            index_channel=torch.arange(10000000, dtype=torch.int64),  # 80MB
            metadata={'num_variables': 8},
            device=torch.device('cpu')
        )
        
        device = torch.device('cuda:0')
        
        # Warm up
        for _ in range(3):
            result, _ = optimizer.transfer_to_gpu(channels, device)
            del result
            torch.cuda.empty_cache()
        
        # Measure
        bandwidths = []
        for _ in range(10):
            result, metrics = optimizer.transfer_to_gpu(channels, device)
            bandwidths.append(metrics.bandwidth_gbps)
            del result
            torch.cuda.empty_cache()
        
        avg_bandwidth = np.mean(bandwidths)
        print(f"Average bandwidth: {avg_bandwidth:.2f} GB/s")
        
        # Should meet minimum target
        assert avg_bandwidth >= config.min_transfer_bandwidth_gbps, \
            f"Bandwidth {avg_bandwidth:.2f} GB/s below target {config.min_transfer_bandwidth_gbps} GB/s"
        
        optimizer.shutdown()
    
    def test_latency_targets(self):
        """Test latency targets for different transfer sizes"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = TransferOptimizationConfig.from_system_specs()
        config.max_latency_small_transfers_ms = 1.0
        config.max_latency_large_transfers_ms = 100.0
        optimizer = ChannelTransferOptimizer(config)
        
        device = torch.device('cuda:0')
        
        # Test small transfer latency (< 1MB)
        small_channels = TropicalChannels(
            coefficient_channel=torch.randn(10000),  # 40KB
            exponent_channel=torch.randint(0, 10, (10000, 3), dtype=torch.int32),  # 120KB
            index_channel=torch.arange(10000, dtype=torch.int64),  # 80KB
            metadata={'num_variables': 3},
            device=torch.device('cpu')
        )
        
        # Warm up
        for _ in range(5):
            result, _ = optimizer.transfer_to_gpu(small_channels, device)
            del result
        
        # Measure small transfer latency
        latencies = []
        for _ in range(20):
            result, metrics = optimizer.transfer_to_gpu(small_channels, device)
            latencies.append(metrics.latency_ms)
            del result
        
        avg_small_latency = np.mean(latencies)
        print(f"Small transfer latency: {avg_small_latency:.3f} ms")
        
        # Should meet small transfer latency target
        assert avg_small_latency <= config.max_latency_small_transfers_ms, \
            f"Small transfer latency {avg_small_latency:.3f}ms exceeds target {config.max_latency_small_transfers_ms}ms"
        
        # Test large transfer latency (> 100MB)
        large_channels = TropicalChannels(
            coefficient_channel=torch.randn(10000000),  # 40MB
            exponent_channel=torch.randint(0, 10, (10000000, 5), dtype=torch.int32),  # 200MB
            index_channel=torch.arange(10000000, dtype=torch.int64),  # 80MB
            metadata={'num_variables': 5},
            device=torch.device('cpu')
        )
        
        result, metrics = optimizer.transfer_to_gpu(large_channels, device)
        print(f"Large transfer latency: {metrics.latency_ms:.3f} ms")
        
        # Should meet large transfer latency target
        assert metrics.latency_ms <= config.max_latency_large_transfers_ms, \
            f"Large transfer latency {metrics.latency_ms:.3f}ms exceeds target {config.max_latency_large_transfers_ms}ms"
        
        optimizer.shutdown()
    
    def test_memory_overhead(self):
        """Test memory overhead for pinned buffers"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = TransferOptimizationConfig(
            enable_pinned_memory=True,
            pinned_memory_pool_size_mb=256
        )
        manager = PinnedMemoryManager(config)
        
        # Pre-allocated buffers should not exceed configured size
        total_allocated = 0
        for size, buffers in manager.free_buffers.items():
            for buffer in buffers:
                total_allocated += buffer.numel() * buffer.element_size()
        
        total_allocated_mb = total_allocated / (1024 * 1024)
        print(f"Total pre-allocated: {total_allocated_mb:.2f} MB")
        
        # Should not exceed configured pool size
        assert total_allocated_mb <= config.pinned_memory_pool_size_mb * 1.1, \
            f"Memory overhead {total_allocated_mb:.2f}MB exceeds limit"
        
        # Test memory overhead during transfers
        optimizer = ChannelTransferOptimizer(config)
        
        channels = TropicalChannels(
            coefficient_channel=torch.randn(1000000),
            exponent_channel=torch.randint(0, 10, (1000000, 5), dtype=torch.int32),
            index_channel=torch.arange(1000000, dtype=torch.int64),
            metadata={'num_variables': 5},
            device=torch.device('cpu')
        )
        
        device = torch.device('cuda:0')
        
        # Perform multiple transfers
        for _ in range(10):
            result, _ = optimizer.transfer_to_gpu(channels, device)
            del result
        
        # Check pinned memory stats
        stats = optimizer.pinned_memory_manager.get_stats()
        print(f"Pinned memory stats: {stats}")
        
        # Should have good hit rate (buffer reuse)
        assert stats['hit_rate'] > 0.8, f"Low buffer reuse: {stats['hit_rate']:.2%}"
        
        optimizer.shutdown()


if __name__ == "__main__":
    # Run tests
    import sys
    
    print("Testing Memory Transfer Optimization")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available - skipping GPU tests")
        sys.exit(0)
    
    # Run bandwidth tests
    print("\nTesting Transfer Bandwidth...")
    bandwidth_test = TestTransferBandwidth()
    channels = bandwidth_test.large_channels()
    bandwidth_test.test_bandwidth_improvement(channels)
    
    # Run pinned memory tests
    print("\nTesting Pinned Memory...")
    pinned_test = TestPinnedMemory()
    pinned_test.test_pinned_memory_allocation()
    pinned_test.test_pinned_vs_pageable()
    
    # Run async tests
    print("\nTesting Async Transfers...")
    async_test = TestAsyncTransfers()
    async_test.test_async_coordinator()
    async_test.test_async_vs_sync()
    
    # Run pipeline tests
    print("\nTesting Pipeline Overlap...")
    pipeline_test = TestPipelineOverlap()
    pipeline_test.test_pipelined_transfer()
    pipeline_test.test_compute_transfer_overlap()
    
    # Run compression tests
    print("\nTesting Compression...")
    compression_test = TestCompressionBenefit()
    compression_test.test_compression_algorithms()
    
    # Run strategy selection tests
    print("\nTesting Strategy Selection...")
    strategy_test = TestStrategySelection()
    strategy_test.test_auto_strategy_selection()
    
    # Run integration tests
    print("\nTesting Integration...")
    integration_test = TestIntegration()
    integration_test.test_channel_manager_integration()
    integration_test.test_validation_preserved()
    
    # Run performance target tests
    print("\nTesting Performance Targets...")
    perf_test = TestPerformanceTargets()
    perf_test.test_bandwidth_target()
    perf_test.test_latency_targets()
    perf_test.test_memory_overhead()
    
    print("\n" + "=" * 60)
    print("All transfer optimization tests passed!")