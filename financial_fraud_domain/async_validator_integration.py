"""
Async Validator Integration - Chunk 4: Async Support and Integration
Asynchronous validation support with high-throughput processing,
batch validation, and comprehensive integration utilities.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import queue
from pathlib import Path
import json
import aiofiles
import aiohttp
from collections import defaultdict

# Import enhanced validation components
try:
    from enhanced_validator_framework import (
        TransactionValidationError, ValidationMetricsCollector,
        enhanced_error_handler, enhanced_performance_monitor
    )
    from enhanced_transaction_validator import (
        EnhancedTransactionFieldValidator, EnhancedBusinessRuleValidator,
        ValidationContext, RiskLevel
    )
    from production_validator import (
        ProductionFinancialValidator, SystemHealth, ValidationDiagnostics,
        create_production_validator
    )
    from data_validator import (
        ValidationLevel, ValidationSeverity, ValidationResult, ValidationIssue
    )
    VALIDATION_COMPONENTS = True
except ImportError as e:
    VALIDATION_COMPONENTS = False
    logger.warning(f"Validation components not available: {e}")

# Configure logging
logger = logging.getLogger(__name__)

# ======================== ASYNC VALIDATION CLASSES ========================

@dataclass
class BatchValidationRequest:
    """Batch validation request configuration"""
    request_id: str
    data_batches: List[pd.DataFrame]
    validation_level: ValidationLevel
    validation_context: Optional[ValidationContext] = None
    priority: int = 0  # Higher numbers = higher priority
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class BatchValidationResult:
    """Batch validation result"""
    request_id: str
    results: List[ValidationResult]
    total_batches: int
    successful_batches: int
    failed_batches: int
    total_processing_time: float
    average_batch_time: float
    total_records_processed: int
    total_issues_found: int
    overall_success: bool
    completed_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class AsyncValidationQueue:
    """High-performance async validation queue"""
    
    def __init__(self, 
                 max_concurrent: int = 10,
                 max_queue_size: int = 1000,
                 worker_timeout: float = 300.0):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.worker_timeout = worker_timeout
        
        # Queue management
        self._request_queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._result_futures = {}
        self._active_workers = set()
        self._shutdown_event = asyncio.Event()
        
        # Metrics
        self._processed_requests = 0
        self._failed_requests = 0
        self._total_processing_time = 0.0
        
        # Worker management
        self._workers = []
        self._worker_stats = defaultdict(dict)
        
    async def start_workers(self, validator: 'AsyncFinancialValidator'):
        """Start worker tasks"""
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker_loop(f"worker_{i}", validator))
            self._workers.append(worker)
            logger.info(f"Started async validation worker {i}")
    
    async def _worker_loop(self, worker_id: str, validator: 'AsyncFinancialValidator'):
        """Main worker loop"""
        self._worker_stats[worker_id] = {
            'processed': 0,
            'failed': 0,
            'total_time': 0.0,
            'start_time': time.time()
        }
        
        while not self._shutdown_event.is_set():
            try:
                # Get next request (with timeout to allow shutdown)
                try:
                    priority, request = await asyncio.wait_for(
                        self._request_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process the request
                start_time = time.time()
                try:
                    result = await validator._process_batch_request(request)
                    
                    # Store result
                    if request.request_id in self._result_futures:
                        self._result_futures[request.request_id].set_result(result)
                    
                    # Update metrics
                    processing_time = time.time() - start_time
                    self._processed_requests += 1
                    self._total_processing_time += processing_time
                    
                    self._worker_stats[worker_id]['processed'] += 1
                    self._worker_stats[worker_id]['total_time'] += processing_time
                    
                    logger.debug(f"Worker {worker_id} completed request {request.request_id} in {processing_time:.3f}s")
                    
                except Exception as e:
                    # Handle processing error
                    error_result = BatchValidationResult(
                        request_id=request.request_id,
                        results=[],
                        total_batches=len(request.data_batches),
                        successful_batches=0,
                        failed_batches=len(request.data_batches),
                        total_processing_time=time.time() - start_time,
                        average_batch_time=0.0,
                        total_records_processed=0,
                        total_issues_found=0,
                        overall_success=False,
                        metadata={'error': str(e), 'worker_id': worker_id}
                    )
                    
                    if request.request_id in self._result_futures:
                        self._result_futures[request.request_id].set_result(error_result)
                    
                    self._failed_requests += 1
                    self._worker_stats[worker_id]['failed'] += 1
                    
                    logger.error(f"Worker {worker_id} failed to process request {request.request_id}: {e}")
                
                finally:
                    self._request_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)  # Brief pause before continuing
    
    async def submit_request(self, request: BatchValidationRequest) -> asyncio.Future:
        """Submit validation request to queue"""
        if self._request_queue.full():
            raise ValueError("Validation queue is full")
        
        # Create future for result
        future = asyncio.Future()
        self._result_futures[request.request_id] = future
        
        # Add to queue with priority
        await self._request_queue.put((-request.priority, request))
        
        return future
    
    async def shutdown(self, timeout: float = 30.0):
        """Shutdown the validation queue"""
        logger.info("Shutting down async validation queue")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for workers to finish
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Worker shutdown timed out, cancelling tasks")
            for worker in self._workers:
                worker.cancel()
        
        # Clean up
        self._workers.clear()
        self._result_futures.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            'queue_size': self._request_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'active_workers': len(self._workers),
            'processed_requests': self._processed_requests,
            'failed_requests': self._failed_requests,
            'average_processing_time': (self._total_processing_time / self._processed_requests 
                                     if self._processed_requests > 0 else 0.0),
            'worker_stats': dict(self._worker_stats)
        }

class AsyncFinancialValidator:
    """
    Asynchronous financial data validator with high-throughput processing,
    batch validation, and integration with existing validation infrastructure.
    """
    
    def __init__(self,
                 base_validator: Optional[ProductionFinancialValidator] = None,
                 max_concurrent_validations: int = 10,
                 enable_batch_processing: bool = True,
                 batch_size: int = 1000,
                 enable_streaming: bool = False):
        
        # Initialize base validator
        self.base_validator = base_validator or create_production_validator()
        
        # Async configuration
        self.max_concurrent_validations = max_concurrent_validations
        self.enable_batch_processing = enable_batch_processing
        self.batch_size = batch_size
        self.enable_streaming = enable_streaming
        
        # Initialize async queue
        if enable_batch_processing:
            self.validation_queue = AsyncValidationQueue(
                max_concurrent=max_concurrent_validations
            )
        else:
            self.validation_queue = None
        
        # Async metrics
        self.async_metrics = {
            'total_async_validations': 0,
            'successful_async_validations': 0,
            'failed_async_validations': 0,
            'average_async_time': 0.0,
            'concurrent_validations_peak': 0,
            'batch_validations': 0,
            'streaming_validations': 0
        }
        
        # Concurrency tracking
        self._active_validations = set()
        self._validation_semaphore = asyncio.Semaphore(max_concurrent_validations)
        
        logger.info(f"Async Financial Validator initialized with max_concurrent={max_concurrent_validations}")
    
    async def start(self):
        """Start async validation services"""
        if self.validation_queue:
            await self.validation_queue.start_workers(self)
        logger.info("Async Financial Validator services started")
    
    async def validate_async(self, 
                           data: pd.DataFrame,
                           validation_context: Optional[ValidationContext] = None,
                           validation_level: Optional[ValidationLevel] = None,
                           timeout: Optional[float] = None) -> ValidationResult:
        """
        Asynchronous single validation
        
        Args:
            data: Financial transaction data
            validation_context: Validation context
            validation_level: Validation level override
            timeout: Operation timeout in seconds
            
        Returns:
            Validation result
        """
        
        async with self._validation_semaphore:
            validation_id = f"async_{int(time.time())}_{len(self._active_validations)}"
            self._active_validations.add(validation_id)
            
            try:
                start_time = time.time()
                
                # Update metrics
                self.async_metrics['total_async_validations'] += 1
                self.async_metrics['concurrent_validations_peak'] = max(
                    self.async_metrics['concurrent_validations_peak'],
                    len(self._active_validations)
                )
                
                # Run validation in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                
                # Prepare validation call
                validation_func = self.base_validator.validate_financial_data
                
                # Execute with timeout
                if timeout:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            validation_func,
                            data,
                            validation_context,
                            validation_level
                        ),
                        timeout=timeout
                    )
                else:
                    result = await loop.run_in_executor(
                        None,
                        validation_func,
                        data,
                        validation_context,
                        validation_level
                    )
                
                # Update success metrics
                processing_time = time.time() - start_time
                self.async_metrics['successful_async_validations'] += 1
                self._update_average_time(processing_time)
                
                # Add async metadata
                result.metadata.update({
                    'async_validation': True,
                    'async_processing_time': processing_time,
                    'validation_id': validation_id,
                    'concurrent_validations': len(self._active_validations)
                })
                
                logger.debug(f"Async validation {validation_id} completed successfully in {processing_time:.3f}s")
                return result
                
            except asyncio.TimeoutError:
                self.async_metrics['failed_async_validations'] += 1
                logger.error(f"Async validation {validation_id} timed out")
                raise TransactionValidationError(
                    f"Async validation timed out after {timeout}s",
                    validation_rule="async_timeout"
                )
                
            except Exception as e:
                self.async_metrics['failed_async_validations'] += 1
                logger.error(f"Async validation {validation_id} failed: {e}")
                raise
                
            finally:
                self._active_validations.discard(validation_id)
    
    async def validate_batch_async(self,
                                 data_batches: List[pd.DataFrame],
                                 validation_context: Optional[ValidationContext] = None,
                                 validation_level: Optional[ValidationLevel] = None,
                                 priority: int = 0,
                                 callback: Optional[Callable] = None) -> BatchValidationResult:
        """
        Asynchronous batch validation with queue processing
        
        Args:
            data_batches: List of DataFrames to validate
            validation_context: Validation context
            validation_level: Validation level
            priority: Request priority (higher = more urgent)
            callback: Optional callback for completion
            
        Returns:
            Batch validation result
        """
        
        if not self.validation_queue:
            raise ValueError("Batch processing not enabled")
        
        # Create batch request
        request = BatchValidationRequest(
            request_id=f"batch_{int(time.time())}_{len(data_batches)}",
            data_batches=data_batches,
            validation_level=validation_level or ValidationLevel.COMPREHENSIVE,
            validation_context=validation_context,
            priority=priority,
            callback=callback,
            metadata={
                'total_records': sum(len(batch) for batch in data_batches),
                'batch_count': len(data_batches)
            }
        )
        
        # Submit to queue
        result_future = await self.validation_queue.submit_request(request)
        
        # Wait for result
        result = await result_future
        
        # Update metrics
        self.async_metrics['batch_validations'] += 1
        
        logger.info(f"Batch validation {request.request_id} completed: {result.successful_batches}/{result.total_batches} successful")
        
        return result
    
    async def _process_batch_request(self, request: BatchValidationRequest) -> BatchValidationResult:
        """Process batch validation request"""
        start_time = time.time()
        results = []
        successful_batches = 0
        failed_batches = 0
        total_records = 0
        total_issues = 0
        
        # Process each batch
        for i, batch_data in enumerate(request.data_batches):
            try:
                batch_start = time.time()
                
                # Validate batch
                result = await self.validate_async(
                    batch_data,
                    request.validation_context,
                    request.validation_level,
                    timeout=60.0  # Per-batch timeout
                )
                
                results.append(result)
                successful_batches += 1
                total_records += len(batch_data)
                total_issues += len(result.issues)
                
                logger.debug(f"Batch {i+1}/{len(request.data_batches)} completed in {time.time() - batch_start:.3f}s")
                
            except Exception as e:
                failed_batches += 1
                logger.error(f"Batch {i+1}/{len(request.data_batches)} failed: {e}")
                
                # Create error result for failed batch
                error_result = ValidationResult(
                    is_valid=False,
                    issues=[ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        category="batch_processing_error",
                        message=f"Batch processing failed: {e}",
                        rule_id="batch_error"
                    )],
                    data_quality_score=0.0,
                    validation_level=request.validation_level,
                    compliance_passed=False,
                    timestamp=datetime.now(),
                    processing_time=0.0,
                    validator_version="async_3.0.0",
                    metadata={'batch_error': str(e), 'batch_index': i}
                )
                results.append(error_result)
        
        # Calculate metrics
        total_time = time.time() - start_time
        avg_batch_time = total_time / len(request.data_batches) if request.data_batches else 0.0
        overall_success = failed_batches == 0
        
        # Create batch result
        batch_result = BatchValidationResult(
            request_id=request.request_id,
            results=results,
            total_batches=len(request.data_batches),
            successful_batches=successful_batches,
            failed_batches=failed_batches,
            total_processing_time=total_time,
            average_batch_time=avg_batch_time,
            total_records_processed=total_records,
            total_issues_found=total_issues,
            overall_success=overall_success,
            metadata={
                'request_metadata': request.metadata,
                'validation_level': request.validation_level.value,
                'validation_context': asdict(request.validation_context) if request.validation_context else None
            }
        )
        
        # Execute callback if provided
        if request.callback:
            try:
                if asyncio.iscoroutinefunction(request.callback):
                    await request.callback(batch_result)
                else:
                    request.callback(batch_result)
            except Exception as e:
                logger.error(f"Batch callback failed: {e}")
        
        return batch_result
    
    async def validate_stream_async(self, 
                                  data_stream: AsyncIterator[pd.DataFrame],
                                  validation_context: Optional[ValidationContext] = None,
                                  validation_level: Optional[ValidationLevel] = None,
                                  buffer_size: int = 10) -> AsyncIterator[ValidationResult]:
        """
        Asynchronous streaming validation
        
        Args:
            data_stream: Async iterator of DataFrames
            validation_context: Validation context
            validation_level: Validation level
            buffer_size: Number of concurrent validations
            
        Yields:
            Validation results as they complete
        """
        
        if not self.enable_streaming:
            raise ValueError("Streaming validation not enabled")
        
        buffer = []
        stream_count = 0
        
        async for data_chunk in data_stream:
            stream_count += 1
            
            # Add to buffer
            validation_task = asyncio.create_task(
                self.validate_async(data_chunk, validation_context, validation_level)
            )
            buffer.append((stream_count, validation_task))
            
            # Process buffer when full
            if len(buffer) >= buffer_size:
                # Wait for at least one to complete
                done, pending = await asyncio.wait(
                    [task for _, task in buffer],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Yield completed results
                completed_indices = []
                for i, (chunk_id, task) in enumerate(buffer):
                    if task in done:
                        try:
                            result = await task
                            result.metadata['stream_chunk_id'] = chunk_id
                            yield result
                            completed_indices.append(i)
                        except Exception as e:
                            logger.error(f"Stream validation failed for chunk {chunk_id}: {e}")
                
                # Remove completed tasks from buffer
                for i in reversed(completed_indices):
                    buffer.pop(i)
        
        # Process remaining buffer
        if buffer:
            remaining_tasks = [task for _, task in buffer]
            results = await asyncio.gather(*remaining_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                chunk_id = buffer[i][0]
                if isinstance(result, Exception):
                    logger.error(f"Stream validation failed for chunk {chunk_id}: {result}")
                else:
                    result.metadata['stream_chunk_id'] = chunk_id
                    yield result
        
        # Update streaming metrics
        self.async_metrics['streaming_validations'] += stream_count
        logger.info(f"Streaming validation completed: {stream_count} chunks processed")
    
    def _update_average_time(self, processing_time: float):
        """Update average processing time"""
        current_avg = self.async_metrics['average_async_time']
        total_validations = self.async_metrics['total_async_validations']
        
        if total_validations <= 1:
            self.async_metrics['average_async_time'] = processing_time
        else:
            # Running average
            self.async_metrics['average_async_time'] = (
                (current_avg * (total_validations - 1) + processing_time) / total_validations
            )
    
    async def get_async_health(self) -> Dict[str, Any]:
        """Get async validator health status"""
        base_health = self.base_validator.get_system_health()
        
        async_health = {
            'base_validator_status': base_health.status,
            'active_async_validations': len(self._active_validations),
            'max_concurrent_validations': self.max_concurrent_validations,
            'semaphore_available': self._validation_semaphore._value,
            'batch_processing_enabled': self.enable_batch_processing,
            'streaming_enabled': self.enable_streaming,
            'async_metrics': self.async_metrics.copy()
        }
        
        if self.validation_queue:
            async_health['queue_stats'] = self.validation_queue.get_stats()
        
        return async_health
    
    async def shutdown(self, timeout: float = 60.0):
        """Shutdown async validator"""
        logger.info("Shutting down Async Financial Validator")
        
        try:
            # Shutdown validation queue
            if self.validation_queue:
                await self.validation_queue.shutdown(timeout=timeout/2)
            
            # Wait for active validations to complete
            if self._active_validations:
                logger.info(f"Waiting for {len(self._active_validations)} active validations to complete")
                start_time = time.time()
                while self._active_validations and (time.time() - start_time) < timeout/2:
                    await asyncio.sleep(1)
            
            # Shutdown base validator
            if hasattr(self.base_validator, 'shutdown'):
                self.base_validator.shutdown()
            
            logger.info("Async Financial Validator shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during async validator shutdown: {e}")

# ======================== INTEGRATION UTILITIES ========================

class ValidationIntegrationManager:
    """Integration manager for validation components"""
    
    def __init__(self):
        self.validators = {}
        self.async_validators = {}
        self.integrations = {}
    
    def register_validator(self, name: str, validator: Union[ProductionFinancialValidator, AsyncFinancialValidator]):
        """Register a validator"""
        if isinstance(validator, AsyncFinancialValidator):
            self.async_validators[name] = validator
        else:
            self.validators[name] = validator
        
        logger.info(f"Registered validator: {name}")
    
    def get_validator(self, name: str) -> Union[ProductionFinancialValidator, AsyncFinancialValidator]:
        """Get validator by name"""
        return self.validators.get(name) or self.async_validators.get(name)
    
    async def validate_with_fallback(self, 
                                   data: pd.DataFrame,
                                   primary_validator: str,
                                   fallback_validator: str,
                                   **kwargs) -> ValidationResult:
        """Validate with fallback validator on failure"""
        
        try:
            primary = self.get_validator(primary_validator)
            if not primary:
                raise ValueError(f"Primary validator '{primary_validator}' not found")
            
            if isinstance(primary, AsyncFinancialValidator):
                return await primary.validate_async(data, **kwargs)
            else:
                return primary.validate_financial_data(data, **kwargs)
                
        except Exception as e:
            logger.warning(f"Primary validator '{primary_validator}' failed: {e}, trying fallback")
            
            fallback = self.get_validator(fallback_validator)
            if not fallback:
                raise ValueError(f"Fallback validator '{fallback_validator}' not found")
            
            if isinstance(fallback, AsyncFinancialValidator):
                return await fallback.validate_async(data, **kwargs)
            else:
                return fallback.validate_financial_data(data, **kwargs)
    
    async def shutdown_all(self):
        """Shutdown all registered validators"""
        logger.info("Shutting down all validators")
        
        # Shutdown async validators
        for name, validator in self.async_validators.items():
            try:
                await validator.shutdown()
                logger.info(f"Shutdown async validator: {name}")
            except Exception as e:
                logger.error(f"Error shutting down async validator {name}: {e}")
        
        # Shutdown sync validators
        for name, validator in self.validators.items():
            try:
                if hasattr(validator, 'shutdown'):
                    validator.shutdown()
                logger.info(f"Shutdown validator: {name}")
            except Exception as e:
                logger.error(f"Error shutting down validator {name}: {e}")

# ======================== CONVENIENCE FUNCTIONS ========================

async def create_async_validator(config: Optional[Dict[str, Any]] = None,
                               max_concurrent: int = 10,
                               enable_batch: bool = True) -> AsyncFinancialValidator:
    """Create async financial validator"""
    base_validator = create_production_validator(config)
    async_validator = AsyncFinancialValidator(
        base_validator=base_validator,
        max_concurrent_validations=max_concurrent,
        enable_batch_processing=enable_batch
    )
    await async_validator.start()
    return async_validator

async def validate_financial_data_async(data: pd.DataFrame,
                                      validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE,
                                      timeout: Optional[float] = None) -> ValidationResult:
    """Quick async validation function"""
    async_validator = await create_async_validator()
    try:
        return await async_validator.validate_async(
            data, 
            validation_level=validation_level,
            timeout=timeout
        )
    finally:
        await async_validator.shutdown()

# Export async components
__all__ = [
    # Classes
    'BatchValidationRequest',
    'BatchValidationResult', 
    'AsyncValidationQueue',
    'AsyncFinancialValidator',
    'ValidationIntegrationManager',
    
    # Functions
    'create_async_validator',
    'validate_financial_data_async'
]

if __name__ == "__main__":
    print("Async Validator Integration - Chunk 4: Async Support and Integration loaded")