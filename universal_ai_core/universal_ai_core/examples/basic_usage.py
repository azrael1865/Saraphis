#!/usr/bin/env python3
"""
Basic Usage Examples for Universal AI Core
==========================================

This module demonstrates basic usage patterns for the Universal AI Core system.
Adapted from Saraphis train_and_test_standalone.py patterns, these examples show
how to initialize, configure, and use the core system for various domains.

Examples include:
- System initialization and configuration
- Plugin loading and management
- Data processing and feature extraction
- Model training and evaluation
- Proof system usage
- Knowledge base operations
- Performance monitoring
"""

import logging
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd

# Universal AI Core imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config_manager import get_config_manager, get_config
from core.universal_ai_core import UniversalAICore
from core.plugin_manager import PluginManager, PluginType
from core.orchestrator import SystemOrchestrator

logger = logging.getLogger(__name__)


class BasicUsageExamples:
    """
    Basic usage examples for Universal AI Core system.
    
    Demonstrates common patterns adapted from Saraphis usage patterns:
    - System initialization and configuration
    - Plugin management and usage
    - Data processing workflows
    - Model training and evaluation
    - Performance monitoring
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional configuration"""
        self.config_path = config_path
        self.core = None
        self.orchestrator = None
        self.results = {}
        self.performance_history = []
        
        # Setup logging
        self._setup_logging()
        
        # Initialize system
        self._initialize_system()
    
    def _setup_logging(self):
        """Setup logging configuration adapted from Saraphis patterns"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('examples/basic_usage.log')
            ]
        )
        logger.info("Basic usage examples initialized")
    
    def _initialize_system(self):
        """Initialize Universal AI Core system"""
        try:
            # Load configuration
            config_manager = get_config_manager(self.config_path)
            config = config_manager.get_config()
            
            logger.info("Configuration loaded successfully")
            logger.info(f"Environment: {config.environment_config.environment}")
            logger.info(f"Debug mode: {config.environment_config.debug_mode}")
            
            # Initialize core system
            self.core = UniversalAICore(config)
            
            # Initialize orchestrator
            self.orchestrator = SystemOrchestrator(self.core)
            
            logger.info("Universal AI Core system initialized successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    def example_1_basic_configuration(self):
        """
        Example 1: Basic Configuration Management
        
        Demonstrates how to load, modify, and save configurations
        adapted from Saraphis configuration patterns.
        """
        logger.info("=" * 60)
        logger.info("Example 1: Basic Configuration Management")
        logger.info("=" * 60)
        
        try:
            # Get current configuration
            config = get_config()
            
            # Display current settings
            logger.info(f"Current environment: {config.environment_config.environment}")
            logger.info(f"Debug mode: {config.environment_config.debug_mode}")
            logger.info(f"Worker threads: {config.environment_config.worker_threads}")
            logger.info(f"Memory limit: {config.performance_config.memory_limit_mb}MB")
            
            # Modify configuration
            updates = {
                'environment_config': {
                    'worker_threads': 8,
                    'log_level': 'DEBUG'
                },
                'performance_config': {
                    'batch_size': 128,
                    'enable_caching': True
                }
            }
            
            # Apply updates
            config_manager = get_config_manager()
            success = config_manager.update_config(updates, persist=False)
            
            if success:
                logger.info("Configuration updated successfully")
                updated_config = config_manager.get_config()
                logger.info(f"New worker threads: {updated_config.environment_config.worker_threads}")
                logger.info(f"New batch size: {updated_config.performance_config.batch_size}")
            else:
                logger.error("Configuration update failed")
            
            # Plugin-specific configuration
            plugin_config = {
                'model_types': ['random_forest', 'xgboost'],
                'ensemble_method': 'voting',
                'cross_validation_folds': 5
            }
            config.set_plugin_config('molecular_models', plugin_config)
            
            logger.info("Plugin configuration set successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration example failed: {e}")
            return False
    
    def example_2_plugin_management(self):
        """
        Example 2: Plugin Management
        
        Demonstrates how to load, configure, and use plugins
        adapted from Saraphis plugin patterns.
        """
        logger.info("=" * 60)
        logger.info("Example 2: Plugin Management")
        logger.info("=" * 60)
        
        try:
            # Get plugin manager
            plugin_manager = self.core.plugin_manager
            
            # List available plugins
            available_plugins = plugin_manager.list_available_plugins()
            logger.info(f"Available plugins: {len(available_plugins)}")
            
            for plugin_type, plugins in available_plugins.items():
                logger.info(f"  {plugin_type}: {[p['name'] for p in plugins]}")
            
            # Load specific plugins
            plugins_to_load = [
                ('feature_extractors', 'molecular'),
                ('models', 'molecular'),
                ('proof_languages', 'molecular')
            ]
            
            loaded_plugins = {}
            for plugin_type, plugin_name in plugins_to_load:
                try:
                    plugin = plugin_manager.load_plugin(plugin_type, plugin_name)
                    if plugin:
                        loaded_plugins[f"{plugin_type}.{plugin_name}"] = plugin
                        logger.info(f"Loaded plugin: {plugin_type}.{plugin_name}")
                    else:
                        logger.warning(f"Failed to load plugin: {plugin_type}.{plugin_name}")
                except Exception as e:
                    logger.error(f"Error loading plugin {plugin_type}.{plugin_name}: {e}")
            
            # Test plugin functionality
            if 'feature_extractors.molecular' in loaded_plugins:
                molecular_fe = loaded_plugins['feature_extractors.molecular']
                
                # Test with sample molecules
                test_molecules = ['CCO', 'CC(=O)O', 'c1ccccc1']
                
                for smiles in test_molecules:
                    try:
                        result = molecular_fe.extract_features({'smiles': smiles})
                        if result.status.name == 'SUCCESS':
                            logger.info(f"Features extracted for {smiles}: {len(result.features)} features")
                        else:
                            logger.warning(f"Feature extraction failed for {smiles}")
                    except Exception as e:
                        logger.error(f"Error processing {smiles}: {e}")
            
            # Get plugin metadata
            for plugin_name, plugin in loaded_plugins.items():
                try:
                    metadata = plugin.get_metadata()
                    logger.info(f"Plugin {plugin_name} metadata:")
                    logger.info(f"  Version: {metadata.version}")
                    logger.info(f"  Description: {metadata.description}")
                    logger.info(f"  Dependencies: {metadata.dependencies}")
                except Exception as e:
                    logger.error(f"Error getting metadata for {plugin_name}: {e}")
            
            self.results['loaded_plugins'] = loaded_plugins
            return True
            
        except Exception as e:
            logger.error(f"Plugin management example failed: {e}")
            return False
    
    def example_3_data_processing_workflow(self):
        """
        Example 3: Data Processing Workflow
        
        Demonstrates end-to-end data processing workflow
        adapted from Saraphis data processing patterns.
        """
        logger.info("=" * 60)
        logger.info("Example 3: Data Processing Workflow")
        logger.info("=" * 60)
        
        try:
            # Sample data preparation (adapted from train_and_test_standalone.py)
            sample_data = self._prepare_sample_data()
            
            # Load feature extractor plugin
            plugin_manager = self.core.plugin_manager
            fe_plugin = plugin_manager.get_plugin('feature_extractors', 'molecular')
            
            if not fe_plugin:
                logger.warning("Molecular feature extractor not available, using mock data")
                return self._mock_data_processing_workflow()
            
            # Process data in batches (performance optimization pattern)
            batch_size = self.core.config.performance_config.batch_size
            total_samples = len(sample_data)
            processed_results = []
            
            logger.info(f"Processing {total_samples} samples in batches of {batch_size}")
            
            start_time = time.time()
            
            for i in range(0, total_samples, batch_size):
                batch = sample_data[i:i + batch_size]
                batch_start = time.time()
                
                # Process batch
                batch_results = []
                for sample in batch:
                    try:
                        result = fe_plugin.extract_features(sample)
                        if result.status.name == 'SUCCESS':
                            batch_results.append({
                                'input': sample,
                                'features': result.features,
                                'metadata': result.metadata
                            })
                        else:
                            logger.warning(f"Feature extraction failed for sample: {sample}")
                    except Exception as e:
                        logger.error(f"Error processing sample {sample}: {e}")
                
                processed_results.extend(batch_results)
                
                batch_time = time.time() - batch_start
                logger.info(f"Batch {i//batch_size + 1}: {len(batch_results)}/{len(batch)} samples processed in {batch_time:.2f}s")
            
            total_time = time.time() - start_time
            success_rate = len(processed_results) / total_samples
            
            logger.info(f"Data processing completed:")
            logger.info(f"  Total samples: {total_samples}")
            logger.info(f"  Successful: {len(processed_results)}")
            logger.info(f"  Success rate: {success_rate:.2%}")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Throughput: {total_samples/total_time:.2f} samples/s")
            
            # Store results
            self.results['data_processing'] = {
                'total_samples': total_samples,
                'successful_samples': len(processed_results),
                'success_rate': success_rate,
                'processing_time': total_time,
                'throughput': total_samples / total_time,
                'results': processed_results[:5]  # Store first 5 for inspection
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Data processing workflow example failed: {e}")
            return False
    
    def example_4_model_training_evaluation(self):
        """
        Example 4: Model Training and Evaluation
        
        Demonstrates model training and evaluation workflow
        adapted from Saraphis model training patterns.
        """
        logger.info("=" * 60)
        logger.info("Example 4: Model Training and Evaluation")
        logger.info("=" * 60)
        
        try:
            # Check if we have processed data from previous example
            if 'data_processing' not in self.results:
                logger.warning("No processed data available, running data processing first")
                if not self.example_3_data_processing_workflow():
                    return self._mock_model_training()
            
            # Get model plugin
            plugin_manager = self.core.plugin_manager
            model_plugin = plugin_manager.get_plugin('models', 'molecular')
            
            if not model_plugin:
                logger.warning("Molecular model plugin not available, using mock training")
                return self._mock_model_training()
            
            # Prepare training data
            training_data = self._prepare_training_data()
            
            # Configure model training
            training_config = {
                'model_types': ['random_forest', 'xgboost'],
                'ensemble_method': 'voting',
                'cross_validation_folds': 3,
                'test_size': 0.2,
                'random_state': 42
            }
            
            logger.info("Starting model training...")
            logger.info(f"Training configuration: {training_config}")
            
            start_time = time.time()
            
            # Train models
            training_result = model_plugin.train_model(training_data, training_config)
            
            training_time = time.time() - start_time
            
            if training_result.status.name == 'SUCCESS':
                logger.info(f"Model training completed in {training_time:.2f}s")
                
                # Display training metrics
                metrics = training_result.metrics
                logger.info("Training metrics:")
                for model_type, model_metrics in metrics.items():
                    logger.info(f"  {model_type}:")
                    for metric_name, metric_value in model_metrics.items():
                        logger.info(f"    {metric_name}: {metric_value:.4f}")
                
                # Test predictions
                test_samples = training_data['X_test'][:5] if 'X_test' in training_data else []
                
                if len(test_samples) > 0:
                    logger.info("Testing predictions on sample data...")
                    
                    for i, sample in enumerate(test_samples):
                        try:
                            prediction_result = model_plugin.predict({'features': sample})
                            if prediction_result.status.name == 'SUCCESS':
                                prediction = prediction_result.prediction
                                confidence = prediction_result.confidence
                                logger.info(f"  Sample {i+1}: Prediction = {prediction:.4f}, Confidence = {confidence:.4f}")
                        except Exception as e:
                            logger.error(f"Prediction failed for sample {i+1}: {e}")
                
                # Store training results
                self.results['model_training'] = {
                    'training_time': training_time,
                    'metrics': metrics,
                    'model_config': training_config,
                    'model_id': training_result.model_id
                }
                
                return True
            else:
                logger.error(f"Model training failed: {training_result.error_message}")
                return False
            
        except Exception as e:
            logger.error(f"Model training example failed: {e}")
            return False
    
    def example_5_proof_system_usage(self):
        """
        Example 5: Proof System Usage
        
        Demonstrates proof system usage adapted from Saraphis proof patterns.
        """
        logger.info("=" * 60)
        logger.info("Example 5: Proof System Usage")
        logger.info("=" * 60)
        
        try:
            # Get proof language plugin
            plugin_manager = self.core.plugin_manager
            proof_plugin = plugin_manager.get_plugin('proof_languages', 'molecular')
            
            if not proof_plugin:
                logger.warning("Molecular proof plugin not available, using mock proof system")
                return self._mock_proof_system_usage()
            
            # Test molecular rule validation
            test_molecules = [
                {'smiles': 'CCO', 'name': 'ethanol'},
                {'smiles': 'CC(=O)O', 'name': 'acetic_acid'},
                {'smiles': 'c1ccccc1', 'name': 'benzene'},
                {'smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 'name': 'ibuprofen'}
            ]
            
            # Test drug-likeness rules
            logger.info("Testing drug-likeness rule validation...")
            
            for molecule in test_molecules:
                try:
                    # Create proof for drug-likeness
                    proof_context = {
                        'molecule': molecule,
                        'rules': ['lipinski_rule_of_five', 'veber_rules'],
                        'confidence_threshold': 0.8
                    }
                    
                    proof_result = proof_plugin.construct_proof(proof_context)
                    
                    if proof_result.status.name == 'SUCCESS':
                        logger.info(f"Proof for {molecule['name']}:")
                        logger.info(f"  Valid: {proof_result.proof_valid}")
                        logger.info(f"  Confidence: {proof_result.confidence:.4f}")
                        
                        if hasattr(proof_result, 'rule_violations'):
                            violations = proof_result.rule_violations
                            if violations:
                                logger.info(f"  Rule violations: {violations}")
                            else:
                                logger.info("  No rule violations")
                    else:
                        logger.warning(f"Proof construction failed for {molecule['name']}")
                        
                except Exception as e:
                    logger.error(f"Error processing {molecule['name']}: {e}")
            
            # Test complex proof scenarios
            logger.info("Testing complex proof scenarios...")
            
            complex_scenarios = [
                {
                    'name': 'ADMET_compliance',
                    'context': {
                        'rules': ['admet_rules', 'reactive_groups'],
                        'thresholds': {'bioavailability': 0.3, 'toxicity': 0.1}
                    }
                },
                {
                    'name': 'synthetic_accessibility',
                    'context': {
                        'rules': ['synthetic_complexity', 'reaction_feasibility'],
                        'target_complexity': 'medium'
                    }
                }
            ]
            
            scenario_results = {}
            for scenario in complex_scenarios:
                try:
                    start_time = time.time()
                    result = proof_plugin.construct_proof(scenario['context'])
                    proof_time = time.time() - start_time
                    
                    scenario_results[scenario['name']] = {
                        'success': result.status.name == 'SUCCESS',
                        'proof_time': proof_time,
                        'result': result
                    }
                    
                    logger.info(f"Scenario {scenario['name']}: {result.status.name} in {proof_time:.4f}s")
                    
                except Exception as e:
                    logger.error(f"Scenario {scenario['name']} failed: {e}")
                    scenario_results[scenario['name']] = {'success': False, 'error': str(e)}
            
            # Store proof results
            self.results['proof_system'] = {
                'molecule_proofs': len(test_molecules),
                'scenario_results': scenario_results,
                'proof_plugin_available': True
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Proof system example failed: {e}")
            return False
    
    async def example_6_async_operations(self):
        """
        Example 6: Asynchronous Operations
        
        Demonstrates async operations adapted from Saraphis async patterns.
        """
        logger.info("=" * 60)
        logger.info("Example 6: Asynchronous Operations")
        logger.info("=" * 60)
        
        try:
            # Simulate async data processing tasks
            async def process_data_async(data_id: int, processing_time: float):
                logger.info(f"Starting async processing for data {data_id}")
                await asyncio.sleep(processing_time)  # Simulate processing
                return {'data_id': data_id, 'result': f'processed_{data_id}', 'time': processing_time}
            
            # Create multiple async tasks
            tasks = []
            processing_times = [0.5, 1.0, 0.3, 0.8, 0.6]  # Different processing times
            
            for i, proc_time in enumerate(processing_times):
                task = process_data_async(i, proc_time)
                tasks.append(task)
            
            # Execute tasks concurrently
            start_time = time.time()
            logger.info(f"Starting {len(tasks)} async tasks...")
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            total_time = time.time() - start_time
            sequential_time = sum(processing_times)
            
            # Process results
            successful_results = []
            failed_results = []
            
            for result in results:
                if isinstance(result, Exception):
                    failed_results.append(str(result))
                else:
                    successful_results.append(result)
            
            logger.info(f"Async operations completed:")
            logger.info(f"  Total tasks: {len(tasks)}")
            logger.info(f"  Successful: {len(successful_results)}")
            logger.info(f"  Failed: {len(failed_results)}")
            logger.info(f"  Concurrent time: {total_time:.2f}s")
            logger.info(f"  Sequential time would be: {sequential_time:.2f}s")
            logger.info(f"  Time saved: {sequential_time - total_time:.2f}s ({(1 - total_time/sequential_time)*100:.1f}%)")
            
            # Test async with orchestrator
            if self.orchestrator:
                logger.info("Testing orchestrator async operations...")
                
                # Create orchestrator tasks
                orchestrator_tasks = [
                    {'type': 'feature_extraction', 'data': {'id': i, 'content': f'sample_{i}'}}
                    for i in range(3)
                ]
                
                try:
                    orchestrator_results = await self.orchestrator.execute_async_workflow(orchestrator_tasks)
                    logger.info(f"Orchestrator processed {len(orchestrator_results)} tasks")
                except Exception as e:
                    logger.warning(f"Orchestrator async test failed: {e}")
            
            # Store async results
            self.results['async_operations'] = {
                'total_tasks': len(tasks),
                'successful_tasks': len(successful_results),
                'failed_tasks': len(failed_results),
                'concurrent_time': total_time,
                'sequential_time': sequential_time,
                'time_saved_percent': (1 - total_time/sequential_time) * 100
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Async operations example failed: {e}")
            return False
    
    def example_7_performance_monitoring(self):
        """
        Example 7: Performance Monitoring
        
        Demonstrates performance monitoring adapted from Saraphis monitoring patterns.
        """
        logger.info("=" * 60)
        logger.info("Example 7: Performance Monitoring")
        logger.info("=" * 60)
        
        try:
            # Monitor system performance during operations
            from core.performance_monitor import PerformanceMonitor
            
            monitor = PerformanceMonitor()
            
            # Start monitoring
            monitor.start_monitoring()
            
            # Simulate various operations
            operations = [
                {'name': 'data_loading', 'duration': 1.0},
                {'name': 'feature_extraction', 'duration': 2.0},
                {'name': 'model_training', 'duration': 3.0},
                {'name': 'prediction', 'duration': 0.5},
                {'name': 'validation', 'duration': 1.5}
            ]
            
            operation_metrics = {}
            
            for operation in operations:
                logger.info(f"Executing operation: {operation['name']}")
                
                # Track operation
                with monitor.track_operation(operation['name']) as tracker:
                    # Simulate work
                    start_time = time.time()
                    time.sleep(operation['duration'])
                    
                    # Simulate memory usage
                    tracker.add_metric('memory_usage_mb', np.random.randint(100, 500))
                    tracker.add_metric('cpu_usage_percent', np.random.randint(20, 80))
                    
                    actual_duration = time.time() - start_time
                    operation_metrics[operation['name']] = actual_duration
            
            # Get performance summary
            performance_summary = monitor.get_performance_summary()
            
            logger.info("Performance monitoring results:")
            logger.info(f"  Total operations: {len(operations)}")
            logger.info(f"  Monitoring duration: {performance_summary.get('total_time', 0):.2f}s")
            
            # Display operation metrics
            for op_name, duration in operation_metrics.items():
                logger.info(f"  {op_name}: {duration:.2f}s")
            
            # System resource usage
            system_metrics = monitor.get_system_metrics()
            if system_metrics:
                logger.info("System metrics:")
                for metric_name, metric_value in system_metrics.items():
                    logger.info(f"  {metric_name}: {metric_value}")
            
            # Store performance history
            self.performance_history.append({
                'timestamp': time.time(),
                'operations': operation_metrics,
                'system_metrics': system_metrics,
                'summary': performance_summary
            })
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            # Store performance results
            self.results['performance_monitoring'] = {
                'operations_tracked': len(operations),
                'total_monitoring_time': sum(operation_metrics.values()),
                'average_operation_time': np.mean(list(operation_metrics.values())),
                'operation_metrics': operation_metrics,
                'system_metrics': system_metrics
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Performance monitoring example failed: {e}")
            return False
    
    def run_all_examples(self):
        """Run all examples in sequence"""
        logger.info("Starting Universal AI Core Basic Usage Examples")
        logger.info("=" * 80)
        
        examples = [
            ('Configuration Management', self.example_1_basic_configuration),
            ('Plugin Management', self.example_2_plugin_management),
            ('Data Processing Workflow', self.example_3_data_processing_workflow),
            ('Model Training and Evaluation', self.example_4_model_training_evaluation),
            ('Proof System Usage', self.example_5_proof_system_usage),
            ('Performance Monitoring', self.example_7_performance_monitoring)
        ]
        
        results_summary = {}
        total_start_time = time.time()
        
        for example_name, example_func in examples:
            try:
                logger.info(f"\nRunning example: {example_name}")
                start_time = time.time()
                
                success = example_func()
                
                execution_time = time.time() - start_time
                results_summary[example_name] = {
                    'success': success,
                    'execution_time': execution_time
                }
                
                if success:
                    logger.info(f"✓ {example_name} completed successfully in {execution_time:.2f}s")
                else:
                    logger.error(f"✗ {example_name} failed after {execution_time:.2f}s")
                    
            except Exception as e:
                logger.error(f"✗ {example_name} crashed: {e}")
                results_summary[example_name] = {
                    'success': False,
                    'execution_time': 0,
                    'error': str(e)
                }
        
        # Run async example separately
        try:
            logger.info("\nRunning example: Asynchronous Operations")
            start_time = time.time()
            
            success = asyncio.run(self.example_6_async_operations())
            
            execution_time = time.time() - start_time
            results_summary['Asynchronous Operations'] = {
                'success': success,
                'execution_time': execution_time
            }
            
            if success:
                logger.info(f"✓ Asynchronous Operations completed successfully in {execution_time:.2f}s")
            else:
                logger.error(f"✗ Asynchronous Operations failed after {execution_time:.2f}s")
                
        except Exception as e:
            logger.error(f"✗ Asynchronous Operations crashed: {e}")
            results_summary['Asynchronous Operations'] = {
                'success': False,
                'execution_time': 0,
                'error': str(e)
            }
        
        total_time = time.time() - total_start_time
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("EXAMPLES EXECUTION SUMMARY")
        logger.info("=" * 80)
        
        successful_examples = sum(1 for result in results_summary.values() if result['success'])
        total_examples = len(results_summary)
        
        logger.info(f"Total examples: {total_examples}")
        logger.info(f"Successful: {successful_examples}")
        logger.info(f"Failed: {total_examples - successful_examples}")
        logger.info(f"Success rate: {successful_examples/total_examples:.1%}")
        logger.info(f"Total execution time: {total_time:.2f}s")
        
        for example_name, result in results_summary.items():
            status = "✓" if result['success'] else "✗"
            logger.info(f"  {status} {example_name}: {result['execution_time']:.2f}s")
        
        return results_summary
    
    # Helper methods for mock data and fallback operations
    
    def _prepare_sample_data(self):
        """Prepare sample data for examples"""
        return [
            {'smiles': 'CCO', 'target': 0.5},
            {'smiles': 'CC(=O)O', 'target': 0.3},
            {'smiles': 'c1ccccc1', 'target': 0.8},
            {'smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 'target': 0.7},
            {'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'target': 0.6}
        ]
    
    def _prepare_training_data(self):
        """Prepare training data for model examples"""
        sample_data = self._prepare_sample_data()
        
        # Mock feature matrix and targets
        n_samples = len(sample_data)
        n_features = 100
        
        X = np.random.randn(n_samples, n_features)
        y = np.array([item['target'] for item in sample_data])
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': [f'feature_{i}' for i in range(n_features)]
        }
    
    def _mock_data_processing_workflow(self):
        """Mock data processing workflow when plugins unavailable"""
        logger.info("Using mock data processing workflow")
        
        sample_data = self._prepare_sample_data()
        
        processed_results = []
        for sample in sample_data:
            processed_results.append({
                'input': sample,
                'features': np.random.randn(50),  # Mock features
                'metadata': {'processing_time': 0.1}
            })
        
        self.results['data_processing'] = {
            'total_samples': len(sample_data),
            'successful_samples': len(processed_results),
            'success_rate': 1.0,
            'processing_time': 0.5,
            'throughput': len(sample_data) / 0.5,
            'results': processed_results[:5]
        }
        
        return True
    
    def _mock_model_training(self):
        """Mock model training when plugins unavailable"""
        logger.info("Using mock model training")
        
        # Mock training metrics
        mock_metrics = {
            'random_forest': {
                'r2_score': 0.85,
                'rmse': 0.12,
                'mae': 0.09
            },
            'xgboost': {
                'r2_score': 0.88,
                'rmse': 0.11,
                'mae': 0.08
            }
        }
        
        self.results['model_training'] = {
            'training_time': 2.5,
            'metrics': mock_metrics,
            'model_config': {'model_types': ['random_forest', 'xgboost']},
            'model_id': 'mock_model_123'
        }
        
        return True
    
    def _mock_proof_system_usage(self):
        """Mock proof system usage when plugins unavailable"""
        logger.info("Using mock proof system")
        
        self.results['proof_system'] = {
            'molecule_proofs': 4,
            'scenario_results': {
                'ADMET_compliance': {'success': True, 'proof_time': 0.5},
                'synthetic_accessibility': {'success': True, 'proof_time': 0.3}
            },
            'proof_plugin_available': False
        }
        
        return True


def main():
    """Main function to run basic usage examples"""
    try:
        # Initialize examples
        examples = BasicUsageExamples()
        
        # Run all examples
        results = examples.run_all_examples()
        
        # Save results
        results_file = Path('examples/basic_usage_results.json')
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    main()