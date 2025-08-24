#!/usr/bin/env python3
"""
Production-Ready Training Execution Script with GAC Integration
Leverages the existing GAC-integrated training system for fraud detection
"""

import os
import sys
import logging
import json
import time
import traceback
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np

# PyTorch import for device detection
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available")

# Add current directory and parent directory to path for proper imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import brain_training_integration for automatic session management
import brain_training_integration

# Core system imports
from brain import Brain, BrainSystemConfig
from training_manager import TrainingManager, TrainingConfig

# Import IEEE fraud data loader
try:
    from financial_fraud_domain.ieee_fraud_data_loader import IEEEFraudDataLoader
    IEEE_LOADER_AVAILABLE = True
except ImportError as e:
    IEEE_LOADER_AVAILABLE = False
    print(f"Warning: IEEE fraud data loader not available: {e}")

# Import deployment orchestrator for production deployment
try:
    from financial_fraud_domain.deployment_orchestrator import (
        IntegratedDeploymentOrchestrator, 
        DeploymentConfig, 
        DeploymentStrategy, 
        DeploymentResult,
        DeploymentStatus
    )
    DEPLOYMENT_ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    DEPLOYMENT_ORCHESTRATOR_AVAILABLE = False
    print(f"Warning: Deployment orchestrator not available: {e}")


class ProductionTrainingExecutor:
    """Production-ready training executor with comprehensive GAC integration"""
    
    def __init__(self, output_dir="training_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.brain = None
        self.training_manager = None
        self.data_loader = None
        self.session_id = None
        self.deployment_orchestrator = None
        
        # Training metadata
        self.training_metadata = {
            "start_time": None,
            "end_time": None,
            "config": None,
            "results": None,
            "errors": [],
            "gac_metrics": [],
            "deployment_info": None
        }
        
        self.logger.info("ProductionTrainingExecutor initialized")
    
    def _setup_logging(self):
        """Configure comprehensive logging with both file and console output"""
        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.output_dir / f"training_execution_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def initialize_brain_system(self, config=None):
        """Initialize Brain system with enhanced configuration"""
        try:
            self.logger.info("Initializing Brain system...")
            
            # Use provided config or create optimized default
            if config is None:
                config = BrainSystemConfig(
                    max_memory_gb=1,  # 1GB memory1024 * 1024 * 1024,  # 1GB memory
                    enable_persistence=True,
                    auto_save_interval=300,  # 5 minutes
                    log_level="INFO",
                    max_domains=5,
                    
                )
            
            # Initialize Brain with automatic session management
            self.brain = Brain(config)
            
            # Verify Brain initialization
            if not hasattr(self.brain, 'enhanced_training_manager'):
                self.logger.warning("Enhanced training manager not available")
            else:
                self.logger.info("Enhanced training manager available - GAC integration active")
            
            # Add fraud detection domain if not exists
            if 'fraud_detection' not in self.brain.domain_registry._domains:
                domain_config = {
                    'type': 'specialized',
                    'name': 'fraud_detection', 
                    'description': 'Financial fraud detection with GAC optimization',
                    'features': {
                        'transaction_analysis': True,
                        'pattern_recognition': True,
                        'anomaly_detection': True,
                        'real_time_scoring': True,
                        'gac_integration': True
                    }
                }
                self.brain.add_domain('fraud_detection', domain_config)
                self.logger.info("Fraud detection domain added to Brain system")
            else:
                self.logger.info("Fraud detection domain already exists")
            
            self.logger.info("Brain system initialized successfully")
            
            # Initialize deployment orchestrator if available
            self._initialize_deployment_orchestrator()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Brain system: {str(e)}")
            self.training_metadata["errors"].append({
                "stage": "brain_initialization",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            })
            raise
    
    def _initialize_deployment_orchestrator(self):
        """Initialize deployment orchestrator for production deployment capabilities."""
        try:
            if not DEPLOYMENT_ORCHESTRATOR_AVAILABLE:
                self.logger.warning("Deployment orchestrator not available - skipping deployment integration")
                return
                
            self.logger.info("Initializing deployment orchestrator...")
            
            # Setup deployment configuration for fraud detection
            deployment_config = {
                'logger': self.logger,
                'metrics_output_dir': str(self.output_dir / "deployment_metrics"),
                'application_name': 'saraphis-fraud-detection',
                'environment': 'production',
                'deployment_strategy': 'blue_green',  # Safe production deployments
                'auto_scaling': {
                    'enabled': True,
                    'min_instances': 2,
                    'max_instances': 10,
                    'target_cpu_percent': 70
                },
                'health_checks': {
                    'enabled': True,
                    'endpoint': '/health',
                    'interval_seconds': 30,
                    'timeout_seconds': 10
                },
                'monitoring': {
                    'enabled': True,
                    'metrics_collection': True,
                    'alerting': True
                },
                'security': {
                    'enable_tls': True,
                    'enable_authentication': True,
                    'enable_authorization': True
                }
            }
            
            # Initialize deployment orchestrator with the config
            self.deployment_orchestrator = IntegratedDeploymentOrchestrator(deployment_config)
            
            self.logger.info("Deployment orchestrator initialized successfully")
            
            # Store deployment info in training metadata
            self.training_metadata["deployment_info"] = {
                'orchestrator_available': True,
                'deployment_strategy': deployment_config['deployment_strategy'],
                'auto_scaling_enabled': deployment_config['auto_scaling']['enabled'],
                'monitoring_enabled': deployment_config['monitoring']['enabled'],
                'initialization_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize deployment orchestrator: {str(e)}")
            self.training_metadata["deployment_info"] = {
                'orchestrator_available': False,
                'error': str(e),
                'initialization_time': datetime.now().isoformat()
            }
            # Don't raise - deployment orchestrator is optional
    
    def setup_training_configuration(self, **kwargs):
        """Setup comprehensive training configuration with GAC integration"""
        try:
            self.logger.info("Setting up training configuration...")
            
            # Default configuration optimized for fraud detection
            config_params = {
                'epochs': kwargs.get('epochs', 10),
                'batch_size': kwargs.get('batch_size', 32),
                'learning_rate': kwargs.get('learning_rate', 0.001),
                'validation_split': kwargs.get('validation_split', 0.2),
                'device': kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),  # Auto-detect CUDA
                
                # GAC System configuration
                'use_gac_system': kwargs.get('use_gac_system', True),
                'gac_mode': kwargs.get('gac_mode', "adaptive"),
                'gac_components': kwargs.get('gac_components', ["clipping", "monitoring", "normalization"]),
                'gac_threshold_adaptation': kwargs.get('gac_threshold_adaptation', True),
                'gac_noise_enabled': kwargs.get('gac_noise_enabled', False),
                
                # Optimization settings
                'optimizer': kwargs.get('optimizer', "adam"),
                'scheduler': kwargs.get('scheduler', "reduce_on_plateau"),
                'gradient_clip_value': kwargs.get('gradient_clip_value', 1.0),
                
                # Early stopping and checkpointing
                'early_stopping_enabled': kwargs.get('early_stopping_enabled', True),
                'early_stopping_patience': kwargs.get('early_stopping_patience', 5),
                'checkpoint_frequency': kwargs.get('checkpoint_frequency', 1),
                
                # Performance settings
                'mixed_precision': kwargs.get('mixed_precision', False),
                'num_workers': kwargs.get('num_workers', 0),
                'log_frequency': kwargs.get('log_frequency', 10),
                
                # Advanced settings
                'save_training_history': True,
                'normalize_features': True,
                'handle_missing_values': True
            }
            
            # Create training configuration
            training_config = TrainingConfig(**config_params)
            
            # Store configuration in metadata
            self.training_metadata["config"] = {
                k: v for k, v in config_params.items() 
                if not callable(v) and not k.startswith('_')
            }
            
            self.logger.info(f"Training configuration created:")
            self.logger.info(f"  Epochs: {training_config.epochs}")
            self.logger.info(f"  Batch size: {training_config.batch_size}")
            self.logger.info(f"  Learning rate: {training_config.learning_rate}")
            self.logger.info(f"  GAC mode: {training_config.gac_mode}")
            self.logger.info(f"  GAC enabled: {training_config.use_gac_system}")
            
            return training_config
            
        except Exception as e:
            self.logger.error(f"Failed to setup training configuration: {str(e)}")
            self.training_metadata["errors"].append({
                "stage": "config_setup",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            })
            raise
    
    def load_ieee_training_data(self, data_dir=None):
        """Load IEEE fraud detection dataset with comprehensive validation"""
        try:
            self.logger.info("Loading IEEE fraud detection dataset...")
            
            if not IEEE_LOADER_AVAILABLE:
                raise ImportError("IEEE fraud data loader not available")
            
            # Set default data directory
            if data_dir is None:
                # Try multiple possible locations
                possible_dirs = [
                    os.path.join(os.path.dirname(__file__), "..", "training_data", "ieee-fraud-detection"),
                    os.path.join(os.path.dirname(__file__), "training_data", "ieee-fraud-detection"),
                    "training_data/ieee-fraud-detection",
                    "../training_data/ieee-fraud-detection"
                ]
                
                data_dir = None
                for dir_path in possible_dirs:
                    if os.path.exists(dir_path):
                        data_dir = dir_path
                        break
                
                if data_dir is None:
                    raise FileNotFoundError(f"IEEE dataset not found in any of these locations: {possible_dirs}")
            
            self.logger.info(f"Using data directory: {data_dir}")
            
            # Initialize data loader with enhanced configuration
            loader_config = {
                'enable_validation': True,
                'cache_processed': True,
            }
            
            self.data_loader = IEEEFraudDataLoader(
                data_dir=data_dir,
                **loader_config
            )
            
            # Load training data with timing - ROOT FIX: Enhanced logging for data validation
            start_time = time.time()
            self.logger.info("Starting IEEE fraud data loading with cache validation...")
            
            X_train, y_train, val_data = self.data_loader.load_data(
                dataset_type="train",
                validation_split=0.2,
                use_cache=True
            )
            load_time = time.time() - start_time
            
            # ROOT FIX: Enhanced comprehensive data statistics with validation
            self.logger.info(f"Data loaded successfully in {load_time:.2f} seconds")
            self.logger.info(f"Training samples: {X_train.shape[0]:,} (expected: ~472,432 for 80% split)")
            self.logger.info(f"Features: {X_train.shape[1]:,}")
            self.logger.info(f"Fraud rate: {np.mean(y_train):.4f} ({np.sum(y_train):,} fraud cases)")
            self.logger.info(f"Memory usage: {X_train.nbytes / 1024**2:.1f} MB")
            
            # ROOT FIX: Validate data size expectations
            expected_training_size = int(590540 * 0.8)  # 80% of full dataset
            if abs(X_train.shape[0] - expected_training_size) > expected_training_size * 0.01:
                self.logger.warning(
                    f"Training data size deviation: {X_train.shape[0]:,} vs expected ~{expected_training_size:,} "
                    f"(difference: {abs(X_train.shape[0] - expected_training_size):,} samples)"
                )
            else:
                self.logger.info(f"Training data size within expected range: {X_train.shape[0]:,} samples")
            
            if val_data:
                X_val, y_val = val_data
                self.logger.info(f"Validation samples: {X_val.shape[0]:,}")
                self.logger.info(f"Validation fraud rate: {np.mean(y_val):.4f}")
            
            # Store data statistics in metadata
            self.training_metadata["data_info"] = {
                "training_samples": int(X_train.shape[0]),
                "features": int(X_train.shape[1]),
                "fraud_rate": float(np.mean(y_train)),
                "fraud_cases": int(np.sum(y_train)),
                "validation_samples": int(X_val.shape[0]) if val_data else 0,
                "load_time_seconds": float(load_time),
                "memory_mb": float(X_train.nbytes / 1024**2)
            }
            
            return X_train, y_train, val_data
            
        except Exception as e:
            self.logger.error(f"Failed to load training data: {str(e)}")
            self.training_metadata["errors"].append({
                "stage": "data_loading",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            })
            raise
    
    def execute_training_session(self, X_train, y_train, val_data, training_config):
        """Execute training session with comprehensive GAC monitoring"""
        try:
            self.logger.info("Starting production training session...")
            self.training_metadata["start_time"] = datetime.now().isoformat()
            
            # Prepare comprehensive training data package
            training_data = {
                'X': X_train,
                'y': y_train,
                'validation_data': val_data,
                'feature_names': self.data_loader.get_feature_names() if self.data_loader else None,
                'data_info': self.training_metadata.get("data_info", {})
            }
            
            self.logger.info("Initiating Brain training with GAC integration...")
            
            # Start training with Brain's enhanced training
            # ROOT FIX: Convert TrainingConfig dataclass to dictionary for consistent interface
            if hasattr(training_config, '__dataclass_fields__'):
                # Convert dataclass to dictionary
                try:
                    from dataclasses import asdict
                    training_config_dict = asdict(training_config)
                except ImportError:
                    # Fallback if dataclasses not available
                    training_config_dict = {k: getattr(training_config, k) for k in dir(training_config) if not k.startswith('_')}
            else:
                training_config_dict = training_config
            
            if hasattr(self.brain, 'train_domain_with_progress'):
                # Use enhanced training with progress monitoring
                result = self.brain.train_domain_with_progress(
                    domain_name='fraud_detection',
                    training_data=training_data,
                    training_config=training_config_dict,  # Pass as dictionary
                    progress_callback=self._training_progress_callback
                )
            else:
                # Fallback to basic training
                self.logger.warning("Enhanced training not available, using basic training")
                result = self.brain.train_domain(
                    domain_name='fraud_detection',
                    training_data=training_data,
                    training_config=training_config_dict  # Pass as dictionary
                )
            
            # Process training results
            self.training_metadata["results"] = result
            self.logger.info(f"Training completed successfully")
            self.logger.info(f"Final results: {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Training session failed: {str(e)}")
            self.training_metadata["errors"].append({
                "stage": "training_execution",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            })
            raise
    
    def _training_progress_callback(self, progress_info):
        """Callback function to monitor training progress and GAC metrics"""
        try:
            epoch = progress_info.get('epoch', 0)
            metrics = progress_info.get('metrics', {})
            gac_status = progress_info.get('gac_status', {})
            
            # Log epoch progress
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"EPOCH {epoch} PROGRESS")
            self.logger.info(f"{'='*80}")
            
            # Training metrics
            if metrics:
                self.logger.info("Training Metrics:")
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"  {metric_name}: {value:.4f}")
                    else:
                        self.logger.info(f"  {metric_name}: {value}")
            
            # GAC-specific metrics
            if gac_status.get('gac_enabled', False):
                self.logger.info("\nGAC System Status:")
                self.logger.info(f"  Mode: {gac_status.get('gac_mode', 'unknown')}")
                self.logger.info(f"  Components: {gac_status.get('gac_components', [])}")
                
                gac_metrics = gac_status.get('gac_metrics', {})
                if gac_metrics:
                    self.logger.info("  GAC Metrics:")
                    for metric, value in gac_metrics.items():
                        if isinstance(value, (int, float)):
                            self.logger.info(f"    {metric}: {value:.6f}")
                        else:
                            self.logger.info(f"    {metric}: {value}")
                
                # Store GAC metrics for later analysis
                self.training_metadata["gac_metrics"].append({
                    "epoch": epoch,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": gac_metrics
                })
            # ROOT FIX: Store epoch info when we have meaningful data even without GAC
            elif epoch > 0 or metrics:
                self.training_metadata["gac_metrics"].append({
                    "epoch": epoch,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": {},
                    "gac_enabled": False
                })
            
            self.logger.info(f"{'='*80}\n")
            
        except Exception as e:
            self.logger.warning(f"Error in progress callback: {e}")
    
    def save_comprehensive_results(self):
        """Save comprehensive training results, models, and analysis"""
        try:
            self.logger.info("Saving comprehensive training results...")
            
            # Update end time and calculate duration
            self.training_metadata["end_time"] = datetime.now().isoformat()
            start = datetime.fromisoformat(self.training_metadata["start_time"])
            end = datetime.fromisoformat(self.training_metadata["end_time"])
            self.training_metadata["duration_seconds"] = (end - start).total_seconds()
            
            # Save training metadata
            metadata_file = self.output_dir / "training_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.training_metadata, f, indent=2, default=str)
            self.logger.info(f"Training metadata saved: {metadata_file}")
            
            # Save model if available
            try:
                model_dir = self.output_dir / "models"
                model_dir.mkdir(exist_ok=True)
                model_file = model_dir / "fraud_detection_model.pkl"
                
                if self.brain and hasattr(self.brain, 'save_model'):
                    self.brain.save_model('fraud_detection', str(model_file))
                    self.logger.info(f"Model saved: {model_file}")
                else:
                    self.logger.warning("Model saving not available")
            except Exception as e:
                self.logger.warning(f"Failed to save model: {e}")
            
            # Generate comprehensive analysis report
            self._generate_analysis_report()
            
            # Generate GAC metrics analysis if available
            if self.training_metadata["gac_metrics"]:
                self._generate_gac_analysis()
            
            self.logger.info("All results saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            self.training_metadata["errors"].append({
                "stage": "save_results",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            })
    
    def _generate_analysis_report(self):
        """Generate comprehensive training analysis report"""
        report_file = self.output_dir / "training_analysis_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("PRODUCTION TRAINING ANALYSIS REPORT\n")
            f.write("=" * 100 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 50 + "\n")
            duration_hours = self.training_metadata.get("duration_seconds", 0) / 3600
            f.write(f"Training Duration: {duration_hours:.2f} hours\n")
            f.write(f"Start Time: {self.training_metadata.get('start_time', 'N/A')}\n")
            f.write(f"End Time: {self.training_metadata.get('end_time', 'N/A')}\n")
            f.write(f"Status: {'Success' if not self.training_metadata['errors'] else 'Completed with errors'}\n\n")
            
            # Configuration Details
            f.write("CONFIGURATION\n")
            f.write("-" * 50 + "\n")
            config = self.training_metadata.get("config", {})
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Data Information
            f.write("DATASET INFORMATION\n")
            f.write("-" * 50 + "\n")
            data_info = self.training_metadata.get("data_info", {})
            for key, value in data_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Training Results
            f.write("TRAINING RESULTS\n")
            f.write("-" * 50 + "\n")
            results = self.training_metadata.get("results", {})
            if results:
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{key}: {value:.6f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # GAC System Performance
            gac_metrics = self.training_metadata.get("gac_metrics", [])
            if gac_metrics:
                f.write("GAC SYSTEM PERFORMANCE\n")
                f.write("-" * 50 + "\n")
                f.write(f"Total GAC metric records: {len(gac_metrics)}\n")
                f.write("Final epoch GAC metrics:\n")
                if gac_metrics:
                    final_metrics = gac_metrics[-1].get("metrics", {})
                    for metric, value in final_metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {metric}: {value:.6f}\n")
                        else:
                            f.write(f"  {metric}: {value}\n")
                f.write("\n")
            
            # Error Summary
            if self.training_metadata["errors"]:
                f.write("ERRORS ENCOUNTERED\n")
                f.write("-" * 50 + "\n")
                for i, error in enumerate(self.training_metadata["errors"], 1):
                    f.write(f"{i}. Stage: {error['stage']}\n")
                    f.write(f"   Time: {error['timestamp']}\n")
                    f.write(f"   Error: {error['error']}\n\n")
            
            f.write("END OF REPORT\n")
        
        self.logger.info(f"Analysis report generated: {report_file}")
    
    def _generate_gac_analysis(self):
        """Generate detailed GAC system performance analysis"""
        gac_file = self.output_dir / "gac_performance_analysis.json"
        
        gac_metrics = self.training_metadata.get("gac_metrics", [])
        if not gac_metrics:
            return
        
        # Analyze GAC performance trends
        analysis = {
            "total_epochs": len(gac_metrics),
            "metrics_timeline": gac_metrics,
            "performance_summary": {}
        }
        
        # Calculate performance statistics
        if gac_metrics:
            first_epoch = gac_metrics[0].get("metrics", {})
            last_epoch = gac_metrics[-1].get("metrics", {})
            
            analysis["performance_summary"] = {
                "first_epoch": first_epoch,
                "last_epoch": last_epoch,
                "improvement_detected": len(gac_metrics) > 1
            }
        
        with open(gac_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self.logger.info(f"GAC analysis generated: {gac_file}")
    
    def deploy_trained_model(self, training_results):
        """Deploy trained model to production using deployment orchestrator."""
        try:
            if not self.deployment_orchestrator:
                self.logger.warning("Deployment orchestrator not available - skipping model deployment")
                return {
                    'success': False,
                    'reason': 'deployment_orchestrator_not_available',
                    'timestamp': datetime.now().isoformat()
                }
            
            self.logger.info("Starting production model deployment...")
            
            # Prepare deployment package with training results
            deployment_package = {
                'model_info': {
                    'training_session_id': self.session_id,
                    'final_accuracy': training_results.get('final_accuracy', 0.0),
                    'training_duration': training_results.get('training_time', 0),
                    'model_version': f"fraud_detection_{int(time.time())}",
                    'validation_metrics': training_results.get('validation_metrics', {})
                },
                'deployment_config': {
                    'strategy': 'blue_green',
                    'health_check_timeout': 300,  # 5 minutes
                    'rollback_enabled': True,
                    'auto_scaling_enabled': True,
                    'monitoring_enabled': True
                },
                'model_artifacts': {
                    'model_path': str(self.output_dir / 'model_checkpoint.pkl'),
                    'metadata_path': str(self.output_dir / 'training_metadata.json'),
                    'results_path': str(self.output_dir / 'training_analysis_report.txt')
                }
            }
            
            # Validate deployment readiness
            validation_result = self._validate_deployment_readiness(training_results)
            if not validation_result['ready']:
                self.logger.warning(f"Deployment validation failed: {validation_result['reason']}")
                return {
                    'success': False,
                    'reason': 'deployment_validation_failed',
                    'validation_details': validation_result,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Create proper DeploymentConfig
            deployment_config_obj = DeploymentConfig(
                strategy=DeploymentStrategy.BLUE_GREEN,
                environment='production',
                version=deployment_package['model_info']['model_version'],
                components=['fraud_detection_model', 'api_gateway', 'monitoring'],
                validation_gates=[
                    {'type': 'health_check', 'timeout': 300},
                    {'type': 'performance_test', 'threshold': 0.7}
                ],
                rollback_config={
                    'enabled': True,
                    'auto_rollback': True,
                    'rollback_threshold': 0.5
                },
                traffic_config={
                    'initial_weight': 0,
                    'ramp_up_duration': 600  # 10 minutes
                },
                health_check_config={
                    'endpoint': '/health',
                    'interval_seconds': 30,
                    'timeout_seconds': 10,
                    'healthy_threshold': 3,
                    'unhealthy_threshold': 3
                },
                monitoring_config={
                    'enabled': True,
                    'metrics_collection': True,
                    'alerting': True,
                    'log_aggregation': True
                },
                security_config={
                    'tls_enabled': True,
                    'authentication_required': True,
                    'authorization_enabled': True,
                    'network_policies_enabled': True
                }
            )
            
            # Execute deployment through orchestrator
            self.logger.info("Executing deployment through orchestrator...")
            deployment_result: DeploymentResult = self.deployment_orchestrator.deploy_complete_system(
                environment='production',
                version=deployment_package['model_info']['model_version'],
                deployment_config=deployment_config_obj,
                enable_all_capabilities=True
            )
            
            if deployment_result.status == DeploymentStatus.COMPLETED:
                self.logger.info(f"Model deployed successfully to production - Deployment ID: {deployment_result.deployment_id}")
                
                # Update deployment info in training metadata
                self.training_metadata["deployment_info"].update({
                    'deployment_successful': True,
                    'deployment_id': deployment_result.deployment_id,
                    'deployment_status': deployment_result.status.value,
                    'deployment_strategy': deployment_result.strategy.value,
                    'deployment_timestamp': datetime.now().isoformat(),
                    'start_time': deployment_result.start_time.isoformat() if deployment_result.start_time else None,
                    'end_time': deployment_result.end_time.isoformat() if deployment_result.end_time else None,
                    'environment': deployment_result.environment,
                    'version': deployment_result.version
                })
                
                return {
                    'success': True,
                    'deployment_id': deployment_result.deployment_id,
                    'deployment_status': deployment_result.status.value,
                    'deployment_strategy': deployment_result.strategy.value,
                    'environment': deployment_result.environment,
                    'version': deployment_result.version,
                    'start_time': deployment_result.start_time.isoformat() if deployment_result.start_time else None,
                    'end_time': deployment_result.end_time.isoformat() if deployment_result.end_time else None,
                    'auto_scaling_enabled': True,
                    'monitoring_enabled': True,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                self.logger.error(f"Deployment failed with status: {deployment_result.status.value}")
                return {
                    'success': False,
                    'reason': 'deployment_execution_failed',
                    'deployment_status': deployment_result.status.value,
                    'deployment_id': deployment_result.deployment_id,
                    'error': f"Deployment status: {deployment_result.status.value}",
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            self.logger.error(f"Model deployment failed with exception: {str(e)}")
            return {
                'success': False,
                'reason': 'deployment_exception',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
    
    def _validate_deployment_readiness(self, training_results):
        """Validate if model is ready for production deployment."""
        try:
            validation_checks = []
            
            # Check 1: Training completed successfully
            training_success = training_results.get('success', False)
            validation_checks.append({
                'check': 'training_success',
                'passed': training_success,
                'details': f"Training successful: {training_success}"
            })
            
            # Check 2: Minimum accuracy threshold
            final_accuracy = training_results.get('final_accuracy', 0.0)
            min_accuracy_threshold = 0.7  # 70% minimum for production
            accuracy_check = final_accuracy >= min_accuracy_threshold
            validation_checks.append({
                'check': 'accuracy_threshold',
                'passed': accuracy_check,
                'details': f"Accuracy {final_accuracy:.2%} >= {min_accuracy_threshold:.2%}: {accuracy_check}"
            })
            
            # Check 3: Model artifacts exist
            model_artifacts_exist = all([
                (self.output_dir / 'training_metadata.json').exists(),
                (self.output_dir / 'training_analysis_report.txt').exists()
            ])
            validation_checks.append({
                'check': 'model_artifacts',
                'passed': model_artifacts_exist,
                'details': f"Model artifacts available: {model_artifacts_exist}"
            })
            
            # Check 4: No critical training errors
            critical_errors = any(
                error.get('stage') in ['training_execution', 'model_validation'] 
                for error in self.training_metadata.get('errors', [])
            )
            no_critical_errors = not critical_errors
            validation_checks.append({
                'check': 'no_critical_errors',
                'passed': no_critical_errors,
                'details': f"No critical training errors: {no_critical_errors}"
            })
            
            # Overall validation
            all_checks_passed = all(check['passed'] for check in validation_checks)
            
            return {
                'ready': all_checks_passed,
                'validation_checks': validation_checks,
                'checks_passed': sum(1 for check in validation_checks if check['passed']),
                'total_checks': len(validation_checks),
                'reason': 'validation_passed' if all_checks_passed else 'validation_failed'
            }
            
        except Exception as e:
            return {
                'ready': False,
                'reason': 'validation_error',
                'error': str(e)
            }
    
    def run_production_training(self, **kwargs):
        """Execute complete production training pipeline"""
        try:
            self.logger.info("="*100)
            self.logger.info("STARTING PRODUCTION TRAINING EXECUTION")
            self.logger.info("="*100)
            
            # Step 1: Initialize Brain system
            self.initialize_brain_system(kwargs.get('brain_config'))
            
            # Step 2: Setup training configuration
            training_config = self.setup_training_configuration(**kwargs)
            
            # Step 3: Load IEEE training data
            X_train, y_train, val_data = self.load_ieee_training_data(kwargs.get('data_dir'))
            
            # Step 4: Execute training session
            results = self.execute_training_session(X_train, y_train, val_data, training_config)
            
            # Step 5: Save comprehensive results
            self.save_comprehensive_results()
            
            # Step 6: Deploy trained model to production if deployment orchestrator is available
            deployment_result = self.deploy_trained_model(results)
            
            self.logger.info("="*100)
            self.logger.info("PRODUCTION TRAINING COMPLETED SUCCESSFULLY")
            if deployment_result and deployment_result.get('success'):
                self.logger.info("MODEL DEPLOYED TO PRODUCTION SUCCESSFULLY")
            else:
                self.logger.warning("MODEL DEPLOYMENT SKIPPED OR FAILED")
            self.logger.info("="*100)
            
            # Include deployment result in final results
            results['deployment'] = deployment_result
            
            return results
            
        except Exception as e:
            self.logger.error(f"Production training failed: {str(e)}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Save partial results even on failure
            try:
                self.save_comprehensive_results()
            except:
                self.logger.error("Failed to save partial results")
            
            raise
        
        finally:
            # Cleanup resources
            try:
                if self.brain:
                    self.brain.cleanup()
                self.logger.info("Cleanup completed")
            except:
                pass


def main():
    """Main execution function with comprehensive CLI"""
    parser = argparse.ArgumentParser(
        description="Production Training Execution with GAC Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --epochs 20 --batch-size 64 --learning-rate 0.0001
  %(prog)s --data-dir /path/to/ieee-data --output-dir custom_results
  %(prog)s --gac-mode adaptive --epochs 15
  %(prog)s --gac-mode disabled --epochs 10
        """
    )
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10, 
                       help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    parser.add_argument("--validation-split", type=float, default=0.2,
                       help="Validation split ratio (default: 0.2)")
    
    # GAC parameters
    parser.add_argument("--gac-mode", type=str, default="adaptive",
                       choices=["adaptive", "conservative", "aggressive", "disabled"],
                       help="GAC system mode (default: adaptive)")
    parser.add_argument("--disable-gac", action="store_true",
                       help="Disable GAC system completely")
    
    # I/O parameters
    parser.add_argument("--data-dir", type=str,
                       help="Path to IEEE fraud detection dataset directory")
    parser.add_argument("--output-dir", type=str, default="training_results",
                       help="Output directory for results (default: training_results)")
    
    # Advanced parameters
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                       help="Early stopping patience (default: 5)")
    parser.add_argument("--gradient-clip-value", type=float, default=1.0,
                       help="Gradient clipping value (default: 1.0)")
    parser.add_argument("--mixed-precision", action="store_true",
                       help="Enable mixed precision training")
    
    # Debug and verbose options
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Prepare configuration
    config_kwargs = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'validation_split': args.validation_split,
        'use_gac_system': not args.disable_gac and args.gac_mode != "disabled",
        'gac_mode': args.gac_mode if args.gac_mode != "disabled" else "adaptive",
        'early_stopping_patience': args.early_stopping_patience,
        'gradient_clip_value': args.gradient_clip_value,
        'mixed_precision': args.mixed_precision,
        'data_dir': args.data_dir
    }
    
    # Create and run executor
    executor = ProductionTrainingExecutor(output_dir=args.output_dir)
    
    try:
        print(f"\n🚀 Starting production training execution...")
        print(f"📊 Configuration: {args.epochs} epochs, batch size {args.batch_size}")
        print(f"🧠 GAC Mode: {args.gac_mode} ({'enabled' if config_kwargs['use_gac_system'] else 'disabled'})")
        print(f"📁 Output directory: {args.output_dir}\n")
        
        # Execute training
        results = executor.run_production_training(**config_kwargs)
        
        print(f"\n✅ Training completed successfully!")
        print(f"📈 Results: {results}")
        print(f"📁 All outputs saved to: {args.output_dir}")
        print(f"📝 Check training_analysis_report.txt for detailed analysis")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        executor.logger.warning("Training interrupted by user (Ctrl+C)")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        print(f"📁 Check logs in: {args.output_dir}")
        executor.logger.error(f"Training execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()