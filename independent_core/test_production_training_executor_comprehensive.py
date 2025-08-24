#!/usr/bin/env python3
"""
Comprehensive test suite for ProductionTrainingExecutor
Tests all methods, edge cases, and integration scenarios
"""

import unittest
import tempfile
import shutil
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the component to test
from production_training_execution import ProductionTrainingExecutor


class TestProductionTrainingExecutorBasic(unittest.TestCase):
    """Test basic functionality of ProductionTrainingExecutor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.executor = ProductionTrainingExecutor(output_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test executor initialization"""
        self.assertIsNotNone(self.executor)
        self.assertEqual(str(self.executor.output_dir), self.test_dir)
        self.assertIsNotNone(self.executor.logger)
        self.assertIsNone(self.executor.brain)
        self.assertIsNone(self.executor.training_manager)
        self.assertIsNone(self.executor.data_loader)
        self.assertIsNone(self.executor.session_id)
        self.assertIsNone(self.executor.deployment_orchestrator)
    
    def test_output_directory_creation(self):
        """Test that output directory is created"""
        test_output_dir = os.path.join(self.test_dir, "test_output")
        executor = ProductionTrainingExecutor(output_dir=test_output_dir)
        self.assertTrue(os.path.exists(test_output_dir))
    
    def test_training_metadata_structure(self):
        """Test training metadata initialization"""
        metadata = self.executor.training_metadata
        self.assertIn("start_time", metadata)
        self.assertIn("end_time", metadata)
        self.assertIn("config", metadata)
        self.assertIn("results", metadata)
        self.assertIn("errors", metadata)
        self.assertIn("gac_metrics", metadata)
        self.assertIn("deployment_info", metadata)
        self.assertIsInstance(metadata["errors"], list)
        self.assertIsInstance(metadata["gac_metrics"], list)
    
    def test_logging_setup(self):
        """Test logging configuration"""
        # Check that log file is created
        log_files = list(Path(self.test_dir).glob("training_execution_*.log"))
        self.assertTrue(len(log_files) > 0, "Log file should be created")


class TestBrainSystemInitialization(unittest.TestCase):
    """Test Brain system initialization"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.executor = ProductionTrainingExecutor(output_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('production_training_execution.Brain')
    @patch('production_training_execution.BrainSystemConfig')
    def test_initialize_brain_system_default_config(self, mock_config, mock_brain):
        """Test Brain system initialization with default config"""
        mock_brain_instance = MagicMock()
        mock_brain_instance.domain_registry._domains = {}
        mock_brain.return_value = mock_brain_instance
        
        self.executor.initialize_brain_system()
        
        # Verify Brain was initialized
        mock_brain.assert_called_once()
        self.assertEqual(self.executor.brain, mock_brain_instance)
        
        # Verify domain was added
        mock_brain_instance.add_domain.assert_called_once_with(
            'fraud_detection',
            unittest.mock.ANY
        )
    
    @patch('production_training_execution.Brain')
    def test_initialize_brain_system_with_config(self, mock_brain):
        """Test Brain system initialization with custom config"""
        mock_brain_instance = MagicMock()
        mock_brain_instance.domain_registry._domains = {'fraud_detection': {}}
        mock_brain.return_value = mock_brain_instance
        
        custom_config = MagicMock()
        self.executor.initialize_brain_system(config=custom_config)
        
        mock_brain.assert_called_once_with(custom_config)
        self.assertEqual(self.executor.brain, mock_brain_instance)
        
        # Should not add domain if it already exists
        mock_brain_instance.add_domain.assert_not_called()
    
    @patch('production_training_execution.Brain')
    def test_initialize_brain_system_error_handling(self, mock_brain):
        """Test Brain system initialization error handling"""
        mock_brain.side_effect = Exception("Brain initialization failed")
        
        with self.assertRaises(Exception):
            self.executor.initialize_brain_system()
        
        # Check error was logged
        self.assertTrue(len(self.executor.training_metadata["errors"]) > 0)
        error = self.executor.training_metadata["errors"][0]
        self.assertEqual(error["stage"], "brain_initialization")


class TestTrainingConfiguration(unittest.TestCase):
    """Test training configuration setup"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.executor = ProductionTrainingExecutor(output_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('production_training_execution.torch')
    @patch('production_training_execution.TrainingConfig')
    def test_setup_training_configuration_default(self, mock_config_class, mock_torch):
        """Test training configuration with defaults"""
        mock_torch.cuda.is_available.return_value = False
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance
        
        config = self.executor.setup_training_configuration()
        
        # Verify TrainingConfig was created with expected defaults
        mock_config_class.assert_called_once()
        call_args = mock_config_class.call_args[1]
        self.assertEqual(call_args['epochs'], 10)
        self.assertEqual(call_args['batch_size'], 32)
        self.assertEqual(call_args['learning_rate'], 0.001)
        self.assertEqual(call_args['device'], 'cpu')
        self.assertTrue(call_args['use_gac_system'])
        
        # Check metadata was updated
        self.assertIsNotNone(self.executor.training_metadata["config"])
    
    @patch('production_training_execution.torch')
    @patch('production_training_execution.TrainingConfig')
    def test_setup_training_configuration_custom(self, mock_config_class, mock_torch):
        """Test training configuration with custom parameters"""
        mock_torch.cuda.is_available.return_value = True
        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance
        
        config = self.executor.setup_training_configuration(
            epochs=20,
            batch_size=64,
            learning_rate=0.0001,
            use_gac_system=False,
            gac_mode="disabled"
        )
        
        # Verify custom parameters were used
        call_args = mock_config_class.call_args[1]
        self.assertEqual(call_args['epochs'], 20)
        self.assertEqual(call_args['batch_size'], 64)
        self.assertEqual(call_args['learning_rate'], 0.0001)
        self.assertEqual(call_args['device'], 'cuda')
        self.assertFalse(call_args['use_gac_system'])
    
    @patch('production_training_execution.TrainingConfig')
    def test_setup_training_configuration_error_handling(self, mock_config_class):
        """Test training configuration error handling"""
        mock_config_class.side_effect = Exception("Config creation failed")
        
        with self.assertRaises(Exception):
            self.executor.setup_training_configuration()
        
        # Check error was logged
        self.assertTrue(len(self.executor.training_metadata["errors"]) > 0)
        error = self.executor.training_metadata["errors"][0]
        self.assertEqual(error["stage"], "config_setup")


class TestDataLoading(unittest.TestCase):
    """Test IEEE data loading functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.executor = ProductionTrainingExecutor(output_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('production_training_execution.IEEEFraudDataLoader')
    @patch('production_training_execution.IEEE_LOADER_AVAILABLE', True)
    def test_load_ieee_training_data_success(self, mock_loader_class):
        """Test successful IEEE data loading"""
        # Mock data loader
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        
        # Mock data
        X_train = np.random.rand(472432, 100)
        y_train = np.random.randint(0, 2, 472432)
        X_val = np.random.rand(118108, 100)
        y_val = np.random.randint(0, 2, 118108)
        
        mock_loader.load_data.return_value = (X_train, y_train, (X_val, y_val))
        mock_loader.get_feature_names.return_value = [f"feature_{i}" for i in range(100)]
        
        # Load data
        result = self.executor.load_ieee_training_data()
        
        # Verify results
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].shape, X_train.shape)
        self.assertEqual(result[1].shape, y_train.shape)
        self.assertIsNotNone(result[2])
        
        # Check metadata
        data_info = self.executor.training_metadata["data_info"]
        self.assertEqual(data_info["training_samples"], 472432)
        self.assertEqual(data_info["features"], 100)
        self.assertEqual(data_info["validation_samples"], 118108)
    
    @patch('production_training_execution.IEEE_LOADER_AVAILABLE', False)
    def test_load_ieee_training_data_loader_not_available(self):
        """Test data loading when loader is not available"""
        with self.assertRaises(ImportError):
            self.executor.load_ieee_training_data()
        
        # Check error was logged
        self.assertTrue(len(self.executor.training_metadata["errors"]) > 0)
        error = self.executor.training_metadata["errors"][0]
        self.assertEqual(error["stage"], "data_loading")
    
    @patch('production_training_execution.os.path.exists')
    @patch('production_training_execution.IEEE_LOADER_AVAILABLE', True)
    def test_load_ieee_training_data_no_directory(self, mock_exists):
        """Test data loading when directory doesn't exist"""
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            self.executor.load_ieee_training_data()


class TestTrainingExecution(unittest.TestCase):
    """Test training execution functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.executor = ProductionTrainingExecutor(output_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_execute_training_session_with_progress(self):
        """Test training session execution with progress monitoring"""
        # Setup mock brain
        mock_brain = MagicMock()
        mock_brain.train_domain_with_progress.return_value = {
            'success': True,
            'final_accuracy': 0.85,
            'training_time': 100.5
        }
        self.executor.brain = mock_brain
        
        # Setup mock data loader
        mock_loader = MagicMock()
        mock_loader.get_feature_names.return_value = ["feature1", "feature2"]
        self.executor.data_loader = mock_loader
        
        # Mock data
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        val_data = (np.random.rand(20, 10), np.random.randint(0, 2, 20))
        
        # Mock config
        mock_config = MagicMock()
        mock_config.__dataclass_fields__ = {}  # Indicate it's a dataclass
        
        # Execute training
        with patch('production_training_execution.asdict', return_value={'epochs': 10}):
            result = self.executor.execute_training_session(
                X_train, y_train, val_data, mock_config
            )
        
        # Verify training was called
        mock_brain.train_domain_with_progress.assert_called_once()
        self.assertEqual(result['success'], True)
        self.assertEqual(result['final_accuracy'], 0.85)
    
    def test_execute_training_session_fallback(self):
        """Test training session with fallback to basic training"""
        # Setup mock brain without enhanced training
        mock_brain = MagicMock()
        del mock_brain.train_domain_with_progress  # Remove enhanced method
        mock_brain.train_domain.return_value = {
            'success': True,
            'final_accuracy': 0.80
        }
        self.executor.brain = mock_brain
        
        # Mock data
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        val_data = None
        
        # Execute training with dict config
        result = self.executor.execute_training_session(
            X_train, y_train, val_data, {'epochs': 5}
        )
        
        # Verify basic training was called
        mock_brain.train_domain.assert_called_once()
        self.assertEqual(result['success'], True)
    
    def test_training_progress_callback(self):
        """Test training progress callback"""
        progress_info = {
            'epoch': 5,
            'metrics': {
                'loss': 0.123,
                'accuracy': 0.89
            },
            'gac_status': {
                'gac_enabled': True,
                'gac_mode': 'adaptive',
                'gac_components': ['clipping', 'monitoring'],
                'gac_metrics': {
                    'gradient_norm': 0.456,
                    'clipping_rate': 0.1
                }
            }
        }
        
        # Call progress callback
        self.executor._training_progress_callback(progress_info)
        
        # Check GAC metrics were stored
        self.assertEqual(len(self.executor.training_metadata["gac_metrics"]), 1)
        gac_record = self.executor.training_metadata["gac_metrics"][0]
        self.assertEqual(gac_record["epoch"], 5)
        self.assertEqual(gac_record["metrics"]["gradient_norm"], 0.456)


class TestResultsSaving(unittest.TestCase):
    """Test results saving functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.executor = ProductionTrainingExecutor(output_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_save_comprehensive_results(self):
        """Test saving comprehensive results"""
        # Setup metadata
        self.executor.training_metadata["start_time"] = datetime.now().isoformat()
        self.executor.training_metadata["config"] = {"epochs": 10}
        self.executor.training_metadata["results"] = {"accuracy": 0.85}
        self.executor.training_metadata["gac_metrics"] = [
            {"epoch": 1, "metrics": {"gradient_norm": 0.5}}
        ]
        
        # Mock brain
        mock_brain = MagicMock()
        self.executor.brain = mock_brain
        
        # Save results
        self.executor.save_comprehensive_results()
        
        # Verify metadata file was created
        metadata_file = Path(self.test_dir) / "training_metadata.json"
        self.assertTrue(metadata_file.exists())
        
        # Verify content
        with open(metadata_file, 'r') as f:
            saved_metadata = json.load(f)
        self.assertIn("duration_seconds", saved_metadata)
        self.assertEqual(saved_metadata["config"]["epochs"], 10)
        
        # Verify analysis report was created
        report_file = Path(self.test_dir) / "training_analysis_report.txt"
        self.assertTrue(report_file.exists())
        
        # Verify GAC analysis was created
        gac_file = Path(self.test_dir) / "gac_performance_analysis.json"
        self.assertTrue(gac_file.exists())
    
    def test_generate_analysis_report(self):
        """Test analysis report generation"""
        # Setup metadata
        self.executor.training_metadata = {
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": 3600,
            "config": {"epochs": 10, "batch_size": 32},
            "data_info": {"training_samples": 1000, "features": 50},
            "results": {"final_accuracy": 0.85},
            "gac_metrics": [{"epoch": 1, "metrics": {"gradient_norm": 0.5}}],
            "errors": []
        }
        
        # Generate report
        self.executor._generate_analysis_report()
        
        # Verify report file exists and has content
        report_file = Path(self.test_dir) / "training_analysis_report.txt"
        self.assertTrue(report_file.exists())
        
        with open(report_file, 'r') as f:
            content = f.read()
        self.assertIn("PRODUCTION TRAINING ANALYSIS REPORT", content)
        self.assertIn("EXECUTIVE SUMMARY", content)
        self.assertIn("CONFIGURATION", content)
        self.assertIn("epochs: 10", content)


class TestDeployment(unittest.TestCase):
    """Test deployment functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.executor = ProductionTrainingExecutor(output_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('production_training_execution.IntegratedDeploymentOrchestrator')
    @patch('production_training_execution.DEPLOYMENT_ORCHESTRATOR_AVAILABLE', True)
    def test_deploy_trained_model_success(self, mock_orchestrator_class):
        """Test successful model deployment"""
        # Setup mock orchestrator
        mock_orchestrator = MagicMock()
        mock_deployment_result = MagicMock()
        mock_deployment_result.status.value = 'COMPLETED'
        mock_deployment_result.deployment_id = 'deploy-123'
        mock_deployment_result.strategy.value = 'blue_green'
        mock_deployment_result.environment = 'production'
        mock_deployment_result.version = 'v1.0'
        mock_deployment_result.start_time = datetime.now()
        mock_deployment_result.end_time = datetime.now()
        
        mock_orchestrator.deploy_complete_system.return_value = mock_deployment_result
        self.executor.deployment_orchestrator = mock_orchestrator
        
        # Create required files
        (Path(self.test_dir) / 'training_metadata.json').touch()
        (Path(self.test_dir) / 'training_analysis_report.txt').touch()
        
        # Deploy
        training_results = {
            'success': True,
            'final_accuracy': 0.85,
            'training_time': 100
        }
        
        result = self.executor.deploy_trained_model(training_results)
        
        # Verify deployment succeeded
        self.assertTrue(result['success'])
        self.assertEqual(result['deployment_id'], 'deploy-123')
        self.assertEqual(result['deployment_strategy'], 'blue_green')
    
    def test_deploy_trained_model_no_orchestrator(self):
        """Test deployment when orchestrator is not available"""
        self.executor.deployment_orchestrator = None
        
        training_results = {'success': True, 'final_accuracy': 0.85}
        result = self.executor.deploy_trained_model(training_results)
        
        self.assertFalse(result['success'])
        self.assertEqual(result['reason'], 'deployment_orchestrator_not_available')
    
    def test_validate_deployment_readiness(self):
        """Test deployment readiness validation"""
        # Setup metadata
        self.executor.training_metadata["errors"] = []
        
        # Create required files
        (Path(self.test_dir) / 'training_metadata.json').touch()
        (Path(self.test_dir) / 'training_analysis_report.txt').touch()
        
        # Test with good results
        training_results = {
            'success': True,
            'final_accuracy': 0.85
        }
        
        validation = self.executor._validate_deployment_readiness(training_results)
        
        self.assertTrue(validation['ready'])
        self.assertEqual(validation['reason'], 'validation_passed')
        self.assertEqual(validation['checks_passed'], 4)
        
        # Test with low accuracy
        training_results['final_accuracy'] = 0.5
        validation = self.executor._validate_deployment_readiness(training_results)
        
        self.assertFalse(validation['ready'])
        self.assertEqual(validation['reason'], 'validation_failed')


class TestProductionPipeline(unittest.TestCase):
    """Test complete production training pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.executor = ProductionTrainingExecutor(output_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('production_training_execution.IEEEFraudDataLoader')
    @patch('production_training_execution.Brain')
    @patch('production_training_execution.TrainingConfig')
    @patch('production_training_execution.IEEE_LOADER_AVAILABLE', True)
    def test_run_production_training_complete(self, mock_config_class, mock_brain_class, mock_loader_class):
        """Test complete production training pipeline"""
        # Setup mocks
        mock_brain = MagicMock()
        mock_brain.domain_registry._domains = {}
        mock_brain.train_domain.return_value = {
            'success': True,
            'final_accuracy': 0.85
        }
        mock_brain_class.return_value = mock_brain
        
        mock_loader = MagicMock()
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        val_data = (np.random.rand(20, 10), np.random.randint(0, 2, 20))
        mock_loader.load_data.return_value = (X_train, y_train, val_data)
        mock_loader.get_feature_names.return_value = ["feature1", "feature2"]
        mock_loader_class.return_value = mock_loader
        
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config
        
        # Mock os.path.exists to return True for data directory
        with patch('production_training_execution.os.path.exists', return_value=True):
            # Run production training
            results = self.executor.run_production_training(
                epochs=5,
                batch_size=16
            )
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertEqual(results['success'], True)
        self.assertEqual(results['final_accuracy'], 0.85)
        
        # Verify all steps were executed
        mock_brain_class.assert_called()
        mock_loader_class.assert_called()
        mock_config_class.assert_called()
        mock_brain.train_domain.assert_called()
        
        # Verify cleanup was called
        mock_brain.cleanup.assert_called()
    
    @patch('production_training_execution.Brain')
    def test_run_production_training_error_handling(self, mock_brain_class):
        """Test error handling in production training"""
        mock_brain_class.side_effect = Exception("Brain initialization failed")
        
        with self.assertRaises(Exception):
            self.executor.run_production_training()
        
        # Verify partial results were attempted to be saved
        # Even on failure, should try to save what we have
        self.assertTrue(len(self.executor.training_metadata["errors"]) > 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.executor = ProductionTrainingExecutor(output_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_missing_data_loader_attributes(self):
        """Test handling of missing data loader attributes"""
        # Execute training without data loader
        X_train = np.random.rand(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        mock_brain = MagicMock()
        mock_brain.train_domain.return_value = {'success': True}
        self.executor.brain = mock_brain
        
        result = self.executor.execute_training_session(
            X_train, y_train, None, {'epochs': 5}
        )
        
        # Should still work without data loader
        self.assertIsNotNone(result)
    
    def test_deployment_validation_with_errors(self):
        """Test deployment validation with critical errors"""
        self.executor.training_metadata["errors"] = [
            {
                "stage": "training_execution",
                "error": "Critical error",
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        training_results = {
            'success': True,
            'final_accuracy': 0.85
        }
        
        validation = self.executor._validate_deployment_readiness(training_results)
        
        # Should not be ready due to critical errors
        self.assertFalse(validation['ready'])
    
    def test_progress_callback_error_handling(self):
        """Test progress callback error handling"""
        # Call with invalid progress info
        self.executor._training_progress_callback({})
        
        # Should not crash
        self.assertEqual(len(self.executor.training_metadata["gac_metrics"]), 0)
        
        # Call with partial info
        self.executor._training_progress_callback({'epoch': 1})
        
        # Should still create a record
        self.assertEqual(len(self.executor.training_metadata["gac_metrics"]), 1)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestProductionTrainingExecutorBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestBrainSystemInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingExecution))
    suite.addTests(loader.loadTestsFromTestCase(TestResultsSaving))
    suite.addTests(loader.loadTestsFromTestCase(TestDeployment))
    suite.addTests(loader.loadTestsFromTestCase(TestProductionPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)