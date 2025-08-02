#!/usr/bin/env python3
"""
Device Routing Efficiency Test
Tests that training is properly routed to CPU or GPU based on efficiency
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import numpy as np
from typing import Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTestModel(nn.Module):
    """Simple model for testing device routing"""
    def __init__(self, input_size: int = 100, hidden_size: int = 50, output_size: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class DeviceRoutingTester:
    """Test device routing efficiency"""
    
    def __init__(self):
        self.results = {}
        self.device_info = self._get_device_info()
    
    def _get_device_info(self) -> Dict[str, any]:
        """Get comprehensive device information"""
        info = {
            'cpu': {
                'available': True,
                'cores': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'architecture': torch.get_num_threads()
            },
            'cuda': {
                'available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
            }
        }
        
        if torch.cuda.is_available():
            info['cuda'].update({
                'device_name': torch.cuda.get_device_name(),
                'memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'compute_capability': torch.cuda.get_device_capability()
            })
        
        return info
    
    def _generate_test_data(self, batch_size: int = 32, input_size: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic test data"""
        X = torch.randn(batch_size, input_size)
        y = torch.randint(0, 10, (batch_size,))
        return X, y
    
    def _test_device_performance(self, device: str, model: nn.Module, 
                                data_loader: torch.utils.data.DataLoader, 
                                epochs: int = 5) -> Dict[str, float]:
        """Test performance on specific device"""
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Warm up
        model.train()
        for _ in range(3):
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                break
        
        # Performance test
        start_time = time.time()
        total_loss = 0.0
        batch_count = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
            
            total_loss += epoch_loss
            batch_count += epoch_batches
        
        end_time = time.time()
        
        return {
            'total_time': end_time - start_time,
            'avg_loss': total_loss / batch_count,
            'batches_per_second': batch_count / (end_time - start_time),
            'device': device
        }
    
    def _test_cuda_functionality(self) -> Dict[str, any]:
        """Test CUDA functionality comprehensively"""
        cuda_tests = {
            'basic_availability': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'memory_allocated': 0,
            'memory_cached': 0,
            'tensor_creation': False,
            'model_transfer': False,
            'training_step': False,
            'compute_capability': None,
            'architecture_status': 'unknown'
        }
        
        if not torch.cuda.is_available():
            return cuda_tests
        
        try:
            # Get device properties
            cuda_tests['compute_capability'] = torch.cuda.get_device_capability(0)
            major, minor = cuda_tests['compute_capability']
            
            # Determine architecture status
            if major >= 12:
                cuda_tests['architecture_status'] = 'new_architecture'
                logger.info(f"Detected new GPU architecture: sm_{major}{minor}")
            elif major >= 8:
                cuda_tests['architecture_status'] = 'supported'
            else:
                cuda_tests['architecture_status'] = 'legacy'
            
            # Test tensor creation
            test_tensor = torch.randn(100, 100).cuda()
            cuda_tests['tensor_creation'] = True
            
            # Test model transfer
            model = SimpleTestModel().cuda()
            cuda_tests['model_transfer'] = True
            
            # Test training step with architecture-aware handling
            if cuda_tests['architecture_status'] == 'new_architecture':
                try:
                    # Try with smaller operations for new architectures
                    optimizer = optim.Adam(model.parameters())
                    criterion = nn.CrossEntropyLoss()
                    
                    X = torch.randn(16, 100).cuda()  # Smaller batch
                    y = torch.randint(0, 10, (16,)).cuda()
                    
                    optimizer.zero_grad()
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
                    
                    cuda_tests['training_step'] = True
                    logger.info("Training step successful on new architecture")
                    
                except Exception as e:
                    logger.warning(f"Training step failed on new architecture: {e}")
                    cuda_tests['training_step'] = False
            else:
                # Standard test for supported architectures
                optimizer = optim.Adam(model.parameters())
                criterion = nn.CrossEntropyLoss()
                
                X = torch.randn(32, 100).cuda()
                y = torch.randint(0, 10, (32,)).cuda()
                
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                cuda_tests['training_step'] = True
            
            # Memory info
            cuda_tests['memory_allocated'] = torch.cuda.memory_allocated() / (1024**2)  # MB
            cuda_tests['memory_cached'] = torch.cuda.memory_reserved() / (1024**2)  # MB
            
        except Exception as e:
            logger.warning(f"CUDA test failed: {e}")
        
        return cuda_tests
    
    def _get_optimal_device_recommendation(self) -> str:
        """Recommend optimal device based on tests"""
        cuda_tests = self._test_cuda_functionality()
        
        if not cuda_tests['basic_availability']:
            logger.info("CUDA not available, recommending CPU")
            return 'cpu'
        
        # Handle new architecture detection
        if cuda_tests['architecture_status'] == 'new_architecture':
            if cuda_tests['training_step']:
                logger.info("New GPU architecture detected but training step successful - recommending CUDA with fallback")
                return 'cuda'
            else:
                logger.info("New GPU architecture detected but training step failed - recommending CPU")
                return 'cpu'
        
        if not cuda_tests['training_step']:
            logger.warning("CUDA available but training step failed, recommending CPU")
            return 'cpu'
        
        # Check memory requirements
        model_size_mb = 50  # Approximate model size
        if cuda_tests['memory_allocated'] + model_size_mb > 1000:  # 1GB threshold
            logger.info("CUDA memory usage high, recommending CPU")
            return 'cpu'
        
        logger.info("CUDA tests passed, recommending CUDA")
        return 'cuda'
    
    def run_comprehensive_test(self) -> Dict[str, any]:
        """Run comprehensive device routing test"""
        logger.info("Starting comprehensive device routing test...")
        
        # Device information
        logger.info(f"Device Info: {self.device_info}")
        
        # CUDA functionality test
        cuda_tests = self._test_cuda_functionality()
        logger.info(f"CUDA Tests: {cuda_tests}")
        
        # Generate test data
        X, y = self._generate_test_data(batch_size=64, input_size=100)
        dataset = torch.utils.data.TensorDataset(X, y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Test CPU performance
        logger.info("Testing CPU performance...")
        cpu_model = SimpleTestModel()
        cpu_results = self._test_device_performance('cpu', cpu_model, data_loader)
        self.results['cpu'] = cpu_results
        
        # Test CUDA performance if available
        if cuda_tests['training_step']:
            logger.info("Testing CUDA performance...")
            cuda_model = SimpleTestModel()
            cuda_results = self._test_device_performance('cuda', cuda_model, data_loader)
            self.results['cuda'] = cuda_results
        
        # Get optimal device recommendation
        optimal_device = self._get_optimal_device_recommendation()
        
        # Performance comparison
        comparison = self._compare_performance()
        
        # Final recommendation
        final_recommendation = {
            'optimal_device': optimal_device,
            'reasoning': self._get_recommendation_reasoning(comparison, cuda_tests),
            'performance_comparison': comparison,
            'device_info': self.device_info,
            'cuda_tests': cuda_tests
        }
        
        self.results['recommendation'] = final_recommendation
        return self.results
    
    def _compare_performance(self) -> Dict[str, any]:
        """Compare CPU vs CUDA performance"""
        if 'cuda' not in self.results:
            return {'cpu_only': True, 'recommendation': 'cpu'}
        
        cpu_perf = self.results['cpu']
        cuda_perf = self.results['cuda']
        
        # Calculate speedup
        speedup = cpu_perf['total_time'] / cuda_perf['total_time']
        
        # Determine winner
        if speedup > 1.2:  # 20% faster threshold
            winner = 'cuda'
            advantage = f"{speedup:.2f}x faster"
        elif speedup < 0.8:  # 20% slower threshold
            winner = 'cpu'
            advantage = f"{1/speedup:.2f}x faster"
        else:
            winner = 'cpu'  # Prefer CPU for similar performance
            advantage = "similar performance, preferring CPU"
        
        return {
            'cpu_performance': cpu_perf,
            'cuda_performance': cuda_perf,
            'speedup': speedup,
            'winner': winner,
            'advantage': advantage,
            'cpu_only': False
        }
    
    def _get_recommendation_reasoning(self, comparison: Dict, cuda_tests: Dict) -> str:
        """Get reasoning for device recommendation"""
        if comparison.get('cpu_only', False):
            return "CUDA not available or failed tests"
        
        if cuda_tests.get('memory_allocated', 0) > 800:  # 800MB threshold
            return "CUDA memory usage too high"
        
        if comparison['winner'] == 'cuda':
            return f"CUDA is {comparison['advantage']}"
        else:
            return f"CPU is {comparison['advantage']}"
    
    def print_results(self):
        """Print formatted test results"""
        print("\n" + "="*60)
        print("DEVICE ROUTING EFFICIENCY TEST RESULTS")
        print("="*60)
        
        print(f"\nDEVICE INFORMATION:")
        print(f"CPU Cores: {self.device_info['cpu']['cores']}")
        print(f"CPU Memory: {self.device_info['cpu']['memory_gb']:.1f} GB")
        print(f"CUDA Available: {self.device_info['cuda']['available']}")
        
        if self.device_info['cuda']['available']:
            print(f"CUDA Device: {self.device_info['cuda']['device_name']}")
            print(f"CUDA Memory: {self.device_info['cuda']['memory_gb']:.1f} GB")
        
        print(f"\nCUDA FUNCTIONALITY TESTS:")
        cuda_tests = self.results.get('recommendation', {}).get('cuda_tests', {})
        for test, result in cuda_tests.items():
            if test == 'compute_capability' and result:
                print(f"  {test}: sm_{result[0]}{result[1]}")
            elif test == 'architecture_status':
                status_emoji = "🆕" if result == 'new_architecture' else "✅" if result == 'supported' else "⚠️"
                print(f"  {test}: {status_emoji} {result}")
            else:
                print(f"  {test}: {result}")
        
        print(f"\nPERFORMANCE RESULTS:")
        if 'cpu' in self.results:
            cpu = self.results['cpu']
            print(f"CPU - Time: {cpu['total_time']:.2f}s, Batches/sec: {cpu['batches_per_second']:.2f}")
        
        if 'cuda' in self.results:
            cuda = self.results['cuda']
            print(f"CUDA - Time: {cuda['total_time']:.2f}s, Batches/sec: {cuda['batches_per_second']:.2f}")
        
        print(f"\nPERFORMANCE COMPARISON:")
        comparison = self.results.get('recommendation', {}).get('performance_comparison', {})
        if not comparison.get('cpu_only', False):
            print(f"Speedup: {comparison.get('speedup', 0):.2f}x")
            print(f"Winner: {comparison.get('winner', 'unknown')}")
            print(f"Advantage: {comparison.get('advantage', 'unknown')}")
        
        print(f"\nFINAL RECOMMENDATION:")
        recommendation = self.results.get('recommendation', {})
        print(f"Optimal Device: {recommendation.get('optimal_device', 'unknown')}")
        print(f"Reasoning: {recommendation.get('reasoning', 'unknown')}")
        
        print("\n" + "="*60)

def main():
    """Main test execution"""
    print("Starting Device Routing Efficiency Test...")
    
    tester = DeviceRoutingTester()
    results = tester.run_comprehensive_test()
    tester.print_results()
    
    # Return results for potential use
    return results

if __name__ == "__main__":
    main() 