"""
Integration example showing how layer analyzer fits into the compression pipeline
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from layer_analyzer import DenseLayerAnalyzer, CompressionMethod, LayerAnalysisResult


@dataclass
class ModelAnalysisReport:
    """Complete analysis report for a neural network model"""
    model_name: str
    total_parameters: int
    total_layers: int
    analyzed_layers: int
    layer_results: Dict[str, LayerAnalysisResult]
    compression_summary: Dict[CompressionMethod, int]
    estimated_total_compression: float
    
    def print_summary(self):
        """Print analysis summary"""
        print(f"\nModel Analysis Report: {self.model_name}")
        print(f"{'='*60}")
        print(f"Total parameters: {self.total_parameters:,}")
        print(f"Total layers: {self.total_layers}")
        print(f"Analyzed layers: {self.analyzed_layers}")
        
        print(f"\nCompression Method Distribution:")
        for method, count in self.compression_summary.items():
            percentage = (count / self.analyzed_layers * 100) if self.analyzed_layers > 0 else 0
            print(f"  {method.value}: {count} layers ({percentage:.1f}%)")
        
        print(f"\nEstimated total compression: {self.estimated_total_compression:.2f}x")
        
        print(f"\nPer-Layer Recommendations:")
        for name, result in self.layer_results.items():
            rec = result.compression_recommendation
            print(f"  {name}: {rec.method.value} (confidence: {rec.confidence:.2f}, ratio: {rec.estimated_compression_ratio:.2f}x)")


class ModelCompressionAnalyzer:
    """Analyzes entire neural network models for compression"""
    
    def __init__(self, analyzer_config: Optional[Dict[str, Any]] = None):
        """Initialize model analyzer"""
        self.layer_analyzer = DenseLayerAnalyzer(analyzer_config)
    
    def analyze_model(self, model: nn.Module, model_name: str = "unnamed") -> ModelAnalysisReport:
        """
        Analyze all linear layers in a model
        
        Args:
            model: PyTorch model to analyze
            model_name: Name for the report
            
        Returns:
            ModelAnalysisReport with complete analysis
        """
        layer_results = {}
        compression_summary = {method: 0 for method in CompressionMethod}
        total_parameters = sum(p.numel() for p in model.parameters())
        total_compressible_params = 0
        weighted_compression_sum = 0.0
        
        # Find all linear layers
        linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((name, module))
        
        # Analyze each layer
        for layer_name, layer in linear_layers:
            try:
                result = self.layer_analyzer.analyze_layer(layer, layer_name)
                layer_results[layer_name] = result
                
                # Update summary
                method = result.compression_recommendation.method
                compression_summary[method] += 1
                
                # Track compression ratios weighted by parameter count
                if method != CompressionMethod.NONE:
                    params = result.parameter_count
                    ratio = result.compression_recommendation.estimated_compression_ratio
                    total_compressible_params += params
                    weighted_compression_sum += params * ratio
                
            except Exception as e:
                print(f"Warning: Failed to analyze layer {layer_name}: {e}")
        
        # Calculate overall compression estimate
        if total_compressible_params > 0:
            avg_compression = weighted_compression_sum / total_compressible_params
            # Account for non-linear layers (assume no compression)
            non_linear_params = total_parameters - total_compressible_params
            if total_parameters > 0:
                estimated_total_compression = (
                    (total_compressible_params * avg_compression + non_linear_params) / 
                    total_parameters
                )
            else:
                estimated_total_compression = 1.0
        else:
            estimated_total_compression = 1.0
        
        return ModelAnalysisReport(
            model_name=model_name,
            total_parameters=total_parameters,
            total_layers=len(list(model.modules())),
            analyzed_layers=len(linear_layers),
            layer_results=layer_results,
            compression_summary=compression_summary,
            estimated_total_compression=1.0 / estimated_total_compression if estimated_total_compression > 0 else 1.0
        )


def create_example_model():
    """Create an example model for testing"""
    
    class ExampleNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Low-rank layer
            self.fc1 = nn.Linear(784, 128)
            with torch.no_grad():
                U = torch.randn(128, 10)
                V = torch.randn(10, 784) 
                self.fc1.weight.data = U @ V
            
            # Dense layer
            self.fc2 = nn.Linear(128, 64)
            
            # Sparse output layer
            self.fc3 = nn.Linear(64, 10)
            with torch.no_grad():
                self.fc3.weight.data = torch.randn(10, 64) * 0.1
                mask = torch.rand(10, 64) > 0.3
                self.fc3.weight.data[mask] = 0
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    return ExampleNet()


def integration_with_compression_pipeline(model: nn.Module, analyzer: ModelCompressionAnalyzer):
    """
    Example of how the analyzer integrates with compression pipeline
    
    In a real system, this would:
    1. Analyze the model
    2. Route each layer to appropriate compressor based on recommendation
    3. Coordinate compression with training if needed
    """
    
    print("\n" + "="*80)
    print("COMPRESSION PIPELINE INTEGRATION EXAMPLE")
    print("="*80)
    
    # Step 1: Analyze model
    report = analyzer.analyze_model(model, "ExampleNet")
    report.print_summary()
    
    # Step 2: Demonstrate routing logic
    print(f"\n{'='*60}")
    print("Compression Routing Plan:")
    print(f"{'='*60}")
    
    from collections import defaultdict
    routing_plan = defaultdict(list)
    
    for layer_name, result in report.layer_results.items():
        method = result.compression_recommendation.method
        routing_plan[method].append({
            'layer_name': layer_name,
            'confidence': result.compression_recommendation.confidence,
            'estimated_ratio': result.compression_recommendation.estimated_compression_ratio
        })
    
    # Display routing plan
    for method, layers in routing_plan.items():
        if layers:
            print(f"\n{method.value.upper()} Compression:")
            for layer_info in layers:
                print(f"  - {layer_info['layer_name']}: "
                      f"confidence={layer_info['confidence']:.2f}, "
                      f"ratio={layer_info['estimated_ratio']:.2f}x")
    
    # Step 3: Show how this would feed into existing systems
    print(f"\n{'='*60}")
    print("Integration Points:")
    print(f"{'='*60}")
    print("1. Tropical layers -> TropicalMathematicalOperations")
    print("2. P-adic layers -> HybridPadicCompressionSystem") 
    print("3. Hybrid layers -> DynamicSwitchingManager")
    print("4. Results -> TrainingCompressionCoordinator")
    
    return report


def main():
    """Run integration example"""
    
    # Initialize analyzer
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'energy_threshold': 0.99,
        'numerical_tolerance': 1e-6,
        'near_zero_threshold': 1e-8
    }
    
    analyzer = ModelCompressionAnalyzer(config)
    
    # Create example model
    model = create_example_model()
    
    # Run integration example
    report = integration_with_compression_pipeline(model, analyzer)
    
    print("\n" + "="*80)
    print("INTEGRATION EXAMPLE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()