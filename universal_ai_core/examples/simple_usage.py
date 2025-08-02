"""
Simple Universal Core Usage Examples
=====================================

This shows how the Universal Core is SUPPOSED to work:
- Simple, clean core
- External modules hook in easily
- No built-in domain logic
- Minimal, focused API
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universal_ai_core.simple_api import create_simple_api, Module
from typing import Any, Dict
import json


# Example 1: Simple Text Processing Module

class TextCleanerModule(Module):
    """External module for cleaning text."""
    
    def process(self, data: Any) -> Any:
        """Clean text data."""
        if isinstance(data, str):
            # Simple text cleaning
            cleaned = data.strip().lower()
            return cleaned
        elif isinstance(data, dict) and "text" in data:
            data["text"] = data["text"].strip().lower()
            data["cleaned_by"] = self.name
            return data
        else:
            return data


class SentimentModule(Module):
    """External module for sentiment analysis."""
    
    def process(self, data: Any) -> Any:
        """Simple sentiment analysis."""
        if isinstance(data, str):
            text = data
        elif isinstance(data, dict) and "text" in data:
            text = data["text"]
        else:
            return data
        
        # Super simple sentiment (just for demo)
        positive_words = ["good", "great", "awesome", "love", "excellent"]
        negative_words = ["bad", "terrible", "hate", "awful", "horrible"]
        
        sentiment_score = 0
        for word in positive_words:
            if word in text:
                sentiment_score += 1
        for word in negative_words:
            if word in text:
                sentiment_score -= 1
        
        if isinstance(data, dict):
            data["sentiment_score"] = sentiment_score
            data["sentiment"] = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
            data["analyzed_by"] = self.name
            return data
        else:
            return {
                "text": text,
                "sentiment_score": sentiment_score,
                "sentiment": "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
            }


# Example 2: Data Processing Module

class DataValidatorModule(Module):
    """External module for validating data."""
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data."""
        if data is None:
            return False
        if isinstance(data, str) and len(data.strip()) == 0:
            return False
        return True
    
    def process(self, data: Any) -> Any:
        """Validate and mark data."""
        if isinstance(data, dict):
            data["validated"] = True
            data["validated_by"] = self.name
        
        return data


class LoggingModule(Module):
    """External module for logging processing steps."""
    
    def process(self, data: Any) -> Any:
        """Log processing step."""
        print(f"[{self.name}] Processing: {str(data)[:100]}...")
        
        if isinstance(data, dict):
            if "processing_log" not in data:
                data["processing_log"] = []
            data["processing_log"].append({
                "module": self.name,
                "step": len(data["processing_log"]) + 1
            })
        
        return data


# Example 3: Domain-Specific Module (but externally defined)

class MolecularModule(Module):
    """External molecular analysis module."""
    
    def process(self, data: Any) -> Any:
        """Process molecular data."""
        if isinstance(data, dict) and "smiles" in data:
            smiles = data["smiles"]
            
            # Simple molecular weight calculation (just for demo)
            # In reality, this would use RDKit or similar
            molecular_weight = len(smiles) * 12  # Fake calculation
            
            data["molecular_weight"] = molecular_weight
            data["processed_by"] = self.name
            
            return data
        else:
            return data


def demo_simple_text_processing():
    """Demo 1: Simple text processing pipeline."""
    print("=== Demo 1: Simple Text Processing ===")
    
    # Create the core
    api = create_simple_api()
    
    # Create and register external modules
    cleaner = TextCleanerModule("text_cleaner")
    sentiment = SentimentModule("sentiment_analyzer")
    logger_module = LoggingModule("logger")
    
    api.register_module(cleaner)
    api.register_module(sentiment)
    api.register_module(logger_module)
    
    # Create a pipeline
    pipeline = api.create_pipeline("text_analysis")
    pipeline.add_module(logger_module)
    pipeline.add_module(cleaner)
    pipeline.add_module(sentiment)
    
    # Process some text
    test_texts = [
        "This is GREAT and awesome!",
        "This is terrible and awful.",
        "This is just normal text."
    ]
    
    for text in test_texts:
        print(f"\nInput: {text}")
        result = api.process_with_pipeline("text_analysis", {"text": text})
        print(f"Result: {json.dumps(result, indent=2)}")
    
    print(f"\nSystem Status: {api.status()}")


def demo_modular_data_processing():
    """Demo 2: Modular data processing."""
    print("\n=== Demo 2: Modular Data Processing ===")
    
    # Create core
    api = create_simple_api()
    
    # Create modules
    validator = DataValidatorModule("validator")
    logger_module = LoggingModule("logger")
    
    api.register_module(validator)
    api.register_module(logger_module)
    
    # Process individual modules
    test_data = {"input": "some data", "value": 123}
    
    print(f"Original: {test_data}")
    
    # Process with individual modules
    result = api.process_with_module("validator", test_data.copy())
    print(f"After validation: {result}")
    
    result = api.process_with_module("logger", result)
    print(f"After logging: {result}")
    
    # Or create a pipeline
    pipeline = api.create_pipeline("data_processing")
    pipeline.add_module(validator)
    pipeline.add_module(logger_module)
    
    result = api.process_with_pipeline("data_processing", test_data.copy())
    print(f"Pipeline result: {json.dumps(result, indent=2)}")


def demo_domain_specific_modules():
    """Demo 3: Domain-specific modules (externally defined)."""
    print("\n=== Demo 3: Domain-Specific Modules ===")
    
    # Create core
    api = create_simple_api()
    
    # This is how you'd add domain-specific functionality:
    # Create external modules and register them
    
    molecular_module = MolecularModule("molecular_analyzer")
    validator = DataValidatorModule("validator")
    
    api.register_module(molecular_module)
    api.register_module(validator)
    
    # Create domain-specific pipeline
    pipeline = api.create_pipeline("molecular_analysis")
    pipeline.add_module(validator)
    pipeline.add_module(molecular_module)
    
    # Process molecular data
    molecular_data = {
        "smiles": "CCO",  # Ethanol
        "name": "ethanol"
    }
    
    print(f"Input: {molecular_data}")
    result = api.process_with_pipeline("molecular_analysis", molecular_data)
    print(f"Result: {json.dumps(result, indent=2)}")


def demo_event_system():
    """Demo 4: Event system."""
    print("\n=== Demo 4: Event System ===")
    
    # Create core
    api = create_simple_api()
    
    # Register event handlers
    def on_module_registered(event_name, data):
        print(f"📦 Module registered: {data['module_name']}")
    
    def on_module_processed(event_name, data):
        print(f"⚡ Module '{data['module_name']}' processed in {data['processing_time']:.3f}s")
    
    api.on("module_registered", on_module_registered)
    api.on("module_processed", on_module_processed)
    
    # Register a module (will trigger event)
    test_module = TextCleanerModule("test_cleaner")
    api.register_module(test_module)
    
    # Process data (will trigger event)
    result = api.process_with_module("test_cleaner", "  HELLO WORLD  ")
    print(f"Processed result: '{result}'")


def demo_configuration():
    """Demo 5: Configuration system."""
    print("\n=== Demo 5: Configuration ===")
    
    # Create core with config
    config = {
        "debug": True,
        "max_pipeline_steps": 10,
        "custom_setting": "my_value"
    }
    
    api = create_simple_api(config)
    
    print(f"Debug mode: {api.get_config('debug')}")
    print(f"Custom setting: {api.get_config('custom_setting')}")
    print(f"All config: {api.get_config()}")
    
    # Update config
    api.set_config("runtime_setting", "new_value")
    print(f"Updated config: {api.get_config()}")


if __name__ == "__main__":
    # Run all demos
    demo_simple_text_processing()
    demo_modular_data_processing()
    demo_domain_specific_modules()
    demo_event_system()
    demo_configuration()
    
    print("\n🎉 All demos completed!")
    print("\nThis is what a Universal Core should look like:")
    print("✅ Simple, focused API")
    print("✅ External modules hook in easily") 
    print("✅ No built-in domain logic")
    print("✅ Modular and extensible")
    print("✅ Clean separation of concerns")