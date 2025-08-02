#!/usr/bin/env python3
"""
Custom Plugins Examples for Universal AI Core
==============================================

This module demonstrates how to create and use custom plugins with the Universal AI Core system.
Adapted from Saraphis plugin patterns, these examples show how to implement, register, and
manage custom plugins for extending system capabilities.

Examples include:
- Custom plugin implementation patterns
- Plugin registration and lifecycle management
- Inter-plugin communication
- Plugin configuration and validation
- Dynamic plugin loading and unloading
- Plugin testing and debugging
"""

import logging
import time
import json
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading
import importlib.util

# Universal AI Core imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.plugin_manager import PluginManager
from plugins.base import (
    PluginInterface, FeatureExtractorPlugin, ModelPlugin, 
    ProofLanguagePlugin, KnowledgeBasePlugin
)

logger = logging.getLogger(__name__)


@dataclass
class CustomPluginConfig:
    """Configuration for custom plugins"""
    plugin_name: str = ""
    plugin_version: str = "1.0.0"
    plugin_type: str = ""
    enabled: bool = True
    dependencies: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    hooks: List[str] = field(default_factory=list)
    priority: int = 100
    timeout_seconds: int = 30


class CustomTextAnalyzerPlugin(FeatureExtractorPlugin):
    """
    Custom text analysis plugin example.
    Demonstrates feature extraction plugin implementation
    adapted from Saraphis molecular descriptor patterns.
    """
    
    def __init__(self, config: Optional[CustomPluginConfig] = None):
        super().__init__(config)
        self.config = config or CustomPluginConfig(
            plugin_name="custom_text_analyzer",
            plugin_type="feature_extractors"
        )
        self.word_embeddings = {}
        self.feature_cache = {}
        self._setup_analyzer()
    
    def _setup_analyzer(self):
        """Setup text analysis components"""
        try:
            # Simulate loading word embeddings
            self.word_embeddings = {
                'hello': [0.1, 0.2, 0.3],
                'world': [0.4, 0.5, 0.6],
                'test': [0.7, 0.8, 0.9],
                'example': [0.2, 0.4, 0.6]
            }
            logger.info("Text analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize text analyzer: {e}")
    
    def extract_features(self, data: Dict[str, Any]) -> 'FeatureExtractionResult':
        """Extract features from text data"""
        from plugins.base import FeatureExtractionResult, OperationStatus, FeatureType
        
        try:
            start_time = time.time()
            
            # Get text from input data
            text = data.get('text', '')
            if not text:
                return FeatureExtractionResult(
                    status=OperationStatus.ERROR,
                    error_message="No text provided in input data"
                )
            
            # Check cache
            cache_key = hash(text)
            if cache_key in self.feature_cache:
                cached_result = self.feature_cache[cache_key]
                logger.info(f"Features retrieved from cache for text: {text[:50]}...")
                return cached_result
            
            # Extract various text features
            features = {}
            
            # Basic text statistics
            words = text.lower().split()
            features.update({
                'text_length': len(text),
                'word_count': len(words),
                'unique_words': len(set(words)),
                'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
                'sentence_count': len([s for s in text.split('.') if s.strip()]),
                'char_count': len(text.replace(' ', ''))
            })
            
            # Word frequency features
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Most common words
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            for i, (word, freq) in enumerate(sorted_words[:10]):
                features[f'top_word_{i+1}_freq'] = freq
            
            # Embedding-based features
            word_vectors = []
            for word in words:
                if word in self.word_embeddings:
                    word_vectors.append(self.word_embeddings[word])
            
            if word_vectors:
                # Average word embedding
                import numpy as np
                avg_embedding = np.mean(word_vectors, axis=0)
                for i, val in enumerate(avg_embedding):
                    features[f'embedding_dim_{i+1}'] = val
            
            # Linguistic features
            features.update({
                'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
                'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
                'punctuation_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0,
                'vowel_ratio': sum(1 for c in text.lower() if c in 'aeiou') / len(text) if text else 0
            })
            
            processing_time = time.time() - start_time
            
            # Create result
            result = FeatureExtractionResult(
                features=features,
                feature_names=list(features.keys()),
                feature_types={name: FeatureType.NUMERICAL for name in features.keys()},
                status=OperationStatus.SUCCESS,
                processing_time=processing_time,
                metadata={
                    'plugin_name': self.config.plugin_name,
                    'plugin_version': self.config.plugin_version,
                    'text_preview': text[:100] + "..." if len(text) > 100 else text,
                    'feature_count': len(features)
                }
            )
            
            # Cache result
            self.feature_cache[cache_key] = result
            
            logger.info(f"Extracted {len(features)} features from text in {processing_time:.4f}s")
            return result
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return FeatureExtractionResult(
                status=OperationStatus.ERROR,
                error_message=str(e)
            )
    
    def get_metadata(self) -> 'PluginMetadata':
        """Get plugin metadata"""
        from plugins.base import PluginMetadata
        
        return PluginMetadata(
            name=self.config.plugin_name,
            version=self.config.plugin_version,
            description="Custom text analysis feature extractor with linguistic and embedding features",
            author="Universal AI Core Examples",
            plugin_type="feature_extractors",
            dependencies=["numpy"],
            supported_formats=["text"],
            configuration_schema={
                "enable_embeddings": {"type": "boolean", "default": True},
                "max_text_length": {"type": "integer", "default": 10000},
                "cache_enabled": {"type": "boolean", "default": True}
            }
        )


class CustomSentimentModelPlugin(ModelPlugin):
    """
    Custom sentiment analysis model plugin example.
    Demonstrates model plugin implementation adapted from Saraphis model patterns.
    """
    
    def __init__(self, config: Optional[CustomPluginConfig] = None):
        super().__init__(config)
        self.config = config or CustomPluginConfig(
            plugin_name="custom_sentiment_model",
            plugin_type="models"
        )
        self.model = None
        self.is_trained = False
        self.sentiment_weights = {}
        self._setup_model()
    
    def _setup_model(self):
        """Setup sentiment analysis model"""
        try:
            # Initialize simple rule-based sentiment model
            self.sentiment_weights = {
                'positive_words': ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best'],
                'negative_words': ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'disappointing'],
                'neutral_words': ['okay', 'fine', 'normal', 'average', 'standard']
            }
            self.is_trained = True
            logger.info("Sentiment model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize sentiment model: {e}")
    
    def train_model(self, training_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> 'TrainingResult':
        """Train sentiment analysis model"""
        from plugins.base import TrainingResult, OperationStatus
        
        try:
            start_time = time.time()
            
            # Extract training data
            texts = training_data.get('texts', [])
            labels = training_data.get('labels', [])
            
            if not texts or not labels:
                return TrainingResult(
                    status=OperationStatus.ERROR,
                    error_message="Training data must contain 'texts' and 'labels'"
                )
            
            logger.info(f"Training sentiment model on {len(texts)} samples")
            
            # Simulate training by updating sentiment weights
            # In a real implementation, this would train an actual model
            positive_samples = [texts[i] for i, label in enumerate(labels) if label > 0.5]
            negative_samples = [texts[i] for i, label in enumerate(labels) if label < 0.5]
            
            # Update word weights based on training data
            for text in positive_samples:
                words = text.lower().split()
                for word in words:
                    if word not in self.sentiment_weights['positive_words']:
                        # Simple heuristic: add words that appear frequently in positive samples
                        pass
            
            training_time = time.time() - start_time
            
            # Calculate mock metrics
            mock_metrics = {
                'accuracy': 0.85 + (hash(str(texts)) % 100) / 1000,  # Deterministic but varied
                'precision': 0.82 + (hash(str(labels)) % 100) / 1000,
                'recall': 0.88 + (hash(str(texts + labels)) % 100) / 1000,
                'f1_score': 0.85 + (hash(str(texts[:5])) % 100) / 1000
            }
            
            self.is_trained = True
            
            return TrainingResult(
                status=OperationStatus.SUCCESS,
                model_id=f"sentiment_model_{int(time.time())}",
                metrics=mock_metrics,
                training_time=training_time,
                metadata={
                    'plugin_name': self.config.plugin_name,
                    'training_samples': len(texts),
                    'positive_samples': len(positive_samples),
                    'negative_samples': len(negative_samples)
                }
            )
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return TrainingResult(
                status=OperationStatus.ERROR,
                error_message=str(e)
            )
    
    def predict(self, data: Dict[str, Any]) -> 'PredictionResult':
        """Make sentiment predictions"""
        from plugins.base import PredictionResult, OperationStatus
        
        try:
            if not self.is_trained:
                return PredictionResult(
                    status=OperationStatus.ERROR,
                    error_message="Model must be trained before making predictions"
                )
            
            text = data.get('text', '')
            if not text:
                return PredictionResult(
                    status=OperationStatus.ERROR,
                    error_message="No text provided for prediction"
                )
            
            start_time = time.time()
            
            # Simple rule-based sentiment analysis
            words = text.lower().split()
            positive_score = sum(1 for word in words if word in self.sentiment_weights['positive_words'])
            negative_score = sum(1 for word in words if word in self.sentiment_weights['negative_words'])
            
            total_words = len(words)
            if total_words == 0:
                sentiment_score = 0.5
            else:
                sentiment_score = (positive_score - negative_score + total_words) / (2 * total_words)
                sentiment_score = max(0, min(1, sentiment_score))  # Clamp to [0, 1]
            
            # Calculate confidence based on the strength of sentiment words
            confidence = min(1.0, (positive_score + negative_score) / max(1, total_words) * 3)
            
            prediction_time = time.time() - start_time
            
            return PredictionResult(
                prediction=sentiment_score,
                confidence=confidence,
                status=OperationStatus.SUCCESS,
                prediction_time=prediction_time,
                metadata={
                    'plugin_name': self.config.plugin_name,
                    'positive_words_found': positive_score,
                    'negative_words_found': negative_score,
                    'total_words': total_words,
                    'sentiment_label': 'positive' if sentiment_score > 0.6 else 'negative' if sentiment_score < 0.4 else 'neutral'
                }
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return PredictionResult(
                status=OperationStatus.ERROR,
                error_message=str(e)
            )
    
    def get_metadata(self) -> 'PluginMetadata':
        """Get plugin metadata"""
        from plugins.base import PluginMetadata
        
        return PluginMetadata(
            name=self.config.plugin_name,
            version=self.config.plugin_version,
            description="Custom sentiment analysis model using rule-based approach",
            author="Universal AI Core Examples",
            plugin_type="models",
            dependencies=[],
            supported_formats=["text"],
            configuration_schema={
                "sentiment_threshold": {"type": "float", "default": 0.5},
                "confidence_boost": {"type": "float", "default": 1.0}
            }
        )


class CustomValidationProofPlugin(ProofLanguagePlugin):
    """
    Custom validation proof language plugin example.
    Demonstrates proof system implementation adapted from Saraphis proof patterns.
    """
    
    def __init__(self, config: Optional[CustomPluginConfig] = None):
        super().__init__(config)
        self.config = config or CustomPluginConfig(
            plugin_name="custom_validation_proof",
            plugin_type="proof_languages"
        )
        self.validation_rules = {}
        self.proof_cache = {}
        self._setup_rules()
    
    def _setup_rules(self):
        """Setup validation rules"""
        try:
            self.validation_rules = {
                'text_length': {
                    'min_length': 10,
                    'max_length': 10000,
                    'description': 'Text must be between 10 and 10000 characters'
                },
                'sentiment_consistency': {
                    'min_confidence': 0.7,
                    'description': 'Sentiment prediction must have confidence >= 0.7'
                },
                'word_frequency': {
                    'max_repetition_ratio': 0.3,
                    'description': 'No word should appear more than 30% of the time'
                },
                'content_quality': {
                    'min_unique_words': 5,
                    'max_special_chars_ratio': 0.2,
                    'description': 'Content must have quality indicators'
                }
            }
            logger.info("Validation rules initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize validation rules: {e}")
    
    def construct_proof(self, context: Dict[str, Any]) -> 'ProofResult':
        """Construct validation proof"""
        from plugins.base import ProofResult, OperationStatus
        
        try:
            start_time = time.time()
            
            # Get data to validate
            text = context.get('text', '')
            sentiment_result = context.get('sentiment_result', {})
            rules_to_check = context.get('rules', list(self.validation_rules.keys()))
            
            if not text:
                return ProofResult(
                    status=OperationStatus.ERROR,
                    error_message="No text provided for validation"
                )
            
            # Check cache
            cache_key = hash(f"{text}_{str(sorted(rules_to_check))}")
            if cache_key in self.proof_cache:
                logger.info("Proof retrieved from cache")
                return self.proof_cache[cache_key]
            
            # Validate against each rule
            proof_steps = []
            violations = []
            all_valid = True
            
            for rule_name in rules_to_check:
                if rule_name not in self.validation_rules:
                    continue
                
                rule = self.validation_rules[rule_name]
                step_result = self._validate_rule(rule_name, rule, text, sentiment_result)
                
                proof_steps.append({
                    'rule': rule_name,
                    'description': rule['description'],
                    'result': step_result['valid'],
                    'details': step_result['details']
                })
                
                if not step_result['valid']:
                    violations.append({
                        'rule': rule_name,
                        'violation': step_result['violation']
                    })
                    all_valid = False
            
            # Calculate overall confidence
            valid_rules = sum(1 for step in proof_steps if step['result'])
            confidence = valid_rules / len(proof_steps) if proof_steps else 0.0
            
            construction_time = time.time() - start_time
            
            # Create proof result
            result = ProofResult(
                proof_valid=all_valid,
                confidence=confidence,
                status=OperationStatus.SUCCESS,
                proof_steps=proof_steps,
                construction_time=construction_time,
                metadata={
                    'plugin_name': self.config.plugin_name,
                    'rules_checked': len(proof_steps),
                    'violations_found': len(violations),
                    'violations': violations,
                    'text_preview': text[:100] + "..." if len(text) > 100 else text
                }
            )
            
            # Cache result
            self.proof_cache[cache_key] = result
            
            logger.info(f"Validation proof completed: {valid_rules}/{len(proof_steps)} rules passed")
            return result
            
        except Exception as e:
            logger.error(f"Proof construction failed: {e}")
            return ProofResult(
                status=OperationStatus.ERROR,
                error_message=str(e)
            )
    
    def _validate_rule(self, rule_name: str, rule: Dict[str, Any], 
                      text: str, sentiment_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a specific rule"""
        try:
            if rule_name == 'text_length':
                text_len = len(text)
                valid = rule['min_length'] <= text_len <= rule['max_length']
                return {
                    'valid': valid,
                    'details': f"Text length: {text_len}",
                    'violation': f"Text length {text_len} outside range [{rule['min_length']}, {rule['max_length']}]" if not valid else None
                }
            
            elif rule_name == 'sentiment_consistency':
                confidence = sentiment_result.get('confidence', 0.0)
                valid = confidence >= rule['min_confidence']
                return {
                    'valid': valid,
                    'details': f"Sentiment confidence: {confidence:.3f}",
                    'violation': f"Sentiment confidence {confidence:.3f} below threshold {rule['min_confidence']}" if not valid else None
                }
            
            elif rule_name == 'word_frequency':
                words = text.lower().split()
                if not words:
                    return {'valid': True, 'details': "No words to check", 'violation': None}
                
                word_counts = {}
                for word in words:
                    word_counts[word] = word_counts.get(word, 0) + 1
                
                max_freq = max(word_counts.values())
                max_ratio = max_freq / len(words)
                valid = max_ratio <= rule['max_repetition_ratio']
                
                return {
                    'valid': valid,
                    'details': f"Max word repetition ratio: {max_ratio:.3f}",
                    'violation': f"Word repetition ratio {max_ratio:.3f} exceeds limit {rule['max_repetition_ratio']}" if not valid else None
                }
            
            elif rule_name == 'content_quality':
                words = text.lower().split()
                unique_words = len(set(words))
                special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
                special_ratio = special_chars / len(text) if text else 0
                
                valid_unique = unique_words >= rule['min_unique_words']
                valid_special = special_ratio <= rule['max_special_chars_ratio']
                valid = valid_unique and valid_special
                
                violations = []
                if not valid_unique:
                    violations.append(f"Only {unique_words} unique words, need >= {rule['min_unique_words']}")
                if not valid_special:
                    violations.append(f"Special characters ratio {special_ratio:.3f} exceeds {rule['max_special_chars_ratio']}")
                
                return {
                    'valid': valid,
                    'details': f"Unique words: {unique_words}, Special chars ratio: {special_ratio:.3f}",
                    'violation': "; ".join(violations) if violations else None
                }
            
            else:
                return {
                    'valid': True,
                    'details': "Unknown rule",
                    'violation': None
                }
                
        except Exception as e:
            return {
                'valid': False,
                'details': f"Rule validation error: {str(e)}",
                'violation': f"Failed to validate rule {rule_name}: {str(e)}"
            }
    
    def get_metadata(self) -> 'PluginMetadata':
        """Get plugin metadata"""
        from plugins.base import PluginMetadata
        
        return PluginMetadata(
            name=self.config.plugin_name,
            version=self.config.plugin_version,
            description="Custom validation proof language for text quality and consistency checks",
            author="Universal AI Core Examples",
            plugin_type="proof_languages",
            dependencies=[],
            supported_formats=["text"],
            configuration_schema={
                "validation_rules": {"type": "object"},
                "cache_enabled": {"type": "boolean", "default": True}
            }
        )


class PluginOrchestrator:
    """
    Plugin orchestrator for managing complex workflows.
    Demonstrates plugin coordination patterns adapted from Saraphis.
    """
    
    def __init__(self, plugin_manager: PluginManager):
        self.plugin_manager = plugin_manager
        self.workflows = {}
        self.execution_history = []
    
    def register_workflow(self, workflow_name: str, workflow_steps: List[Dict[str, Any]]):
        """Register a multi-plugin workflow"""
        self.workflows[workflow_name] = workflow_steps
        logger.info(f"Registered workflow '{workflow_name}' with {len(workflow_steps)} steps")
    
    def execute_workflow(self, workflow_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a multi-plugin workflow"""
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        
        workflow_steps = self.workflows[workflow_name]
        logger.info(f"Executing workflow '{workflow_name}' with {len(workflow_steps)} steps")
        
        start_time = time.time()
        current_data = input_data.copy()
        step_results = []
        
        for i, step in enumerate(workflow_steps):
            step_start = time.time()
            
            try:
                plugin_type = step['plugin_type']
                plugin_name = step['plugin_name']
                operation = step['operation']
                step_config = step.get('config', {})
                
                logger.info(f"Step {i+1}: {plugin_type}.{plugin_name}.{operation}")
                
                # Get plugin
                plugin = self.plugin_manager.get_plugin(plugin_type, plugin_name)
                if not plugin:
                    raise ValueError(f"Plugin {plugin_type}.{plugin_name} not found")
                
                # Execute operation
                if operation == 'extract_features':
                    result = plugin.extract_features(current_data)
                    if result.status.name == 'SUCCESS':
                        current_data['features'] = result.features
                        current_data['feature_metadata'] = result.metadata
                    else:
                        raise Exception(f"Feature extraction failed: {result.error_message}")
                
                elif operation == 'predict':
                    result = plugin.predict(current_data)
                    if result.status.name == 'SUCCESS':
                        current_data['prediction'] = result.prediction
                        current_data['confidence'] = result.confidence
                        current_data['prediction_metadata'] = result.metadata
                    else:
                        raise Exception(f"Prediction failed: {result.error_message}")
                
                elif operation == 'construct_proof':
                    result = plugin.construct_proof(current_data)
                    if result.status.name == 'SUCCESS':
                        current_data['proof_valid'] = result.proof_valid
                        current_data['proof_confidence'] = result.confidence
                        current_data['proof_steps'] = result.proof_steps
                        current_data['proof_metadata'] = result.metadata
                    else:
                        raise Exception(f"Proof construction failed: {result.error_message}")
                
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                
                step_time = time.time() - step_start
                step_results.append({
                    'step': i + 1,
                    'plugin': f"{plugin_type}.{plugin_name}",
                    'operation': operation,
                    'success': True,
                    'execution_time': step_time
                })
                
                logger.info(f"Step {i+1} completed in {step_time:.4f}s")
                
            except Exception as e:
                step_time = time.time() - step_start
                error_msg = f"Step {i+1} failed: {str(e)}"
                logger.error(error_msg)
                
                step_results.append({
                    'step': i + 1,
                    'plugin': f"{plugin_type}.{plugin_name}",
                    'operation': operation,
                    'success': False,
                    'error': str(e),
                    'execution_time': step_time
                })
                
                # Stop workflow on error
                break
        
        total_time = time.time() - start_time
        
        # Record execution
        execution_record = {
            'workflow_name': workflow_name,
            'start_time': start_time,
            'total_time': total_time,
            'steps': step_results,
            'final_data': current_data,
            'success': all(step['success'] for step in step_results)
        }
        
        self.execution_history.append(execution_record)
        
        logger.info(f"Workflow '{workflow_name}' completed in {total_time:.2f}s")
        return current_data


class CustomPluginsExample:
    """
    Comprehensive examples for custom plugins.
    Demonstrates patterns adapted from Saraphis plugin architecture.
    """
    
    def __init__(self):
        self.plugin_manager = PluginManager()
        self.orchestrator = PluginOrchestrator(self.plugin_manager)
        self.custom_plugins = {}
        self.results = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
    
    def example_1_create_custom_plugins(self):
        """Example 1: Create and Register Custom Plugins"""
        logger.info("=" * 60)
        logger.info("Example 1: Create and Register Custom Plugins")
        logger.info("=" * 60)
        
        try:
            # Create custom plugins
            text_analyzer = CustomTextAnalyzerPlugin()
            sentiment_model = CustomSentimentModelPlugin()
            validation_proof = CustomValidationProofPlugin()
            
            # Register plugins with plugin manager
            plugins_to_register = [
                ('feature_extractors', 'text_analyzer', text_analyzer),
                ('models', 'sentiment_model', sentiment_model),
                ('proof_languages', 'validation_proof', validation_proof)
            ]
            
            for plugin_type, plugin_name, plugin_instance in plugins_to_register:
                success = self.plugin_manager.register_plugin(plugin_type, plugin_name, plugin_instance)
                if success:
                    logger.info(f"✓ Registered {plugin_type}.{plugin_name}")
                    self.custom_plugins[f"{plugin_type}.{plugin_name}"] = plugin_instance
                else:
                    logger.error(f"✗ Failed to register {plugin_type}.{plugin_name}")
            
            # Verify plugin registration
            available_plugins = self.plugin_manager.list_available_plugins()
            logger.info(f"Total available plugins: {sum(len(plugins) for plugins in available_plugins.values())}")
            
            for plugin_type, plugins in available_plugins.items():
                logger.info(f"  {plugin_type}: {[p['name'] for p in plugins]}")
            
            # Get plugin metadata
            for plugin_key, plugin in self.custom_plugins.items():
                metadata = plugin.get_metadata()
                logger.info(f"Plugin {plugin_key} metadata:")
                logger.info(f"  Description: {metadata.description}")
                logger.info(f"  Version: {metadata.version}")
                logger.info(f"  Dependencies: {metadata.dependencies}")
            
            self.results['plugin_registration'] = {
                'registered_plugins': len(self.custom_plugins),
                'plugin_types': list(available_plugins.keys()),
                'custom_plugins': list(self.custom_plugins.keys())
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Plugin creation example failed: {e}")
            return False
    
    def example_2_test_individual_plugins(self):
        """Example 2: Test Individual Plugin Functionality"""
        logger.info("=" * 60)
        logger.info("Example 2: Test Individual Plugin Functionality")
        logger.info("=" * 60)
        
        try:
            # Test data
            test_texts = [
                "This is an excellent example of great text analysis!",
                "This terrible example shows awful text processing.",
                "This is a normal, average piece of text for testing.",
                "Amazing wonderful fantastic great excellent brilliant superb!",
                "Bad horrible awful terrible disgusting disappointing worst!"
            ]
            
            plugin_results = {}
            
            # Test text analyzer plugin
            logger.info("Testing Text Analyzer Plugin")
            text_analyzer = self.custom_plugins.get('feature_extractors.text_analyzer')
            if text_analyzer:
                for i, text in enumerate(test_texts):
                    result = text_analyzer.extract_features({'text': text})
                    if result.status.name == 'SUCCESS':
                        logger.info(f"  Text {i+1}: {len(result.features)} features extracted")
                        logger.info(f"    Features: {list(result.features.keys())[:5]}...")
                        plugin_results[f'text_analyzer_test_{i+1}'] = {
                            'success': True,
                            'feature_count': len(result.features),
                            'processing_time': result.processing_time
                        }
                    else:
                        logger.error(f"  Text {i+1}: Feature extraction failed")
                        plugin_results[f'text_analyzer_test_{i+1}'] = {'success': False}
            
            # Test sentiment model plugin
            logger.info("Testing Sentiment Model Plugin")
            sentiment_model = self.custom_plugins.get('models.sentiment_model')
            if sentiment_model:
                # Train model first
                training_data = {
                    'texts': test_texts,
                    'labels': [0.9, 0.1, 0.5, 0.95, 0.05]  # Manual sentiment labels
                }
                
                training_result = sentiment_model.train_model(training_data)
                if training_result.status.name == 'SUCCESS':
                    logger.info(f"  Model trained successfully in {training_result.training_time:.4f}s")
                    logger.info(f"  Training metrics: {training_result.metrics}")
                    
                    # Test predictions
                    for i, text in enumerate(test_texts):
                        pred_result = sentiment_model.predict({'text': text})
                        if pred_result.status.name == 'SUCCESS':
                            sentiment = pred_result.prediction
                            confidence = pred_result.confidence
                            logger.info(f"  Text {i+1}: Sentiment = {sentiment:.3f}, Confidence = {confidence:.3f}")
                            plugin_results[f'sentiment_model_test_{i+1}'] = {
                                'success': True,
                                'sentiment': sentiment,
                                'confidence': confidence
                            }
                        else:
                            logger.error(f"  Text {i+1}: Prediction failed")
                            plugin_results[f'sentiment_model_test_{i+1}'] = {'success': False}
                else:
                    logger.error("  Model training failed")
            
            # Test validation proof plugin
            logger.info("Testing Validation Proof Plugin")
            validation_proof = self.custom_plugins.get('proof_languages.validation_proof')
            if validation_proof:
                for i, text in enumerate(test_texts):
                    # Create context with sentiment result
                    sentiment_result = plugin_results.get(f'sentiment_model_test_{i+1}', {})
                    context = {
                        'text': text,
                        'sentiment_result': sentiment_result,
                        'rules': ['text_length', 'content_quality']
                    }
                    
                    proof_result = validation_proof.construct_proof(context)
                    if proof_result.status.name == 'SUCCESS':
                        valid = proof_result.proof_valid
                        confidence = proof_result.confidence
                        logger.info(f"  Text {i+1}: Valid = {valid}, Confidence = {confidence:.3f}")
                        if hasattr(proof_result, 'metadata') and 'violations' in proof_result.metadata:
                            violations = proof_result.metadata['violations']
                            if violations:
                                logger.info(f"    Violations: {[v['rule'] for v in violations]}")
                        plugin_results[f'validation_proof_test_{i+1}'] = {
                            'success': True,
                            'valid': valid,
                            'confidence': confidence
                        }
                    else:
                        logger.error(f"  Text {i+1}: Proof construction failed")
                        plugin_results[f'validation_proof_test_{i+1}'] = {'success': False}
            
            self.results['individual_plugin_tests'] = plugin_results
            
            # Summary
            successful_tests = sum(1 for result in plugin_results.values() if result.get('success', False))
            total_tests = len(plugin_results)
            logger.info(f"Individual plugin tests: {successful_tests}/{total_tests} successful")
            
            return True
            
        except Exception as e:
            logger.error(f"Individual plugin test example failed: {e}")
            return False
    
    def example_3_workflow_orchestration(self):
        """Example 3: Multi-Plugin Workflow Orchestration"""
        logger.info("=" * 60)
        logger.info("Example 3: Multi-Plugin Workflow Orchestration")
        logger.info("=" * 60)
        
        try:
            # Define comprehensive text analysis workflow
            workflow_steps = [
                {
                    'plugin_type': 'feature_extractors',
                    'plugin_name': 'text_analyzer',
                    'operation': 'extract_features',
                    'config': {}
                },
                {
                    'plugin_type': 'models',
                    'plugin_name': 'sentiment_model',
                    'operation': 'predict',
                    'config': {}
                },
                {
                    'plugin_type': 'proof_languages',
                    'plugin_name': 'validation_proof',
                    'operation': 'construct_proof',
                    'config': {'rules': ['text_length', 'sentiment_consistency', 'content_quality']}
                }
            ]
            
            # Register workflow
            self.orchestrator.register_workflow('text_analysis_pipeline', workflow_steps)
            
            # Test workflow with various inputs
            test_cases = [
                {
                    'name': 'positive_text',
                    'text': 'This is an absolutely wonderful and amazing example of excellent text analysis capabilities!'
                },
                {
                    'name': 'negative_text',
                    'text': 'This terrible awful horrible example shows the worst possible text processing results.'
                },
                {
                    'name': 'neutral_text',
                    'text': 'This is a standard example of normal text processing with average results.'
                },
                {
                    'name': 'short_text',
                    'text': 'Short.'
                },
                {
                    'name': 'repetitive_text',
                    'text': 'test test test test test test test test test test test test test'
                }
            ]
            
            workflow_results = {}
            
            for test_case in test_cases:
                logger.info(f"Running workflow for: {test_case['name']}")
                
                try:
                    result = self.orchestrator.execute_workflow(
                        'text_analysis_pipeline',
                        {'text': test_case['text']}
                    )
                    
                    # Extract key results
                    analysis_summary = {
                        'feature_count': len(result.get('features', {})),
                        'sentiment': result.get('prediction', 0.0),
                        'sentiment_confidence': result.get('confidence', 0.0),
                        'validation_passed': result.get('proof_valid', False),
                        'validation_confidence': result.get('proof_confidence', 0.0)
                    }
                    
                    workflow_results[test_case['name']] = {
                        'success': True,
                        'analysis': analysis_summary
                    }
                    
                    logger.info(f"  ✓ Analysis completed: {analysis_summary}")
                    
                except Exception as e:
                    logger.error(f"  ✗ Workflow failed: {str(e)}")
                    workflow_results[test_case['name']] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Analyze workflow performance
            execution_history = self.orchestrator.execution_history
            if execution_history:
                avg_execution_time = sum(record['total_time'] for record in execution_history) / len(execution_history)
                success_rate = sum(1 for record in execution_history if record['success']) / len(execution_history)
                
                logger.info(f"Workflow performance:")
                logger.info(f"  Executions: {len(execution_history)}")
                logger.info(f"  Success rate: {success_rate:.1%}")
                logger.info(f"  Average execution time: {avg_execution_time:.4f}s")
            
            self.results['workflow_orchestration'] = {
                'workflow_results': workflow_results,
                'execution_count': len(execution_history),
                'success_rate': success_rate if execution_history else 0.0,
                'avg_execution_time': avg_execution_time if execution_history else 0.0
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Workflow orchestration example failed: {e}")
            return False
    
    def example_4_plugin_communication(self):
        """Example 4: Inter-Plugin Communication and Data Flow"""
        logger.info("=" * 60)
        logger.info("Example 4: Inter-Plugin Communication and Data Flow")
        logger.info("=" * 60)
        
        try:
            # Demonstrate data flow between plugins
            test_text = "This fascinating example demonstrates excellent plugin communication patterns!"
            
            logger.info("Demonstrating step-by-step plugin communication:")
            
            # Step 1: Feature extraction
            logger.info("Step 1: Feature Extraction")
            text_analyzer = self.custom_plugins.get('feature_extractors.text_analyzer')
            feature_result = text_analyzer.extract_features({'text': test_text})
            
            if feature_result.status.name == 'SUCCESS':
                logger.info(f"  ✓ Extracted {len(feature_result.features)} features")
                logger.info(f"  ✓ Processing time: {feature_result.processing_time:.4f}s")
                
                # Show some key features
                key_features = {k: v for k, v in list(feature_result.features.items())[:5]}
                logger.info(f"  Sample features: {key_features}")
            else:
                logger.error("  ✗ Feature extraction failed")
                return False
            
            # Step 2: Sentiment prediction using features
            logger.info("Step 2: Sentiment Prediction")
            sentiment_model = self.custom_plugins.get('models.sentiment_model')
            
            # Create enhanced input with features
            prediction_input = {
                'text': test_text,
                'features': feature_result.features,
                'feature_metadata': feature_result.metadata
            }
            
            prediction_result = sentiment_model.predict(prediction_input)
            
            if prediction_result.status.name == 'SUCCESS':
                logger.info(f"  ✓ Sentiment: {prediction_result.prediction:.3f}")
                logger.info(f"  ✓ Confidence: {prediction_result.confidence:.3f}")
                logger.info(f"  ✓ Processing time: {prediction_result.prediction_time:.4f}s")
            else:
                logger.error("  ✗ Sentiment prediction failed")
                return False
            
            # Step 3: Comprehensive validation
            logger.info("Step 3: Comprehensive Validation")
            validation_proof = self.custom_plugins.get('proof_languages.validation_proof')
            
            # Create comprehensive context
            validation_context = {
                'text': test_text,
                'features': feature_result.features,
                'feature_metadata': feature_result.metadata,
                'sentiment_result': {
                    'prediction': prediction_result.prediction,
                    'confidence': prediction_result.confidence,
                    'metadata': prediction_result.metadata
                },
                'rules': ['text_length', 'sentiment_consistency', 'word_frequency', 'content_quality']
            }
            
            proof_result = validation_proof.construct_proof(validation_context)
            
            if proof_result.status.name == 'SUCCESS':
                logger.info(f"  ✓ Validation passed: {proof_result.proof_valid}")
                logger.info(f"  ✓ Validation confidence: {proof_result.confidence:.3f}")
                logger.info(f"  ✓ Processing time: {proof_result.construction_time:.4f}s")
                
                # Show validation details
                if hasattr(proof_result, 'proof_steps'):
                    passed_rules = sum(1 for step in proof_result.proof_steps if step['result'])
                    total_rules = len(proof_result.proof_steps)
                    logger.info(f"  ✓ Rules passed: {passed_rules}/{total_rules}")
            else:
                logger.error("  ✗ Validation failed")
                return False
            
            # Demonstrate data aggregation
            logger.info("Step 4: Data Aggregation and Summary")
            
            aggregated_result = {
                'input_text': test_text,
                'analysis': {
                    'features': {
                        'count': len(feature_result.features),
                        'extraction_time': feature_result.processing_time,
                        'key_features': {k: v for k, v in list(feature_result.features.items())[:3]}
                    },
                    'sentiment': {
                        'score': prediction_result.prediction,
                        'confidence': prediction_result.confidence,
                        'prediction_time': prediction_result.prediction_time,
                        'label': prediction_result.metadata.get('sentiment_label', 'unknown')
                    },
                    'validation': {
                        'passed': proof_result.proof_valid,
                        'confidence': proof_result.confidence,
                        'validation_time': proof_result.construction_time,
                        'rules_checked': len(proof_result.proof_steps) if hasattr(proof_result, 'proof_steps') else 0
                    }
                },
                'total_processing_time': (
                    feature_result.processing_time +
                    prediction_result.prediction_time +
                    proof_result.construction_time
                ),
                'overall_score': (
                    proof_result.confidence * 0.4 +
                    prediction_result.confidence * 0.3 +
                    (1.0 if len(feature_result.features) > 10 else 0.5) * 0.3
                )
            }
            
            logger.info(f"  ✓ Overall analysis score: {aggregated_result['overall_score']:.3f}")
            logger.info(f"  ✓ Total processing time: {aggregated_result['total_processing_time']:.4f}s")
            
            self.results['plugin_communication'] = {
                'success': True,
                'aggregated_result': aggregated_result,
                'communication_flow': [
                    'feature_extraction → features',
                    'features + text → sentiment_prediction',
                    'features + sentiment + text → validation',
                    'all_results → aggregated_analysis'
                ]
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Plugin communication example failed: {e}")
            return False
    
    def run_all_examples(self):
        """Run all custom plugin examples"""
        logger.info("Starting Custom Plugins Examples")
        logger.info("=" * 80)
        
        examples = [
            ('Create Custom Plugins', self.example_1_create_custom_plugins),
            ('Test Individual Plugins', self.example_2_test_individual_plugins),
            ('Workflow Orchestration', self.example_3_workflow_orchestration),
            ('Plugin Communication', self.example_4_plugin_communication)
        ]
        
        results_summary = {}
        
        for example_name, example_func in examples:
            try:
                logger.info(f"\nRunning: {example_name}")
                start_time = time.time()
                
                success = example_func()
                execution_time = time.time() - start_time
                
                results_summary[example_name] = {
                    'success': success,
                    'execution_time': execution_time
                }
                
                if success:
                    logger.info(f"✓ {example_name} completed in {execution_time:.2f}s")
                else:
                    logger.error(f"✗ {example_name} failed after {execution_time:.2f}s")
                    
            except Exception as e:
                logger.error(f"✗ {example_name} crashed: {e}")
                results_summary[example_name] = {'success': False, 'error': str(e)}
        
        # Save results
        results_file = Path('examples/custom_plugins_results.json')
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        return results_summary


def main():
    """Main function to run custom plugins examples"""
    try:
        examples = CustomPluginsExample()
        results = examples.run_all_examples()
        return results
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    main()