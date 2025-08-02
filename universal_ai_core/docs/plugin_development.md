# Plugin Development Guide

## Overview

The Universal AI Core plugin system provides a flexible architecture for extending functionality across different domains. This guide covers how to create, register, and deploy custom plugins based on the established patterns from the molecular analysis system.

## Plugin Architecture

### Plugin Types

The system supports four main plugin types:

1. **Feature Extractors**: Extract meaningful features from raw data
2. **Models**: Machine learning models for prediction and analysis
3. **Proof Languages**: Generate formal proofs and validation logic
4. **Knowledge Bases**: Store and query domain-specific knowledge

### Base Plugin Structure

All plugins inherit from the base plugin class:

```python
from universal_ai_core.plugins.base import BasePlugin
from abc import abstractmethod

class BasePlugin:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.plugin_type = None  # Set in subclass
        self.domain = None       # Set in subclass
        self.version = "1.0.0"
        self.enabled = config.get("enabled", True)
    
    @abstractmethod
    def process(self, data: Any) -> Dict[str, Any]:
        """Main processing method - implement in subclass"""
        pass
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """Validate plugin configuration"""
        return True, []
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata"""
        return {
            "type": self.plugin_type,
            "domain": self.domain,
            "version": self.version,
            "enabled": self.enabled
        }
```

## Creating Feature Extractor Plugins

### Template Structure

```python
from universal_ai_core.plugins.base import BasePlugin
from typing import Dict, Any, List
import logging

class CustomFeatureExtractorPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.plugin_type = "feature_extractors"
        self.domain = "custom_domain"  # Your domain name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Domain-specific configuration
        self.feature_types = config.get("feature_types", ["basic"])
        self.normalize = config.get("normalize", True)
        self.cache_features = config.get("cache_features", True)
    
    def extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from input data"""
        try:
            # Validate input data
            if not self._validate_input(data):
                raise ValueError("Invalid input data format")
            
            # Extract different types of features
            features = {}
            
            if "basic" in self.feature_types:
                features["basic_features"] = self._extract_basic_features(data)
            
            if "advanced" in self.feature_types:
                features["advanced_features"] = self._extract_advanced_features(data)
            
            # Normalize if requested
            if self.normalize:
                features = self._normalize_features(features)
            
            return {
                "features": features,
                "feature_count": self._count_features(features),
                "processing_time": 0.1,  # Track actual processing time
                "metadata": {
                    "domain": self.domain,
                    "feature_types": self.feature_types,
                    "normalized": self.normalize
                }
            }
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return {
                "features": {},
                "feature_count": 0,
                "error": str(e),
                "processing_time": 0.0
            }
    
    def _validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data structure"""
        # Implement domain-specific validation
        required_fields = ["input_data"]  # Define required fields
        return all(field in data for field in required_fields)
    
    def _extract_basic_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract basic numerical features"""
        # Implement basic feature extraction logic
        input_data = data["input_data"]
        
        # Example: convert text to numerical features
        if isinstance(input_data, str):
            return [len(input_data), input_data.count(' '), len(set(input_data))]
        elif isinstance(input_data, list):
            return [len(input_data), sum(input_data) if all(isinstance(x, (int, float)) for x in input_data) else 0]
        else:
            return [1.0, 0.0, 0.0]  # Default features
    
    def _extract_advanced_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract advanced domain-specific features"""
        # Implement advanced feature extraction logic
        # This is where you add domain expertise
        return [0.5, 0.3, 0.8, 0.2]  # Example features
    
    def _normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize feature values"""
        import numpy as np
        
        normalized_features = {}
        for feature_type, feature_values in features.items():
            if isinstance(feature_values, list):
                # Min-max normalization
                arr = np.array(feature_values)
                if arr.max() > arr.min():
                    normalized = (arr - arr.min()) / (arr.max() - arr.min())
                    normalized_features[feature_type] = normalized.tolist()
                else:
                    normalized_features[feature_type] = feature_values
            else:
                normalized_features[feature_type] = feature_values
        
        return normalized_features
    
    def _count_features(self, features: Dict[str, Any]) -> int:
        """Count total number of features extracted"""
        total = 0
        for feature_values in features.values():
            if isinstance(feature_values, list):
                total += len(feature_values)
            else:
                total += 1
        return total
    
    # Required method from base class
    def process(self, data: Any) -> Dict[str, Any]:
        """Main processing entry point"""
        return self.extract_features(data)
```

### Example: Text Analysis Feature Extractor

```python
class TextAnalysisFeatureExtractorPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.plugin_type = "feature_extractors"
        self.domain = "text_analysis"
        
        # Text-specific configuration
        self.include_sentiment = config.get("include_sentiment", True)
        self.include_readability = config.get("include_readability", True)
        self.max_length = config.get("max_length", 10000)
    
    def extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text analysis features"""
        text = data.get("text", "")
        
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        features = {}
        
        # Basic text features
        features["basic_stats"] = [
            len(text),                    # Character count
            len(text.split()),           # Word count
            len(text.split('.')),        # Sentence count
            text.count('\n'),            # Line count
        ]
        
        # Sentiment analysis
        if self.include_sentiment:
            features["sentiment"] = self._analyze_sentiment(text)
        
        # Readability metrics
        if self.include_readability:
            features["readability"] = self._calculate_readability(text)
        
        return {
            "features": features,
            "feature_count": self._count_features(features),
            "processing_time": 0.05
        }
    
    def _analyze_sentiment(self, text: str) -> List[float]:
        """Simple sentiment analysis"""
        positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
        negative_words = ["bad", "terrible", "awful", "horrible", "worst"]
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_words = len(words)
        if total_words == 0:
            return [0.0, 0.0, 0.0]
        
        return [
            positive_count / total_words,  # Positive ratio
            negative_count / total_words,  # Negative ratio
            (positive_count - negative_count) / total_words  # Net sentiment
        ]
    
    def _calculate_readability(self, text: str) -> List[float]:
        """Calculate readability metrics"""
        words = text.split()
        sentences = text.split('.')
        
        if not words or not sentences:
            return [0.0, 0.0]
        
        avg_words_per_sentence = len(words) / len(sentences)
        avg_chars_per_word = sum(len(word) for word in words) / len(words)
        
        return [avg_words_per_sentence, avg_chars_per_word]
    
    def process(self, data: Any) -> Dict[str, Any]:
        return self.extract_features(data)
```

## Creating Model Plugins

### Template Structure

```python
from universal_ai_core.plugins.base import BasePlugin
import numpy as np
from typing import Dict, Any, Optional
import pickle
import os

class CustomModelPlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.plugin_type = "models"
        self.domain = "custom_domain"
        
        # Model configuration
        self.model_type = config.get("model_type", "neural_network")
        self.model_path = config.get("model_path", None)
        self.auto_load = config.get("auto_load", True)
        
        # Initialize model
        self.model = None
        if self.auto_load and self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)
        elif self.auto_load:
            self.model = self._create_default_model()
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using the model"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Prepare input data
            input_data = self._prepare_input(features)
            
            # Make prediction
            predictions = self._predict_internal(input_data)
            
            # Post-process results
            processed_predictions = self._post_process_predictions(predictions)
            
            return {
                "predictions": processed_predictions,
                "confidence": self._calculate_confidence(predictions),
                "model_type": self.model_type,
                "prediction_count": len(processed_predictions),
                "processing_time": 0.1
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {
                "predictions": [],
                "error": str(e),
                "processing_time": 0.0
            }
    
    def train(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the model"""
        try:
            X = training_data.get("features", [])
            y = training_data.get("labels", [])
            
            if not X or not y:
                raise ValueError("Training data missing features or labels")
            
            # Prepare training data
            X_processed = self._prepare_training_data(X)
            y_processed = self._prepare_labels(y)
            
            # Train model
            if self.model is None:
                self.model = self._create_default_model()
            
            training_result = self._train_internal(X_processed, y_processed)
            
            return {
                "status": "success",
                "training_samples": len(X),
                "model_type": self.model_type,
                "training_time": training_result.get("training_time", 0.0),
                "model_info": training_result.get("model_info", {})
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def save_model(self, model_path: str) -> bool:
        """Save the trained model"""
        try:
            if self.model is None:
                return False
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            self.model_path = model_path
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, model_path: str) -> bool:
        """Load a pre-trained model"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.model_path = model_path
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def _create_default_model(self):
        """Create default model based on configuration"""
        if self.model_type == "neural_network":
            return self._create_neural_network()
        elif self.model_type == "random_forest":
            return self._create_random_forest()
        else:
            return self._create_simple_model()
    
    def _create_neural_network(self):
        """Create neural network model"""
        try:
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        except ImportError:
            return self._create_simple_model()
    
    def _create_random_forest(self):
        """Create random forest model"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        except ImportError:
            return self._create_simple_model()
    
    def _create_simple_model(self):
        """Create simple linear model as fallback"""
        try:
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        except ImportError:
            # Ultimate fallback - simple statistical model
            return {"type": "mean_predictor", "trained": False}
    
    def _prepare_input(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare input features for prediction"""
        # Flatten all features into a single array
        feature_values = []
        for key, values in features.items():
            if isinstance(values, list):
                feature_values.extend(values)
            elif isinstance(values, (int, float)):
                feature_values.append(values)
        
        return np.array(feature_values).reshape(1, -1)
    
    def _predict_internal(self, input_data: np.ndarray):
        """Internal prediction method"""
        if hasattr(self.model, 'predict'):
            return self.model.predict(input_data)
        elif isinstance(self.model, dict) and self.model.get("type") == "mean_predictor":
            # Simple mean predictor fallback
            return [np.mean(input_data)]
        else:
            return [0.0]  # Default prediction
    
    def _post_process_predictions(self, predictions) -> List[float]:
        """Post-process raw predictions"""
        if isinstance(predictions, np.ndarray):
            return predictions.tolist()
        elif isinstance(predictions, list):
            return predictions
        else:
            return [float(predictions)]
    
    def _calculate_confidence(self, predictions) -> float:
        """Calculate prediction confidence"""
        # Simple confidence calculation - can be enhanced
        if hasattr(self.model, 'predict_proba'):
            try:
                # For classification models
                return 0.8  # Placeholder
            except:
                pass
        
        # Default confidence for regression
        return 0.7
    
    def _prepare_training_data(self, X: List) -> np.ndarray:
        """Prepare training features"""
        return np.array(X)
    
    def _prepare_labels(self, y: List) -> np.ndarray:
        """Prepare training labels"""
        return np.array(y)
    
    def _train_internal(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Internal training method"""
        start_time = time.time()
        
        if hasattr(self.model, 'fit'):
            self.model.fit(X, y)
            training_time = time.time() - start_time
            
            return {
                "training_time": training_time,
                "model_info": {
                    "samples": len(X),
                    "features": X.shape[1] if len(X.shape) > 1 else 1,
                    "model_type": type(self.model).__name__
                }
            }
        elif isinstance(self.model, dict):
            # Simple statistical model
            self.model["mean"] = np.mean(y)
            self.model["trained"] = True
            
            return {
                "training_time": time.time() - start_time,
                "model_info": {"type": "mean_predictor", "mean_value": self.model["mean"]}
            }
        
        return {"training_time": 0.0, "model_info": {}}
    
    def process(self, data: Any) -> Dict[str, Any]:
        """Main processing entry point"""
        if "features" in data:
            return self.predict(data)
        elif "training_data" in data:
            return self.train(data["training_data"])
        else:
            return {"error": "Invalid input data format"}
```

## Creating Proof Language Plugins

### Template Structure

```python
from universal_ai_core.plugins.base import BasePlugin
from typing import Dict, Any, List
from datetime import datetime

class CustomProofLanguagePlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.plugin_type = "proof_languages"
        self.domain = "custom_domain"
        
        # Proof configuration
        self.proof_types = config.get("proof_types", ["validation", "compliance"])
        self.include_evidence = config.get("include_evidence", True)
        self.formal_notation = config.get("formal_notation", False)
    
    def generate_proof(self, proof_request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate formal proof based on request"""
        try:
            proof_type = proof_request.get("proof_type", "validation")
            data = proof_request.get("data", {})
            assertions = proof_request.get("assertions", [])
            
            if proof_type not in self.proof_types:
                raise ValueError(f"Unsupported proof type: {proof_type}")
            
            # Generate proof based on type
            if proof_type == "validation":
                proof = self._generate_validation_proof(data, assertions)
            elif proof_type == "compliance":
                proof = self._generate_compliance_proof(data, assertions)
            else:
                proof = self._generate_generic_proof(data, assertions)
            
            return {
                "proof_type": proof_type,
                "statements": proof["statements"],
                "conclusion": proof["conclusion"],
                "confidence": proof["confidence"],
                "evidence": proof.get("evidence", []) if self.include_evidence else [],
                "timestamp": datetime.now().isoformat(),
                "formal_notation": self.formal_notation
            }
            
        except Exception as e:
            self.logger.error(f"Proof generation failed: {e}")
            return {
                "proof_type": "error",
                "statements": [],
                "conclusion": f"Proof generation failed: {e}",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _generate_validation_proof(self, data: Dict[str, Any], assertions: List[str]) -> Dict[str, Any]:
        """Generate validation proof"""
        statements = []
        evidence = []
        
        # Check basic data validity
        if data:
            statements.append("P1: Input data is non-empty")
            evidence.append("Data contains {} fields".format(len(data)))
        else:
            statements.append("P1: Input data is empty")
            evidence.append("No data provided for validation")
        
        # Validate assertions
        for i, assertion in enumerate(assertions):
            statement = f"P{i+2}: Assertion '{assertion}' is "
            if self._validate_assertion(data, assertion):
                statement += "satisfied"
                evidence.append(f"Assertion {i+1} validation passed")
            else:
                statement += "not satisfied"
                evidence.append(f"Assertion {i+1} validation failed")
            statements.append(statement)
        
        # Generate conclusion
        all_valid = data and all(self._validate_assertion(data, a) for a in assertions)
        conclusion = "Therefore, the data is valid" if all_valid else "Therefore, the data is invalid"
        confidence = 1.0 if all_valid else 0.0
        
        return {
            "statements": statements,
            "conclusion": conclusion,
            "confidence": confidence,
            "evidence": evidence
        }
    
    def _generate_compliance_proof(self, data: Dict[str, Any], assertions: List[str]) -> Dict[str, Any]:
        """Generate compliance proof"""
        statements = []
        evidence = []
        
        # Check compliance requirements
        compliance_rules = [
            "data_completeness",
            "format_compliance",
            "range_validation"
        ]
        
        satisfied_rules = 0
        for i, rule in enumerate(compliance_rules):
            statement = f"C{i+1}: Compliance rule '{rule}' is "
            if self._check_compliance_rule(data, rule):
                statement += "satisfied"
                evidence.append(f"Rule {rule} compliance verified")
                satisfied_rules += 1
            else:
                statement += "not satisfied"
                evidence.append(f"Rule {rule} compliance failed")
            statements.append(statement)
        
        # Generate conclusion
        compliance_ratio = satisfied_rules / len(compliance_rules)
        if compliance_ratio >= 0.8:
            conclusion = "Therefore, the data is compliant"
            confidence = compliance_ratio
        else:
            conclusion = "Therefore, the data is not compliant"
            confidence = 1.0 - compliance_ratio
        
        return {
            "statements": statements,
            "conclusion": conclusion,
            "confidence": confidence,
            "evidence": evidence
        }
    
    def _generate_generic_proof(self, data: Dict[str, Any], assertions: List[str]) -> Dict[str, Any]:
        """Generate generic proof"""
        statements = ["G1: Generic proof requested"]
        evidence = ["Proof type not specifically handled"]
        conclusion = "Generic proof completed"
        confidence = 0.5
        
        return {
            "statements": statements,
            "conclusion": conclusion,
            "confidence": confidence,
            "evidence": evidence
        }
    
    def _validate_assertion(self, data: Dict[str, Any], assertion: str) -> bool:
        """Validate a specific assertion against data"""
        # Simple assertion validation - enhance based on domain
        if "required" in assertion.lower():
            required_field = assertion.lower().replace("required", "").strip()
            return required_field in data
        elif "positive" in assertion.lower():
            # Check for positive values
            for value in data.values():
                if isinstance(value, (int, float)) and value > 0:
                    return True
            return False
        elif "non_empty" in assertion.lower():
            return bool(data)
        else:
            # Default: assume assertion is satisfied
            return True
    
    def _check_compliance_rule(self, data: Dict[str, Any], rule: str) -> bool:
        """Check specific compliance rule"""
        if rule == "data_completeness":
            return len(data) > 0
        elif rule == "format_compliance":
            # Check if data has expected structure
            return isinstance(data, dict)
        elif rule == "range_validation":
            # Check if numerical values are in reasonable ranges
            for value in data.values():
                if isinstance(value, (int, float)):
                    if value < -1000 or value > 1000:  # Example range
                        return False
            return True
        else:
            return True  # Unknown rule, assume compliant
    
    def process(self, data: Any) -> Dict[str, Any]:
        """Main processing entry point"""
        return self.generate_proof(data)
```

## Creating Knowledge Base Plugins

### Template Structure

```python
from universal_ai_core.plugins.base import BasePlugin
from typing import Dict, Any, List, Optional
import json
import sqlite3
from pathlib import Path

class CustomKnowledgeBasePlugin(BasePlugin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.plugin_type = "knowledge_bases"
        self.domain = "custom_domain"
        
        # Knowledge base configuration
        self.db_path = config.get("db_path", "knowledge_base.db")
        self.enable_caching = config.get("enable_caching", True)
        self.max_results = config.get("max_results", 100)
        
        # Initialize database
        self._init_database()
        
        # Cache for frequent queries
        self.query_cache = {} if self.enable_caching else None
    
    def query(self, query_request: Dict[str, Any]) -> Dict[str, Any]:
        """Query the knowledge base"""
        try:
            query_type = query_request.get("type", "search")
            query_params = query_request.get("params", {})
            
            # Check cache first
            cache_key = self._generate_cache_key(query_request)
            if self.query_cache and cache_key in self.query_cache:
                return self.query_cache[cache_key]
            
            # Execute query based on type
            if query_type == "search":
                results = self._search_knowledge(query_params)
            elif query_type == "lookup":
                results = self._lookup_knowledge(query_params)
            elif query_type == "similarity":
                results = self._similarity_search(query_params)
            else:
                results = self._generic_query(query_params)
            
            response = {
                "query_type": query_type,
                "results": results,
                "result_count": len(results),
                "query_time": 0.05,
                "cache_hit": False
            }
            
            # Cache result
            if self.query_cache and len(results) <= self.max_results:
                self.query_cache[cache_key] = response
            
            return response
            
        except Exception as e:
            self.logger.error(f"Knowledge base query failed: {e}")
            return {
                "query_type": "error",
                "results": [],
                "result_count": 0,
                "error": str(e)
            }
    
    def add_knowledge(self, knowledge_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add new knowledge to the database"""
        try:
            knowledge_type = knowledge_data.get("type", "fact")
            content = knowledge_data.get("content", {})
            metadata = knowledge_data.get("metadata", {})
            
            # Store in database
            knowledge_id = self._store_knowledge(knowledge_type, content, metadata)
            
            # Clear relevant cache entries
            if self.query_cache:
                self._clear_related_cache(knowledge_type)
            
            return {
                "status": "success",
                "knowledge_id": knowledge_id,
                "knowledge_type": knowledge_type,
                "storage_time": 0.02
            }
            
        except Exception as e:
            self.logger.error(f"Failed to add knowledge: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def update_knowledge(self, knowledge_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing knowledge"""
        try:
            success = self._update_knowledge_record(knowledge_id, updates)
            
            if success:
                # Clear cache
                if self.query_cache:
                    self.query_cache.clear()
                
                return {
                    "status": "success",
                    "knowledge_id": knowledge_id,
                    "update_time": 0.01
                }
            else:
                return {
                    "status": "error",
                    "error": "Knowledge record not found"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to update knowledge: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _init_database(self):
        """Initialize the knowledge database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create knowledge table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    knowledge_id TEXT UNIQUE,
                    type TEXT,
                    content TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_type ON knowledge(type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_id ON knowledge(knowledge_id)')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
    
    def _search_knowledge(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search knowledge by keywords"""
        search_term = params.get("search_term", "")
        knowledge_type = params.get("type", None)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if knowledge_type:
                cursor.execute('''
                    SELECT knowledge_id, type, content, metadata 
                    FROM knowledge 
                    WHERE type = ? AND (content LIKE ? OR metadata LIKE ?)
                    LIMIT ?
                ''', (knowledge_type, f'%{search_term}%', f'%{search_term}%', self.max_results))
            else:
                cursor.execute('''
                    SELECT knowledge_id, type, content, metadata 
                    FROM knowledge 
                    WHERE content LIKE ? OR metadata LIKE ?
                    LIMIT ?
                ''', (f'%{search_term}%', f'%{search_term}%', self.max_results))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "knowledge_id": row[0],
                    "type": row[1],
                    "content": json.loads(row[2]) if row[2] else {},
                    "metadata": json.loads(row[3]) if row[3] else {}
                })
            
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def _lookup_knowledge(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Lookup specific knowledge by ID"""
        knowledge_id = params.get("knowledge_id", "")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT knowledge_id, type, content, metadata 
                FROM knowledge 
                WHERE knowledge_id = ?
            ''', (knowledge_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return [{
                    "knowledge_id": row[0],
                    "type": row[1],
                    "content": json.loads(row[2]) if row[2] else {},
                    "metadata": json.loads(row[3]) if row[3] else {}
                }]
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Lookup failed: {e}")
            return []
    
    def _similarity_search(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        # Simple similarity search implementation
        reference_content = params.get("reference", {})
        threshold = params.get("similarity_threshold", 0.5)
        
        # Get all knowledge entries
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT knowledge_id, type, content, metadata FROM knowledge LIMIT ?', 
                         (self.max_results,))
            
            results = []
            for row in cursor.fetchall():
                content = json.loads(row[2]) if row[2] else {}
                similarity = self._calculate_similarity(reference_content, content)
                
                if similarity >= threshold:
                    results.append({
                        "knowledge_id": row[0],
                        "type": row[1],
                        "content": content,
                        "metadata": json.loads(row[3]) if row[3] else {},
                        "similarity_score": similarity
                    })
            
            conn.close()
            
            # Sort by similarity score
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results
            
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []
    
    def _generic_query(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle generic queries"""
        # Return all knowledge of specified type
        knowledge_type = params.get("type", None)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if knowledge_type:
                cursor.execute('''
                    SELECT knowledge_id, type, content, metadata 
                    FROM knowledge 
                    WHERE type = ?
                    LIMIT ?
                ''', (knowledge_type, self.max_results))
            else:
                cursor.execute('SELECT knowledge_id, type, content, metadata FROM knowledge LIMIT ?', 
                             (self.max_results,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "knowledge_id": row[0],
                    "type": row[1],
                    "content": json.loads(row[2]) if row[2] else {},
                    "metadata": json.loads(row[3]) if row[3] else {}
                })
            
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Generic query failed: {e}")
            return []
    
    def _store_knowledge(self, knowledge_type: str, content: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Store knowledge in database"""
        import uuid
        
        knowledge_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO knowledge (knowledge_id, type, content, metadata)
            VALUES (?, ?, ?, ?)
        ''', (knowledge_id, knowledge_type, json.dumps(content), json.dumps(metadata)))
        
        conn.commit()
        conn.close()
        
        return knowledge_id
    
    def _update_knowledge_record(self, knowledge_id: str, updates: Dict[str, Any]) -> bool:
        """Update knowledge record in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current record
            cursor.execute('SELECT content, metadata FROM knowledge WHERE knowledge_id = ?', (knowledge_id,))
            row = cursor.fetchone()
            
            if not row:
                conn.close()
                return False
            
            # Update content and metadata
            current_content = json.loads(row[0]) if row[0] else {}
            current_metadata = json.loads(row[1]) if row[1] else {}
            
            if "content" in updates:
                current_content.update(updates["content"])
            if "metadata" in updates:
                current_metadata.update(updates["metadata"])
            
            cursor.execute('''
                UPDATE knowledge 
                SET content = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE knowledge_id = ?
            ''', (json.dumps(current_content), json.dumps(current_metadata), knowledge_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Update failed: {e}")
            return False
    
    def _calculate_similarity(self, content1: Dict[str, Any], content2: Dict[str, Any]) -> float:
        """Calculate similarity between two content objects"""
        # Simple similarity calculation - enhance based on domain
        if not content1 or not content2:
            return 0.0
        
        # Compare keys
        keys1 = set(content1.keys())
        keys2 = set(content2.keys())
        key_similarity = len(keys1 & keys2) / len(keys1 | keys2) if keys1 | keys2 else 0.0
        
        # Compare values for common keys
        common_keys = keys1 & keys2
        value_similarity = 0.0
        
        if common_keys:
            matches = 0
            for key in common_keys:
                if content1[key] == content2[key]:
                    matches += 1
            value_similarity = matches / len(common_keys)
        
        return (key_similarity + value_similarity) / 2
    
    def _generate_cache_key(self, query_request: Dict[str, Any]) -> str:
        """Generate cache key for query"""
        return str(hash(json.dumps(query_request, sort_keys=True)))
    
    def _clear_related_cache(self, knowledge_type: str):
        """Clear cache entries related to a knowledge type"""
        if not self.query_cache:
            return
        
        keys_to_remove = []
        for key in self.query_cache:
            # Simple heuristic - clear all cache for now
            keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.query_cache[key]
    
    def process(self, data: Any) -> Dict[str, Any]:
        """Main processing entry point"""
        if "query" in data:
            return self.query(data["query"])
        elif "add_knowledge" in data:
            return self.add_knowledge(data["add_knowledge"])
        elif "update_knowledge" in data:
            return self.update_knowledge(data["knowledge_id"], data["update_knowledge"])
        else:
            return {"error": "Invalid operation"}
```

## Plugin Registration and Loading

### Manual Registration

```python
from universal_ai_core import create_api

# Create API instance
api = create_api()

# Register custom plugin
api.core.plugin_manager.register_plugin("custom_extractor", CustomFeatureExtractorPlugin)

# Load plugin
success = api.core.plugin_manager.load_plugin("feature_extractors", "custom")

# Use plugin
result = api.process_data(data, ["custom_features"])
```

### Automatic Loading

Create a plugin configuration file:

```yaml
# custom_plugins.yaml
plugins:
  feature_extractors:
    custom_domain:
      enabled: true
      feature_types: ["basic", "advanced"]
      normalize: true
      cache_features: true
  
  models:
    custom_domain:
      enabled: true
      model_type: "neural_network"
      auto_load: true
  
  proof_languages:
    custom_domain:
      enabled: true
      proof_types: ["validation", "compliance"]
      include_evidence: true
  
  knowledge_bases:
    custom_domain:
      enabled: true
      db_path: "./custom_knowledge.db"
      enable_caching: true
```

## Testing Plugins

### Unit Tests

```python
import pytest
from unittest.mock import Mock
from your_plugin import CustomFeatureExtractorPlugin

class TestCustomFeatureExtractorPlugin:
    def test_plugin_initialization(self):
        config = {"enabled": True, "feature_types": ["basic"]}
        plugin = CustomFeatureExtractorPlugin(config)
        
        assert plugin.plugin_type == "feature_extractors"
        assert plugin.domain == "custom_domain"
        assert plugin.feature_types == ["basic"]
    
    def test_feature_extraction(self):
        config = {"enabled": True, "feature_types": ["basic"]}
        plugin = CustomFeatureExtractorPlugin(config)
        
        test_data = {"input_data": "test string"}
        result = plugin.extract_features(test_data)
        
        assert result["feature_count"] > 0
        assert "features" in result
        assert result.get("error") is None
    
    def test_invalid_input_handling(self):
        config = {"enabled": True, "feature_types": ["basic"]}
        plugin = CustomFeatureExtractorPlugin(config)
        
        result = plugin.extract_features({})
        assert result["feature_count"] == 0
        assert "error" in result
```

### Integration Tests

```python
def test_plugin_integration():
    from universal_ai_core import create_api
    
    # Create API and register plugin
    api = create_api()
    api.core.plugin_manager.register_plugin("custom", CustomFeatureExtractorPlugin)
    
    # Load plugin
    success = api.core.plugin_manager.load_plugin("feature_extractors", "custom")
    assert success
    
    # Test through API
    test_data = {"input_data": "integration test"}
    result = api.process_data(test_data, ["custom_features"])
    
    assert result.status == "success"
    assert result.feature_count > 0
```

## Best Practices

### Performance Optimization

1. **Efficient Data Processing**
```python
def _extract_features_efficiently(self, data: Dict[str, Any]) -> List[float]:
    # Use vectorized operations when possible
    import numpy as np
    
    # Batch process multiple items
    if isinstance(data.get("items"), list):
        return self._batch_process(data["items"])
    else:
        return self._single_process(data)
```

2. **Memory Management**
```python
def extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Process data in chunks for large datasets
        if self._is_large_dataset(data):
            return self._process_in_chunks(data)
        else:
            return self._process_all_at_once(data)
    finally:
        # Clean up temporary resources
        self._cleanup_temp_resources()
```

3. **Caching**
```python
def __init__(self, config: Dict[str, Any]):
    super().__init__(config)
    self.feature_cache = {} if config.get("enable_caching", True) else None

def extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
    # Check cache first
    cache_key = self._generate_cache_key(data)
    if self.feature_cache and cache_key in self.feature_cache:
        return self.feature_cache[cache_key]
    
    # Extract features
    result = self._extract_features_internal(data)
    
    # Cache result
    if self.feature_cache:
        self.feature_cache[cache_key] = result
    
    return result
```

### Error Handling

```python
def extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Validate input
        if not self._validate_input(data):
            return self._create_error_result("Invalid input data")
        
        # Extract features with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self._extract_features_internal(data)
            except TemporaryError as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
                
    except Exception as e:
        self.logger.error(f"Feature extraction failed: {e}")
        return self._create_error_result(str(e))

def _create_error_result(self, error_message: str) -> Dict[str, Any]:
    return {
        "features": {},
        "feature_count": 0,
        "error": error_message,
        "processing_time": 0.0
    }
```

### Configuration Management

```python
def validate_config(self) -> Tuple[bool, List[str]]:
    """Validate plugin configuration"""
    errors = []
    
    # Check required fields
    required_fields = ["enabled"]
    for field in required_fields:
        if field not in self.config:
            errors.append(f"Missing required field: {field}")
    
    # Validate field values
    if "feature_types" in self.config:
        valid_types = ["basic", "advanced", "statistical"]
        for ftype in self.config["feature_types"]:
            if ftype not in valid_types:
                errors.append(f"Invalid feature type: {ftype}")
    
    return len(errors) == 0, errors
```

## Deployment and Distribution

### Plugin Package Structure

```
my_custom_plugin/
├── __init__.py
├── feature_extractor.py
├── model.py
├── proof_language.py
├── knowledge_base.py
├── config/
│   ├── default_config.yaml
│   └── schema.json
├── tests/
│   ├── test_feature_extractor.py
│   ├── test_model.py
│   └── test_integration.py
├── docs/
│   ├── README.md
│   └── usage_examples.md
├── requirements.txt
└── setup.py
```

### Setup.py for Distribution

```python
from setuptools import setup, find_packages

setup(
    name="my-custom-plugin",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Custom plugin for Universal AI Core",
    packages=find_packages(),
    install_requires=[
        "universal-ai-core>=1.0.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
```

### Installation and Usage

```bash
# Install plugin
pip install my-custom-plugin

# Or install from source
git clone https://github.com/your-repo/my-custom-plugin
cd my-custom-plugin
pip install -e .
```

```python
# Use in Universal AI Core
from universal_ai_core import create_api
from my_custom_plugin import CustomFeatureExtractorPlugin

api = create_api()
api.core.plugin_manager.register_plugin("custom_extractor", CustomFeatureExtractorPlugin)
api.core.plugin_manager.load_plugin("feature_extractors", "custom")

result = api.process_data(data, ["custom_features"])
```

This guide provides a comprehensive foundation for developing plugins for the Universal AI Core system. Each plugin type has specific responsibilities and interfaces, but all follow the same general patterns for configuration, error handling, and integration with the core system.