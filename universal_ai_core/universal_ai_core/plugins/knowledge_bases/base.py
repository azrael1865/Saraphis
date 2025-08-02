#!/usr/bin/env python3
"""
Knowledge Base Plugin Base Classes
==================================

This module provides abstract base classes for knowledge base plugins in the Universal AI Core system.
Adapted from existing knowledge base patterns in the Saraphis codebase, made domain-agnostic.

Base Classes:
- KnowledgeBasePlugin: Abstract base for all knowledge base implementations
- KnowledgeItem: Individual knowledge representation
- QueryResult: Standardized query result container
- KnowledgeMetadata: Knowledge base metadata and versioning
"""

import logging
import time
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from enum import Enum
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """Types of knowledge that can be stored"""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    METACOGNITIVE = "metacognitive"
    RULE = "rule"
    PATTERN = "pattern"
    RELATIONSHIP = "relationship"
    CONSTRAINT = "constraint"


class QueryType(Enum):
    """Types of knowledge base queries"""
    EXACT_MATCH = "exact_match"
    FUZZY_SEARCH = "fuzzy_search"
    SEMANTIC_SEARCH = "semantic_search"
    SIMILARITY_SEARCH = "similarity_search"
    PATTERN_MATCH = "pattern_match"
    INFERENCE = "inference"
    REASONING = "reasoning"


class KnowledgeFormat(Enum):
    """Formats for knowledge representation"""
    TEXT = "text"
    JSON = "json"
    RDF = "rdf"
    GRAPH = "graph"
    VECTOR = "vector"
    LOGIC = "logic"
    ONTOLOGY = "ontology"


class OperationStatus(Enum):
    """Status of knowledge base operations"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class KnowledgeItem:
    """Individual piece of knowledge in the knowledge base"""
    id: str
    content: Any
    knowledge_type: KnowledgeType
    format: KnowledgeFormat
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    relationships: List[str] = field(default_factory=list)  # IDs of related items
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate knowledge item after initialization"""
        if not self.id:
            # Generate ID from content hash
            content_str = json.dumps(self.content, default=str, sort_keys=True)
            self.id = hashlib.md5(content_str.encode()).hexdigest()[:16]
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert knowledge item to dictionary"""
        result = {
            'id': self.id,
            'content': self.content,
            'knowledge_type': self.knowledge_type.value,
            'format': self.format.value,
            'metadata': self.metadata,
            'tags': self.tags,
            'confidence': self.confidence,
            'source': self.source,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            'relationships': self.relationships
        }
        
        if self.embedding is not None:
            result['embedding'] = self.embedding.tolist()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeItem':
        """Create knowledge item from dictionary"""
        # Convert string enums back to enum objects
        knowledge_type = KnowledgeType(data['knowledge_type'])
        format_type = KnowledgeFormat(data['format'])
        
        # Convert datetime strings back to datetime objects
        created_at = datetime.fromisoformat(data['created_at'])
        updated_at = datetime.fromisoformat(data['updated_at'])
        
        # Convert embedding back to numpy array if present
        embedding = None
        if 'embedding' in data and data['embedding'] is not None:
            embedding = np.array(data['embedding'])
        
        return cls(
            id=data['id'],
            content=data['content'],
            knowledge_type=knowledge_type,
            format=format_type,
            metadata=data.get('metadata', {}),
            tags=data.get('tags', []),
            confidence=data.get('confidence', 1.0),
            source=data.get('source', ''),
            created_at=created_at,
            updated_at=updated_at,
            version=data.get('version', 1),
            relationships=data.get('relationships', []),
            embedding=embedding
        )


@dataclass
class QueryResult:
    """Result container for knowledge base queries"""
    items: List[KnowledgeItem]
    query: str
    query_type: QueryType
    total_results: int
    retrieved_count: int
    query_time: float
    confidence_scores: List[float] = field(default_factory=list)
    similarity_scores: List[float] = field(default_factory=list)
    status: OperationStatus = OperationStatus.SUCCESS
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate query result"""
        if self.status == OperationStatus.SUCCESS and not self.items and self.total_results > 0:
            self.status = OperationStatus.PARTIAL
        
        if len(self.confidence_scores) > 0 and len(self.confidence_scores) != len(self.items):
            raise ValueError("Confidence scores length must match items length")
        
        if len(self.similarity_scores) > 0 and len(self.similarity_scores) != len(self.items):
            raise ValueError("Similarity scores length must match items length")


@dataclass
class KnowledgeBaseMetadata:
    """Metadata for knowledge base plugins"""
    name: str
    version: str
    author: str
    description: str
    supported_knowledge_types: List[KnowledgeType]
    supported_formats: List[KnowledgeFormat]
    supported_query_types: List[QueryType]
    storage_backend: str
    indexing_method: str = "none"
    vector_dimension: Optional[int] = None
    max_capacity: Optional[int] = None
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    plugin_id: str = ""
    
    def __post_init__(self):
        """Generate plugin ID if not provided"""
        if not self.plugin_id:
            content = f"{self.name}:{self.version}:{self.storage_backend}"
            self.plugin_id = hashlib.md5(content.encode()).hexdigest()


class KnowledgeBasePlugin(ABC):
    """
    Abstract base class for knowledge base plugins.
    
    This class defines the interface that all knowledge base implementations must follow.
    Adapted from existing knowledge management patterns in the Saraphis codebase.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the knowledge base plugin.
        
        Args:
            config: Plugin-specific configuration dictionary
        """
        self.config = config or {}
        self._metadata = self._create_metadata()
        self._is_initialized = False
        self._is_connected = False
        self._knowledge_count = 0
        self._query_cache = {}
        self._indices = {}
        self._relationships = {}
        self._statistics = {}
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Initialized knowledge base: {self._metadata.name}")
    
    @abstractmethod
    def _create_metadata(self) -> KnowledgeBaseMetadata:
        """
        Create metadata for this knowledge base plugin.
        
        Returns:
            KnowledgeBaseMetadata instance with plugin information
        """
        pass
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the knowledge base storage backend.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the knowledge base storage backend"""
        pass
    
    @abstractmethod
    def store_knowledge(self, item: KnowledgeItem) -> bool:
        """
        Store a knowledge item in the knowledge base.
        
        Args:
            item: Knowledge item to store
            
        Returns:
            True if storage successful, False otherwise
        """
        pass
    
    @abstractmethod
    def retrieve_knowledge(self, item_id: str) -> Optional[KnowledgeItem]:
        """
        Retrieve a specific knowledge item by ID.
        
        Args:
            item_id: Unique identifier of the knowledge item
            
        Returns:
            KnowledgeItem if found, None otherwise
        """
        pass
    
    @abstractmethod
    def query_knowledge(self, query: str, query_type: QueryType = QueryType.FUZZY_SEARCH,
                       max_results: int = 10, **kwargs) -> QueryResult:
        """
        Query the knowledge base for relevant items.
        
        Args:
            query: Query string or pattern
            query_type: Type of query to perform
            max_results: Maximum number of results to return
            **kwargs: Additional query parameters
            
        Returns:
            QueryResult containing matching knowledge items
        """
        pass
    
    @abstractmethod
    def update_knowledge(self, item_id: str, updated_item: KnowledgeItem) -> bool:
        """
        Update an existing knowledge item.
        
        Args:
            item_id: ID of the item to update
            updated_item: Updated knowledge item
            
        Returns:
            True if update successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_knowledge(self, item_id: str) -> bool:
        """
        Delete a knowledge item from the knowledge base.
        
        Args:
            item_id: ID of the item to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        pass
    
    def initialize(self) -> bool:
        """
        Initialize the knowledge base plugin.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self._perform_initialization()
            self._is_initialized = True
            logger.info(f"Knowledge base {self._metadata.name} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base {self._metadata.name}: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the knowledge base plugin and clean up resources"""
        try:
            if self._is_connected:
                self.disconnect()
            self._perform_shutdown()
            self._is_initialized = False
            logger.info(f"Knowledge base {self._metadata.name} shutdown successfully")
        except Exception as e:
            logger.error(f"Error shutting down knowledge base {self._metadata.name}: {e}")
    
    def get_metadata(self) -> KnowledgeBaseMetadata:
        """Get knowledge base metadata"""
        return self._metadata
    
    def is_initialized(self) -> bool:
        """Check if knowledge base is initialized"""
        return self._is_initialized
    
    def is_connected(self) -> bool:
        """Check if connected to storage backend"""
        return self._is_connected
    
    def batch_store_knowledge(self, items: List[KnowledgeItem]) -> Dict[str, bool]:
        """
        Store multiple knowledge items in batch.
        
        Args:
            items: List of knowledge items to store
            
        Returns:
            Dictionary mapping item IDs to success status
        """
        results = {}
        for item in items:
            try:
                success = self.store_knowledge(item)
                results[item.id] = success
            except Exception as e:
                logger.error(f"Error storing knowledge item {item.id}: {e}")
                results[item.id] = False
        
        return results
    
    def get_knowledge_by_tags(self, tags: List[str], 
                            match_all: bool = False) -> List[KnowledgeItem]:
        """
        Retrieve knowledge items by tags.
        
        Args:
            tags: List of tags to search for
            match_all: If True, item must have all tags; if False, any tag
            
        Returns:
            List of matching knowledge items
        """
        # Default implementation using query - override for optimized search
        tag_query = " AND ".join(tags) if match_all else " OR ".join(tags)
        result = self.query_knowledge(f"tags:{tag_query}", QueryType.PATTERN_MATCH)
        return result.items
    
    def get_related_knowledge(self, item_id: str, 
                            max_depth: int = 2) -> List[KnowledgeItem]:
        """
        Get knowledge items related to a specific item.
        
        Args:
            item_id: ID of the source item
            max_depth: Maximum relationship depth to traverse
            
        Returns:
            List of related knowledge items
        """
        related_items = []
        visited = set()
        to_process = [(item_id, 0)]
        
        while to_process:
            current_id, depth = to_process.pop(0)
            
            if current_id in visited or depth >= max_depth:
                continue
            
            visited.add(current_id)
            item = self.retrieve_knowledge(current_id)
            
            if item:
                if depth > 0:  # Don't include the source item
                    related_items.append(item)
                
                # Add related items to process
                for related_id in item.relationships:
                    if related_id not in visited:
                        to_process.append((related_id, depth + 1))
        
        return related_items
    
    def add_relationship(self, item1_id: str, item2_id: str, 
                        relationship_type: str = "related") -> bool:
        """
        Add a relationship between two knowledge items.
        
        Args:
            item1_id: ID of the first item
            item2_id: ID of the second item
            relationship_type: Type of relationship
            
        Returns:
            True if relationship added successfully
        """
        try:
            # Retrieve both items
            item1 = self.retrieve_knowledge(item1_id)
            item2 = self.retrieve_knowledge(item2_id)
            
            if not item1 or not item2:
                return False
            
            # Add bidirectional relationship
            if item2_id not in item1.relationships:
                item1.relationships.append(item2_id)
                item1.updated_at = datetime.utcnow()
                self.update_knowledge(item1_id, item1)
            
            if item1_id not in item2.relationships:
                item2.relationships.append(item1_id)
                item2.updated_at = datetime.utcnow()
                self.update_knowledge(item2_id, item2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding relationship between {item1_id} and {item2_id}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.
        
        Returns:
            Dictionary with statistics about the knowledge base
        """
        return {
            "total_items": self._knowledge_count,
            "is_initialized": self._is_initialized,
            "is_connected": self._is_connected,
            "cache_size": len(self._query_cache),
            "indices_count": len(self._indices),
            "plugin_name": self._metadata.name,
            "plugin_version": self._metadata.version,
            "storage_backend": self._metadata.storage_backend,
            **self._statistics
        }
    
    def clear_cache(self) -> None:
        """Clear the query cache"""
        self._query_cache.clear()
        logger.info(f"Cleared query cache for {self._metadata.name}")
    
    def create_index(self, field: str, index_type: str = "hash") -> bool:
        """
        Create an index on a specific field for faster queries.
        
        Args:
            field: Field name to index
            index_type: Type of index to create
            
        Returns:
            True if index created successfully
        """
        try:
            # Basic implementation - override in subclasses for specific backends
            self._indices[field] = {
                "type": index_type,
                "created_at": datetime.utcnow()
            }
            logger.info(f"Created {index_type} index on field '{field}'")
            return True
        except Exception as e:
            logger.error(f"Error creating index on field '{field}': {e}")
            return False
    
    def export_knowledge(self, filepath: str, format: str = "json") -> bool:
        """
        Export knowledge base to file.
        
        Args:
            filepath: Path to export file
            format: Export format (json, csv, etc.)
            
        Returns:
            True if export successful
        """
        # Basic implementation - override in subclasses for specific formats
        try:
            if format == "json":
                # This would need to be implemented by subclass to retrieve all items
                logger.warning("Export functionality must be implemented by subclass")
                return False
            else:
                raise ValueError(f"Unsupported export format: {format}")
        except Exception as e:
            logger.error(f"Error exporting knowledge base: {e}")
            return False
    
    def import_knowledge(self, filepath: str, format: str = "json") -> bool:
        """
        Import knowledge from file.
        
        Args:
            filepath: Path to import file
            format: Import format (json, csv, etc.)
            
        Returns:
            True if import successful
        """
        try:
            if format == "json":
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    # List of knowledge items
                    items = [KnowledgeItem.from_dict(item_data) for item_data in data]
                    results = self.batch_store_knowledge(items)
                    success_count = sum(results.values())
                    logger.info(f"Imported {success_count}/{len(items)} knowledge items")
                    return success_count == len(items)
                
                return False
            else:
                raise ValueError(f"Unsupported import format: {format}")
                
        except Exception as e:
            logger.error(f"Error importing knowledge: {e}")
            return False
    
    def _validate_config(self) -> None:
        """Validate plugin configuration. Override in subclasses."""
        pass
    
    def _perform_initialization(self) -> None:
        """Perform plugin-specific initialization. Override in subclasses."""
        pass
    
    def _perform_shutdown(self) -> None:
        """Perform plugin-specific shutdown. Override in subclasses."""
        pass
    
    def _generate_cache_key(self, query: str, query_type: QueryType, **kwargs) -> str:
        """Generate cache key for query results"""
        cache_data = {
            "query": query,
            "query_type": query_type.value,
            **kwargs
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[QueryResult]:
        """Check if query result is cached"""
        cached_item = self._query_cache.get(cache_key)
        if cached_item:
            result, timestamp = cached_item
            # Check if cache is still valid (default 1 hour)
            cache_ttl = self.config.get('cache_ttl', 3600)
            if time.time() - timestamp < cache_ttl:
                return result
            else:
                # Remove expired cache entry
                del self._query_cache[cache_key]
        return None
    
    def _store_in_cache(self, cache_key: str, result: QueryResult) -> None:
        """Store query result in cache"""
        max_cache_size = self.config.get('max_cache_size', 1000)
        if len(self._query_cache) < max_cache_size:
            self._query_cache[cache_key] = (result, time.time())


# Example implementation for testing
class InMemoryKnowledgeBase(KnowledgeBasePlugin):
    """In-memory knowledge base implementation for testing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._storage = {}  # Simple dict storage
        self._tag_index = {}  # Tag to item IDs mapping
    
    def _create_metadata(self) -> KnowledgeBaseMetadata:
        return KnowledgeBaseMetadata(
            name="InMemoryKnowledgeBase",
            version="1.0.0",
            author="Universal AI Core",
            description="In-memory knowledge base for testing",
            supported_knowledge_types=list(KnowledgeType),
            supported_formats=list(KnowledgeFormat),
            supported_query_types=[QueryType.EXACT_MATCH, QueryType.FUZZY_SEARCH, QueryType.PATTERN_MATCH],
            storage_backend="memory",
            indexing_method="hash",
            capabilities=["full_text_search", "tag_search", "relationship_traversal"]
        )
    
    def connect(self) -> bool:
        """Connect to in-memory storage (always succeeds)"""
        self._is_connected = True
        return True
    
    def disconnect(self) -> None:
        """Disconnect from in-memory storage"""
        self._is_connected = False
    
    def store_knowledge(self, item: KnowledgeItem) -> bool:
        """Store knowledge item in memory"""
        try:
            self._storage[item.id] = item
            self._knowledge_count = len(self._storage)
            
            # Update tag index
            for tag in item.tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(item.id)
            
            return True
        except Exception as e:
            logger.error(f"Error storing knowledge item {item.id}: {e}")
            return False
    
    def retrieve_knowledge(self, item_id: str) -> Optional[KnowledgeItem]:
        """Retrieve knowledge item from memory"""
        return self._storage.get(item_id)
    
    def query_knowledge(self, query: str, query_type: QueryType = QueryType.FUZZY_SEARCH,
                       max_results: int = 10, **kwargs) -> QueryResult:
        """Query knowledge items in memory"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, query_type, max_results=max_results, **kwargs)
            cached_result = self._check_cache(cache_key)
            if cached_result:
                return cached_result
            
            matching_items = []
            
            if query_type == QueryType.EXACT_MATCH:
                # Exact content match
                for item in self._storage.values():
                    if str(item.content).lower() == query.lower():
                        matching_items.append(item)
            
            elif query_type == QueryType.FUZZY_SEARCH:
                # Simple fuzzy search in content
                query_lower = query.lower()
                for item in self._storage.values():
                    if query_lower in str(item.content).lower():
                        matching_items.append(item)
            
            elif query_type == QueryType.PATTERN_MATCH:
                # Pattern matching (e.g., for tags)
                if query.startswith("tags:"):
                    tag_query = query[5:]  # Remove "tags:" prefix
                    if " AND " in tag_query:
                        required_tags = tag_query.split(" AND ")
                        for item in self._storage.values():
                            if all(tag in item.tags for tag in required_tags):
                                matching_items.append(item)
                    elif " OR " in tag_query:
                        possible_tags = tag_query.split(" OR ")
                        for item in self._storage.values():
                            if any(tag in item.tags for tag in possible_tags):
                                matching_items.append(item)
                    else:
                        # Single tag
                        for item in self._storage.values():
                            if tag_query in item.tags:
                                matching_items.append(item)
            
            # Sort by confidence and limit results
            matching_items.sort(key=lambda x: x.confidence, reverse=True)
            matching_items = matching_items[:max_results]
            
            query_time = time.time() - start_time
            
            result = QueryResult(
                items=matching_items,
                query=query,
                query_type=query_type,
                total_results=len(matching_items),
                retrieved_count=len(matching_items),
                query_time=query_time,
                confidence_scores=[item.confidence for item in matching_items]
            )
            
            # Store in cache
            self._store_in_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            query_time = time.time() - start_time
            return QueryResult(
                items=[],
                query=query,
                query_type=query_type,
                total_results=0,
                retrieved_count=0,
                query_time=query_time,
                status=OperationStatus.ERROR,
                error_message=str(e)
            )
    
    def update_knowledge(self, item_id: str, updated_item: KnowledgeItem) -> bool:
        """Update knowledge item in memory"""
        try:
            if item_id not in self._storage:
                return False
            
            old_item = self._storage[item_id]
            
            # Update tag index
            for tag in old_item.tags:
                if tag in self._tag_index:
                    self._tag_index[tag].discard(item_id)
                    if not self._tag_index[tag]:
                        del self._tag_index[tag]
            
            for tag in updated_item.tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(item_id)
            
            # Update item
            updated_item.updated_at = datetime.utcnow()
            updated_item.version = old_item.version + 1
            self._storage[item_id] = updated_item
            
            return True
        except Exception as e:
            logger.error(f"Error updating knowledge item {item_id}: {e}")
            return False
    
    def delete_knowledge(self, item_id: str) -> bool:
        """Delete knowledge item from memory"""
        try:
            if item_id not in self._storage:
                return False
            
            item = self._storage[item_id]
            
            # Update tag index
            for tag in item.tags:
                if tag in self._tag_index:
                    self._tag_index[tag].discard(item_id)
                    if not self._tag_index[tag]:
                        del self._tag_index[tag]
            
            # Remove item
            del self._storage[item_id]
            self._knowledge_count = len(self._storage)
            
            return True
        except Exception as e:
            logger.error(f"Error deleting knowledge item {item_id}: {e}")
            return False


# Plugin factory function
def create_knowledge_base(kb_type: str, config: Optional[Dict[str, Any]] = None) -> KnowledgeBasePlugin:
    """
    Factory function to create knowledge base plugins.
    
    Args:
        kb_type: Type of knowledge base to create
        config: Configuration for the knowledge base
        
    Returns:
        KnowledgeBasePlugin instance
    """
    knowledge_bases = {
        'memory': InMemoryKnowledgeBase,
    }
    
    if kb_type not in knowledge_bases:
        raise ValueError(f"Unknown knowledge base type: {kb_type}")
    
    return knowledge_bases[kb_type](config)


if __name__ == "__main__":
    # Test the knowledge base plugin base classes
    print("🧠 Knowledge Base Plugin Base Classes Test")
    print("=" * 50)
    
    # Create test knowledge base
    kb = create_knowledge_base('memory')
    
    # Initialize and connect
    init_success = kb.initialize()
    connect_success = kb.connect()
    print(f"✅ KB initialized: {init_success}, connected: {connect_success}")
    
    # Test metadata
    metadata = kb.get_metadata()
    print(f"📋 KB: {metadata.name} v{metadata.version}")
    print(f"💾 Backend: {metadata.storage_backend}")
    
    # Create test knowledge items
    items = [
        KnowledgeItem(
            id="fact1",
            content="The capital of France is Paris",
            knowledge_type=KnowledgeType.FACTUAL,
            format=KnowledgeFormat.TEXT,
            tags=["geography", "france", "capital"],
            confidence=0.95
        ),
        KnowledgeItem(
            id="rule1",
            content="If X is a bird and X can fly, then X is not a penguin",
            knowledge_type=KnowledgeType.RULE,
            format=KnowledgeFormat.LOGIC,
            tags=["logic", "birds", "reasoning"],
            confidence=0.85
        ),
        KnowledgeItem(
            id="pattern1",
            content="European capitals are often cultural centers",
            knowledge_type=KnowledgeType.PATTERN,
            format=KnowledgeFormat.TEXT,
            tags=["geography", "europe", "culture"],
            confidence=0.8
        )
    ]
    
    # Store knowledge items
    for item in items:
        success = kb.store_knowledge(item)
        print(f"📝 Stored {item.id}: {success}")
    
    # Test retrieval
    retrieved = kb.retrieve_knowledge("fact1")
    print(f"🔍 Retrieved fact1: {'✅' if retrieved else '❌'}")
    
    # Test queries
    query_result = kb.query_knowledge("Paris", QueryType.FUZZY_SEARCH)
    print(f"🔎 Query 'Paris': {len(query_result.items)} results")
    
    # Test tag search
    tag_result = kb.query_knowledge("tags:geography", QueryType.PATTERN_MATCH)
    print(f"🏷️ Tag query 'geography': {len(tag_result.items)} results")
    
    # Test relationships
    relationship_added = kb.add_relationship("fact1", "pattern1", "related")
    print(f"🔗 Added relationship: {relationship_added}")
    
    # Test related knowledge
    related = kb.get_related_knowledge("fact1")
    print(f"🔗 Related to fact1: {len(related)} items")
    
    # Test statistics
    stats = kb.get_statistics()
    print(f"📊 Total items: {stats['total_items']}")
    
    # Shutdown
    kb.disconnect()
    kb.shutdown()
    print("\n✅ Knowledge base plugin test completed!")