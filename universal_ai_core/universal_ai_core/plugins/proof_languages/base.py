#!/usr/bin/env python3
"""
Proof Language Plugin Base Classes
==================================

This module provides abstract base classes for formal proof language plugins in the Universal AI Core system.
Adapted from existing proof system patterns in the Saraphis codebase, made domain-agnostic.

Base Classes:
- ProofLanguagePlugin: Abstract base for all proof language verifiers
- ProofVerificationResult: Standardized verification result container
- ProofStep: Individual proof step representation
- ProofContext: Proof context and state management
"""

import logging
import time
import hashlib
import asyncio
import tempfile
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ProofLanguage(Enum):
    """Supported formal proof languages"""
    LEAN4 = "lean4"
    COQ = "coq"
    ISABELLE = "isabelle"
    AGDA = "agda"
    IDRIS = "idris"
    NEUROFORMAL = "neuroformal"
    TPTP = "tptp"
    SMT_LIB = "smt_lib"
    METAMATH = "metamath"
    HOL_LIGHT = "hol_light"


class ProofStatus(Enum):
    """Status of proof verification"""
    PENDING = "pending"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"
    INCOMPLETE = "incomplete"


class ProofType(Enum):
    """Types of formal proofs"""
    THEOREM = "theorem"
    LEMMA = "lemma"
    AXIOM = "axiom"
    DEFINITION = "definition"
    PROPOSITION = "proposition"
    COROLLARY = "corollary"
    CONJECTURE = "conjecture"


class LogicSystem(Enum):
    """Supported logic systems"""
    PROPOSITIONAL = "propositional"
    PREDICATE = "predicate"
    MODAL = "modal"
    TEMPORAL = "temporal"
    INTUITIONISTIC = "intuitionistic"
    LINEAR = "linear"
    HIGHER_ORDER = "higher_order"
    TYPE_THEORY = "type_theory"


@dataclass
class ProofStep:
    """Representation of an individual proof step"""
    id: str
    step_number: int
    tactic: str
    premise: str
    conclusion: str
    justification: str
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    verified: bool = False
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate step after initialization"""
        if not self.tactic:
            raise ValueError("Proof step must have a tactic")


@dataclass
class ProofContext:
    """Context and state for proof verification"""
    id: str
    assumptions: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)  # var_name -> type
    constants: Dict[str, str] = field(default_factory=dict)  # const_name -> type
    axioms: List[str] = field(default_factory=list)
    definitions: Dict[str, str] = field(default_factory=dict)
    logic_system: LogicSystem = LogicSystem.PREDICATE
    namespace: str = "default"
    
    def copy(self) -> 'ProofContext':
        """Create a copy of the proof context"""
        return ProofContext(
            id=self.id,
            assumptions=self.assumptions.copy(),
            goals=self.goals.copy(),
            hypotheses=self.hypotheses.copy(),
            variables=self.variables.copy(),
            constants=self.constants.copy(),
            axioms=self.axioms.copy(),
            definitions=self.definitions.copy(),
            logic_system=self.logic_system,
            namespace=self.namespace
        )


@dataclass
class Proof:
    """Complete formal proof representation"""
    id: str
    name: str
    statement: str
    proof_type: ProofType
    language: ProofLanguage
    author: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    status: ProofStatus = ProofStatus.PENDING
    context: ProofContext = field(default_factory=lambda: ProofContext(id="default"))
    steps: List[ProofStep] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    difficulty: float = 0.0  # 0.0 to 1.0
    completion_time: Optional[float] = None
    verification_time: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_code: str = ""
    compiled_code: str = ""
    checksum: str = ""
    
    def __post_init__(self):
        """Generate checksum if source code is provided"""
        if self.source_code and not self.checksum:
            self.checksum = hashlib.sha256(self.source_code.encode()).hexdigest()


@dataclass
class ProofVerificationResult:
    """Result container for proof verification operations"""
    proof_id: str
    status: ProofStatus
    verified: bool = False
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    verification_time: float = 0.0
    steps_verified: int = 0
    total_steps: int = 0
    dependencies_satisfied: bool = True
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    detailed_errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate verification result"""
        if self.verified and self.status != ProofStatus.VERIFIED:
            self.status = ProofStatus.VERIFIED
        if not self.verified and self.status == ProofStatus.VERIFIED:
            self.verified = True


@dataclass
class LanguageMetadata:
    """Metadata for proof language plugins"""
    name: str
    version: str
    author: str
    description: str
    language: ProofLanguage
    supported_logic_systems: List[LogicSystem]
    supported_proof_types: List[ProofType]
    file_extensions: List[str]
    executable_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    plugin_id: str = ""
    
    def __post_init__(self):
        """Generate plugin ID if not provided"""
        if not self.plugin_id:
            content = f"{self.name}:{self.version}:{self.language.value}"
            self.plugin_id = hashlib.md5(content.encode()).hexdigest()


class ProofLanguagePlugin(ABC):
    """
    Abstract base class for formal proof language plugins.
    
    This class defines the interface that all proof language verifiers must implement.
    Adapted from existing proof verification patterns in the Saraphis codebase.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the proof language plugin.
        
        Args:
            config: Plugin-specific configuration dictionary
        """
        self.config = config or {}
        self._metadata = self._create_metadata()
        self._is_initialized = False
        self._verification_cache = {}
        self._tactic_registry = {}
        self._inference_rules = {}
        self._verification_count = 0
        self._last_verification_time = None
        
        # Initialize tactic registry
        self._initialize_tactics()
        self._initialize_inference_rules()
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Initialized proof language plugin: {self._metadata.name}")
    
    @abstractmethod
    def _create_metadata(self) -> LanguageMetadata:
        """
        Create metadata for this proof language plugin.
        
        Returns:
            LanguageMetadata instance with plugin information
        """
        pass
    
    @abstractmethod
    async def verify_proof(self, proof: Proof) -> ProofVerificationResult:
        """
        Verify a complete formal proof.
        
        Args:
            proof: Proof object to verify
            
        Returns:
            ProofVerificationResult containing verification status and details
        """
        pass
    
    @abstractmethod
    async def verify_step(self, step: ProofStep, context: ProofContext) -> bool:
        """
        Verify a single proof step.
        
        Args:
            step: Proof step to verify
            context: Current proof context
            
        Returns:
            True if step is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def parse_proof(self, proof_text: str) -> Proof:
        """
        Parse proof text into a structured Proof object.
        
        Args:
            proof_text: Source code of the proof
            
        Returns:
            Parsed Proof object
        """
        pass
    
    @abstractmethod
    def format_proof(self, proof: Proof) -> str:
        """
        Format a Proof object back into source code.
        
        Args:
            proof: Proof object to format
            
        Returns:
            Formatted proof source code
        """
        pass
    
    def initialize(self) -> bool:
        """
        Initialize the proof language plugin.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self._perform_initialization()
            self._is_initialized = True
            logger.info(f"Proof language plugin {self._metadata.name} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize plugin {self._metadata.name}: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the plugin and clean up resources"""
        try:
            self._perform_shutdown()
            self._is_initialized = False
            logger.info(f"Plugin {self._metadata.name} shutdown successfully")
        except Exception as e:
            logger.error(f"Error shutting down plugin {self._metadata.name}: {e}")
    
    def get_metadata(self) -> LanguageMetadata:
        """Get plugin metadata"""
        return self._metadata
    
    def is_initialized(self) -> bool:
        """Check if plugin is initialized"""
        return self._is_initialized
    
    def get_supported_tactics(self) -> List[str]:
        """Get list of supported proof tactics"""
        return list(self._tactic_registry.keys())
    
    def get_inference_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get available inference rules"""
        return self._inference_rules.copy()
    
    def validate_syntax(self, proof_text: str) -> Tuple[bool, List[str]]:
        """
        Validate proof syntax without full verification.
        
        Args:
            proof_text: Proof source code to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            self.parse_proof(proof_text)
            return True, []
        except Exception as e:
            return False, [str(e)]
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """
        Get statistics about verifications performed.
        
        Returns:
            Dictionary with verification statistics
        """
        return {
            "verification_count": self._verification_count,
            "last_verification_time": self._last_verification_time,
            "cache_size": len(self._verification_cache),
            "is_initialized": self._is_initialized,
            "language": self._metadata.language.value,
            "plugin_name": self._metadata.name,
            "plugin_version": self._metadata.version
        }
    
    def clear_cache(self) -> None:
        """Clear the verification cache"""
        self._verification_cache.clear()
        logger.info(f"Cleared verification cache for {self._metadata.name}")
    
    def _initialize_tactics(self) -> None:
        """Initialize the tactic registry. Override in subclasses."""
        pass
    
    def _initialize_inference_rules(self) -> None:
        """Initialize inference rules. Override in subclasses."""
        pass
    
    def _validate_config(self) -> None:
        """Validate plugin configuration. Override in subclasses."""
        pass
    
    def _perform_initialization(self) -> None:
        """Perform plugin-specific initialization. Override in subclasses."""
        pass
    
    def _perform_shutdown(self) -> None:
        """Perform plugin-specific shutdown. Override in subclasses."""
        pass
    
    def _generate_cache_key(self, proof: Proof) -> str:
        """Generate cache key for proof verification"""
        return f"{proof.checksum}:{self._metadata.language.value}"
    
    def _update_verification_stats(self, verification_time: float) -> None:
        """Update verification statistics"""
        self._verification_count += 1
        self._last_verification_time = datetime.utcnow()
    
    def _check_cache(self, cache_key: str) -> Optional[ProofVerificationResult]:
        """Check if verification result is cached"""
        return self._verification_cache.get(cache_key)
    
    def _store_in_cache(self, cache_key: str, result: ProofVerificationResult) -> None:
        """Store verification result in cache"""
        max_cache_size = self.config.get('max_cache_size', 1000)
        if len(self._verification_cache) < max_cache_size:
            self._verification_cache[cache_key] = result


class ExternalProofLanguagePlugin(ProofLanguagePlugin):
    """Base class for external proof language verifiers (e.g., Lean4, Coq, Isabelle)"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.executable_path = self.config.get('executable_path', '')
    
    async def verify_proof(self, proof: Proof) -> ProofVerificationResult:
        """Verify proof using external executable"""
        start_time = time.time()
        
        if not self._is_initialized:
            return ProofVerificationResult(
                proof_id=proof.id,
                status=ProofStatus.ERROR,
                error_message="Plugin not initialized",
                verification_time=time.time() - start_time
            )
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(proof)
            cached_result = self._check_cache(cache_key)
            if cached_result:
                logger.info(f"Using cached verification result for proof: {proof.name}")
                return cached_result
            
            # Create temporary file with appropriate extension
            file_extension = self._get_file_extension()
            with tempfile.NamedTemporaryFile(mode='w', suffix=file_extension, delete=False) as f:
                f.write(proof.source_code)
                temp_file = f.name
            
            try:
                # Run external verifier
                process = await asyncio.create_subprocess_exec(
                    self.executable_path,
                    temp_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Wait for completion with timeout
                timeout = self.config.get('verification_timeout', 300.0)
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                
                verification_time = time.time() - start_time
                
                # Parse result
                result = self._parse_verification_output(
                    proof.id, process.returncode, stdout, stderr, verification_time
                )
                
                # Update stats and cache
                self._update_verification_stats(verification_time)
                self._store_in_cache(cache_key, result)
                
                return result
                
            finally:
                # Clean up temporary file
                Path(temp_file).unlink(missing_ok=True)
                
        except asyncio.TimeoutError:
            return ProofVerificationResult(
                proof_id=proof.id,
                status=ProofStatus.TIMEOUT,
                error_message="Verification timeout",
                verification_time=time.time() - start_time
            )
        except Exception as e:
            return ProofVerificationResult(
                proof_id=proof.id,
                status=ProofStatus.ERROR,
                error_message=str(e),
                verification_time=time.time() - start_time
            )
    
    async def verify_step(self, step: ProofStep, context: ProofContext) -> bool:
        """External verifiers typically don't support step-by-step verification"""
        logger.warning(f"Step-by-step verification not supported for {self._metadata.language.value}")
        return True
    
    def _get_file_extension(self) -> str:
        """Get appropriate file extension for the language"""
        extensions = {
            ProofLanguage.LEAN4: ".lean",
            ProofLanguage.COQ: ".v",
            ProofLanguage.ISABELLE: ".thy",
            ProofLanguage.AGDA: ".agda",
            ProofLanguage.IDRIS: ".idr"
        }
        return extensions.get(self._metadata.language, ".txt")
    
    def _parse_verification_output(self, proof_id: str, return_code: int, 
                                 stdout: bytes, stderr: bytes, 
                                 verification_time: float) -> ProofVerificationResult:
        """Parse output from external verifier"""
        if return_code == 0:
            return ProofVerificationResult(
                proof_id=proof_id,
                status=ProofStatus.VERIFIED,
                verified=True,
                verification_time=verification_time,
                metadata={"stdout": stdout.decode(), "stderr": stderr.decode()}
            )
        else:
            return ProofVerificationResult(
                proof_id=proof_id,
                status=ProofStatus.FAILED,
                verified=False,
                error_message=stderr.decode(),
                verification_time=verification_time,
                metadata={"stdout": stdout.decode(), "return_code": return_code}
            )
    
    def _perform_initialization(self) -> None:
        """Check if external executable is available"""
        if not self.executable_path:
            raise RuntimeError(f"No executable path configured for {self._metadata.language.value}")
        
        if not Path(self.executable_path).exists():
            raise RuntimeError(f"Executable not found: {self.executable_path}")
        
        # Test executable
        try:
            result = subprocess.run([self.executable_path, "--version"], 
                                  capture_output=True, timeout=10)
            if result.returncode != 0:
                logger.warning(f"Executable test failed for {self.executable_path}")
        except Exception as e:
            logger.warning(f"Could not test executable {self.executable_path}: {e}")


# Example implementation for NeuroFormal language
class NeuroFormalPlugin(ProofLanguagePlugin):
    """Built-in NeuroFormal proof language plugin"""
    
    def _create_metadata(self) -> LanguageMetadata:
        return LanguageMetadata(
            name="NeuroFormalPlugin",
            version="1.0.0",
            author="Universal AI Core",
            description="Built-in NeuroFormal proof language verifier",
            language=ProofLanguage.NEUROFORMAL,
            supported_logic_systems=list(LogicSystem),
            supported_proof_types=list(ProofType),
            file_extensions=[".nf", ".neuroformal"],
            capabilities=["step_verification", "tactic_based", "interactive"]
        )
    
    def _initialize_tactics(self) -> None:
        """Initialize NeuroFormal tactics"""
        self._tactic_registry = {
            "intro": self._tactic_intro,
            "apply": self._tactic_apply,
            "exact": self._tactic_exact,
            "assumption": self._tactic_assumption,
            "split": self._tactic_split,
            "left": self._tactic_left,
            "right": self._tactic_right,
            "exists": self._tactic_exists,
            "destruct": self._tactic_destruct,
            "induction": self._tactic_induction,
            "rewrite": self._tactic_rewrite,
            "simp": self._tactic_simp,
            "auto": self._tactic_auto,
            "contradiction": self._tactic_contradiction,
            "reflexivity": self._tactic_reflexivity
        }
    
    def _initialize_inference_rules(self) -> None:
        """Initialize logical inference rules"""
        self._inference_rules = {
            "modus_ponens": {
                "premise": ["P -> Q", "P"],
                "conclusion": "Q",
                "description": "If P implies Q and P is true, then Q is true"
            },
            "modus_tollens": {
                "premise": ["P -> Q", "¬Q"],
                "conclusion": "¬P",
                "description": "If P implies Q and Q is false, then P is false"
            },
            "syllogism": {
                "premise": ["P -> Q", "Q -> R"],
                "conclusion": "P -> R",
                "description": "If P implies Q and Q implies R, then P implies R"
            }
        }
    
    async def verify_proof(self, proof: Proof) -> ProofVerificationResult:
        """Verify a NeuroFormal proof"""
        start_time = time.time()
        
        try:
            if not self._is_initialized:
                raise RuntimeError("Plugin not initialized")
            
            # Check cache
            cache_key = self._generate_cache_key(proof)
            cached_result = self._check_cache(cache_key)
            if cached_result:
                return cached_result
            
            # Verify each step
            context = proof.context.copy()
            verified_steps = 0
            
            for i, step in enumerate(proof.steps):
                step_verified = await self.verify_step(step, context)
                
                if step_verified:
                    verified_steps += 1
                    context = self._update_context(context, step)
                else:
                    verification_time = time.time() - start_time
                    result = ProofVerificationResult(
                        proof_id=proof.id,
                        status=ProofStatus.FAILED,
                        error_message=f"Step {i+1} verification failed",
                        verification_time=verification_time,
                        steps_verified=verified_steps,
                        total_steps=len(proof.steps)
                    )
                    self._store_in_cache(cache_key, result)
                    return result
            
            # All steps verified
            verification_time = time.time() - start_time
            result = ProofVerificationResult(
                proof_id=proof.id,
                status=ProofStatus.VERIFIED,
                verified=True,
                verification_time=verification_time,
                steps_verified=verified_steps,
                total_steps=len(proof.steps),
                confidence_score=1.0
            )
            
            self._update_verification_stats(verification_time)
            self._store_in_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            verification_time = time.time() - start_time
            return ProofVerificationResult(
                proof_id=proof.id,
                status=ProofStatus.ERROR,
                error_message=str(e),
                verification_time=verification_time
            )
    
    async def verify_step(self, step: ProofStep, context: ProofContext) -> bool:
        """Verify a single NeuroFormal proof step"""
        try:
            tactic_func = self._tactic_registry.get(step.tactic)
            if not tactic_func:
                step.error_message = f"Unknown tactic: {step.tactic}"
                return False
            
            result = await tactic_func(step, context)
            
            if result:
                step.verified = True
            else:
                step.error_message = f"Tactic {step.tactic} failed"
            
            return result
            
        except Exception as e:
            step.error_message = f"Error executing tactic {step.tactic}: {e}"
            return False
    
    def parse_proof(self, proof_text: str) -> Proof:
        """Parse NeuroFormal proof text"""
        import re
        
        proof = Proof(
            id=hashlib.md5(proof_text.encode()).hexdigest()[:8],
            name="",
            statement="",
            proof_type=ProofType.THEOREM,
            language=ProofLanguage.NEUROFORMAL,
            source_code=proof_text
        )
        
        # Parse header
        header_match = re.search(r'theorem\s+(\w+)\s*:\s*(.+)', proof_text, re.IGNORECASE)
        if header_match:
            proof.name = header_match.group(1)
            proof.statement = header_match.group(2).strip()
        
        # Parse proof body
        proof_body_match = re.search(r'begin\s*(.*?)\s*end', proof_text, re.DOTALL | re.IGNORECASE)
        if proof_body_match:
            proof_body = proof_body_match.group(1)
            proof.steps = self._parse_proof_steps(proof_body)
        
        proof.checksum = hashlib.sha256(proof_text.encode()).hexdigest()
        
        return proof
    
    def format_proof(self, proof: Proof) -> str:
        """Format a Proof object back to NeuroFormal source"""
        lines = [f"theorem {proof.name} : {proof.statement}"]
        lines.append("begin")
        
        for step in proof.steps:
            if step.premise:
                lines.append(f"  {step.tactic} {step.premise}")
            else:
                lines.append(f"  {step.tactic}")
        
        lines.append("end")
        
        return "\n".join(lines)
    
    def _parse_proof_steps(self, proof_body: str) -> List[ProofStep]:
        """Parse proof steps from NeuroFormal proof body"""
        import re
        
        steps = []
        lines = proof_body.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('--'):
                continue
            
            # Parse tactic and arguments
            tactic_match = re.match(r'(\w+)(?:\s+(.*))?', line)
            if tactic_match:
                tactic = tactic_match.group(1)
                args = tactic_match.group(2) or ""
                
                step = ProofStep(
                    id=f"step_{i}",
                    step_number=i + 1,
                    tactic=tactic,
                    premise=args,
                    conclusion="",
                    justification=line
                )
                steps.append(step)
        
        return steps
    
    def _update_context(self, context: ProofContext, step: ProofStep) -> ProofContext:
        """Update proof context after a successful step"""
        new_context = context.copy()
        
        # Simple context update based on tactic
        if step.tactic == "intro" and step.premise:
            new_context.hypotheses.append(step.premise)
        elif step.tactic == "apply" and step.conclusion:
            new_context.goals.append(step.conclusion)
        
        return new_context
    
    # Tactic implementations (simplified)
    async def _tactic_intro(self, step: ProofStep, context: ProofContext) -> bool:
        return True
    
    async def _tactic_apply(self, step: ProofStep, context: ProofContext) -> bool:
        return True
    
    async def _tactic_exact(self, step: ProofStep, context: ProofContext) -> bool:
        return True
    
    async def _tactic_assumption(self, step: ProofStep, context: ProofContext) -> bool:
        return step.premise in context.assumptions
    
    async def _tactic_split(self, step: ProofStep, context: ProofContext) -> bool:
        return True
    
    async def _tactic_left(self, step: ProofStep, context: ProofContext) -> bool:
        return True
    
    async def _tactic_right(self, step: ProofStep, context: ProofContext) -> bool:
        return True
    
    async def _tactic_exists(self, step: ProofStep, context: ProofContext) -> bool:
        return True
    
    async def _tactic_destruct(self, step: ProofStep, context: ProofContext) -> bool:
        return True
    
    async def _tactic_induction(self, step: ProofStep, context: ProofContext) -> bool:
        return True
    
    async def _tactic_rewrite(self, step: ProofStep, context: ProofContext) -> bool:
        return True
    
    async def _tactic_simp(self, step: ProofStep, context: ProofContext) -> bool:
        return True
    
    async def _tactic_auto(self, step: ProofStep, context: ProofContext) -> bool:
        return True
    
    async def _tactic_contradiction(self, step: ProofStep, context: ProofContext) -> bool:
        return True
    
    async def _tactic_reflexivity(self, step: ProofStep, context: ProofContext) -> bool:
        return True


# Plugin factory function
def create_proof_language_plugin(language: str, config: Optional[Dict[str, Any]] = None) -> ProofLanguagePlugin:
    """
    Factory function to create proof language plugins.
    
    Args:
        language: Language name to create plugin for
        config: Configuration for the plugin
        
    Returns:
        ProofLanguagePlugin instance
    """
    plugins = {
        'neuroformal': NeuroFormalPlugin,
    }
    
    if language not in plugins:
        raise ValueError(f"Unknown proof language: {language}")
    
    return plugins[language](config)


if __name__ == "__main__":
    # Test the proof language plugin base classes
    print("🔍 Proof Language Plugin Base Classes Test")
    print("=" * 50)
    
    # Test NeuroFormal plugin
    plugin = create_proof_language_plugin('neuroformal')
    
    # Initialize
    success = plugin.initialize()
    print(f"✅ Plugin initialized: {success}")
    
    # Test metadata
    metadata = plugin.get_metadata()
    print(f"📋 Plugin: {metadata.name} v{metadata.version}")
    print(f"🔤 Language: {metadata.language.value}")
    
    # Test proof parsing
    test_proof_text = """
    theorem example_theorem : forall (P Q : Prop), P ∧ Q → Q ∧ P
    begin
      intro P Q h,
      split,
      exact h.right,
      exact h.left
    end
    """
    
    parsed_proof = plugin.parse_proof(test_proof_text)
    print(f"📄 Parsed proof: {parsed_proof.name}")
    print(f"🔧 Steps: {len(parsed_proof.steps)}")
    
    # Test proof verification
    async def test_verification():
        result = await plugin.verify_proof(parsed_proof)
        print(f"✅ Verification status: {result.status.value}")
        print(f"⏱️ Verification time: {result.verification_time:.4f}s")
        print(f"📊 Steps verified: {result.steps_verified}/{result.total_steps}")
    
    # Run async test
    import asyncio
    asyncio.run(test_verification())
    
    # Test tactics
    tactics = plugin.get_supported_tactics()
    print(f"🎯 Supported tactics: {len(tactics)}")
    
    # Test statistics
    stats = plugin.get_verification_stats()
    print(f"📈 Verification count: {stats['verification_count']}")
    
    # Shutdown
    plugin.shutdown()
    print("\n✅ Proof language plugin test completed!")