"""
Financial Fraud Domain Integration Example
Demonstrates how to integrate the Financial Fraud Domain with the Core Brain System
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# Import core system components (these would come from independent_core)
from dataclasses import dataclass
from enum import Enum

# Import our domain
from domain_registration import (
    FinancialFraudDomain,
    FraudTaskType,
    register_fraud_domain
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Example Task Models (would come from core system)
@dataclass
class BuildTask:
    """Core system BuildTask model"""
    id: str
    type: str  # Will use FraudTaskType values
    action: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    priority: str
    ai_analysis: Optional[Dict[str, Any]] = None
    ai_metadata: Optional[Dict[str, Any]] = None


@dataclass
class TaskContext:
    """Core system TaskContext model"""
    build_id: str
    user_id: str
    session_id: str
    environment: str
    metadata: Dict[str, Any]


# Example Fraud Detection Executor
class FraudDetectionExecutor:
    """
    Example executor for fraud detection tasks
    Would extend TaskExecutor from core system
    """
    
    def __init__(self, domain: FinancialFraudDomain):
        self.domain = domain
        self.name = "FraudDetectionExecutor"
    
    async def execute(self, task: BuildTask, context: TaskContext) -> Dict[str, Any]:
        """Execute a fraud detection task"""
        logger.info(f"Executing fraud detection task: {task.id}")
        
        try:
            # Extract task parameters
            transaction_data = task.parameters.get("transaction_data", {})
            
            # Simulate fraud detection logic
            risk_score = await self._calculate_risk_score(transaction_data)
            fraud_indicators = await self._detect_fraud_patterns(transaction_data)
            compliance_status = await self._check_compliance(transaction_data)
            
            # Update domain metrics
            self.domain.metrics["tasks_processed"] += 1
            if risk_score > self.domain.configuration.alert_threshold:
                self.domain.metrics["fraud_detected"] += 1
            
            result = {
                "task_id": task.id,
                "status": "completed",
                "risk_score": risk_score,
                "fraud_indicators": fraud_indicators,
                "compliance_status": compliance_status,
                "timestamp": datetime.now().isoformat(),
                "alert_generated": risk_score > self.domain.configuration.alert_threshold
            }
            
            # Generate alert if needed
            if result["alert_generated"]:
                await self._generate_alert(task, result)
            
            return result
        
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "task_id": task.id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _calculate_risk_score(self, transaction_data: Dict[str, Any]) -> float:
        """Calculate fraud risk score"""
        # Simulate ML model prediction
        amount = transaction_data.get("amount", 0)
        merchant_risk = transaction_data.get("merchant_risk", 0.5)
        
        # Simple risk calculation (would use actual ML model)
        if amount > 10000:
            base_risk = 0.7
        elif amount > 5000:
            base_risk = 0.5
        else:
            base_risk = 0.3
        
        risk_score = min(base_risk * merchant_risk * 1.5, 1.0)
        return round(risk_score, 3)
    
    async def _detect_fraud_patterns(self, transaction_data: Dict[str, Any]) -> List[str]:
        """Detect fraud patterns in transaction"""
        patterns = []
        
        # Check for suspicious patterns
        if transaction_data.get("country") != transaction_data.get("card_country"):
            patterns.append("cross_border_transaction")
        
        if transaction_data.get("amount", 0) > 5000:
            patterns.append("high_value_transaction")
        
        if transaction_data.get("velocity_check_failed"):
            patterns.append("velocity_limit_exceeded")
        
        return patterns
    
    async def _check_compliance(self, transaction_data: Dict[str, Any]) -> Dict[str, bool]:
        """Check compliance requirements"""
        return {
            "PCI_DSS": True,  # Simplified - would check actual compliance
            "SOX": True,
            "GDPR": transaction_data.get("gdpr_consent", False)
        }
    
    async def _generate_alert(self, task: BuildTask, result: Dict[str, Any]) -> None:
        """Generate fraud alert"""
        alert = {
            "alert_id": f"ALERT-{task.id}",
            "task_id": task.id,
            "risk_score": result["risk_score"],
            "fraud_indicators": result["fraud_indicators"],
            "timestamp": datetime.now().isoformat(),
            "priority": "HIGH" if result["risk_score"] > 0.9 else "MEDIUM"
        }
        
        logger.warning(f"FRAUD ALERT GENERATED: {json.dumps(alert, indent=2)}")
        
        # Would send to notification channels
        if self.domain.configuration.webhook_enabled:
            # Send webhook notification
            pass


# Example Usage and Integration
class FraudDomainIntegration:
    """
    Example integration between Financial Fraud Domain and Core Brain System
    """
    
    def __init__(self):
        self.fraud_domain: Optional[FinancialFraudDomain] = None
        self.executors: Dict[str, Any] = {}
        self.task_queue: List[BuildTask] = []
    
    async def initialize(self):
        """Initialize the fraud domain integration"""
        logger.info("Initializing Financial Fraud Domain Integration")
        
        # Create mock domain registry (would use actual DomainRegistry)
        class MockDomainRegistry:
            def __init__(self):
                self.domains = {}
            
            async def register_domain(self, domain_id, domain, info):
                self.domains[domain_id] = domain
                logger.info(f"Domain registered with registry: {domain_id}")
        
        # Register fraud domain
        registry = MockDomainRegistry()
        self.fraud_domain = await register_fraud_domain(registry)
        
        # Initialize executors
        self.executors[FraudTaskType.TRANSACTION_ANALYSIS.value] = FraudDetectionExecutor(self.fraud_domain)
        
        logger.info("Fraud Domain Integration initialized successfully")
    
    async def process_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a transaction through the fraud detection system"""
        # Create a fraud detection task
        task = BuildTask(
            id=f"TASK-{datetime.now().timestamp()}",
            type=FraudTaskType.TRANSACTION_ANALYSIS.value,
            action="analyze_transaction",
            parameters={"transaction_data": transaction_data},
            dependencies=[],
            priority="HIGH",
            ai_analysis={
                "suggested_checks": ["velocity", "geolocation", "merchant_reputation"],
                "confidence": 0.85
            }
        )
        
        # Create task context
        context = TaskContext(
            build_id=f"BUILD-{datetime.now().timestamp()}",
            user_id="system",
            session_id="fraud-detection-session",
            environment="production",
            metadata={"source": "api", "version": "1.0"}
        )
        
        # Execute the task
        executor = self.executors.get(task.type)
        if executor:
            result = await executor.execute(task, context)
            return result
        else:
            raise ValueError(f"No executor found for task type: {task.type}")
    
    async def run_batch_analysis(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run batch analysis on multiple transactions"""
        logger.info(f"Running batch analysis on {len(transactions)} transactions")
        
        results = []
        for transaction in transactions:
            try:
                result = await self.process_transaction(transaction)
                results.append(result)
                
                # Add small delay to simulate processing
                await asyncio.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error processing transaction: {e}")
                results.append({
                    "status": "error",
                    "error": str(e),
                    "transaction": transaction
                })
        
        return results
    
    async def get_domain_status(self) -> Dict[str, Any]:
        """Get current domain status and metrics"""
        if not self.fraud_domain:
            return {"status": "not_initialized"}
        
        health = await self.fraud_domain.health_check()
        metrics = self.fraud_domain.get_metrics()
        
        return {
            "domain_info": self.fraud_domain.get_info(),
            "health": health,
            "metrics": metrics,
            "executors": list(self.executors.keys())
        }
    
    async def shutdown(self):
        """Shutdown the fraud domain integration"""
        if self.fraud_domain:
            await self.fraud_domain.shutdown()
            logger.info("Fraud Domain Integration shutdown complete")


# Example usage script
async def main():
    """Demonstrate the Financial Fraud Domain integration"""
    
    # Initialize integration
    integration = FraudDomainIntegration()
    await integration.initialize()
    
    # Example transactions to analyze
    test_transactions = [
        {
            "transaction_id": "TXN-001",
            "amount": 15000,
            "currency": "USD",
            "merchant": "Unknown Merchant",
            "merchant_risk": 0.8,
            "country": "US",
            "card_country": "UK",
            "timestamp": datetime.now().isoformat()
        },
        {
            "transaction_id": "TXN-002",
            "amount": 250,
            "currency": "USD",
            "merchant": "Amazon",
            "merchant_risk": 0.1,
            "country": "US",
            "card_country": "US",
            "gdpr_consent": True,
            "timestamp": datetime.now().isoformat()
        },
        {
            "transaction_id": "TXN-003",
            "amount": 8500,
            "currency": "EUR",
            "merchant": "Suspicious Store",
            "merchant_risk": 0.9,
            "country": "RU",
            "card_country": "US",
            "velocity_check_failed": True,
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    # Process transactions
    logger.info("\n=== Processing Test Transactions ===")
    results = await integration.run_batch_analysis(test_transactions)
    
    # Display results
    for i, result in enumerate(results):
        logger.info(f"\nTransaction {i+1} Result:")
        logger.info(json.dumps(result, indent=2))
    
    # Get domain status
    logger.info("\n=== Domain Status ===")
    status = await integration.get_domain_status()
    logger.info(json.dumps({
        "health": status["health"],
        "metrics": status["metrics"]
    }, indent=2))
    
    # Update domain configuration
    logger.info("\n=== Updating Domain Configuration ===")
    integration.fraud_domain.update_configuration({
        "alert_threshold": 0.8,
        "model_threshold": 0.85
    })
    logger.info("Configuration updated")
    
    # Process another transaction with new thresholds
    high_risk_transaction = {
        "transaction_id": "TXN-004",
        "amount": 5000,
        "currency": "USD",
        "merchant": "New Merchant",
        "merchant_risk": 0.85,
        "country": "CN",
        "card_country": "US",
        "timestamp": datetime.now().isoformat()
    }
    
    result = await integration.process_transaction(high_risk_transaction)
    logger.info(f"\nHigh Risk Transaction Result:")
    logger.info(json.dumps(result, indent=2))
    
    # Final metrics
    logger.info("\n=== Final Metrics ===")
    final_metrics = integration.fraud_domain.get_metrics()
    logger.info(json.dumps(final_metrics, indent=2))
    
    # Shutdown
    await integration.shutdown()


if __name__ == "__main__":
    asyncio.run(main())