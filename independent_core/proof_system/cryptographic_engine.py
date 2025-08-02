"""
Cryptographic Proof Engine
Generates cryptographic proofs for data integrity and authenticity
"""

import logging
import time
import hashlib
import hmac
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import secrets

logger = logging.getLogger(__name__)


class CryptographicProofEngine:
    """Cryptographic proof engine for data integrity validation"""
    
    def __init__(self):
        """Initialize cryptographic engine"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.key = self._generate_key()
        
    def _generate_key(self) -> bytes:
        """Generate cryptographic key for HMAC operations"""
        return secrets.token_bytes(32)  # 256-bit key
        
    def generate_proof(self, data: Dict[str, Any], previous_hash: str = None) -> Dict[str, Any]:
        """Generate cryptographic proof for data"""
        start_time = time.time()
        
        # Serialize data for hashing
        serialized_data = self._serialize_data(data)
        
        # Generate timestamp
        timestamp = datetime.now().isoformat()
        
        # Generate nonce for uniqueness
        nonce = secrets.token_hex(16)
        
        # Create hash input
        hash_input = {
            'data': serialized_data,
            'timestamp': timestamp,
            'nonce': nonce,
            'previous_hash': previous_hash or '0' * 64
        }
        
        # Generate SHA-256 hash
        data_hash = self._generate_hash(hash_input)
        
        # Generate HMAC for authentication
        hmac_signature = self._generate_hmac(data_hash)
        
        proof_time = time.time() - start_time
        
        return {
            'hash': data_hash,
            'hmac': hmac_signature,
            'timestamp': timestamp,
            'nonce': nonce,
            'previous_hash': previous_hash or '0' * 64,
            'algorithm': 'SHA-256',
            'generation_time_ms': proof_time * 1000
        }
        
    def verify_proof(self, data: Dict[str, Any], proof: Dict[str, Any]) -> bool:
        """Verify cryptographic proof for data"""
        try:
            # Recreate hash input
            serialized_data = self._serialize_data(data)
            hash_input = {
                'data': serialized_data,
                'timestamp': proof['timestamp'],
                'nonce': proof['nonce'],
                'previous_hash': proof['previous_hash']
            }
            
            # Regenerate hash
            expected_hash = self._generate_hash(hash_input)
            
            # Verify hash matches
            if expected_hash != proof['hash']:
                return False
                
            # Verify HMAC
            expected_hmac = self._generate_hmac(expected_hash)
            return hmac.compare_digest(expected_hmac, proof['hmac'])
            
        except Exception as e:
            self.logger.error(f"Proof verification failed: {str(e)}")
            return False
            
    def _serialize_data(self, data: Dict[str, Any]) -> str:
        """Serialize data for consistent hashing"""
        # Convert to JSON with sorted keys for consistency
        return json.dumps(data, sort_keys=True, separators=(',', ':'))
        
    def _generate_hash(self, data: Dict[str, Any]) -> str:
        """Generate SHA-256 hash of data"""
        serialized = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()
        
    def _generate_hmac(self, data: str) -> str:
        """Generate HMAC signature"""
        return hmac.new(
            self.key,
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
    def verify_chain(self, chain: List[Dict[str, Any]]) -> bool:
        """Verify blockchain-style chain integrity"""
        if not chain:
            return True
            
        for i, block in enumerate(chain):
            # Verify individual block
            if not self.verify_proof(block['data'], block['proof']):
                self.logger.error(f"Block {i} failed proof verification")
                return False
                
            # Verify chain linkage
            if i > 0:
                expected_prev = chain[i-1]['proof']['hash']
                actual_prev = block['proof']['previous_hash']
                if expected_prev != actual_prev:
                    self.logger.error(f"Block {i} chain linkage broken")
                    return False
                    
        return True
        
    def build_merkle_tree(self, transactions: List[Dict[str, Any]]) -> str:
        """Build Merkle tree root hash for batch of transactions"""
        if not transactions:
            return '0' * 64
            
        # Hash all transactions
        hashes = []
        for tx in transactions:
            tx_hash = self._generate_hash(tx)
            hashes.append(tx_hash)
            
        # Build tree bottom-up
        while len(hashes) > 1:
            next_level = []
            
            # Process pairs
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    # Hash pair
                    combined = hashes[i] + hashes[i + 1]
                    combined_hash = hashlib.sha256(combined.encode()).hexdigest()
                    next_level.append(combined_hash)
                else:
                    # Odd number, duplicate last hash
                    combined = hashes[i] + hashes[i]
                    combined_hash = hashlib.sha256(combined.encode()).hexdigest()
                    next_level.append(combined_hash)
                    
            hashes = next_level
            
        return hashes[0] if hashes else '0' * 64
        
    def get_merkle_proof(self, transactions: List[Dict[str, Any]], target_index: int) -> List[str]:
        """Get Merkle proof path for specific transaction"""
        if not transactions or target_index >= len(transactions):
            return []
            
        # Build tree with proof path tracking
        hashes = [self._generate_hash(tx) for tx in transactions]
        proof_path = []
        current_index = target_index
        
        while len(hashes) > 1:
            next_level = []
            
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    # Add sibling to proof path if this is our target's level
                    if i == current_index or i + 1 == current_index:
                        sibling_index = i + 1 if i == current_index else i
                        if sibling_index < len(hashes):
                            proof_path.append(hashes[sibling_index])
                            
                    combined = hashes[i] + hashes[i + 1]
                    combined_hash = hashlib.sha256(combined.encode()).hexdigest()
                    next_level.append(combined_hash)
                else:
                    combined = hashes[i] + hashes[i]
                    combined_hash = hashlib.sha256(combined.encode()).hexdigest()
                    next_level.append(combined_hash)
                    
            # Update index for next level
            current_index = current_index // 2
            hashes = next_level
            
        return proof_path
        
    def generate_key_pair(self) -> Tuple[str, str]:
        """Generate simulated key pair for digital signatures"""
        # Simplified key generation for demonstration
        private_key = secrets.token_hex(32)
        public_key = hashlib.sha256(private_key.encode()).hexdigest()
        return private_key, public_key
        
    def sign_data(self, data: Dict[str, Any], private_key: str) -> str:
        """Generate digital signature for data"""
        serialized_data = self._serialize_data(data)
        
        # Simplified signing using HMAC with private key
        signature = hmac.new(
            private_key.encode(),
            serialized_data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
        
    def verify_signature(self, data: Dict[str, Any], signature: str, public_key: str) -> bool:
        """Verify digital signature"""
        try:
            # For this simplified implementation, we'll regenerate private key from public key
            # In a real implementation, this would use proper public key cryptography
            serialized_data = self._serialize_data(data)
            
            # This is a simplified verification - not cryptographically secure
            expected_signature = hmac.new(
                public_key.encode(),
                serialized_data.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            self.logger.error(f"Signature verification failed: {str(e)}")
            return False
            
    def generate_batch_proof(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate batch proof for multiple transactions"""
        start_time = time.time()
        
        # Generate Merkle root
        merkle_root = self.build_merkle_tree(transactions)
        
        # Generate batch metadata
        batch_metadata = {
            'transaction_count': len(transactions),
            'batch_timestamp': datetime.now().isoformat(),
            'batch_id': secrets.token_hex(16)
        }
        
        # Generate proof for batch metadata
        batch_proof = self.generate_proof(batch_metadata)
        
        # Include Merkle root in proof
        batch_proof['merkle_root'] = merkle_root
        batch_proof['batch_metadata'] = batch_metadata
        batch_proof['generation_time_ms'] = (time.time() - start_time) * 1000
        
        return batch_proof
        
    def verify_batch_proof(self, transactions: List[Dict[str, Any]], proof: Dict[str, Any]) -> bool:
        """Verify batch proof for multiple transactions"""
        try:
            # Verify Merkle root
            expected_merkle = self.build_merkle_tree(transactions)
            if expected_merkle != proof['merkle_root']:
                return False
                
            # Verify batch metadata proof
            batch_metadata = proof['batch_metadata']
            return self.verify_proof(batch_metadata, proof)
            
        except Exception as e:
            self.logger.error(f"Batch proof verification failed: {str(e)}")
            return False
            
    def generate_proof_for_transaction(self, transaction: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive cryptographic proof for single transaction"""
        # Generate basic proof
        basic_proof = self.generate_proof(transaction)
        
        # Add transaction-specific elements
        transaction_id = transaction.get('transaction_id', secrets.token_hex(8))
        
        # Generate digital signature
        private_key, public_key = self.generate_key_pair()
        signature = self.sign_data(transaction, private_key)
        
        comprehensive_proof = {
            'engine_type': 'cryptographic',
            'transaction_id': transaction_id,
            'basic_proof': basic_proof,
            'digital_signature': {
                'signature': signature,
                'public_key': public_key,
                'algorithm': 'HMAC-SHA256'
            },
            'proof_metadata': {
                'engine_version': '1.0.0',
                'generation_timestamp': datetime.now().isoformat(),
                'cryptographic_strength': 'SHA-256/HMAC'
            }
        }
        
        if context:
            comprehensive_proof['context'] = context
            
        return comprehensive_proof