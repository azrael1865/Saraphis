"""
Comprehensive test suite for CryptographicProofEngine
Tests all methods, edge cases, error handling, and performance
"""

import unittest
import json
import time
import secrets
import hashlib
from unittest.mock import patch, MagicMock
from datetime import datetime
import threading
import hmac
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proof_system.cryptographic_engine import CryptographicProofEngine


class TestCryptographicProofEngine(unittest.TestCase):
    """Test suite for CryptographicProofEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = CryptographicProofEngine()
        
        # Sample test data
        self.sample_data = {
            'transaction_id': 'tx_001',
            'amount': 100.50,
            'sender': 'alice',
            'receiver': 'bob',
            'metadata': {'currency': 'USD'}
        }
        
        self.sample_data_2 = {
            'transaction_id': 'tx_002',
            'amount': 50.25,
            'sender': 'bob',
            'receiver': 'charlie'
        }
        
        self.empty_data = {}
        
        self.complex_data = {
            'nested': {
                'level1': {
                    'level2': {
                        'value': 123,
                        'array': [1, 2, 3]
                    }
                }
            },
            'unicode': 'Hello 世界 🌍',
            'special_chars': '!@#$%^&*()',
            'null_value': None,
            'boolean': True,
            'float': 3.14159
        }
        
    def test_initialization(self):
        """Test engine initialization"""
        engine = CryptographicProofEngine()
        self.assertIsNotNone(engine.key)
        self.assertIsInstance(engine.key, bytes)
        self.assertEqual(len(engine.key), 32)  # 256-bit key
        self.assertIsNotNone(engine.logger)
        
    def test_generate_key(self):
        """Test cryptographic key generation"""
        key1 = self.engine._generate_key()
        key2 = self.engine._generate_key()
        
        self.assertIsInstance(key1, bytes)
        self.assertIsInstance(key2, bytes)
        self.assertEqual(len(key1), 32)
        self.assertEqual(len(key2), 32)
        self.assertNotEqual(key1, key2)  # Keys should be unique
        
    def test_serialize_data(self):
        """Test data serialization"""
        # Test with normal data
        serialized = self.engine._serialize_data(self.sample_data)
        self.assertIsInstance(serialized, str)
        
        # Verify JSON structure
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized, self.sample_data)
        
        # Test with empty data
        empty_serialized = self.engine._serialize_data(self.empty_data)
        self.assertEqual(empty_serialized, '{}')
        
        # Test with complex data
        complex_serialized = self.engine._serialize_data(self.complex_data)
        complex_deserialized = json.loads(complex_serialized)
        self.assertEqual(complex_deserialized, self.complex_data)
        
        # Test consistency - same data should produce same serialization
        serialized1 = self.engine._serialize_data(self.sample_data)
        serialized2 = self.engine._serialize_data(self.sample_data)
        self.assertEqual(serialized1, serialized2)
        
    def test_generate_hash(self):
        """Test hash generation"""
        hash1 = self.engine._generate_hash(self.sample_data)
        
        # Verify hash properties
        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 64)  # SHA-256 produces 64 hex chars
        self.assertTrue(all(c in '0123456789abcdef' for c in hash1))
        
        # Test consistency
        hash2 = self.engine._generate_hash(self.sample_data)
        self.assertEqual(hash1, hash2)
        
        # Test different data produces different hash
        hash3 = self.engine._generate_hash(self.sample_data_2)
        self.assertNotEqual(hash1, hash3)
        
        # Test empty data
        empty_hash = self.engine._generate_hash(self.empty_data)
        self.assertEqual(len(empty_hash), 64)
        
    def test_generate_hmac(self):
        """Test HMAC generation"""
        data = "test_data"
        hmac1 = self.engine._generate_hmac(data)
        
        # Verify HMAC properties
        self.assertIsInstance(hmac1, str)
        self.assertEqual(len(hmac1), 64)  # HMAC-SHA256 produces 64 hex chars
        
        # Test consistency
        hmac2 = self.engine._generate_hmac(data)
        self.assertEqual(hmac1, hmac2)
        
        # Test different data produces different HMAC
        hmac3 = self.engine._generate_hmac("different_data")
        self.assertNotEqual(hmac1, hmac3)
        
        # Test with empty string
        empty_hmac = self.engine._generate_hmac("")
        self.assertEqual(len(empty_hmac), 64)
        
    def test_generate_proof(self):
        """Test proof generation"""
        proof = self.engine.generate_proof(self.sample_data)
        
        # Verify proof structure
        self.assertIsInstance(proof, dict)
        required_fields = ['hash', 'hmac', 'timestamp', 'nonce', 
                          'previous_hash', 'algorithm', 'generation_time_ms']
        for field in required_fields:
            self.assertIn(field, proof)
            
        # Verify field properties
        self.assertEqual(len(proof['hash']), 64)
        self.assertEqual(len(proof['hmac']), 64)
        self.assertEqual(len(proof['nonce']), 32)
        self.assertEqual(proof['algorithm'], 'SHA-256')
        self.assertIsInstance(proof['generation_time_ms'], float)
        self.assertGreaterEqual(proof['generation_time_ms'], 0)
        
        # Test with previous hash
        proof_with_prev = self.engine.generate_proof(
            self.sample_data, 
            previous_hash='a' * 64
        )
        self.assertEqual(proof_with_prev['previous_hash'], 'a' * 64)
        
        # Test without previous hash
        proof_without_prev = self.engine.generate_proof(self.sample_data)
        self.assertEqual(proof_without_prev['previous_hash'], '0' * 64)
        
        # Test uniqueness (nonce should make each proof unique)
        proof1 = self.engine.generate_proof(self.sample_data)
        proof2 = self.engine.generate_proof(self.sample_data)
        self.assertNotEqual(proof1['hash'], proof2['hash'])
        self.assertNotEqual(proof1['nonce'], proof2['nonce'])
        
    def test_verify_proof(self):
        """Test proof verification"""
        # Generate valid proof
        proof = self.engine.generate_proof(self.sample_data)
        
        # Verify valid proof
        self.assertTrue(self.engine.verify_proof(self.sample_data, proof))
        
        # Test with modified data
        modified_data = self.sample_data.copy()
        modified_data['amount'] = 200.00
        self.assertFalse(self.engine.verify_proof(modified_data, proof))
        
        # Test with modified proof hash
        modified_proof = proof.copy()
        modified_proof['hash'] = 'a' * 64
        self.assertFalse(self.engine.verify_proof(self.sample_data, modified_proof))
        
        # Test with modified HMAC
        modified_proof = proof.copy()
        modified_proof['hmac'] = 'b' * 64
        self.assertFalse(self.engine.verify_proof(self.sample_data, modified_proof))
        
        # Test with missing fields
        incomplete_proof = {'hash': proof['hash']}
        self.assertFalse(self.engine.verify_proof(self.sample_data, incomplete_proof))
        
        # Test with invalid proof structure
        self.assertFalse(self.engine.verify_proof(self.sample_data, {}))
        self.assertFalse(self.engine.verify_proof(self.sample_data, None))
        self.assertFalse(self.engine.verify_proof(self.sample_data, "invalid"))
        
    def test_verify_chain(self):
        """Test blockchain chain verification"""
        # Create valid chain
        chain = []
        prev_hash = None
        
        for i in range(3):
            data = {'block': i, 'data': f'block_{i}'}
            proof = self.engine.generate_proof(data, prev_hash)
            chain.append({'data': data, 'proof': proof})
            prev_hash = proof['hash']
            
        # Verify valid chain
        self.assertTrue(self.engine.verify_chain(chain))
        
        # Test empty chain
        self.assertTrue(self.engine.verify_chain([]))
        
        # Test single block chain
        single_chain = [chain[0]]
        self.assertTrue(self.engine.verify_chain(single_chain))
        
        # Test with broken linkage
        broken_chain = chain.copy()
        broken_chain[2]['proof']['previous_hash'] = 'invalid_hash'
        self.assertFalse(self.engine.verify_chain(broken_chain))
        
        # Test with modified data
        modified_chain = chain.copy()
        modified_chain[1]['data']['modified'] = True
        self.assertFalse(self.engine.verify_chain(modified_chain))
        
        # Test with invalid proof
        invalid_chain = chain.copy()
        invalid_chain[0]['proof']['hash'] = 'invalid'
        self.assertFalse(self.engine.verify_chain(invalid_chain))
        
    def test_build_merkle_tree(self):
        """Test Merkle tree construction"""
        # Test with multiple transactions
        transactions = [
            {'tx': 1, 'amount': 10},
            {'tx': 2, 'amount': 20},
            {'tx': 3, 'amount': 30},
            {'tx': 4, 'amount': 40}
        ]
        
        root = self.engine.build_merkle_tree(transactions)
        self.assertIsInstance(root, str)
        self.assertEqual(len(root), 64)
        
        # Test consistency
        root2 = self.engine.build_merkle_tree(transactions)
        self.assertEqual(root, root2)
        
        # Test with different transactions
        transactions2 = transactions.copy()
        transactions2[0]['amount'] = 15
        root3 = self.engine.build_merkle_tree(transactions2)
        self.assertNotEqual(root, root3)
        
        # Test with empty list
        empty_root = self.engine.build_merkle_tree([])
        self.assertEqual(empty_root, '0' * 64)
        
        # Test with single transaction
        single_root = self.engine.build_merkle_tree([transactions[0]])
        self.assertEqual(len(single_root), 64)
        
        # Test with odd number of transactions
        odd_transactions = transactions[:3]
        odd_root = self.engine.build_merkle_tree(odd_transactions)
        self.assertEqual(len(odd_root), 64)
        
        # Test with power of 2 transactions
        power2_transactions = transactions[:4]
        power2_root = self.engine.build_merkle_tree(power2_transactions)
        self.assertEqual(len(power2_root), 64)
        
    def test_get_merkle_proof(self):
        """Test Merkle proof generation"""
        transactions = [
            {'tx': 1}, {'tx': 2}, {'tx': 3}, {'tx': 4}
        ]
        
        # Test proof for each transaction
        for i in range(len(transactions)):
            proof_path = self.engine.get_merkle_proof(transactions, i)
            self.assertIsInstance(proof_path, list)
            # For 4 transactions, proof path should have 2 elements
            self.assertEqual(len(proof_path), 2)
            
        # Test with empty transactions
        empty_proof = self.engine.get_merkle_proof([], 0)
        self.assertEqual(empty_proof, [])
        
        # Test with invalid index
        invalid_proof = self.engine.get_merkle_proof(transactions, 10)
        self.assertEqual(invalid_proof, [])
        
        # Test with negative index
        negative_proof = self.engine.get_merkle_proof(transactions, -1)
        self.assertEqual(negative_proof, [])
        
        # Test with single transaction
        single_proof = self.engine.get_merkle_proof([transactions[0]], 0)
        self.assertEqual(len(single_proof), 0)  # No siblings
        
    def test_generate_key_pair(self):
        """Test key pair generation"""
        private_key, public_key = self.engine.generate_key_pair()
        
        # Verify key properties
        self.assertIsInstance(private_key, str)
        self.assertIsInstance(public_key, str)
        self.assertEqual(len(private_key), 64)  # 32 bytes hex
        self.assertEqual(len(public_key), 64)   # SHA-256 hash
        
        # Test uniqueness
        private_key2, public_key2 = self.engine.generate_key_pair()
        self.assertNotEqual(private_key, private_key2)
        self.assertNotEqual(public_key, public_key2)
        
        # Test deterministic public key from private key
        expected_public = hashlib.sha256(private_key.encode()).hexdigest()
        self.assertEqual(public_key, expected_public)
        
    def test_sign_data(self):
        """Test data signing"""
        private_key, public_key = self.engine.generate_key_pair()
        
        # Sign data
        signature = self.engine.sign_data(self.sample_data, private_key)
        
        # Verify signature properties (new format is "signature:verification")
        self.assertIsInstance(signature, str)
        self.assertIn(':', signature)  # Composite signature format
        parts = signature.split(':')
        self.assertEqual(len(parts), 2)
        self.assertEqual(len(parts[0]), 64)  # Signature part
        self.assertEqual(len(parts[1]), 64)  # Verification part
        
        # Test consistency
        signature2 = self.engine.sign_data(self.sample_data, private_key)
        self.assertEqual(signature, signature2)
        
        # Test different data produces different signature
        signature3 = self.engine.sign_data(self.sample_data_2, private_key)
        self.assertNotEqual(signature, signature3)
        
        # Test different key produces different signature
        private_key2, _ = self.engine.generate_key_pair()
        signature4 = self.engine.sign_data(self.sample_data, private_key2)
        self.assertNotEqual(signature, signature4)
        
    def test_verify_signature(self):
        """Test signature verification"""
        private_key, public_key = self.engine.generate_key_pair()
        
        # Sign data with private key
        signature = self.engine.sign_data(self.sample_data, private_key)
        
        # Verify with correct public key - should succeed
        result = self.engine.verify_signature(self.sample_data, signature, public_key)
        self.assertTrue(result)
        
        # Test with modified data - should fail
        modified_data = self.sample_data.copy()
        modified_data['amount'] = 999
        self.assertFalse(self.engine.verify_signature(modified_data, signature, public_key))
        
        # Test with wrong public key - should fail
        wrong_private, wrong_public = self.engine.generate_key_pair()
        self.assertFalse(self.engine.verify_signature(self.sample_data, signature, wrong_public))
        
        # Test with wrong signature format
        wrong_signature = 'a' * 64  # Legacy format without verification part
        self.assertFalse(self.engine.verify_signature(self.sample_data, wrong_signature, public_key))
        
        # Test with corrupted composite signature
        corrupted_signature = 'a' * 64 + ':' + 'b' * 64
        self.assertFalse(self.engine.verify_signature(self.sample_data, corrupted_signature, public_key))
        
        # Test with invalid inputs - None values should raise exceptions
        with self.assertRaises(RuntimeError):
            self.engine.verify_signature(self.sample_data, None, public_key)
        with self.assertRaises(RuntimeError):
            self.engine.verify_signature(self.sample_data, signature, None)
        
    def test_generate_batch_proof(self):
        """Test batch proof generation"""
        transactions = [
            {'tx': i, 'amount': i * 10} for i in range(5)
        ]
        
        batch_proof = self.engine.generate_batch_proof(transactions)
        
        # Verify batch proof structure
        self.assertIsInstance(batch_proof, dict)
        required_fields = ['hash', 'hmac', 'timestamp', 'nonce', 
                          'merkle_root', 'batch_metadata', 'generation_time_ms']
        for field in required_fields:
            self.assertIn(field, batch_proof)
            
        # Verify batch metadata
        metadata = batch_proof['batch_metadata']
        self.assertEqual(metadata['transaction_count'], 5)
        self.assertIn('batch_timestamp', metadata)
        self.assertIn('batch_id', metadata)
        
        # Verify Merkle root
        expected_merkle = self.engine.build_merkle_tree(transactions)
        self.assertEqual(batch_proof['merkle_root'], expected_merkle)
        
        # Test with empty transactions
        empty_batch = self.engine.generate_batch_proof([])
        self.assertEqual(empty_batch['merkle_root'], '0' * 64)
        self.assertEqual(empty_batch['batch_metadata']['transaction_count'], 0)
        
    def test_verify_batch_proof(self):
        """Test batch proof verification"""
        transactions = [
            {'tx': i, 'amount': i * 10} for i in range(5)
        ]
        
        # Generate and verify valid batch proof
        batch_proof = self.engine.generate_batch_proof(transactions)
        self.assertTrue(self.engine.verify_batch_proof(transactions, batch_proof))
        
        # Test with modified transactions
        modified_transactions = transactions.copy()
        modified_transactions[0]['amount'] = 999
        self.assertFalse(self.engine.verify_batch_proof(modified_transactions, batch_proof))
        
        # Test with modified Merkle root
        modified_proof = batch_proof.copy()
        modified_proof['merkle_root'] = 'a' * 64
        self.assertFalse(self.engine.verify_batch_proof(transactions, modified_proof))
        
        # Test with missing metadata
        incomplete_proof = {'merkle_root': batch_proof['merkle_root']}
        self.assertFalse(self.engine.verify_batch_proof(transactions, incomplete_proof))
        
        # Test with empty transactions
        empty_batch_proof = self.engine.generate_batch_proof([])
        self.assertTrue(self.engine.verify_batch_proof([], empty_batch_proof))
        
    def test_generate_proof_for_transaction(self):
        """Test comprehensive transaction proof generation"""
        transaction = {
            'transaction_id': 'tx_test_001',
            'amount': 1000,
            'sender': 'alice',
            'receiver': 'bob'
        }
        
        # Test without context
        proof = self.engine.generate_proof_for_transaction(transaction)
        
        # Verify comprehensive proof structure
        self.assertIsInstance(proof, dict)
        self.assertEqual(proof['engine_type'], 'cryptographic')
        self.assertEqual(proof['transaction_id'], 'tx_test_001')
        self.assertIn('basic_proof', proof)
        self.assertIn('digital_signature', proof)
        self.assertIn('proof_metadata', proof)
        
        # Verify digital signature structure
        sig_data = proof['digital_signature']
        self.assertIn('signature', sig_data)
        self.assertIn('public_key', sig_data)
        self.assertEqual(sig_data['algorithm'], 'HMAC-SHA256')
        
        # Test with context
        context = {'block_height': 100, 'network': 'mainnet'}
        proof_with_context = self.engine.generate_proof_for_transaction(transaction, context)
        self.assertIn('context', proof_with_context)
        self.assertEqual(proof_with_context['context'], context)
        
        # Test without transaction_id
        tx_without_id = {'amount': 500}
        proof_no_id = self.engine.generate_proof_for_transaction(tx_without_id)
        self.assertIn('transaction_id', proof_no_id)
        self.assertEqual(len(proof_no_id['transaction_id']), 16)  # Generated ID
        
    def test_edge_case_empty_data(self):
        """Test with empty data"""
        empty_proof = self.engine.generate_proof({})
        self.assertIsInstance(empty_proof, dict)
        
        # Should still verify
        self.assertTrue(self.engine.verify_proof({}, empty_proof))
        
    def test_edge_case_large_data(self):
        """Test with large data"""
        large_data = {
            f'field_{i}': f'value_{i}' * 100 for i in range(100)
        }
        
        proof = self.engine.generate_proof(large_data)
        self.assertTrue(self.engine.verify_proof(large_data, proof))
        
    def test_edge_case_special_characters(self):
        """Test with special characters and unicode"""
        special_data = {
            'unicode': '你好世界 🌍',
            'special': '!@#$%^&*()_+{}[]|\\:";\'<>?,./`~',
            'newlines': 'line1\nline2\rline3\r\nline4',
            'tabs': 'tab1\ttab2\t\ttab3',
            'null_bytes': 'null\x00byte'
        }
        
        proof = self.engine.generate_proof(special_data)
        self.assertTrue(self.engine.verify_proof(special_data, proof))
        
    def test_edge_case_numerical_precision(self):
        """Test with various numerical values"""
        numerical_data = {
            'large_int': 999999999999999999999999999,
            'small_float': 0.0000000001,
            'negative': -123456789,
            'scientific': 1.23e-10,
            'infinity': float('inf'),
            'nan': float('nan'),
            'zero': 0
        }
        
        # This might fail due to JSON serialization of inf/nan
        try:
            proof = self.engine.generate_proof(numerical_data)
            # If it succeeds, verify it
            self.assertTrue(self.engine.verify_proof(numerical_data, proof))
        except (ValueError, TypeError) as e:
            # Expected for inf/nan in JSON
            pass
            
    def test_thread_safety(self):
        """Test thread safety of the engine"""
        results = []
        errors = []
        
        def worker(engine, data, index):
            try:
                proof = engine.generate_proof(data)
                is_valid = engine.verify_proof(data, proof)
                results.append((index, is_valid))
            except Exception as e:
                errors.append((index, str(e)))
                
        threads = []
        for i in range(10):
            data = {'thread': i, 'data': f'thread_{i}'}
            t = threading.Thread(target=worker, args=(self.engine, data, i))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # All operations should succeed
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)
        self.assertTrue(all(valid for _, valid in results))
        
    def test_performance_proof_generation(self):
        """Test performance of proof generation"""
        iterations = 100
        start_time = time.time()
        
        for i in range(iterations):
            proof = self.engine.generate_proof({'iteration': i})
            
        elapsed = time.time() - start_time
        avg_time = elapsed / iterations
        
        # Should be reasonably fast (less than 10ms per proof)
        self.assertLess(avg_time, 0.01)
        
    def test_performance_merkle_tree(self):
        """Test performance of Merkle tree construction"""
        # Test with various sizes
        sizes = [10, 100, 1000]
        
        for size in sizes:
            transactions = [{'tx': i} for i in range(size)]
            
            start_time = time.time()
            root = self.engine.build_merkle_tree(transactions)
            elapsed = time.time() - start_time
            
            # Should scale reasonably (less than 1ms per transaction)
            self.assertLess(elapsed, size * 0.001)
            
    def test_consistency_across_instances(self):
        """Test consistency across different engine instances"""
        engine1 = CryptographicProofEngine()
        engine2 = CryptographicProofEngine()
        
        # Different engines should produce different proofs (different keys)
        proof1 = engine1.generate_proof(self.sample_data)
        proof2 = engine2.generate_proof(self.sample_data)
        
        # Hashes will differ due to different nonces
        self.assertNotEqual(proof1['hash'], proof2['hash'])
        
        # Each engine should verify its own proofs
        self.assertTrue(engine1.verify_proof(self.sample_data, proof1))
        self.assertTrue(engine2.verify_proof(self.sample_data, proof2))
        
        # But not each other's (different HMAC keys)
        self.assertFalse(engine1.verify_proof(self.sample_data, proof2))
        self.assertFalse(engine2.verify_proof(self.sample_data, proof1))
        
    def test_error_handling_invalid_json(self):
        """Test handling of non-JSON-serializable data"""
        # Create data with non-serializable object
        class CustomObject:
            pass
            
        invalid_data = {'object': CustomObject()}
        
        # Should raise error during serialization
        with self.assertRaises(TypeError):
            self.engine.generate_proof(invalid_data)
            
    def test_error_handling_corrupted_proof(self):
        """Test handling of corrupted proof data"""
        proof = self.engine.generate_proof(self.sample_data)
        
        # Corrupt the proof structure
        corrupted_proofs = [
            {'hash': 'not_64_chars'},  # Invalid hash length
            {'hash': 'z' * 64},  # Invalid hex characters
            None,  # None proof
            "string_proof",  # String instead of dict
            [],  # List instead of dict
        ]
        
        for corrupted in corrupted_proofs:
            self.assertFalse(self.engine.verify_proof(self.sample_data, corrupted))
            
    def test_circular_reference_handling(self):
        """Test handling of circular references in data"""
        # Create circular reference
        circular_data = {'a': 1, 'b': 2}
        circular_data['self'] = circular_data
        
        # Should handle circular reference gracefully
        proof = self.engine.generate_proof(circular_data)
        self.assertIsInstance(proof, dict)
        
        # Verify the proof can be verified
        result = self.engine.verify_proof(circular_data, proof)
        self.assertTrue(result)
        
    def test_deep_circular_reference(self):
        """Test handling of deeply nested circular references"""
        # Create nested circular reference
        data = {'level1': {'level2': {}}}
        data['level1']['level2']['back_to_root'] = data
        
        # Should handle circular reference gracefully
        proof = self.engine.generate_proof(data)
        self.assertIsInstance(proof, dict)
        
        # Verify the proof
        result = self.engine.verify_proof(data, proof)
        self.assertTrue(result)
        
    def test_maximum_depth_protection(self):
        """Test maximum depth protection during serialization"""
        # Create extremely deep nesting (beyond typical limits)
        deep_data = {}
        current = deep_data
        for i in range(150):  # Exceeds default max_depth of 100
            current['level'] = {}
            current = current['level']
        current['value'] = 'very_deep'
        
        # Should raise ValueError for excessive depth
        with self.assertRaises(ValueError) as context:
            self.engine.generate_proof(deep_data)
            
        self.assertIn("Maximum recursion depth", str(context.exception))
        
    def test_circular_reference_in_array(self):
        """Test circular reference handling in arrays"""
        data = {'array': [1, 2, 3]}
        data['array'].append(data)  # Circular reference through array
        
        # Should handle gracefully
        proof = self.engine.generate_proof(data)
        self.assertIsInstance(proof, dict)
        
        # Verify the proof
        result = self.engine.verify_proof(data, proof)
        self.assertTrue(result)
        
    def test_multiple_circular_references(self):
        """Test handling of multiple circular references"""
        data1 = {'name': 'data1'}
        data2 = {'name': 'data2'}
        
        # Create mutual circular references
        data1['ref'] = data2
        data2['ref'] = data1
        
        root_data = {'data1': data1, 'data2': data2}
        
        # Should handle multiple circular references
        proof = self.engine.generate_proof(root_data)
        self.assertIsInstance(proof, dict)
        
        result = self.engine.verify_proof(root_data, proof)
        self.assertTrue(result)


class TestCryptographicProofEngineIntegration(unittest.TestCase):
    """Integration tests for CryptographicProofEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = CryptographicProofEngine()
        
    def test_full_blockchain_simulation(self):
        """Simulate a full blockchain with multiple blocks"""
        blockchain = []
        prev_hash = None
        
        # Create 10 blocks
        for i in range(10):
            # Create block data
            block_data = {
                'block_number': i,
                'timestamp': datetime.now().isoformat(),
                'transactions': [
                    {'from': f'user_{j}', 'to': f'user_{j+1}', 'amount': j * 10}
                    for j in range(5)
                ]
            }
            
            # Generate proof
            proof = self.engine.generate_proof(block_data, prev_hash)
            
            # Add to blockchain
            blockchain.append({
                'data': block_data,
                'proof': proof
            })
            
            prev_hash = proof['hash']
            
        # Verify entire chain
        self.assertTrue(self.engine.verify_chain(blockchain))
        
        # Verify individual blocks
        for block in blockchain:
            self.assertTrue(self.engine.verify_proof(block['data'], block['proof']))
            
    def test_batch_processing_workflow(self):
        """Test complete batch processing workflow"""
        # Create batch of transactions
        transactions = []
        for i in range(50):
            tx = {
                'id': f'tx_{i:03d}',
                'from': f'user_{i % 10}',
                'to': f'user_{(i + 1) % 10}',
                'amount': i * 100,
                'timestamp': datetime.now().isoformat()
            }
            transactions.append(tx)
            
        # Generate batch proof
        batch_proof = self.engine.generate_batch_proof(transactions)
        
        # Verify batch
        self.assertTrue(self.engine.verify_batch_proof(transactions, batch_proof))
        
        # Generate individual Merkle proofs
        for i in range(len(transactions)):
            proof_path = self.engine.get_merkle_proof(transactions, i)
            self.assertIsInstance(proof_path, list)
            
    def test_digital_signature_workflow(self):
        """Test complete digital signature workflow"""
        # Generate key pairs for multiple users
        users = {}
        for name in ['alice', 'bob', 'charlie']:
            private_key, public_key = self.engine.generate_key_pair()
            users[name] = {
                'private_key': private_key,
                'public_key': public_key
            }
            
        # Alice signs a transaction
        transaction = {
            'from': 'alice',
            'to': 'bob',
            'amount': 100,
            'timestamp': datetime.now().isoformat()
        }
        
        alice_signature = self.engine.sign_data(
            transaction, 
            users['alice']['private_key']
        )
        
        # Create comprehensive proof
        proof = self.engine.generate_proof_for_transaction(
            transaction,
            context={'signer': 'alice', 'signature': alice_signature}
        )
        
        self.assertIsInstance(proof, dict)
        self.assertEqual(proof['engine_type'], 'cryptographic')
        
    def test_performance_under_load(self):
        """Test performance under heavy load"""
        start_time = time.time()
        proofs_generated = 0
        proofs_verified = 0
        
        # Generate many proofs
        for i in range(100):
            data = {'batch': i // 10, 'item': i % 10}
            proof = self.engine.generate_proof(data)
            proofs_generated += 1
            
            # Verify every 10th proof
            if i % 10 == 0:
                self.assertTrue(self.engine.verify_proof(data, proof))
                proofs_verified += 1
                
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time (< 1 second)
        self.assertLess(elapsed, 1.0)
        self.assertEqual(proofs_generated, 100)
        self.assertEqual(proofs_verified, 10)


if __name__ == '__main__':
    unittest.main(verbosity=2)