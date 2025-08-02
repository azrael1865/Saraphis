#!/usr/bin/env python3
"""
Network Analysis Feature Extractor Plugin
==========================================

This module provides network traffic analysis feature extraction capabilities for the Universal AI Core system.
Adapted from Saraphis molecular descriptor patterns, specialized for network security analysis,
intrusion detection, and network behavior classification.

Features:
- Packet-level feature extraction and analysis
- Flow-based statistical features
- Protocol distribution analysis
- Temporal pattern extraction
- Anomaly detection features
- Traffic volume and rate analysis
- Connection pattern analysis
- Payload analysis and n-gram extraction
"""

import logging
import sys
import time
import socket
import struct
import hashlib
import math
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter, defaultdict, deque
import threading
import ipaddress
import re

# Try to import network analysis dependencies
try:
    import scapy
    from scapy.all import *
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.inet6 import IPv6
    from scapy.layers.http import HTTP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# Import plugin base classes
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base import FeatureExtractorPlugin, FeatureExtractionResult, FeatureType

logger = logging.getLogger(__name__)


@dataclass
class NetworkFlow:
    """Network flow representation"""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    start_time: float
    end_time: float
    packets: int = 0
    bytes: int = 0
    flags: List[str] = field(default_factory=list)
    payload_sizes: List[int] = field(default_factory=list)
    inter_arrival_times: List[float] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def flow_id(self) -> str:
        return f"{self.src_ip}:{self.src_port}-{self.dst_ip}:{self.dst_port}-{self.protocol}"


@dataclass
class NetworkAnalysisResult:
    """Result container for network analysis"""
    flow_features: np.ndarray = field(default_factory=lambda: np.array([]))
    packet_features: np.ndarray = field(default_factory=lambda: np.array([]))
    temporal_features: np.ndarray = field(default_factory=lambda: np.array([]))
    protocol_features: np.ndarray = field(default_factory=lambda: np.array([]))
    anomaly_features: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_names: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    valid_packets: int = 0
    invalid_packets: int = 0
    flows_detected: int = 0
    error_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class NetworkFlowAnalyzer:
    """
    Network flow statistical analyzer.
    
    Adapted from molecular descriptor calculation patterns.
    """
    
    def __init__(self):
        self.flows = {}  # flow_id -> NetworkFlow
        self.flow_timeout = 300.0  # 5 minutes
        self.logger = logging.getLogger(f"{__name__}.NetworkFlowAnalyzer")
    
    def process_packet(self, packet, timestamp: float):
        """Process a single packet and update flows"""
        try:
            if not SCAPY_AVAILABLE:
                return
            
            # Extract packet information
            packet_info = self._extract_packet_info(packet, timestamp)
            if not packet_info:
                return
            
            flow_id = packet_info['flow_id']
            
            # Update or create flow
            if flow_id in self.flows:
                flow = self.flows[flow_id]
                flow.end_time = timestamp
                flow.packets += 1
                flow.bytes += packet_info['size']
                
                if packet_info['payload_size'] > 0:
                    flow.payload_sizes.append(packet_info['payload_size'])
                
                # Calculate inter-arrival time
                if flow.packets > 1:
                    inter_arrival = timestamp - flow.end_time
                    flow.inter_arrival_times.append(inter_arrival)
                
                if 'flags' in packet_info:
                    flow.flags.extend(packet_info['flags'])
            else:
                # Create new flow
                flow = NetworkFlow(
                    src_ip=packet_info['src_ip'],
                    dst_ip=packet_info['dst_ip'],
                    src_port=packet_info['src_port'],
                    dst_port=packet_info['dst_port'],
                    protocol=packet_info['protocol'],
                    start_time=timestamp,
                    end_time=timestamp,
                    packets=1,
                    bytes=packet_info['size']
                )
                
                if packet_info['payload_size'] > 0:
                    flow.payload_sizes.append(packet_info['payload_size'])
                
                if 'flags' in packet_info:
                    flow.flags.extend(packet_info['flags'])
                
                self.flows[flow_id] = flow
            
            # Cleanup old flows
            self._cleanup_old_flows(timestamp)
            
        except Exception as e:
            self.logger.error(f"Error processing packet: {e}")
    
    def _extract_packet_info(self, packet, timestamp: float) -> Optional[Dict[str, Any]]:
        """Extract relevant information from packet"""
        try:
            info = {
                'timestamp': timestamp,
                'size': len(packet),
                'payload_size': 0
            }
            
            # Extract IP layer information
            if IP in packet:
                ip_layer = packet[IP]
                info.update({
                    'src_ip': ip_layer.src,
                    'dst_ip': ip_layer.dst,
                    'protocol': ip_layer.proto,
                    'ttl': ip_layer.ttl,
                    'length': ip_layer.len
                })
                
                # Extract transport layer information
                if TCP in packet:
                    tcp_layer = packet[TCP]
                    info.update({
                        'src_port': tcp_layer.sport,
                        'dst_port': tcp_layer.dport,
                        'protocol': 'TCP',
                        'flags': self._extract_tcp_flags(tcp_layer.flags),
                        'window_size': tcp_layer.window,
                        'seq': tcp_layer.seq,
                        'ack': tcp_layer.ack
                    })
                    
                    # Calculate payload size
                    if hasattr(tcp_layer, 'payload') and tcp_layer.payload:
                        info['payload_size'] = len(tcp_layer.payload)
                
                elif UDP in packet:
                    udp_layer = packet[UDP]
                    info.update({
                        'src_port': udp_layer.sport,
                        'dst_port': udp_layer.dport,
                        'protocol': 'UDP',
                        'length': udp_layer.len
                    })
                    
                    # Calculate payload size
                    if hasattr(udp_layer, 'payload') and udp_layer.payload:
                        info['payload_size'] = len(udp_layer.payload)
                
                elif ICMP in packet:
                    icmp_layer = packet[ICMP]
                    info.update({
                        'src_port': 0,
                        'dst_port': 0,
                        'protocol': 'ICMP',
                        'type': icmp_layer.type,
                        'code': icmp_layer.code
                    })
                
                # Generate flow ID
                flow_key = (info['src_ip'], info['dst_ip'], 
                           info.get('src_port', 0), info.get('dst_port', 0), 
                           info.get('protocol', 'OTHER'))
                info['flow_id'] = '-'.join(map(str, flow_key))
                
                return info
            
        except Exception as e:
            self.logger.debug(f"Error extracting packet info: {e}")
        
        return None
    
    def _extract_tcp_flags(self, flags: int) -> List[str]:
        """Extract TCP flags as string list"""
        flag_names = []
        if flags & 0x01: flag_names.append('FIN')
        if flags & 0x02: flag_names.append('SYN')
        if flags & 0x04: flag_names.append('RST')
        if flags & 0x08: flag_names.append('PSH')
        if flags & 0x10: flag_names.append('ACK')
        if flags & 0x20: flag_names.append('URG')
        if flags & 0x40: flag_names.append('ECE')
        if flags & 0x80: flag_names.append('CWR')
        return flag_names
    
    def _cleanup_old_flows(self, current_time: float):
        """Remove flows that have timed out"""
        expired_flows = []
        for flow_id, flow in self.flows.items():
            if current_time - flow.end_time > self.flow_timeout:
                expired_flows.append(flow_id)
        
        for flow_id in expired_flows:
            del self.flows[flow_id]
    
    def extract_flow_features(self) -> Tuple[np.ndarray, List[str]]:
        """Extract statistical features from network flows"""
        if not self.flows:
            return np.zeros(30), [f"flow_feature_{i}" for i in range(30)]
        
        try:
            # Aggregate flow statistics
            durations = [flow.duration for flow in self.flows.values()]
            packets_per_flow = [flow.packets for flow in self.flows.values()]
            bytes_per_flow = [flow.bytes for flow in self.flows.values()]
            
            # Protocol distribution
            protocols = [flow.protocol for flow in self.flows.values()]
            protocol_counts = Counter(protocols)
            tcp_ratio = protocol_counts.get('TCP', 0) / len(protocols)
            udp_ratio = protocol_counts.get('UDP', 0) / len(protocols)
            icmp_ratio = protocol_counts.get('ICMP', 0) / len(protocols)
            
            # Port analysis
            src_ports = [flow.src_port for flow in self.flows.values() if flow.src_port > 0]
            dst_ports = [flow.dst_port for flow in self.flows.values() if flow.dst_port > 0]
            unique_src_ports = len(set(src_ports)) if src_ports else 0
            unique_dst_ports = len(set(dst_ports)) if dst_ports else 0
            
            # Well-known port usage
            well_known_src = sum(1 for p in src_ports if p < 1024) / max(len(src_ports), 1)
            well_known_dst = sum(1 for p in dst_ports if p < 1024) / max(len(dst_ports), 1)
            
            # Flow size statistics
            avg_duration = np.mean(durations) if durations else 0
            max_duration = np.max(durations) if durations else 0
            std_duration = np.std(durations) if durations else 0
            
            avg_packets = np.mean(packets_per_flow) if packets_per_flow else 0
            max_packets = np.max(packets_per_flow) if packets_per_flow else 0
            std_packets = np.std(packets_per_flow) if packets_per_flow else 0
            
            avg_bytes = np.mean(bytes_per_flow) if bytes_per_flow else 0
            max_bytes = np.max(bytes_per_flow) if bytes_per_flow else 0
            std_bytes = np.std(bytes_per_flow) if bytes_per_flow else 0
            
            # Inter-arrival time statistics
            all_inter_arrivals = []
            for flow in self.flows.values():
                all_inter_arrivals.extend(flow.inter_arrival_times)
            
            avg_inter_arrival = np.mean(all_inter_arrivals) if all_inter_arrivals else 0
            std_inter_arrival = np.std(all_inter_arrivals) if all_inter_arrivals else 0
            
            # Payload size statistics
            all_payload_sizes = []
            for flow in self.flows.values():
                all_payload_sizes.extend(flow.payload_sizes)
            
            avg_payload = np.mean(all_payload_sizes) if all_payload_sizes else 0
            max_payload = np.max(all_payload_sizes) if all_payload_sizes else 0
            std_payload = np.std(all_payload_sizes) if all_payload_sizes else 0
            
            # Connection patterns
            total_flows = len(self.flows)
            active_flows = sum(1 for flow in self.flows.values() if flow.packets > 1)
            
            features = [
                total_flows, active_flows, tcp_ratio, udp_ratio, icmp_ratio,
                unique_src_ports, unique_dst_ports, well_known_src, well_known_dst,
                avg_duration, max_duration, std_duration,
                avg_packets, max_packets, std_packets,
                avg_bytes, max_bytes, std_bytes,
                avg_inter_arrival, std_inter_arrival,
                avg_payload, max_payload, std_payload
            ]
            
            # Pad to fixed size
            target_size = 30
            while len(features) < target_size:
                features.append(0.0)
            
            feature_names = [
                'total_flows', 'active_flows', 'tcp_ratio', 'udp_ratio', 'icmp_ratio',
                'unique_src_ports', 'unique_dst_ports', 'well_known_src', 'well_known_dst',
                'avg_duration', 'max_duration', 'std_duration',
                'avg_packets', 'max_packets', 'std_packets',
                'avg_bytes', 'max_bytes', 'std_bytes',
                'avg_inter_arrival', 'std_inter_arrival',
                'avg_payload', 'max_payload', 'std_payload'
            ] + [f"flow_feature_{i}" for i in range(len(features) - 23)]
            
            return np.array(features), feature_names
            
        except Exception as e:
            self.logger.error(f"Error extracting flow features: {e}")
            return np.zeros(30), [f"flow_feature_{i}" for i in range(30)]


class PacketStatisticalAnalyzer:
    """
    Packet-level statistical analyzer.
    
    Adapted from molecular statistical analysis patterns.
    """
    
    def __init__(self):
        self.packet_sizes = []
        self.inter_packet_times = []
        self.protocols = []
        self.ttl_values = []
        self.last_timestamp = None
        self.logger = logging.getLogger(f"{__name__}.PacketStatisticalAnalyzer")
    
    def process_packet(self, packet, timestamp: float):
        """Process packet for statistical analysis"""
        try:
            if not SCAPY_AVAILABLE:
                return
            
            # Packet size
            self.packet_sizes.append(len(packet))
            
            # Inter-packet time
            if self.last_timestamp is not None:
                inter_time = timestamp - self.last_timestamp
                self.inter_packet_times.append(inter_time)
            self.last_timestamp = timestamp
            
            # Protocol
            if IP in packet:
                self.protocols.append(packet[IP].proto)
                self.ttl_values.append(packet[IP].ttl)
            
        except Exception as e:
            self.logger.error(f"Error in packet statistical analysis: {e}")
    
    def extract_statistical_features(self) -> Tuple[np.ndarray, List[str]]:
        """Extract statistical features from packets"""
        try:
            features = []
            
            # Packet size statistics
            if self.packet_sizes:
                features.extend([
                    len(self.packet_sizes),  # Total packets
                    np.mean(self.packet_sizes),
                    np.std(self.packet_sizes),
                    np.min(self.packet_sizes),
                    np.max(self.packet_sizes),
                    np.median(self.packet_sizes)
                ])
            else:
                features.extend([0.0] * 6)
            
            # Inter-packet time statistics
            if self.inter_packet_times:
                features.extend([
                    np.mean(self.inter_packet_times),
                    np.std(self.inter_packet_times),
                    np.min(self.inter_packet_times),
                    np.max(self.inter_packet_times)
                ])
            else:
                features.extend([0.0] * 4)
            
            # Protocol distribution
            if self.protocols:
                protocol_counts = Counter(self.protocols)
                total_protocols = len(self.protocols)
                tcp_ratio = protocol_counts.get(6, 0) / total_protocols  # TCP = 6
                udp_ratio = protocol_counts.get(17, 0) / total_protocols  # UDP = 17
                icmp_ratio = protocol_counts.get(1, 0) / total_protocols  # ICMP = 1
                unique_protocols = len(protocol_counts)
                
                features.extend([tcp_ratio, udp_ratio, icmp_ratio, unique_protocols])
            else:
                features.extend([0.0] * 4)
            
            # TTL statistics
            if self.ttl_values:
                features.extend([
                    np.mean(self.ttl_values),
                    np.std(self.ttl_values),
                    len(set(self.ttl_values))  # Unique TTL values
                ])
            else:
                features.extend([0.0] * 3)
            
            # Ensure fixed size
            target_size = 20
            while len(features) < target_size:
                features.append(0.0)
            
            feature_names = [
                'total_packets', 'avg_packet_size', 'std_packet_size', 'min_packet_size',
                'max_packet_size', 'median_packet_size', 'avg_inter_time', 'std_inter_time',
                'min_inter_time', 'max_inter_time', 'tcp_ratio', 'udp_ratio', 'icmp_ratio',
                'unique_protocols', 'avg_ttl', 'std_ttl', 'unique_ttls'
            ] + [f"packet_stat_{i}" for i in range(len(features) - 17)]
            
            return np.array(features), feature_names
            
        except Exception as e:
            self.logger.error(f"Error extracting packet statistics: {e}")
            return np.zeros(20), [f"packet_stat_{i}" for i in range(20)]


class NetworkAnomalyDetector:
    """
    Network anomaly detection feature extractor.
    
    Adapted from molecular anomaly detection patterns.
    """
    
    def __init__(self):
        self.baseline_stats = {}
        self.anomaly_threshold = 2.0  # Standard deviations
        self.logger = logging.getLogger(f"{__name__}.NetworkAnomalyDetector")
    
    def detect_anomalies(self, flows: Dict[str, NetworkFlow], packets_stats: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Detect network anomalies"""
        try:
            anomaly_features = []
            
            # Flow-based anomalies
            if flows:
                flow_anomalies = self._detect_flow_anomalies(flows)
                anomaly_features.extend(flow_anomalies)
            else:
                anomaly_features.extend([0.0] * 10)
            
            # Packet-based anomalies
            packet_anomalies = self._detect_packet_anomalies(packets_stats)
            anomaly_features.extend(packet_anomalies)
            
            # Port scan detection
            port_scan_score = self._detect_port_scanning(flows)
            anomaly_features.append(port_scan_score)
            
            # DDoS detection
            ddos_score = self._detect_ddos_patterns(flows)
            anomaly_features.append(ddos_score)
            
            # Ensure fixed size
            target_size = 20
            while len(anomaly_features) < target_size:
                anomaly_features.append(0.0)
            
            feature_names = [
                'flow_duration_anomaly', 'flow_packets_anomaly', 'flow_bytes_anomaly',
                'inter_arrival_anomaly', 'payload_size_anomaly', 'protocol_anomaly',
                'port_anomaly', 'connection_rate_anomaly', 'bandwidth_anomaly',
                'flow_symmetry_anomaly', 'packet_size_anomaly', 'packet_rate_anomaly',
                'ttl_anomaly', 'protocol_mix_anomaly', 'port_scan_score', 'ddos_score'
            ] + [f"anomaly_feature_{i}" for i in range(len(anomaly_features) - 16)]
            
            return np.array(anomaly_features), feature_names
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return np.zeros(20), [f"anomaly_feature_{i}" for i in range(20)]
    
    def _detect_flow_anomalies(self, flows: Dict[str, NetworkFlow]) -> List[float]:
        """Detect flow-level anomalies"""
        anomalies = []
        
        try:
            # Extract flow metrics
            durations = [flow.duration for flow in flows.values()]
            packets = [flow.packets for flow in flows.values()]
            bytes_list = [flow.bytes for flow in flows.values()]
            
            # Calculate z-scores for anomaly detection
            duration_anomaly = self._calculate_anomaly_score(durations)
            packets_anomaly = self._calculate_anomaly_score(packets)
            bytes_anomaly = self._calculate_anomaly_score(bytes_list)
            
            # Inter-arrival time anomalies
            all_inter_arrivals = []
            for flow in flows.values():
                all_inter_arrivals.extend(flow.inter_arrival_times)
            inter_arrival_anomaly = self._calculate_anomaly_score(all_inter_arrivals)
            
            # Payload size anomalies
            all_payloads = []
            for flow in flows.values():
                all_payloads.extend(flow.payload_sizes)
            payload_anomaly = self._calculate_anomaly_score(all_payloads)
            
            anomalies.extend([
                duration_anomaly, packets_anomaly, bytes_anomaly,
                inter_arrival_anomaly, payload_anomaly
            ])
            
            # Protocol anomalies
            protocols = [flow.protocol for flow in flows.values()]
            protocol_dist = Counter(protocols)
            protocol_entropy = self._calculate_entropy(list(protocol_dist.values()))
            anomalies.append(protocol_entropy)
            
            # Port anomalies
            dst_ports = [flow.dst_port for flow in flows.values() if flow.dst_port > 0]
            unique_dst_ports = len(set(dst_ports))
            port_diversity = unique_dst_ports / max(len(dst_ports), 1)
            anomalies.append(port_diversity)
            
            # Connection rate anomaly
            if flows:
                time_window = max(flow.end_time for flow in flows.values()) - min(flow.start_time for flow in flows.values())
                connection_rate = len(flows) / max(time_window, 1)
                anomalies.append(connection_rate)
            else:
                anomalies.append(0.0)
            
            # Bandwidth anomaly
            total_bytes = sum(flow.bytes for flow in flows.values())
            if flows:
                time_window = max(flow.end_time for flow in flows.values()) - min(flow.start_time for flow in flows.values())
                bandwidth = total_bytes / max(time_window, 1)
                anomalies.append(bandwidth)
            else:
                anomalies.append(0.0)
            
            # Flow symmetry (bidirectional flows)
            flow_pairs = self._find_bidirectional_flows(flows)
            symmetry_ratio = len(flow_pairs) / max(len(flows), 1)
            anomalies.append(symmetry_ratio)
            
        except Exception as e:
            self.logger.error(f"Error in flow anomaly detection: {e}")
            anomalies = [0.0] * 10
        
        return anomalies
    
    def _detect_packet_anomalies(self, packet_stats: Dict[str, Any]) -> List[float]:
        """Detect packet-level anomalies"""
        try:
            # Extract packet statistics
            packet_sizes = packet_stats.get('packet_sizes', [])
            inter_times = packet_stats.get('inter_packet_times', [])
            ttls = packet_stats.get('ttl_values', [])
            protocols = packet_stats.get('protocols', [])
            
            # Calculate anomaly scores
            size_anomaly = self._calculate_anomaly_score(packet_sizes)
            rate_anomaly = self._calculate_anomaly_score([1/t for t in inter_times if t > 0])
            ttl_anomaly = self._calculate_anomaly_score(ttls)
            
            # Protocol mix anomaly
            if protocols:
                protocol_counts = Counter(protocols)
                protocol_entropy = self._calculate_entropy(list(protocol_counts.values()))
            else:
                protocol_entropy = 0.0
            
            return [size_anomaly, rate_anomaly, ttl_anomaly, protocol_entropy]
            
        except Exception as e:
            self.logger.error(f"Error in packet anomaly detection: {e}")
            return [0.0] * 4
    
    def _detect_port_scanning(self, flows: Dict[str, NetworkFlow]) -> float:
        """Detect port scanning behavior"""
        try:
            # Group flows by source IP
            src_ip_flows = defaultdict(list)
            for flow in flows.values():
                src_ip_flows[flow.src_ip].append(flow)
            
            max_scan_score = 0.0
            
            for src_ip, ip_flows in src_ip_flows.items():
                # Check for many unique destination ports from single source
                dst_ports = set(flow.dst_port for flow in ip_flows if flow.dst_port > 0)
                dst_ips = set(flow.dst_ip for flow in ip_flows)
                
                if len(dst_ports) > 10 and len(dst_ips) == 1:  # Many ports, single target
                    scan_score = min(len(dst_ports) / 100.0, 1.0)  # Normalize to 0-1
                    max_scan_score = max(max_scan_score, scan_score)
                elif len(dst_ports) > 5 and len(dst_ips) > 1:  # Multiple targets
                    scan_score = min((len(dst_ports) * len(dst_ips)) / 500.0, 1.0)
                    max_scan_score = max(max_scan_score, scan_score)
            
            return max_scan_score
            
        except Exception as e:
            self.logger.error(f"Error in port scan detection: {e}")
            return 0.0
    
    def _detect_ddos_patterns(self, flows: Dict[str, NetworkFlow]) -> float:
        """Detect DDoS attack patterns"""
        try:
            if not flows:
                return 0.0
            
            # Group flows by destination IP
            dst_ip_flows = defaultdict(list)
            for flow in flows.values():
                dst_ip_flows[flow.dst_ip].append(flow)
            
            max_ddos_score = 0.0
            
            for dst_ip, ip_flows in dst_ip_flows.items():
                # Check for many sources targeting single destination
                src_ips = set(flow.src_ip for flow in ip_flows)
                total_packets = sum(flow.packets for flow in ip_flows)
                
                if len(src_ips) > 20:  # Many different sources
                    # Calculate concentration of traffic
                    concentration = total_packets / len(src_ips)
                    ddos_score = min(concentration / 1000.0, 1.0)  # Normalize
                    max_ddos_score = max(max_ddos_score, ddos_score)
            
            return max_ddos_score
            
        except Exception as e:
            self.logger.error(f"Error in DDoS detection: {e}")
            return 0.0
    
    def _calculate_anomaly_score(self, values: List[float]) -> float:
        """Calculate anomaly score based on z-score"""
        if not values or len(values) < 2:
            return 0.0
        
        try:
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:
                return 0.0
            
            # Calculate max z-score
            z_scores = [(val - mean_val) / std_val for val in values]
            max_z_score = max(abs(z) for z in z_scores)
            
            # Normalize to 0-1 range
            return min(max_z_score / 3.0, 1.0)  # 3 standard deviations = max
            
        except Exception:
            return 0.0
    
    def _calculate_entropy(self, values: List[int]) -> float:
        """Calculate Shannon entropy"""
        if not values:
            return 0.0
        
        total = sum(values)
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in values:
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _find_bidirectional_flows(self, flows: Dict[str, NetworkFlow]) -> List[Tuple[str, str]]:
        """Find bidirectional flow pairs"""
        flow_pairs = []
        processed = set()
        
        for flow_id, flow in flows.items():
            if flow_id in processed:
                continue
            
            # Look for reverse flow
            reverse_key = f"{flow.dst_ip}:{flow.dst_port}-{flow.src_ip}:{flow.src_port}-{flow.protocol}"
            
            if reverse_key in flows:
                flow_pairs.append((flow_id, reverse_key))
                processed.add(flow_id)
                processed.add(reverse_key)
        
        return flow_pairs


class NetworkFeatureExtractor(FeatureExtractorPlugin):
    """
    Network analysis feature extractor plugin.
    
    Adapted from Saraphis molecular feature extraction patterns,
    specialized for network traffic analysis and intrusion detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the network feature extractor"""
        super().__init__(config)
        
        # Configuration
        self.use_flow_analysis = self.config.get('use_flow_analysis', True)
        self.use_packet_analysis = self.config.get('use_packet_analysis', True)
        self.use_anomaly_detection = self.config.get('use_anomaly_detection', True)
        
        self.max_packets = self.config.get('max_packets', 100000)
        self.flow_timeout = self.config.get('flow_timeout', 300.0)
        
        # Initialize analyzers
        self.flow_analyzer = NetworkFlowAnalyzer() if self.use_flow_analysis else None
        self.packet_analyzer = PacketStatisticalAnalyzer() if self.use_packet_analysis else None
        self.anomaly_detector = NetworkAnomalyDetector() if self.use_anomaly_detection else None
        
        # Feature caching
        self.feature_cache = {}
        self.cache_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'packets_processed': 0,
            'flows_detected': 0,
            'cache_hits': 0,
            'extraction_time': 0.0,
            'invalid_packets': 0
        }
        
        self.logger.info(f"🌐 Network Feature Extractor initialized (Scapy: {SCAPY_AVAILABLE})")
    
    def get_metadata(self):
        """Get plugin metadata"""
        from ..base import PluginMetadata, PluginDependency, PluginVersion
        
        dependencies = []
        if SCAPY_AVAILABLE:
            dependencies.append(PluginDependency(name="scapy", version_requirement="*"))
        
        return PluginMetadata(
            name="NetworkFeatureExtractor",
            version=PluginVersion(1, 0, 0),
            author="Universal AI Core",
            description="Network traffic analysis and intrusion detection features",
            plugin_type="feature_extractor",
            entry_point=f"{__name__}:NetworkFeatureExtractor",
            dependencies=dependencies,
            capabilities=["flow_analysis", "packet_analysis", "anomaly_detection", "intrusion_detection"],
            hooks=["before_extraction", "after_extraction"],
            configuration_schema={
                "use_flow_analysis": {"type": "boolean", "default": True},
                "use_packet_analysis": {"type": "boolean", "default": True},
                "max_packets": {"type": "integer", "default": 100000}
            }
        )
    
    def extract_features(self, input_data: Any) -> FeatureExtractionResult:
        """Extract features from network traffic data"""
        start_time = time.time()
        
        try:
            # Parse input data
            packets = self._parse_input_data(input_data)
            if not packets:
                return FeatureExtractionResult(
                    features={},
                    feature_names=[],
                    feature_types=[],
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="No valid network packets found in input data"
                )
            
            # Process packets
            result = self._process_network_traffic(packets)
            
            if result.valid_packets > 0:
                # Combine all feature types
                features, feature_names, feature_types = self._combine_features(result)
                
                processing_time = time.time() - start_time
                
                # Update statistics
                self.stats['packets_processed'] += result.valid_packets
                self.stats['flows_detected'] += result.flows_detected
                self.stats['extraction_time'] += processing_time
                self.stats['invalid_packets'] += result.invalid_packets
                
                return FeatureExtractionResult(
                    features=features,
                    feature_names=feature_names,
                    feature_types=feature_types,
                    processing_time=processing_time,
                    success=True,
                    metadata={
                        'total_packets': len(packets),
                        'valid_packets': result.valid_packets,
                        'invalid_packets': result.invalid_packets,
                        'flows_detected': result.flows_detected,
                        'feature_dimensions': {
                            'flow_features': len(result.flow_features) if result.flow_features.size > 0 else 0,
                            'packet_features': len(result.packet_features) if result.packet_features.size > 0 else 0,
                            'anomaly_features': len(result.anomaly_features) if result.anomaly_features.size > 0 else 0
                        }
                    }
                )
            else:
                return FeatureExtractionResult(
                    features={},
                    feature_names=[],
                    feature_types=[],
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="No valid features extracted from network traffic"
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error extracting network features: {e}")
            return FeatureExtractionResult(
                features={},
                feature_names=[],
                feature_types=[],
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _parse_input_data(self, input_data: Any) -> List[Tuple[Any, float]]:
        """Parse input data to extract packets with timestamps"""
        packets = []
        
        if not SCAPY_AVAILABLE:
            self.logger.error("Scapy not available - network analysis disabled")
            return packets
        
        try:
            if isinstance(input_data, str):
                # PCAP file path
                packets_raw = rdpcap(input_data)
                packets = [(pkt, float(pkt.time)) for pkt in packets_raw]
            elif isinstance(input_data, list):
                # List of packets or PCAP files
                for item in input_data:
                    if isinstance(item, str):
                        packets_raw = rdpcap(item)
                        packets.extend([(pkt, float(pkt.time)) for pkt in packets_raw])
                    elif hasattr(item, 'time'):  # Scapy packet
                        packets.append((item, float(item.time)))
            elif hasattr(input_data, '__iter__'):
                # Iterable of packets
                current_time = time.time()
                for i, pkt in enumerate(input_data):
                    timestamp = current_time + i * 0.001  # Artificial timestamps
                    packets.append((pkt, timestamp))
            elif isinstance(input_data, dict):
                # Dictionary with packet data
                if 'pcap_file' in input_data:
                    packets_raw = rdpcap(input_data['pcap_file'])
                    packets = [(pkt, float(pkt.time)) for pkt in packets_raw]
                elif 'packets' in input_data:
                    packets_data = input_data['packets']
                    current_time = time.time()
                    for i, pkt in enumerate(packets_data):
                        timestamp = current_time + i * 0.001
                        packets.append((pkt, timestamp))
        except Exception as e:
            self.logger.error(f"Error parsing network input data: {e}")
        
        # Limit number of packets
        if len(packets) > self.max_packets:
            packets = packets[:self.max_packets]
            self.logger.warning(f"Limited packets to {self.max_packets}")
        
        return packets
    
    def _process_network_traffic(self, packets: List[Tuple[Any, float]]) -> NetworkAnalysisResult:
        """Process network traffic and extract features"""
        result = NetworkAnalysisResult()
        
        try:
            valid_count = 0
            invalid_count = 0
            
            # Reset analyzers
            if self.flow_analyzer:
                self.flow_analyzer.flows.clear()
            if self.packet_analyzer:
                self.packet_analyzer.packet_sizes.clear()
                self.packet_analyzer.inter_packet_times.clear()
                self.packet_analyzer.protocols.clear()
                self.packet_analyzer.ttl_values.clear()
                self.packet_analyzer.last_timestamp = None
            
            # Process each packet
            for packet, timestamp in packets:
                try:
                    # Flow analysis
                    if self.flow_analyzer:
                        self.flow_analyzer.process_packet(packet, timestamp)
                    
                    # Packet analysis
                    if self.packet_analyzer:
                        self.packet_analyzer.process_packet(packet, timestamp)
                    
                    valid_count += 1
                    
                except Exception as e:
                    self.logger.debug(f"Error processing packet: {e}")
                    invalid_count += 1
            
            # Extract flow features
            if self.flow_analyzer:
                flow_features, flow_names = self.flow_analyzer.extract_flow_features()
                result.flow_features = flow_features
                result.feature_names.extend(flow_names)
                result.flows_detected = len(self.flow_analyzer.flows)
            
            # Extract packet features
            if self.packet_analyzer:
                packet_features, packet_names = self.packet_analyzer.extract_statistical_features()
                result.packet_features = packet_features
            
            # Extract anomaly features
            if self.anomaly_detector and self.flow_analyzer:
                packet_stats = {
                    'packet_sizes': self.packet_analyzer.packet_sizes if self.packet_analyzer else [],
                    'inter_packet_times': self.packet_analyzer.inter_packet_times if self.packet_analyzer else [],
                    'ttl_values': self.packet_analyzer.ttl_values if self.packet_analyzer else [],
                    'protocols': self.packet_analyzer.protocols if self.packet_analyzer else []
                }
                
                anomaly_features, anomaly_names = self.anomaly_detector.detect_anomalies(
                    self.flow_analyzer.flows, packet_stats
                )
                result.anomaly_features = anomaly_features
            
            result.valid_packets = valid_count
            result.invalid_packets = invalid_count
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing network traffic: {e}")
            result.invalid_packets = len(packets)
            result.error_messages.append(str(e))
            return result
    
    def _combine_features(self, result: NetworkAnalysisResult) -> Tuple[Dict[str, np.ndarray], List[str], List[FeatureType]]:
        """Combine all extracted features"""
        features = {}
        feature_names = []
        feature_types = []
        
        # Add flow features
        if result.flow_features.size > 0:
            features['flow_features'] = result.flow_features
            flow_names = [f'flow_{i}' for i in range(len(result.flow_features))]
            feature_names.extend(flow_names)
            feature_types.extend([FeatureType.NUMERICAL] * len(result.flow_features))
        
        # Add packet features
        if result.packet_features.size > 0:
            features['packet_features'] = result.packet_features
            packet_names = [f'packet_{i}' for i in range(len(result.packet_features))]
            feature_names.extend(packet_names)
            feature_types.extend([FeatureType.NUMERICAL] * len(result.packet_features))
        
        # Add anomaly features
        if result.anomaly_features.size > 0:
            features['anomaly_features'] = result.anomaly_features
            anomaly_names = [f'anomaly_{i}' for i in range(len(result.anomaly_features))]
            feature_names.extend(anomaly_names)
            feature_types.extend([FeatureType.NUMERICAL] * len(result.anomaly_features))
        
        # Add combined features if multiple types are available
        if len(features) > 1:
            all_features = []
            for feature_array in features.values():
                all_features.extend(feature_array)
            features['combined_network'] = np.array(all_features)
        
        return features, feature_names, feature_types
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get plugin capabilities"""
        return {
            'flow_analysis': self.use_flow_analysis,
            'packet_analysis': self.use_packet_analysis,
            'anomaly_detection': self.use_anomaly_detection,
            'intrusion_detection': True,
            'ddos_detection': True,
            'port_scan_detection': True,
            'batch_processing': True,
            'pcap_support': SCAPY_AVAILABLE
        }
    
    def get_hooks(self) -> Dict[str, Callable]:
        """Get plugin hooks"""
        return {
            'before_extraction': self._before_extraction_hook,
            'after_extraction': self._after_extraction_hook
        }
    
    def _before_extraction_hook(self, input_data: Any) -> Any:
        """Hook called before feature extraction"""
        self.logger.debug("🔗 Before network feature extraction")
        return input_data
    
    def _after_extraction_hook(self, result: FeatureExtractionResult) -> FeatureExtractionResult:
        """Hook called after feature extraction"""
        self.logger.debug("🔗 After network feature extraction")
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear feature cache"""
        with self.cache_lock:
            self.feature_cache.clear()
        self.logger.info("🧹 Cleared network feature cache")
    
    def health_check(self) -> bool:
        """Perform health check"""
        try:
            # Check if Scapy is available and working
            if not SCAPY_AVAILABLE:
                return False
            
            # Test basic packet creation
            test_packet = IP(src="192.168.1.1", dst="192.168.1.2")/TCP(sport=80, dport=443)
            
            # Test analyzers
            if self.flow_analyzer:
                self.flow_analyzer.process_packet(test_packet, time.time())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False


# Plugin metadata for discovery
__plugin_metadata__ = {
    "name": "NetworkFeatureExtractor",
    "version": "1.0.0",
    "author": "Universal AI Core", 
    "description": "Network traffic analysis and intrusion detection features",
    "plugin_type": "feature_extractor",
    "entry_point": f"{__name__}:NetworkFeatureExtractor",
    "dependencies": [
        {"name": "scapy", "optional": True}
    ],
    "capabilities": ["flow_analysis", "packet_analysis", "anomaly_detection", "intrusion_detection"],
    "hooks": ["before_extraction", "after_extraction"]
}


if __name__ == "__main__":
    # Test the network feature extractor
    print("🌐 NETWORK FEATURE EXTRACTOR TEST")
    print("=" * 50)
    
    # Test configuration
    config = {
        'use_flow_analysis': True,
        'use_packet_analysis': True,
        'use_anomaly_detection': True,
        'max_packets': 1000
    }
    
    # Initialize extractor
    extractor = NetworkFeatureExtractor(config)
    
    # Create test network traffic
    if SCAPY_AVAILABLE:
        print(f"\n🌐 Creating test network traffic...")
        test_packets = []
        
        # Generate some test packets
        for i in range(10):
            # TCP packet
            tcp_pkt = IP(src=f"192.168.1.{i+1}", dst="192.168.1.100")/TCP(sport=1024+i, dport=80)
            test_packets.append((tcp_pkt, time.time() + i * 0.1))
            
            # UDP packet  
            udp_pkt = IP(src=f"192.168.1.{i+1}", dst="192.168.1.200")/UDP(sport=2048+i, dport=53)
            test_packets.append((udp_pkt, time.time() + i * 0.1 + 0.05))
        
        # Test feature extraction
        print(f"\n🔍 Testing with {len(test_packets)} packets...")
        result = extractor.extract_features(test_packets)
        
        if result.success:
            print(f"✅ Extraction successful!")
            print(f"📊 Processing time: {result.processing_time:.3f}s")
            print(f"🧮 Features extracted:")
            for feature_type, features in result.features.items():
                if isinstance(features, np.ndarray):
                    print(f"  {feature_type}: {features.shape}")
                else:
                    print(f"  {feature_type}: {type(features)}")
            
            # Show metadata
            if result.metadata:
                print(f"📋 Metadata:")
                for key, value in result.metadata.items():
                    print(f"  {key}: {value}")
        else:
            print(f"❌ Extraction failed: {result.error_message}")
    else:
        print("❌ Scapy not available - skipping packet test")
        
        # Test with dummy data
        result = extractor.extract_features([])
        print(f"Test with empty data: {'✅' if not result.success else '❌'}")
    
    # Test health check
    health = extractor.health_check()
    print(f"\n🏥 Health check: {'✅' if health else '❌'}")
    
    # Show capabilities
    capabilities = extractor.get_capabilities()
    print(f"\n🔧 Capabilities:")
    for key, value in capabilities.items():
        print(f"  {key}: {value}")
    
    # Show statistics
    stats = extractor.get_statistics()
    print(f"\n📊 Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Network feature extractor test completed!")