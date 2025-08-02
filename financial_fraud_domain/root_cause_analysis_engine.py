"""
Root Cause Analysis Engine for Accuracy Degradation Investigation
Advanced causal inference, decision tree analysis, and temporal pattern detection
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
import time
import threading
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.tree import DecisionTreeClassifier, export_text, _tree
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Import accuracy tracking components
try:
    from accuracy_tracking_db import MetricType
except ImportError:
    # Fallback for missing MetricType
    class MetricType:
        ACCURACY = "accuracy"


class RootCauseAnalysisEngine:
    """
    Engine for performing comprehensive root cause analysis.
    Supports causal inference, decision trees, correlation analysis, and timeline analysis.
    """
    
    def __init__(self, logger: logging.Logger = None):
        """Initialize the root cause analysis engine."""
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._lock = threading.Lock()
    
    def perform_root_cause_analysis(self, accuracy_degradation_events: Dict[str, Any], 
                                   causal_analysis_config: Dict[str, Any],
                                   monitoring_system=None, accuracy_db=None) -> Dict[str, Any]:
        """Perform comprehensive root cause analysis"""
        with self._lock:
            # Prepare data for analysis
            prepared_data = self._prepare_root_cause_data(accuracy_degradation_events, monitoring_system, accuracy_db)
            
            # Get analysis methods
            methods = causal_analysis_config.get("methods", 
                ["causal_inference", "decision_trees", "correlation_analysis", "timeline_analysis"])
            
            # Initialize results structure
            analysis_results = {
                "timeline_analysis": None,
                "correlation_analysis": None,
                "causal_inference": None,
                "decision_tree_analysis": None
            }
            
            # Perform each analysis
            if "timeline_analysis" in methods:
                try:
                    analysis_results["timeline_analysis"] = self._perform_timeline_analysis(prepared_data)
                except Exception as e:
                    self.logger.warning(f"Timeline analysis failed: {e}")
                    analysis_results["timeline_analysis"] = {"status": "failed", "error": str(e)}
            
            if "correlation_analysis" in methods:
                try:
                    analysis_results["correlation_analysis"] = self._perform_degradation_correlation_analysis(
                        prepared_data, analysis_results["timeline_analysis"]
                    )
                except Exception as e:
                    self.logger.warning(f"Correlation analysis failed: {e}")
                    analysis_results["correlation_analysis"] = {"status": "failed", "error": str(e)}
            
            if "causal_inference" in methods:
                try:
                    analysis_results["causal_inference"] = self._perform_causal_inference(
                        prepared_data, analysis_results["correlation_analysis"], causal_analysis_config
                    )
                except Exception as e:
                    self.logger.warning(f"Causal inference failed: {e}")
                    analysis_results["causal_inference"] = {"status": "failed", "error": str(e)}
            
            if "decision_trees" in methods:
                try:
                    analysis_results["decision_tree_analysis"] = self._perform_decision_tree_analysis(
                        prepared_data, causal_analysis_config
                    )
                except Exception as e:
                    self.logger.warning(f"Decision tree analysis failed: {e}")
                    analysis_results["decision_tree_analysis"] = {"status": "failed", "error": str(e)}
            
            return {
                "analysis_results": analysis_results,
                "prepared_data_info": {
                    "total_events": prepared_data["total_events"],
                    "time_range": prepared_data.get("time_range", {}),
                    "models_affected": list(prepared_data["models"].keys())
                }
            }
    
    def _prepare_root_cause_data(self, accuracy_degradation_events: Dict[str, Any],
                                monitoring_system=None, accuracy_db=None) -> Dict[str, Any]:
        """Prepare degradation event data for analysis"""
        prepared = {
            "events": [],
            "models": {},
            "metrics": {},
            "time_range": {},
            "total_events": 0,
            "event_types": {},
            "context_data": {}
        }
        
        # Process events
        events = accuracy_degradation_events.get("events", [])
        
        for event in events:
            # Parse timestamp
            try:
                timestamp = pd.to_datetime(event.get("timestamp"))
            except:
                self.logger.warning(f"Invalid timestamp in event: {event}")
                continue
            
            # Create structured event
            structured_event = {
                "timestamp": timestamp,
                "model_id": event.get("model_id", "unknown"),
                "metric": event.get("metric", "accuracy"),
                "value": event.get("value"),
                "previous_value": event.get("previous_value"),
                "degradation_amount": None,
                "degradation_rate": None,
                "event_type": event.get("event_type", "degradation"),
                "severity": event.get("severity", "unknown"),
                "context": event.get("context", {})
            }
            
            # Calculate degradation metrics
            if structured_event["value"] is not None and structured_event["previous_value"] is not None:
                structured_event["degradation_amount"] = structured_event["previous_value"] - structured_event["value"]
                if structured_event["previous_value"] != 0:
                    structured_event["degradation_rate"] = (
                        structured_event["degradation_amount"] / structured_event["previous_value"]
                    )
            
            prepared["events"].append(structured_event)
            
            # Track by model and metric
            model_id = structured_event["model_id"]
            if model_id not in prepared["models"]:
                prepared["models"][model_id] = []
            prepared["models"][model_id].append(structured_event)
            
            metric = structured_event["metric"]
            if metric not in prepared["metrics"]:
                prepared["metrics"][metric] = []
            prepared["metrics"][metric].append(structured_event)
            
            # Track event types
            event_type = structured_event["event_type"]
            prepared["event_types"][event_type] = prepared["event_types"].get(event_type, 0) + 1
        
        # Sort events by timestamp
        prepared["events"].sort(key=lambda x: x["timestamp"])
        prepared["total_events"] = len(prepared["events"])
        
        # Determine time range
        if prepared["events"]:
            prepared["time_range"] = {
                "start": prepared["events"][0]["timestamp"],
                "end": prepared["events"][-1]["timestamp"],
                "duration_days": (prepared["events"][-1]["timestamp"] - prepared["events"][0]["timestamp"]).days
            }
        
        # Retrieve context data
        if prepared["time_range"]:
            try:
                system_events = self._retrieve_system_events(prepared["time_range"], monitoring_system)
                prepared["context_data"]["system_events"] = system_events
                
                performance_metrics = self._retrieve_performance_metrics(
                    prepared["models"].keys(), prepared["time_range"], accuracy_db
                )
                prepared["context_data"]["performance_metrics"] = performance_metrics
                
            except Exception as e:
                self.logger.warning(f"Failed to retrieve context data: {e}")
        
        return prepared
    
    def _retrieve_system_events(self, time_range: Dict[str, Any], monitoring_system=None) -> List[Dict[str, Any]]:
        """Retrieve system events from monitoring system"""
        if not monitoring_system:
            return self._generate_synthetic_system_events(time_range)
        
        try:
            events = monitoring_system.get_events({
                "start_time": time_range["start"].isoformat(),
                "end_time": time_range["end"].isoformat(),
                "event_types": ["deployment", "config_change", "system_error", "data_drift", "resource_alert"]
            })
            return events
        except:
            return self._generate_synthetic_system_events(time_range)
    
    def _generate_synthetic_system_events(self, time_range: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate synthetic system events for testing"""
        events = []
        
        event_templates = [
            {"type": "deployment", "component": "model_server", "impact": "high"},
            {"type": "config_change", "component": "feature_pipeline", "impact": "medium"},
            {"type": "data_drift", "component": "input_data", "impact": "high"},
            {"type": "system_error", "component": "database", "impact": "low"},
            {"type": "resource_alert", "component": "cpu_usage", "impact": "medium"}
        ]
        
        num_events = np.random.poisson(5)
        
        for _ in range(num_events):
            template = np.random.choice(event_templates)
            
            time_offset = np.random.uniform(0, 1)
            timestamp = time_range["start"] + (time_range["end"] - time_range["start"]) * time_offset
            
            events.append({
                "timestamp": timestamp.isoformat(),
                "event_type": template["type"],
                "component": template["component"],
                "impact_level": template["impact"],
                "details": {
                    "description": f"{template['type']} in {template['component']}",
                    "duration_minutes": np.random.randint(5, 120)
                }
            })
        
        return sorted(events, key=lambda x: x["timestamp"])
    
    def _retrieve_performance_metrics(self, model_ids: List[str], time_range: Dict[str, Any], 
                                     accuracy_db=None) -> Dict[str, Any]:
        """Retrieve performance metrics for models"""
        if not accuracy_db:
            return self._generate_synthetic_performance_metrics(model_ids, time_range)
        
        try:
            metrics = {}
            for model_id in model_ids:
                model_metrics = accuracy_db.query_accuracy_metrics({
                    "model_id": model_id,
                    "start_time": time_range["start"].isoformat(),
                    "end_time": time_range["end"].isoformat()
                })
                metrics[model_id] = model_metrics
            return metrics
        except:
            return self._generate_synthetic_performance_metrics(model_ids, time_range)
    
    def _generate_synthetic_performance_metrics(self, model_ids: List[str], 
                                               time_range: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthetic performance metrics - FALLBACK ONLY"""
        # Explicit warning for synthetic data usage
        self.logger.warning(
            f"USING DUMMY DATA: No real performance metrics available for models {model_ids}. "
            f"Synthetic performance data should NOT be used for actual root cause analysis or production decisions."
        )
        
        metrics = {}
        
        for model_id in model_ids:
            num_points = min(100, time_range.get("duration_days", 30))
            timestamps = pd.date_range(time_range["start"], time_range["end"], periods=num_points)
            
            base_accuracy = 0.85 + np.random.random() * 0.1
            degradation_rate = np.random.uniform(0.0001, 0.001)
            noise_level = 0.01
            
            values = []
            for i, ts in enumerate(timestamps):
                value = base_accuracy - degradation_rate * i + np.random.normal(0, noise_level)
                values.append(max(0.0, min(1.0, value)))
            
            metrics[model_id] = {
                "timestamps": [ts.isoformat() for ts in timestamps],
                "accuracy": values,
                "sample_count": [1000 + np.random.randint(-100, 100) for _ in range(num_points)]
            }
        
        return metrics
    
    def _get_real_performance_metrics(self, model_ids: List[str], time_range: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Attempt to retrieve real performance metrics from accuracy database"""
        try:
            if not hasattr(self, 'accuracy_db') or not self.accuracy_db:
                return None
            
            start_date = datetime.fromisoformat(time_range["start"])
            end_date = datetime.fromisoformat(time_range["end"])
            
            all_metrics = {}
            
            for model_id in model_ids:
                metrics = self.accuracy_db.get_accuracy_metrics(
                    model_id=model_id,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if metrics:
                    accuracy_values = [float(m.metric_value) for m in metrics if m.metric_type == MetricType.ACCURACY]
                    timestamps = [m.timestamp.isoformat() for m in metrics if m.metric_type == MetricType.ACCURACY]
                    
                    if accuracy_values and len(accuracy_values) > 10:  # Require minimum data
                        all_metrics[model_id] = {
                            "timestamps": timestamps,
                            "accuracy": accuracy_values,
                            "sample_count": [m.sample_size or 1000 for m in metrics if m.metric_type == MetricType.ACCURACY]
                        }
            
            return all_metrics if all_metrics else None
            
        except Exception as e:
            self.logger.debug(f"Could not retrieve real performance metrics: {e}")
            return None
    
    def _perform_timeline_analysis(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform timeline analysis to establish temporal relationships"""
        timeline_results = {
            "event_sequences": [],
            "temporal_clusters": [],
            "event_patterns": {},
            "critical_periods": [],
            "timeline_summary": {}
        }
        
        if not prepared_data["events"]:
            return timeline_results
        
        # Identify event sequences
        sequences = self._identify_event_sequences(prepared_data["events"])
        timeline_results["event_sequences"] = sequences
        
        # Find temporal clusters
        clusters = self._find_temporal_clusters(prepared_data["events"])
        timeline_results["temporal_clusters"] = clusters
        
        # Detect event patterns
        patterns = self._detect_temporal_patterns(prepared_data["events"], sequences)
        timeline_results["event_patterns"] = patterns
        
        # Identify critical periods
        critical_periods = self._identify_critical_periods(
            prepared_data["events"], clusters, prepared_data.get("context_data", {})
        )
        timeline_results["critical_periods"] = critical_periods
        
        # Generate timeline summary
        summary = self._generate_timeline_summary(sequences, clusters, patterns, critical_periods)
        timeline_results["timeline_summary"] = summary
        
        return timeline_results
    
    def _identify_event_sequences(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify sequences of related events"""
        sequences = []
        
        # Group events by model and metric
        event_groups = {}
        for event in events:
            key = (event["model_id"], event["metric"])
            if key not in event_groups:
                event_groups[key] = []
            event_groups[key].append(event)
        
        # Analyze each group for sequences
        for (model_id, metric), group_events in event_groups.items():
            if len(group_events) < 2:
                continue
            
            group_events.sort(key=lambda x: x["timestamp"])
            
            current_sequence = [group_events[0]]
            
            for i in range(1, len(group_events)):
                prev_event = group_events[i-1]
                curr_event = group_events[i]
                
                time_diff = (curr_event["timestamp"] - prev_event["timestamp"]).days
                
                if time_diff <= 7:
                    current_sequence.append(curr_event)
                else:
                    if len(current_sequence) >= 2:
                        sequences.append({
                            "model_id": model_id,
                            "metric": metric,
                            "events": current_sequence,
                            "duration_days": (current_sequence[-1]["timestamp"] - 
                                            current_sequence[0]["timestamp"]).days,
                            "total_degradation": sum(e.get("degradation_amount", 0) 
                                                   for e in current_sequence),
                            "sequence_type": self._classify_sequence(current_sequence)
                        })
                    current_sequence = [curr_event]
            
            if len(current_sequence) >= 2:
                sequences.append({
                    "model_id": model_id,
                    "metric": metric,
                    "events": current_sequence,
                    "duration_days": (current_sequence[-1]["timestamp"] - 
                                    current_sequence[0]["timestamp"]).days,
                    "total_degradation": sum(e.get("degradation_amount", 0) 
                                           for e in current_sequence),
                    "sequence_type": self._classify_sequence(current_sequence)
                })
        
        return sequences
    
    def _classify_sequence(self, sequence: List[Dict[str, Any]]) -> str:
        """Classify the type of event sequence"""
        if len(sequence) < 2:
            return "single"
        
        degradations = [e.get("degradation_amount", 0) for e in sequence]
        
        if all(d > 0 for d in degradations):
            if len(degradations) >= 3:
                diffs = [degradations[i+1] - degradations[i] for i in range(len(degradations)-1)]
                if all(d > 0 for d in diffs):
                    return "accelerating_degradation"
                elif all(d < 0 for d in diffs):
                    return "decelerating_degradation"
            return "continuous_degradation"
        elif any(d < 0 for d in degradations):
            return "mixed_performance"
        
        return "stable_degradation"
    
    def _find_temporal_clusters(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find temporal clusters of events"""
        if len(events) < 2:
            return []
        
        # Convert timestamps to numeric values
        first_timestamp = events[0]["timestamp"]
        time_features = np.array([
            [(e["timestamp"] - first_timestamp).total_seconds() / 3600]
            for e in events
        ])
        
        # Perform clustering
        clustering = DBSCAN(eps=24, min_samples=2)
        cluster_labels = clustering.fit_predict(time_features)
        
        clusters = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:
                continue
            
            cluster_events = [events[i] for i, l in enumerate(cluster_labels) if l == label]
            cluster_models = set(e["model_id"] for e in cluster_events)
            cluster_metrics = set(e["metric"] for e in cluster_events)
            
            clusters.append({
                "cluster_id": int(label),
                "events": cluster_events,
                "event_count": len(cluster_events),
                "time_span_hours": (cluster_events[-1]["timestamp"] - 
                                  cluster_events[0]["timestamp"]).total_seconds() / 3600,
                "affected_models": list(cluster_models),
                "affected_metrics": list(cluster_metrics),
                "cluster_type": "multi_model" if len(cluster_models) > 1 else "single_model",
                "severity": self._assess_cluster_severity(cluster_events)
            })
        
        return sorted(clusters, key=lambda x: x["events"][0]["timestamp"])
    
    def _assess_cluster_severity(self, events: List[Dict[str, Any]]) -> str:
        """Assess severity of event cluster"""
        degradations = [e.get("degradation_amount", 0) for e in events if e.get("degradation_amount")]
        
        if not degradations:
            return "unknown"
        
        avg_degradation = np.mean(degradations)
        max_degradation = np.max(degradations)
        
        if max_degradation > 0.1 or avg_degradation > 0.05:
            return "critical"
        elif max_degradation > 0.05 or avg_degradation > 0.02:
            return "high"
        elif max_degradation > 0.02 or avg_degradation > 0.01:
            return "medium"
        else:
            return "low"
    
    def _detect_temporal_patterns(self, events: List[Dict[str, Any]], 
                                 sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect temporal patterns in events"""
        patterns = {
            "recurring_patterns": [],
            "periodicity_analysis": {},
            "cascade_effects": [],
            "pattern_summary": {}
        }
        
        # Check for recurring patterns
        hourly_pattern = {}
        daily_pattern = {}
        
        for event in events:
            hour = event["timestamp"].hour
            day = event["timestamp"].dayofweek
            
            hourly_pattern[hour] = hourly_pattern.get(hour, 0) + 1
            daily_pattern[day] = daily_pattern.get(day, 0) + 1
        
        # Identify peak hours/days
        if hourly_pattern:
            peak_hour = max(hourly_pattern.items(), key=lambda x: x[1])
            if peak_hour[1] > len(events) * 0.2:
                patterns["recurring_patterns"].append({
                    "type": "hourly",
                    "peak_hour": peak_hour[0],
                    "concentration": peak_hour[1] / len(events),
                    "pattern": "degradation_peaks_at_specific_hour"
                })
        
        if daily_pattern:
            peak_day = max(daily_pattern.items(), key=lambda x: x[1])
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            if peak_day[1] > len(events) * 0.2:
                patterns["recurring_patterns"].append({
                    "type": "daily",
                    "peak_day": day_names[peak_day[0]],
                    "concentration": peak_day[1] / len(events),
                    "pattern": "degradation_peaks_on_specific_day"
                })
        
        # Cascade effects
        cascade_effects = self._identify_cascade_effects(events, sequences)
        patterns["cascade_effects"] = cascade_effects
        
        patterns["pattern_summary"] = {
            "has_recurring_patterns": len(patterns["recurring_patterns"]) > 0,
            "has_cascade_effects": len(cascade_effects) > 0,
            "dominant_pattern": self._identify_dominant_pattern(patterns)
        }
        
        return patterns
    
    def _identify_cascade_effects(self, events: List[Dict[str, Any]], 
                                 sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify cascade effects"""
        cascade_effects = []
        
        for i, trigger_event in enumerate(events[:-1]):
            subsequent_events = []
            
            for j in range(i + 1, len(events)):
                following_event = events[j]
                time_diff = (following_event["timestamp"] - trigger_event["timestamp"]).total_seconds() / 3600
                
                if time_diff > 24:
                    break
                
                if (following_event["model_id"] != trigger_event["model_id"] or 
                    following_event["metric"] != trigger_event["metric"]):
                    subsequent_events.append({
                        "event": following_event,
                        "time_lag_hours": time_diff
                    })
            
            if len(subsequent_events) >= 2:
                cascade_effects.append({
                    "trigger_event": trigger_event,
                    "subsequent_events": subsequent_events,
                    "affected_models": list(set(e["event"]["model_id"] for e in subsequent_events)),
                    "cascade_span_hours": max(e["time_lag_hours"] for e in subsequent_events),
                    "cascade_type": "multi_model_cascade" if len(set(e["event"]["model_id"] 
                                                                    for e in subsequent_events)) > 1 
                                                         else "single_model_cascade"
                })
        
        return cascade_effects
    
    def _identify_dominant_pattern(self, patterns: Dict[str, Any]) -> str:
        """Identify the dominant temporal pattern"""
        if patterns["cascade_effects"]:
            return "cascade_degradation"
        elif patterns["recurring_patterns"]:
            return "recurring_degradation"
        else:
            return "sporadic_degradation"
    
    def _identify_critical_periods(self, events: List[Dict[str, Any]], 
                                  clusters: List[Dict[str, Any]], 
                                  context_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify critical time periods"""
        critical_periods = []
        
        if events:
            window_hours = 24
            window_slide = 6
            
            start_time = events[0]["timestamp"]
            end_time = events[-1]["timestamp"]
            
            current_time = start_time
            while current_time < end_time:
                window_end = current_time + timedelta(hours=window_hours)
                
                window_events = [e for e in events if current_time <= e["timestamp"] < window_end]
                
                if len(window_events) >= 3:
                    merged = False
                    for period in critical_periods:
                        if (current_time <= period["end_time"] and 
                            window_end >= period["start_time"]):
                            period["start_time"] = min(period["start_time"], current_time)
                            period["end_time"] = max(period["end_time"], window_end)
                            period["events"].extend(window_events)
                            period["event_count"] = len(set(tuple(e.items()) for e in period["events"]))
                            merged = True
                            break
                    
                    if not merged:
                        critical_periods.append({
                            "start_time": current_time,
                            "end_time": window_end,
                            "events": window_events,
                            "event_count": len(window_events),
                            "severity": self._assess_period_severity(window_events),
                            "affected_models": list(set(e["model_id"] for e in window_events))
                        })
                
                current_time += timedelta(hours=window_slide)
        
        # Add context from system events
        if context_data.get("system_events"):
            for period in critical_periods:
                overlapping_system_events = []
                for sys_event in context_data["system_events"]:
                    sys_timestamp = pd.to_datetime(sys_event["timestamp"])
                    if period["start_time"] <= sys_timestamp <= period["end_time"]:
                        overlapping_system_events.append(sys_event)
                
                period["system_events"] = overlapping_system_events
                period["has_system_events"] = len(overlapping_system_events) > 0
        
        return sorted(critical_periods, key=lambda x: x["event_count"], reverse=True)
    
    def _assess_period_severity(self, events: List[Dict[str, Any]]) -> str:
        """Assess severity of a time period"""
        if not events:
            return "low"
        
        event_count = len(events)
        avg_degradation = np.mean([e.get("degradation_amount", 0) for e in events 
                                  if e.get("degradation_amount")])
        
        if event_count >= 5 or avg_degradation > 0.05:
            return "critical"
        elif event_count >= 3 or avg_degradation > 0.02:
            return "high"
        elif event_count >= 2 or avg_degradation > 0.01:
            return "medium"
        else:
            return "low"
    
    def _generate_timeline_summary(self, sequences: List[Dict[str, Any]], 
                                  clusters: List[Dict[str, Any]], 
                                  patterns: Dict[str, Any], 
                                  critical_periods: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of timeline analysis"""
        summary = {
            "sequence_count": len(sequences),
            "cluster_count": len(clusters),
            "critical_period_count": len(critical_periods),
            "temporal_characteristics": []
        }
        
        if sequences:
            sequence_types = [s["sequence_type"] for s in sequences]
            if sequence_types:
                most_common_type = max(set(sequence_types), key=sequence_types.count)
                summary["temporal_characteristics"].append(
                    f"Dominant sequence type: {most_common_type}"
                )
        
        if clusters:
            multi_model_clusters = [c for c in clusters if c["cluster_type"] == "multi_model"]
            if multi_model_clusters:
                summary["temporal_characteristics"].append(
                    f"{len(multi_model_clusters)} multi-model event clusters detected"
                )
        
        if patterns.get("cascade_effects"):
            summary["temporal_characteristics"].append(
                f"{len(patterns['cascade_effects'])} cascade effects identified"
            )
        
        if patterns.get("recurring_patterns"):
            for pattern in patterns["recurring_patterns"]:
                summary["temporal_characteristics"].append(
                    f"Recurring {pattern['type']} pattern detected"
                )
        
        return summary
    
    def _perform_degradation_correlation_analysis(self, prepared_data: Dict[str, Any], 
                                                 timeline_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform correlation analysis on degradation events"""
        correlation_results = {
            "event_correlations": {},
            "factor_correlations": {},
            "cross_model_correlations": {},
            "environmental_correlations": {},
            "correlation_summary": {}
        }
        
        # Event-to-event correlations
        event_correlations = self._analyze_event_correlations(prepared_data["events"])
        correlation_results["event_correlations"] = event_correlations
        
        # Factor correlations
        if prepared_data.get("context_data"):
            factor_correlations = self._analyze_factor_correlations(
                prepared_data["events"], prepared_data["context_data"]
            )
            correlation_results["factor_correlations"] = factor_correlations
        
        # Cross-model correlations
        if len(prepared_data["models"]) > 1:
            cross_model = self._analyze_cross_model_correlations(prepared_data["models"])
            correlation_results["cross_model_correlations"] = cross_model
        
        # Environmental correlations
        environmental = self._analyze_environmental_correlations(
            prepared_data["events"], prepared_data.get("context_data", {}), timeline_results
        )
        correlation_results["environmental_correlations"] = environmental
        
        # Generate summary
        summary = self._generate_correlation_summary(correlation_results)
        correlation_results["correlation_summary"] = summary
        
        return correlation_results
    
    def _analyze_event_correlations(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlations between different types of events"""
        event_correlations = {
            "severity_correlations": {},
            "metric_correlations": {},
            "temporal_correlations": {}
        }
        
        by_severity = {}
        by_metric = {}
        
        for event in events:
            severity = event.get("severity", "unknown")
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(event)
            
            metric = event.get("metric", "unknown")
            if metric not in by_metric:
                by_metric[metric] = []
            by_metric[metric].append(event)
        
        # Analyze severity patterns
        if len(by_severity) > 1:
            for severity, sev_events in by_severity.items():
                if len(sev_events) >= 2:
                    time_diffs = []
                    for i in range(1, len(sev_events)):
                        diff = (sev_events[i]["timestamp"] - sev_events[i-1]["timestamp"]).total_seconds() / 3600
                        time_diffs.append(diff)
                    
                    if time_diffs:
                        avg_time_diff = np.mean(time_diffs)
                        event_correlations["severity_correlations"][severity] = {
                            "event_count": len(sev_events),
                            "avg_time_between_events_hours": float(avg_time_diff),
                            "clustering_tendency": "high" if avg_time_diff < 24 else "low"
                        }
        
        return event_correlations
    
    def _analyze_factor_correlations(self, events: List[Dict[str, Any]], 
                                   context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlations with contextual factors"""
        factor_correlations = {
            "system_event_correlations": {},
            "performance_metric_correlations": {},
            "identified_factors": []
        }
        
        # Correlate with system events
        if context_data.get("system_events"):
            system_events = context_data["system_events"]
            
            system_event_types = {}
            for sys_event in system_events:
                event_type = sys_event.get("event_type", "unknown")
                if event_type not in system_event_types:
                    system_event_types[event_type] = []
                system_event_types[event_type].append(sys_event)
            
            for event_type, type_events in system_event_types.items():
                correlated_degradations = 0
                
                for sys_event in type_events:
                    sys_timestamp = pd.to_datetime(sys_event["timestamp"])
                    
                    for deg_event in events:
                        time_diff = abs((deg_event["timestamp"] - sys_timestamp).total_seconds() / 3600)
                        if time_diff <= 24:
                            correlated_degradations += 1
                
                if correlated_degradations > 0:
                    correlation_ratio = correlated_degradations / len(events)
                    factor_correlations["system_event_correlations"][event_type] = {
                        "system_event_count": len(type_events),
                        "correlated_degradation_count": correlated_degradations,
                        "correlation_ratio": float(correlation_ratio),
                        "potential_cause": correlation_ratio > 0.3
                    }
                    
                    if correlation_ratio > 0.3:
                        factor_correlations["identified_factors"].append({
                            "factor_type": "system_event",
                            "factor_name": event_type,
                            "correlation_strength": float(correlation_ratio),
                            "confidence": "high" if correlation_ratio > 0.5 else "medium"
                        })
        
        return factor_correlations
    
    def _analyze_cross_model_correlations(self, models_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze correlations across different models"""
        cross_model_correlations = {
            "simultaneous_degradations": [],
            "model_interaction_matrix": {},
            "shared_root_causes": []
        }
        
        model_ids = list(models_data.keys())
        interaction_matrix = np.zeros((len(model_ids), len(model_ids)))
        
        for i, model1 in enumerate(model_ids):
            for j, model2 in enumerate(model_ids):
                if i != j:
                    simultaneous_count = 0
                    
                    for event1 in models_data[model1]:
                        for event2 in models_data[model2]:
                            time_diff = abs((event1["timestamp"] - event2["timestamp"]).total_seconds() / 3600)
                            if time_diff <= 1:
                                simultaneous_count += 1
                    
                    if simultaneous_count > 0:
                        interaction_score = simultaneous_count / min(len(models_data[model1]), 
                                                                    len(models_data[model2]))
                        interaction_matrix[i, j] = interaction_score
                        
                        if interaction_score > 0.3:
                            cross_model_correlations["simultaneous_degradations"].append({
                                "model_pair": [model1, model2],
                                "simultaneous_events": simultaneous_count,
                                "interaction_score": float(interaction_score),
                                "correlation_type": "strong" if interaction_score > 0.5 else "moderate"
                            })
        
        cross_model_correlations["model_interaction_matrix"] = {
            "model_ids": model_ids,
            "matrix": interaction_matrix.tolist()
        }
        
        return cross_model_correlations
    
    def _analyze_environmental_correlations(self, events: List[Dict[str, Any]], 
                                          context_data: Dict[str, Any], 
                                          timeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlations with environmental factors"""
        environmental_correlations = {
            "temporal_patterns": {},
            "workload_correlations": {},
            "infrastructure_correlations": {}
        }
        
        # Time-based patterns
        hourly_degradations = {}
        daily_degradations = {}
        
        for event in events:
            hour = event["timestamp"].hour
            day = event["timestamp"].dayofweek
            
            hourly_degradations[hour] = hourly_degradations.get(hour, 0) + 1
            daily_degradations[day] = daily_degradations.get(day, 0) + 1
        
        # Identify peak periods
        if hourly_degradations:
            peak_hours = sorted(hourly_degradations.items(), key=lambda x: x[1], reverse=True)[:3]
            environmental_correlations["temporal_patterns"]["peak_hours"] = [
                {"hour": hour, "event_count": count, "percentage": count/len(events)}
                for hour, count in peak_hours
            ]
        
        if daily_degradations:
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            peak_days = sorted(daily_degradations.items(), key=lambda x: x[1], reverse=True)[:3]
            environmental_correlations["temporal_patterns"]["peak_days"] = [
                {"day": day_names[day], "event_count": count, "percentage": count/len(events)}
                for day, count in peak_days
            ]
        
        return environmental_correlations
    
    def _generate_correlation_summary(self, correlation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of correlation analysis"""
        summary = {
            "strongest_correlations": [],
            "identified_patterns": [],
            "correlation_insights": []
        }
        
        # Extract strongest correlations
        all_correlations = []
        
        if correlation_results.get("factor_correlations", {}).get("identified_factors"):
            for factor in correlation_results["factor_correlations"]["identified_factors"]:
                all_correlations.append({
                    "type": factor["factor_type"],
                    "name": factor["factor_name"],
                    "strength": factor["correlation_strength"],
                    "source": "factor_analysis"
                })
        
        if correlation_results.get("cross_model_correlations", {}).get("simultaneous_degradations"):
            for degradation in correlation_results["cross_model_correlations"]["simultaneous_degradations"]:
                all_correlations.append({
                    "type": "cross_model",
                    "name": f"{degradation['model_pair'][0]}__{degradation['model_pair'][1]}",
                    "strength": degradation["interaction_score"],
                    "source": "cross_model_analysis"
                })
        
        all_correlations.sort(key=lambda x: x["strength"], reverse=True)
        summary["strongest_correlations"] = all_correlations[:5]
        
        if correlation_results.get("environmental_correlations", {}).get("temporal_patterns"):
            summary["identified_patterns"].append("temporal_concentration")
        
        if summary["strongest_correlations"]:
            summary["correlation_insights"].append(
                f"Found {len(summary['strongest_correlations'])} strong correlations"
            )
        
        return summary
    
    def _perform_causal_inference(self, prepared_data: Dict[str, Any], 
                                 correlation_results: Dict[str, Any], 
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform causal inference to determine causal relationships"""
        causal_results = {
            "causal_relationships": [],
            "causal_graph": {},
            "confidence_scores": {},
            "causal_chains": []
        }
        
        # Extract potential causes
        potential_causes = []
        
        if correlation_results and correlation_results.get("factor_correlations"):
            factors = correlation_results["factor_correlations"].get("identified_factors", [])
            for factor in factors:
                potential_causes.append({
                    "cause_type": factor["factor_type"],
                    "cause_name": factor["factor_name"],
                    "correlation_strength": factor["correlation_strength"]
                })
        
        # Add system events as potential causes
        if prepared_data.get("context_data", {}).get("system_events"):
            for sys_event in prepared_data["context_data"]["system_events"]:
                potential_causes.append({
                    "cause_type": "system_event",
                    "cause_name": sys_event.get("event_type"),
                    "cause_timestamp": sys_event.get("timestamp"),
                    "impact_level": sys_event.get("impact_level", "unknown")
                })
        
        # Perform causal analysis
        for event in prepared_data["events"][:50]:
            event_causes = self._infer_event_causes(event, potential_causes, prepared_data)
            
            for cause in event_causes:
                causal_results["causal_relationships"].append({
                    "cause": cause["cause"],
                    "effect": {
                        "model_id": event["model_id"],
                        "metric": event["metric"],
                        "degradation": event.get("degradation_amount", 0),
                        "timestamp": event["timestamp"].isoformat()
                    },
                    "confidence": cause["confidence"],
                    "evidence": cause["evidence"],
                    "causal_mechanism": cause.get("mechanism", "unknown")
                })
        
        # Build causal graph
        causal_graph = self._build_causal_graph(causal_results["causal_relationships"])
        causal_results["causal_graph"] = causal_graph
        
        # Calculate confidence scores
        confidence_scores = self._calculate_causal_confidence(causal_results["causal_relationships"])
        causal_results["confidence_scores"] = confidence_scores
        
        return causal_results
    
    def _infer_event_causes(self, event: Dict[str, Any], potential_causes: List[Dict[str, Any]], 
                          prepared_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Infer causes for a specific degradation event"""
        event_causes = []
        
        for cause in potential_causes:
            causal_score = 0
            evidence = []
            
            # Temporal proximity
            if "cause_timestamp" in cause:
                cause_time = pd.to_datetime(cause["cause_timestamp"])
                time_diff = (event["timestamp"] - cause_time).total_seconds() / 3600
                
                if 0 < time_diff < 24:
                    proximity_score = 1 / (1 + time_diff/24)
                    causal_score += proximity_score * 0.4
                    evidence.append(f"Temporal proximity: {time_diff:.1f} hours")
            
            # Correlation strength
            if "correlation_strength" in cause:
                causal_score += cause["correlation_strength"] * 0.3
                evidence.append(f"Correlation strength: {cause['correlation_strength']:.2f}")
            
            # Impact level match
            if cause.get("impact_level") == "high" and event.get("severity") in ["high", "critical"]:
                causal_score += 0.2
                evidence.append("Impact level matches severity")
            
            # Cause type specific scoring
            if cause["cause_type"] == "system_event":
                if cause["cause_name"] in ["deployment", "config_change"]:
                    causal_score += 0.1
                    evidence.append("High-risk system event type")
            
            # Determine mechanism
            mechanism = self._determine_causal_mechanism(cause, event)
            
            if causal_score > 0.3:
                event_causes.append({
                    "cause": cause,
                    "confidence": min(causal_score, 1.0),
                    "evidence": evidence,
                    "mechanism": mechanism
                })
        
        event_causes.sort(key=lambda x: x["confidence"], reverse=True)
        return event_causes[:3]
    
    def _determine_causal_mechanism(self, cause: Dict[str, Any], effect: Dict[str, Any]) -> str:
        """Determine the causal mechanism"""
        cause_type = cause.get("cause_type")
        cause_name = cause.get("cause_name", "")
        
        if cause_type == "system_event":
            if cause_name == "deployment":
                return "code_change_impact"
            elif cause_name == "config_change":
                return "configuration_drift"
            elif cause_name == "data_drift":
                return "distribution_shift"
            elif cause_name == "resource_alert":
                return "resource_constraint"
        elif cause_type == "data_quality":
            return "data_quality_degradation"
        elif cause_type == "cross_model":
            return "cascading_failure"
        
        return "unknown_mechanism"
    
    def _build_causal_graph(self, causal_relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a causal graph from relationships"""
        nodes = {}
        edges = []
        
        for relationship in causal_relationships:
            # Add cause node
            cause_id = f"{relationship['cause']['cause_type']}_{relationship['cause']['cause_name']}"
            if cause_id not in nodes:
                nodes[cause_id] = {
                    "id": cause_id,
                    "type": relationship["cause"]["cause_type"],
                    "name": relationship["cause"]["cause_name"],
                    "node_type": "cause"
                }
            
            # Add effect node
            effect_id = f"{relationship['effect']['model_id']}_{relationship['effect']['metric']}"
            if effect_id not in nodes:
                nodes[effect_id] = {
                    "id": effect_id,
                    "type": "degradation",
                    "model_id": relationship["effect"]["model_id"],
                    "metric": relationship["effect"]["metric"],
                    "node_type": "effect"
                }
            
            # Add edge
            edges.append({
                "source": cause_id,
                "target": effect_id,
                "confidence": relationship["confidence"],
                "mechanism": relationship["causal_mechanism"]
            })
        
        return {
            "nodes": list(nodes.values()),
            "edges": edges
        }
    
    def _calculate_causal_confidence(self, causal_relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate confidence scores for causal analysis"""
        confidence_scores = {
            "overall_confidence": 0,
            "confidence_by_cause_type": {},
            "high_confidence_relationships": 0
        }
        
        if not causal_relationships:
            return confidence_scores
        
        confidences = [r["confidence"] for r in causal_relationships]
        confidence_scores["overall_confidence"] = float(np.mean(confidences))
        
        # Group by cause type
        by_type = {}
        for relationship in causal_relationships:
            cause_type = relationship["cause"]["cause_type"]
            if cause_type not in by_type:
                by_type[cause_type] = []
            by_type[cause_type].append(relationship["confidence"])
        
        for cause_type, type_confidences in by_type.items():
            confidence_scores["confidence_by_cause_type"][cause_type] = {
                "mean_confidence": float(np.mean(type_confidences)),
                "count": len(type_confidences)
            }
        
        confidence_scores["high_confidence_relationships"] = sum(1 for c in confidences if c > 0.7)
        
        return confidence_scores
    
    def _perform_decision_tree_analysis(self, prepared_data: Dict[str, Any], 
                                       config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform decision tree analysis"""
        decision_tree_results = {
            "decision_trees": [],
            "important_features": [],
            "decision_rules": [],
            "tree_performance": {}
        }
        
        # Prepare features
        features_df = self._prepare_features_for_decision_tree(prepared_data)
        
        if features_df is None or len(features_df) < 10:
            return {
                "status": "insufficient_data",
                "message": "Not enough data for decision tree analysis"
            }
        
        # Build trees for different targets
        targets = ["has_degradation", "severity_level"]
        
        for target in targets:
            if target not in features_df.columns:
                continue
            
            tree_result = self._build_decision_tree(features_df, target, config)
            if tree_result:
                decision_tree_results["decision_trees"].append(tree_result)
        
        # Extract important features
        all_importances = {}
        for tree in decision_tree_results["decision_trees"]:
            for feature, importance in tree.get("feature_importances", {}).items():
                if feature not in all_importances:
                    all_importances[feature] = []
                all_importances[feature].append(importance)
        
        important_features = []
        for feature, importances in all_importances.items():
            avg_importance = np.mean(importances)
            important_features.append({
                "feature": feature,
                "importance": float(avg_importance),
                "appears_in_trees": len(importances)
            })
        
        important_features.sort(key=lambda x: x["importance"], reverse=True)
        decision_tree_results["important_features"] = important_features[:10]
        
        return decision_tree_results
    
    def _prepare_features_for_decision_tree(self, prepared_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare feature DataFrame for decision tree"""
        features_list = []
        
        for event in prepared_data["events"]:
            features = {
                "hour_of_day": event["timestamp"].hour,
                "day_of_week": event["timestamp"].dayofweek,
                "is_weekend": event["timestamp"].dayofweek >= 5,
                "model_id": event["model_id"],
                "metric": event["metric"],
                "has_degradation": event.get("degradation_amount", 0) > 0,
                "degradation_amount": event.get("degradation_amount", 0),
                "severity_level": self._encode_severity(event.get("severity", "unknown"))
            }
            
            # Add context features
            if prepared_data.get("context_data", {}).get("system_events"):
                recent_system_events = self._count_recent_system_events(
                    event["timestamp"], prepared_data["context_data"]["system_events"]
                )
                features.update(recent_system_events)
            
            features_list.append(features)
        
        if not features_list:
            return None
        
        df = pd.DataFrame(features_list)
        
        # Encode categorical variables
        categorical_columns = ["model_id", "metric"]
        for col in categorical_columns:
            if col in df.columns:
                df[f"{col}_encoded"] = pd.Categorical(df[col]).codes
                df.drop(col, axis=1, inplace=True)
        
        return df
    
    def _encode_severity(self, severity: str) -> int:
        """Encode severity level as numeric"""
        severity_map = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
            "unknown": 0
        }
        return severity_map.get(severity, 0)
    
    def _count_recent_system_events(self, event_timestamp: pd.Timestamp, 
                                   system_events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count system events in time windows"""
        counts = {
            "system_events_1h": 0,
            "system_events_24h": 0,
            "deployments_24h": 0,
            "config_changes_24h": 0,
            "errors_24h": 0
        }
        
        for sys_event in system_events:
            sys_timestamp = pd.to_datetime(sys_event["timestamp"])
            time_diff = (event_timestamp - sys_timestamp).total_seconds() / 3600
            
            if 0 < time_diff <= 24:
                counts["system_events_24h"] += 1
                
                if time_diff <= 1:
                    counts["system_events_1h"] += 1
                
                event_type = sys_event.get("event_type", "")
                if event_type == "deployment":
                    counts["deployments_24h"] += 1
                elif event_type == "config_change":
                    counts["config_changes_24h"] += 1
                elif event_type == "system_error":
                    counts["errors_24h"] += 1
        
        return counts
    
    def _build_decision_tree(self, features_df: pd.DataFrame, target: str, 
                            config: Dict[str, Any]) -> Dict[str, Any]:
        """Build a decision tree for the specified target"""
        try:
            feature_columns = [col for col in features_df.columns if col != target]
            X = features_df[feature_columns]
            y = features_df[target]
            
            if target == "has_degradation" and y.value_counts().min() < 5:
                return None
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )
            
            max_depth = config.get("max_depth", 5)
            min_samples_leaf = config.get("min_samples_leaf", 5)
            
            tree = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            
            tree.fit(X_train, y_train)
            
            y_pred = tree.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            feature_importances = {
                feature_columns[i]: float(importance)
                for i, importance in enumerate(tree.feature_importances_)
                if importance > 0.01
            }
            
            return {
                "target": target,
                "accuracy": float(accuracy),
                "feature_importances": feature_importances,
                "tree_depth": tree.get_depth(),
                "n_leaves": tree.get_n_leaves(),
                "training_samples": len(X_train)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to build decision tree for {target}: {e}")
            return None