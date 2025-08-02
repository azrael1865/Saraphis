"""
Visualization Engine Module for Saraphis Fraud Detection System
Phase 6C-5: Interactive Accuracy Visualizations Implementation
Handles creation of interactive charts with real-time updates and cross-visualization linking
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

from enhanced_fraud_core_exceptions import (
    FraudCoreError, ValidationError, ProcessingError,
    ModelError, DataError, ConfigurationError
)

# Import accuracy tracking components
try:
    from accuracy_tracking_db import MetricType
except ImportError:
    # Fallback for missing MetricType
    class MetricType:
        ACCURACY = "accuracy"


class AccuracyAnalyticsError(FraudCoreError):
    """Custom exception for accuracy analytics operations"""
    pass


class VisualizationEngine:
    """
    Specialized module for creating interactive accuracy visualizations.
    Handles chart creation, interactivity features, real-time updates, and export capabilities.
    """
    
    def __init__(self, logger: logging.Logger = None):
        """Initialize VisualizationEngine"""
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._lock = None  # Will be set by parent orchestrator
        self.cache = {}  # Cache for analysis results
    
    def set_lock(self, lock):
        """Set thread lock from parent orchestrator"""
        self._lock = lock
    
    def set_cache_reference(self, cache_ref):
        """Set reference to analysis cache"""
        self.cache = cache_ref
    
    def create_interactive_accuracy_visualizations(self, 
                                                 visualization_config: Dict[str, Any], 
                                                 data_sources: List[str], 
                                                 interactivity_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interactive visualizations with drill-down, filtering, and real-time updates.
        
        Args:
            visualization_config: Configuration for visualization types and settings
            data_sources: List of data sources to use for visualizations
            interactivity_options: Options for interactivity features
            
        Returns:
            Dict containing visualization configurations, interactive elements, data bindings
            
        Raises:
            ProcessingError: When visualization creation fails
            ValidationError: When input validation fails
        """
        start_time = time.time()
        
        try:
            # Input validation
            self._validate_visualization_inputs(visualization_config, data_sources, interactivity_options)
            
            lock_context = self._lock if self._lock else type('DummyLock', (), {'__enter__': lambda self: None, '__exit__': lambda self, *args: None})()
            
            with lock_context:
                self.logger.info(f"Creating interactive accuracy visualizations", extra={
                    "operation": "create_interactive_accuracy_visualizations",
                    "num_data_sources": len(data_sources),
                    "chart_types": visualization_config.get("chart_types", [])
                })
                
                # Initialize results structure
                visualization_results = {
                    "analysis_type": "interactive_accuracy_visualizations",
                    "created_at": datetime.now().isoformat(),
                    "visualization_configs": {},
                    "interactive_elements": {},
                    "data_bindings": {},
                    "real_time_config": {},
                    "export_config": {},
                    "summary": {},
                    "metadata": {
                        "execution_time_ms": 0,
                        "visualizations_created": 0,
                        "data_points_processed": 0,
                        "interactivity_features": [],
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                # Gather data from specified sources
                aggregated_data = self._gather_visualization_data(data_sources)
                visualization_results["metadata"]["data_points_processed"] = aggregated_data.get("total_data_points", 0)
                
                # Get chart types to create
                chart_types = visualization_config.get("chart_types", 
                    ["line", "bar", "heatmap", "scatter", "box_plot", "violin_plot"])
                
                # Create each visualization type
                for chart_type in chart_types:
                    try:
                        viz_config = self._create_visualization_config(
                            chart_type,
                            aggregated_data,
                            visualization_config,
                            interactivity_options
                        )
                        
                        visualization_results["visualization_configs"][chart_type] = viz_config
                        visualization_results["metadata"]["visualizations_created"] += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error creating {chart_type} visualization: {e}")
                        visualization_results["visualization_configs"][chart_type] = {
                            "status": "error",
                            "error": str(e)
                        }
                
                # Create interactive elements
                interactive_elements = self._create_interactive_elements(
                    visualization_results["visualization_configs"],
                    interactivity_options
                )
                visualization_results["interactive_elements"] = interactive_elements
                visualization_results["metadata"]["interactivity_features"] = list(interactive_elements.keys())
                
                # Create data bindings
                data_bindings = self._create_data_bindings(
                    visualization_results["visualization_configs"],
                    aggregated_data,
                    interactivity_options
                )
                visualization_results["data_bindings"] = data_bindings
                
                # Configure real-time updates
                if interactivity_options.get("enable_real_time", False):
                    real_time_config = self._configure_real_time_updates(
                        visualization_results["visualization_configs"],
                        interactivity_options
                    )
                    visualization_results["real_time_config"] = real_time_config
                
                # Configure export options
                export_config = self._configure_export_options(
                    visualization_results["visualization_configs"],
                    visualization_config.get("export_formats", ["png", "svg", "pdf", "json"])
                )
                visualization_results["export_config"] = export_config
                
                # Generate visualization summary
                summary = self._generate_visualization_summary(
                    visualization_results,
                    aggregated_data
                )
                visualization_results["summary"] = summary
                
                # Add execution metadata
                execution_time = int((time.time() - start_time) * 1000)
                visualization_results["metadata"]["execution_time_ms"] = execution_time
                visualization_results["status"] = "success"
                
                self.logger.info(f"Interactive visualizations created successfully", extra={
                    "operation": "create_interactive_accuracy_visualizations",
                    "execution_time_ms": execution_time,
                    "visualizations_created": visualization_results["metadata"]["visualizations_created"],
                    "interactivity_features": len(visualization_results["metadata"]["interactivity_features"])
                })
                
                return visualization_results
                
        except ValidationError as e:
            self.logger.error(f"Validation failed in visualization creation: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in visualization creation: {e}")
            raise ProcessingError(f"Visualization creation failed: {str(e)}")
    
    def _validate_visualization_inputs(self, visualization_config: Dict[str, Any],
                                     data_sources: List[str],
                                     interactivity_options: Dict[str, Any]) -> None:
        """Validate visualization inputs"""
        if not data_sources:
            raise ValidationError("No data sources specified")
            
        # Validate data sources
        valid_sources = ["accuracy_metrics", "trend_analysis", "model_comparison", 
                        "anomaly_detection", "performance_statistics", "drift_analysis",
                        "feature_analysis", "stability_analysis", "forecast_data"]
        invalid_sources = [s for s in data_sources if s not in valid_sources]
        if invalid_sources:
            raise ValidationError(f"Invalid data sources: {invalid_sources}")
        
        # Validate chart types
        if "chart_types" in visualization_config:
            valid_chart_types = ["line", "bar", "heatmap", "scatter", "box_plot", 
                               "violin_plot", "area", "bubble", "radar", "sankey"]
            invalid_types = [t for t in visualization_config["chart_types"] 
                            if t not in valid_chart_types]
            if invalid_types:
                raise ValidationError(f"Invalid chart types: {invalid_types}")
        
        # Validate interactivity options
        if interactivity_options:
            valid_features = ["zoom", "pan", "filter", "drill_down", "hover", 
                             "selection", "brushing", "animation", "real_time"]
            invalid_features = [f for f in interactivity_options.keys() 
                              if f not in valid_features and not f.startswith("enable_")]
            if invalid_features:
                self.logger.warning(f"Unknown interactivity features will be ignored: {invalid_features}")
    
    def _gather_visualization_data(self, data_sources: List[str]) -> Dict[str, Any]:
        """Gather data from multiple sources for visualization"""
        aggregated_data = {
            "sources": {},
            "total_data_points": 0,
            "time_range": {},
            "models": set()
        }
        
        for source in data_sources:
            try:
                if source == "accuracy_metrics":
                    # Get accuracy metrics data
                    metrics_data = self._get_cached_analysis_results("accuracy_metrics")
                    if not metrics_data:
                        metrics_data = self._generate_sample_accuracy_data_for_viz()
                    
                    aggregated_data["sources"]["accuracy_metrics"] = metrics_data
                    aggregated_data["total_data_points"] += len(metrics_data.get("data_points", []))
                    
                elif source == "trend_analysis":
                    # Get trend analysis results
                    trend_data = self._get_cached_analysis_results("trend_analysis")
                    if not trend_data:
                        trend_data = self._generate_sample_trend_data_for_viz()
                    
                    aggregated_data["sources"]["trend_analysis"] = trend_data
                    
                elif source == "model_comparison":
                    # Get model comparison data
                    comparison_data = self._get_cached_analysis_results("model_comparison")
                    if not comparison_data:
                        comparison_data = self._generate_sample_comparison_data_for_viz()
                    
                    aggregated_data["sources"]["model_comparison"] = comparison_data
                    
                elif source == "anomaly_detection":
                    # Get anomaly detection results
                    anomaly_data = self._get_cached_analysis_results("anomaly_detection")
                    if not anomaly_data:
                        anomaly_data = self._generate_sample_anomaly_data_for_viz()
                    
                    aggregated_data["sources"]["anomaly_detection"] = anomaly_data
                    
                elif source == "performance_statistics":
                    # Get performance statistics
                    stats_data = self._get_cached_analysis_results("performance_statistics")
                    if not stats_data:
                        stats_data = self._generate_sample_stats_data_for_viz()
                    
                    aggregated_data["sources"]["performance_statistics"] = stats_data
                    
                elif source == "drift_analysis":
                    # Get drift analysis data
                    drift_data = self._get_cached_analysis_results("drift_analysis")
                    if not drift_data:
                        drift_data = self._generate_sample_drift_data_for_viz()
                    
                    aggregated_data["sources"]["drift_analysis"] = drift_data
                    
                elif source == "feature_analysis":
                    # Get feature analysis data
                    feature_data = self._get_cached_analysis_results("feature_analysis")
                    if not feature_data:
                        feature_data = self._generate_sample_feature_data_for_viz()
                    
                    aggregated_data["sources"]["feature_analysis"] = feature_data
                    
                elif source == "stability_analysis":
                    # Get stability analysis data
                    stability_data = self._get_cached_analysis_results("stability_analysis")
                    if not stability_data:
                        stability_data = self._generate_sample_stability_data_for_viz()
                    
                    aggregated_data["sources"]["stability_analysis"] = stability_data
                    
                elif source == "forecast_data":
                    # Get forecast data
                    forecast_data = self._get_cached_analysis_results("forecast_data")
                    if not forecast_data:
                        forecast_data = self._generate_sample_forecast_data_for_viz()
                    
                    aggregated_data["sources"]["forecast_data"] = forecast_data
                    
            except Exception as e:
                self.logger.error(f"Error gathering data from source {source}: {e}")
                aggregated_data["sources"][source] = {"error": str(e)}
        
        # Extract models and time range
        self._extract_metadata_from_sources(aggregated_data)
        
        return aggregated_data
    
    def _create_visualization_config(self, chart_type: str, 
                                   aggregated_data: Dict[str, Any],
                                   visualization_config: Dict[str, Any],
                                   interactivity_options: Dict[str, Any]) -> Dict[str, Any]:
        """Create configuration for specific visualization type"""
        viz_config = {
            "chart_type": chart_type,
            "id": f"viz_{chart_type}_{int(time.time() * 1000)}",
            "title": self._generate_chart_title(chart_type, aggregated_data),
            "data": {},
            "layout": {},
            "style": {},
            "interactions": {},
            "animations": {},
            "responsive": True
        }
        
        # Chart-specific configurations
        if chart_type == "line":
            viz_config.update(self._create_line_chart_config(aggregated_data, visualization_config))
            
        elif chart_type == "bar":
            viz_config.update(self._create_bar_chart_config(aggregated_data, visualization_config))
            
        elif chart_type == "heatmap":
            viz_config.update(self._create_heatmap_config(aggregated_data, visualization_config))
            
        elif chart_type == "scatter":
            viz_config.update(self._create_scatter_plot_config(aggregated_data, visualization_config))
            
        elif chart_type == "box_plot":
            viz_config.update(self._create_box_plot_config(aggregated_data, visualization_config))
            
        elif chart_type == "violin_plot":
            viz_config.update(self._create_violin_plot_config(aggregated_data, visualization_config))
            
        # Apply common styling
        viz_config["style"] = self._apply_visualization_styling(
            chart_type,
            visualization_config.get("theme", "professional")
        )
        
        # Apply interactivity options
        viz_config["interactions"] = self._apply_interactivity_options(
            chart_type,
            interactivity_options
        )
        
        # Configure animations
        if visualization_config.get("enable_animations", True):
            viz_config["animations"] = self._configure_animations(chart_type)
        
        return viz_config
    
    def _create_line_chart_config(self, aggregated_data: Dict[str, Any],
                                visualization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create line chart configuration"""
        config_update = {
            "data": {
                "series": [],
                "categories": []
            },
            "layout": {
                "xAxis": {
                    "type": "datetime",
                    "title": "Time",
                    "gridLines": True
                },
                "yAxis": {
                    "title": "Accuracy",
                    "min": 0,
                    "max": 1,
                    "gridLines": True
                },
                "legend": {
                    "position": "top",
                    "interactive": True
                }
            }
        }
        
        # Add accuracy metrics series
        if "accuracy_metrics" in aggregated_data["sources"]:
            metrics_data = aggregated_data["sources"]["accuracy_metrics"]
            if "data_points" in metrics_data:
                # Group by model
                model_series = {}
                for point in metrics_data["data_points"]:
                    model_id = point.get("model_id", "unknown")
                    if model_id not in model_series:
                        model_series[model_id] = {
                            "name": f"Model {model_id}",
                            "data": [],
                            "type": "line",
                            "smooth": True
                        }
                    
                    model_series[model_id]["data"].append({
                        "x": point.get("timestamp"),
                        "y": point.get("accuracy", 0),
                        "metadata": {
                            "precision": point.get("precision"),
                            "recall": point.get("recall"),
                            "f1_score": point.get("f1_score")
                        }
                    })
                
                config_update["data"]["series"] = list(model_series.values())
        
        # Add trend lines if available
        if "trend_analysis" in aggregated_data["sources"]:
            trend_data = aggregated_data["sources"]["trend_analysis"]
            if "forecast" in trend_data:
                config_update["data"]["series"].append({
                    "name": "Forecast",
                    "data": [
                        {"x": t, "y": v}
                        for t, v in zip(trend_data["forecast"].get("timestamps", []),
                                       trend_data["forecast"].get("values", []))
                    ],
                    "type": "line",
                    "dashStyle": "dash",
                    "color": "rgba(255, 0, 0, 0.5)"
                })
        
        return config_update
    
    def _create_bar_chart_config(self, aggregated_data: Dict[str, Any],
                               visualization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create bar chart configuration"""
        config_update = {
            "data": {
                "series": [],
                "categories": []
            },
            "layout": {
                "xAxis": {
                    "type": "category",
                    "title": "Models",
                    "labels": {
                        "rotation": -45
                    }
                },
                "yAxis": {
                    "title": "Average Accuracy",
                    "min": 0,
                    "max": 1
                },
                "barMode": visualization_config.get("bar_mode", "grouped")
            }
        }
        
        # Add model comparison data
        if "model_comparison" in aggregated_data["sources"]:
            comparison_data = aggregated_data["sources"]["model_comparison"]
            if "model_scores" in comparison_data:
                categories = []
                accuracy_values = []
                precision_values = []
                recall_values = []
                
                for model_id, scores in comparison_data["model_scores"].items():
                    categories.append(model_id)
                    accuracy_values.append(scores.get("accuracy", 0))
                    precision_values.append(scores.get("precision", 0))
                    recall_values.append(scores.get("recall", 0))
                
                config_update["data"]["categories"] = categories
                config_update["data"]["series"] = [
                    {
                        "name": "Accuracy",
                        "data": accuracy_values,
                        "color": "#4472C4"
                    },
                    {
                        "name": "Precision",
                        "data": precision_values,
                        "color": "#ED7D31"
                    },
                    {
                        "name": "Recall",
                        "data": recall_values,
                        "color": "#A5A5A5"
                    }
                ]
        
        return config_update
    
    def _create_heatmap_config(self, aggregated_data: Dict[str, Any],
                             visualization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create heatmap configuration"""
        config_update = {
            "data": {
                "values": [],
                "xLabels": [],
                "yLabels": []
            },
            "layout": {
                "xAxis": {
                    "title": "Time Period",
                    "type": "category"
                },
                "yAxis": {
                    "title": "Models",
                    "type": "category"
                },
                "colorScale": {
                    "scheme": "viridis",
                    "min": 0,
                    "max": 1,
                    "showScale": True
                }
            }
        }
        
        # Create accuracy heatmap over time
        if "accuracy_metrics" in aggregated_data["sources"]:
            metrics_data = aggregated_data["sources"]["accuracy_metrics"]
            if "data_points" in metrics_data:
                # Organize data by model and time
                heatmap_data = {}
                time_periods = set()
                models = set()
                
                for point in metrics_data["data_points"]:
                    model_id = point.get("model_id", "unknown")
                    timestamp = point.get("timestamp", "")
                    # Convert to day for aggregation
                    day = timestamp.split("T")[0] if "T" in timestamp else timestamp
                    
                    models.add(model_id)
                    time_periods.add(day)
                    
                    if model_id not in heatmap_data:
                        heatmap_data[model_id] = {}
                    
                    # Average if multiple values per day
                    if day in heatmap_data[model_id]:
                        heatmap_data[model_id][day] = (
                            heatmap_data[model_id][day] + point.get("accuracy", 0)
                        ) / 2
                    else:
                        heatmap_data[model_id][day] = point.get("accuracy", 0)
                
                # Convert to matrix format
                sorted_models = sorted(models)
                sorted_periods = sorted(time_periods)
                
                values = []
                for model in sorted_models:
                    row = []
                    for period in sorted_periods:
                        value = heatmap_data.get(model, {}).get(period, None)
                        row.append(value)
                    values.append(row)
                
                config_update["data"]["values"] = values
                config_update["data"]["xLabels"] = sorted_periods
                config_update["data"]["yLabels"] = sorted_models
        
        return config_update
    
    def _create_scatter_plot_config(self, aggregated_data: Dict[str, Any],
                                  visualization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create scatter plot configuration"""
        config_update = {
            "data": {
                "series": []
            },
            "layout": {
                "xAxis": {
                    "title": visualization_config.get("x_metric", "Precision"),
                    "min": 0,
                    "max": 1
                },
                "yAxis": {
                    "title": visualization_config.get("y_metric", "Recall"),
                    "min": 0,
                    "max": 1
                },
                "showGrid": True
            }
        }
        
        # Create precision vs recall scatter plot
        if "performance_statistics" in aggregated_data["sources"]:
            stats_data = aggregated_data["sources"]["performance_statistics"]
            if "model_metrics" in stats_data:
                for model_id, metrics in stats_data["model_metrics"].items():
                    series_data = []
                    
                    # Get precision and recall values
                    precision_values = metrics.get("precision_values", [])
                    recall_values = metrics.get("recall_values", [])
                    
                    for i in range(min(len(precision_values), len(recall_values))):
                        series_data.append({
                            "x": precision_values[i],
                            "y": recall_values[i],
                            "size": 10,
                            "metadata": {
                                "model_id": model_id,
                                "index": i
                            }
                        })
                    
                    if series_data:
                        config_update["data"]["series"].append({
                            "name": f"Model {model_id}",
                            "data": series_data,
                            "type": "scatter"
                        })
        
        return config_update
    
    def _create_box_plot_config(self, aggregated_data: Dict[str, Any],
                              visualization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create box plot configuration"""
        config_update = {
            "data": {
                "series": []
            },
            "layout": {
                "xAxis": {
                    "type": "category",
                    "title": "Models"
                },
                "yAxis": {
                    "title": "Accuracy Distribution",
                    "min": 0,
                    "max": 1
                }
            }
        }
        
        # Create box plots for model accuracy distributions
        if "performance_statistics" in aggregated_data["sources"]:
            stats_data = aggregated_data["sources"]["performance_statistics"]
            if "model_metrics" in stats_data:
                box_data = []
                
                for model_id, metrics in stats_data["model_metrics"].items():
                    accuracy_values = metrics.get("accuracy_values", [])
                    
                    if accuracy_values:
                        # Calculate box plot statistics
                        q1 = np.percentile(accuracy_values, 25)
                        median = np.percentile(accuracy_values, 50)
                        q3 = np.percentile(accuracy_values, 75)
                        min_val = np.min(accuracy_values)
                        max_val = np.max(accuracy_values)
                        
                        box_data.append({
                            "x": model_id,
                            "low": min_val,
                            "q1": q1,
                            "median": median,
                            "q3": q3,
                            "high": max_val,
                            "outliers": self._detect_outliers(accuracy_values)
                        })
                
                if box_data:
                    config_update["data"]["series"] = [{
                        "name": "Accuracy Distribution",
                        "data": box_data,
                        "type": "boxplot"
                    }]
        
        return config_update
    
    def _create_violin_plot_config(self, aggregated_data: Dict[str, Any],
                                 visualization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create violin plot configuration"""
        config_update = {
            "data": {
                "series": []
            },
            "layout": {
                "xAxis": {
                    "type": "category",
                    "title": "Models"
                },
                "yAxis": {
                    "title": "Accuracy Distribution",
                    "min": 0,
                    "max": 1
                }
            }
        }
        
        # Create violin plots for model accuracy distributions
        if "performance_statistics" in aggregated_data["sources"]:
            stats_data = aggregated_data["sources"]["performance_statistics"]
            if "model_metrics" in stats_data:
                violin_data = []
                
                for model_id, metrics in stats_data["model_metrics"].items():
                    accuracy_values = metrics.get("accuracy_values", [])
                    
                    if accuracy_values:
                        # Calculate kernel density estimation
                        kde_points = self._calculate_kde(accuracy_values)
                        
                        violin_data.append({
                            "x": model_id,
                            "y": accuracy_values,
                            "kde": kde_points,
                            "box": {
                                "median": np.median(accuracy_values),
                                "q1": np.percentile(accuracy_values, 25),
                                "q3": np.percentile(accuracy_values, 75)
                            }
                        })
                
                if violin_data:
                    config_update["data"]["series"] = [{
                        "name": "Accuracy Distribution",
                        "data": violin_data,
                        "type": "violin"
                    }]
        
        return config_update
    
    def _create_interactive_elements(self, visualization_configs: Dict[str, Any],
                                   interactivity_options: Dict[str, Any]) -> Dict[str, Any]:
        """Create interactive elements for visualizations"""
        interactive_elements = {}
        
        # Zoom and Pan controls
        if interactivity_options.get("zoom", True) or interactivity_options.get("pan", True):
            interactive_elements["zoom_pan"] = {
                "enabled": True,
                "zoom": {
                    "enabled": interactivity_options.get("zoom", True),
                    "type": "xy",
                    "wheelZoom": True
                },
                "pan": {
                    "enabled": interactivity_options.get("pan", True),
                    "mode": "xy"
                }
            }
        
        # Filter controls
        if interactivity_options.get("filter", True):
            interactive_elements["filters"] = {
                "enabled": True,
                "model_filter": {
                    "type": "multi_select",
                    "options": self._extract_model_options(visualization_configs),
                    "default": "all"
                },
                "time_filter": {
                    "type": "date_range",
                    "format": "YYYY-MM-DD",
                    "default": "last_30_days"
                },
                "metric_filter": {
                    "type": "single_select",
                    "options": ["accuracy", "precision", "recall", "f1_score"],
                    "default": "accuracy"
                }
            }
        
        # Drill-down capabilities
        if interactivity_options.get("drill_down", True):
            interactive_elements["drill_down"] = {
                "enabled": True,
                "levels": ["model", "time_period", "metric"],
                "actions": {
                    "click": "drill_to_detail",
                    "double_click": "drill_to_raw_data"
                }
            }
        
        # Hover tooltips
        if interactivity_options.get("hover", True):
            interactive_elements["tooltips"] = {
                "enabled": True,
                "content_template": {
                    "model": "{model_id}",
                    "timestamp": "{timestamp}",
                    "accuracy": "{accuracy:.4f}",
                    "precision": "{precision:.4f}",
                    "recall": "{recall:.4f}"
                },
                "position": "follow_cursor",
                "animation": "fade"
            }
        
        # Selection and brushing
        if interactivity_options.get("selection", True) or interactivity_options.get("brushing", True):
            interactive_elements["selection"] = {
                "enabled": True,
                "mode": "rectangle",
                "multiple": True,
                "brushing": {
                    "enabled": interactivity_options.get("brushing", True),
                    "linked_charts": True
                }
            }
        
        # Cross-chart linking
        interactive_elements["cross_chart_linking"] = {
            "enabled": True,
            "link_types": ["filter", "highlight", "zoom"],
            "sync_axes": True
        }
        
        return interactive_elements
    
    def _create_data_bindings(self, visualization_configs: Dict[str, Any],
                            aggregated_data: Dict[str, Any],
                            interactivity_options: Dict[str, Any]) -> Dict[str, Any]:
        """Create data bindings for dynamic updates"""
        data_bindings = {
            "data_sources": {},
            "update_triggers": {},
            "transformation_rules": {},
            "caching_strategy": {}
        }
        
        # Define data source bindings
        for viz_type, config in visualization_configs.items():
            if "error" not in config:
                data_bindings["data_sources"][viz_type] = {
                    "primary_source": self._identify_primary_source(viz_type, aggregated_data),
                    "secondary_sources": self._identify_secondary_sources(viz_type, aggregated_data),
                    "refresh_rate": self._determine_refresh_rate(viz_type, interactivity_options),
                    "data_keys": self._extract_data_keys(config)
                }
        
        # Define update triggers
        data_bindings["update_triggers"] = {
            "user_interaction": {
                "filter_change": ["all_charts"],
                "time_range_change": ["time_series_charts"],
                "model_selection": ["model_specific_charts"]
            },
            "data_refresh": {
                "scheduled": {
                    "interval": "5_minutes",
                    "charts": ["real_time_charts"]
                },
                "on_demand": {
                    "trigger": "user_request",
                    "charts": ["all_charts"]
                }
            }
        }
        
        # Define transformation rules
        data_bindings["transformation_rules"] = {
            "aggregation": {
                "time_grouping": "auto_detect",
                "model_grouping": "by_model_id",
                "metric_calculation": "mean_with_confidence_intervals"
            },
            "filtering": {
                "data_quality_threshold": 0.8,
                "outlier_detection": True,
                "missing_data_handling": "interpolate"
            },
            "normalization": {
                "metric_scaling": "0_to_1",
                "time_alignment": "common_timebase"
            }
        }
        
        # Define caching strategy
        data_bindings["caching_strategy"] = {
            "cache_levels": ["raw_data", "processed_data", "rendered_data"],
            "cache_duration": {
                "raw_data": "1_hour",
                "processed_data": "30_minutes",
                "rendered_data": "5_minutes"
            },
            "invalidation_triggers": ["data_update", "filter_change", "config_change"]
        }
        
        return data_bindings
    
    def _configure_real_time_updates(self, visualization_configs: Dict[str, Any],
                                   interactivity_options: Dict[str, Any]) -> Dict[str, Any]:
        """Configure real-time update capabilities"""
        real_time_config = {
            "enabled": True,
            "websocket_config": {},
            "polling_config": {},
            "update_strategy": {},
            "performance_settings": {}
        }
        
        # WebSocket configuration
        real_time_config["websocket_config"] = {
            "endpoint": "/ws/accuracy_updates",
            "protocols": ["accuracy-v1"],
            "heartbeat_interval": 30,
            "reconnect_attempts": 5,
            "buffer_size": 1024
        }
        
        # Polling fallback configuration
        real_time_config["polling_config"] = {
            "enabled": True,
            "interval": interactivity_options.get("refresh_interval", 30),
            "endpoints": {
                "accuracy_metrics": "/api/accuracy/recent",
                "model_status": "/api/models/status",
                "alerts": "/api/alerts/recent"
            }
        }
        
        # Update strategy
        real_time_config["update_strategy"] = {
            "update_mode": "incremental",
            "batch_size": 100,
            "animation_duration": 500,
            "smooth_transitions": True,
            "preserve_user_interactions": True
        }
        
        # Performance settings
        real_time_config["performance_settings"] = {
            "max_data_points": 1000,
            "data_compression": True,
            "lazy_loading": True,
            "virtual_scrolling": True,
            "debounce_interval": 100
        }
        
        return real_time_config
    
    def _configure_export_options(self, visualization_configs: Dict[str, Any],
                                export_formats: List[str]) -> Dict[str, Any]:
        """Configure export options for visualizations"""
        export_config = {
            "enabled_formats": export_formats,
            "format_configs": {},
            "export_settings": {}
        }
        
        # Format-specific configurations
        for format_type in export_formats:
            if format_type == "png":
                export_config["format_configs"]["png"] = {
                    "resolution": "high",
                    "dpi": 300,
                    "background": "white",
                    "width": 1200,
                    "height": 800
                }
            elif format_type == "svg":
                export_config["format_configs"]["svg"] = {
                    "scalable": True,
                    "embed_fonts": True,
                    "optimize": True
                }
            elif format_type == "pdf":
                export_config["format_configs"]["pdf"] = {
                    "page_size": "A4",
                    "orientation": "landscape",
                    "include_title": True,
                    "include_metadata": True
                }
            elif format_type == "json":
                export_config["format_configs"]["json"] = {
                    "include_data": True,
                    "include_config": True,
                    "pretty_print": True,
                    "compression": "gzip"
                }
            elif format_type == "html":
                export_config["format_configs"]["html"] = {
                    "standalone": True,
                    "include_interactivity": True,
                    "cdn_libraries": False,
                    "responsive": True
                }
        
        # General export settings
        export_config["export_settings"] = {
            "filename_template": "accuracy_viz_{chart_type}_{timestamp}",
            "include_timestamp": True,
            "batch_export": True,
            "custom_styling": True
        }
        
        return export_config
    
    def _generate_visualization_summary(self, visualization_results: Dict[str, Any],
                                      aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of created visualizations"""
        summary = {
            "overview": {},
            "chart_summary": {},
            "data_summary": {},
            "interactivity_summary": {},
            "recommendations": []
        }
        
        # Overview
        summary["overview"] = {
            "total_visualizations": visualization_results["metadata"]["visualizations_created"],
            "data_sources_used": len(aggregated_data["sources"]),
            "interactivity_features": len(visualization_results["metadata"]["interactivity_features"]),
            "real_time_enabled": "real_time_config" in visualization_results and visualization_results["real_time_config"]["enabled"]
        }
        
        # Chart summary
        chart_types = list(visualization_results["visualization_configs"].keys())
        summary["chart_summary"] = {
            "chart_types_created": chart_types,
            "most_data_rich": self._identify_most_data_rich_chart(visualization_results["visualization_configs"]),
            "interactive_charts": [ct for ct in chart_types if ct in visualization_results["visualization_configs"] and visualization_results["visualization_configs"][ct].get("interactions")],
            "exportable_charts": chart_types  # All charts are exportable
        }
        
        # Data summary
        summary["data_summary"] = {
            "total_data_points": aggregated_data["total_data_points"],
            "models_analyzed": len(aggregated_data.get("models", [])),
            "time_range_covered": aggregated_data.get("time_range", {}),
            "data_quality_score": self._calculate_data_quality_score(aggregated_data)
        }
        
        # Interactivity summary
        summary["interactivity_summary"] = {
            "zoom_pan_enabled": "zoom_pan" in visualization_results["interactive_elements"],
            "filtering_enabled": "filters" in visualization_results["interactive_elements"],
            "drill_down_enabled": "drill_down" in visualization_results["interactive_elements"],
            "cross_chart_linking": "cross_chart_linking" in visualization_results["interactive_elements"],
            "real_time_updates": visualization_results.get("real_time_config", {}).get("enabled", False)
        }
        
        # Recommendations
        summary["recommendations"] = self._generate_visualization_recommendations(
            visualization_results, aggregated_data
        )
        
        return summary
    
    # Helper methods for data generation and processing
    
    def _get_cached_analysis_results(self, analysis_type: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis results"""
        return self.cache.get(analysis_type)
    
    def _generate_sample_accuracy_data_for_viz(self) -> Dict[str, Any]:
        """Generate sample accuracy data for visualization - FALLBACK ONLY"""
        # Try to get real accuracy data first
        try:
            if hasattr(self, 'accuracy_db') and self.accuracy_db:
                real_data = self._get_real_accuracy_data_for_viz()
                if real_data and len(real_data.get('data_points', [])) > 0:
                    return real_data
        except Exception as e:
            self.logger.warning(f"Failed to retrieve real accuracy data for visualization: {e}")
        
        # Fallback warning
        self.logger.warning(
            "USING DUMMY DATA: No real accuracy data available for visualization. "
            "These synthetic metrics should NOT be used for production analysis."
        )
        
        data_points = []
        models = ["model_1", "model_2", "model_3"]
        
        for i in range(90):  # 90 days of data
            timestamp = (datetime.now() - timedelta(days=89-i)).isoformat()
            
            for model_id in models:
                base_accuracy = 0.94 + (hash(model_id) % 100) / 1000
                accuracy = base_accuracy + np.random.normal(0, 0.02)
                precision = accuracy + np.random.normal(0, 0.01)
                recall = accuracy + np.random.normal(0, 0.015)
                f1_score = 2 * (precision * recall) / (precision + recall)
                
                data_points.append({
                    "model_id": model_id,
                    "timestamp": timestamp,
                    "accuracy": max(0, min(1, accuracy)),
                    "precision": max(0, min(1, precision)),
                    "recall": max(0, min(1, recall)),
                    "f1_score": max(0, min(1, f1_score))
                })
        
        return {
            "data_points": data_points,
            "_is_dummy_data": True,
            "_warning": "Synthetic visualization data - not for production analysis"
        }
    
    def _get_real_accuracy_data_for_viz(self) -> Optional[Dict[str, Any]]:
        """Attempt to retrieve real accuracy data for visualization"""
        try:
            start_date = datetime.now() - timedelta(days=90)
            real_metrics = self.accuracy_db.get_accuracy_metrics(
                model_id="fraud_model_viz",
                start_date=start_date
            )
            
            if real_metrics and len(real_metrics) > 50:  # Require substantial data
                data_points = []
                for metric in real_metrics:
                    data_points.append({
                        "model_id": metric.model_id,
                        "timestamp": metric.timestamp.isoformat(),
                        "accuracy": float(metric.metric_value) if metric.metric_type == MetricType.ACCURACY else None,
                        "precision": None,  # Would need separate query
                        "recall": None,     # Would need separate query
                        "f1_score": None    # Would need separate query
                    })
                
                return {
                    "data_points": data_points,
                    "_is_real_data": True
                }
        except Exception as e:
            self.logger.debug(f"Could not retrieve real visualization data: {e}")
        
        return None
    
    def _generate_sample_trend_data_for_viz(self) -> Dict[str, Any]:
        """Generate sample trend data for visualization"""
        return {
            "forecast": {
                "timestamps": [(datetime.now() + timedelta(days=i)).isoformat() for i in range(1, 31)],
                "values": [0.94 + np.random.normal(0, 0.01) for _ in range(30)]
            }
        }
    
    def _generate_sample_comparison_data_for_viz(self) -> Dict[str, Any]:
        """Generate sample model comparison data"""
        models = ["model_1", "model_2", "model_3", "model_4"]
        model_scores = {}
        
        for model_id in models:
            base_score = 0.90 + (hash(model_id) % 100) / 1000
            model_scores[model_id] = {
                "accuracy": base_score + np.random.normal(0, 0.02),
                "precision": base_score + np.random.normal(0, 0.015),
                "recall": base_score + np.random.normal(0, 0.02)
            }
        
        return {"model_scores": model_scores}
    
    def _generate_sample_anomaly_data_for_viz(self) -> Dict[str, Any]:
        """Generate sample anomaly data"""
        return {"anomalies": []}
    
    def _generate_sample_stats_data_for_viz(self) -> Dict[str, Any]:
        """Generate sample statistics data"""
        models = ["model_1", "model_2", "model_3"]
        model_metrics = {}
        
        for model_id in models:
            base_accuracy = 0.94 + (hash(model_id) % 100) / 1000
            accuracy_values = [base_accuracy + np.random.normal(0, 0.02) for _ in range(30)]
            precision_values = [a + np.random.normal(0, 0.01) for a in accuracy_values]
            recall_values = [a + np.random.normal(0, 0.015) for a in accuracy_values]
            
            model_metrics[model_id] = {
                "accuracy_values": accuracy_values,
                "precision_values": precision_values,
                "recall_values": recall_values
            }
        
        return {"model_metrics": model_metrics}
    
    def _generate_sample_drift_data_for_viz(self) -> Dict[str, Any]:
        """Generate sample drift data"""
        return {"drift_detected": False}
    
    def _generate_sample_feature_data_for_viz(self) -> Dict[str, Any]:
        """Generate sample feature data"""
        return {"feature_importance": {}}
    
    def _generate_sample_stability_data_for_viz(self) -> Dict[str, Any]:
        """Generate sample stability data"""
        return {"stability_score": 0.95}
    
    def _generate_sample_forecast_data_for_viz(self) -> Dict[str, Any]:
        """Generate sample forecast data"""
        return {
            "forecast_values": [0.94 + np.random.normal(0, 0.01) for _ in range(30)],
            "confidence_intervals": {
                "lower": [0.92 + np.random.normal(0, 0.01) for _ in range(30)],
                "upper": [0.96 + np.random.normal(0, 0.01) for _ in range(30)]
            }
        }
    
    def _extract_metadata_from_sources(self, aggregated_data: Dict[str, Any]) -> None:
        """Extract metadata from data sources"""
        # Extract models
        models = set()
        timestamps = []
        
        for source_name, source_data in aggregated_data["sources"].items():
            if "data_points" in source_data:
                for point in source_data["data_points"]:
                    if "model_id" in point:
                        models.add(point["model_id"])
                    if "timestamp" in point:
                        timestamps.append(point["timestamp"])
        
        aggregated_data["models"] = models
        
        if timestamps:
            aggregated_data["time_range"] = {
                "start": min(timestamps),
                "end": max(timestamps)
            }
    
    def _generate_chart_title(self, chart_type: str, aggregated_data: Dict[str, Any]) -> str:
        """Generate appropriate title for chart type"""
        title_map = {
            "line": "Accuracy Trends Over Time",
            "bar": "Model Performance Comparison",
            "heatmap": "Accuracy Heatmap by Model and Time",
            "scatter": "Precision vs Recall Analysis",
            "box_plot": "Accuracy Distribution by Model",
            "violin_plot": "Accuracy Distribution Density"
        }
        
        base_title = title_map.get(chart_type, f"{chart_type.title()} Visualization")
        
        # Add context based on available data
        if aggregated_data.get("models"):
            model_count = len(aggregated_data["models"])
            base_title += f" ({model_count} Models)"
        
        return base_title
    
    def _apply_visualization_styling(self, chart_type: str, theme: str) -> Dict[str, Any]:
        """Apply styling based on chart type and theme"""
        base_style = {
            "theme": theme,
            "color_palette": self._get_color_palette(theme),
            "font_family": "Arial, sans-serif",
            "title_size": "16px",
            "axis_label_size": "12px",
            "legend_size": "11px"
        }
        
        # Chart-specific styling
        if chart_type == "line":
            base_style.update({
                "line_width": 2,
                "point_size": 4,
                "grid_opacity": 0.3
            })
        elif chart_type == "bar":
            base_style.update({
                "bar_border_width": 1,
                "bar_spacing": 0.1
            })
        elif chart_type == "heatmap":
            base_style.update({
                "cell_border_width": 1,
                "color_interpolation": "smooth"
            })
        
        return base_style
    
    def _get_color_palette(self, theme: str) -> List[str]:
        """Get color palette for theme"""
        palettes = {
            "professional": ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5"],
            "vibrant": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
            "minimal": ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
        }
        return palettes.get(theme, palettes["professional"])
    
    def _apply_interactivity_options(self, chart_type: str, 
                                   interactivity_options: Dict[str, Any]) -> Dict[str, Any]:
        """Apply interactivity options to chart"""
        interactions = {}
        
        # Hover interactions
        if interactivity_options.get("hover", True):
            interactions["hover"] = {
                "enabled": True,
                "highlight_on_hover": True,
                "tooltip_follow_cursor": True
            }
        
        # Click interactions
        if interactivity_options.get("drill_down", True):
            interactions["click"] = {
                "enabled": True,
                "action": "drill_down",
                "multi_select": True
            }
        
        # Selection interactions
        if interactivity_options.get("selection", True):
            interactions["selection"] = {
                "enabled": True,
                "mode": "rectangle" if chart_type in ["scatter", "line"] else "single",
                "persistent": True
            }
        
        return interactions
    
    def _configure_animations(self, chart_type: str) -> Dict[str, Any]:
        """Configure animations for chart type"""
        animations = {
            "enabled": True,
            "duration": 800,
            "easing": "ease-in-out"
        }
        
        if chart_type == "line":
            animations.update({
                "draw_animation": "left_to_right",
                "point_animation": "fade_in"
            })
        elif chart_type == "bar":
            animations.update({
                "draw_animation": "bottom_to_top",
                "stagger_delay": 50
            })
        elif chart_type == "heatmap":
            animations.update({
                "draw_animation": "fade_in",
                "cell_delay": 10
            })
        
        return animations
    
    def _extract_model_options(self, visualization_configs: Dict[str, Any]) -> List[str]:
        """Extract model options from visualization configs"""
        models = set()
        
        for config in visualization_configs.values():
            if "data" in config and "series" in config["data"]:
                for series in config["data"]["series"]:
                    if "metadata" in series:
                        for point in series.get("data", []):
                            if isinstance(point, dict) and "metadata" in point:
                                model_id = point["metadata"].get("model_id")
                                if model_id:
                                    models.add(model_id)
        
        return sorted(list(models))
    
    def _identify_primary_source(self, viz_type: str, aggregated_data: Dict[str, Any]) -> str:
        """Identify primary data source for visualization type"""
        source_mapping = {
            "line": "accuracy_metrics",
            "bar": "model_comparison",
            "heatmap": "accuracy_metrics",
            "scatter": "performance_statistics",
            "box_plot": "performance_statistics",
            "violin_plot": "performance_statistics"
        }
        
        return source_mapping.get(viz_type, "accuracy_metrics")
    
    def _identify_secondary_sources(self, viz_type: str, aggregated_data: Dict[str, Any]) -> List[str]:
        """Identify secondary data sources for visualization type"""
        if viz_type == "line":
            return ["trend_analysis", "forecast_data"]
        elif viz_type == "scatter":
            return ["model_comparison"]
        else:
            return []
    
    def _determine_refresh_rate(self, viz_type: str, interactivity_options: Dict[str, Any]) -> int:
        """Determine refresh rate for visualization type"""
        if interactivity_options.get("enable_real_time", False):
            return interactivity_options.get("refresh_interval", 30)
        else:
            return 300  # 5 minutes default
    
    def _extract_data_keys(self, config: Dict[str, Any]) -> List[str]:
        """Extract data keys from visualization config"""
        keys = []
        
        if "data" in config:
            if "series" in config["data"]:
                keys.append("series")
            if "categories" in config["data"]:
                keys.append("categories")
            if "values" in config["data"]:
                keys.append("values")
        
        return keys
    
    def _detect_outliers(self, values: List[float]) -> List[float]:
        """Detect outliers in values using IQR method"""
        if len(values) < 4:
            return []
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [v for v in values if v < lower_bound or v > upper_bound]
        return outliers
    
    def _calculate_kde(self, values: List[float]) -> List[Dict[str, float]]:
        """Calculate kernel density estimation for violin plots"""
        # Simplified KDE calculation
        min_val = min(values)
        max_val = max(values)
        
        # Create points for KDE
        x_points = np.linspace(min_val, max_val, 50)
        kde_points = []
        
        for x in x_points:
            # Simple Gaussian kernel
            density = sum(np.exp(-0.5 * ((x - v) / 0.1) ** 2) for v in values)
            kde_points.append({"x": float(x), "density": float(density)})
        
        return kde_points
    
    def _identify_most_data_rich_chart(self, visualization_configs: Dict[str, Any]) -> str:
        """Identify chart with most data points"""
        max_points = 0
        most_data_rich = None
        
        for chart_type, config in visualization_configs.items():
            if "error" not in config and "data" in config:
                data_count = 0
                
                if "series" in config["data"]:
                    for series in config["data"]["series"]:
                        data_count += len(series.get("data", []))
                
                if data_count > max_points:
                    max_points = data_count
                    most_data_rich = chart_type
        
        return most_data_rich or "none"
    
    def _calculate_data_quality_score(self, aggregated_data: Dict[str, Any]) -> float:
        """Calculate overall data quality score"""
        quality_factors = []
        
        # Check data completeness
        total_sources = len(aggregated_data["sources"])
        sources_with_data = sum(1 for source in aggregated_data["sources"].values() 
                               if "error" not in source)
        
        if total_sources > 0:
            completeness = sources_with_data / total_sources
            quality_factors.append(completeness)
        
        # Check data volume
        if aggregated_data["total_data_points"] > 100:
            quality_factors.append(1.0)
        elif aggregated_data["total_data_points"] > 50:
            quality_factors.append(0.8)
        else:
            quality_factors.append(0.6)
        
        # Check time coverage
        if aggregated_data.get("time_range"):
            quality_factors.append(0.9)
        else:
            quality_factors.append(0.5)
        
        return np.mean(quality_factors) if quality_factors else 0.0
    
    def _generate_visualization_recommendations(self, visualization_results: Dict[str, Any],
                                             aggregated_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations for visualization improvements"""
        recommendations = []
        
        # Data quality recommendations
        data_quality = self._calculate_data_quality_score(aggregated_data)
        if data_quality < 0.8:
            recommendations.append("Consider improving data quality by adding more data sources")
        
        # Interactivity recommendations
        if not visualization_results.get("real_time_config", {}).get("enabled"):
            recommendations.append("Enable real-time updates for more dynamic visualizations")
        
        # Chart type recommendations
        created_charts = len(visualization_results["visualization_configs"])
        if created_charts < 3:
            recommendations.append("Add more chart types for comprehensive analysis")
        
        # Export recommendations
        export_formats = visualization_results.get("export_config", {}).get("enabled_formats", [])
        if "pdf" not in export_formats:
            recommendations.append("Add PDF export for professional reporting")
        
        return recommendations