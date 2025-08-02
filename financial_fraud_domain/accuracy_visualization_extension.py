"""
Accuracy Visualization Extension for Saraphis Financial Fraud Detection System
Extends the VisualizationDashboardEngine to add comprehensive accuracy monitoring and visualization.

Integrates with existing accuracy tracking infrastructure to provide:
- Real-time accuracy dashboards with confusion matrix visualization
- Training metrics monitoring and validation accuracy tracking  
- Model comparison and performance analysis
- Export capabilities for reporting and analysis

Author: Saraphis Development Team
Version: 1.0.0
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dataclasses import dataclass, asdict
import asyncio
import threading
from collections import deque
import logging

# Import existing components from the Saraphis system
from accuracy_tracking_db import AccuracyTrackingDatabase
from real_time_accuracy_monitor import RealTimeAccuracyMonitor
from advanced_accuracy_monitoring import DashboardManager, DashboardWidget, DashboardWidgetType
# Import core visualization engine from new location
import sys
from pathlib import Path
core_dir = Path(__file__).parent.parent / 'independent_core'
sys.path.insert(0, str(core_dir))
from dashboard_integration.visualization_dashboard_engine import VisualizationDashboardEngine

logger = logging.getLogger(__name__)


class AccuracyVisualizationExtension:
    """Extension for VisualizationDashboardEngine to add accuracy calculation and visualization"""
    
    def __init__(self, dashboard_engine: VisualizationDashboardEngine):
        self.dashboard_engine = dashboard_engine
        self.accuracy_db = AccuracyTrackingDatabase()
        self.real_time_monitor = RealTimeAccuracyMonitor()
        self.dashboard_manager = DashboardManager()
        self.accuracy_cache = deque(maxlen=1000)
        self.confusion_matrix_cache = {}
        self._setup_accuracy_callbacks()
        self._initialize_accuracy_widgets()
        
    def _initialize_accuracy_widgets(self):
        """Initialize accuracy-specific widgets in the dashboard"""
        # Create accuracy trend widget
        accuracy_trend_widget = DashboardWidget(
            widget_id="accuracy_trend_main",
            title="Model Accuracy Trend",
            widget_type=DashboardWidgetType.TIME_SERIES,
            config={
                "metric": "accuracy",
                "time_range": "24h",
                "update_interval": 30,
                "show_confidence_bands": True
            },
            data_source="accuracy_tracking",
            display_options={
                "height": 400,
                "show_legend": True
            }
        )
        
        # Create confusion matrix heatmap widget
        confusion_matrix_widget = DashboardWidget(
            widget_id="confusion_matrix_heatmap",
            title="Confusion Matrix",
            widget_type=DashboardWidgetType.HEATMAP,
            config={
                "matrix_type": "normalized",
                "color_scale": "Blues",
                "show_values": True,
                "update_interval": 60
            },
            data_source="confusion_matrix",
            display_options={
                "height": 400,
                "color_continuous_scale": "Blues"
            }
        )
        
        # Create accuracy metrics card widget
        accuracy_metrics_card = DashboardWidget(
            widget_id="accuracy_metrics_card",
            title="Accuracy Metrics",
            widget_type=DashboardWidgetType.METRIC_CARD,
            config={
                "metrics": ["accuracy", "precision", "recall", "f1_score"],
                "show_trend": True,
                "comparison_period": "1h"
            },
            data_source="real_time_metrics",
            display_options={
                "columns": 4,
                "show_sparklines": True
            }
        )
        
        # Create model comparison widget
        model_comparison_widget = DashboardWidget(
            widget_id="accuracy_model_comparison",
            title="Model Accuracy Comparison",
            widget_type=DashboardWidgetType.MODEL_COMPARISON,
            config={
                "models": [],
                "metrics": ["accuracy", "precision", "recall"],
                "visualization": "radar"
            },
            data_source="model_metrics",
            display_options={
                "chart_type": "radar",
                "show_values": True
            }
        )
        
        # Register widgets with dashboard manager
        self.dashboard_manager.widgets = {
            "accuracy_trend_main": accuracy_trend_widget,
            "confusion_matrix_heatmap": confusion_matrix_widget,
            "accuracy_metrics_card": accuracy_metrics_card,
            "accuracy_model_comparison": model_comparison_widget
        }
        
    def _setup_accuracy_callbacks(self):
        """Set up Dash callbacks for accuracy visualization"""
        app = self.dashboard_engine._dash_app
        
        if app is None:
            logger.warning("Dash app not initialized, callbacks will be registered later")
            return
        
        # Callback for accuracy trend visualization
        @app.callback(
            Output('accuracy-trend-graph', 'figure'),
            [Input('accuracy-update-interval', 'n_intervals'),
             Input('model-selector', 'value'),
             Input('time-range-selector', 'value')]
        )
        def update_accuracy_trend(n_intervals, model_id, time_range):
            return self._generate_accuracy_trend_figure(model_id, time_range)
        
        # Callback for confusion matrix heatmap
        @app.callback(
            Output('confusion-matrix-heatmap', 'figure'),
            [Input('accuracy-update-interval', 'n_intervals'),
             Input('model-selector', 'value'),
             Input('matrix-normalization', 'value')]
        )
        def update_confusion_matrix(n_intervals, model_id, normalization):
            return self._generate_confusion_matrix_figure(model_id, normalization)
        
        # Callback for accuracy metrics cards
        @app.callback(
            Output('accuracy-metrics-cards', 'children'),
            [Input('accuracy-update-interval', 'n_intervals'),
             Input('model-selector', 'value')]
        )
        def update_accuracy_metrics(n_intervals, model_id):
            return self._generate_accuracy_metrics_cards(model_id)
        
        # Callback for model comparison
        @app.callback(
            Output('model-comparison-chart', 'figure'),
            [Input('accuracy-update-interval', 'n_intervals'),
             Input('comparison-models', 'value'),
             Input('comparison-metric', 'value')]
        )
        def update_model_comparison(n_intervals, model_ids, metric):
            return self._generate_model_comparison_figure(model_ids, metric)
        
        # WebSocket callback for real-time accuracy updates
        if self.dashboard_engine._socket_io:
            @self.dashboard_engine._socket_io.on('request_accuracy_update')
            def handle_accuracy_update_request(data):
                model_id = data.get('model_id')
                accuracy_data = self._get_real_time_accuracy(model_id)
                self.dashboard_engine._socket_io.emit('accuracy_update', {
                    'model_id': model_id,
                    'accuracy_data': accuracy_data,
                    'timestamp': datetime.now().isoformat()
                })
    
    def _generate_accuracy_trend_figure(self, model_id: str, time_range: str) -> go.Figure:
        """Generate accuracy trend visualization"""
        # Get historical accuracy data
        end_time = datetime.now()
        if time_range == '1h':
            start_time = end_time - timedelta(hours=1)
        elif time_range == '24h':
            start_time = end_time - timedelta(days=1)
        elif time_range == '7d':
            start_time = end_time - timedelta(days=7)
        else:
            start_time = end_time - timedelta(days=30)
        
        # Query accuracy data from database
        accuracy_data = self.accuracy_db.get_accuracy_metrics(
            model_id=model_id,
            start_time=start_time,
            end_time=end_time
        )
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        if accuracy_data:
            df = pd.DataFrame(accuracy_data)
            
            # Accuracy trace
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['accuracy'],
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
            
            # Precision trace
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['precision'],
                    mode='lines+markers',
                    name='Precision',
                    line=dict(color='green', width=2),
                    marker=dict(size=4)
                ),
                row=1, col=2
            )
            
            # Recall trace
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['recall'],
                    mode='lines+markers',
                    name='Recall',
                    line=dict(color='orange', width=2),
                    marker=dict(size=4)
                ),
                row=2, col=1
            )
            
            # F1-Score trace
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['f1_score'],
                    mode='lines+markers',
                    name='F1-Score',
                    line=dict(color='red', width=2),
                    marker=dict(size=4)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Accuracy Metrics Trend - Model: {model_id}",
            height=600,
            showlegend=False,
            template='plotly_white'
        )
        
        # Update y-axes to show percentage
        for row in [1, 2]:
            for col in [1, 2]:
                fig.update_yaxes(
                    range=[0, 1],
                    tickformat='.1%',
                    row=row, col=col
                )
        
        return fig
    
    def _generate_confusion_matrix_figure(self, model_id: str, normalization: str = 'true') -> go.Figure:
        """Generate confusion matrix heatmap"""
        # Get latest confusion matrix from database
        try:
            cm_data = self.accuracy_db.get_latest_confusion_matrix(model_id)
        except Exception as e:
            logger.error(f"Error getting confusion matrix: {e}")
            cm_data = None
        
        if cm_data:
            cm = np.array([
                [cm_data['true_negatives'], cm_data['false_positives']],
                [cm_data['false_negatives'], cm_data['true_positives']]
            ])
            
            # Apply normalization if requested
            if normalization == 'true':
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_text = [[f'{val:.2%}' for val in row] for row in cm]
            elif normalization == 'pred':
                cm = cm.astype('float') / cm.sum(axis=0)
                cm_text = [[f'{val:.2%}' for val in row] for row in cm]
            else:
                cm_text = [[str(val) for val in row] for row in cm]
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Legitimate', 'Predicted Fraud'],
                y=['Actual Legitimate', 'Actual Fraud'],
                text=cm_text,
                texttemplate='%{text}',
                colorscale='Blues',
                showscale=True
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Confusion Matrix - Model: {model_id}",
                xaxis_title="Predicted Label",
                yaxis_title="Actual Label",
                height=400,
                template='plotly_white'
            )
        else:
            # Empty figure if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No confusion matrix data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            fig.update_layout(
                title=f"Confusion Matrix - Model: {model_id}",
                height=400,
                template='plotly_white'
            )
        
        return fig
    
    def _generate_accuracy_metrics_cards(self, model_id: str) -> List[dbc.Card]:
        """Generate accuracy metric cards"""
        # Get real-time accuracy metrics
        metrics = self._get_real_time_accuracy(model_id)
        
        # Get comparison metrics (1 hour ago)
        try:
            comparison_metrics = self.real_time_monitor.calculate_live_accuracy(
                model_id,
                time_window=timedelta(hours=1)
            )
        except Exception as e:
            logger.warning(f"Could not get comparison metrics: {e}")
            comparison_metrics = None
        
        cards = []
        metric_info = [
            ('accuracy', 'Accuracy', 'primary', 'fas fa-bullseye'),
            ('precision', 'Precision', 'success', 'fas fa-crosshairs'),
            ('recall', 'Recall', 'warning', 'fas fa-search'),
            ('f1_score', 'F1-Score', 'info', 'fas fa-balance-scale')
        ]
        
        for metric_key, metric_name, color, icon in metric_info:
            current_value = metrics.get(metric_key, 0)
            previous_value = comparison_metrics.get(metric_key, 0) if comparison_metrics else current_value
            
            # Calculate trend
            if previous_value > 0:
                trend = ((current_value - previous_value) / previous_value) * 100
                trend_icon = 'fas fa-arrow-up' if trend > 0 else 'fas fa-arrow-down'
                trend_color = 'success' if trend > 0 else 'danger'
            else:
                trend = 0
                trend_icon = 'fas fa-minus'
                trend_color = 'secondary'
            
            card = dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className=f"{icon} fa-2x mb-3", style={'color': f'var(--bs-{color})'}),
                        html.H4(metric_name, className="card-title"),
                        html.H2(f"{current_value:.2%}", className="mb-2"),
                        html.Div([
                            html.I(className=f"{trend_icon} me-1", style={'color': f'var(--bs-{trend_color})'}),
                            html.Span(f"{abs(trend):.1f}%", className=f"text-{trend_color}"),
                            html.Small(" vs 1h ago", className="text-muted ms-1")
                        ])
                    ], className="text-center")
                ])
            ], className="h-100", style={'minHeight': '180px'})
            
            cards.append(dbc.Col(card, width=3))
        
        return cards
    
    def _generate_model_comparison_figure(self, model_ids: List[str], metric: str = 'accuracy') -> go.Figure:
        """Generate model comparison visualization"""
        if not model_ids:
            fig = go.Figure()
            fig.add_annotation(
                text="Please select models to compare",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            fig.update_layout(
                title="Model Comparison",
                height=400,
                template='plotly_white'
            )
            return fig
        
        # Get metrics for each model
        comparison_data = []
        for model_id in model_ids:
            try:
                metrics = self.real_time_monitor.calculate_live_accuracy(model_id)
                if metrics:
                    comparison_data.append({
                        'model_id': model_id,
                        'accuracy': metrics.get('accuracy', 0),
                        'precision': metrics.get('precision', 0),
                        'recall': metrics.get('recall', 0),
                        'f1_score': metrics.get('f1_score', 0)
                    })
            except Exception as e:
                logger.warning(f"Could not get metrics for model {model_id}: {e}")
        
        if not comparison_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for selected models",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            fig.update_layout(
                title="Model Comparison",
                height=400,
                template='plotly_white'
            )
            return fig
        
        # Create comparison visualization based on metric type
        if metric == 'all':
            # Radar chart for all metrics
            categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            fig = go.Figure()
            
            for data in comparison_data:
                values = [
                    data['accuracy'],
                    data['precision'],
                    data['recall'],
                    data['f1_score']
                ]
                values.append(values[0])  # Complete the circle
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=data['model_id']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickformat='.0%'
                    )
                ),
                showlegend=True,
                title="Model Performance Comparison - All Metrics",
                height=500,
                template='plotly_white'
            )
        else:
            # Bar chart for single metric
            df = pd.DataFrame(comparison_data)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=df['model_id'],
                    y=df[metric],
                    text=[f'{val:.2%}' for val in df[metric]],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f"Model Comparison - {metric.capitalize()}",
                xaxis_title="Model ID",
                yaxis_title=metric.capitalize(),
                yaxis_tickformat='.0%',
                yaxis_range=[0, 1],
                height=400,
                template='plotly_white'
            )
        
        return fig
    
    def _get_real_time_accuracy(self, model_id: str) -> Dict[str, float]:
        """Get real-time accuracy metrics for a model"""
        # Try to get from cache first
        cache_key = f"{model_id}_latest"
        if cache_key in self.confusion_matrix_cache:
            cached_data = self.confusion_matrix_cache[cache_key]
            if (datetime.now() - cached_data['timestamp']).seconds < 30:
                return cached_data['metrics']
        
        # Calculate fresh metrics
        try:
            metrics = self.real_time_monitor.calculate_live_accuracy(model_id)
        except Exception as e:
            logger.error(f"Error calculating live accuracy for {model_id}: {e}")
            metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
        
        # Update cache
        self.confusion_matrix_cache[cache_key] = {
            'metrics': metrics,
            'timestamp': datetime.now()
        }
        
        return metrics
    
    def create_accuracy_dashboard_layout(self) -> html.Div:
        """Create the accuracy dashboard layout"""
        return html.Div([
            dbc.Container([
                # Header
                dbc.Row([
                    dbc.Col([
                        html.H1("Accuracy Monitoring Dashboard", className="mb-4"),
                        html.Hr()
                    ])
                ]),
                
                # Controls
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Select Model:"),
                        dcc.Dropdown(
                            id='model-selector',
                            options=[],  # Will be populated dynamically
                            value=None,
                            clearable=False
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Time Range:"),
                        dcc.Dropdown(
                            id='time-range-selector',
                            options=[
                                {'label': 'Last Hour', 'value': '1h'},
                                {'label': 'Last 24 Hours', 'value': '24h'},
                                {'label': 'Last 7 Days', 'value': '7d'},
                                {'label': 'Last 30 Days', 'value': '30d'}
                            ],
                            value='24h',
                            clearable=False
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Auto-refresh:"),
                        dbc.Switch(
                            id='auto-refresh-switch',
                            label="Enabled",
                            value=True
                        )
                    ], width=4)
                ], className="mb-4"),
                
                # Metrics Cards
                dbc.Row(id='accuracy-metrics-cards', className="mb-4"),
                
                # Main Charts
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Graph(id='accuracy-trend-graph')
                            ])
                        ])
                    ], width=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                dbc.Label("Matrix Normalization:"),
                                dcc.RadioItems(
                                    id='matrix-normalization',
                                    options=[
                                        {'label': 'True', 'value': 'true'},
                                        {'label': 'Predicted', 'value': 'pred'},
                                        {'label': 'None', 'value': 'none'}
                                    ],
                                    value='true',
                                    inline=True,
                                    className="mb-2"
                                ),
                                dcc.Graph(id='confusion-matrix-heatmap')
                            ])
                        ])
                    ], width=4)
                ], className="mb-4"),
                
                # Model Comparison
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Model Comparison", className="card-title mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Select Models:"),
                                        dcc.Dropdown(
                                            id='comparison-models',
                                            options=[],  # Will be populated dynamically
                                            value=[],
                                            multi=True
                                        )
                                    ], width=8),
                                    dbc.Col([
                                        dbc.Label("Comparison Type:"),
                                        dcc.Dropdown(
                                            id='comparison-metric',
                                            options=[
                                                {'label': 'All Metrics', 'value': 'all'},
                                                {'label': 'Accuracy', 'value': 'accuracy'},
                                                {'label': 'Precision', 'value': 'precision'},
                                                {'label': 'Recall', 'value': 'recall'},
                                                {'label': 'F1-Score', 'value': 'f1_score'}
                                            ],
                                            value='all',
                                            clearable=False
                                        )
                                    ], width=4)
                                ], className="mb-3"),
                                dcc.Graph(id='model-comparison-chart')
                            ])
                        ])
                    ])
                ]),
                
                # Update interval component
                dcc.Interval(
                    id='accuracy-update-interval',
                    interval=30*1000,  # 30 seconds
                    disabled=False
                )
                
            ], fluid=True)
        ])
    
    def handle_training_accuracy_update(self, model_id: str, epoch: int, 
                                      train_metrics: Dict, val_metrics: Dict):
        """Handle accuracy updates during training"""
        try:
            # Record training accuracy
            if train_metrics:
                self.accuracy_db.record_accuracy_metrics(
                    model_id=model_id,
                    model_version=f"epoch_{epoch}",
                    y_true=train_metrics.get('y_true', []),
                    y_pred=train_metrics.get('y_pred', []),
                    y_proba=train_metrics.get('y_proba'),
                    data_type='training'
                )
            
            # Record validation accuracy
            if val_metrics:
                self.accuracy_db.record_accuracy_metrics(
                    model_id=model_id,
                    model_version=f"epoch_{epoch}",
                    y_true=val_metrics.get('y_true', []),
                    y_pred=val_metrics.get('y_pred', []),
                    y_proba=val_metrics.get('y_proba'),
                    data_type='validation'
                )
            
            # Emit real-time update
            if self.dashboard_engine._socket_io:
                self.dashboard_engine._socket_io.emit('training_accuracy_update', {
                    'model_id': model_id,
                    'epoch': epoch,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"Error handling training accuracy update: {e}")


# Integration function to be called from main application
def integrate_accuracy_visualization(dashboard_engine: VisualizationDashboardEngine):
    """Main integration function to add accuracy visualization to the dashboard"""
    accuracy_extension = AccuracyVisualizationExtension(dashboard_engine)
    
    # Return the extension instance for further use
    return accuracy_extension