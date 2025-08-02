"""
Forecasting Engine for Predictive Accuracy Analysis
Advanced time series forecasting with ARIMA, Prophet, LSTM, and ensemble methods
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
import time
import threading
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# Time series forecasting imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import acf, pacf, adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    warnings.warn("ARIMA models not available. Install statsmodels for ARIMA support.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False
        warnings.warn("Prophet not available. Install prophet for Prophet forecasting.")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    warnings.warn("TensorFlow not available. Install tensorflow for LSTM forecasting.")


class ForecastingEngine:
    """
    Engine for performing advanced time series forecasting.
    Supports ARIMA, Prophet, LSTM, and ensemble methods for accuracy prediction.
    """
    
    def __init__(self, logger: logging.Logger = None):
        """Initialize the forecasting engine."""
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._lock = threading.Lock()
    
    def generate_forecasts(self, historical_data: Dict[str, Any], 
                          forecasting_models: List[str], 
                          forecast_horizon: str) -> Dict[str, Any]:
        """Generate forecasts using multiple models"""
        with self._lock:
            # Prepare data
            prepared_data = self._prepare_forecasting_data(historical_data)
            
            # Set forecast periods based on horizon
            periods = self._get_forecast_periods(forecast_horizon)
            
            # Generate forecasts with each model
            forecast_results = {}
            model_performance = {}
            
            for model_name in forecasting_models:
                if model_name == "ensemble":
                    continue  # Handle ensemble separately
                    
                try:
                    self.logger.info(f"Generating {model_name} forecast")
                    
                    if model_name == "arima" and ARIMA_AVAILABLE:
                        forecast_result, performance = self._generate_arima_forecast(
                            prepared_data, periods
                        )
                    elif model_name == "prophet" and PROPHET_AVAILABLE:
                        forecast_result, performance = self._generate_prophet_forecast(
                            prepared_data, periods
                        )
                    elif model_name == "lstm" and LSTM_AVAILABLE:
                        forecast_result, performance = self._generate_lstm_forecast(
                            prepared_data, periods
                        )
                    elif model_name == "simple_nn":
                        forecast_result, performance = self._generate_simple_nn_forecast(
                            prepared_data, periods
                        )
                    else:
                        self.logger.warning(f"Model {model_name} not available, skipping")
                        continue
                    
                    forecast_results[model_name] = forecast_result
                    model_performance[model_name] = performance
                    
                except Exception as e:
                    self.logger.error(f"Error generating {model_name} forecast: {e}")
                    continue
            
            # Generate ensemble forecast if requested and we have multiple models
            ensemble_forecast = None
            if "ensemble" in forecasting_models and len(forecast_results) > 1:
                ensemble_forecast = self._generate_ensemble_forecast(
                    forecast_results, model_performance, periods
                )
            
            return {
                "individual_forecasts": forecast_results,
                "ensemble_forecast": ensemble_forecast,
                "model_performance": model_performance,
                "data_info": prepared_data["data_info"]
            }
    
    def _prepare_forecasting_data(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for forecasting"""
        try:
            # Extract time series data
            if "time_series" in historical_data:
                ts_data = historical_data["time_series"]
            elif "accuracy_values" in historical_data:
                ts_data = historical_data["accuracy_values"]
            else:
                raise ValueError("No time series data found")
            
            # Convert to pandas DataFrame
            if isinstance(ts_data, dict):
                if "timestamps" in ts_data and "values" in ts_data:
                    df = pd.DataFrame({
                        "ds": pd.to_datetime(ts_data["timestamps"]),
                        "y": ts_data["values"]
                    })
                else:
                    # Assume dict keys are timestamps
                    timestamps = list(ts_data.keys())
                    values = list(ts_data.values())
                    df = pd.DataFrame({
                        "ds": pd.to_datetime(timestamps),
                        "y": values
                    })
            elif isinstance(ts_data, list):
                # Create timestamps if not provided
                df = pd.DataFrame({
                    "ds": pd.date_range(start="2023-01-01", periods=len(ts_data), freq="D"),
                    "y": ts_data
                })
            else:
                raise ValueError("Unsupported time series data format")
            
            # Sort by timestamp and remove duplicates
            df = df.sort_values("ds").drop_duplicates(subset=["ds"]).reset_index(drop=True)
            
            # Handle missing values
            df["y"] = df["y"].interpolate(method="linear")
            df = df.dropna()
            
            if len(df) < 10:
                raise ValueError("Insufficient data for forecasting (minimum 10 points required)")
            
            # Detect frequency
            time_diffs = df["ds"].diff().dropna()
            freq = time_diffs.mode().iloc[0] if len(time_diffs) > 0 else pd.Timedelta(days=1)
            
            # Split train/test for validation
            test_size = min(int(len(df) * 0.2), 30)  # 20% or max 30 points
            train_df = df.iloc[:-test_size]
            test_df = df.iloc[-test_size:]
            
            return {
                "full_data": df,
                "train_data": train_df,
                "test_data": test_df,
                "frequency": freq,
                "data_info": {
                    "total_points": len(df),
                    "train_points": len(train_df),
                    "test_points": len(test_df),
                    "date_range": {
                        "start": df["ds"].min().isoformat(),
                        "end": df["ds"].max().isoformat()
                    },
                    "frequency_detected": str(freq)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise
    
    def _get_forecast_periods(self, horizon: str) -> int:
        """Get number of forecast periods based on horizon"""
        horizon_map = {
            "short": 7,    # 1 week
            "medium": 30,  # 1 month
            "long": 90,    # 3 months
            "weekly": 7,
            "monthly": 30,
            "quarterly": 90
        }
        return horizon_map.get(horizon.lower(), 30)
    
    def _generate_arima_forecast(self, prepared_data: Dict[str, Any], 
                                periods: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate ARIMA forecast"""
        train_data = prepared_data["train_data"]
        test_data = prepared_data["test_data"]
        
        # Prepare time series
        ts = train_data.set_index("ds")["y"]
        
        # Check stationarity
        adf_result = adfuller(ts.dropna())
        is_stationary = adf_result[1] < 0.05
        
        # Auto-select ARIMA order
        best_order = self._select_arima_order(ts)
        
        # Fit ARIMA model
        model = ARIMA(ts, order=best_order)
        fitted_model = model.fit()
        
        # Generate forecast
        forecast_result = fitted_model.forecast(steps=periods, alpha=0.05)
        forecast_values = forecast_result.values if hasattr(forecast_result, 'values') else forecast_result
        
        # Generate confidence intervals
        forecast_ci = fitted_model.get_forecast(steps=periods, alpha=0.05).conf_int()
        
        # Generate out-of-sample forecast for validation
        validation_forecast = fitted_model.forecast(steps=len(test_data))
        
        # Calculate performance metrics
        performance = self._calculate_forecast_performance(
            test_data["y"].values, validation_forecast
        )
        
        # Create forecast dates
        last_date = train_data["ds"].iloc[-1]
        freq = prepared_data["frequency"]
        forecast_dates = pd.date_range(start=last_date + freq, periods=periods, freq=freq)
        
        forecast_result = {
            "model_type": "arima",
            "order": best_order,
            "forecast_dates": forecast_dates.tolist(),
            "forecast_values": forecast_values.tolist() if hasattr(forecast_values, 'tolist') else list(forecast_values),
            "confidence_intervals": {
                "lower": forecast_ci.iloc[:, 0].tolist(),
                "upper": forecast_ci.iloc[:, 1].tolist()
            },
            "model_diagnostics": {
                "aic": fitted_model.aic,
                "bic": fitted_model.bic,
                "is_stationary": is_stationary,
                "adf_pvalue": adf_result[1]
            }
        }
        
        return forecast_result, performance
    
    def _generate_prophet_forecast(self, prepared_data: Dict[str, Any], 
                                  periods: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate Prophet forecast"""
        train_data = prepared_data["train_data"].copy()
        test_data = prepared_data["test_data"]
        
        # Initialize and fit Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )
        
        model.fit(train_data)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract forecast for new periods only
        forecast_future = forecast.tail(periods)
        
        # Generate validation forecast
        validation_future = model.make_future_dataframe(periods=len(test_data))
        validation_forecast = model.predict(validation_future)
        validation_values = validation_forecast.tail(len(test_data))["yhat"].values
        
        # Calculate performance metrics
        performance = self._calculate_forecast_performance(
            test_data["y"].values, validation_values
        )
        
        forecast_result = {
            "model_type": "prophet",
            "forecast_dates": forecast_future["ds"].dt.strftime('%Y-%m-%d').tolist(),
            "forecast_values": forecast_future["yhat"].tolist(),
            "confidence_intervals": {
                "lower": forecast_future["yhat_lower"].tolist(),
                "upper": forecast_future["yhat_upper"].tolist()
            },
            "components": {
                "trend": forecast_future["trend"].tolist(),
                "seasonal": {
                    "yearly": forecast_future.get("yearly", [0]*periods),
                    "weekly": forecast_future.get("weekly", [0]*periods)
                }
            },
            "changepoints": {
                "dates": model.changepoints.dt.strftime('%Y-%m-%d').tolist(),
                "deltas": model.params["delta"].tolist() if "delta" in model.params else []
            }
        }
        
        return forecast_result, performance
    
    def _generate_lstm_forecast(self, prepared_data: Dict[str, Any], 
                               periods: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate LSTM forecast"""
        train_data = prepared_data["train_data"]
        test_data = prepared_data["test_data"]
        
        # Prepare sequences for LSTM
        sequence_length = min(10, len(train_data) // 3)
        
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:(i + seq_length)])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)
        
        # Scale data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(train_data[["y"]])
        
        # Create sequences
        X_train, y_train = create_sequences(scaled_train.flatten(), sequence_length)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        # Build LSTM model
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train model
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        # Generate forecast
        last_sequence = scaled_train[-sequence_length:].flatten()
        forecast_scaled = []
        
        for _ in range(periods):
            next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
            forecast_scaled.append(next_pred[0, 0])
            last_sequence = np.append(last_sequence[1:], next_pred[0, 0])
        
        # Inverse transform
        forecast_values = scaler.inverse_transform(
            np.array(forecast_scaled).reshape(-1, 1)
        ).flatten()
        
        # Generate validation forecast
        scaled_test = scaler.transform(test_data[["y"]])
        validation_forecast = []
        val_sequence = scaled_train[-sequence_length:].flatten()
        
        for _ in range(len(test_data)):
            next_pred = model.predict(val_sequence.reshape(1, sequence_length, 1), verbose=0)
            validation_forecast.append(next_pred[0, 0])
            val_sequence = np.append(val_sequence[1:], next_pred[0, 0])
        
        validation_values = scaler.inverse_transform(
            np.array(validation_forecast).reshape(-1, 1)
        ).flatten()
        
        # Calculate performance metrics
        performance = self._calculate_forecast_performance(
            test_data["y"].values, validation_values
        )
        
        # Create forecast dates
        last_date = train_data["ds"].iloc[-1]
        freq = prepared_data["frequency"]
        forecast_dates = pd.date_range(start=last_date + freq, periods=periods, freq=freq)
        
        # Estimate confidence intervals (simple approach)
        residuals = validation_values - test_data["y"].values
        std_residual = np.std(residuals)
        ci_lower = forecast_values - 1.96 * std_residual
        ci_upper = forecast_values + 1.96 * std_residual
        
        forecast_result = {
            "model_type": "lstm",
            "sequence_length": sequence_length,
            "forecast_dates": forecast_dates.tolist(),
            "forecast_values": forecast_values.tolist(),
            "confidence_intervals": {
                "lower": ci_lower.tolist(),
                "upper": ci_upper.tolist()
            },
            "model_info": {
                "epochs": 50,
                "architecture": "LSTM-50-LSTM-50-Dense-25-Dense-1",
                "sequence_length": sequence_length
            }
        }
        
        return forecast_result, performance
    
    def _generate_simple_nn_forecast(self, prepared_data: Dict[str, Any], 
                                    periods: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate simple neural network forecast (fallback when TensorFlow unavailable)"""
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import MinMaxScaler
        
        train_data = prepared_data["train_data"]
        test_data = prepared_data["test_data"]
        
        # Create features (lagged values)
        def create_features(data, lags=5):
            features = []
            targets = []
            
            for i in range(lags, len(data)):
                features.append(data[i-lags:i])
                targets.append(data[i])
                
            return np.array(features), np.array(targets)
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(train_data[["y"]])
        
        # Create features
        lags = min(5, len(train_data) // 3)
        X_train, y_train = create_features(scaled_train.flatten(), lags)
        
        # Train model
        model = MLPRegressor(
            hidden_layer_sizes=(50, 25),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )
        
        model.fit(X_train, y_train)
        
        # Generate forecast
        last_values = scaled_train[-lags:].flatten()
        forecast_scaled = []
        
        for _ in range(periods):
            next_pred = model.predict([last_values])[0]
            forecast_scaled.append(next_pred)
            last_values = np.append(last_values[1:], next_pred)
        
        # Inverse transform
        forecast_values = scaler.inverse_transform(
            np.array(forecast_scaled).reshape(-1, 1)
        ).flatten()
        
        # Generate validation forecast
        validation_forecast = []
        val_values = scaled_train[-lags:].flatten()
        
        for _ in range(len(test_data)):
            next_pred = model.predict([val_values])[0]
            validation_forecast.append(next_pred)
            val_values = np.append(val_values[1:], next_pred)
        
        validation_values = scaler.inverse_transform(
            np.array(validation_forecast).reshape(-1, 1)
        ).flatten()
        
        # Calculate performance metrics
        performance = self._calculate_forecast_performance(
            test_data["y"].values, validation_values
        )
        
        # Create forecast dates
        last_date = train_data["ds"].iloc[-1]
        freq = prepared_data["frequency"]
        forecast_dates = pd.date_range(start=last_date + freq, periods=periods, freq=freq)
        
        # Estimate confidence intervals
        residuals = validation_values - test_data["y"].values
        std_residual = np.std(residuals)
        ci_lower = forecast_values - 1.96 * std_residual
        ci_upper = forecast_values + 1.96 * std_residual
        
        forecast_result = {
            "model_type": "simple_nn",
            "lags": lags,
            "forecast_dates": forecast_dates.tolist(),
            "forecast_values": forecast_values.tolist(),
            "confidence_intervals": {
                "lower": ci_lower.tolist(),
                "upper": ci_upper.tolist()
            },
            "model_info": {
                "architecture": "MLP(50,25)",
                "lags": lags,
                "iterations": model.n_iter_
            }
        }
        
        return forecast_result, performance
    
    def _generate_ensemble_forecast(self, individual_forecasts: Dict[str, Any], 
                                   model_performance: Dict[str, Any], 
                                   periods: int) -> Dict[str, Any]:
        """Generate ensemble forecast combining multiple models"""
        if len(individual_forecasts) < 2:
            return None
        
        # Extract forecast values from each model
        forecasts = {}
        weights = {}
        
        for model_name, forecast in individual_forecasts.items():
            forecasts[model_name] = np.array(forecast["forecast_values"])
            # Weight by inverse RMSE (better models get higher weight)
            rmse = model_performance[model_name].get("rmse", 1.0)
            weights[model_name] = 1.0 / (rmse + 1e-10)
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Generate ensemble forecasts using different methods
        ensemble_methods = {}
        
        # 1. Weighted average
        weighted_forecast = np.zeros(periods)
        for model_name, forecast_values in forecasts.items():
            weighted_forecast += weights[model_name] * forecast_values
        
        ensemble_methods["weighted_average"] = {
            "forecast_values": weighted_forecast.tolist(),
            "weights": weights
        }
        
        # 2. Simple median
        forecast_matrix = np.array(list(forecasts.values()))
        median_forecast = np.median(forecast_matrix, axis=0)
        
        ensemble_methods["median"] = {
            "forecast_values": median_forecast.tolist()
        }
        
        # 3. Trimmed mean (remove best and worst)
        if len(forecasts) >= 3:
            trimmed_forecast = np.mean(
                np.sort(forecast_matrix, axis=0)[1:-1], axis=0
            )
            ensemble_methods["trimmed_mean"] = {
                "forecast_values": trimmed_forecast.tolist()
            }
        
        # Select best ensemble method (use weighted average as default)
        selected_method = "weighted_average"
        selected_forecast = ensemble_methods[selected_method]
        
        # Calculate ensemble confidence intervals
        forecast_std = np.std(forecast_matrix, axis=0)
        ensemble_ci_lower = selected_forecast["forecast_values"] - 1.96 * forecast_std
        ensemble_ci_upper = selected_forecast["forecast_values"] + 1.96 * forecast_std
        
        # Use dates from first available forecast
        forecast_dates = list(individual_forecasts.values())[0]["forecast_dates"]
        
        return {
            "ensemble_methods": ensemble_methods,
            "selected_method": selected_method,
            "forecast_dates": forecast_dates,
            "forecast_values": selected_forecast["forecast_values"],
            "confidence_intervals": {
                "lower": ensemble_ci_lower.tolist(),
                "upper": ensemble_ci_upper.tolist()
            },
            "model_weights": weights,
            "ensemble_variance": forecast_std.tolist()
        }
    
    def _select_arima_order(self, ts: pd.Series) -> Tuple[int, int, int]:
        """Auto-select ARIMA order using AIC"""
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        # Test different orders
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(ts, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    def _calculate_forecast_performance(self, actual: np.ndarray, 
                                       predicted: np.ndarray) -> Dict[str, float]:
        """Calculate forecast performance metrics"""
        try:
            # Ensure arrays have same length
            min_len = min(len(actual), len(predicted))
            actual = actual[:min_len]
            predicted = predicted[:min_len]
            
            # RMSE
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            
            # MAE
            mae = np.mean(np.abs(actual - predicted))
            
            # MAPE
            mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
            
            # Directional accuracy
            if len(actual) > 1:
                actual_direction = np.diff(actual) > 0
                predicted_direction = np.diff(predicted) > 0
                directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
            else:
                directional_accuracy = 50.0
            
            # R-squared
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))
            
            return {
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "directional_accuracy": float(directional_accuracy),
                "r_squared": float(r_squared),
                "sample_size": len(actual)
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating performance metrics: {e}")
            return {
                "rmse": float('inf'),
                "mae": float('inf'),
                "mape": float('inf'),
                "directional_accuracy": 0.0,
                "r_squared": 0.0,
                "sample_size": 0
            }
