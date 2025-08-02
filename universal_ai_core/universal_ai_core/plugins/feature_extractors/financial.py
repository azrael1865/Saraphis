#!/usr/bin/env python3
"""
Financial Analysis Feature Extractor Plugin
===========================================

This module provides financial data analysis feature extraction capabilities for the Universal AI Core system.
Adapted from Saraphis molecular descriptor patterns, specialized for financial market analysis,
time series feature extraction, and quantitative finance applications.

Features:
- Time series technical indicator analysis
- Market microstructure feature extraction
- Risk and volatility analysis features
- Fundamental analysis indicators
- Portfolio optimization features
- High-frequency trading signal extraction
- Economic indicator analysis
- Financial ratio computation
"""

import logging
import sys
import time
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import math
import statistics

# Try to import financial analysis dependencies
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from scipy import stats
    from scipy.fft import fft, ifft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import plugin base classes
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base import FeatureExtractorPlugin, FeatureExtractionResult, FeatureType

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesData:
    """Time series representation for financial data"""
    timestamp: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    symbol: str = ""
    frequency: str = "1D"  # 1min, 5min, 1H, 1D, etc.
    
    @property
    def returns(self) -> np.ndarray:
        """Calculate simple returns"""
        return np.diff(self.close) / self.close[:-1]
    
    @property
    def log_returns(self) -> np.ndarray:
        """Calculate log returns"""
        return np.diff(np.log(self.close))
    
    @property
    def typical_price(self) -> np.ndarray:
        """Calculate typical price (HLC/3)"""
        return (self.high + self.low + self.close) / 3
    
    @property
    def true_range(self) -> np.ndarray:
        """Calculate true range"""
        hl = self.high - self.low
        hc = np.abs(self.high - np.roll(self.close, 1))
        lc = np.abs(self.low - np.roll(self.close, 1))
        tr = np.maximum(hl, np.maximum(hc, lc))
        return tr[1:]  # Remove first element due to roll


@dataclass
class FinancialAnalysisResult:
    """Result container for financial analysis"""
    technical_features: np.ndarray = field(default_factory=lambda: np.array([]))
    fundamental_features: np.ndarray = field(default_factory=lambda: np.array([]))
    risk_features: np.ndarray = field(default_factory=lambda: np.array([]))
    microstructure_features: np.ndarray = field(default_factory=lambda: np.array([]))
    sentiment_features: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_names: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    valid_periods: int = 0
    invalid_periods: int = 0
    error_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TechnicalIndicatorAnalyzer:
    """
    Technical indicator analysis for financial time series.
    
    Adapted from molecular descriptor calculation patterns.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TechnicalIndicatorAnalyzer")
        self.cache = {}
        self.cache_lock = threading.Lock()
    
    def calculate_indicators(self, data: TimeSeriesData, 
                           lookback_periods: List[int] = None) -> Dict[str, np.ndarray]:
        """Calculate comprehensive technical indicators"""
        if lookback_periods is None:
            lookback_periods = [5, 10, 20, 50, 100, 200]
        
        try:
            indicators = {}
            
            # Cache key for this calculation
            cache_key = f"{data.symbol}_{len(data.close)}_{hash(tuple(data.close[-100:]))}"
            
            with self.cache_lock:
                if cache_key in self.cache:
                    return self.cache[cache_key]
            
            # Price-based indicators
            indicators.update(self._calculate_price_indicators(data, lookback_periods))
            
            # Volume-based indicators
            indicators.update(self._calculate_volume_indicators(data, lookback_periods))
            
            # Volatility indicators
            indicators.update(self._calculate_volatility_indicators(data, lookback_periods))
            
            # Momentum indicators
            indicators.update(self._calculate_momentum_indicators(data, lookback_periods))
            
            # Pattern recognition indicators
            indicators.update(self._calculate_pattern_indicators(data))
            
            # Regime detection indicators
            indicators.update(self._calculate_regime_indicators(data))
            
            # Cache results
            with self.cache_lock:
                self.cache[cache_key] = indicators
                # Limit cache size
                if len(self.cache) > 1000:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def _calculate_price_indicators(self, data: TimeSeriesData, 
                                  periods: List[int]) -> Dict[str, np.ndarray]:
        """Calculate price-based technical indicators"""
        indicators = {}
        
        try:
            close = data.close
            high = data.high
            low = data.low
            open_price = data.open
            
            # Moving averages
            for period in periods:
                if len(close) >= period:
                    # Simple Moving Average
                    sma = np.convolve(close, np.ones(period), 'valid') / period
                    indicators[f'sma_{period}'] = self._pad_array(sma, len(close))
                    
                    # Exponential Moving Average
                    ema = self._calculate_ema(close, period)
                    indicators[f'ema_{period}'] = ema
                    
                    # Bollinger Bands
                    bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close, period)
                    indicators[f'bb_upper_{period}'] = bb_upper
                    indicators[f'bb_lower_{period}'] = bb_lower
                    indicators[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
                    indicators[f'bb_position_{period}'] = (close - bb_lower) / (bb_upper - bb_lower)
            
            # Price ratios and relationships
            indicators['hl_ratio'] = (high - low) / close
            indicators['oc_ratio'] = (close - open_price) / open_price
            indicators['price_acceleration'] = np.gradient(np.gradient(close))
            
            # Support and resistance levels
            indicators['resistance_strength'] = self._calculate_resistance_strength(high, close)
            indicators['support_strength'] = self._calculate_support_strength(low, close)
            
            # Gap analysis
            indicators['gap_up'] = np.maximum(0, open_price[1:] - high[:-1])
            indicators['gap_down'] = np.maximum(0, low[:-1] - open_price[1:])
            indicators['gap_up'] = self._pad_array(indicators['gap_up'], len(close))
            indicators['gap_down'] = self._pad_array(indicators['gap_down'], len(close))
            
        except Exception as e:
            self.logger.error(f"Error calculating price indicators: {e}")
        
        return indicators
    
    def _calculate_volume_indicators(self, data: TimeSeriesData,
                                   periods: List[int]) -> Dict[str, np.ndarray]:
        """Calculate volume-based indicators"""
        indicators = {}
        
        try:
            close = data.close
            volume = data.volume
            typical_price = data.typical_price
            
            # Volume moving averages
            for period in periods[:4]:  # Use shorter periods for volume
                if len(volume) >= period:
                    vol_sma = np.convolve(volume, np.ones(period), 'valid') / period
                    indicators[f'volume_sma_{period}'] = self._pad_array(vol_sma, len(volume))
                    
                    # Volume ratio
                    indicators[f'volume_ratio_{period}'] = volume / self._pad_array(vol_sma, len(volume))
            
            # Volume-Price Trend (VPT)
            returns = np.diff(close) / close[:-1]
            vpt = np.cumsum(np.concatenate([[0], returns * volume[1:]]))
            indicators['vpt'] = vpt
            
            # On-Balance Volume (OBV)
            obv = np.zeros_like(close)
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv[i] = obv[i-1] + volume[i]
                elif close[i] < close[i-1]:
                    obv[i] = obv[i-1] - volume[i]
                else:
                    obv[i] = obv[i-1]
            indicators['obv'] = obv
            
            # Money Flow Index (MFI)
            if len(close) >= 14:
                mfi = self._calculate_money_flow_index(data, 14)
                indicators['mfi'] = mfi
            
            # Volume Rate of Change
            for period in [10, 20]:
                if len(volume) > period:
                    vol_roc = (volume[period:] - volume[:-period]) / volume[:-period]
                    indicators[f'volume_roc_{period}'] = self._pad_array(vol_roc, len(volume))
            
            # Accumulation/Distribution Line
            ad_line = self._calculate_accumulation_distribution(data)
            indicators['ad_line'] = ad_line
            
        except Exception as e:
            self.logger.error(f"Error calculating volume indicators: {e}")
        
        return indicators
    
    def _calculate_volatility_indicators(self, data: TimeSeriesData,
                                       periods: List[int]) -> Dict[str, np.ndarray]:
        """Calculate volatility-based indicators"""
        indicators = {}
        
        try:
            close = data.close
            high = data.high
            low = data.low
            returns = data.log_returns
            
            # Historical volatility
            for period in periods[:5]:  # Use first 5 periods
                if len(returns) >= period:
                    rolling_vol = np.array([
                        np.std(returns[max(0, i-period):i]) if i >= period else np.nan
                        for i in range(len(returns))
                    ])
                    # Annualize volatility (assuming daily data)
                    rolling_vol *= np.sqrt(252)
                    indicators[f'volatility_{period}'] = np.concatenate([[np.nan], rolling_vol])
            
            # Average True Range (ATR)
            true_range = data.true_range
            for period in [10, 14, 20]:
                if len(true_range) >= period:
                    atr = np.convolve(true_range, np.ones(period), 'valid') / period
                    indicators[f'atr_{period}'] = self._pad_array(atr, len(close))
            
            # Volatility ratios
            if 'volatility_20' in indicators and 'volatility_50' in indicators:
                indicators['volatility_ratio_20_50'] = indicators['volatility_20'] / np.where(
                    indicators['volatility_50'] != 0, indicators['volatility_50'], 1
                )
            
            # Range-based volatility measures
            indicators['high_low_ratio'] = (high - low) / close
            indicators['close_to_close_vol'] = np.abs(np.diff(close) / close[:-1])
            indicators['close_to_close_vol'] = self._pad_array(indicators['close_to_close_vol'], len(close))
            
            # Parkinson volatility estimator
            parkinson_vol = np.sqrt(
                (1 / (4 * np.log(2))) * np.power(np.log(high / low), 2)
            )
            indicators['parkinson_volatility'] = parkinson_vol
            
            # Garman-Klass volatility estimator
            gk_vol = np.sqrt(
                0.5 * np.power(np.log(high / low), 2) - 
                (2 * np.log(2) - 1) * np.power(np.log(close / data.open), 2)
            )
            indicators['garman_klass_volatility'] = gk_vol
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility indicators: {e}")
        
        return indicators
    
    def _calculate_momentum_indicators(self, data: TimeSeriesData,
                                     periods: List[int]) -> Dict[str, np.ndarray]:
        """Calculate momentum-based indicators"""
        indicators = {}
        
        try:
            close = data.close
            high = data.high
            low = data.low
            
            # Rate of Change (ROC)
            for period in periods[:6]:
                if len(close) > period:
                    roc = (close[period:] - close[:-period]) / close[:-period] * 100
                    indicators[f'roc_{period}'] = self._pad_array(roc, len(close))
            
            # Relative Strength Index (RSI)
            for period in [9, 14, 21]:
                if len(close) >= period + 1:
                    rsi = self._calculate_rsi(close, period)
                    indicators[f'rsi_{period}'] = rsi
            
            # Stochastic Oscillator
            for period in [14, 21]:
                if len(close) >= period:
                    stoch_k, stoch_d = self._calculate_stochastic(data, period, 3)
                    indicators[f'stoch_k_{period}'] = stoch_k
                    indicators[f'stoch_d_{period}'] = stoch_d
            
            # MACD
            macd_line, macd_signal, macd_histogram = self._calculate_macd(close)
            indicators['macd_line'] = macd_line
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_histogram
            
            # Williams %R
            for period in [14, 21]:
                if len(close) >= period:
                    williams_r = self._calculate_williams_r(data, period)
                    indicators[f'williams_r_{period}'] = williams_r
            
            # Commodity Channel Index (CCI)
            for period in [14, 20]:
                if len(close) >= period:
                    cci = self._calculate_cci(data, period)
                    indicators[f'cci_{period}'] = cci
            
            # Momentum
            for period in [10, 20]:
                if len(close) > period:
                    momentum = close[period:] - close[:-period]
                    indicators[f'momentum_{period}'] = self._pad_array(momentum, len(close))
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {e}")
        
        return indicators
    
    def _calculate_pattern_indicators(self, data: TimeSeriesData) -> Dict[str, np.ndarray]:
        """Calculate pattern recognition indicators"""
        indicators = {}
        
        try:
            close = data.close
            high = data.high
            low = data.low
            open_price = data.open
            
            # Candlestick patterns (simplified versions)
            indicators['doji'] = np.abs(close - open_price) / (high - low + 1e-8) < 0.1
            indicators['hammer'] = (
                (close > open_price) & 
                ((close - open_price) / (high - low + 1e-8) > 0.6) &
                ((open_price - low) / (high - low + 1e-8) > 0.6)
            ).astype(float)
            
            indicators['shooting_star'] = (
                (open_price > close) & 
                ((high - open_price) / (high - low + 1e-8) > 0.6) &
                ((close - low) / (high - low + 1e-8) < 0.3)
            ).astype(float)
            
            # Price patterns
            indicators['higher_highs'] = self._detect_higher_highs(high)
            indicators['lower_lows'] = self._detect_lower_lows(low)
            indicators['inside_bars'] = self._detect_inside_bars(high, low)
            indicators['outside_bars'] = self._detect_outside_bars(high, low)
            
            # Trend strength
            indicators['trend_strength'] = self._calculate_trend_strength(close)
            
            # Fractal indicators
            indicators['fractal_highs'] = self._detect_fractal_highs(high)
            indicators['fractal_lows'] = self._detect_fractal_lows(low)
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern indicators: {e}")
        
        return indicators
    
    def _calculate_regime_indicators(self, data: TimeSeriesData) -> Dict[str, np.ndarray]:
        """Calculate market regime indicators"""
        indicators = {}
        
        try:
            close = data.close
            volume = data.volume
            returns = data.log_returns
            
            # Trend regime (using multiple timeframes)
            indicators['trend_regime'] = self._detect_trend_regime(close)
            
            # Volatility regime
            indicators['volatility_regime'] = self._detect_volatility_regime(returns)
            
            # Volume regime
            indicators['volume_regime'] = self._detect_volume_regime(volume)
            
            # Market efficiency measures
            indicators['hurst_exponent'] = self._calculate_hurst_exponent(returns)
            indicators['autocorrelation'] = self._calculate_autocorrelation(returns)
            
            # Regime transition probabilities
            indicators['regime_transition_prob'] = self._calculate_regime_transition_prob(close)
            
        except Exception as e:
            self.logger.error(f"Error calculating regime indicators: {e}")
        
        return indicators
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _calculate_bollinger_bands(self, data: np.ndarray, period: int, 
                                 std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        sma = np.convolve(data, np.ones(period), 'valid') / period
        sma_padded = self._pad_array(sma, len(data))
        
        # Calculate rolling standard deviation
        rolling_std = np.array([
            np.std(data[max(0, i-period):i]) if i >= period else 0
            for i in range(len(data))
        ])
        
        upper_band = sma_padded + std_dev * rolling_std
        lower_band = sma_padded - std_dev * rolling_std
        
        return upper_band, sma_padded, lower_band
    
    def _calculate_rsi(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Relative Strength Index"""
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros(len(data))
        avg_losses = np.zeros(len(data))
        
        # Initial calculation
        if len(gains) >= period:
            avg_gains[period] = np.mean(gains[:period])
            avg_losses[period] = np.mean(losses[:period])
            
            # Smoothed calculation
            for i in range(period + 1, len(data)):
                avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
                avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        
        rs = np.where(avg_losses != 0, avg_gains / avg_losses, 0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_stochastic(self, data: TimeSeriesData, k_period: int, 
                            d_period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic Oscillator"""
        high = data.high
        low = data.low
        close = data.close
        
        k_percent = np.zeros_like(close)
        
        for i in range(k_period - 1, len(close)):
            lowest_low = np.min(low[i - k_period + 1:i + 1])
            highest_high = np.max(high[i - k_period + 1:i + 1])
            
            if highest_high != lowest_low:
                k_percent[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Smooth %K to get %D
        d_percent = np.convolve(k_percent, np.ones(d_period), 'valid') / d_period
        d_percent = self._pad_array(d_percent, len(close))
        
        return k_percent, d_percent
    
    def _calculate_macd(self, data: np.ndarray, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD"""
        ema_fast = self._calculate_ema(data, fast)
        ema_slow = self._calculate_ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        macd_signal = self._calculate_ema(macd_line, signal)
        macd_histogram = macd_line - macd_signal
        
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_williams_r(self, data: TimeSeriesData, period: int) -> np.ndarray:
        """Calculate Williams %R"""
        high = data.high
        low = data.low
        close = data.close
        
        williams_r = np.zeros_like(close)
        
        for i in range(period - 1, len(close)):
            highest_high = np.max(high[i - period + 1:i + 1])
            lowest_low = np.min(low[i - period + 1:i + 1])
            
            if highest_high != lowest_low:
                williams_r[i] = ((highest_high - close[i]) / (highest_high - lowest_low)) * -100
        
        return williams_r
    
    def _calculate_cci(self, data: TimeSeriesData, period: int) -> np.ndarray:
        """Calculate Commodity Channel Index"""
        typical_price = data.typical_price
        
        cci = np.zeros_like(typical_price)
        
        for i in range(period - 1, len(typical_price)):
            tp_sma = np.mean(typical_price[i - period + 1:i + 1])
            mean_deviation = np.mean(np.abs(typical_price[i - period + 1:i + 1] - tp_sma))
            
            if mean_deviation != 0:
                cci[i] = (typical_price[i] - tp_sma) / (0.015 * mean_deviation)
        
        return cci
    
    def _calculate_money_flow_index(self, data: TimeSeriesData, period: int) -> np.ndarray:
        """Calculate Money Flow Index"""
        typical_price = data.typical_price
        volume = data.volume
        
        money_flow = typical_price * volume
        
        positive_flow = np.zeros_like(money_flow)
        negative_flow = np.zeros_like(money_flow)
        
        for i in range(1, len(typical_price)):
            if typical_price[i] > typical_price[i-1]:
                positive_flow[i] = money_flow[i]
            elif typical_price[i] < typical_price[i-1]:
                negative_flow[i] = money_flow[i]
        
        mfi = np.zeros_like(typical_price)
        
        for i in range(period - 1, len(typical_price)):
            pos_sum = np.sum(positive_flow[i - period + 1:i + 1])
            neg_sum = np.sum(negative_flow[i - period + 1:i + 1])
            
            if neg_sum != 0:
                money_ratio = pos_sum / neg_sum
                mfi[i] = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    def _calculate_accumulation_distribution(self, data: TimeSeriesData) -> np.ndarray:
        """Calculate Accumulation/Distribution Line"""
        high = data.high
        low = data.low
        close = data.close
        volume = data.volume
        
        ad_line = np.zeros_like(close)
        
        for i in range(1, len(close)):
            if high[i] != low[i]:
                money_flow_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
                money_flow_volume = money_flow_multiplier * volume[i]
                ad_line[i] = ad_line[i-1] + money_flow_volume
        
        return ad_line
    
    def _detect_higher_highs(self, high: np.ndarray, lookback: int = 5) -> np.ndarray:
        """Detect higher highs pattern"""
        higher_highs = np.zeros_like(high)
        
        for i in range(lookback, len(high)):
            recent_highs = high[i-lookback:i]
            if len(recent_highs) > 0 and high[i] > np.max(recent_highs):
                higher_highs[i] = 1
        
        return higher_highs
    
    def _detect_lower_lows(self, low: np.ndarray, lookback: int = 5) -> np.ndarray:
        """Detect lower lows pattern"""
        lower_lows = np.zeros_like(low)
        
        for i in range(lookback, len(low)):
            recent_lows = low[i-lookback:i]
            if len(recent_lows) > 0 and low[i] < np.min(recent_lows):
                lower_lows[i] = 1
        
        return lower_lows
    
    def _detect_inside_bars(self, high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """Detect inside bars"""
        inside_bars = np.zeros_like(high)
        
        for i in range(1, len(high)):
            if high[i] <= high[i-1] and low[i] >= low[i-1]:
                inside_bars[i] = 1
        
        return inside_bars
    
    def _detect_outside_bars(self, high: np.ndarray, low: np.ndarray) -> np.ndarray:
        """Detect outside bars"""
        outside_bars = np.zeros_like(high)
        
        for i in range(1, len(high)):
            if high[i] > high[i-1] and low[i] < low[i-1]:
                outside_bars[i] = 1
        
        return outside_bars
    
    def _calculate_trend_strength(self, close: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate trend strength"""
        trend_strength = np.zeros_like(close)
        
        for i in range(period, len(close)):
            price_changes = np.diff(close[i-period:i])
            if len(price_changes) > 0:
                positive_changes = np.sum(price_changes > 0)
                negative_changes = np.sum(price_changes < 0)
                total_changes = len(price_changes)
                
                if total_changes > 0:
                    trend_strength[i] = (positive_changes - negative_changes) / total_changes
        
        return trend_strength
    
    def _detect_fractal_highs(self, high: np.ndarray, periods: int = 2) -> np.ndarray:
        """Detect fractal highs"""
        fractal_highs = np.zeros_like(high)
        
        for i in range(periods, len(high) - periods):
            if all(high[i] >= high[i-j] for j in range(1, periods + 1)) and \
               all(high[i] >= high[i+j] for j in range(1, periods + 1)):
                fractal_highs[i] = 1
        
        return fractal_highs
    
    def _detect_fractal_lows(self, low: np.ndarray, periods: int = 2) -> np.ndarray:
        """Detect fractal lows"""
        fractal_lows = np.zeros_like(low)
        
        for i in range(periods, len(low) - periods):
            if all(low[i] <= low[i-j] for j in range(1, periods + 1)) and \
               all(low[i] <= low[i+j] for j in range(1, periods + 1)):
                fractal_lows[i] = 1
        
        return fractal_lows
    
    def _detect_trend_regime(self, close: np.ndarray, short: int = 20, long: int = 50) -> np.ndarray:
        """Detect trend regime"""
        if len(close) < long:
            return np.zeros_like(close)
        
        sma_short = np.convolve(close, np.ones(short), 'valid') / short
        sma_long = np.convolve(close, np.ones(long), 'valid') / long
        
        # Pad arrays to match original length
        sma_short = self._pad_array(sma_short, len(close))
        sma_long = self._pad_array(sma_long, len(close))
        
        regime = np.where(sma_short > sma_long, 1, -1)  # 1 = uptrend, -1 = downtrend
        return regime.astype(float)
    
    def _detect_volatility_regime(self, returns: np.ndarray, period: int = 20) -> np.ndarray:
        """Detect volatility regime"""
        if len(returns) < period:
            return np.zeros_like(returns)
        
        rolling_vol = np.array([
            np.std(returns[max(0, i-period):i]) if i >= period else 0
            for i in range(len(returns))
        ])
        
        vol_median = np.median(rolling_vol[rolling_vol > 0])
        regime = np.where(rolling_vol > vol_median, 1, 0)  # 1 = high vol, 0 = low vol
        
        return np.concatenate([[0], regime])  # Pad for returns array
    
    def _detect_volume_regime(self, volume: np.ndarray, period: int = 20) -> np.ndarray:
        """Detect volume regime"""
        if len(volume) < period:
            return np.zeros_like(volume)
        
        rolling_vol = np.convolve(volume, np.ones(period), 'valid') / period
        rolling_vol = self._pad_array(rolling_vol, len(volume))
        
        regime = np.where(volume > rolling_vol, 1, 0)  # 1 = high volume, 0 = low volume
        return regime.astype(float)
    
    def _calculate_hurst_exponent(self, returns: np.ndarray, max_lag: int = 20) -> np.ndarray:
        """Calculate Hurst exponent for trend persistence"""
        if len(returns) < max_lag * 2:
            return np.full_like(returns, 0.5)
        
        hurst_values = np.zeros_like(returns)
        
        for i in range(max_lag * 2, len(returns)):
            data_slice = returns[i-max_lag*2:i]
            if len(data_slice) > max_lag:
                try:
                    lags = range(2, min(max_lag, len(data_slice) // 2))
                    tau = []
                    rs_values = []
                    
                    for lag in lags:
                        subset = data_slice[:len(data_slice)//lag*lag]
                        reshaped = subset.reshape(-1, lag)
                        
                        mean_vals = np.mean(reshaped, axis=1)
                        std_vals = np.std(reshaped, axis=1)
                        
                        rs = np.mean(np.where(std_vals > 0, 
                                            np.ptp(np.cumsum(reshaped - mean_vals[:, np.newaxis], axis=1), axis=1) / std_vals,
                                            0))
                        
                        if rs > 0:
                            rs_values.append(rs)
                            tau.append(lag)
                    
                    if len(rs_values) > 1:
                        log_rs = np.log(rs_values)
                        log_tau = np.log(tau)
                        hurst_values[i] = np.polyfit(log_tau, log_rs, 1)[0]
                    else:
                        hurst_values[i] = 0.5
                        
                except Exception:
                    hurst_values[i] = 0.5
            else:
                hurst_values[i] = 0.5
        
        return hurst_values
    
    def _calculate_autocorrelation(self, returns: np.ndarray, lag: int = 1) -> np.ndarray:
        """Calculate autocorrelation"""
        if len(returns) <= lag:
            return np.zeros_like(returns)
        
        autocorr = np.zeros_like(returns)
        
        for i in range(lag, len(returns)):
            if i >= 20:  # Need sufficient data for correlation
                data_slice = returns[i-20:i]
                if len(data_slice) > lag:
                    x = data_slice[:-lag]
                    y = data_slice[lag:]
                    
                    if len(x) > 0 and len(y) > 0 and np.std(x) > 0 and np.std(y) > 0:
                        autocorr[i] = np.corrcoef(x, y)[0, 1]
        
        return autocorr
    
    def _calculate_regime_transition_prob(self, close: np.ndarray, period: int = 50) -> np.ndarray:
        """Calculate regime transition probability"""
        if len(close) < period:
            return np.zeros_like(close)
        
        regime = self._detect_trend_regime(close)
        transition_prob = np.zeros_like(close)
        
        for i in range(period, len(regime)):
            regime_slice = regime[i-period:i]
            transitions = np.sum(np.abs(np.diff(regime_slice)))
            transition_prob[i] = transitions / (period - 1)
        
        return transition_prob
    
    def _calculate_resistance_strength(self, high: np.ndarray, close: np.ndarray, 
                                     window: int = 20) -> np.ndarray:
        """Calculate resistance strength"""
        resistance = np.zeros_like(high)
        
        for i in range(window, len(high)):
            recent_highs = high[i-window:i]
            current_price = close[i]
            
            # Count how many times price approached but didn't break recent highs
            resistance_levels = recent_highs[recent_highs > current_price * 1.01]
            resistance[i] = len(resistance_levels) / window
        
        return resistance
    
    def _calculate_support_strength(self, low: np.ndarray, close: np.ndarray,
                                  window: int = 20) -> np.ndarray:
        """Calculate support strength"""
        support = np.zeros_like(low)
        
        for i in range(window, len(low)):
            recent_lows = low[i-window:i]
            current_price = close[i]
            
            # Count how many times price approached but didn't break recent lows
            support_levels = recent_lows[recent_lows < current_price * 0.99]
            support[i] = len(support_levels) / window
        
        return support
    
    def _pad_array(self, arr: np.ndarray, target_length: int) -> np.ndarray:
        """Pad array to target length"""
        if len(arr) >= target_length:
            return arr
        
        padding_length = target_length - len(arr)
        padding = np.full(padding_length, arr[0] if len(arr) > 0 else 0)
        
        return np.concatenate([padding, arr])


class RiskAnalyzer:
    """
    Risk and portfolio analysis adapted from molecular property analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RiskAnalyzer")
    
    def calculate_risk_metrics(self, data: TimeSeriesData, 
                             benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Calculate comprehensive risk metrics"""
        try:
            metrics = {}
            returns = data.log_returns
            
            if len(returns) == 0:
                return metrics
            
            # Basic risk metrics
            metrics.update(self._calculate_basic_risk_metrics(returns))
            
            # Drawdown analysis
            metrics.update(self._calculate_drawdown_metrics(data.close))
            
            # Value at Risk and Expected Shortfall
            metrics.update(self._calculate_var_metrics(returns))
            
            # Higher moment risk measures
            metrics.update(self._calculate_higher_moment_metrics(returns))
            
            # Tail risk measures
            metrics.update(self._calculate_tail_risk_metrics(returns))
            
            # If benchmark provided, calculate relative risk metrics
            if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                metrics.update(self._calculate_relative_risk_metrics(returns, benchmark_returns))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_basic_risk_metrics(self, returns: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate basic risk metrics"""
        metrics = {}
        
        if len(returns) == 0:
            return metrics
        
        # Rolling volatility (annualized)
        for window in [20, 60, 252]:
            if len(returns) >= window:
                rolling_vol = np.array([
                    np.std(returns[max(0, i-window):i]) * np.sqrt(252) if i >= window else np.nan
                    for i in range(len(returns))
                ])
                metrics[f'volatility_{window}d'] = np.concatenate([[np.nan], rolling_vol])
        
        # Sharpe ratio (assuming risk-free rate = 0)
        for window in [60, 252]:
            if len(returns) >= window:
                rolling_sharpe = np.array([
                    (np.mean(returns[max(0, i-window):i]) * 252) / 
                    (np.std(returns[max(0, i-window):i]) * np.sqrt(252)) 
                    if i >= window and np.std(returns[max(0, i-window):i]) > 0 else np.nan
                    for i in range(len(returns))
                ])
                metrics[f'sharpe_ratio_{window}d'] = np.concatenate([[np.nan], rolling_sharpe])
        
        # Sortino ratio (downside deviation)
        for window in [60, 252]:
            if len(returns) >= window:
                rolling_sortino = np.array([
                    self._calculate_sortino_ratio(returns[max(0, i-window):i]) 
                    if i >= window else np.nan
                    for i in range(len(returns))
                ])
                metrics[f'sortino_ratio_{window}d'] = np.concatenate([[np.nan], rolling_sortino])
        
        return metrics
    
    def _calculate_drawdown_metrics(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate drawdown metrics"""
        metrics = {}
        
        # Running maximum (peak)
        running_max = np.maximum.accumulate(prices)
        
        # Drawdown
        drawdown = (prices - running_max) / running_max
        metrics['drawdown'] = drawdown
        
        # Maximum drawdown
        max_dd = np.minimum.accumulate(drawdown)
        metrics['max_drawdown'] = max_dd
        
        # Drawdown duration
        dd_duration = np.zeros_like(prices)
        current_duration = 0
        
        for i in range(len(drawdown)):
            if drawdown[i] < 0:
                current_duration += 1
            else:
                current_duration = 0
            dd_duration[i] = current_duration
        
        metrics['drawdown_duration'] = dd_duration
        
        # Recovery factor (total return / max drawdown)
        total_return = (prices - prices[0]) / prices[0]
        recovery_factor = np.where(max_dd < 0, total_return / abs(max_dd), np.inf)
        metrics['recovery_factor'] = recovery_factor
        
        return metrics
    
    def _calculate_var_metrics(self, returns: np.ndarray, 
                             confidence_levels: List[float] = None) -> Dict[str, np.ndarray]:
        """Calculate Value at Risk and Expected Shortfall"""
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
        
        metrics = {}
        
        for window in [60, 252]:
            if len(returns) >= window:
                for confidence in confidence_levels:
                    var_values = np.array([
                        np.percentile(returns[max(0, i-window):i], (1-confidence)*100) 
                        if i >= window else np.nan
                        for i in range(len(returns))
                    ])
                    metrics[f'var_{int(confidence*100)}_{window}d'] = np.concatenate([[np.nan], var_values])
                    
                    # Expected Shortfall (Conditional VaR)
                    es_values = np.array([
                        np.mean(returns[max(0, i-window):i][
                            returns[max(0, i-window):i] <= np.percentile(returns[max(0, i-window):i], (1-confidence)*100)
                        ]) if i >= window else np.nan
                        for i in range(len(returns))
                    ])
                    metrics[f'expected_shortfall_{int(confidence*100)}_{window}d'] = np.concatenate([[np.nan], es_values])
        
        return metrics
    
    def _calculate_higher_moment_metrics(self, returns: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate skewness and kurtosis"""
        metrics = {}
        
        for window in [60, 252]:
            if len(returns) >= window:
                # Rolling skewness
                rolling_skew = np.array([
                    stats.skew(returns[max(0, i-window):i]) if i >= window and SCIPY_AVAILABLE else np.nan
                    for i in range(len(returns))
                ])
                metrics[f'skewness_{window}d'] = np.concatenate([[np.nan], rolling_skew])
                
                # Rolling kurtosis
                rolling_kurt = np.array([
                    stats.kurtosis(returns[max(0, i-window):i]) if i >= window and SCIPY_AVAILABLE else np.nan
                    for i in range(len(returns))
                ])
                metrics[f'kurtosis_{window}d'] = np.concatenate([[np.nan], rolling_kurt])
        
        return metrics
    
    def _calculate_tail_risk_metrics(self, returns: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate tail risk metrics"""
        metrics = {}
        
        # Tail ratio (95th percentile / 5th percentile)
        for window in [60, 252]:
            if len(returns) >= window:
                tail_ratio = np.array([
                    np.percentile(returns[max(0, i-window):i], 95) / 
                    abs(np.percentile(returns[max(0, i-window):i], 5))
                    if i >= window and np.percentile(returns[max(0, i-window):i], 5) != 0 else np.nan
                    for i in range(len(returns))
                ])
                metrics[f'tail_ratio_{window}d'] = np.concatenate([[np.nan], tail_ratio])
        
        # Extreme returns frequency
        threshold = 2 * np.std(returns) if len(returns) > 0 else 0
        extreme_returns = np.abs(returns) > threshold
        
        for window in [60, 252]:
            if len(returns) >= window:
                extreme_freq = np.array([
                    np.sum(extreme_returns[max(0, i-window):i]) / window if i >= window else np.nan
                    for i in range(len(returns))
                ])
                metrics[f'extreme_returns_freq_{window}d'] = np.concatenate([[np.nan], extreme_freq])
        
        return metrics
    
    def _calculate_relative_risk_metrics(self, returns: np.ndarray, 
                                       benchmark_returns: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate risk metrics relative to benchmark"""
        metrics = {}
        
        # Active returns
        active_returns = returns - benchmark_returns
        
        # Tracking error
        for window in [60, 252]:
            if len(active_returns) >= window:
                tracking_error = np.array([
                    np.std(active_returns[max(0, i-window):i]) * np.sqrt(252) if i >= window else np.nan
                    for i in range(len(active_returns))
                ])
                metrics[f'tracking_error_{window}d'] = tracking_error
        
        # Information ratio
        for window in [60, 252]:
            if len(active_returns) >= window:
                info_ratio = np.array([
                    (np.mean(active_returns[max(0, i-window):i]) * 252) / 
                    (np.std(active_returns[max(0, i-window):i]) * np.sqrt(252))
                    if i >= window and np.std(active_returns[max(0, i-window):i]) > 0 else np.nan
                    for i in range(len(active_returns))
                ])
                metrics[f'information_ratio_{window}d'] = info_ratio
        
        # Beta
        for window in [60, 252]:
            if len(returns) >= window:
                rolling_beta = np.array([
                    np.cov(returns[max(0, i-window):i], benchmark_returns[max(0, i-window):i])[0, 1] / 
                    np.var(benchmark_returns[max(0, i-window):i])
                    if i >= window and np.var(benchmark_returns[max(0, i-window):i]) > 0 else np.nan
                    for i in range(len(returns))
                ])
                metrics[f'beta_{window}d'] = rolling_beta
        
        return metrics
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return np.nan
        
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return np.inf
        
        downside_deviation = np.sqrt(np.mean(negative_returns ** 2))
        if downside_deviation == 0:
            return np.inf
        
        return (np.mean(returns) * 252) / (downside_deviation * np.sqrt(252))


class FinancialFeatureExtractorPlugin(FeatureExtractorPlugin):
    """
    Financial feature extractor plugin for quantitative analysis.
    
    Provides comprehensive feature extraction for financial time series data,
    adapted from molecular descriptor calculation patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the financial feature extractor plugin"""
        super().__init__(config)
        
        # Configuration
        self.enable_technical_analysis = self.config.get('technical_analysis', True)
        self.enable_risk_analysis = self.config.get('risk_analysis', True)
        self.lookback_periods = self.config.get('lookback_periods', [5, 10, 20, 50, 100, 200])
        self.cache_enabled = self.config.get('cache_enabled', True)
        
        # Initialize analyzers
        self.technical_analyzer = TechnicalIndicatorAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        
        # Statistics
        self.stats = {
            'features_extracted': 0,
            'time_series_processed': 0,
            'cache_hits': 0,
            'processing_time_total': 0.0
        }
        
        self.logger.info(f"💰 Financial Feature Extractor initialized")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata"""
        return {
            "name": "FinancialFeatureExtractorPlugin",
            "version": "1.0.0",
            "description": "Financial time series feature extraction for quantitative analysis",
            "supported_features": [
                FeatureType.NUMERICAL,
                FeatureType.CATEGORICAL,
                FeatureType.TIME_SERIES
            ],
            "capabilities": [
                "technical_indicators",
                "risk_metrics",
                "market_microstructure",
                "regime_detection",
                "pattern_recognition",
                "volatility_analysis"
            ],
            "dependencies": {
                "talib": TALIB_AVAILABLE,
                "yfinance": YFINANCE_AVAILABLE,
                "scipy": SCIPY_AVAILABLE
            }
        }
    
    def extract_features(self, data: Any, feature_types: Optional[List[FeatureType]] = None,
                        **kwargs) -> FeatureExtractionResult:
        """Extract financial features from time series data"""
        start_time = time.time()
        
        try:
            self.logger.info(f"💹 Extracting financial features")
            
            # Convert input data to TimeSeriesData
            if isinstance(data, dict):
                ts_data = self._convert_dict_to_timeseries(data)
            elif isinstance(data, pd.DataFrame):
                ts_data = self._convert_dataframe_to_timeseries(data)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            result = FinancialAnalysisResult()
            all_features = []
            feature_names = []
            
            # Technical analysis features
            if self.enable_technical_analysis:
                tech_features, tech_names = self._extract_technical_features(ts_data)
                all_features.extend(tech_features)
                feature_names.extend(tech_names)
                result.technical_features = np.array(tech_features).T if tech_features else np.array([])
            
            # Risk analysis features
            if self.enable_risk_analysis:
                risk_features, risk_names = self._extract_risk_features(ts_data)
                all_features.extend(risk_features)
                feature_names.extend(risk_names)
                result.risk_features = np.array(risk_features).T if risk_features else np.array([])
            
            # Market microstructure features
            micro_features, micro_names = self._extract_microstructure_features(ts_data)
            all_features.extend(micro_features)
            feature_names.extend(micro_names)
            result.microstructure_features = np.array(micro_features).T if micro_features else np.array([])
            
            # Fundamental analysis features (if available)
            fund_features, fund_names = self._extract_fundamental_features(ts_data, kwargs)
            all_features.extend(fund_features)
            feature_names.extend(fund_names)
            result.fundamental_features = np.array(fund_features).T if fund_features else np.array([])
            
            # Combine all features
            if all_features:
                combined_features = np.column_stack(all_features)
            else:
                combined_features = np.array([])
            
            # Update result
            result.feature_names = feature_names
            result.processing_time = time.time() - start_time
            result.valid_periods = len(ts_data.close)
            result.metadata = {
                'symbol': ts_data.symbol,
                'frequency': ts_data.frequency,
                'date_range': f"{ts_data.timestamp[0]} to {ts_data.timestamp[-1]}",
                'total_features': len(feature_names)
            }
            
            # Update statistics
            self.stats['features_extracted'] += len(feature_names)
            self.stats['time_series_processed'] += 1
            self.stats['processing_time_total'] += result.processing_time
            
            return FeatureExtractionResult(
                success=True,
                features=combined_features,
                feature_names=feature_names,
                feature_types=[FeatureType.NUMERICAL] * len(feature_names),
                extraction_time=result.processing_time,
                metadata=result.metadata
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting financial features: {e}")
            return FeatureExtractionResult(
                success=False,
                error_message=str(e),
                extraction_time=time.time() - start_time
            )
    
    def _convert_dict_to_timeseries(self, data: Dict[str, Any]) -> TimeSeriesData:
        """Convert dictionary to TimeSeriesData"""
        required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        return TimeSeriesData(
            timestamp=pd.to_datetime(data['timestamp']),
            open=np.array(data['open']),
            high=np.array(data['high']),
            low=np.array(data['low']),
            close=np.array(data['close']),
            volume=np.array(data['volume']),
            symbol=data.get('symbol', ''),
            frequency=data.get('frequency', '1D')
        )
    
    def _convert_dataframe_to_timeseries(self, df: pd.DataFrame) -> TimeSeriesData:
        """Convert DataFrame to TimeSeriesData"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return TimeSeriesData(
            timestamp=pd.to_datetime(df.index),
            open=df['open'].values,
            high=df['high'].values,
            low=df['low'].values,
            close=df['close'].values,
            volume=df['volume'].values,
            symbol=getattr(df, 'symbol', ''),
            frequency=getattr(df, 'frequency', '1D')
        )
    
    def _extract_technical_features(self, data: TimeSeriesData) -> Tuple[List[np.ndarray], List[str]]:
        """Extract technical analysis features"""
        try:
            indicators = self.technical_analyzer.calculate_indicators(data, self.lookback_periods)
            
            features = []
            names = []
            
            for name, values in indicators.items():
                if isinstance(values, np.ndarray) and len(values) == len(data.close):
                    # Handle NaN values
                    clean_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                    features.append(clean_values)
                    names.append(f"tech_{name}")
            
            return features, names
            
        except Exception as e:
            self.logger.error(f"Error extracting technical features: {e}")
            return [], []
    
    def _extract_risk_features(self, data: TimeSeriesData) -> Tuple[List[np.ndarray], List[str]]:
        """Extract risk analysis features"""
        try:
            risk_metrics = self.risk_analyzer.calculate_risk_metrics(data)
            
            features = []
            names = []
            
            for name, values in risk_metrics.items():
                if isinstance(values, np.ndarray) and len(values) == len(data.close):
                    # Handle NaN values
                    clean_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                    features.append(clean_values)
                    names.append(f"risk_{name}")
            
            return features, names
            
        except Exception as e:
            self.logger.error(f"Error extracting risk features: {e}")
            return [], []
    
    def _extract_microstructure_features(self, data: TimeSeriesData) -> Tuple[List[np.ndarray], List[str]]:
        """Extract market microstructure features"""
        try:
            features = []
            names = []
            
            # Spread proxies
            spread_proxy = (data.high - data.low) / data.close
            features.append(spread_proxy)
            names.append("spread_proxy")
            
            # Volume-weighted average price
            vwap = np.cumsum(data.typical_price * data.volume) / np.cumsum(data.volume)
            features.append(vwap)
            names.append("vwap")
            
            # Price impact measures
            volume_buckets = np.percentile(data.volume, [25, 50, 75])
            price_impact = np.zeros_like(data.close)
            
            for i in range(1, len(data.close)):
                volume_quartile = np.searchsorted(volume_buckets, data.volume[i])
                price_change = abs(data.close[i] - data.close[i-1]) / data.close[i-1]
                price_impact[i] = price_change / (volume_quartile + 1)
            
            features.append(price_impact)
            names.append("price_impact")
            
            # Order flow imbalance proxy
            ofi_proxy = (data.close - data.open) / (data.high - data.low + 1e-8)
            features.append(ofi_proxy)
            names.append("order_flow_imbalance")
            
            # Liquidity measures
            amihud_illiquidity = np.abs(data.returns) / (data.volume + 1e-8)
            amihud_illiquidity = np.concatenate([[0], amihud_illiquidity])
            features.append(amihud_illiquidity)
            names.append("amihud_illiquidity")
            
            # Volume synchronicity
            volume_sync = np.zeros_like(data.volume)
            for i in range(5, len(data.volume)):
                recent_volume = data.volume[i-5:i]
                volume_sync[i] = np.corrcoef(recent_volume, np.arange(len(recent_volume)))[0, 1]
            
            volume_sync = np.nan_to_num(volume_sync)
            features.append(volume_sync)
            names.append("volume_synchronicity")
            
            return features, names
            
        except Exception as e:
            self.logger.error(f"Error extracting microstructure features: {e}")
            return [], []
    
    def _extract_fundamental_features(self, data: TimeSeriesData, 
                                    kwargs: Dict[str, Any]) -> Tuple[List[np.ndarray], List[str]]:
        """Extract fundamental analysis features"""
        try:
            features = []
            names = []
            
            # If fundamental data is provided in kwargs
            fundamental_data = kwargs.get('fundamental_data', {})
            
            if fundamental_data:
                for metric_name, values in fundamental_data.items():
                    if isinstance(values, (list, np.ndarray)) and len(values) == len(data.close):
                        features.append(np.array(values))
                        names.append(f"fundamental_{metric_name}")
            
            # Basic price-based fundamental proxies
            if len(data.close) > 252:  # Need at least 1 year of data
                # Price-to-moving-average ratios (fundamental proxies)
                ma_252 = np.convolve(data.close, np.ones(252), 'valid') / 252
                ma_252_padded = np.concatenate([np.full(251, ma_252[0]), ma_252])
                
                price_to_ma = data.close / ma_252_padded
                features.append(price_to_ma)
                names.append("price_to_yearly_ma")
                
                # Relative price position in yearly range
                for i in range(252, len(data.close)):
                    yearly_high = np.max(data.high[i-252:i])
                    yearly_low = np.min(data.low[i-252:i])
                    
                    if yearly_high != yearly_low:
                        price_position = (data.close[i] - yearly_low) / (yearly_high - yearly_low)
                    else:
                        price_position = 0.5
                
                yearly_position = np.concatenate([np.full(252, 0.5), 
                                                [0.5] * (len(data.close) - 252)])
                
                # Recalculate properly
                yearly_position = np.zeros_like(data.close)
                for i in range(252, len(data.close)):
                    yearly_high = np.max(data.high[i-252:i])
                    yearly_low = np.min(data.low[i-252:i])
                    
                    if yearly_high != yearly_low:
                        yearly_position[i] = (data.close[i] - yearly_low) / (yearly_high - yearly_low)
                    else:
                        yearly_position[i] = 0.5
                
                features.append(yearly_position)
                names.append("yearly_price_position")
            
            return features, names
            
        except Exception as e:
            self.logger.error(f"Error extracting fundamental features: {e}")
            return [], []
    
    def get_feature_importance(self, features: np.ndarray, 
                             feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance based on variance and correlation"""
        try:
            importance = {}
            
            if len(features) == 0 or len(feature_names) == 0:
                return importance
            
            # Calculate variance-based importance
            feature_vars = np.var(features, axis=0)
            max_var = np.max(feature_vars) if len(feature_vars) > 0 else 1.0
            
            for i, name in enumerate(feature_names):
                if i < len(feature_vars):
                    # Normalize by maximum variance
                    var_importance = feature_vars[i] / max_var if max_var > 0 else 0
                    
                    # Add stability factor (lower is better for stability)
                    stability = 1.0 / (1.0 + np.std(features[:, i])) if features.shape[0] > i else 0
                    
                    # Combined importance
                    importance[name] = 0.7 * var_importance + 0.3 * stability
            
            return importance
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get plugin statistics"""
        stats = self.stats.copy()
        stats['average_processing_time'] = (
            stats['processing_time_total'] / stats['time_series_processed'] 
            if stats['time_series_processed'] > 0 else 0
        )
        return stats
    
    def health_check(self) -> bool:
        """Perform health check"""
        try:
            # Test basic functionality
            test_data = TimeSeriesData(
                timestamp=pd.date_range('2020-01-01', periods=100, freq='D'),
                open=np.random.randn(100) + 100,
                high=np.random.randn(100) + 102,
                low=np.random.randn(100) + 98,
                close=np.random.randn(100) + 100,
                volume=np.random.randint(1000, 10000, 100),
                symbol="TEST"
            )
            
            # Ensure high >= low and other constraints
            test_data.high = np.maximum(test_data.high, test_data.low + 1)
            test_data.high = np.maximum(test_data.high, np.maximum(test_data.open, test_data.close))
            test_data.low = np.minimum(test_data.low, np.minimum(test_data.open, test_data.close))
            
            # Test technical analysis
            indicators = self.technical_analyzer.calculate_indicators(test_data, [5, 10])
            
            # Test risk analysis
            risk_metrics = self.risk_analyzer.calculate_risk_metrics(test_data)
            
            return len(indicators) > 0 or len(risk_metrics) > 0
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False


# Plugin metadata for discovery
__plugin_metadata__ = {
    "name": "FinancialFeatureExtractorPlugin",
    "version": "1.0.0",
    "author": "Universal AI Core",
    "description": "Financial time series feature extraction for quantitative analysis",
    "plugin_type": "feature_extractor",
    "entry_point": f"{__name__}:FinancialFeatureExtractorPlugin",
    "dependencies": [
        {"name": "pandas", "optional": False},
        {"name": "numpy", "optional": False},
        {"name": "talib", "optional": True},
        {"name": "yfinance", "optional": True},
        {"name": "scipy", "optional": True}
    ],
    "capabilities": [
        "technical_indicators",
        "risk_metrics",
        "market_microstructure",
        "regime_detection",
        "pattern_recognition",
        "volatility_analysis"
    ],
    "hooks": []
}


if __name__ == "__main__":
    # Test the financial feature extractor
    print("💰 FINANCIAL FEATURE EXTRACTOR TEST")
    print("=" * 50)
    
    # Initialize plugin
    config = {
        'technical_analysis': True,
        'risk_analysis': True,
        'lookback_periods': [5, 10, 20, 50],
        'cache_enabled': True
    }
    
    financial_extractor = FinancialFeatureExtractorPlugin(config)
    
    # Generate test financial data
    np.random.seed(42)
    n_periods = 252  # One year of daily data
    
    # Simulate price data with realistic patterns
    base_price = 100
    returns = np.random.normal(0.0005, 0.02, n_periods)  # 0.05% daily return, 2% volatility
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])  # Remove initial price
    
    # Create OHLC data
    high_noise = np.random.exponential(0.01, n_periods)
    low_noise = np.random.exponential(0.01, n_periods)
    
    test_data = {
        'timestamp': pd.date_range('2023-01-01', periods=n_periods, freq='D'),
        'open': prices * (1 + np.random.normal(0, 0.005, n_periods)),
        'high': prices * (1 + high_noise),
        'low': prices * (1 - low_noise),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, n_periods),
        'symbol': 'TEST_STOCK',
        'frequency': '1D'
    }
    
    # Ensure OHLC constraints
    test_data['high'] = np.maximum.reduce([test_data['open'], test_data['high'], 
                                          test_data['low'], test_data['close']])
    test_data['low'] = np.minimum.reduce([test_data['open'], test_data['high'], 
                                         test_data['low'], test_data['close']])
    
    print(f"\n📊 Test data: {n_periods} periods")
    print(f"💵 Price range: ${test_data['close'].min():.2f} - ${test_data['close'].max():.2f}")
    print(f"📈 Total return: {((test_data['close'][-1] / test_data['close'][0]) - 1) * 100:.2f}%")
    
    # Extract features
    print(f"\n🔍 Extracting financial features...")
    result = financial_extractor.extract_features(test_data)
    
    if result.success:
        print(f"✅ Feature extraction successful!")
        print(f"📊 Features extracted: {len(result.feature_names)}")
        print(f"⏱️ Processing time: {result.extraction_time:.3f}s")
        print(f"📐 Feature matrix shape: {result.features.shape}")
        
        # Show sample features
        print(f"\n🎯 Sample features:")
        for i, name in enumerate(result.feature_names[:10]):
            if i < result.features.shape[1]:
                non_zero_count = np.count_nonzero(result.features[:, i])
                print(f"  {name}: {non_zero_count}/{len(result.features)} non-zero values")
        
        # Feature importance
        if len(result.features) > 0:
            importance = financial_extractor.get_feature_importance(
                result.features, result.feature_names
            )
            
            # Top 5 most important features
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n🏆 Top 5 important features:")
            for name, score in top_features:
                print(f"  {name}: {score:.3f}")
    else:
        print(f"❌ Feature extraction failed: {result.error_message}")
    
    # Test health check
    health = financial_extractor.health_check()
    print(f"\n🏥 Health check: {'✅' if health else '❌'}")
    
    # Show statistics
    stats = financial_extractor.get_statistics()
    print(f"\n📊 Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Financial feature extractor test completed!")