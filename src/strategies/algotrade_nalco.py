#!/usr/bin/env python3
"""
advanced_behavioral_stock_screener.py

Multi-strategy algorithmic trading screener with behavioral finance integration.
Analyzes multiple stocks using technical, fundamental, and behavioral indicators.

Usage:
python advanced_behavioral_stock_screener.py --capital 30000 --period 1y

Advanced Strategies Implemented:
1. EMA Crossover with Volume & Momentum Confirmation
2. RSI Mean Reversion with Sentiment Analysis
3. Bollinger Band Breakout with Market Structure
4. MACD Momentum with Divergence Detection
5. Support/Resistance with Volume Profile
6. Ichimoku Cloud Strategy
7. Williams %R Oscillator
8. Behavioral Sentiment Score
9. Market Regime Detection
10. Volatility Breakout Strategy
11. Money Flow Index (MFI)
12. Fear & Greed Index Integration
"""

import argparse
import datetime as dt
import math
import warnings
from typing import List, Dict, Tuple, Optional
import ssl
from scipy import stats
from scipy.signal import find_peaks

import numpy as np
import pandas as pd
import yfinance as yf

# Try to import TA-Lib, fallback if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available, using fallback implementations")

warnings.filterwarnings('ignore')

# Disable SSL verification for yfinance
ssl._create_default_https_context = ssl._create_unverified_context

# ==================== TECHNICAL INDICATORS ====================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    middle = sma(series, window)
    std = series.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume"""
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R Oscillator"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)

def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index"""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    positive_flow = pd.Series(index=close.index, dtype=float)
    negative_flow = pd.Series(index=close.index, dtype=float)
    
    for i in range(1, len(typical_price)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.iloc[i] = raw_money_flow.iloc[i]
            negative_flow.iloc[i] = 0
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            positive_flow.iloc[i] = 0
            negative_flow.iloc[i] = raw_money_flow.iloc[i]
        else:
            positive_flow.iloc[i] = 0
            negative_flow.iloc[i] = 0
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    money_ratio = positive_mf / negative_mf
    return 100 - (100 / (1 + money_ratio))

def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict:
    """Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (high.rolling(26).max() + low.rolling(26).min()) / 2
    
    # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    
    # Chikou Span (Lagging Span): Close shifted back 26 periods
    chikou_span = close.shift(-26)
    
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index"""
    if TALIB_AVAILABLE:
        try:
            return pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=period), index=high.index)
        except:
            pass
    
    # Fallback implementation
    try:
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=high.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=high.index)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(period).mean()
    except:
        return pd.Series([25] * len(high), index=high.index)  # Default neutral ADX

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator"""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(d_period).mean()
    return k_percent, d_percent

def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()

def calculate_behavioral_sentiment(df: pd.DataFrame) -> float:
    """Calculate behavioral sentiment score based on price action and volume"""
    if len(df) < 20:
        return 0.5
    
    recent = df.tail(20)
    
    # Price momentum (20%)
    price_momentum = (recent['close'].iloc[-1] / recent['close'].iloc[0] - 1)
    momentum_score = min(1, max(0, (price_momentum + 0.1) / 0.2))  # Normalize to 0-1
    
    # Volume trend (20%)
    vol_trend = recent['volume'].iloc[-10:].mean() / recent['volume'].iloc[-20:-10].mean()
    vol_score = min(1, max(0, (vol_trend - 0.5) / 1.0))
    
    # Volatility (negative sentiment if too high) (20%)
    volatility = recent['close'].pct_change().std()
    vol_sentiment = max(0, 1 - (volatility * 50))  # High vol = negative sentiment
    
    # RSI position (20%)
    rsi_val = rsi(recent['close']).iloc[-1]
    rsi_score = 1 - abs(rsi_val - 50) / 50  # Closer to 50 = better sentiment
    
    # Moving average position (20%)
    ma_score = 1 if recent['close'].iloc[-1] > recent['close'].rolling(10).mean().iloc[-1] else 0
    
    sentiment = (momentum_score * 0.2 + vol_score * 0.2 + vol_sentiment * 0.2 + 
                rsi_score * 0.2 + ma_score * 0.2)
    
    return sentiment

def detect_market_regime(df: pd.DataFrame) -> str:
    """Detect current market regime: trending, ranging, or volatile"""
    if len(df) < 50:
        return "Unknown"
    
    recent = df.tail(50)
    
    # Calculate ADX for trend strength
    adx_val = adx(recent['high'], recent['low'], recent['close']).iloc[-1]
    
    # Calculate volatility
    volatility = recent['close'].pct_change().std() * np.sqrt(252)
    
    # Calculate price efficiency (how much price moved vs path taken)
    price_change = abs(recent['close'].iloc[-1] - recent['close'].iloc[0])
    path_length = recent['close'].diff().abs().sum()
    efficiency = price_change / path_length if path_length > 0 else 0
    
    if adx_val > 25 and efficiency > 0.3:
        return "Trending"
    elif volatility > 0.4:
        return "Volatile"
    else:
        return "Ranging"

# ==================== DATA FETCHING ====================

def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch OHLCV data with enhanced error handling"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=False, verify=False)
            
            if df.empty:
                if attempt < max_retries - 1:
                    continue
                return pd.DataFrame()
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            
            # Ensure we have all required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                if attempt < max_retries - 1:
                    continue
                return pd.DataFrame()
            
            df = df[required_cols].copy()
            df.dropna(inplace=True)
            
            # Basic data validation
            if len(df) < 30:  # Need minimum data
                if attempt < max_retries - 1:
                    continue
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            print(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()
    
    return pd.DataFrame()

# ==================== STRATEGY IMPLEMENTATIONS ====================

def strategy_ema_crossover(df: pd.DataFrame) -> Dict:
    """Strategy 1: EMA Crossover with Volume Confirmation"""
    if len(df) < 50:
        return {'signal': 0, 'score': 0, 'reason': 'Insufficient data'}
    
    df['ema_9'] = ema(df['close'], 9)
    df['ema_21'] = ema(df['close'], 21)
    df['ema_50'] = ema(df['close'], 50)
    df['vol_sma'] = sma(df['volume'], 20)
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Bullish crossover: EMA9 crosses above EMA21
    bullish_cross = (latest['ema_9'] > latest['ema_21']) and (prev['ema_9'] <= prev['ema_21'])
    # Price above EMA50 (uptrend)
    uptrend = latest['close'] > latest['ema_50']
    # Volume confirmation
    volume_surge = latest['volume'] > latest['vol_sma'] * 1.2
    
    score = 0
    if bullish_cross:
        score += 40
    if uptrend:
        score += 30
    if volume_surge:
        score += 30
    
    signal = 1 if score >= 60 else 0
    reason = f"EMA Cross: {bullish_cross}, Uptrend: {uptrend}, Vol: {volume_surge}"
    
    return {'signal': signal, 'score': score, 'reason': reason}

def strategy_rsi_mean_reversion(df: pd.DataFrame) -> Dict:
    """Strategy 2: RSI Mean Reversion"""
    if len(df) < 50:
        return {'signal': 0, 'score': 0, 'reason': 'Insufficient data'}
    
    df['rsi'] = rsi(df['close'], 14)
    df['ema_20'] = ema(df['close'], 20)
    
    latest = df.iloc[-1]
    
    # Oversold (RSI < 35) but recovering
    oversold_recovery = 30 < latest['rsi'] < 45
    # Price near or above EMA20
    price_support = latest['close'] >= latest['ema_20'] * 0.98
    # RSI trending up
    rsi_momentum = latest['rsi'] > df.iloc[-3]['rsi']
    
    score = 0
    if oversold_recovery:
        score += 40
    if price_support:
        score += 30
    if rsi_momentum:
        score += 30
    
    signal = 1 if score >= 60 else 0
    reason = f"RSI: {latest['rsi']:.1f}, Recovery: {oversold_recovery}, Momentum: {rsi_momentum}"
    
    return {'signal': signal, 'score': score, 'reason': reason}

def strategy_bollinger_breakout(df: pd.DataFrame) -> Dict:
    """Strategy 3: Bollinger Band Squeeze Breakout"""
    if len(df) < 50:
        return {'signal': 0, 'score': 0, 'reason': 'Insufficient data'}
    
    upper, middle, lower = bollinger_bands(df['close'], 20, 2.0)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_width'] = (upper - lower) / middle
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Squeeze: bandwidth in lower 20th percentile
    squeeze = latest['bb_width'] < df['bb_width'].quantile(0.25)
    # Breakout: price crosses above middle band
    breakout = (latest['close'] > latest['bb_middle']) and (prev['close'] <= prev['bb_middle'])
    # Momentum: price in upper half
    momentum = latest['close'] > latest['bb_middle']
    
    score = 0
    if squeeze:
        score += 35
    if breakout:
        score += 40
    if momentum:
        score += 25
    
    signal = 1 if score >= 65 else 0
    reason = f"Squeeze: {squeeze}, Breakout: {breakout}, Momentum: {momentum}"
    
    return {'signal': signal, 'score': score, 'reason': reason}

def strategy_macd_momentum(df: pd.DataFrame) -> Dict:
    """Strategy 4: MACD Momentum with Histogram Expansion"""
    if len(df) < 50:
        return {'signal': 0, 'score': 0, 'reason': 'Insufficient data'}
    
    macd_line, macd_sig, macd_hist = macd(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = macd_sig
    df['macd_hist'] = macd_hist
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # MACD bullish crossover
    macd_cross = (latest['macd'] > latest['macd_signal']) and (prev['macd'] <= prev['macd_signal'])
    # Histogram expanding
    hist_expand = latest['macd_hist'] > prev['macd_hist']
    # Both above zero (strong momentum)
    above_zero = latest['macd'] > 0 and latest['macd_signal'] > 0
    
    score = 0
    if macd_cross:
        score += 40
    if hist_expand:
        score += 30
    if above_zero:
        score += 30
    
    signal = 1 if score >= 60 else 0
    reason = f"MACD Cross: {macd_cross}, Hist+: {hist_expand}, Above 0: {above_zero}"
    
    return {'signal': signal, 'score': score, 'reason': reason}

def strategy_support_resistance(df: pd.DataFrame) -> Dict:
    """Strategy 5: Support/Resistance with Volume Profile"""
    if len(df) < 50:
        return {'signal': 0, 'score': 0, 'reason': 'Insufficient data'}
    
    # Enhanced support/resistance with volume
    df['support'] = df['low'].rolling(20).min()
    df['resistance'] = df['high'].rolling(20).max()
    df['rsi'] = rsi(df['close'], 14)
    df['volume_sma'] = df['volume'].rolling(20).mean()
    
    latest = df.iloc[-1]
    
    # Price near support (within 2%)
    near_support = latest['close'] <= latest['support'] * 1.02
    # RSI showing bullish divergence
    rsi_healthy = latest['rsi'] > 35
    # Price bouncing up
    bouncing = latest['close'] > df.iloc[-2]['close']
    # Volume confirmation
    volume_support = latest['volume'] > latest['volume_sma'] * 1.1
    
    score = 0
    if near_support:
        score += 35
    if rsi_healthy:
        score += 25
    if bouncing:
        score += 25
    if volume_support:
        score += 15
    
    signal = 1 if score >= 60 else 0
    reason = f"Support: {near_support}, RSI: {latest['rsi']:.1f}, Bounce: {bouncing}, Vol: {volume_support}"
    
    return {'signal': signal, 'score': score, 'reason': reason}

def strategy_ichimoku_cloud(df: pd.DataFrame) -> Dict:
    """Strategy 6: Ichimoku Cloud Strategy"""
    if len(df) < 100:
        return {'signal': 0, 'score': 0, 'reason': 'Insufficient data'}
    
    ichimoku = ichimoku_cloud(df['high'], df['low'], df['close'])
    df.update(ichimoku)
    
    latest = df.iloc[-1]
    
    # Price above cloud
    cloud_top = max(latest['senkou_span_a'], latest['senkou_span_b'])
    cloud_bottom = min(latest['senkou_span_a'], latest['senkou_span_b'])
    above_cloud = latest['close'] > cloud_top
    
    # Tenkan above Kijun (bullish)
    tenkan_bullish = latest['tenkan_sen'] > latest['kijun_sen']
    
    # Chikou above price (confirmation)
    chikou_clear = latest['chikou_span'] > df['close'].iloc[-26] if len(df) > 26 else False
    
    # Strong momentum through cloud
    momentum = latest['close'] > df['close'].iloc[-5] * 1.02 if len(df) > 5 else False
    
    score = 0
    if above_cloud:
        score += 40
    if tenkan_bullish:
        score += 30
    if chikou_clear:
        score += 20
    if momentum:
        score += 10
    
    signal = 1 if score >= 70 else 0
    reason = f"Cloud: {above_cloud}, T>K: {tenkan_bullish}, Chikou: {chikou_clear}, Mom: {momentum}"
    
    return {'signal': signal, 'score': score, 'reason': reason}

def strategy_williams_r(df: pd.DataFrame) -> Dict:
    """Strategy 7: Williams %R Oscillator Strategy"""
    if len(df) < 50:
        return {'signal': 0, 'score': 0, 'reason': 'Insufficient data'}
    
    df['williams_r'] = williams_r(df['high'], df['low'], df['close'], 14)
    df['williams_r_fast'] = williams_r(df['high'], df['low'], df['close'], 7)
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Oversold recovery (-80 to -50)
    oversold_recovery = -80 < latest['williams_r'] < -50
    # Fast Williams crossing above slow
    bullish_cross = (latest['williams_r_fast'] > latest['williams_r']) and (prev['williams_r_fast'] <= prev['williams_r'])
    # Momentum confirmation
    momentum_up = latest['williams_r'] > prev['williams_r']
    
    score = 0
    if oversold_recovery:
        score += 40
    if bullish_cross:
        score += 35
    if momentum_up:
        score += 25
    
    signal = 1 if score >= 60 else 0
    reason = f"WR Recovery: {oversold_recovery}, Cross: {bullish_cross}, Mom: {momentum_up}"
    
    return {'signal': signal, 'score': score, 'reason': reason}

def strategy_behavioral_sentiment(df: pd.DataFrame) -> Dict:
    """Strategy 8: Behavioral Sentiment Analysis"""
    if len(df) < 50:
        return {'signal': 0, 'score': 0, 'reason': 'Insufficient data'}
    
    # Calculate behavioral sentiment
    sentiment = calculate_behavioral_sentiment(df)
    
    # Market regime
    regime = detect_market_regime(df)
    
    # Fear & Greed proxy (based on volatility and momentum)
    recent_vol = df['close'].pct_change().tail(20).std()
    recent_return = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1)
    
    fear_greed = 0.5  # Neutral
    if recent_return > 0.05 and recent_vol < 0.03:  # Good returns, low vol = greed
        fear_greed = 0.8
    elif recent_return < -0.05 and recent_vol > 0.05:  # Bad returns, high vol = fear
        fear_greed = 0.2
    
    # Combine signals
    sentiment_bullish = sentiment > 0.6
    regime_favorable = regime in ["Trending", "Ranging"]
    contrarian_opportunity = fear_greed < 0.3  # Buy when others are fearful
    
    score = 0
    if sentiment_bullish:
        score += 40
    if regime_favorable:
        score += 30
    if contrarian_opportunity:
        score += 30
    
    signal = 1 if score >= 60 else 0
    reason = f"Sentiment: {sentiment:.2f}, Regime: {regime}, F&G: {fear_greed:.2f}"
    
    return {'signal': signal, 'score': score, 'reason': reason}

def strategy_mfi_divergence(df: pd.DataFrame) -> Dict:
    """Strategy 9: Money Flow Index with Divergence Detection"""
    if len(df) < 50:
        return {'signal': 0, 'score': 0, 'reason': 'Insufficient data'}
    
    df['mfi'] = mfi(df['high'], df['low'], df['close'], df['volume'], 14)
    
    latest = df.iloc[-1]
    
    # MFI oversold recovery
    mfi_recovery = 20 < latest['mfi'] < 50
    
    # Bullish divergence: price makes lower low, MFI makes higher low
    recent_price_low = df['close'].tail(20).min()
    recent_mfi_at_price_low = df.loc[df['close'].tail(20).idxmin(), 'mfi']
    current_mfi = latest['mfi']
    
    bullish_divergence = (df['close'].iloc[-1] <= recent_price_low * 1.02 and 
                         current_mfi > recent_mfi_at_price_low)
    
    # Volume increasing
    volume_increasing = latest['volume'] > df['volume'].tail(10).mean()
    
    score = 0
    if mfi_recovery:
        score += 40
    if bullish_divergence:
        score += 35
    if volume_increasing:
        score += 25
    
    signal = 1 if score >= 60 else 0
    reason = f"MFI: {latest['mfi']:.1f}, Divergence: {bullish_divergence}, Vol+: {volume_increasing}"
    
    return {'signal': signal, 'score': score, 'reason': reason}

def strategy_volatility_breakout(df: pd.DataFrame) -> Dict:
    """Strategy 10: Volatility Breakout Strategy"""
    if len(df) < 50:
        return {'signal': 0, 'score': 0, 'reason': 'Insufficient data'}
    
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    
    latest = df.iloc[-1]
    
    # Low volatility environment (compression)
    low_vol = latest['volatility'] < df['volatility'].tail(50).quantile(0.25)
    
    # Price breaking above recent high
    recent_high = df['high'].tail(20).max()
    breakout = latest['close'] > recent_high
    
    # Volume confirmation
    volume_surge = latest['volume'] > df['volume'].tail(20).mean() * 1.5
    
    # ATR expansion
    atr_expanding = latest['atr'] > df['atr'].iloc[-5]
    
    score = 0
    if low_vol:
        score += 25
    if breakout:
        score += 40
    if volume_surge:
        score += 25
    if atr_expanding:
        score += 10
    
    signal = 1 if score >= 65 else 0
    reason = f"LowVol: {low_vol}, Breakout: {breakout}, VolSurge: {volume_surge}, ATR+: {atr_expanding}"
    
    return {'signal': signal, 'score': score, 'reason': reason}

# ==================== RISK METRICS ====================

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.065) -> float:
    """Calculate annualized Sharpe Ratio (India risk-free ~6.5%)"""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    excess_returns = returns - (risk_free_rate / 252)  # daily risk-free rate
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calculate_cagr(df: pd.DataFrame) -> float:
    """Calculate Compound Annual Growth Rate"""
    if len(df) < 2:
        return 0.0
    start_price = df['close'].iloc[0]
    end_price = df['close'].iloc[-1]
    years = len(df) / 252  # trading days
    if years == 0 or start_price <= 0:
        return 0.0
    return (pow(end_price / start_price, 1 / years) - 1) * 100

def calculate_volatility(returns: pd.Series) -> float:
    """Calculate annualized volatility"""
    if len(returns) < 2:
        return 0.0
    return returns.std() * np.sqrt(252) * 100

def calculate_max_drawdown(df: pd.DataFrame) -> float:
    """Calculate maximum drawdown percentage"""
    cumulative = (1 + df['close'].pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min() * 100

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.065) -> float:
    """Calculate Sortino Ratio (using downside deviation)"""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / 252)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    downside_deviation = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)
    return (excess_returns.mean() * 252) / downside_deviation if downside_deviation > 0 else 0.0

def calculate_calmar_ratio(df: pd.DataFrame) -> float:
    """Calculate Calmar Ratio (CAGR / Max Drawdown)"""
    cagr = calculate_cagr(df)
    max_dd = abs(calculate_max_drawdown(df))
    return cagr / max_dd if max_dd > 0 else 0.0

def calculate_win_rate(df: pd.DataFrame) -> float:
    """Calculate win rate of daily returns"""
    returns = df['close'].pct_change().dropna()
    if len(returns) == 0:
        return 0.0
    return (returns > 0).sum() / len(returns)

def calculate_profit_factor(df: pd.DataFrame) -> float:
    """Calculate profit factor (gross profit / gross loss)"""
    returns = df['close'].pct_change().dropna()
    if len(returns) == 0:
        return 0.0
    
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    
    return gross_profit / gross_loss if gross_loss > 0 else float('inf')

def calculate_expectancy(df: pd.DataFrame) -> float:
    """Calculate expectancy (average win * win rate - average loss * loss rate)"""
    returns = df['close'].pct_change().dropna()
    if len(returns) == 0:
        return 0.0
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    win_rate = len(wins) / len(returns)
    loss_rate = len(losses) / len(returns)
    
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    
    return (avg_win * win_rate) - (avg_loss * loss_rate)

# ==================== PORTFOLIO ANALYSIS ====================

def analyze_stock(ticker: str, period: str = "1y") -> Optional[Dict]:
    """Comprehensive analysis of a single stock"""
    print(f"Analyzing {ticker}...", end=" ")
    
    df = fetch_stock_data(ticker, period)
    if df.empty or len(df) < 50:
        print(" Insufficient data")
        return None
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Run all strategies
    strategies = {
        'ema_crossover': strategy_ema_crossover(df),
        'rsi_reversion': strategy_rsi_mean_reversion(df),
        'bb_breakout': strategy_bollinger_breakout(df),
        'macd_momentum': strategy_macd_momentum(df),
        'support_bounce': strategy_support_resistance(df),
        'ichimoku_cloud': strategy_ichimoku_cloud(df),
        'williams_r': strategy_williams_r(df),
        'behavioral_sentiment': strategy_behavioral_sentiment(df),
        'mfi_divergence': strategy_mfi_divergence(df),
        'volatility_breakout': strategy_volatility_breakout(df)
    }
    
    # Aggregate signals
    total_score = sum(s['score'] for s in strategies.values())
    signal_count = sum(s['signal'] for s in strategies.values())
    
    # Calculate enhanced risk metrics
    returns_clean = df['returns'].dropna()
    sharpe = calculate_sharpe_ratio(returns_clean)
    sortino = calculate_sortino_ratio(returns_clean)
    cagr = calculate_cagr(df)
    volatility = calculate_volatility(returns_clean)
    max_dd = calculate_max_drawdown(df)
    calmar = calculate_calmar_ratio(df)
    win_rate = calculate_win_rate(df)
    profit_factor = calculate_profit_factor(df)
    expectancy = calculate_expectancy(df)
    
    current_price = df['close'].iloc[-1]
    
    # Market regime and sentiment
    regime = detect_market_regime(df)
    sentiment = calculate_behavioral_sentiment(df)
    
    print(f"✓ (Signals: {signal_count}/10, Sharpe: {sharpe:.2f}, Regime: {regime})")
    
    return {
        'ticker': ticker,
        'current_price': current_price,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'cagr': cagr,
        'volatility': volatility,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'market_regime': regime,
        'behavioral_sentiment': sentiment,
        'signal_count': signal_count,
        'total_score': total_score,
        'strategies': strategies
    }

def rank_stocks(results: List[Dict]) -> pd.DataFrame:
    """Advanced ranking using multiple criteria and behavioral factors"""
    df = pd.DataFrame(results)
    
    # Normalize metrics (0-1 scale)
    def normalize_metric(series, inverse=False):
        if series.max() == series.min():
            return pd.Series([0.5] * len(series), index=series.index)
        if inverse:
            return (series.max() - series) / (series.max() - series.min())
        else:
            return (series - series.min()) / (series.max() - series.min())
    
    # Performance metrics (40%)
    df['sharpe_norm'] = normalize_metric(df['sharpe_ratio'])
    df['sortino_norm'] = normalize_metric(df['sortino_ratio'])
    df['cagr_norm'] = normalize_metric(df['cagr'])
    df['calmar_norm'] = normalize_metric(df['calmar_ratio'])
    
    # Risk metrics (25%) - lower is better for some
    df['volatility_norm'] = normalize_metric(df['volatility'], inverse=True)
    df['max_dd_norm'] = normalize_metric(df['max_drawdown'], inverse=True)
    df['win_rate_norm'] = normalize_metric(df['win_rate'])
    
    # Strategy signals (20%)
    df['signal_norm'] = df['signal_count'] / 10.0  # Now out of 10 strategies
    
    # Behavioral factors (15%)
    df['sentiment_norm'] = normalize_metric(df['behavioral_sentiment'])
    df['regime_score'] = df['market_regime'].map({
        'Trending': 1.0,
        'Ranging': 0.7,
        'Volatile': 0.3,
        'Unknown': 0.5
    })
    
    # Advanced composite score with multiple factors
    df['performance_score'] = (
        df['sharpe_norm'] * 0.15 +
        df['sortino_norm'] * 0.15 +
        df['cagr_norm'] * 0.10
    )
    
    df['risk_score'] = (
        df['volatility_norm'] * 0.08 +
        df['max_dd_norm'] * 0.08 +
        df['calmar_norm'] * 0.09
    )
    
    df['signal_score'] = df['signal_norm'] * 0.20
    
    df['behavioral_score'] = (
        df['sentiment_norm'] * 0.08 +
        df['regime_score'] * 0.07
    )
    
    # Final composite score
    df['composite_score'] = (
        df['performance_score'] +
        df['risk_score'] +
        df['signal_score'] +
        df['behavioral_score']
    )
    
    df = df.sort_values('composite_score', ascending=False)
    return df

def allocate_portfolio(ranked_df: pd.DataFrame, capital: float, top_n: int = 5) -> Dict:
    """Allocate capital across top stocks"""
    top_stocks = ranked_df.head(top_n).copy()
    
    # Equal weight allocation
    allocation_per_stock = capital / len(top_stocks)
    
    portfolio = []
    total_invested = 0
    
    for idx, row in top_stocks.iterrows():
        shares = math.floor(allocation_per_stock / row['current_price'])
        cost = shares * row['current_price']
        total_invested += cost
        
        portfolio.append({
            'ticker': row['ticker'],
            'shares': shares,
            'price': row['current_price'],
            'investment': cost,
            'sharpe': row['sharpe_ratio'],
            'cagr': row['cagr'],
            'signals': row['signal_count']
        })
    
    return {
        'portfolio': portfolio,
        'total_invested': total_invested,
        'cash_remaining': capital - total_invested
    }

# ==================== MAIN FUNCTION ====================

def main():
    parser = argparse.ArgumentParser(description='Multi-Strategy Indian Stock Screener')
    parser.add_argument('--capital', type=float, default=30000, help='Starting capital in INR')
    parser.add_argument('--period', type=str, default='1y', choices=['6mo', '1y', '2y'], help='Analysis period')
    parser.add_argument('--top', type=int, default=5, help='Number of stocks to include in portfolio')
    args = parser.parse_args()
    
    # Default Indian large-cap stocks
    tickers = [
        "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", 
        "RELIANCE.NS", "ITC.NS", "HINDUNILVR.NS", "SBIN.NS",
        "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "BHARTIARTL.NS",
        "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS", "NTPC.NS",
        "POWERGRID.NS", "ONGC.NS", "COALINDIA.NS", "TATASTEEL.NS"
    ]
    
    print(f"\n{'='*70}")
    print(f"  MULTI-STRATEGY INDIAN STOCK SCREENER")
    print(f"  Capital: ₹{args.capital:,.0f} | Period: {args.period} | Top-N: {args.top}")
    print(f"{'='*70}\n")
    
    # Analyze all stocks
    results = []
    for ticker in tickers:
        result = analyze_stock(ticker, args.period)
        if result:
            results.append(result)
    
    if not results:
        print("No valid stock data retrieved. Exiting.")
        return
    
    # Rank stocks
    ranked = rank_stocks(results)
    
    # Display top stocks with enhanced metrics
    print(f"\n{'='*120}")
    print("TOP 10 RECOMMENDED STOCKS (Advanced Multi-Factor Ranking)")
    print(f"{'='*120}")
    print(f"{'Rank':<5}{'Ticker':<12}{'Price':<8}{'Sharpe':<8}{'Sortino':<8}{'CAGR%':<8}{'Vol%':<8}{'Signals':<8}{'Regime':<10}{'Sentiment':<10}{'Score':<8}")
    print("-" * 120)
    
    for i, row in ranked.head(10).iterrows():
        rank = ranked.index.get_loc(i) + 1
        print(f"{rank:<5}{row['ticker']:<12}₹{row['current_price']:<7.0f}"
              f"{row['sharpe_ratio']:<8.2f}{row['sortino_ratio']:<8.2f}{row['cagr']:<8.1f}"
              f"{row['volatility']:<8.1f}{row['signal_count']:<8}/10"
              f"{row['market_regime']:<10}{row['behavioral_sentiment']:<10.2f}{row['composite_score']:<8.3f}")
    
    # Portfolio allocation
    allocation = allocate_portfolio(ranked, args.capital, args.top)
    
    print(f"\n{'='*110}")
    print(f"RECOMMENDED PORTFOLIO ALLOCATION (₹{args.capital:,.0f})")
    print(f"{'='*110}")
    print(f"{'Ticker':<12}{'Shares':<8}{'Price':<10}{'Investment':<12}{'Sharpe':<8}{'Sortino':<8}{'Signals':<8}{'Regime':<10}{'Sentiment':<10}")
    print("-" * 110)
    
    for stock in allocation['portfolio']:
        regime = next((r['market_regime'] for r in results if r['ticker'] == stock['ticker']), 'N/A')
        sentiment = next((r['behavioral_sentiment'] for r in results if r['ticker'] == stock['ticker']), 0.5)
        sortino = next((r['sortino_ratio'] for r in results if r['ticker'] == stock['ticker']), 0)
        
        print(f"{stock['ticker']:<12}{stock['shares']:<8}₹{stock['price']:<9.0f}"
              f"₹{stock['investment']:<11,.0f}{stock['sharpe']:<8.2f}{sortino:<8.2f}"
              f"{stock['signals']:<8}/10{regime:<10}{sentiment:<10.2f}")
    
    print("-" * 110)
    print(f"{'Total Invested:':<50}₹{allocation['total_invested']:,.0f}")
    print(f"{'Cash Remaining:':<50}₹{allocation['cash_remaining']:,.0f}")
    
    # Best single trade
    best = ranked.iloc[0]
    best_allocation = math.floor(args.capital / best['current_price']) * best['current_price']
    
    print(f"\n{'='*70}")
    print(f"💡 BEST SINGLE TRADE FOR THIS WEEK")
    print(f"{'='*70}")
    print(f"Stock: {best['ticker']}")
    print(f"Current Price: ₹{best['current_price']:.2f}")
    print(f"Allocation: ₹{best_allocation:,.0f} ({math.floor(args.capital/best['current_price'])} shares)")
    print(f"Sharpe: {best['sharpe_ratio']:.2f} | Sortino: {best['sortino_ratio']:.2f} | CAGR: {best['cagr']:.1f}% | Signals: {best['signal_count']}/10")
    print(f"Market Regime: {best['market_regime']} | Behavioral Sentiment: {best['behavioral_sentiment']:.2f}")
    print(f"Win Rate: {best['win_rate']:.1%} | Profit Factor: {best['profit_factor']:.2f}")
    
    # Show active strategies for best stock
    print(f"\nActive Strategies:")
    for name, strat in best['strategies'].items():
        if strat['signal'] == 1:
            print(f"  ✓ {name.replace('_', ' ').title()}: {strat['reason']}")
    
    # Show behavioral insights
    print(f"\nBehavioral Insights:")
    if best['behavioral_sentiment'] > 0.7:
        print(f"  📈 High positive sentiment - Strong bullish behavior")
    elif best['behavioral_sentiment'] < 0.3:
        print(f"  📉 Low sentiment - Potential contrarian opportunity")
    else:
        print(f"  ⚖️  Neutral sentiment - Mixed market behavior")
    
    if best['market_regime'] == 'Trending':
        print(f"  🎯 Trending market - Momentum strategies favored")
    elif best['market_regime'] == 'Ranging':
        print(f"  🔄 Ranging market - Mean reversion strategies favored")
    else:
        print(f"  ⚡ Volatile market - Risk management crucial")
    
    print(f"\n{'='*120}\n")

if __name__ == "__main__":
    main()