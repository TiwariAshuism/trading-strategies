#!/usr/bin/env python3
"""
portfolio_analyzer.py

Analyzes your existing stock portfolio and provides actionable recommendations:
- Current P&L and performance metrics
- Risk assessment and diversification score
- Hold/Sell/Add recommendations based on technical and fundamental analysis
- Future price targets and expected returns

Usage:
1. Edit the PORTFOLIO list below with your holdings
2. Run: python portfolio_analyzer.py
3. Or use CSV: python portfolio_analyzer.py --csv my_portfolio.csv

CSV Format:
ticker,total_shares,avg_price,holding_months
HDFCBANK.NS,50,1450.00,6
INFY.NS,100,1380.50,12
"""

import argparse
import datetime as dt
import warnings
from typing import List, Dict, Tuple, Optional
import json
import logging
from scipy import stats
from scipy.cluster.vq import kmeans2
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== YOUR PORTFOLIO (Edit this) ====================
PORTFOLIO = [
    {'ticker': 'TATAMOTORS.NS', 'shares': 46, 'avg_price': 711.24, 'months': 12},
    {'ticker': 'ITC.NS', 'shares': 15, 'avg_price': 412.00, 'months': 12},
    {'ticker': 'TATACHEM.NS', 'shares': 5, 'avg_price': 1095.00, 'months': 12},
    {'ticker': 'ACMESOLAR.NS', 'shares': 4, 'avg_price': 243.00, 'months': 12},
    {'ticker': 'CONTAINERCORP.NS', 'shares': 1, 'avg_price': 888.00, 'months': 12},
]


# ==================== TECHNICAL INDICATORS ====================

def validate_series(series: pd.Series, min_length: int = 10) -> bool:
    """Validate if series has enough valid data"""
    if len(series) < min_length or series.isna().all():
        return False
    return True

def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average with validation"""
    if not validate_series(series, span):
        return pd.Series(dtype=float)
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average with validation"""
    if not validate_series(series, window):
        return pd.Series(dtype=float)
    return series.rolling(window=window).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with validation"""
    if not validate_series(series, period * 2):
        return pd.Series(dtype=float)
    
    try:
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(window=period, min_periods=period).mean()
        ma_down = down.rolling(window=period, min_periods=period).mean()
        rs = ma_up / ma_down
        return 100 - (100 / (1 + rs))
    except Exception as e:
        logger.warning(f"RSI calculation failed: {e}")
        return pd.Series(dtype=float)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD with validation"""
    if not validate_series(series, slow * 2):
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    
    try:
        ema_fast = ema(series, fast)
        ema_slow = ema(series, slow)
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - macd_signal
        return macd_line, macd_signal, macd_hist
    except Exception as e:
        logger.warning(f"MACD calculation failed: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """Bollinger Bands with validation"""
    if not validate_series(series, window):
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    
    try:
        middle = sma(series, window)
        std = series.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower
    except Exception as e:
        logger.warning(f"Bollinger Bands calculation failed: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index (ADX) for trend strength"""
    if not all(validate_series(s, period * 2) for s in [high, low, close]):
        return pd.Series(dtype=float)
    
    try:
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        dm_pos = high.diff()
        dm_neg = -low.diff()
        dm_pos[dm_pos < 0] = 0
        dm_neg[dm_neg < 0] = 0
        
        # Smoothed values
        tr_smooth = tr.rolling(window=period).mean()
        dm_pos_smooth = dm_pos.rolling(window=period).mean()
        dm_neg_smooth = dm_neg.rolling(window=period).mean()
        
        # Directional Indicators
        di_pos = 100 * (dm_pos_smooth / tr_smooth)
        di_neg = 100 * (dm_neg_smooth / tr_smooth)
        
        # ADX
        dx = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg)
        adx = dx.rolling(window=period).mean()
        
        return adx
    except Exception as e:
        logger.warning(f"ADX calculation failed: {e}")
        return pd.Series(dtype=float)

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator"""
    if not all(validate_series(s, k_period) for s in [high, low, close]):
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    try:
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    except Exception as e:
        logger.warning(f"Stochastic calculation failed: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float)

# ==================== ADVANCED RISK METRICS ====================

def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Value at Risk (VaR) using historical method"""
    if len(returns) < 10 or returns.isna().all():
        return 0.0
    
    try:
        clean_returns = returns.dropna()
        if len(clean_returns) < 10:
            return 0.0
        return np.percentile(clean_returns, confidence_level * 100)
    except Exception as e:
        logger.warning(f"VaR calculation failed: {e}")
        return 0.0

def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (CVaR)"""
    if len(returns) < 10 or returns.isna().all():
        return 0.0
    
    try:
        clean_returns = returns.dropna()
        if len(clean_returns) < 10:
            return 0.0
        var = calculate_var(clean_returns, confidence_level)
        return clean_returns[clean_returns <= var].mean()
    except Exception as e:
        logger.warning(f"CVaR calculation failed: {e}")
        return 0.0

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.065) -> float:
    """Calculate Sortino Ratio (using downside deviation)"""
    if len(returns) < 10 or returns.isna().all():
        return 0.0
    
    try:
        clean_returns = returns.dropna()
        if len(clean_returns) < 10:
            return 0.0
        
        excess_returns = clean_returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_deviation = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)
        return (excess_returns.mean() * 252) / downside_deviation if downside_deviation > 0 else 0.0
    except Exception as e:
        logger.warning(f"Sortino ratio calculation failed: {e}")
        return 0.0

def monte_carlo_simulation(current_price: float, returns: pd.Series, days: int = 90, simulations: int = 1000) -> Dict:
    """Monte Carlo price simulation"""
    if len(returns) < 30 or returns.isna().all():
        return {'mean': current_price, 'std': 0, 'percentiles': {}}
    
    try:
        clean_returns = returns.dropna()
        if len(clean_returns) < 30:
            return {'mean': current_price, 'std': 0, 'percentiles': {}}
        
        mean_return = clean_returns.mean()
        std_return = clean_returns.std()
        
        # Generate random returns
        random_returns = np.random.normal(mean_return, std_return, (simulations, days))
        
        # Calculate price paths
        price_paths = current_price * np.exp(np.cumsum(random_returns, axis=1))
        final_prices = price_paths[:, -1]
        
        percentiles = {
            '5%': np.percentile(final_prices, 5),
            '25%': np.percentile(final_prices, 25),
            '50%': np.percentile(final_prices, 50),
            '75%': np.percentile(final_prices, 75),
            '95%': np.percentile(final_prices, 95)
        }
        
        return {
            'mean': float(np.mean(final_prices)),
            'std': float(np.std(final_prices)),
            'percentiles': percentiles
        }
    except Exception as e:
        logger.warning(f"Monte Carlo simulation failed: {e}")
        return {'mean': current_price, 'std': 0, 'percentiles': {}}

def calculate_correlation_matrix(portfolio_returns: Dict[str, pd.Series]) -> pd.DataFrame:
    """Calculate correlation matrix between holdings"""
    try:
        if len(portfolio_returns) < 2:
            return pd.DataFrame()
        
        # Align all return series by date
        df = pd.DataFrame(portfolio_returns)
        df = df.dropna()
        
        if df.empty or len(df) < 10:
            return pd.DataFrame()
        
        return df.corr()
    except Exception as e:
        logger.warning(f"Correlation matrix calculation failed: {e}")
        return pd.DataFrame()

def calculate_diversification_score(correlation_matrix: pd.DataFrame, weights: List[float]) -> float:
    """Advanced diversification score using correlation and clustering"""
    if correlation_matrix.empty or len(weights) < 2:
        return 0.0
    
    try:
        n_assets = len(weights)
        weights_array = np.array(weights)
        
        # Portfolio variance from correlation matrix
        cov_matrix = correlation_matrix.values  # Assuming correlation as proxy for covariance
        portfolio_var = np.dot(weights_array, np.dot(cov_matrix, weights_array))
        
        # Individual asset variance (assuming 1 for correlation matrix)
        individual_vars = np.sum(weights_array ** 2)
        
        # Diversification ratio
        div_ratio = 1 - (portfolio_var / individual_vars) if individual_vars > 0 else 0
        
        # Penalty for concentration
        concentration_penalty = 1 - np.max(weights_array)
        
        # Number of assets bonus
        n_assets_bonus = min(1.0, n_assets / 10)
        
        # Final score (0-100)
        score = (div_ratio * 0.5 + concentration_penalty * 0.3 + n_assets_bonus * 0.2) * 100
        return max(0, min(100, score))
    except Exception as e:
        logger.warning(f"Diversification score calculation failed: {e}")
        return 0.0

# ==================== DATA FETCHING ====================

def validate_portfolio_input(holding: Dict) -> bool:
    """Validate portfolio input data"""
    required_fields = ['ticker', 'shares', 'avg_price', 'months']
    
    for field in required_fields:
        if field not in holding:
            logger.error(f"Missing required field: {field}")
            return False
    
    if holding['shares'] <= 0:
        logger.error(f"Invalid shares for {holding['ticker']}: {holding['shares']}")
        return False
    
    if holding['avg_price'] <= 0:
        logger.error(f"Invalid avg_price for {holding['ticker']}: {holding['avg_price']}")
        return False
    
    if holding['months'] < 0:
        logger.error(f"Invalid months for {holding['ticker']}: {holding['months']}")
        return False
    
    return True

def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch historical data with enhanced error handling"""
    max_retries = 3
    retry_count = 0
    
    # Disable SSL verification for yfinance
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    while retry_count < max_retries:
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True, verify=False)
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            
            # Validate required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns for {ticker}: {missing_cols}")
                return pd.DataFrame()
            
            # Remove rows with all NaN values
            df = df.dropna(how='all')
            
            # Forward fill missing values (common in stock data)
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            if len(df) < 30:  # Need minimum data for analysis
                logger.warning(f"Insufficient data for {ticker}: {len(df)} days")
                return pd.DataFrame()
            
            logger.info(f"Successfully fetched {len(df)} days of data for {ticker}")
            return df
            
        except Exception as e:
            retry_count += 1
            logger.warning(f"Attempt {retry_count} failed for {ticker}: {e}")
            if retry_count >= max_retries:
                logger.error(f"Failed to fetch data for {ticker} after {max_retries} attempts")
                return pd.DataFrame()
    
    return pd.DataFrame()

def get_current_price(ticker: str) -> float:
    """Get latest price with enhanced error handling"""
    max_retries = 3
    retry_count = 0
    
    # Disable SSL verification for yfinance
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    while retry_count < max_retries:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="5d", verify=False)  # Get more data for reliability
            if not data.empty:
                # Use the most recent valid price
                valid_prices = data['Close'].dropna()
                if not valid_prices.empty:
                    price = float(valid_prices.iloc[-1])
                    logger.info(f"Current price for {ticker}: ₹{price:.2f}")
                    return price
            
            logger.warning(f"No price data for {ticker}")
            return 0.0
            
        except Exception as e:
            retry_count += 1
            logger.warning(f"Attempt {retry_count} failed to get price for {ticker}: {e}")
            if retry_count >= max_retries:
                logger.error(f"Failed to get price for {ticker} after {max_retries} attempts")
                return 0.0
    
    return 0.0

# ==================== ANALYSIS FUNCTIONS ====================

def calculate_returns_metrics(df: pd.DataFrame, holding_months: int) -> Dict:
    """Calculate comprehensive historical returns and risk metrics"""
    if df.empty or len(df) < 20:
        return {}
    
    try:
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        clean_returns = df['returns'].dropna()
        
        if len(clean_returns) < 10:
            return {}
        
        # Basic metrics
        annual_return = (df['close'].iloc[-1] / df['close'].iloc[0]) ** (252 / len(df)) - 1
        annual_vol = clean_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        risk_free_rate = 0.065  # 6.5% risk-free rate
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        sortino = calculate_sortino_ratio(clean_returns, risk_free_rate)
        
        # Drawdown analysis
        cumulative = (1 + clean_returns.fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Average drawdown duration
        drawdown_periods = (drawdown < -0.05).astype(int)  # 5% threshold
        dd_duration = 0
        if drawdown_periods.sum() > 0:
            dd_groups = (drawdown_periods != drawdown_periods.shift()).cumsum()
            dd_lengths = drawdown_periods.groupby(dd_groups).sum()
            dd_duration = dd_lengths[dd_lengths > 0].mean() if len(dd_lengths[dd_lengths > 0]) > 0 else 0
        
        # Risk metrics
        var_5 = calculate_var(clean_returns, 0.05) * 100
        cvar_5 = calculate_cvar(clean_returns, 0.05) * 100
        
        # Skewness and Kurtosis
        skewness = clean_returns.skew()
        kurtosis = clean_returns.kurtosis()
        
        # Beta calculation (using market proxy - could be improved with actual market index)
        market_returns = clean_returns  # Simplified - in reality, use market index
        beta = 1.0  # Default beta
        try:
            if len(market_returns) > 20:
                covariance = np.cov(clean_returns[1:], market_returns[:-1])[0, 1]
                market_variance = np.var(market_returns[:-1])
                beta = covariance / market_variance if market_variance > 0 else 1.0
        except:
            beta = 1.0
        
        # Information ratio (simplified)
        info_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Calmar ratio (annual return / max drawdown)
        calmar = abs(annual_return / max_dd) if max_dd < 0 else 0
        
        return {
            'annual_return': annual_return * 100,
            'annual_volatility': annual_vol * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd * 100,
            'avg_drawdown_duration': dd_duration,
            'var_5_percent': var_5,
            'cvar_5_percent': cvar_5,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'beta': beta,
            'information_ratio': info_ratio,
            'calmar_ratio': calmar,
            'total_return': ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
        }
    except Exception as e:
        logger.error(f"Error calculating returns metrics: {e}")
        return {}

def technical_analysis(df: pd.DataFrame) -> Dict:
    """Comprehensive technical analysis with advanced indicators"""
    if df.empty or len(df) < 50:
        return {}
    
    try:
        df = df.copy()
        
        # Moving averages
        df['ema_20'] = ema(df['close'], 20)
        df['ema_50'] = ema(df['close'], 50)
        df['ema_200'] = ema(df['close'], 200)
        df['sma_20'] = sma(df['close'], 20)
        
        # Oscillators
        df['rsi'] = rsi(df['close'], 14)
        df['rsi_9'] = rsi(df['close'], 9)  # Faster RSI
        
        # MACD
        macd_line, macd_sig, macd_hist = macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = macd_sig
        df['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_mid, bb_lower = bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_mid'] = bb_mid
        df['bb_lower'] = bb_lower
        
        # ADX for trend strength
        df['adx'] = adx(df['high'], df['low'], df['close'])
        
        # Stochastic
        stoch_k, stoch_d = stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # Volume indicators
        df['volume_sma'] = sma(df['volume'], 20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Enhanced trend analysis
        trend_signals = []
        trend_score = 0
        
        # Price vs moving averages
        if not pd.isna(latest['ema_20']) and latest['close'] > latest['ema_20']:
            trend_score += 1
            trend_signals.append("Above EMA20")
        if not pd.isna(latest['ema_50']) and latest['close'] > latest['ema_50']:
            trend_score += 1
            trend_signals.append("Above EMA50")
        if not pd.isna(latest['ema_200']) and latest['close'] > latest['ema_200']:
            trend_score += 1
            trend_signals.append("Above EMA200")
        
        # Moving average alignment
        if (not pd.isna(latest['ema_20']) and not pd.isna(latest['ema_50']) and 
            latest['ema_20'] > latest['ema_50']):
            trend_score += 1
            trend_signals.append("EMA20 > EMA50")
        if (not pd.isna(latest['ema_50']) and not pd.isna(latest['ema_200']) and 
            latest['ema_50'] > latest['ema_200']):
            trend_score += 1
            trend_signals.append("EMA50 > EMA200")
        
        trend_strength = (trend_score / 5.0) * 100
        
        # ADX trend strength
        adx_strength = latest['adx'] if not pd.isna(latest['adx']) else 0
        if adx_strength > 25:
            trend_signals.append(f"Strong trend (ADX: {adx_strength:.1f})")
        elif adx_strength > 20:
            trend_signals.append(f"Moderate trend (ADX: {adx_strength:.1f})")
        else:
            trend_signals.append(f"Weak trend (ADX: {adx_strength:.1f})")
        
        # Enhanced momentum analysis
        rsi_val = latest['rsi'] if not pd.isna(latest['rsi']) else 50
        rsi_9_val = latest['rsi_9'] if not pd.isna(latest['rsi_9']) else 50
        stoch_k_val = latest['stoch_k'] if not pd.isna(latest['stoch_k']) else 50
        
        momentum_signals = []
        if rsi_val > 80:
            momentum = "Extremely Overbought"
            momentum_signals.append(f"RSI very high ({rsi_val:.1f})")
        elif rsi_val > 70:
            momentum = "Overbought"
            momentum_signals.append(f"RSI overbought ({rsi_val:.1f})")
        elif rsi_val < 20:
            momentum = "Extremely Oversold"
            momentum_signals.append(f"RSI very low ({rsi_val:.1f})")
        elif rsi_val < 30:
            momentum = "Oversold"
            momentum_signals.append(f"RSI oversold ({rsi_val:.1f})")
        elif rsi_val > 50:
            momentum = "Bullish"
            momentum_signals.append(f"RSI bullish ({rsi_val:.1f})")
        else:
            momentum = "Bearish"
            momentum_signals.append(f"RSI bearish ({rsi_val:.1f})")
        
        # Stochastic confirmation
        if stoch_k_val > 80:
            momentum_signals.append("Stochastic overbought")
        elif stoch_k_val < 20:
            momentum_signals.append("Stochastic oversold")
        
        # MACD analysis
        macd_bullish = (not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']) and 
                       latest['macd'] > latest['macd_signal'])
        macd_signal = "Bullish" if macd_bullish else "Bearish"
        
        # MACD histogram trend
        macd_hist_trend = "Strengthening" if (not pd.isna(latest['macd_histogram']) and 
                                            not pd.isna(prev['macd_histogram']) and
                                            latest['macd_histogram'] > prev['macd_histogram']) else "Weakening"
        
        # Support/Resistance with multiple timeframes
        support_20 = df['close'].rolling(20).min().iloc[-1]
        support_50 = df['close'].rolling(50).min().iloc[-1]
        resistance_20 = df['close'].rolling(20).max().iloc[-1]
        resistance_50 = df['close'].rolling(50).max().iloc[-1]
        
        # Volume analysis
        volume_trend = "High" if latest['volume_ratio'] > 1.5 else "Normal" if latest['volume_ratio'] > 0.8 else "Low"
        
        # Bollinger Bands position
        bb_position = 50  # Default
        if (not pd.isna(latest['bb_upper']) and not pd.isna(latest['bb_lower']) and 
            latest['bb_upper'] != latest['bb_lower']):
            bb_position = ((latest['close'] - latest['bb_lower']) / 
                          (latest['bb_upper'] - latest['bb_lower'])) * 100
        
        # Price momentum (rate of change)
        price_momentum_5d = ((latest['close'] / df['close'].iloc[-6]) - 1) * 100 if len(df) > 5 else 0
        price_momentum_20d = ((latest['close'] / df['close'].iloc[-21]) - 1) * 100 if len(df) > 20 else 0
        
        return {
            'trend_strength': trend_strength,
            'trend_signals': trend_signals,
            'adx': adx_strength,
            'rsi': rsi_val,
            'rsi_9': rsi_9_val,
            'stochastic_k': stoch_k_val,
            'momentum': momentum,
            'momentum_signals': momentum_signals,
            'macd_signal': macd_signal,
            'macd_histogram_trend': macd_hist_trend,
            'support_20d': support_20,
            'support_50d': support_50,
            'resistance_20d': resistance_20,
            'resistance_50d': resistance_50,
            'volume_trend': volume_trend,
            'volume_ratio': latest['volume_ratio'] if not pd.isna(latest['volume_ratio']) else 1.0,
            'bb_position': bb_position,
            'price_vs_ema20': ((latest['close'] / latest['ema_20']) - 1) * 100 if not pd.isna(latest['ema_20']) else 0,
            'price_vs_ema50': ((latest['close'] / latest['ema_50']) - 1) * 100 if not pd.isna(latest['ema_50']) else 0,
            'price_vs_ema200': ((latest['close'] / latest['ema_200']) - 1) * 100 if not pd.isna(latest['ema_200']) else 0,
            'price_momentum_5d': price_momentum_5d,
            'price_momentum_20d': price_momentum_20d
        }
    except Exception as e:
        logger.error(f"Error in technical analysis: {e}")
        return {}

def predict_target_price(df: pd.DataFrame, current_price: float, months_ahead: int = 3) -> Dict:
    """Advanced target price prediction using multiple methods and Monte Carlo"""
    if df.empty or len(df) < 50:
        return {}
    
    try:
        df = df.copy()
        df['returns'] = df['close'].pct_change()
        days_ahead = months_ahead * 30
        
        # Method 1: Linear regression on recent trend (multiple timeframes)
        targets_linear = {}
        for period in [30, 60, 90]:
            if len(df) >= period:
                recent = df.tail(period).copy()
                recent['index'] = range(len(recent))
                x = recent['index'].values
                y = recent['close'].values
                if len(x) > 10:
                    z = np.polyfit(x, y, 1)
                    slope = z[0]
                    targets_linear[f'{period}d'] = current_price + (slope * days_ahead)
        
        linear_target = np.mean(list(targets_linear.values())) if targets_linear else current_price
        
        # Method 2: Exponential smoothing
        alpha = 0.3
        exp_smooth = df['close'].ewm(alpha=alpha).mean().iloc[-1]
        trend = df['close'].ewm(alpha=alpha).mean().diff().ewm(alpha=alpha).mean().iloc[-1]
        exp_target = exp_smooth + (trend * days_ahead)
        
        # Method 3: Moving average convergence
        ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
        ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
        ema_200 = df['close'].ewm(span=200).mean().iloc[-1]
        ma_target = (ema_20 * 0.5 + ema_50 * 0.3 + ema_200 * 0.2)
        
        # Method 4: Bollinger bands projection
        bb_upper, bb_mid, bb_lower = bollinger_bands(df['close'])
        bb_target = bb_mid.iloc[-1] if not bb_mid.empty and not pd.isna(bb_mid.iloc[-1]) else current_price
        
        # Method 5: Support/Resistance analysis
        resistance_levels = []
        support_levels = []
        for window in [20, 50, 100]:
            if len(df) >= window:
                resistance_levels.append(df['close'].rolling(window).max().iloc[-1])
                support_levels.append(df['close'].rolling(window).min().iloc[-1])
        
        resistance_target = np.mean(resistance_levels) if resistance_levels else current_price
        support_level = np.mean(support_levels) if support_levels else current_price
        
        # Method 6: Monte Carlo simulation
        mc_result = monte_carlo_simulation(current_price, df['returns'], days_ahead)
        mc_target = mc_result.get('mean', current_price)
        mc_percentiles = mc_result.get('percentiles', {})
        
        # Method 7: Mean reversion model
        long_term_mean = df['close'].mean()
        current_deviation = (current_price - long_term_mean) / long_term_mean
        reversion_factor = 0.1  # 10% reversion towards mean
        mean_reversion_target = current_price - (current_deviation * long_term_mean * reversion_factor)
        
        # Weighted average of all methods
        methods = [
            ('linear', linear_target, 0.20),
            ('exponential', exp_target, 0.15),
            ('moving_average', ma_target, 0.15),
            ('bollinger', bb_target, 0.10),
            ('resistance', resistance_target, 0.10),
            ('monte_carlo', mc_target, 0.20),
            ('mean_reversion', mean_reversion_target, 0.10)
        ]
        
        # Filter out invalid targets and calculate weighted average
        valid_methods = [(name, target, weight) for name, target, weight in methods 
                        if not pd.isna(target) and target > 0]
        
        if valid_methods:
            total_weight = sum(weight for _, _, weight in valid_methods)
            avg_target = sum(target * weight for _, target, weight in valid_methods) / total_weight
        else:
            avg_target = current_price
        
        # Calculate confidence intervals
        all_targets = [target for _, target, _ in valid_methods]
        target_std = np.std(all_targets) if len(all_targets) > 1 else 0
        
        # Risk-adjusted targets
        conservative_target = avg_target - target_std
        aggressive_target = avg_target + target_std
        
        # Expected returns
        expected_return = ((avg_target / current_price) - 1) * 100
        conservative_return = ((conservative_target / current_price) - 1) * 100
        aggressive_return = ((aggressive_target / current_price) - 1) * 100
        
        # Probability analysis from Monte Carlo
        prob_analysis = {}
        if mc_percentiles:
            prob_analysis = {
                'prob_positive': sum(1 for p in mc_percentiles.values() if p > current_price) / len(mc_percentiles) * 100,
                'prob_10_percent_gain': sum(1 for p in mc_percentiles.values() if p > current_price * 1.1) / len(mc_percentiles) * 100,
                'prob_20_percent_gain': sum(1 for p in mc_percentiles.values() if p > current_price * 1.2) / len(mc_percentiles) * 100
            }
        
        return {
            'linear_target': linear_target,
            'exponential_target': exp_target,
            'ma_target': ma_target,
            'bb_target': bb_target,
            'resistance_target': resistance_target,
            'monte_carlo_target': mc_target,
            'mean_reversion_target': mean_reversion_target,
            'avg_target': avg_target,
            'conservative_target': conservative_target,
            'aggressive_target': aggressive_target,
            'expected_return': expected_return,
            'conservative_return': conservative_return,
            'aggressive_return': aggressive_return,
            'target_std': target_std,
            'support_level': support_level,
            'monte_carlo_percentiles': mc_percentiles,
            'probability_analysis': prob_analysis,
            'prediction_confidence': min(100, max(20, 100 - (target_std / avg_target * 100))) if avg_target > 0 else 50
        }
    except Exception as e:
        logger.error(f"Error in price prediction: {e}")
        return {'avg_target': current_price, 'expected_return': 0}

def recommendation_engine(stock_data: Dict, holding_months: int) -> Dict:
    """Advanced recommendation engine with multiple factors and risk assessment"""
    try:
        current_return = stock_data.get('current_return', 0)
        tech = stock_data.get('technical', {})
        target = stock_data.get('target', {})
        historical = stock_data.get('historical', {})
        
        score = 0
        reasons = []
        risk_factors = []
        opportunity_factors = []
        
        # Factor 1: Current Performance Analysis (Weight: 25%)
        if current_return > 30:
            score += 3
            opportunity_factors.append(f"Exceptional gains ({current_return:.1f}%)")
        elif current_return > 20:
            score += 2
            opportunity_factors.append(f"Strong gains ({current_return:.1f}%)")
        elif current_return > 10:
            score += 1
            opportunity_factors.append(f"Moderate gains ({current_return:.1f}%)")
        elif current_return > 0:
            score += 0.5
            reasons.append(f"Small gains ({current_return:.1f}%)")
        elif current_return > -5:
            score -= 0.5
            reasons.append(f"Minor losses ({current_return:.1f}%)")
        elif current_return > -15:
            score -= 1.5
            risk_factors.append(f"Moderate losses ({current_return:.1f}%)")
        else:
            score -= 3
            risk_factors.append(f"Heavy losses ({current_return:.1f}%)")
        
        # Factor 2: Technical Trend Analysis (Weight: 20%)
        trend_strength = tech.get('trend_strength', 50)
        adx = tech.get('adx', 0)
        
        if trend_strength > 80 and adx > 25:
            score += 3
            opportunity_factors.append(f"Very strong uptrend (Strength: {trend_strength:.0f}%, ADX: {adx:.1f})")
        elif trend_strength > 70:
            score += 2
            opportunity_factors.append(f"Strong uptrend ({trend_strength:.0f}%)")
        elif trend_strength > 60:
            score += 1
            opportunity_factors.append(f"Moderate uptrend ({trend_strength:.0f}%)")
        elif trend_strength < 30:
            score -= 2
            risk_factors.append(f"Weak trend ({trend_strength:.0f}%)")
        elif trend_strength < 40:
            score -= 1
            risk_factors.append(f"Downtrend ({trend_strength:.0f}%)")
        
        # Factor 3: Momentum and Oscillators (Weight: 15%)
        rsi = tech.get('rsi', 50)
        stoch_k = tech.get('stochastic_k', 50)
        
        # RSI analysis
        if rsi > 85:
            score -= 2
            risk_factors.append(f"Severely overbought (RSI: {rsi:.1f})")
        elif rsi > 75:
            score -= 1
            risk_factors.append(f"Overbought (RSI: {rsi:.1f})")
        elif rsi < 15:
            score += 2
            opportunity_factors.append(f"Severely oversold - bounce likely (RSI: {rsi:.1f})")
        elif rsi < 25:
            score += 1.5
            opportunity_factors.append(f"Oversold (RSI: {rsi:.1f})")
        elif 40 <= rsi <= 60:
            score += 0.5
            reasons.append(f"Neutral momentum (RSI: {rsi:.1f})")
        
        # Stochastic confirmation
        if stoch_k > 80 and rsi > 70:
            score -= 0.5
            risk_factors.append("Multiple overbought signals")
        elif stoch_k < 20 and rsi < 30:
            score += 0.5
            opportunity_factors.append("Multiple oversold signals")
        
        # Factor 4: Price Prediction and Expected Returns (Weight: 20%)
        expected_return = target.get('expected_return', 0)
        prediction_confidence = target.get('prediction_confidence', 50)
        conservative_return = target.get('conservative_return', 0)
        
        if expected_return > 25 and prediction_confidence > 70:
            score += 3
            opportunity_factors.append(f"High upside potential ({expected_return:.1f}%, confidence: {prediction_confidence:.0f}%)")
        elif expected_return > 15:
            score += 2
            opportunity_factors.append(f"Good upside ({expected_return:.1f}%)")
        elif expected_return > 8:
            score += 1
            opportunity_factors.append(f"Moderate upside ({expected_return:.1f}%)")
        elif expected_return > 0:
            score += 0.5
            reasons.append(f"Limited upside ({expected_return:.1f}%)")
        elif expected_return > -8:
            score -= 1
            risk_factors.append(f"Potential downside ({expected_return:.1f}%)")
        else:
            score -= 2.5
            risk_factors.append(f"High downside risk ({expected_return:.1f}%)")
        
        # Conservative scenario check
        if conservative_return < -10:
            score -= 1
            risk_factors.append(f"Conservative scenario shows significant risk ({conservative_return:.1f}%)")
        
        # Factor 5: Risk Metrics (Weight: 10%)
        sharpe_ratio = historical.get('sharpe_ratio', 0)
        sortino_ratio = historical.get('sortino_ratio', 0)
        max_drawdown = historical.get('max_drawdown', 0)
        var_5 = historical.get('var_5_percent', 0)
        
        if sharpe_ratio > 1.5:
            score += 1.5
            opportunity_factors.append(f"Excellent risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")
        elif sharpe_ratio > 1.0:
            score += 1
            opportunity_factors.append(f"Good risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")
        elif sharpe_ratio < 0:
            score -= 1
            risk_factors.append(f"Poor risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")
        
        if max_drawdown < -30:
            score -= 1.5
            risk_factors.append(f"High historical drawdown ({max_drawdown:.1f}%)")
        elif max_drawdown < -20:
            score -= 0.5
            risk_factors.append(f"Moderate drawdown risk ({max_drawdown:.1f}%)")
        
        # Factor 6: Volume and Market Structure (Weight: 5%)
        volume_trend = tech.get('volume_trend', 'Normal')
        volume_ratio = tech.get('volume_ratio', 1.0)
        
        if volume_trend == 'High' and volume_ratio > 2.0:
            if trend_strength > 60:
                score += 1
                opportunity_factors.append(f"Strong volume support ({volume_ratio:.1f}x avg)")
            else:
                score -= 0.5
                risk_factors.append("High volume on weakness")
        
        # Factor 7: Holding Period Considerations (Weight: 5%)
        if holding_months < 1:
            reasons.append("Very new position")
        elif holding_months < 6:
            reasons.append("Short-term holding")
            if current_return < -10:
                score -= 0.5  # Penalty for quick losses
        elif holding_months > 24:
            reasons.append("Long-term holding")
            if current_return > 20:
                score += 0.5  # Reward for long-term gains
            elif current_return < 0:
                score -= 0.5  # Question long-term underperformance
        
        # Factor 8: MACD and Price Momentum
        macd_signal = tech.get('macd_signal', 'Neutral')
        macd_hist_trend = tech.get('macd_histogram_trend', 'Neutral')
        price_momentum_20d = tech.get('price_momentum_20d', 0)
        
        if macd_signal == "Bullish" and macd_hist_trend == "Strengthening":
            score += 1.5
            opportunity_factors.append("Strong MACD bullish momentum")
        elif macd_signal == "Bullish":
            score += 1
            opportunity_factors.append("MACD bullish signal")
        elif macd_signal == "Bearish" and macd_hist_trend == "Strengthening":
            score -= 1.5
            risk_factors.append("Strong MACD bearish momentum")
        
        if price_momentum_20d > 10:
            score += 0.5
            opportunity_factors.append(f"Strong 20-day momentum (+{price_momentum_20d:.1f}%)")
        elif price_momentum_20d < -10:
            score -= 0.5
            risk_factors.append(f"Weak 20-day momentum ({price_momentum_20d:.1f}%)")
        
        # Generate recommendation based on total score
        if score >= 6:
            action = "🟢 STRONG BUY"
            confidence = "Very High"
            action_detail = "Add to position aggressively"
        elif score >= 4:
            action = "🟢 BUY/ADD"
            confidence = "High"
            action_detail = "Consider adding to position"
        elif score >= 2:
            action = "🟢 HOLD/LIGHT ADD"
            confidence = "Medium-High"
            action_detail = "Hold current position, consider small additions"
        elif score >= 0:
            action = "🟡 HOLD"
            confidence = "Medium"
            action_detail = "Maintain current position"
        elif score >= -2:
            action = "🟠 HOLD/WATCH"
            confidence = "Medium-Low"
            action_detail = "Hold but monitor closely"
        elif score >= -4:
            action = "� REDUCE/TRIM"
            confidence = "Low"
            action_detail = "Consider reducing position size"
        else:
            action = "�🔴 SELL/EXIT"
            confidence = "Very Low"
            action_detail = "Consider exiting position"
        
        # Risk level assessment
        risk_score = len(risk_factors) - len(opportunity_factors)
        if risk_score > 2:
            risk_level = "High"
        elif risk_score > 0:
            risk_level = "Medium"
        elif risk_score < -2:
            risk_level = "Low"
        else:
            risk_level = "Medium"
        
        return {
            'action': action,
            'action_detail': action_detail,
            'confidence': confidence,
            'risk_level': risk_level,
            'score': round(score, 1),
            'reasons': reasons,
            'opportunity_factors': opportunity_factors,
            'risk_factors': risk_factors,
            'recommendation_summary': f"{action} - {confidence} confidence, {risk_level} risk"
        }
    except Exception as e:
        logger.error(f"Error in recommendation engine: {e}")
        return {
            'action': '🟡 HOLD',
            'confidence': 'Low',
            'score': 0,
            'reasons': ['Analysis incomplete due to data issues']
        }

# ==================== PORTFOLIO ANALYSIS ====================

def analyze_holding(holding: Dict) -> Dict:
    """Analyze a single holding with comprehensive validation"""
    if not validate_portfolio_input(holding):
        return None
    
    ticker = holding['ticker']
    shares = holding['shares']
    avg_price = holding['avg_price']
    months = holding['months']
    
    print(f"\nAnalyzing {ticker}...", end=" ")
    
    try:
        # Fetch data
        df = fetch_stock_data(ticker, period="2y")
        current_price = get_current_price(ticker)
        
        if df.empty or current_price == 0:
            print("❌ Failed to fetch data")
            logger.error(f"No data available for {ticker}")
            return None
        
        # Calculate basic metrics
        invested = shares * avg_price
        current_value = shares * current_price
        pnl = current_value - invested
        pnl_pct = (pnl / invested) * 100
        
        # Annualized return with validation
        if months > 0:
            annualized_return = ((current_price / avg_price) ** (12 / months) - 1) * 100
        else:
            annualized_return = 0
            logger.warning(f"Invalid holding period for {ticker}: {months} months")
        
        # Historical metrics
        hist_metrics = calculate_returns_metrics(df, months)
        
        # Technical analysis
        tech = technical_analysis(df)
        
        # Target price prediction
        target = predict_target_price(df, current_price, months_ahead=3)
        
        # Compile stock data
        stock_data = {
            'ticker': ticker,
            'shares': shares,
            'avg_price': avg_price,
            'current_price': current_price,
            'invested': invested,
            'current_value': current_value,
            'pnl': pnl,
            'current_return': pnl_pct,
            'annualized_return': annualized_return,
            'holding_months': months,
            'historical': hist_metrics,
            'technical': tech,
            'target': target,
            'returns_series': df['close'].pct_change().dropna() if 'close' in df.columns else pd.Series()
        }
        
        # Generate recommendation
        recommendation = recommendation_engine(stock_data, months)
        stock_data['recommendation'] = recommendation
        
        print(f"✓ (Return: {pnl_pct:+.1f}%, Action: {recommendation['action']})")
        logger.info(f"Successfully analyzed {ticker}: {pnl_pct:+.1f}% return")
        
        return stock_data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error analyzing {ticker}: {e}")
        return None

def portfolio_summary(results: List[Dict]) -> Dict:
    """Calculate comprehensive portfolio-level metrics"""
    if not results:
        return {}
    
    try:
        # Basic portfolio metrics
        total_invested = sum(r['invested'] for r in results)
        total_value = sum(r['current_value'] for r in results)
        total_pnl = total_value - total_invested
        total_return = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
        
        # Position weights
        weights = [r['current_value'] / total_value for r in results] if total_value > 0 else [1/len(results)] * len(results)
        
        # Weighted performance metrics
        weighted_annual_return = sum(r.get('annualized_return', 0) * w for r, w in zip(results, weights))
        
        # Portfolio risk metrics
        returns = [r['current_return'] for r in results]
        portfolio_volatility = np.std(returns) if len(returns) > 1 else 0
        
        # Collect return series for correlation analysis
        portfolio_returns = {}
        for r in results:
            if 'returns_series' in r and not r['returns_series'].empty:
                portfolio_returns[r['ticker']] = r['returns_series']
        
        # Correlation analysis
        correlation_matrix = calculate_correlation_matrix(portfolio_returns)
        avg_correlation = 0
        if not correlation_matrix.empty:
            # Get average correlation (excluding diagonal)
            corr_values = correlation_matrix.values
            mask = ~np.eye(corr_values.shape[0], dtype=bool)
            avg_correlation = np.mean(corr_values[mask]) if mask.any() else 0
        
        # Advanced diversification score
        diversification_score = calculate_diversification_score(correlation_matrix, weights)
        
        # Concentration metrics
        n_stocks = len(results)
        max_weight = max(weights) * 100 if weights else 0
        top_3_concentration = sum(sorted(weights, reverse=True)[:min(3, len(weights))]) * 100
        
        # Risk-adjusted returns
        portfolio_sharpe = 0
        portfolio_sortino = 0
        portfolio_var = 0
        portfolio_cvar = 0
        
        if portfolio_returns:
            # Create portfolio return series (weighted average)
            aligned_returns = pd.DataFrame(portfolio_returns).fillna(0)
            if not aligned_returns.empty:
                portfolio_return_series = aligned_returns.dot(weights[:len(aligned_returns.columns)])
                
                if len(portfolio_return_series) > 10:
                    annual_ret = portfolio_return_series.mean() * 252
                    annual_vol = portfolio_return_series.std() * np.sqrt(252)
                    portfolio_sharpe = (annual_ret - 0.065) / annual_vol if annual_vol > 0 else 0
                    portfolio_sortino = calculate_sortino_ratio(portfolio_return_series)
                    portfolio_var = calculate_var(portfolio_return_series) * 100
                    portfolio_cvar = calculate_cvar(portfolio_return_series) * 100
        
        # Performance distribution
        winners = sum(1 for r in results if r['current_return'] > 0)
        losers = sum(1 for r in results if r['current_return'] < 0)
        big_winners = sum(1 for r in results if r['current_return'] > 20)
        big_losers = sum(1 for r in results if r['current_return'] < -20)
        
        # Sector/Risk clustering (simplified)
        volatilities = [r.get('historical', {}).get('annual_volatility', 20) for r in results]
        try:
            if len(volatilities) > 1:
                # Simple clustering based on volatility
                vol_array = np.array(volatilities).reshape(-1, 1)
                if len(vol_array) >= 2:
                    scaler = StandardScaler()
                    vol_scaled = scaler.fit_transform(vol_array)
                    n_clusters = min(3, len(vol_array))
                    centroids, labels = kmeans2(vol_scaled, n_clusters, minit='++')
                    risk_clusters = len(set(labels))
                else:
                    risk_clusters = 1
            else:
                risk_clusters = 1
        except:
            risk_clusters = 1
        
        # Expected portfolio return (forward-looking)
        expected_returns = [r.get('target', {}).get('expected_return', 0) for r in results]
        weighted_expected_return = sum(ret * w for ret, w in zip(expected_returns, weights))
        
        # Risk level assessment
        high_risk_positions = sum(1 for r in results if r.get('recommendation', {}).get('risk_level', 'Medium') == 'High')
        risk_level_score = high_risk_positions / len(results) * 100
        
        # Performance rating with more nuanced scoring
        rating_score = 0
        if total_return > 30:
            rating_score += 5
        elif total_return > 20:
            rating_score += 4
        elif total_return > 10:
            rating_score += 3
        elif total_return > 0:
            rating_score += 2
        elif total_return > -10:
            rating_score += 1
        
        # Adjust for risk
        if portfolio_sharpe > 1.5:
            rating_score += 1
        elif portfolio_sharpe < 0:
            rating_score -= 1
        
        # Adjust for diversification
        if diversification_score > 70:
            rating_score += 1
        elif diversification_score < 30:
            rating_score -= 1
        
        if rating_score >= 6:
            rating = "⭐⭐⭐⭐⭐ Exceptional"
        elif rating_score >= 5:
            rating = "⭐⭐⭐⭐⭐ Excellent"
        elif rating_score >= 4:
            rating = "⭐⭐⭐⭐ Good"
        elif rating_score >= 3:
            rating = "⭐⭐⭐ Average"
        elif rating_score >= 2:
            rating = "⭐⭐ Below Average"
        elif rating_score >= 1:
            rating = "⭐⭐ Poor"
        else:
            rating = "⭐ Very Poor"
        
        return {
            'total_invested': total_invested,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'weighted_annual_return': weighted_annual_return,
            'weighted_expected_return': weighted_expected_return,
            'portfolio_volatility': portfolio_volatility,
            'portfolio_sharpe': portfolio_sharpe,
            'portfolio_sortino': portfolio_sortino,
            'portfolio_var': portfolio_var,
            'portfolio_cvar': portfolio_cvar,
            'diversification_score': diversification_score,
            'avg_correlation': avg_correlation,
            'max_weight': max_weight,
            'top_3_concentration': top_3_concentration,
            'n_stocks': n_stocks,
            'risk_clusters': risk_clusters,
            'winners': winners,
            'losers': losers,
            'big_winners': big_winners,
            'big_losers': big_losers,
            'risk_level_score': risk_level_score,
            'rating': rating,
            'rating_score': rating_score,
            'correlation_matrix': correlation_matrix
        }
    except Exception as e:
        logger.error(f"Error calculating portfolio summary: {e}")
        return {
            'total_invested': sum(r.get('invested', 0) for r in results),
            'total_value': sum(r.get('current_value', 0) for r in results),
            'rating': "⭐ Analysis Incomplete"
        }

def print_detailed_report(results: List[Dict], summary: Dict):
    """Print comprehensive enhanced report"""
    
    print(f"\n{'='*100}")
    print(f"{'ADVANCED PORTFOLIO PERFORMANCE ANALYSIS':^100}")
    print(f"{'='*100}")
    
    # Portfolio Overview
    print(f"\n{'📈 PORTFOLIO OVERVIEW':<50}")
    print(f"{'─'*50}")
    print(f"{'Total Invested:':<30} ₹{summary['total_invested']:>15,.2f}")
    print(f"{'Current Value:':<30} ₹{summary['total_value']:>15,.2f}")
    print(f"{'Total P&L:':<30} ₹{summary['total_pnl']:>15,.2f} ({summary['total_return']:+.2f}%)")
    print(f"{'Expected Return (3m):':<30} {summary.get('weighted_expected_return', 0):>15.2f}%")
    print(f"{'Portfolio Rating:':<30} {summary['rating']:>15}")
    
    # Risk Metrics
    print(f"\n{'⚠️  RISK ANALYSIS':<50}")
    print(f"{'─'*50}")
    print(f"{'Portfolio Volatility:':<30} {summary['portfolio_volatility']:>15.2f}%")
    print(f"{'Sharpe Ratio:':<30} {summary.get('portfolio_sharpe', 0):>15.2f}")
    print(f"{'Sortino Ratio:':<30} {summary.get('portfolio_sortino', 0):>15.2f}")
    print(f"{'VaR (5%):':<30} {summary.get('portfolio_var', 0):>15.2f}%")
    print(f"{'CVaR (5%):':<30} {summary.get('portfolio_cvar', 0):>15.2f}%")
    print(f"{'High Risk Positions:':<30} {summary.get('risk_level_score', 0):>15.1f}%")
    
    # Diversification Analysis
    print(f"\n{'🎯 DIVERSIFICATION ANALYSIS':<50}")
    print(f"{'─'*50}")
    print(f"{'Number of Holdings:':<30} {summary['n_stocks']:>15}")
    print(f"{'Diversification Score:':<30} {summary['diversification_score']:>15.1f}/100")
    print(f"{'Average Correlation:':<30} {summary.get('avg_correlation', 0):>15.2f}")
    print(f"{'Max Position Weight:':<30} {summary.get('max_weight', 0):>15.1f}%")
    print(f"{'Top 3 Concentration:':<30} {summary.get('top_3_concentration', 0):>15.1f}%")
    print(f"{'Risk Clusters:':<30} {summary.get('risk_clusters', 1):>15}")
    
    # Performance Distribution
    print(f"\n{'📊 PERFORMANCE DISTRIBUTION':<50}")
    print(f"{'─'*50}")
    print(f"{'Winners:':<30} {summary.get('winners', 0):>15} stocks")
    print(f"{'Losers:':<30} {summary.get('losers', 0):>15} stocks")
    print(f"{'Big Winners (>20%):':<30} {summary.get('big_winners', 0):>15} stocks")
    print(f"{'Big Losers (<-20%):':<30} {summary.get('big_losers', 0):>15} stocks")
    
    # Correlation Matrix
    if 'correlation_matrix' in summary and not summary['correlation_matrix'].empty:
        print(f"\n{'🔗 CORRELATION MATRIX':<50}")
        print(f"{'─'*50}")
        corr_matrix = summary['correlation_matrix']
        print(corr_matrix.round(2).to_string())
    
    # Individual stock details
    print(f"\n{'='*100}")
    print(f"{'DETAILED STOCK ANALYSIS':^100}")
    print(f"{'='*100}")
    
    for r in results:
        print(f"\n{'─'*100}")
        rec = r.get('recommendation', {})
        print(f"📊 {r['ticker']} - {rec.get('action', 'N/A')} | Risk: {rec.get('risk_level', 'N/A')} | Score: {rec.get('score', 0)}")
        print(f"{'─'*100}")
        
        # Basic Position Info
        print(f"{'Position:':<25} {r['shares']} shares @ ₹{r['avg_price']:.2f} avg price")
        print(f"{'Current Price:':<25} ₹{r['current_price']:.2f}")
        print(f"{'Investment:':<25} ₹{r['invested']:,.2f}")
        print(f"{'Current Value:':<25} ₹{r['current_value']:,.2f}")
        print(f"{'P&L:':<25} ₹{r['pnl']:,.2f} ({r['current_return']:+.2f}%)")
        print(f"{'Annualized Return:':<25} {r['annualized_return']:.2f}%")
        print(f"{'Holding Period:':<25} {r['holding_months']} months")
        
        # Enhanced Technical Analysis
        tech = r.get('technical', {})
        hist = r.get('historical', {})
        
        print(f"\n{'🔍 Technical Indicators:'}:")
        print(f"  • Trend Strength: {tech.get('trend_strength', 0):.1f}/100 (ADX: {tech.get('adx', 0):.1f})")
        print(f"  • RSI: {tech.get('rsi', 50):.1f} ({tech.get('momentum', 'N/A')})")
        print(f"  • Stochastic: {tech.get('stochastic_k', 50):.1f}")
        print(f"  • MACD: {tech.get('macd_signal', 'N/A')} ({tech.get('macd_histogram_trend', 'N/A')})")
        print(f"  • Volume Trend: {tech.get('volume_trend', 'N/A')} ({tech.get('volume_ratio', 1):.1f}x)")
        print(f"  • Price Momentum (20d): {tech.get('price_momentum_20d', 0):+.2f}%")
        
        print(f"\n{'📈 Price Levels:'}:")
        print(f"  • vs EMA20: {tech.get('price_vs_ema20', 0):+.2f}%")
        print(f"  • vs EMA50: {tech.get('price_vs_ema50', 0):+.2f}%")
        print(f"  • vs EMA200: {tech.get('price_vs_ema200', 0):+.2f}%")
        print(f"  • Support (20d): ₹{tech.get('support_20d', 0):.2f}")
        print(f"  • Resistance (20d): ₹{tech.get('resistance_20d', 0):.2f}")
        
        # Advanced Price Targets
        target = r.get('target', {})
        print(f"\n{'🎯 Price Targets (3-month):'}:")
        print(f"  • Conservative: ₹{target.get('conservative_target', 0):.2f} ({target.get('conservative_return', 0):+.2f}%)")
        print(f"  • Average: ₹{target.get('avg_target', 0):.2f} ({target.get('expected_return', 0):+.2f}%)")
        print(f"  • Aggressive: ₹{target.get('aggressive_target', 0):.2f} ({target.get('aggressive_return', 0):+.2f}%)")
        print(f"  • Prediction Confidence: {target.get('prediction_confidence', 50):.0f}%")
        
        # Monte Carlo Results
        mc_percentiles = target.get('monte_carlo_percentiles', {})
        if mc_percentiles:
            print(f"  • Monte Carlo Percentiles:")
            for pct, price in mc_percentiles.items():
                print(f"    - {pct}: ₹{price:.2f}")
        
        # Risk Metrics
        if hist:
            print(f"\n{'⚠️  Risk Metrics:'}:")
            print(f"  • Sharpe Ratio: {hist.get('sharpe_ratio', 0):.2f}")
            print(f"  • Sortino Ratio: {hist.get('sortino_ratio', 0):.2f}")
            print(f"  • Max Drawdown: {hist.get('max_drawdown', 0):.2f}%")
            print(f"  • VaR (5%): {hist.get('var_5_percent', 0):.2f}%")
            print(f"  • CVaR (5%): {hist.get('cvar_5_percent', 0):.2f}%")
            print(f"  • Beta: {hist.get('beta', 1):.2f}")
        
        # Enhanced Recommendation
        print(f"\n{'💡 Recommendation:'} {rec.get('action', 'N/A')}")
        print(f"{'Details:'} {rec.get('action_detail', 'N/A')}")
        print(f"{'Confidence:'} {rec.get('confidence', 'N/A')} | Risk Level: {rec.get('risk_level', 'N/A')}")
        
        opportunity_factors = rec.get('opportunity_factors', [])
        risk_factors = rec.get('risk_factors', [])
        
        if opportunity_factors:
            print(f"{'🟢 Opportunities:'}:")
            for factor in opportunity_factors:
                print(f"  • {factor}")
        
        if risk_factors:
            print(f"{'🔴 Risk Factors:'}:")
            for factor in risk_factors:
                print(f"  • {factor}")
    
    # Enhanced Action Summary
    print(f"\n{'='*100}")
    print(f"{'STRATEGIC RECOMMENDATIONS':^100}")
    print(f"{'='*100}\n")
    
    # Categorize recommendations
    strong_buys = [r for r in results if 'STRONG BUY' in r.get('recommendation', {}).get('action', '')]
    buys = [r for r in results if 'BUY' in r.get('recommendation', {}).get('action', '') and 'STRONG' not in r.get('recommendation', {}).get('action', '')]
    holds = [r for r in results if 'HOLD' in r.get('recommendation', {}).get('action', '') and 'SELL' not in r.get('recommendation', {}).get('action', '')]
    reduces = [r for r in results if 'REDUCE' in r.get('recommendation', {}).get('action', '') or 'TRIM' in r.get('recommendation', {}).get('action', '')]
    sells = [r for r in results if 'SELL' in r.get('recommendation', {}).get('action', '') or 'EXIT' in r.get('recommendation', {}).get('action', '')]
    
    if strong_buys:
        print("� STRONG BUY OPPORTUNITIES:")
        for r in strong_buys:
            target = r.get('target', {})
            print(f"   • {r['ticker']:<15} Target: ₹{target.get('avg_target', 0):.2f} (+{target.get('expected_return', 0):.1f}%) | Confidence: {target.get('prediction_confidence', 50):.0f}%")
    
    if buys:
        print("\n🟢 BUY/ADD POSITIONS:")
        for r in buys:
            target = r.get('target', {})
            print(f"   • {r['ticker']:<15} Target: ₹{target.get('avg_target', 0):.2f} (+{target.get('expected_return', 0):.1f}%) | Risk: {r.get('recommendation', {}).get('risk_level', 'N/A')}")
    
    if holds:
        print("\n🟡 HOLD POSITIONS:")
        for r in holds:
            target = r.get('target', {})
            print(f"   • {r['ticker']:<15} Current: {r['current_return']:+.1f}%, Expected: +{target.get('expected_return', 0):.1f}%")
    
    if reduces:
        print("\n🟠 REDUCE/TRIM POSITIONS:")
        for r in reduces:
            rec = r.get('recommendation', {})
            print(f"   • {r['ticker']:<15} Current: {r['current_return']:+.1f}% | Risk Level: {rec.get('risk_level', 'N/A')}")
    
    if sells:
        print("\n🔴 SELL/EXIT POSITIONS:")
        for r in sells:
            target = r.get('target', {})
            rec = r.get('recommendation', {})
            print(f"   • {r['ticker']:<15} Risk: {target.get('expected_return', 0):.1f}%, Current: {r['current_return']:+.1f}% | Confidence: {rec.get('confidence', 'N/A')}")
    
    # Portfolio Recommendations
    print(f"\n{'🎯 PORTFOLIO OPTIMIZATION SUGGESTIONS:'}")
    print(f"{'─'*50}")
    
    if summary.get('diversification_score', 0) < 50:
        print("• Consider adding more diversified positions to reduce correlation risk")
    if summary.get('max_weight', 0) > 30:
        print(f"• Largest position ({summary.get('max_weight', 0):.1f}%) may be over-concentrated")
    if summary.get('portfolio_sharpe', 0) < 0.5:
        print("• Portfolio risk-adjusted returns could be improved")
    if summary.get('avg_correlation', 0) > 0.7:
        print("• Holdings are highly correlated - consider diversifying across sectors")
    
    print(f"\n{'='*100}")
    print(f"{'Analysis completed with advanced algorithms and risk assessment':^100}")
    print(f"{'='*100}\n")

# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='Portfolio Performance Analyzer')
    parser.add_argument('--csv', type=str, help='CSV file with portfolio (ticker,total_shares,avg_price,holding_months)')
    args = parser.parse_args()
    
    # Load portfolio
    if args.csv:
        try:
            df = pd.read_csv(args.csv)
            portfolio = []
            for _, row in df.iterrows():
                portfolio.append({
                    'ticker': row['ticker'],
                    'shares': int(row['total_shares']),
                    'avg_price': float(row['avg_price']),
                    'months': int(row['holding_months'])
                })
        except Exception as e:
            print(f"Error reading CSV: {e}")
            print("Using default portfolio from code...")
            portfolio = PORTFOLIO
    else:
        portfolio = PORTFOLIO
    
    print(f"\n{'='*90}")
    print(f"{'PORTFOLIO ANALYZER - Analyzing Your Holdings':^90}")
    print(f"{'='*90}")
    print(f"\nTotal Holdings: {len(portfolio)}")
    
    # Analyze each holding
    results = []
    for holding in portfolio:
        result = analyze_holding(holding)
        if result:
            results.append(result)
    
    if not results:
        print("\n❌ No valid holdings to analyze. Check your ticker symbols.")
        return
    
    # Calculate summary
    summary = portfolio_summary(results)
    
    # Print detailed report
    print_detailed_report(results, summary)

if __name__ == "__main__":
    main()