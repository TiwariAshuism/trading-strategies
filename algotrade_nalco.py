#!/usr/bin/env python3
"""
enhanced_stock_screener.py

Multi-strategy algorithmic trading screener for Indian equities.
Analyzes multiple stocks, computes risk metrics, and provides portfolio allocation.

Usage:
python enhanced_stock_screener.py --capital 30000 --period 1y

Strategies Implemented:
1. EMA Crossover with Volume Confirmation
2. RSI Mean Reversion
3. Bollinger Band Breakout
4. MACD Momentum
5. Support/Resistance Bounce
"""

import argparse
import datetime as dt
import math
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

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

# ==================== DATA FETCHING ====================

def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch OHLCV data with error handling"""
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        
        if df.empty:
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
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
    """Strategy 5: Support/Resistance Bounce"""
    if len(df) < 50:
        return {'signal': 0, 'score': 0, 'reason': 'Insufficient data'}
    
    # Find support (recent 20-day low)
    df['support'] = df['low'].rolling(20).min()
    df['resistance'] = df['high'].rolling(20).max()
    df['rsi'] = rsi(df['close'], 14)
    
    latest = df.iloc[-1]
    
    # Price near support (within 2%)
    near_support = latest['close'] <= latest['support'] * 1.02
    # RSI showing bullish divergence (price low but RSI not as low)
    rsi_healthy = latest['rsi'] > 35
    # Price bouncing up
    bouncing = latest['close'] > df.iloc[-2]['close']
    
    score = 0
    if near_support:
        score += 40
    if rsi_healthy:
        score += 30
    if bouncing:
        score += 30
    
    signal = 1 if score >= 60 else 0
    reason = f"Near Support: {near_support}, RSI: {latest['rsi']:.1f}, Bounce: {bouncing}"
    
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

# ==================== PORTFOLIO ANALYSIS ====================

def analyze_stock(ticker: str, period: str = "1y") -> Optional[Dict]:
    """Comprehensive analysis of a single stock"""
    print(f"Analyzing {ticker}...", end=" ")
    
    df = fetch_stock_data(ticker, period)
    if df.empty or len(df) < 50:
        print("❌ Insufficient data")
        return None
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Run all strategies
    strategies = {
        'ema_crossover': strategy_ema_crossover(df),
        'rsi_reversion': strategy_rsi_mean_reversion(df),
        'bb_breakout': strategy_bollinger_breakout(df),
        'macd_momentum': strategy_macd_momentum(df),
        'support_bounce': strategy_support_resistance(df)
    }
    
    # Aggregate signals
    total_score = sum(s['score'] for s in strategies.values())
    signal_count = sum(s['signal'] for s in strategies.values())
    
    # Calculate risk metrics
    sharpe = calculate_sharpe_ratio(df['returns'].dropna())
    cagr = calculate_cagr(df)
    volatility = calculate_volatility(df['returns'].dropna())
    max_dd = calculate_max_drawdown(df)
    
    current_price = df['close'].iloc[-1]
    
    print(f"✓ (Signals: {signal_count}/5, Sharpe: {sharpe:.2f})")
    
    return {
        'ticker': ticker,
        'current_price': current_price,
        'sharpe_ratio': sharpe,
        'cagr': cagr,
        'volatility': volatility,
        'max_drawdown': max_dd,
        'signal_count': signal_count,
        'total_score': total_score,
        'strategies': strategies
    }

def rank_stocks(results: List[Dict]) -> pd.DataFrame:
    """Rank stocks by multiple criteria"""
    df = pd.DataFrame(results)
    
    # Composite score: Sharpe (50%) + CAGR (30%) + Signals (20%)
    df['sharpe_norm'] = (df['sharpe_ratio'] - df['sharpe_ratio'].min()) / (df['sharpe_ratio'].max() - df['sharpe_ratio'].min() + 0.001)
    df['cagr_norm'] = (df['cagr'] - df['cagr'].min()) / (df['cagr'].max() - df['cagr'].min() + 0.001)
    df['signal_norm'] = df['signal_count'] / 5.0
    
    df['composite_score'] = (
        df['sharpe_norm'] * 0.5 + 
        df['cagr_norm'] * 0.3 + 
        df['signal_norm'] * 0.2
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
    
    # Display top stocks
    print(f"\n{'='*70}")
    print("TOP 10 RECOMMENDED STOCKS (Ranked by Composite Score)")
    print(f"{'='*70}")
    print(f"{'Rank':<6}{'Ticker':<15}{'Price':<10}{'Sharpe':<10}{'CAGR%':<10}{'Vol%':<10}{'Signals':<8}")
    print("-" * 70)
    
    for i, row in ranked.head(10).iterrows():
        print(f"{ranked.index.get_loc(i)+1:<6}{row['ticker']:<15}₹{row['current_price']:<9.2f}"
              f"{row['sharpe_ratio']:<10.2f}{row['cagr']:<10.1f}{row['volatility']:<10.1f}{row['signal_count']}/5")
    
    # Portfolio allocation
    allocation = allocate_portfolio(ranked, args.capital, args.top)
    
    print(f"\n{'='*70}")
    print(f"RECOMMENDED PORTFOLIO ALLOCATION (₹{args.capital:,.0f})")
    print(f"{'='*70}")
    print(f"{'Ticker':<15}{'Shares':<10}{'Price':<12}{'Investment':<15}{'Sharpe':<10}{'Signals'}")
    print("-" * 70)
    
    for stock in allocation['portfolio']:
        print(f"{stock['ticker']:<15}{stock['shares']:<10}₹{stock['price']:<11.2f}"
              f"₹{stock['investment']:<14,.0f}{stock['sharpe']:<10.2f}{stock['signals']}/5")
    
    print("-" * 70)
    print(f"{'Total Invested:':<38}₹{allocation['total_invested']:,.0f}")
    print(f"{'Cash Remaining:':<38}₹{allocation['cash_remaining']:,.0f}")
    
    # Best single trade
    best = ranked.iloc[0]
    best_allocation = math.floor(args.capital / best['current_price']) * best['current_price']
    
    print(f"\n{'='*70}")
    print(f"💡 BEST SINGLE TRADE FOR THIS WEEK")
    print(f"{'='*70}")
    print(f"Stock: {best['ticker']}")
    print(f"Current Price: ₹{best['current_price']:.2f}")
    print(f"Allocation: ₹{best_allocation:,.0f} ({math.floor(args.capital/best['current_price'])} shares)")
    print(f"Sharpe Ratio: {best['sharpe_ratio']:.2f} | CAGR: {best['cagr']:.1f}% | Signals: {best['signal_count']}/5")
    
    # Show active strategies for best stock
    print(f"\nActive Strategies:")
    for name, strat in best['strategies'].items():
        if strat['signal'] == 1:
            print(f"  ✓ {name.replace('_', ' ').title()}: {strat['reason']}")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()