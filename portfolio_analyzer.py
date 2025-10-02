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

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ==================== YOUR PORTFOLIO (Edit this) ====================
PORTFOLIO = [
    {'ticker': 'TATAMOTORS.NS', 'shares': 46, 'avg_price': 711.24, 'months': 12},
    {'ticker': 'ITC.NS', 'shares': 15, 'avg_price': 412.00, 'months': 12},
    {'ticker': 'TATACHEM.NS', 'shares': 5, 'avg_price': 1095.00, 'months': 12},
    {'ticker': 'ACMESOLAR.NS', 'shares': 4, 'avg_price': 243.00, 'months': 12},
    {'ticker': 'CONTAINERCORP.NS', 'shares': 1, 'avg_price': 888.00, 'months': 12},
]


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

# ==================== DATA FETCHING ====================

def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch historical data"""
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

def get_current_price(ticker: str) -> float:
    """Get latest price"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return 0.0
    except:
        return 0.0

# ==================== ANALYSIS FUNCTIONS ====================

def calculate_returns_metrics(df: pd.DataFrame, holding_months: int) -> Dict:
    """Calculate historical returns and volatility"""
    if df.empty or len(df) < 20:
        return {}
    
    df['returns'] = df['close'].pct_change()
    
    # Annualized metrics
    annual_return = (df['close'].iloc[-1] / df['close'].iloc[0]) ** (252 / len(df)) - 1
    annual_vol = df['returns'].std() * np.sqrt(252)
    sharpe = (annual_return - 0.065) / annual_vol if annual_vol > 0 else 0
    
    # Max drawdown
    cumulative = (1 + df['returns']).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    return {
        'annual_return': annual_return * 100,
        'annual_volatility': annual_vol * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd * 100
    }

def technical_analysis(df: pd.DataFrame) -> Dict:
    """Comprehensive technical analysis"""
    if df.empty or len(df) < 50:
        return {}
    
    # Add indicators
    df['ema_20'] = ema(df['close'], 20)
    df['ema_50'] = ema(df['close'], 50)
    df['ema_200'] = ema(df['close'], 200)
    df['rsi'] = rsi(df['close'], 14)
    
    macd_line, macd_sig, macd_hist = macd(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = macd_sig
    
    bb_upper, bb_mid, bb_lower = bollinger_bands(df['close'])
    df['bb_upper'] = bb_upper
    df['bb_mid'] = bb_mid
    df['bb_lower'] = bb_lower
    
    latest = df.iloc[-1]
    
    # Trend analysis
    trend_score = 0
    if latest['close'] > latest['ema_20']:
        trend_score += 1
    if latest['close'] > latest['ema_50']:
        trend_score += 1
    if latest['close'] > latest['ema_200']:
        trend_score += 1
    if latest['ema_20'] > latest['ema_50']:
        trend_score += 1
    if latest['ema_50'] > latest['ema_200']:
        trend_score += 1
    
    trend_strength = (trend_score / 5.0) * 100
    
    # Momentum
    rsi_val = latest['rsi']
    if rsi_val > 70:
        momentum = "Overbought"
    elif rsi_val < 30:
        momentum = "Oversold"
    elif rsi_val > 50:
        momentum = "Bullish"
    else:
        momentum = "Bearish"
    
    # MACD signal
    macd_signal = "Bullish" if latest['macd'] > latest['macd_signal'] else "Bearish"
    
    # Support/Resistance
    support = df['close'].rolling(50).min().iloc[-1]
    resistance = df['close'].rolling(50).max().iloc[-1]
    
    # Price position in BB
    bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
    
    return {
        'trend_strength': trend_strength,
        'rsi': rsi_val,
        'momentum': momentum,
        'macd_signal': macd_signal,
        'support': support,
        'resistance': resistance,
        'bb_position': bb_position * 100,
        'price_vs_ema20': ((latest['close'] / latest['ema_20']) - 1) * 100,
        'price_vs_ema50': ((latest['close'] / latest['ema_50']) - 1) * 100,
        'price_vs_ema200': ((latest['close'] / latest['ema_200']) - 1) * 100
    }

def predict_target_price(df: pd.DataFrame, current_price: float, months_ahead: int = 3) -> Dict:
    """Predict target price using multiple methods"""
    if df.empty or len(df) < 50:
        return {}
    
    # Method 1: Linear regression on recent trend
    recent = df.tail(60).copy()
    recent['index'] = range(len(recent))
    x = recent['index'].values
    y = recent['close'].values
    z = np.polyfit(x, y, 1)
    slope = z[0]
    
    days_ahead = months_ahead * 30
    linear_target = current_price + (slope * days_ahead)
    
    # Method 2: Moving average projection
    ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
    ema_200 = df['close'].ewm(span=200).mean().iloc[-1]
    ma_target = (ema_50 + ema_200) / 2
    
    # Method 3: Bollinger bands
    bb_upper, bb_mid, bb_lower = bollinger_bands(df['close'])
    bb_target = bb_mid.iloc[-1]
    
    # Method 4: Support/Resistance
    resistance = df['close'].rolling(50).max().iloc[-1]
    
    # Average target
    avg_target = np.mean([linear_target, ma_target, bb_target, resistance])
    
    return {
        'linear_target': linear_target,
        'ma_target': ma_target,
        'bb_target': bb_target,
        'resistance_target': resistance,
        'avg_target': avg_target,
        'expected_return': ((avg_target / current_price) - 1) * 100
    }

def recommendation_engine(stock_data: Dict, holding_months: int) -> Dict:
    """Generate HOLD/SELL/ADD recommendation"""
    current_return = stock_data['current_return']
    tech = stock_data['technical']
    target = stock_data['target']
    
    score = 0
    reasons = []
    
    # Factor 1: Current profit/loss
    if current_return > 20:
        score += 2
        reasons.append(f"Strong gains ({current_return:.1f}%)")
    elif current_return > 10:
        score += 1
        reasons.append(f"Moderate gains ({current_return:.1f}%)")
    elif current_return < -10:
        score -= 2
        reasons.append(f"Heavy losses ({current_return:.1f}%)")
    elif current_return < -5:
        score -= 1
        reasons.append(f"Losses ({current_return:.1f}%)")
    
    # Factor 2: Trend strength
    if tech['trend_strength'] > 70:
        score += 2
        reasons.append("Strong uptrend")
    elif tech['trend_strength'] > 50:
        score += 1
        reasons.append("Mild uptrend")
    elif tech['trend_strength'] < 30:
        score -= 2
        reasons.append("Weak/downtrend")
    
    # Factor 3: RSI (overbought/oversold)
    if tech['rsi'] > 75:
        score -= 1
        reasons.append("Overbought (RSI>75)")
    elif tech['rsi'] < 30:
        score += 1
        reasons.append("Oversold - potential bounce")
    
    # Factor 4: Expected return
    if target['expected_return'] > 15:
        score += 2
        reasons.append(f"High upside ({target['expected_return']:.1f}%)")
    elif target['expected_return'] > 5:
        score += 1
        reasons.append(f"Moderate upside ({target['expected_return']:.1f}%)")
    elif target['expected_return'] < -5:
        score -= 2
        reasons.append(f"Downside risk ({target['expected_return']:.1f}%)")
    
    # Factor 5: Holding period
    if holding_months < 3:
        reasons.append("Short holding period")
    elif holding_months > 12:
        reasons.append("Long-term holding")
        if current_return > 15:
            score += 1  # Reward long-term winners
    
    # Factor 6: MACD
    if tech['macd_signal'] == "Bullish":
        score += 1
    else:
        score -= 1
    
    # Generate recommendation
    if score >= 4:
        action = "🟢 STRONG BUY/ADD"
        confidence = "High"
    elif score >= 2:
        action = "🟢 HOLD/ADD"
        confidence = "Medium-High"
    elif score >= 0:
        action = "🟡 HOLD"
        confidence = "Medium"
    elif score >= -2:
        action = "🟠 HOLD/REDUCE"
        confidence = "Medium-Low"
    else:
        action = "🔴 SELL/EXIT"
        confidence = "High"
    
    return {
        'action': action,
        'confidence': confidence,
        'score': score,
        'reasons': reasons
    }

# ==================== PORTFOLIO ANALYSIS ====================

def analyze_holding(holding: Dict) -> Dict:
    """Analyze a single holding"""
    ticker = holding['ticker']
    shares = holding['shares']
    avg_price = holding['avg_price']
    months = holding['months']
    
    print(f"\nAnalyzing {ticker}...", end=" ")
    
    # Fetch data
    df = fetch_stock_data(ticker, period="2y")
    current_price = get_current_price(ticker)
    
    if df.empty or current_price == 0:
        print("❌ Failed to fetch data")
        return None
    
    # Calculate metrics
    invested = shares * avg_price
    current_value = shares * current_price
    pnl = current_value - invested
    pnl_pct = (pnl / invested) * 100
    
    # Annualized return
    annualized_return = ((current_price / avg_price) ** (12 / months) - 1) * 100 if months > 0 else 0
    
    # Historical metrics
    hist_metrics = calculate_returns_metrics(df, months)
    
    # Technical analysis
    tech = technical_analysis(df)
    
    # Target price
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
        'target': target
    }
    
    # Generate recommendation
    recommendation = recommendation_engine(stock_data, months)
    stock_data['recommendation'] = recommendation
    
    print(f"✓ (Return: {pnl_pct:+.1f}%, Action: {recommendation['action']})")
    
    return stock_data

def portfolio_summary(results: List[Dict]) -> Dict:
    """Calculate portfolio-level metrics"""
    total_invested = sum(r['invested'] for r in results)
    total_value = sum(r['current_value'] for r in results)
    total_pnl = total_value - total_invested
    total_return = (total_pnl / total_invested) * 100
    
    # Weighted returns
    weights = [r['current_value'] / total_value for r in results]
    weighted_annual_return = sum(r['annualized_return'] * w for r, w in zip(results, weights))
    
    # Risk metrics
    returns = [r['current_return'] for r in results]
    portfolio_volatility = np.std(returns) if len(returns) > 1 else 0
    
    # Diversification score (0-100)
    # Based on: number of stocks, sector spread (proxy: volatility of returns), concentration
    n_stocks = len(results)
    concentration = max(weights) * 100  # Highest single stock weight
    
    diversification = min(100, (n_stocks * 15) - concentration + (portfolio_volatility * 0.5))
    
    # Performance rating
    if total_return > 20:
        rating = "⭐⭐⭐⭐⭐ Excellent"
    elif total_return > 10:
        rating = "⭐⭐⭐⭐ Good"
    elif total_return > 0:
        rating = "⭐⭐⭐ Average"
    elif total_return > -10:
        rating = "⭐⭐ Poor"
    else:
        rating = "⭐ Very Poor"
    
    return {
        'total_invested': total_invested,
        'total_value': total_value,
        'total_pnl': total_pnl,
        'total_return': total_return,
        'weighted_annual_return': weighted_annual_return,
        'portfolio_volatility': portfolio_volatility,
        'diversification_score': diversification,
        'concentration': concentration,
        'n_stocks': n_stocks,
        'rating': rating
    }

def print_detailed_report(results: List[Dict], summary: Dict):
    """Print comprehensive report"""
    
    print(f"\n{'='*90}")
    print(f"{'PORTFOLIO PERFORMANCE SUMMARY':^90}")
    print(f"{'='*90}")
    
    print(f"\n{'Total Invested:':<30} ₹{summary['total_invested']:>15,.2f}")
    print(f"{'Current Value:':<30} ₹{summary['total_value']:>15,.2f}")
    print(f"{'Total P&L:':<30} ₹{summary['total_pnl']:>15,.2f} ({summary['total_return']:+.2f}%)")
    print(f"{'Weighted Annual Return:':<30} {summary['weighted_annual_return']:>15.2f}%")
    print(f"{'Portfolio Rating:':<30} {summary['rating']:>15}")
    print(f"{'Diversification Score:':<30} {summary['diversification_score']:>15.1f}/100")
    print(f"{'Number of Holdings:':<30} {summary['n_stocks']:>15}")
    
    # Individual stock details
    print(f"\n{'='*90}")
    print(f"{'INDIVIDUAL STOCK ANALYSIS':^90}")
    print(f"{'='*90}")
    
    for r in results:
        print(f"\n{'─'*90}")
        print(f"📊 {r['ticker']} - {r['recommendation']['action']}")
        print(f"{'─'*90}")
        
        print(f"{'Position:':<25} {r['shares']} shares @ ₹{r['avg_price']:.2f} avg")
        print(f"{'Current Price:':<25} ₹{r['current_price']:.2f}")
        print(f"{'Investment:':<25} ₹{r['invested']:,.2f}")
        print(f"{'Current Value:':<25} ₹{r['current_value']:,.2f}")
        print(f"{'P&L:':<25} ₹{r['pnl']:,.2f} ({r['current_return']:+.2f}%)")
        print(f"{'Annualized Return:':<25} {r['annualized_return']:.2f}%")
        print(f"{'Holding Period:':<25} {r['holding_months']} months")
        
        tech = r['technical']
        print(f"\n{'Technical Indicators:'}:")
        print(f"  • Trend Strength: {tech['trend_strength']:.1f}/100")
        print(f"  • RSI: {tech['rsi']:.1f} ({tech['momentum']})")
        print(f"  • MACD: {tech['macd_signal']}")
        print(f"  • Price vs EMA20: {tech['price_vs_ema20']:+.2f}%")
        print(f"  • Price vs EMA50: {tech['price_vs_ema50']:+.2f}%")
        
        target = r['target']
        print(f"\n{'Price Targets (3-month):'}:")
        print(f"  • Average Target: ₹{target['avg_target']:.2f} ({target['expected_return']:+.2f}%)")
        print(f"  • Resistance: ₹{target['resistance_target']:.2f}")
        print(f"  • Support: ₹{tech['support']:.2f}")
        
        rec = r['recommendation']
        print(f"\n{'Recommendation:'} {rec['action']} (Confidence: {rec['confidence']})")
        print(f"{'Reasons:'}:")
        for reason in rec['reasons']:
            print(f"  • {reason}")
    
    # Action summary
    print(f"\n{'='*90}")
    print(f"{'RECOMMENDED ACTIONS':^90}")
    print(f"{'='*90}\n")
    
    holds = [r for r in results if 'HOLD' in r['recommendation']['action'] and 'SELL' not in r['recommendation']['action']]
    buys = [r for r in results if 'BUY' in r['recommendation']['action'] or 'ADD' in r['recommendation']['action']]
    sells = [r for r in results if 'SELL' in r['recommendation']['action'] or 'REDUCE' in r['recommendation']['action']]
    
    if buys:
        print("🟢 STOCKS TO BUY/ADD MORE:")
        for r in buys:
            print(f"   • {r['ticker']:<15} Target: ₹{r['target']['avg_target']:.2f} (+{r['target']['expected_return']:.1f}%)")
    
    if holds:
        print("\n🟡 STOCKS TO HOLD:")
        for r in holds:
            print(f"   • {r['ticker']:<15} Current: {r['current_return']:+.1f}%, Target: +{r['target']['expected_return']:.1f}%")
    
    if sells:
        print("\n🔴 STOCKS TO SELL/REDUCE:")
        for r in sells:
            print(f"   • {r['ticker']:<15} Risk: {r['target']['expected_return']:.1f}%, Current: {r['current_return']:+.1f}%")
    
    print(f"\n{'='*90}\n")

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