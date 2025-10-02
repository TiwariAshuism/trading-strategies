#!/usr/bin/env python3
"""
Configuration file for Advanced Short-Term Trading Strategy
Customize all parameters, weights, and thresholds here.
"""

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class StrategyConfig:
    """Configuration class for the Advanced Short-Term Strategy"""
    
    # Data fetching parameters
    DATA_PERIOD = "6mo"  # Data period for analysis
    DATA_INTERVAL = "1h"  # Data interval (1h, 1d, etc.)
    
    # Monte Carlo simulation parameters
    MC_SIMULATIONS = 1000  # Number of Monte Carlo simulations
    MC_DAYS = 3  # Forecast horizon in days
    MC_MEAN_REVERSION = -0.1  # Mean reversion factor
    
    # Technical indicator parameters
    MA_SHORT_PERIOD = 20  # Short moving average period
    MA_LONG_PERIOD = 50   # Long moving average period
    RSI_PERIOD = 14       # RSI calculation period
    RSI_OVERSOLD = 30     # RSI oversold threshold
    RSI_OVERBOUGHT = 70   # RSI overbought threshold
    MACD_FAST = 12        # MACD fast EMA
    MACD_SLOW = 26        # MACD slow EMA
    MACD_SIGNAL = 9       # MACD signal line
    BB_PERIOD = 20        # Bollinger Bands period
    BB_STD = 2            # Bollinger Bands standard deviation
    VOLUME_MULTIPLIER = 1.5  # High volume threshold multiplier
    
    # Strategy component weights (must sum to 1.0)
    WEIGHTS = {
        'monte_carlo': 0.25,   # Monte Carlo simulation weight
        'technical': 0.25,     # Technical indicators weight
        'sentiment': 0.20,     # Sentiment analysis weight
        'volume': 0.15,        # Volume analysis weight
        'candlestick': 0.15    # Candlestick patterns weight
    }
    
    # Signal thresholds
    SIGNAL_THRESHOLDS = {
        'very_strong_buy': 60,    # Very strong buy signal threshold
        'strong_buy': 30,         # Strong buy signal threshold
        'moderate_buy': 10,       # Moderate buy signal threshold
        'moderate_sell': -10,     # Moderate sell signal threshold
        'strong_sell': -30,       # Strong sell signal threshold
        'very_strong_sell': -60   # Very strong sell signal threshold
    }
    
    # Monte Carlo thresholds
    MC_THRESHOLDS = {
        'very_bullish': 0.7,    # Very bullish probability threshold
        'bullish': 0.6,         # Bullish probability threshold
        'bearish': 0.4,         # Bearish probability threshold
        'very_bearish': 0.3     # Very bearish probability threshold
    }
    
    # Risk management parameters
    RISK_MANAGEMENT = {
        'stop_loss_multiplier': 2.0,    # Stop loss = price ± (volatility × multiplier)
        'take_profit_multiplier': 3.0,  # Take profit = price ± (volatility × multiplier)
        'max_position_risk': 0.02,      # Maximum position risk (2% of portfolio)
        'min_risk_reward': 1.5          # Minimum risk/reward ratio
    }
    
    # Sentiment analysis settings
    SENTIMENT_CONFIG = {
        'news_sources': [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.investing.com/rss/news.rss"
        ],
        'max_news_items': 5,        # Maximum news items to analyze per source
        'sentiment_multiplier': 3,   # Sentiment score multiplier
        'min_sentiment_strength': 0.1  # Minimum sentiment to be considered significant
    }
    
    # Candlestick pattern settings
    CANDLESTICK_CONFIG = {
        'shadow_body_ratio': 2.0,      # Shadow to body ratio for hammer/shooting star
        'small_shadow_threshold': 0.1,  # Small shadow threshold
        'doji_threshold': 0.1           # Doji body size threshold
    }
    
    # Default stock symbols for quick testing
    DEFAULT_SYMBOLS = [
        'RELIANCE.NS',
        'TCS.NS', 
        'INFY.NS',
        'HDFCBANK.NS',
        'ICICIBANK.NS',
        'ITC.NS',
        'HINDUNILVR.NS',
        'LT.NS',
        'SBIN.NS',
        'BHARTIARTL.NS'
    ]
    
    # Logging configuration
    LOGGING_CONFIG = {
        'level': 'INFO',  # Logging level
        'format': '%(asctime)s - %(levelname)s - %(message)s'
    }
    
    # Chart settings
    CHART_CONFIG = {
        'figure_size': (16, 12),
        'colors': {
            'buy_signal': 'green',
            'sell_signal': 'red',
            'hold_signal': 'orange',
            'price': 'black',
            'ema_short': 'blue',
            'ema_long': 'red',
            'rsi': 'purple',
            'volume': 'skyblue'
        }
    }

# Create global config instance
CONFIG = StrategyConfig()

def get_config() -> StrategyConfig:
    """Get the global configuration instance"""
    return CONFIG

def update_weights(**kwargs):
    """Update strategy weights and ensure they sum to 1.0"""
    CONFIG.WEIGHTS.update(kwargs)
    total = sum(CONFIG.WEIGHTS.values())
    if abs(total - 1.0) > 0.01:
        print(f"⚠️ Warning: Weights sum to {total:.3f}, normalizing...")
        for key in CONFIG.WEIGHTS:
            CONFIG.WEIGHTS[key] /= total

def print_config():
    """Print current configuration"""
    print("\n📋 CURRENT STRATEGY CONFIGURATION")
    print("=" * 50)
    
    print(f"\n🔧 Data Parameters:")
    print(f"  Period: {CONFIG.DATA_PERIOD}")
    print(f"  Interval: {CONFIG.DATA_INTERVAL}")
    
    print(f"\n🎲 Monte Carlo:")
    print(f"  Simulations: {CONFIG.MC_SIMULATIONS}")
    print(f"  Days: {CONFIG.MC_DAYS}")
    
    print(f"\n📊 Weights:")
    for component, weight in CONFIG.WEIGHTS.items():
        print(f"  {component}: {weight:.2%}")
    
    print(f"\n Signal Thresholds:")
    for threshold, value in CONFIG.SIGNAL_THRESHOLDS.items():
        print(f"  {threshold}: {value}")
    
    print(f"\n⚠️ Risk Management:")
    for param, value in CONFIG.RISK_MANAGEMENT.items():
        print(f"  {param}: {value}")

# Example of how to customize configuration
def customize_for_aggressive_trading():
    """Customize config for aggressive trading"""
    update_weights(
        monte_carlo=0.35,
        technical=0.30, 
        sentiment=0.25,
        volume=0.05,
        candlestick=0.05
    )
    CONFIG.SIGNAL_THRESHOLDS['strong_buy'] = 20
    CONFIG.SIGNAL_THRESHOLDS['strong_sell'] = -20
    CONFIG.RISK_MANAGEMENT['stop_loss_multiplier'] = 1.5
    CONFIG.RISK_MANAGEMENT['take_profit_multiplier'] = 4.0

def customize_for_conservative_trading():
    """Customize config for conservative trading"""
    update_weights(
        monte_carlo=0.20,
        technical=0.20,
        sentiment=0.15,
        volume=0.25,
        candlestick=0.20
    )
    CONFIG.SIGNAL_THRESHOLDS['strong_buy'] = 40
    CONFIG.SIGNAL_THRESHOLDS['strong_sell'] = -40
    CONFIG.RISK_MANAGEMENT['stop_loss_multiplier'] = 2.5
    CONFIG.RISK_MANAGEMENT['take_profit_multiplier'] = 2.5

if __name__ == "__main__":
    print_config()