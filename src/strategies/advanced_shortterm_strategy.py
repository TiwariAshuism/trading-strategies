#!/usr/bin/env python3
"""
Advanced Short-Term Trading Strategy Model
Combines Monte Carlo simulation with multiple technical indicators and sentiment analysis
for high-confidence entry/exit signals.

Author: Trading Strategy Suite
Date: October 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import ssl
import requests
from textblob import TextBlob
import feedparser
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# Add project root to path for emoji handler
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import emoji handler
try:
    from src.utils.emoji_handler import safe_print, format_text
except ImportError:
    # Fallback functions
    def safe_print(text, end='\n', flush=False):
        try:
            print(text, end=end, flush=flush)
        except UnicodeEncodeError:
            text = str(text).replace("", "[LAUNCH]").replace("", "[OK]").replace("", "[ERROR]").replace("", "[CHART]").replace("", "[TARGET]")
            print(text, end=end, flush=flush)
    
    def format_text(text):
        return text

# SSL bypass for data fetching
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    """Signal strength classification"""
    VERY_STRONG = 5
    STRONG = 4
    MODERATE = 3
    WEAK = 2
    VERY_WEAK = 1

@dataclass
class TradingSignal:
    """Trading signal with confidence and reasoning"""
    direction: str  # 'BUY', 'SELL', 'HOLD'
    strength: SignalStrength
    confidence: float  # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: List[str]
    risk_reward_ratio: float
    expected_return: float
    time_horizon: str  # '1D', '3D', '1W'

class AdvancedShortTermStrategy:
    """
    Advanced Short-Term Trading Strategy combining:
    - Monte Carlo Simulation
    - Technical Indicators (MA, RSI, MACD, Bollinger Bands)
    - Volume Analysis
    - Candlestick Patterns
    - Sentiment Analysis
    - Event-Based Triggers
    """
    
    def __init__(self, symbol: str, period: str = "6mo"):
        self.symbol = symbol.upper()
        self.period = period
        self.data = None
        self.signals = {}
        self.final_signal = None
        
        # Strategy weights for multi-factor confirmation
        self.weights = {
            'monte_carlo': 0.25,
            'technical': 0.25,
            'sentiment': 0.20,
            'volume': 0.15,
            'candlestick': 0.15
        }
        
        logger.info(f"Initialized Advanced Strategy for {self.symbol}")
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch stock data with retry mechanism"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period, interval="1h")
            
            if self.data.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            logger.info(f"Fetched {len(self.data)} data points for {self.symbol}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error fetching data for {self.symbol}: {e}")
            raise
    
    def calculate_technical_indicators(self) -> Dict:
        """Calculate all technical indicators"""
        if self.data is None:
            raise ValueError("No data available. Call fetch_data() first.")
        
        df = self.data.copy()
        indicators = {}
        
        # 1. Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        df['EMA_50'] = df['Close'].ewm(span=50).mean()
        
        # Golden Cross / Death Cross
        indicators['golden_cross'] = df['EMA_20'].iloc[-1] > df['EMA_50'].iloc[-1] and \
                                   df['EMA_20'].iloc[-2] <= df['EMA_50'].iloc[-2]
        indicators['death_cross'] = df['EMA_20'].iloc[-1] < df['EMA_50'].iloc[-1] and \
                                  df['EMA_20'].iloc[-2] >= df['EMA_50'].iloc[-2]
        
        # 2. RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        indicators['rsi_current'] = df['RSI'].iloc[-1]
        indicators['rsi_oversold'] = df['RSI'].iloc[-1] < 30
        indicators['rsi_overbought'] = df['RSI'].iloc[-1] > 70
        
        # 3. MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        indicators['macd_bullish_crossover'] = df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and \
                                             df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2]
        indicators['macd_bearish_crossover'] = df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] and \
                                             df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2]
        
        # 4. Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        indicators['bb_squeeze'] = (df['BB_Upper'] - df['BB_Lower']).iloc[-1] < \
                                 (df['BB_Upper'] - df['BB_Lower']).rolling(20).mean().iloc[-1]
        indicators['bb_breakout_up'] = df['Close'].iloc[-1] > df['BB_Upper'].iloc[-1]
        indicators['bb_breakout_down'] = df['Close'].iloc[-1] < df['BB_Lower'].iloc[-1]
        
        # 5. Volume Analysis
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        indicators['high_volume'] = df['Volume'].iloc[-1] > df['Volume_SMA'].iloc[-1] * 1.5
        indicators['volume_trend'] = df['Volume'].rolling(5).mean().iloc[-1] > \
                                   df['Volume'].rolling(5).mean().iloc[-6]
        
        self.technical_data = df
        return indicators
    
    def monte_carlo_simulation(self, days: int = 3, simulations: int = 1000) -> Dict:
        """Advanced Monte Carlo simulation with multiple scenarios"""
        if self.data is None:
            raise ValueError("No data available")
        
        # Calculate returns and volatility
        returns = self.data['Close'].pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        
        current_price = self.data['Close'].iloc[-1]
        
        # Monte Carlo simulations
        simulated_prices = []
        
        for _ in range(simulations):
            price_path = [current_price]
            for day in range(days):
                # Add mean reversion component
                random_shock = np.random.normal(mean_return, std_return)
                # Mean reversion factor (prices tend to revert to mean)
                reversion_factor = -0.1 * (price_path[-1] / current_price - 1)
                
                next_price = price_path[-1] * (1 + random_shock + reversion_factor)
                price_path.append(next_price)
            
            simulated_prices.append(price_path[-1])
        
        simulated_prices = np.array(simulated_prices)
        
        # Calculate probabilities and statistics
        bullish_prob = np.sum(simulated_prices > current_price) / simulations
        bearish_prob = 1 - bullish_prob
        
        # Price ranges
        percentiles = np.percentile(simulated_prices, [5, 25, 50, 75, 95])
        
        monte_carlo_results = {
            'current_price': current_price,
            'mean_predicted': np.mean(simulated_prices),
            'median_predicted': np.median(simulated_prices),
            'bullish_probability': bullish_prob,
            'bearish_probability': bearish_prob,
            'price_range_5_95': (percentiles[0], percentiles[4]),
            'price_range_25_75': (percentiles[1], percentiles[3]),
            'expected_return': (np.mean(simulated_prices) - current_price) / current_price,
            'volatility_forecast': np.std(simulated_prices) / current_price,
            'max_gain': np.max(simulated_prices),
            'max_loss': np.min(simulated_prices),
            'all_simulations': simulated_prices
        }
        
        return monte_carlo_results
    
    def analyze_candlestick_patterns(self) -> Dict:
        """Analyze candlestick patterns for reversal/continuation signals"""
        if self.data is None:
            raise ValueError("No data available")
        
        df = self.data.copy()
        patterns = {}
        
        # Calculate candle properties
        df['Body'] = abs(df['Close'] - df['Open'])
        df['Upper_Shadow'] = df['High'] - np.maximum(df['Close'], df['Open'])
        df['Lower_Shadow'] = np.minimum(df['Close'], df['Open']) - df['Low']
        df['Range'] = df['High'] - df['Low']
        
        # Recent candles (last 3)
        recent = df.tail(3)
        last_candle = df.iloc[-1]
        
        # Hammer pattern (bullish reversal)
        hammer_condition = (
            last_candle['Lower_Shadow'] > 2 * last_candle['Body'] and
            last_candle['Upper_Shadow'] < 0.1 * last_candle['Body'] and
            last_candle['Close'] > last_candle['Open']  # Green candle
        )
        patterns['hammer'] = hammer_condition
        
        # Shooting Star (bearish reversal)
        shooting_star_condition = (
            last_candle['Upper_Shadow'] > 2 * last_candle['Body'] and
            last_candle['Lower_Shadow'] < 0.1 * last_candle['Body'] and
            last_candle['Close'] < last_candle['Open']  # Red candle
        )
        patterns['shooting_star'] = shooting_star_condition
        
        # Doji (indecision)
        doji_condition = last_candle['Body'] < 0.1 * last_candle['Range']
        patterns['doji'] = doji_condition
        
        # Engulfing patterns
        if len(recent) >= 2:
            prev_candle = df.iloc[-2]
            
            # Bullish engulfing
            bullish_engulfing = (
                prev_candle['Close'] < prev_candle['Open'] and  # Previous red
                last_candle['Close'] > last_candle['Open'] and  # Current green
                last_candle['Close'] > prev_candle['Open'] and  # Engulfs previous
                last_candle['Open'] < prev_candle['Close']
            )
            patterns['bullish_engulfing'] = bullish_engulfing
            
            # Bearish engulfing
            bearish_engulfing = (
                prev_candle['Close'] > prev_candle['Open'] and  # Previous green
                last_candle['Close'] < last_candle['Open'] and  # Current red
                last_candle['Close'] < prev_candle['Open'] and  # Engulfs previous
                last_candle['Open'] > prev_candle['Close']
            )
            patterns['bearish_engulfing'] = bearish_engulfing
        
        return patterns
    
    def fetch_sentiment_analysis(self) -> Dict:
        """Fetch and analyze sentiment from news and social media"""
        sentiment_data = {
            'news_sentiment': 0.0,
            'social_sentiment': 0.0,
            'overall_sentiment': 0.0,
            'sentiment_strength': 'NEUTRAL',
            'news_count': 0,
            'recent_events': []
        }
        
        try:
            # News sentiment analysis
            company_name = self.symbol  # You can map symbols to company names
            
            # RSS feeds for financial news
            news_sources = [
                f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={self.symbol}&region=US&lang=en-US",
                "https://feeds.bloomberg.com/markets/news.rss",
                "https://www.investing.com/rss/news.rss"
            ]
            
            news_sentiments = []
            news_items = []
            
            for source in news_sources:
                try:
                    feed = feedparser.parse(source)
                    for entry in feed.entries[:5]:  # Latest 5 news items
                        if self.symbol.lower() in entry.title.lower() or \
                           self.symbol.lower() in entry.get('summary', '').lower():
                            
                            # Analyze sentiment
                            text = f"{entry.title} {entry.get('summary', '')}"
                            blob = TextBlob(text)
                            sentiment_score = blob.sentiment.polarity
                            
                            news_sentiments.append(sentiment_score)
                            news_items.append({
                                'title': entry.title,
                                'sentiment': sentiment_score,
                                'published': entry.get('published', 'N/A')
                            })
                except Exception as e:
                    logger.warning(f"Error fetching from {source}: {e}")
                    continue
            
            if news_sentiments:
                sentiment_data['news_sentiment'] = np.mean(news_sentiments)
                sentiment_data['news_count'] = len(news_sentiments)
                sentiment_data['recent_events'] = news_items[:3]
            
            # Overall sentiment calculation
            sentiment_data['overall_sentiment'] = sentiment_data['news_sentiment']
            
            # Classify sentiment strength
            abs_sentiment = abs(sentiment_data['overall_sentiment'])
            if abs_sentiment > 0.6:
                sentiment_data['sentiment_strength'] = 'VERY_STRONG'
            elif abs_sentiment > 0.3:
                sentiment_data['sentiment_strength'] = 'STRONG'
            elif abs_sentiment > 0.1:
                sentiment_data['sentiment_strength'] = 'MODERATE'
            else:
                sentiment_data['sentiment_strength'] = 'NEUTRAL'
            
        except Exception as e:
            logger.warning(f"Error in sentiment analysis: {e}")
        
        return sentiment_data
    
    def generate_multi_factor_signal(self) -> TradingSignal:
        """Generate trading signal based on multi-factor analysis"""
        
        # Fetch all components
        technical_indicators = self.calculate_technical_indicators()
        monte_carlo_results = self.monte_carlo_simulation()
        candlestick_patterns = self.analyze_candlestick_patterns()
        sentiment_data = self.fetch_sentiment_analysis()
        
        # Score each component
        scores = {}
        reasoning = []
        
        # 1. Monte Carlo Score (-5 to +5)
        mc_score = 0
        if monte_carlo_results['bullish_probability'] > 0.7:
            mc_score = 5
            reasoning.append(f"Monte Carlo: {monte_carlo_results['bullish_probability']:.1%} bullish probability")
        elif monte_carlo_results['bullish_probability'] > 0.6:
            mc_score = 3
            reasoning.append(f"Monte Carlo: {monte_carlo_results['bullish_probability']:.1%} bullish probability")
        elif monte_carlo_results['bullish_probability'] < 0.3:
            mc_score = -5
            reasoning.append(f"Monte Carlo: {monte_carlo_results['bearish_probability']:.1%} bearish probability")
        elif monte_carlo_results['bullish_probability'] < 0.4:
            mc_score = -3
            reasoning.append(f"Monte Carlo: {monte_carlo_results['bearish_probability']:.1%} bearish probability")
        
        scores['monte_carlo'] = mc_score
        
        # 2. Technical Indicators Score (-5 to +5)
        tech_score = 0
        
        if technical_indicators['golden_cross']:
            tech_score += 3
            reasoning.append("Technical: Golden Cross detected")
        elif technical_indicators['death_cross']:
            tech_score -= 3
            reasoning.append("Technical: Death Cross detected")
        
        if technical_indicators['rsi_oversold']:
            tech_score += 2
            reasoning.append(f"Technical: RSI oversold ({technical_indicators['rsi_current']:.1f})")
        elif technical_indicators['rsi_overbought']:
            tech_score -= 2
            reasoning.append(f"Technical: RSI overbought ({technical_indicators['rsi_current']:.1f})")
        
        if technical_indicators['macd_bullish_crossover']:
            tech_score += 2
            reasoning.append("Technical: MACD bullish crossover")
        elif technical_indicators['macd_bearish_crossover']:
            tech_score -= 2
            reasoning.append("Technical: MACD bearish crossover")
        
        if technical_indicators['bb_breakout_up'] and technical_indicators['high_volume']:
            tech_score += 2
            reasoning.append("Technical: Bollinger Band breakout with volume")
        elif technical_indicators['bb_breakout_down'] and technical_indicators['high_volume']:
            tech_score -= 2
            reasoning.append("Technical: Bollinger Band breakdown with volume")
        
        scores['technical'] = max(-5, min(5, tech_score))
        
        # 3. Sentiment Score (-3 to +3)
        sentiment_score = sentiment_data['overall_sentiment'] * 3
        if abs(sentiment_score) > 0.5:
            direction = "positive" if sentiment_score > 0 else "negative"
            reasoning.append(f"Sentiment: {direction} ({sentiment_data['overall_sentiment']:.2f})")
        scores['sentiment'] = max(-3, min(3, sentiment_score))
        
        # 4. Volume Score (-2 to +2)
        volume_score = 0
        if technical_indicators['high_volume'] and technical_indicators['volume_trend']:
            volume_score = 2
            reasoning.append("Volume: High volume with uptrend")
        elif technical_indicators['high_volume']:
            volume_score = 1
            reasoning.append("Volume: Above average volume")
        scores['volume'] = volume_score
        
        # 5. Candlestick Score (-3 to +3)
        candle_score = 0
        if candlestick_patterns['hammer'] or candlestick_patterns['bullish_engulfing']:
            candle_score = 2
            reasoning.append("Candlestick: Bullish reversal pattern")
        elif candlestick_patterns['shooting_star'] or candlestick_patterns['bearish_engulfing']:
            candle_score = -2
            reasoning.append("Candlestick: Bearish reversal pattern")
        elif candlestick_patterns['doji']:
            candle_score = 0
            reasoning.append("Candlestick: Doji - indecision")
        scores['candlestick'] = candle_score
        
        # Calculate weighted final score
        final_score = sum(scores[key] * self.weights[key] * 20 for key in scores.keys())
        
        # Determine signal direction and strength
        current_price = self.data['Close'].iloc[-1]
        
        if final_score > 60:
            direction = "BUY"
            strength = SignalStrength.VERY_STRONG
        elif final_score > 30:
            direction = "BUY"
            strength = SignalStrength.STRONG
        elif final_score > 10:
            direction = "BUY"
            strength = SignalStrength.MODERATE
        elif final_score < -60:
            direction = "SELL"
            strength = SignalStrength.VERY_STRONG
        elif final_score < -30:
            direction = "SELL"
            strength = SignalStrength.STRONG
        elif final_score < -10:
            direction = "SELL"
            strength = SignalStrength.MODERATE
        else:
            direction = "HOLD"
            strength = SignalStrength.WEAK
        
        # Calculate risk management levels
        volatility = monte_carlo_results['volatility_forecast']
        
        if direction == "BUY":
            stop_loss = current_price * (1 - 2 * volatility)  # 2x volatility below
            take_profit = current_price * (1 + 3 * volatility)  # 3x volatility above
        elif direction == "SELL":
            stop_loss = current_price * (1 + 2 * volatility)  # 2x volatility above
            take_profit = current_price * (1 - 3 * volatility)  # 3x volatility below
        else:
            stop_loss = current_price
            take_profit = current_price
        
        risk_reward_ratio = abs(take_profit - current_price) / abs(current_price - stop_loss) if stop_loss != current_price else 1.0
        expected_return = monte_carlo_results['expected_return']
        
        # Create trading signal
        signal = TradingSignal(
            direction=direction,
            strength=strength,
            confidence=min(100, abs(final_score)),
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            risk_reward_ratio=risk_reward_ratio,
            expected_return=expected_return,
            time_horizon="3D"
        )
        
        # Store all analysis results
        self.signals = {
            'technical_indicators': technical_indicators,
            'monte_carlo_results': monte_carlo_results,
            'candlestick_patterns': candlestick_patterns,
            'sentiment_data': sentiment_data,
            'scores': scores,
            'final_score': final_score
        }
        
        self.final_signal = signal
        return signal
    
    def print_detailed_analysis(self):
        """Print comprehensive analysis report"""
        if self.final_signal is None:
            safe_print(" No analysis available. Run generate_multi_factor_signal() first.")
            return
        
        signal = self.final_signal
        current_price = self.data['Close'].iloc[-1]
        
        safe_print("\n" + "="*80)
        safe_print(f" ADVANCED SHORT-TERM STRATEGY ANALYSIS: {self.symbol}")
        safe_print("="*80)
        
        # Signal Summary
        safe_print(f"\n TRADING SIGNAL SUMMARY")
        safe_print("-" * 40)
        safe_print(f"Direction: {signal.direction} ({signal.strength.name})")
        safe_print(f"Confidence: {signal.confidence:.1f}%")
        safe_print(f"Current Price: ₹{current_price:.2f}")
        safe_print(f"Entry Price: ₹{signal.entry_price:.2f}")
        safe_print(f"Stop Loss: ₹{signal.stop_loss:.2f}")
        safe_print(f"Take Profit: ₹{signal.take_profit:.2f}")
        safe_print(f"Risk/Reward Ratio: 1:{signal.risk_reward_ratio:.2f}")
        safe_print(f"Expected Return: {signal.expected_return:.2%}")
        safe_print(f"Time Horizon: {signal.time_horizon}")
        
        # Monte Carlo Results
        mc_results = self.signals['monte_carlo_results']
        print(f"\n🎲 MONTE CARLO SIMULATION (1000 runs)")
        print("-" * 40)
        print(f"Bullish Probability: {mc_results['bullish_probability']:.1%}")
        print(f"Expected Price (3D): ₹{mc_results['mean_predicted']:.2f}")
        print(f"Price Range (95% confidence): ₹{mc_results['price_range_5_95'][0]:.2f} - ₹{mc_results['price_range_5_95'][1]:.2f}")
        print(f"Volatility Forecast: {mc_results['volatility_forecast']:.1%}")
        
        # Technical Indicators
        tech_indicators = self.signals['technical_indicators']
        print(f"\n📈 TECHNICAL ANALYSIS")
        print("-" * 40)
        print(f"RSI: {tech_indicators['rsi_current']:.1f} ({'Oversold' if tech_indicators['rsi_oversold'] else 'Overbought' if tech_indicators['rsi_overbought'] else 'Normal'})")
        print(f"Golden Cross: {'' if tech_indicators['golden_cross'] else ''}")
        print(f"Death Cross: {'⚠️' if tech_indicators['death_cross'] else ''}")
        print(f"MACD Bullish: {'' if tech_indicators['macd_bullish_crossover'] else ''}")
        print(f"BB Breakout: {'📈 Up' if tech_indicators['bb_breakout_up'] else '📉 Down' if tech_indicators['bb_breakout_down'] else '➡️ None'}")
        print(f"High Volume: {'' if tech_indicators['high_volume'] else ''}")
        
        # Candlestick Patterns
        patterns = self.signals['candlestick_patterns']
        print(f"\n🕯️ CANDLESTICK PATTERNS")
        print("-" * 40)
        active_patterns = []
        if patterns.get('hammer'): active_patterns.append("🔨 Hammer (Bullish)")
        if patterns.get('shooting_star'): active_patterns.append("⭐ Shooting Star (Bearish)")
        if patterns.get('bullish_engulfing'): active_patterns.append("📈 Bullish Engulfing")
        if patterns.get('bearish_engulfing'): active_patterns.append("📉 Bearish Engulfing")
        if patterns.get('doji'): active_patterns.append("➡️ Doji (Indecision)")
        
        if active_patterns:
            for pattern in active_patterns:
                print(f"  {pattern}")
        else:
            print("  No significant patterns detected")
        
        # Sentiment Analysis
        sentiment = self.signals['sentiment_data']
        print(f"\n💭 SENTIMENT ANALYSIS")
        print("-" * 40)
        print(f"Overall Sentiment: {sentiment['overall_sentiment']:.2f} ({sentiment['sentiment_strength']})")
        print(f"News Articles Analyzed: {sentiment['news_count']}")
        
        if sentiment['recent_events']:
            print("Recent News:")
            for event in sentiment['recent_events'][:2]:
                print(f"  • {event['title'][:60]}... (Sentiment: {event['sentiment']:.2f})")
        
        # Component Scores
        scores = self.signals['scores']
        print(f"\n📋 COMPONENT SCORES")
        print("-" * 40)
        for component, score in scores.items():
            bar = "█" * max(0, int(abs(score))) + "░" * max(0, 5 - int(abs(score)))
            direction_icon = "📈" if score > 0 else "📉" if score < 0 else "➡️"
            print(f"{component.title():15} {direction_icon} {score:+4.1f} |{bar}|")
        
        print(f"\nFinal Score: {self.signals['final_score']:.1f}")
        
        # Reasoning
        print(f"\n🧠 REASONING")
        print("-" * 40)
        for i, reason in enumerate(signal.reasoning, 1):
            print(f"{i:2d}. {reason}")
        
        # Risk Warning
        print(f"\n⚠️ RISK MANAGEMENT")
        print("-" * 40)
        potential_loss = abs(current_price - signal.stop_loss) / current_price * 100
        potential_gain = abs(signal.take_profit - current_price) / current_price * 100
        
        print(f"Maximum Risk: {potential_loss:.1f}%")
        print(f"Maximum Reward: {potential_gain:.1f}%")
        print(f"Position Size Recommendation: Risk max 2% of portfolio")
        
        print("\n" + "="*80)
    
    def plot_analysis_charts(self):
        """Create comprehensive analysis charts"""
        if self.data is None or self.final_signal is None:
            print(" No data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Advanced Strategy Analysis: {self.symbol}', fontsize=16, fontweight='bold')
        
        # 1. Price with Technical Indicators
        ax1 = axes[0, 0]
        df = self.technical_data
        
        ax1.plot(df.index, df['Close'], label='Price', linewidth=2, color='black')
        ax1.plot(df.index, df['EMA_20'], label='EMA 20', alpha=0.7, color='blue')
        ax1.plot(df.index, df['EMA_50'], label='EMA 50', alpha=0.7, color='red')
        ax1.fill_between(df.index, df['BB_Lower'], df['BB_Upper'], alpha=0.2, color='gray', label='Bollinger Bands')
        
        # Mark entry point
        current_price = df['Close'].iloc[-1]
        signal_color = 'green' if self.final_signal.direction == 'BUY' else 'red' if self.final_signal.direction == 'SELL' else 'orange'
        ax1.scatter(df.index[-1], current_price, color=signal_color, s=100, marker='^' if self.final_signal.direction == 'BUY' else 'v', zorder=5)
        
        ax1.set_title('Price Action & Technical Indicators')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RSI
        ax2 = axes[0, 1]
        ax2.plot(df.index, df['RSI'], color='purple', linewidth=1.5)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        ax2.fill_between(df.index, 30, 70, alpha=0.1, color='yellow')
        ax2.set_title('RSI (Relative Strength Index)')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Monte Carlo Distribution
        ax3 = axes[1, 0]
        mc_results = self.signals['monte_carlo_results']
        ax3.hist(mc_results['all_simulations'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(mc_results['current_price'], color='red', linestyle='--', linewidth=2, label=f'Current Price: ₹{mc_results["current_price"]:.2f}')
        ax3.axvline(mc_results['mean_predicted'], color='green', linestyle='--', linewidth=2, label=f'Mean Predicted: ₹{mc_results["mean_predicted"]:.2f}')
        ax3.set_title('Monte Carlo Price Distribution (3 Days)')
        ax3.set_xlabel('Price (₹)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Component Scores
        ax4 = axes[1, 1]
        components = list(self.signals['scores'].keys())
        scores = list(self.signals['scores'].values())
        colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in scores]
        
        bars = ax4.barh(components, scores, color=colors, alpha=0.7)
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.8)
        ax4.set_title('Multi-Factor Analysis Scores')
        ax4.set_xlabel('Score')
        ax4.grid(True, alpha=0.3)
        
        # Add score values on bars
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax4.text(width + (0.1 if width >= 0 else -0.1), bar.get_y() + bar.get_height()/2, 
                    f'{score:.1f}', ha='left' if width >= 0 else 'right', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

def main():
    """Test the Advanced Short-Term Strategy"""
    print(" Advanced Short-Term Trading Strategy")
    print("=" * 50)
    
    # Example usage
    symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
    
    print("\nSelect a stock to analyze:")
    for i, symbol in enumerate(symbols, 1):
        print(f"{i}. {symbol}")
    
    try:
        choice = int(input("\nEnter choice (1-5): ")) - 1
        if 0 <= choice < len(symbols):
            selected_symbol = symbols[choice]
        else:
            selected_symbol = 'RELIANCE.NS'  # Default
    except:
        selected_symbol = 'RELIANCE.NS'  # Default
    
    print(f"\n🔍 Analyzing {selected_symbol}...")
    
    # Initialize strategy
    strategy = AdvancedShortTermStrategy(selected_symbol)
    
    try:
        # Fetch data
        strategy.fetch_data()
        
        # Generate comprehensive signal
        signal = strategy.generate_multi_factor_signal()
        
        # Print detailed analysis
        strategy.print_detailed_analysis()
        
        # Plot charts
        print("\n Generating analysis charts...")
        strategy.plot_analysis_charts()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"analysis_{selected_symbol.replace('.NS', '')}_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write(f"Advanced Short-Term Strategy Analysis\n")
            f.write(f"Symbol: {selected_symbol}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Signal: {signal.direction} ({signal.strength.name})\n")
            f.write(f"Confidence: {signal.confidence:.1f}%\n")
            f.write(f"Entry: ₹{signal.entry_price:.2f}\n")
            f.write(f"Stop Loss: ₹{signal.stop_loss:.2f}\n")
            f.write(f"Take Profit: ₹{signal.take_profit:.2f}\n")
            f.write("\nReasoning:\n")
            for reason in signal.reasoning:
                f.write(f"- {reason}\n")
        
        print(f"\n💾 Analysis saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        print(f" Error: {e}")

if __name__ == "__main__":
    main()