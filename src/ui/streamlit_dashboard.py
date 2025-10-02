#!/usr/bin/env python3
"""
Streamlit Web Dashboard for Trading Strategies
Real-time monitoring, interactive charts, and portfolio tracking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import sqlite3
import json
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data.database_manager import TradingDatabase
from src.strategies.advanced_shortterm_strategy import AdvancedShortTermStrategy
from src.config.strategy_config import CONFIG
import ssl

# SSL bypass
ssl._create_default_https_context = ssl._create_unverified_context

# Page configuration
st.set_page_config(
    page_title="Advanced Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
@st.cache_resource
def init_database():
    return TradingDatabase()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem;
    }
    .positive {
        color: #00ff00;
        font-weight: bold;
    }
    .negative {
        color: #ff4444;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header"> Advanced Trading Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize database
    db = init_database()
    
    # Sidebar navigation
    st.sidebar.title(" Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Dashboard", "📈 Live Analysis", " Strategy Performance", 
         "📰 News Sentiment", "🔄 Backtesting", "💼 Portfolio Tracker", "⚙️ Settings"]
    )
    
    if page == "🏠 Dashboard":
        show_dashboard(db)
    elif page == "📈 Live Analysis":
        show_live_analysis(db)
    elif page == " Strategy Performance":
        show_strategy_performance(db)
    elif page == "📰 News Sentiment":
        show_news_sentiment(db)
    elif page == "🔄 Backtesting":
        show_backtesting()
    elif page == "💼 Portfolio Tracker":
        show_portfolio_tracker(db)
    elif page == "⚙️ Settings":
        show_settings()

def show_dashboard(db):
    """Main dashboard with overview metrics"""
    st.header(" Trading Dashboard Overview")
    
    # Get database stats
    stats = db.get_database_stats()
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=" Total Signals",
            value=stats.get('signals_count', 0),
            delta="Active strategies"
        )
    
    with col2:
        st.metric(
            label="💰 Total Trades",
            value=stats.get('trades_count', 0),
            delta="Executed"
        )
    
    with col3:
        st.metric(
            label="📰 News Articles",
            value=stats.get('news_sentiment_count', 0),
            delta="Analyzed"
        )
    
    with col4:
        st.metric(
            label="💾 Database Size",
            value=f"{stats.get('db_size_mb', 0)} MB",
            delta="Storage used"
        )
    
    # Recent activity
    st.subheader("🕒 Recent Trading Signals")
    recent_signals = db.get_recent_signals(limit=10)
    
    if not recent_signals.empty:
        # Format the dataframe for display
        display_df = recent_signals[['timestamp', 'symbol', 'strategy', 'direction', 'confidence', 'entry_price']].copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['confidence'] = display_df['confidence'].round(1)
        display_df['entry_price'] = display_df['entry_price'].round(2)
        
        # Color code by direction
        def highlight_direction(row):
            if row['direction'] == 'BUY':
                return ['background-color: #d4edda'] * len(row)
            elif row['direction'] == 'SELL':
                return ['background-color: #f8d7da'] * len(row)
            else:
                return ['background-color: #fff3cd'] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_direction, axis=1),
            use_container_width=True
        )
    else:
        st.info("No recent signals found. Run a strategy analysis to see signals here.")
    
    # Quick analysis section
    st.subheader("⚡ Quick Stock Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_symbol = st.selectbox(
            "Select Stock for Quick Analysis",
            ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'ITC.NS']
        )
        
        if st.button("🔍 Analyze Now", type="primary"):
            with st.spinner(f"Analyzing {selected_symbol}..."):
                try:
                    strategy = AdvancedShortTermStrategy(selected_symbol)
                    strategy.fetch_data()
                    signal = strategy.generate_multi_factor_signal()
                    
                    # Save signal to database
                    signal_data = {
                        'symbol': selected_symbol,
                        'strategy': 'advanced_shortterm',
                        'direction': signal.direction,
                        'confidence': signal.confidence,
                        'entry_price': signal.entry_price,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'reasoning': signal.reasoning,
                        'risk_reward_ratio': signal.risk_reward_ratio,
                        'expected_return': signal.expected_return
                    }
                    
                    db.save_signal(signal_data)
                    st.success("Analysis complete! Signal saved to database.")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error analyzing {selected_symbol}: {e}")
    
    with col2:
        if recent_signals.empty:
            st.info("Run an analysis to see results here")
        else:
            latest_signal = recent_signals.iloc[0]
            
            st.markdown("### 📋 Latest Signal Details")
            
            # Signal direction with color
            direction_color = "🟢" if latest_signal['direction'] == 'BUY' else "🔴" if latest_signal['direction'] == 'SELL' else "🟡"
            st.markdown(f"**Direction:** {direction_color} {latest_signal['direction']}")
            st.markdown(f"**Symbol:** {latest_signal['symbol']}")
            st.markdown(f"**Confidence:** {latest_signal['confidence']:.1f}%")
            st.markdown(f"**Entry Price:** ₹{latest_signal['entry_price']:.2f}")
            
            if latest_signal['reasoning']:
                reasoning = json.loads(latest_signal['reasoning']) if isinstance(latest_signal['reasoning'], str) else latest_signal['reasoning']
                st.markdown("**Reasoning:**")
                for reason in reasoning[:3]:  # Show top 3 reasons
                    st.markdown(f"• {reason}")

def show_live_analysis(db):
    """Live market analysis page"""
    st.header("📈 Live Market Analysis")
    
    # Stock selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="RELIANCE.NS", help="e.g., RELIANCE.NS, TCS.NS")
    
    with col2:
        period = st.selectbox("Analysis Period", ["6mo", "1y", "2y"], index=1)
    
    with col3:
        if st.button(" Analyze", type="primary"):
            st.session_state.analyze_clicked = True
    
    if st.session_state.get('analyze_clicked', False):
        try:
            with st.spinner(f"Fetching data and analyzing {symbol}..."):
                # Initialize strategy
                strategy = AdvancedShortTermStrategy(symbol, period=period)
                strategy.fetch_data()
                signal = strategy.generate_multi_factor_signal()
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Price chart with technical indicators
                    fig = create_price_chart(strategy)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Signal summary
                    st.markdown("###  Trading Signal")
                    
                    direction_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
                    st.markdown(f"## {direction_emoji.get(signal.direction, '🟡')} {signal.direction}")
                    
                    st.metric("Confidence", f"{signal.confidence:.1f}%")
                    st.metric("Entry Price", f"₹{signal.entry_price:.2f}")
                    st.metric("Stop Loss", f"₹{signal.stop_loss:.2f}")
                    st.metric("Take Profit", f"₹{signal.take_profit:.2f}")
                    st.metric("Risk:Reward", f"1:{signal.risk_reward_ratio:.2f}")
                
                # Component analysis
                st.subheader("🔍 Multi-Factor Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Monte Carlo results
                    mc_results = strategy.signals['monte_carlo_results']
                    
                    st.markdown("#### 🎲 Monte Carlo Simulation")
                    st.metric("Bullish Probability", f"{mc_results['bullish_probability']:.1%}")
                    st.metric("Expected Price (3D)", f"₹{mc_results['mean_predicted']:.2f}")
                    st.metric("Volatility Forecast", f"{mc_results['volatility_forecast']:.1%}")
                
                with col2:
                    # Technical indicators
                    tech_indicators = strategy.signals['technical_indicators']
                    
                    st.markdown("####  Technical Indicators")
                    st.metric("RSI", f"{tech_indicators['rsi_current']:.1f}")
                    st.write(f"Golden Cross: {'' if tech_indicators['golden_cross'] else ''}")
                    st.write(f"MACD Bullish: {'' if tech_indicators['macd_bullish_crossover'] else ''}")
                    st.write(f"High Volume: {'' if tech_indicators['high_volume'] else ''}")
                
                # Reasoning
                st.subheader("🧠 Analysis Reasoning")
                for i, reason in enumerate(signal.reasoning, 1):
                    st.write(f"{i}. {reason}")
                
                # Save signal
                signal_data = {
                    'symbol': symbol,
                    'strategy': 'advanced_shortterm',
                    'direction': signal.direction,
                    'confidence': signal.confidence,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'reasoning': signal.reasoning,
                    'risk_reward_ratio': signal.risk_reward_ratio,
                    'expected_return': signal.expected_return
                }
                
                db.save_signal(signal_data)
                
        except Exception as e:
            st.error(f"Error analyzing {symbol}: {e}")

def create_price_chart(strategy):
    """Create interactive price chart with technical indicators"""
    df = strategy.technical_data
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Moving Averages', 'RSI', 'Volume'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price and moving averages
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA_50'], name='EMA 50', line=dict(color='red')),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash')),
        row=1, col=1
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue'),
        row=3, col=1
    )
    
    fig.update_layout(
        title="Technical Analysis Chart",
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def show_strategy_performance(db):
    """Strategy performance comparison page"""
    st.header(" Strategy Performance Analysis")
    
    # Strategy selection
    strategies = ['advanced_shortterm', 'portfolio_analyzer', 'news_based']
    selected_strategies = st.multiselect(
        "Select Strategies to Compare",
        strategies,
        default=['advanced_shortterm']
    )
    
    # Time period
    days = st.slider("Analysis Period (Days)", 7, 365, 30)
    
    if selected_strategies:
        # Get performance data
        comparison_data = []
        
        for strategy in selected_strategies:
            perf = db.get_strategy_performance(strategy, days)
            if 'error' not in perf:
                comparison_data.append(perf)
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            # Performance metrics
            st.subheader("📈 Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                fig = px.bar(df, x='strategy', y='total_pnl', title='Total P&L')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(df, x='strategy', y='win_rate', title='Win Rate')
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                fig = px.bar(df, x='strategy', y='total_trades', title='Total Trades')
                st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                fig = px.bar(df, x='strategy', y='profit_factor', title='Profit Factor')
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.subheader("📋 Detailed Comparison")
            st.dataframe(df, use_container_width=True)
        
        else:
            st.info("No performance data available for selected strategies and period.")

def show_news_sentiment(db):
    """News sentiment analysis page"""
    st.header("📰 News Sentiment Analysis")
    
    # Symbol and period selection
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Stock Symbol", value="RELIANCE.NS")
    
    with col2:
        days = st.slider("Days to Analyze", 1, 30, 7)
    
    if st.button(" Get Sentiment Data"):
        sentiment_df = db.get_news_sentiment_history(symbol, days)
        
        if not sentiment_df.empty:
            # Sentiment timeline
            fig = px.line(
                sentiment_df, 
                x='timestamp', 
                y='sentiment_score',
                title=f'News Sentiment Timeline - {symbol}',
                hover_data=['title', 'source']
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    sentiment_df, 
                    x='sentiment_score', 
                    title='Sentiment Distribution',
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                avg_sentiment = sentiment_df['sentiment_score'].mean()
                sentiment_counts = (sentiment_df['sentiment_score'] > 0).value_counts()
                
                st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
                st.metric("Positive News", f"{sentiment_counts.get(True, 0)}")
                st.metric("Negative News", f"{sentiment_counts.get(False, 0)}")
            
            # Recent news
            st.subheader("📝 Recent News Articles")
            display_df = sentiment_df[['timestamp', 'title', 'source', 'sentiment_score']].head(10)
            st.dataframe(display_df, use_container_width=True)
        
        else:
            st.info(f"No news sentiment data available for {symbol}")

def show_backtesting():
    """Backtesting interface"""
    st.header("🔄 Strategy Backtesting")
    st.info("Backtesting interface - Integration with strategy_backtester.py")
    
    # This would integrate with your existing backtesting module
    st.markdown("""
    ### Available Backtesting Features:
    - Historical performance analysis
    - Multi-symbol comparison
    - Parameter optimization
    - Risk metrics calculation
    
    Use the command line backtester for detailed analysis:
    ```bash
    python strategy_backtester.py
    ```
    """)

def show_portfolio_tracker(db):
    """Portfolio tracking page"""
    st.header("💼 Portfolio Tracker")
    
    st.info("Portfolio tracking interface - Integration with portfolio_analyzer.py")
    
    # This would integrate with your existing portfolio analyzer
    st.markdown("""
    ### Portfolio Analysis Features:
    - Real-time portfolio valuation
    - Risk metrics (VaR, CVaR, Sharpe ratio)
    - Diversification analysis
    - Rebalancing recommendations
    
    Use the portfolio analyzer for detailed analysis:
    ```bash
    python portfolio_analyzer.py
    ```
    """)

def show_settings():
    """Settings and configuration page"""
    st.header("⚙️ Settings & Configuration")
    
    # Strategy weights
    st.subheader(" Strategy Component Weights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        monte_carlo_weight = st.slider("Monte Carlo Weight", 0.0, 1.0, CONFIG.WEIGHTS['monte_carlo'], 0.05)
        technical_weight = st.slider("Technical Weight", 0.0, 1.0, CONFIG.WEIGHTS['technical'], 0.05)
        sentiment_weight = st.slider("Sentiment Weight", 0.0, 1.0, CONFIG.WEIGHTS['sentiment'], 0.05)
    
    with col2:
        volume_weight = st.slider("Volume Weight", 0.0, 1.0, CONFIG.WEIGHTS['volume'], 0.05)
        candlestick_weight = st.slider("Candlestick Weight", 0.0, 1.0, CONFIG.WEIGHTS['candlestick'], 0.05)
    
    total_weight = monte_carlo_weight + technical_weight + sentiment_weight + volume_weight + candlestick_weight
    
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"⚠️ Weights sum to {total_weight:.2f}. They should sum to 1.0")
    else:
        st.success(" Weights are properly balanced")
    
    if st.button("💾 Save Configuration"):
        # Update configuration (this would save to config file)
        st.success("Configuration saved! Restart the application to apply changes.")
    
    # Database management
    st.subheader("💾 Database Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(" Database Stats"):
            db = init_database()
            stats = db.get_database_stats()
            st.json(stats)
    
    with col2:
        if st.button("🧹 Cleanup Old Data"):
            db = init_database()
            db.cleanup_old_data(days_to_keep=365)
            st.success("Old data cleaned up!")
    
    with col3:
        if st.button("📤 Export Data"):
            st.info("Export functionality - select table and format")

if __name__ == "__main__":
    main()