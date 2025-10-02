#!/usr/bin/env python3
"""
Simplified Real-Time Market Data Feed
Synchronous version that avoids asyncio complications.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import threading
import time
import json
import requests
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, asdict
from src.data.database_manager import TradingDatabase

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    bid: float = 0.0
    ask: float = 0.0
    change: float = 0.0
    change_percent: float = 0.0

@dataclass
class NewsData:
    """News data structure"""
    title: str
    summary: str
    source: str
    timestamp: datetime
    symbols: List[str]
    sentiment_score: float = 0.0
    url: str = ""

class SimpleDataFeed:
    """
    Simplified real-time market data and news feed manager
    Pure synchronous implementation without WebSocket complexity
    """
    
    def __init__(self, symbols: List[str], update_interval: int = 30):
        self.symbols = symbols
        self.update_interval = update_interval
        self.running = False
        self.subscribers = []
        self.latest_data = {}
        self.db = TradingDatabase()
        
        logger.info(f"Initialized simple data feed for {len(symbols)} symbols")
    
    def subscribe(self, callback: Callable):
        """Subscribe to data updates"""
        self.subscribers.append(callback)
        logger.info(f"New subscriber added. Total: {len(self.subscribers)}")
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from data updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info(f"Subscriber removed. Total: {len(self.subscribers)}")
    
    def start_feed(self):
        """Start the real-time data feed"""
        if self.running:
            logger.warning("Feed is already running")
            return
        
        self.running = True
        
        # Start data collection thread
        data_thread = threading.Thread(target=self._data_collection_loop, daemon=True)
        data_thread.start()
        
        # Start news collection thread
        news_thread = threading.Thread(target=self._news_collection_loop, daemon=True)
        news_thread.start()
        
        logger.info("Simple data feed started")
    
    def stop_feed(self):
        """Stop the real-time data feed"""
        self.running = False
        logger.info("Simple data feed stopped")
    
    def _data_collection_loop(self):
        """Main data collection loop"""
        while self.running:
            try:
                # Fetch data for all symbols
                for symbol in self.symbols:
                    if not self.running:
                        break
                    
                    market_data = self._fetch_market_data(symbol)
                    if market_data:
                        self.latest_data[symbol] = market_data
                        self._notify_subscribers('market_data', market_data)
                    
                    # Small delay between symbols
                    time.sleep(1)
                
                # Wait for next update cycle
                if self.running:
                    time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                if self.running:
                    time.sleep(self.update_interval)
    
    def _news_collection_loop(self):
        """News collection loop (runs less frequently)"""
        while self.running:
            try:
                # Collect news every 5 minutes
                news_items = self._fetch_news_data()
                for news in news_items:
                    if not self.running:
                        break
                    self._notify_subscribers('news_data', news)
                
                # Wait 5 minutes for next news update
                for _ in range(300):  # 300 seconds = 5 minutes
                    if not self.running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in news collection loop: {e}")
                if self.running:
                    time.sleep(300)  # Wait 5 minutes before retry
    
    def _fetch_market_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch current market data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current data (last 2 days to calculate change)
            hist = ticker.history(period="2d", interval="1m")
            if hist.empty:
                return None
            
            current = hist.iloc[-1]
            previous = hist.iloc[-2] if len(hist) > 1 else current
            
            # Calculate changes
            change = current['Close'] - previous['Close']
            change_percent = (change / previous['Close']) * 100 if previous['Close'] != 0 else 0
            
            # Get real-time info (if available)
            try:
                info = ticker.info
                bid = info.get('bid', current['Close'])
                ask = info.get('ask', current['Close'])
            except:
                bid = current['Close']
                ask = current['Close']
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=datetime.now(),
                price=current['Close'],
                volume=int(current['Volume']),
                bid=bid,
                ask=ask,
                change=change,
                change_percent=change_percent
            )
            
            # Cache data in database
            self._cache_market_data(market_data)
            
            return market_data
            
        except Exception as e:
            logger.warning(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _fetch_news_data(self) -> List[NewsData]:
        """Fetch latest news data"""
        news_items = []
        
        try:
            import feedparser
            from textblob import TextBlob
            
            news_sources = [
                "https://feeds.bloomberg.com/markets/news.rss",
                "https://www.investing.com/rss/news.rss",
                "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
            ]
            
            for source_url in news_sources:
                try:
                    feed = feedparser.parse(source_url)
                    source_name = feed.feed.get('title', 'Unknown')
                    
                    for entry in feed.entries[:3]:  # Latest 3 items per source
                        if not self.running:
                            break
                        
                        # Extract symbols mentioned in title/summary
                        text = f"{entry.title} {entry.get('summary', '')}"
                        symbols_mentioned = []
                        
                        for symbol in self.symbols:
                            symbol_base = symbol.replace('.NS', '').replace('.BO', '')
                            if symbol_base.lower() in text.lower():
                                symbols_mentioned.append(symbol)
                        
                        # Calculate sentiment
                        blob = TextBlob(text)
                        sentiment_score = blob.sentiment.polarity
                        
                        news_data = NewsData(
                            title=entry.title,
                            summary=entry.get('summary', ''),
                            source=source_name,
                            timestamp=datetime(
                                *entry.get('published_parsed', time.gmtime())[:6]
                            ),
                            symbols=symbols_mentioned,
                            sentiment_score=sentiment_score,
                            url=entry.get('link', '')
                        )
                        
                        news_items.append(news_data)
                        
                        # Save to database
                        if symbols_mentioned:
                            for symbol in symbols_mentioned:
                                try:
                                    self.db.save_news_sentiment({
                                        'symbol': symbol,
                                        'title': news_data.title,
                                        'source': news_data.source,
                                        'sentiment_score': news_data.sentiment_score,
                                        'url': news_data.url
                                    })
                                except Exception as e:
                                    logger.debug(f"Error saving news sentiment: {e}")
                
                except Exception as e:
                    logger.warning(f"Error fetching from {source_url}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in news data collection: {e}")
        
        return news_items
    
    def _cache_market_data(self, market_data: MarketData):
        """Cache market data in database"""
        try:
            import sqlite3
            with sqlite3.connect(self.db.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO market_data_cache 
                    (symbol, date, open_price, high_price, low_price, close_price, volume, adj_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    market_data.symbol,
                    market_data.timestamp.date(),
                    market_data.price,  # Using current price as OHLC for simplicity
                    market_data.price,
                    market_data.price,
                    market_data.price,
                    market_data.volume,
                    market_data.price
                ))
                conn.commit()
        except Exception as e:
            logger.debug(f"Error caching market data: {e}")
    
    def _notify_subscribers(self, data_type: str, data):
        """Notify all subscribers of new data"""
        for callback in self.subscribers:
            try:
                callback(data_type, data)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def get_latest_data(self, symbol: str = None) -> Dict:
        """Get latest data for symbol(s)"""
        if symbol:
            return self.latest_data.get(symbol)
        return self.latest_data.copy()

class MarketDataSubscriber:
    """Example subscriber for market data"""
    
    def __init__(self, name: str):
        self.name = name
    
    def handle_data(self, data_type: str, data):
        """Handle incoming data updates"""
        if data_type == 'market_data':
            logger.info(f"{self.name}: {data.symbol} - ₹{data.price:.2f} ({data.change_percent:+.2f}%)")
        elif data_type == 'news_data':
            if data.symbols:
                logger.info(f"{self.name}: News for {data.symbols}: {data.title[:50]}... (Sentiment: {data.sentiment_score:.2f})")

def main():
    """Test the simple data feed"""
    # Initialize with popular Indian stocks
    symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
    
    # Create data feed
    feed = SimpleDataFeed(symbols, update_interval=30)  # Update every 30 seconds
    
    # Create subscriber
    subscriber = MarketDataSubscriber("Test Subscriber")
    feed.subscribe(subscriber.handle_data)
    
    # Start data feed
    feed.start_feed()
    
    print(" Simple data feed started!")
    print(f" Tracking: {', '.join(symbols)}")
    print("📰 News updates every 5 minutes")
    print("💹 Price updates every 30 seconds")
    print("\nPress Ctrl+C to stop...")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping data feed...")
        feed.stop_feed()
        print(" Data feed stopped.")

if __name__ == "__main__":
    main()