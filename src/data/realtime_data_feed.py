#!/usr/bin/env python3
"""
Real-Time Market Data Feed
WebSocket connections for live price updates and news feeds.
"""

import asyncio
import websockets
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import threading
import time
from typing import Dict, List, Callable, Optional
import requests
from dataclasses import dataclass, asdict
import sqlite3
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

class RealTimeDataFeed:
    """
    Real-time market data and news feed manager
    Simulates live data using yfinance and news APIs
    """
    
    def __init__(self, symbols: List[str], update_interval: int = 5):
        self.symbols = symbols
        self.update_interval = update_interval
        self.running = False
        self.subscribers = []
        self.latest_data = {}
        self.db = TradingDatabase()
        
        # WebSocket clients
        self.websocket_clients = set()
        
        logger.info(f"Initialized real-time feed for {len(symbols)} symbols")
    
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
        
        logger.info("Real-time data feed started")
    
    def stop_feed(self):
        """Stop the real-time data feed"""
        self.running = False
        logger.info("Real-time data feed stopped")
    
    def _data_collection_loop(self):
        """Main data collection loop"""
        while self.running:
            try:
                # Fetch data for all symbols
                for symbol in self.symbols:
                    market_data = self._fetch_market_data(symbol)
                    if market_data:
                        self.latest_data[symbol] = market_data
                        self._notify_subscribers('market_data', market_data)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                time.sleep(self.update_interval)
    
    def _news_collection_loop(self):
        """News collection loop (runs less frequently)"""
        while self.running:
            try:
                # Collect news every 5 minutes
                news_items = self._fetch_news_data()
                for news in news_items:
                    self._notify_subscribers('news_data', news)
                
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in news collection loop: {e}")
                time.sleep(300)
    
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
            info = ticker.info
            bid = info.get('bid', current['Close'])
            ask = info.get('ask', current['Close'])
            
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
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _fetch_news_data(self) -> List[NewsData]:
        """Fetch latest news data"""
        news_items = []
        
        try:
            # RSS news sources
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
                    
                    for entry in feed.entries[:5]:  # Latest 5 items
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
                                self.db.save_news_sentiment({
                                    'symbol': symbol,
                                    'title': news_data.title,
                                    'source': news_data.source,
                                    'sentiment_score': news_data.sentiment_score,
                                    'url': news_data.url
                                })
                
                except Exception as e:
                    logger.warning(f"Error fetching from {source_url}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in news data collection: {e}")
        
        return news_items
    
    def _cache_market_data(self, market_data: MarketData):
        """Cache market data in database"""
        try:
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
            logger.warning(f"Error caching market data: {e}")
    
    def _notify_subscribers(self, data_type: str, data):
        """Notify all subscribers of new data"""
        for callback in self.subscribers:
            try:
                callback(data_type, data)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
        
        # Notify WebSocket clients (safely handle event loop)
        self._schedule_websocket_notification(data_type, data)
    
    def _schedule_websocket_notification(self, data_type: str, data):
        """Schedule WebSocket notification safely"""
        if not self.websocket_clients:
            return
        
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # Create task in the running loop
            loop.create_task(self._notify_websocket_clients(data_type, data))
        except RuntimeError:
            # No running event loop, create a new one in a thread
            def run_notification():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self._notify_websocket_clients(data_type, data))
                    loop.close()
                except Exception as e:
                    logger.warning(f"Error in WebSocket notification: {e}")
            
            # Run in a separate thread to avoid blocking
            threading.Thread(target=run_notification, daemon=True).start()
    
    async def _notify_websocket_clients(self, data_type: str, data):
        """Notify WebSocket clients"""
        if self.websocket_clients:
            message = {
                'type': data_type,
                'data': asdict(data) if hasattr(data, '__dict__') else data,
                'timestamp': datetime.now().isoformat()
            }
            
            # Convert datetime objects to strings
            if isinstance(data, (MarketData, NewsData)):
                message['data']['timestamp'] = data.timestamp.isoformat()
            
            message_json = json.dumps(message, default=str)
            
            # Send to all connected clients
            disconnected_clients = set()
            for client in self.websocket_clients:
                try:
                    await client.send(message_json)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
                except Exception as e:
                    logger.error(f"Error sending to WebSocket client: {e}")
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected_clients
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections"""
        self.websocket_clients.add(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.websocket_clients)}")
        
        try:
            # Send current data to new client
            for symbol, data in self.latest_data.items():
                message = {
                    'type': 'market_data',
                    'data': asdict(data),
                    'timestamp': datetime.now().isoformat()
                }
                await websocket.send(json.dumps(message, default=str))
            
            # Keep connection alive
            await websocket.wait_closed()
        
        except websockets.exceptions.ConnectionClosed:
            pass
        
        finally:
            self.websocket_clients.discard(websocket)
            logger.info(f"WebSocket client disconnected. Total: {len(self.websocket_clients)}")
    
    def get_latest_data(self, symbol: str = None) -> Dict:
        """Get latest data for symbol(s)"""
        if symbol:
            return self.latest_data.get(symbol)
        return self.latest_data
    
    def start_websocket_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket server for real-time data streaming"""
        async def server():
            try:
                logger.info(f"Starting WebSocket server on ws://{host}:{port}")
                await websockets.serve(self.websocket_handler, host, port)
            except Exception as e:
                logger.error(f"Error starting WebSocket server: {e}")
        
        # Run server in separate thread
        def run_server():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(server())
                loop.run_forever()
            except Exception as e:
                logger.error(f"Error in WebSocket server thread: {e}")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        logger.info("WebSocket server thread started")

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
    """Test the real-time data feed"""
    # Initialize with popular Indian stocks
    symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
    
    # Create data feed
    feed = RealTimeDataFeed(symbols, update_interval=10)  # Update every 10 seconds
    
    # Create subscriber
    subscriber = MarketDataSubscriber("Test Subscriber")
    feed.subscribe(subscriber.handle_data)
    
    # Start WebSocket server
    feed.start_websocket_server()
    
    # Start data feed
    feed.start_feed()
    
    print(" Real-time data feed started!")
    print(f" Tracking: {', '.join(symbols)}")
    print("🌐 WebSocket server: ws://localhost:8765")
    print("📰 News updates every 5 minutes")
    print("💹 Price updates every 10 seconds")
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