#!/usr/bin/env python3
"""
Database Manager for Trading Strategies
Handles data persistence, historical tracking, and performance analytics.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import hashlib
import sys

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
            text = str(text).replace("✅", "[OK]").replace("", "[ERROR]").replace("💾", "[DATA]")
            print(text, end=end, flush=flush)
    
    def format_text(text):
        return text

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingDatabase:
    """
    Comprehensive database manager for trading system
    Stores signals, trades, backtest results, and performance metrics
    """
    
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with all required tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Signals table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL,
                    entry_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    reasoning TEXT,
                    risk_reward_ratio REAL,
                    expected_return REAL,
                    signal_data TEXT  -- JSON data
                )
            """)
            
            # Trades table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_date DATETIME,
                    exit_date DATETIME,
                    entry_price REAL,
                    exit_price REAL,
                    quantity INTEGER,
                    pnl REAL,
                    return_pct REAL,
                    close_reason TEXT,
                    commission REAL DEFAULT 0,
                    FOREIGN KEY (signal_id) REFERENCES signals (id)
                )
            """)
            
            # Portfolio performance table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    strategy TEXT NOT NULL,
                    total_value REAL,
                    daily_return REAL,
                    cumulative_return REAL,
                    drawdown REAL,
                    portfolio_data TEXT  -- JSON data
                )
            """)
            
            # Backtest results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    strategy TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    period_start DATE,
                    period_end DATE,
                    total_return REAL,
                    annualized_return REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    profit_factor REAL,
                    config_hash TEXT,  -- Configuration hash
                    detailed_results TEXT  -- JSON data
                )
            """)
            
            # News sentiment tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    title TEXT,
                    source TEXT,
                    sentiment_score REAL,
                    confidence REAL,
                    url TEXT,
                    content_hash TEXT UNIQUE
                )
            """)
            
            # Market data cache
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume INTEGER,
                    adj_close REAL,
                    UNIQUE(symbol, date)
                )
            """)
            
            # Strategy performance metrics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS strategy_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    strategy TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    period_days INTEGER,
                    additional_data TEXT  -- JSON data
                )
            """)
            
            conn.commit()
        
        logger.info(f"Database initialized: {self.db_path}")
    
    def save_signal(self, signal_data: Dict) -> int:
        """Save trading signal to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO signals (
                    symbol, strategy, direction, confidence, entry_price,
                    stop_loss, take_profit, reasoning, risk_reward_ratio,
                    expected_return, signal_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_data['symbol'],
                signal_data['strategy'],
                signal_data['direction'],
                signal_data['confidence'],
                signal_data['entry_price'],
                signal_data['stop_loss'],
                signal_data['take_profit'],
                json.dumps(signal_data.get('reasoning', [])),
                signal_data['risk_reward_ratio'],
                signal_data['expected_return'],
                json.dumps(signal_data)
            ))
            signal_id = cursor.lastrowid
            conn.commit()
        
        logger.info(f"Signal saved with ID: {signal_id}")
        return signal_id
    
    def save_trade(self, trade_data: Dict) -> int:
        """Save executed trade to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO trades (
                    signal_id, symbol, strategy, direction, entry_date,
                    exit_date, entry_price, exit_price, quantity, pnl,
                    return_pct, close_reason, commission
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data.get('signal_id'),
                trade_data['symbol'],
                trade_data['strategy'],
                trade_data['direction'],
                trade_data['entry_date'],
                trade_data['exit_date'],
                trade_data['entry_price'],
                trade_data['exit_price'],
                trade_data['quantity'],
                trade_data['pnl'],
                trade_data['return_pct'],
                trade_data['close_reason'],
                trade_data.get('commission', 0)
            ))
            trade_id = cursor.lastrowid
            conn.commit()
        
        logger.info(f"Trade saved with ID: {trade_id}")
        return trade_id
    
    def save_backtest_result(self, backtest_data: Dict) -> int:
        """Save backtest results to database"""
        # Create configuration hash for comparison
        config_str = json.dumps(backtest_data.get('config', {}), sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO backtest_results (
                    strategy, symbol, period_start, period_end, total_return,
                    annualized_return, max_drawdown, sharpe_ratio, win_rate,
                    total_trades, profit_factor, config_hash, detailed_results
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                backtest_data['strategy'],
                backtest_data['symbol'],
                backtest_data['period_start'],
                backtest_data['period_end'],
                backtest_data['total_return'],
                backtest_data['annualized_return'],
                backtest_data['max_drawdown'],
                backtest_data['sharpe_ratio'],
                backtest_data['win_rate'],
                backtest_data['total_trades'],
                backtest_data['profit_factor'],
                config_hash,
                json.dumps(backtest_data)
            ))
            backtest_id = cursor.lastrowid
            conn.commit()
        
        logger.info(f"Backtest result saved with ID: {backtest_id}")
        return backtest_id
    
    def save_news_sentiment(self, news_data: Dict) -> Optional[int]:
        """Save news sentiment data (avoid duplicates)"""
        # Create content hash to avoid duplicates
        content = news_data.get('title', '') + news_data.get('url', '')
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO news_sentiment (
                        symbol, title, source, sentiment_score, confidence, url, content_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    news_data['symbol'],
                    news_data['title'],
                    news_data['source'],
                    news_data['sentiment_score'],
                    news_data.get('confidence', 0.5),
                    news_data.get('url', ''),
                    content_hash
                ))
                news_id = cursor.lastrowid
                conn.commit()
                return news_id
        except sqlite3.IntegrityError:
            # Duplicate content
            return None
    
    def get_recent_signals(self, strategy: str = None, limit: int = 10) -> pd.DataFrame:
        """Get recent trading signals"""
        query = """
            SELECT * FROM signals 
            WHERE (? IS NULL OR strategy = ?)
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=[strategy, strategy, limit])
        
        return df
    
    def get_strategy_performance(self, strategy: str, days: int = 30) -> Dict:
        """Get strategy performance metrics"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get trades in period
            trades_df = pd.read_sql_query("""
                SELECT * FROM trades 
                WHERE strategy = ? AND entry_date >= ? AND entry_date <= ?
                ORDER BY entry_date
            """, conn, params=[strategy, start_date, end_date])
            
            # Get signals in period
            signals_df = pd.read_sql_query("""
                SELECT * FROM signals 
                WHERE strategy = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """, conn, params=[strategy, start_date, end_date])
        
        if trades_df.empty:
            return {'error': 'No trades found for the specified period'}
        
        # Calculate performance metrics
        total_pnl = trades_df['pnl'].sum()
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * (total_trades - winning_trades))) if avg_loss != 0 else np.inf
        
        return {
            'strategy': strategy,
            'period_days': days,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_signals': len(signals_df)
        }
    
    def get_portfolio_history(self, strategy: str, days: int = 30) -> pd.DataFrame:
        """Get portfolio performance history"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT * FROM portfolio_performance 
                WHERE strategy = ? AND date >= ? AND date <= ?
                ORDER BY date
            """, conn, params=[strategy, start_date, end_date])
        
        return df
    
    def compare_strategies(self, strategies: List[str], days: int = 30) -> pd.DataFrame:
        """Compare performance of multiple strategies"""
        comparison_data = []
        
        for strategy in strategies:
            perf = self.get_strategy_performance(strategy, days)
            if 'error' not in perf:
                comparison_data.append(perf)
        
        return pd.DataFrame(comparison_data)
    
    def get_news_sentiment_history(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Get news sentiment history for a symbol"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT timestamp, title, source, sentiment_score, confidence
                FROM news_sentiment 
                WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
            """, conn, params=[symbol, start_date, end_date])
        
        return df
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old data to keep database size manageable"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with sqlite3.connect(self.db_path) as conn:
            # Clean old signals (keep recent ones)
            conn.execute("DELETE FROM signals WHERE timestamp < ?", (cutoff_date,))
            
            # Clean old news sentiment (keep recent ones)
            conn.execute("DELETE FROM news_sentiment WHERE timestamp < ?", (cutoff_date,))
            
            # Clean old portfolio performance (keep aggregated monthly data)
            conn.execute("""
                DELETE FROM portfolio_performance 
                WHERE date < ? AND id NOT IN (
                    SELECT MIN(id) FROM portfolio_performance 
                    WHERE date < ? 
                    GROUP BY strftime('%Y-%m', date), strategy
                )
            """, (cutoff_date, cutoff_date))
            
            conn.commit()
        
        logger.info(f"Cleaned up data older than {days_to_keep} days")
    
    def export_data(self, table_name: str, output_file: str):
        """Export table data to CSV"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            df.to_csv(output_file, index=False)
        
        logger.info(f"Exported {table_name} to {output_file}")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        tables = ['signals', 'trades', 'portfolio_performance', 'backtest_results', 
                 'news_sentiment', 'market_data_cache']
        
        with sqlite3.connect(self.db_path) as conn:
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats[f"{table}_count"] = count
        
        # Database file size
        db_size = Path(self.db_path).stat().st_size / (1024 * 1024)  # MB
        stats['db_size_mb'] = round(db_size, 2)
        
        return stats

def main():
    """Test database functionality"""
    db = TradingDatabase()
    
    # Test signal saving
    test_signal = {
        'symbol': 'RELIANCE.NS',
        'strategy': 'advanced_shortterm',
        'direction': 'BUY',
        'confidence': 85.5,
        'entry_price': 2850.00,
        'stop_loss': 2800.00,
        'take_profit': 2950.00,
        'reasoning': ['Technical bullish', 'Positive sentiment'],
        'risk_reward_ratio': 2.0,
        'expected_return': 0.035
    }
    
    signal_id = db.save_signal(test_signal)
    print(f"Test signal saved with ID: {signal_id}")
    
    # Test trade saving
    test_trade = {
        'signal_id': signal_id,
        'symbol': 'RELIANCE.NS',
        'strategy': 'advanced_shortterm',
        'direction': 'BUY',
        'entry_date': datetime.now(),
        'exit_date': datetime.now() + timedelta(days=2),
        'entry_price': 2850.00,
        'exit_price': 2920.00,
        'quantity': 10,
        'pnl': 700.00,
        'return_pct': 0.0246,
        'close_reason': 'Take Profit',
        'commission': 25.00
    }
    
    trade_id = db.save_trade(test_trade)
    print(f"Test trade saved with ID: {trade_id}")
    
    # Get recent signals
    recent_signals = db.get_recent_signals(limit=5)
    print(f"\nRecent signals:\n{recent_signals}")
    
    # Get database stats
    stats = db.get_database_stats()
    print(f"\nDatabase stats: {stats}")

if __name__ == "__main__":
    main()