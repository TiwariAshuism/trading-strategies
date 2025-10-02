#!/usr/bin/env python3
"""
Backtesting Module for Advanced Short-Term Trading Strategy
Test historical performance and optimize parameters.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import ssl
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
from src.strategies.advanced_shortterm_strategy import AdvancedShortTermStrategy
from src.config.strategy_config import CONFIG

# SSL bypass
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Backtest results container"""
    symbol: str
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    trades: List[Dict]
    equity_curve: pd.Series
    benchmark_return: float

class StrategyBacktester:
    """Comprehensive backtesting system for the Advanced Short-Term Strategy"""
    
    def __init__(self, symbol: str, start_date: str, end_date: str, initial_capital: float = 100000):
        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = None
        self.results = None
        
    def fetch_historical_data(self, interval: str = "1d") -> pd.DataFrame:
        """Fetch historical data for backtesting"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(start=self.start_date, end=self.end_date, interval=interval)
            
            if self.data.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            logger.info(f"Fetched {len(self.data)} data points for {self.symbol}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def run_backtest(self, rebalance_frequency: int = 3) -> BacktestResult:
        """
        Run comprehensive backtest
        
        Args:
            rebalance_frequency: Days between strategy signals (default 3 for short-term)
        """
        if self.data is None:
            self.fetch_historical_data()
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        position_size = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        trades = []
        equity_curve = []
        
        # Convert data to required format for strategy
        data_length = len(self.data)
        
        print(f"\n🔄 Running backtest for {self.symbol}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ₹{self.initial_capital:,.2f}")
        print(f"Rebalance Frequency: {rebalance_frequency} days")
        print("-" * 60)
        
        # Iterate through historical data
        for i in range(60, data_length, rebalance_frequency):  # Start after 60 days for indicators
            current_date = self.data.index[i]
            current_price = self.data['Close'].iloc[i]
            
            # Create subset of data for strategy analysis (up to current date)
            historical_data = self.data.iloc[:i+1]
            
            try:
                # Initialize strategy with historical data
                strategy = AdvancedShortTermStrategy(self.symbol, period="1y")
                strategy.data = historical_data.tail(252)  # Last 252 days for analysis
                
                # Generate signal
                signal = strategy.generate_multi_factor_signal()
                
                # Close existing position if stop loss or take profit hit
                if position != 0:
                    should_close = False
                    close_reason = ""
                    
                    if position == 1:  # Long position
                        if current_price <= stop_loss:
                            should_close = True
                            close_reason = "Stop Loss"
                        elif current_price >= take_profit:
                            should_close = True
                            close_reason = "Take Profit"
                    elif position == -1:  # Short position
                        if current_price >= stop_loss:
                            should_close = True
                            close_reason = "Stop Loss"
                        elif current_price <= take_profit:
                            should_close = True
                            close_reason = "Take Profit"
                    
                    # Check for signal reversal
                    if (position == 1 and signal.direction == "SELL") or \
                       (position == -1 and signal.direction == "BUY"):
                        should_close = True
                        close_reason = "Signal Reversal"
                    
                    # Close position
                    if should_close:
                        pnl = (current_price - entry_price) * position_size * position
                        capital += pnl
                        
                        trade_record = {
                            'entry_date': trades[-1]['entry_date'] if trades else current_date,
                            'exit_date': current_date,
                            'direction': 'LONG' if position == 1 else 'SHORT',
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'position_size': position_size,
                            'pnl': pnl,
                            'return': pnl / (entry_price * position_size) * (1 if position == 1 else -1),
                            'close_reason': close_reason,
                            'capital_after': capital
                        }
                        trades.append(trade_record)
                        
                        # Reset position
                        position = 0
                        position_size = 0
                
                # Open new position based on signal
                if position == 0 and signal.direction in ["BUY", "SELL"] and signal.confidence > 50:
                    # Calculate position size (risk 2% of capital)
                    risk_amount = capital * CONFIG.RISK_MANAGEMENT['max_position_risk']
                    
                    if signal.direction == "BUY":
                        position = 1
                        entry_price = current_price
                        stop_loss = signal.stop_loss
                        take_profit = signal.take_profit
                        
                        # Position size based on stop loss distance
                        stop_distance = abs(entry_price - stop_loss)
                        position_size = min(risk_amount / stop_distance, capital * 0.95 / entry_price)
                        
                    elif signal.direction == "SELL":
                        position = -1
                        entry_price = current_price
                        stop_loss = signal.stop_loss
                        take_profit = signal.take_profit
                        
                        # Position size for short
                        stop_distance = abs(stop_loss - entry_price)
                        position_size = min(risk_amount / stop_distance, capital * 0.95 / entry_price)
                
                # Record equity curve
                if position != 0:
                    unrealized_pnl = (current_price - entry_price) * position_size * position
                    current_equity = capital + unrealized_pnl
                else:
                    current_equity = capital
                
                equity_curve.append({
                    'date': current_date,
                    'equity': current_equity,
                    'position': position,
                    'price': current_price
                })
                
            except Exception as e:
                logger.warning(f"Error processing date {current_date}: {e}")
                continue
        
        # Close any remaining position
        if position != 0:
            final_price = self.data['Close'].iloc[-1]
            pnl = (final_price - entry_price) * position_size * position
            capital += pnl
            
            trade_record = {
                'entry_date': trades[-1]['entry_date'] if trades else self.data.index[-1],
                'exit_date': self.data.index[-1],
                'direction': 'LONG' if position == 1 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': final_price,
                'position_size': position_size,
                'pnl': pnl,
                'return': pnl / (entry_price * position_size) * (1 if position == 1 else -1),
                'close_reason': 'End of Backtest',
                'capital_after': capital
            }
            trades.append(trade_record)
        
        # Calculate performance metrics
        if not trades:
            logger.warning("No trades executed during backtest period")
            return None
        
        # Convert equity curve to Series
        equity_df = pd.DataFrame(equity_curve)
        equity_series = equity_df.set_index('date')['equity']
        
        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        
        # Annualized return
        days = (pd.to_datetime(self.end_date) - pd.to_datetime(self.start_date)).days
        annualized_return = (1 + total_return) ** (365.25 / days) - 1
        
        # Maximum drawdown
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (assuming 0% risk-free rate)
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Trade statistics
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = len([t for t in trades if t['pnl'] < 0])
        win_rate = winning_trades / len(trades) if trades else 0
        
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else np.inf
        
        # Benchmark (buy and hold)
        benchmark_return = (self.data['Close'].iloc[-1] - self.data['Close'].iloc[0]) / self.data['Close'].iloc[0]
        
        # Create result object
        result = BacktestResult(
            symbol=self.symbol,
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            total_trades=len(trades),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=trades,
            equity_curve=equity_series,
            benchmark_return=benchmark_return
        )
        
        self.results = result
        return result
    
    def print_backtest_results(self):
        """Print comprehensive backtest results"""
        if self.results is None:
            print(" No backtest results available")
            return
        
        result = self.results
        
        print("\n" + "="*80)
        print(f"📊 BACKTEST RESULTS: {result.symbol}")
        print("="*80)
        
        # Performance Summary
        print(f"\n💰 PERFORMANCE SUMMARY")
        print("-" * 40)
        print(f"Initial Capital: ₹{self.initial_capital:,.2f}")
        print(f"Final Capital: ₹{self.initial_capital * (1 + result.total_return):,.2f}")
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Annualized Return: {result.annualized_return:.2%}")
        print(f"Benchmark (Buy & Hold): {result.benchmark_return:.2%}")
        print(f"Alpha: {result.total_return - result.benchmark_return:.2%}")
        
        # Risk Metrics
        print(f"\n⚠️ RISK METRICS")
        print("-" * 40)
        print(f"Maximum Drawdown: {result.max_drawdown:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Volatility: {result.equity_curve.pct_change().std() * np.sqrt(252):.2%}")
        
        # Trade Statistics
        print(f"\n📈 TRADE STATISTICS")
        print("-" * 40)
        print(f"Total Trades: {result.total_trades}")
        print(f"Winning Trades: {result.winning_trades}")
        print(f"Losing Trades: {result.losing_trades}")
        print(f"Win Rate: {result.win_rate:.1%}")
        print(f"Average Win: ₹{result.avg_win:,.2f}")
        print(f"Average Loss: ₹{result.avg_loss:,.2f}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        
        # Best and Worst Trades
        if result.trades:
            best_trade = max(result.trades, key=lambda x: x['pnl'])
            worst_trade = min(result.trades, key=lambda x: x['pnl'])
            
            print(f"\n🏆 BEST/WORST TRADES")
            print("-" * 40)
            print(f"Best Trade: ₹{best_trade['pnl']:,.2f} ({best_trade['return']:.1%}) - {best_trade['direction']} on {best_trade['entry_date'].strftime('%Y-%m-%d')}")
            print(f"Worst Trade: ₹{worst_trade['pnl']:,.2f} ({worst_trade['return']:.1%}) - {worst_trade['direction']} on {worst_trade['entry_date'].strftime('%Y-%m-%d')}")
        
        print("\n" + "="*80)
    
    def plot_backtest_results(self):
        """Create comprehensive backtest visualization"""
        if self.results is None:
            print(" No backtest results available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Backtest Results: {self.symbol}', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve vs Benchmark
        ax1 = axes[0, 0]
        
        # Strategy equity curve
        strategy_returns = (self.results.equity_curve / self.initial_capital - 1) * 100
        ax1.plot(strategy_returns.index, strategy_returns.values, label='Strategy', linewidth=2, color='blue')
        
        # Benchmark (buy and hold)
        price_data = self.data['Close'].reindex(strategy_returns.index, method='ffill')
        benchmark_returns = (price_data / price_data.iloc[0] - 1) * 100
        ax1.plot(benchmark_returns.index, benchmark_returns.values, label='Buy & Hold', linewidth=2, color='gray', alpha=0.7)
        
        ax1.set_title('Strategy vs Benchmark Returns')
        ax1.set_ylabel('Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = axes[0, 1]
        running_max = self.results.equity_curve.expanding().max()
        drawdown = (self.results.equity_curve - running_max) / running_max * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade Returns Distribution
        ax3 = axes[1, 0]
        trade_returns = [t['return'] * 100 for t in self.results.trades]
        ax3.hist(trade_returns, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax3.set_title('Trade Returns Distribution')
        ax3.set_xlabel('Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # 4. Monthly Returns Heatmap
        ax4 = axes[1, 1]
        monthly_returns = self.results.equity_curve.resample('M').last().pct_change().dropna()
        if len(monthly_returns) > 12:
            monthly_data = monthly_returns.values.reshape(-1, 12) * 100
            sns.heatmap(monthly_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax4)
            ax4.set_title('Monthly Returns Heatmap (%)')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Year')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor heatmap', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Monthly Returns')
        
        plt.tight_layout()
        plt.show()

def run_multi_symbol_backtest(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Run backtest on multiple symbols and compare results"""
    results = []
    
    print(f"\n🔄 Running Multi-Symbol Backtest")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date} to {end_date}")
    print("="*60)
    
    for symbol in symbols:
        try:
            print(f"\n📊 Backtesting {symbol}...")
            backtester = StrategyBacktester(symbol, start_date, end_date)
            result = backtester.run_backtest()
            
            if result:
                results.append({
                    'Symbol': symbol,
                    'Total Return': f"{result.total_return:.2%}",
                    'Annualized Return': f"{result.annualized_return:.2%}",
                    'Max Drawdown': f"{result.max_drawdown:.2%}",
                    'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                    'Win Rate': f"{result.win_rate:.1%}",
                    'Total Trades': result.total_trades,
                    'Profit Factor': f"{result.profit_factor:.2f}",
                    'Benchmark Return': f"{result.benchmark_return:.2%}"
                })
            
        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {e}")
            results.append({
                'Symbol': symbol,
                'Total Return': 'ERROR',
                'Annualized Return': 'ERROR',
                'Max Drawdown': 'ERROR',
                'Sharpe Ratio': 'ERROR',
                'Win Rate': 'ERROR',
                'Total Trades': 0,
                'Profit Factor': 'ERROR',
                'Benchmark Return': 'ERROR'
            })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    print(f"\n📈 MULTI-SYMBOL BACKTEST RESULTS")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    return comparison_df

def main():
    """Main backtesting function"""
    print("📊 Advanced Short-Term Strategy Backtester")
    print("="*50)
    
    # Configuration
    symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    
    print("\n🎯 Backtest Options:")
    print("1. Single Symbol Detailed Backtest")
    print("2. Multi-Symbol Comparison")
    print("3. Parameter Optimization")
    
    try:
        choice = int(input("\nEnter choice (1-3): "))
        
        if choice == 1:
            # Single symbol detailed backtest
            print(f"\nAvailable symbols: {', '.join(symbols)}")
            symbol = input("Enter symbol (or press Enter for RELIANCE.NS): ").strip().upper()
            if not symbol:
                symbol = "RELIANCE.NS"
            
            print(f"\n🔍 Running detailed backtest for {symbol}...")
            backtester = StrategyBacktester(symbol, start_date, end_date)
            result = backtester.run_backtest()
            
            if result:
                backtester.print_backtest_results()
                backtester.plot_backtest_results()
            
        elif choice == 2:
            # Multi-symbol comparison
            comparison_df = run_multi_symbol_backtest(symbols, start_date, end_date)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_comparison_{timestamp}.csv"
            comparison_df.to_csv(filename, index=False)
            print(f"\n💾 Results saved to: {filename}")
            
        elif choice == 3:
            # Parameter optimization placeholder
            print("\n🔧 Parameter optimization feature coming soon!")
            print("This will test different weight combinations and thresholds.")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f" Error: {e}")

if __name__ == "__main__":
    main()