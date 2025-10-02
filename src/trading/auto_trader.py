#!/usr/bin/env python3
"""
Automated Trading Executor
Integrates with broker APIs for automated order placement and portfolio management.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import requests
from src.data.database_manager import TradingDatabase
from src.strategies.advanced_shortterm_strategy import AdvancedShortTermStrategy, TradingSignal

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_MARKET = "STOP_LOSS_MARKET"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    PLACED = "PLACED"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class Order:
    """Order structure"""
    symbol: str
    quantity: int
    order_type: OrderType
    side: str  # BUY or SELL
    price: float = 0.0
    trigger_price: float = 0.0
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Position:
    """Position structure"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    pnl: float
    pnl_percent: float
    side: str  # LONG or SHORT

class MockBrokerAPI:
    """
    Mock broker API for testing
    Replace with actual broker API integration (Zerodha Kite, Angel Broking, etc.)
    """
    
    def __init__(self, initial_balance: float = 100000):
        self.balance = initial_balance
        self.positions = {}
        self.orders = {}
        self.order_counter = 1000
        
        logger.info(f"Mock Broker initialized with balance: ₹{initial_balance:,.2f}")
    
    def place_order(self, order: Order) -> str:
        """Place an order"""
        order.order_id = f"ORD{self.order_counter}"
        self.order_counter += 1
        
        # Simulate order placement
        order.status = OrderStatus.PLACED
        self.orders[order.order_id] = order
        
        # Simulate immediate execution for market orders
        if order.order_type == OrderType.MARKET:
            self._execute_order(order)
        
        logger.info(f"Order placed: {order.order_id} - {order.side} {order.quantity} {order.symbol}")
        return order.order_id
    
    def _execute_order(self, order: Order):
        """Execute an order (mock execution)"""
        try:
            # Calculate transaction cost
            transaction_value = order.price * order.quantity
            brokerage = max(20, transaction_value * 0.0003)  # 0.03% or ₹20, whichever is higher
            
            if order.side == "BUY":
                total_cost = transaction_value + brokerage
                if self.balance >= total_cost:
                    self.balance -= total_cost
                    
                    # Update position
                    if order.symbol in self.positions:
                        pos = self.positions[order.symbol]
                        new_quantity = pos.quantity + order.quantity
                        new_avg_price = ((pos.avg_price * pos.quantity) + (order.price * order.quantity)) / new_quantity
                        pos.quantity = new_quantity
                        pos.avg_price = new_avg_price
                    else:
                        self.positions[order.symbol] = Position(
                            symbol=order.symbol,
                            quantity=order.quantity,
                            avg_price=order.price,
                            current_price=order.price,
                            pnl=0.0,
                            pnl_percent=0.0,
                            side="LONG"
                        )
                    
                    order.status = OrderStatus.EXECUTED
                    logger.info(f"BUY order executed: {order.order_id}")
                else:
                    order.status = OrderStatus.REJECTED
                    logger.warning(f"Insufficient balance for order: {order.order_id}")
            
            elif order.side == "SELL":
                if order.symbol in self.positions and self.positions[order.symbol].quantity >= order.quantity:
                    pos = self.positions[order.symbol]
                    
                    # Calculate P&L
                    pnl = (order.price - pos.avg_price) * order.quantity
                    self.balance += (transaction_value - brokerage + pnl)
                    
                    # Update position
                    pos.quantity -= order.quantity
                    if pos.quantity == 0:
                        del self.positions[order.symbol]
                    
                    order.status = OrderStatus.EXECUTED
                    logger.info(f"SELL order executed: {order.order_id}, P&L: ₹{pnl:.2f}")
                else:
                    order.status = OrderStatus.REJECTED
                    logger.warning(f"Insufficient quantity for sell order: {order.order_id}")
        
        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"Error executing order {order.order_id}: {e}")
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        return self.positions.copy()
    
    def get_balance(self) -> float:
        """Get account balance"""
        return self.balance
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == OrderStatus.PLACED:
                order.status = OrderStatus.CANCELLED
                logger.info(f"Order cancelled: {order_id}")
                return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status"""
        return self.orders.get(order_id)

class AutoTrader:
    """
    Automated trading system that executes trades based on strategy signals
    """
    
    def __init__(self, broker_api, initial_capital: float = 100000, max_positions: int = 5):
        self.broker = broker_api
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.db = TradingDatabase()
        self.active_signals = {}  # symbol -> signal
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.running = False
        
        logger.info(f"AutoTrader initialized with ₹{initial_capital:,.2f} capital")
    
    def start_trading(self):
        """Start automated trading"""
        self.running = True
        logger.info("🚀 Automated trading started")
    
    def stop_trading(self):
        """Stop automated trading"""
        self.running = False
        logger.info("🛑 Automated trading stopped")
    
    def process_signal(self, signal: TradingSignal, symbol: str) -> bool:
        """
        Process a trading signal and execute if criteria are met
        
        Args:
            signal: Trading signal from strategy
            symbol: Stock symbol
            
        Returns:
            bool: True if trade was executed
        """
        if not self.running:
            logger.info("Trading is stopped, ignoring signal")
            return False
        
        try:
            # Check signal quality
            if signal.confidence < 70:  # Minimum 70% confidence
                logger.info(f"Signal confidence too low ({signal.confidence:.1f}%) for {symbol}")
                return False
            
            # Check if we already have a position in this symbol
            positions = self.broker.get_positions()
            if symbol in positions:
                logger.info(f"Already have position in {symbol}, skipping")
                return False
            
            # Check maximum positions limit
            if len(positions) >= self.max_positions:
                logger.info(f"Maximum positions ({self.max_positions}) reached, skipping")
                return False
            
            # Calculate position size based on risk management
            balance = self.broker.get_balance()
            risk_amount = balance * self.risk_per_trade
            
            # Calculate stop loss distance
            if signal.direction == "BUY":
                stop_distance = abs(signal.entry_price - signal.stop_loss)
                quantity = int(risk_amount / stop_distance) if stop_distance > 0 else 0
            elif signal.direction == "SELL":
                # For short selling (if supported)
                stop_distance = abs(signal.stop_loss - signal.entry_price)
                quantity = int(risk_amount / stop_distance) if stop_distance > 0 else 0
            else:
                logger.info(f"HOLD signal for {symbol}, no action taken")
                return False
            
            # Ensure minimum quantity and maximum investment
            max_investment = balance * 0.2  # Maximum 20% per position
            max_quantity_by_investment = int(max_investment / signal.entry_price)
            quantity = min(quantity, max_quantity_by_investment)
            
            if quantity < 1:
                logger.info(f"Calculated quantity too small for {symbol}")
                return False
            
            # Place market order
            order = Order(
                symbol=symbol,
                quantity=quantity,
                order_type=OrderType.MARKET,
                side=signal.direction,
                price=signal.entry_price
            )
            
            order_id = self.broker.place_order(order)
            
            if order_id:
                # Store signal for position management
                self.active_signals[symbol] = {
                    'signal': signal,
                    'order_id': order_id,
                    'quantity': quantity,
                    'entry_time': datetime.now()
                }
                
                # Save trade to database
                trade_data = {
                    'symbol': symbol,
                    'strategy': 'advanced_shortterm',
                    'direction': signal.direction,
                    'entry_date': datetime.now(),
                    'entry_price': signal.entry_price,
                    'quantity': quantity,
                    'close_reason': 'AUTO_ENTRY'
                }
                
                self.db.save_trade(trade_data)
                
                # Schedule stop loss and take profit orders
                self._schedule_exit_orders(symbol, signal, quantity)
                
                logger.info(f" Trade executed: {signal.direction} {quantity} {symbol} at ₹{signal.entry_price:.2f}")
                return True
            
        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {e}")
        
        return False
    
    def _schedule_exit_orders(self, symbol: str, signal: TradingSignal, quantity: int):
        """Schedule stop loss and take profit orders"""
        try:
            # Stop loss order
            if signal.direction == "BUY":
                stop_order = Order(
                    symbol=symbol,
                    quantity=quantity,
                    order_type=OrderType.STOP_LOSS_MARKET,
                    side="SELL",
                    trigger_price=signal.stop_loss
                )
            else:  # SELL
                stop_order = Order(
                    symbol=symbol,
                    quantity=quantity,
                    order_type=OrderType.STOP_LOSS_MARKET,
                    side="BUY",
                    trigger_price=signal.stop_loss
                )
            
            stop_order_id = self.broker.place_order(stop_order)
            
            # Take profit order (limit order)
            if signal.direction == "BUY":
                profit_order = Order(
                    symbol=symbol,
                    quantity=quantity,
                    order_type=OrderType.LIMIT,
                    side="SELL",
                    price=signal.take_profit
                )
            else:  # SELL
                profit_order = Order(
                    symbol=symbol,
                    quantity=quantity,
                    order_type=OrderType.LIMIT,
                    side="BUY",
                    price=signal.take_profit
                )
            
            profit_order_id = self.broker.place_order(profit_order)
            
            # Store order IDs for tracking
            if symbol in self.active_signals:
                self.active_signals[symbol]['stop_order_id'] = stop_order_id
                self.active_signals[symbol]['profit_order_id'] = profit_order_id
            
            logger.info(f"Exit orders scheduled for {symbol}: SL={signal.stop_loss:.2f}, TP={signal.take_profit:.2f}")
            
        except Exception as e:
            logger.error(f"Error scheduling exit orders for {symbol}: {e}")
    
    def monitor_positions(self):
        """Monitor active positions and manage exits"""
        positions = self.broker.get_positions()
        
        for symbol, position in positions.items():
            if symbol in self.active_signals:
                signal_data = self.active_signals[symbol]
                signal = signal_data['signal']
                
                # Check if position should be closed based on time or other criteria
                entry_time = signal_data['entry_time']
                time_elapsed = datetime.now() - entry_time
                
                # Close position if held for more than 3 days (for short-term strategy)
                if time_elapsed > timedelta(days=3):
                    self._close_position(symbol, "TIME_LIMIT")
                
                # Update P&L in database
                pnl = position.pnl
                if pnl != 0:  # Position has been closed
                    trade_data = {
                        'symbol': symbol,
                        'strategy': 'advanced_shortterm',
                        'direction': signal.direction,
                        'exit_date': datetime.now(),
                        'exit_price': position.current_price,
                        'quantity': position.quantity,
                        'pnl': pnl,
                        'return_pct': pnl / (signal.entry_price * position.quantity),
                        'close_reason': 'POSITION_CLOSED'
                    }
                    
                    self.db.save_trade(trade_data)
    
    def _close_position(self, symbol: str, reason: str):
        """Close a position manually"""
        try:
            positions = self.broker.get_positions()
            if symbol not in positions:
                logger.warning(f"No position found for {symbol}")
                return
            
            position = positions[symbol]
            
            # Place market order to close position
            close_order = Order(
                symbol=symbol,
                quantity=position.quantity,
                order_type=OrderType.MARKET,
                side="SELL" if position.side == "LONG" else "BUY",
                price=position.current_price
            )
            
            order_id = self.broker.place_order(close_order)
            
            if order_id:
                # Cancel pending exit orders
                if symbol in self.active_signals:
                    signal_data = self.active_signals[symbol]
                    
                    if 'stop_order_id' in signal_data:
                        self.broker.cancel_order(signal_data['stop_order_id'])
                    
                    if 'profit_order_id' in signal_data:
                        self.broker.cancel_order(signal_data['profit_order_id'])
                    
                    del self.active_signals[symbol]
                
                logger.info(f"Position closed for {symbol}, reason: {reason}")
        
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        positions = self.broker.get_positions()
        balance = self.broker.get_balance()
        
        total_investment = 0
        total_current_value = 0
        total_pnl = 0
        
        for symbol, position in positions.items():
            investment = position.avg_price * position.quantity
            current_value = position.current_price * position.quantity
            
            total_investment += investment
            total_current_value += current_value
            total_pnl += position.pnl
        
        total_portfolio_value = balance + total_current_value
        total_return = ((total_portfolio_value - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'initial_capital': self.initial_capital,
            'current_balance': balance,
            'total_investment': total_investment,
            'current_portfolio_value': total_portfolio_value,
            'total_pnl': total_pnl,
            'total_return_percent': total_return,
            'active_positions': len(positions),
            'available_margin': balance
        }
    
    def run_strategy_scan(self, symbols: List[str]):
        """Run strategy analysis on multiple symbols and execute trades"""
        logger.info(f"Running strategy scan on {len(symbols)} symbols")
        
        for symbol in symbols:
            try:
                # Run strategy analysis
                strategy = AdvancedShortTermStrategy(symbol)
                strategy.fetch_data()
                signal = strategy.generate_multi_factor_signal()
                
                # Save signal to database
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
                
                self.db.save_signal(signal_data)
                
                # Process signal for execution
                if signal.direction in ["BUY", "SELL"]:
                    self.process_signal(signal, symbol)
                
                # Small delay between symbols
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        logger.info("Strategy scan completed")

def main():
    """Test the automated trading system"""
    print("🤖 Automated Trading System Demo")
    print("=" * 50)
    
    # Initialize mock broker
    broker = MockBrokerAPI(initial_balance=100000)
    
    # Initialize auto trader
    auto_trader = AutoTrader(broker, initial_capital=100000, max_positions=3)
    auto_trader.start_trading()
    
    # Test symbols
    test_symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
    
    print(f"💼 Initial Portfolio Summary:")
    summary = auto_trader.get_portfolio_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\n🔍 Running strategy scan on: {', '.join(test_symbols)}")
    
    # Run strategy scan
    auto_trader.run_strategy_scan(test_symbols)
    
    print(f"\n📊 Final Portfolio Summary:")
    summary = auto_trader.get_portfolio_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\n💹 Active Positions:")
    positions = broker.get_positions()
    for symbol, position in positions.items():
        print(f"  {symbol}: {position.quantity} shares at ₹{position.avg_price:.2f} (P&L: ₹{position.pnl:.2f})")
    
    print("\n⚠️ Note: This is a mock trading system for demonstration.")
    print("Replace MockBrokerAPI with actual broker API for live trading.")

if __name__ == "__main__":
    main()