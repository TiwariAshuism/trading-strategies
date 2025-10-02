#!/usr/bin/env python3
"""
Master Control Panel for Advanced Trading System
Centralized launcher and coordinator for        print(f" System Status:")
        print(f"   🌐 Web Dashboard: http://localhost:8501")
        print(f"   📡 Data Feed: Running (simple mode)")
        print(f"   💾 Database: trading_data.db") trading components.
"""

import sys
import os
import subprocess
import threading
import time
import argparse
from datetime import datetime
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import emoji handler
try:
    from src.utils.emoji_handler import safe_print, format_text
except ImportError:
    # Fallback if emoji handler not available
    def safe_print(text, end='\n', flush=False):
        try:
            print(text, end=end, flush=flush)
        except UnicodeEncodeError:
            text = str(text).replace("", "[LAUNCH]").replace("", "[OK]").replace("", "[ERROR]").replace("🌐", "[WEB]").replace("📡", "[FEED]").replace("💾", "[DATA]")
            print(text, end=end, flush=flush)
    
    def format_text(text):
        return text

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingSystemController:
    """Master controller for the entire trading system"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.processes = {}
        self.running = False
        
    def start_component(self, component: str) -> bool:
        """Start a specific component"""
        try:
            if component == "dashboard":
                # Start Streamlit dashboard
                cmd = [sys.executable, "-m", "streamlit", "run", "src/ui/streamlit_dashboard.py", "--server.port", "8501"]
                process = subprocess.Popen(cmd, cwd=self.project_root)
                self.processes["dashboard"] = process
                logger.info("🌐 Web Dashboard started on http://localhost:8501")
                
            elif component == "datafeed":
                # Start simple data feed (no WebSocket complexity)
                cmd = [sys.executable, "src/data/simple_data_feed.py"]
                process = subprocess.Popen(cmd, cwd=self.project_root)
                self.processes["datafeed"] = process
                logger.info("📡 Simple Data Feed started (no WebSocket server needed)")
                
            elif component == "autotrader":
                # Start automated trader
                cmd = [sys.executable, "src/trading/auto_trader.py"]
                process = subprocess.Popen(cmd, cwd=self.project_root)
                self.processes["autotrader"] = process
                logger.info("🤖 Automated Trader started")
                
            elif component == "database":
                # Initialize database
                cmd = [sys.executable, "src/data/database_manager.py"]
                subprocess.run(cmd, cwd=self.project_root, check=True)
                logger.info("💾 Database initialized")
                return True
                
            else:
                logger.error(f"Unknown component: {component}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error starting {component}: {e}")
            return False
    
    def stop_component(self, component: str) -> bool:
        """Stop a specific component"""
        try:
            if component in self.processes:
                process = self.processes[component]
                process.terminate()
                process.wait(timeout=10)
                del self.processes[component]
                logger.info(f" {component} stopped")
                return True
            else:
                logger.warning(f"Component {component} not running")
                return False
                
        except Exception as e:
            logger.error(f"Error stopping {component}: {e}")
            return False
    
    def start_full_system(self):
        """Start the complete trading system"""
        logger.info(" Starting Complete Trading System")
        print("=" * 60)
        
        # Initialize database first
        print("1️⃣ Initializing database...")
        if self.start_component("database"):
            print("    Database ready")
        else:
            print("    Database initialization failed")
            return False
        
        # Start data feed
        print("2️⃣ Starting real-time data feed...")
        if self.start_component("datafeed"):
            print("    Data feed running")
            time.sleep(2)
        else:
            print("    Data feed failed to start")
        
        # Start web dashboard
        print("3️⃣ Starting web dashboard...")
        if self.start_component("dashboard"):
            print("    Dashboard running")
            time.sleep(3)
        else:
            print("    Dashboard failed to start")
        
        print("\n System Status:")
        print(f"   🌐 Web Dashboard: http://localhost:8501")
        print(f"   📡 Data Feed: ws://localhost:8765")
        print(f"   💾 Database: trading_data.db")
        
        self.running = True
        
        print("\n📋 Available Commands:")
        print("   • Open http://localhost:8501 for web interface")
        print("   • Press Ctrl+C to stop all services")
        print("   • Check logs for system status")
        
        return True
    
    def stop_full_system(self):
        """Stop all components"""
        logger.info("🛑 Stopping Trading System")
        
        for component in list(self.processes.keys()):
            self.stop_component(component)
        
        self.running = False
        logger.info(" All components stopped")
    
    def run_analysis(self, symbol: str):
        """Run quick analysis for a symbol"""
        try:
            cmd = [
                sys.executable, "-c",
                f"""
from src.strategies.advanced_shortterm_strategy import AdvancedShortTermStrategy
strategy = AdvancedShortTermStrategy('{symbol}')
strategy.fetch_data()
signal = strategy.generate_multi_factor_signal()
strategy.print_detailed_analysis()
"""
            ]
            
            subprocess.run(cmd, cwd=self.project_root, check=True)
            
        except Exception as e:
            logger.error(f"Error running analysis for {symbol}: {e}")
    
    def run_backtest(self, symbol: str = None):
        """Run backtesting"""
        try:
            if symbol:
                cmd = [
                    sys.executable, "-c",
                    f"""
from src.strategies.strategy_backtester import StrategyBacktester
backtester = StrategyBacktester('{symbol}', '2023-01-01', '2024-12-31')
result = backtester.run_backtest()
if result:
    backtester.print_backtest_results()
"""
                ]
            else:
                cmd = [sys.executable, "strategy_backtester.py"]
            
            subprocess.run(cmd, cwd=self.project_root, check=True)
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
    
    def show_status(self):
        """Show system status"""
        print("\n Trading System Status")
        print("=" * 40)
        
        # Check running processes
        active_components = []
        for component, process in self.processes.items():
            if process.poll() is None:  # Process is running
                active_components.append(component)
            else:
                logger.warning(f"{component} process has stopped unexpectedly")
        
        print(f"🟢 Active Components: {', '.join(active_components) if active_components else 'None'}")
        
        # Database status
        db_path = self.project_root / "trading_data.db"
        if db_path.exists():
            db_size = db_path.stat().st_size / (1024 * 1024)  # MB
            print(f"💾 Database: {db_size:.2f} MB")
        else:
            print("💾 Database: Not initialized")
        
        # Quick system health check
        try:
            import requests
            response = requests.get("http://localhost:8501", timeout=5)
            print("🌐 Web Dashboard:  Running")
        except:
            print("🌐 Web Dashboard:  Not accessible")
        
        print(f"⏰ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main control panel interface"""
    parser = argparse.ArgumentParser(description="Advanced Trading System Control Panel")
    parser.add_argument("--start-all", action="store_true", help="Start all components")
    parser.add_argument("--start", choices=["dashboard", "datafeed", "autotrader", "database"], help="Start specific component")
    parser.add_argument("--stop", choices=["dashboard", "datafeed", "autotrader"], help="Stop specific component")
    parser.add_argument("--analyze", type=str, help="Run analysis for symbol")
    parser.add_argument("--backtest", type=str, help="Run backtest for symbol")
    parser.add_argument("--status", action="store_true", help="Show system status")
    
    args = parser.parse_args()
    
    controller = TradingSystemController()
    
    # ASCII Art Header
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║         ADVANCED TRADING SYSTEM CONTROL PANEL          ║
    ║                                                           ║
    ║     Portfolio Analysis  📰 News Sentiment Analysis     ║
    ║     Multi-Strategy      ⚡ Short-Term Signals          ║
    ║    🌐 Web Dashboard       📡 Real-Time Data              ║
    ║    🤖 Auto Trading        💾 Data Persistence           ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        if args.start_all:
            controller.start_full_system()
            
            # Keep system running
            try:
                while controller.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutdown signal received...")
                controller.stop_full_system()
        
        elif args.start:
            controller.start_component(args.start)
        
        elif args.stop:
            controller.stop_component(args.stop)
        
        elif args.analyze:
            print(f"🔍 Running analysis for {args.analyze}...")
            controller.run_analysis(args.analyze)
        
        elif args.backtest:
            print(f" Running backtest for {args.backtest}...")
            controller.run_backtest(args.backtest)
        
        elif args.status:
            controller.show_status()
        
        else:
            # Interactive mode
            print(" Interactive Mode")
            print("Available commands:")
            print("  1. Start full system")
            print("  2. Quick analysis")
            print("  3. Run backtest")
            print("  4. Show status")
            print("  5. Exit")
            
            while True:
                try:
                    choice = input("\nEnter choice (1-5): ").strip()
                    
                    if choice == "1":
                        controller.start_full_system()
                        input("Press Enter to stop system...")
                        controller.stop_full_system()
                        break
                    
                    elif choice == "2":
                        symbol = input("Enter symbol (e.g., RELIANCE.NS): ").strip()
                        if symbol:
                            controller.run_analysis(symbol)
                    
                    elif choice == "3":
                        symbol = input("Enter symbol (or press Enter for interactive): ").strip()
                        controller.run_backtest(symbol if symbol else None)
                    
                    elif choice == "4":
                        controller.show_status()
                    
                    elif choice == "5":
                        break
                    
                    else:
                        print("Invalid choice. Please try again.")
                
                except KeyboardInterrupt:
                    break
    
    except Exception as e:
        logger.error(f"Error in control panel: {e}")
    
    finally:
        # Cleanup
        controller.stop_full_system()
        print("\n👋 Trading System Control Panel Exited")

if __name__ == "__main__":
    main()