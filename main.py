#!/usr/bin/env python3
"""
Main Entry Point for Advanced Trading System
Provides easy access to all system components from the root directory.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        🚀 ADVANCED TRADING SYSTEM                         ║
║                                                           ║
║    Professional Algorithmic Trading Platform             ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    print("📋 Available Commands:")
    print()
    print("🎛️  System Control:")
    print("   python main.py --start-all      Start complete system")
    print("   python main.py --dashboard      Launch web dashboard")
    print("   python main.py --setup          Run system setup")
    print()
    print("📊 Individual Strategies:")
    print("   python main.py --portfolio      Portfolio analyzer")
    print("   python main.py --screener       Multi-strategy screener")
    print("   python main.py --news           News-based trading")
    print("   python main.py --shortterm      Advanced short-term strategy")
    print()
    print("🔧 System Tools:")
    print("   python main.py --test           Run test suite")
    print("   python main.py --diagnostic     System diagnostic")
    print("   python main.py --backtest       Strategy backtesting")
    print()
    
    if len(sys.argv) == 1:
        print("💡 Run with --help or choose an option above")
        return
    
    arg = sys.argv[1]
    
    try:
        if arg == "--start-all":
            from scripts.trading_control_panel import TradingSystemController
            controller = TradingSystemController()
            controller.start_full_system()
            
            try:
                import time
                while controller.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutdown signal received...")
                controller.stop_full_system()
        
        elif arg == "--dashboard":
            os.system("streamlit run src/ui/streamlit_dashboard.py --server.port 8501")
        
        elif arg == "--setup":
            from scripts.setup import TradingSystemSetup
            setup = TradingSystemSetup()
            setup.setup()
        
        elif arg == "--portfolio":
            from src.strategies.portfolio_analyzer import main
            main()
        
        elif arg == "--screener":
            from src.strategies.algotrade_nalco import main
            main()
        
        elif arg == "--news":
            from src.strategies.algotrade_news import main
            main()
        
        elif arg == "--shortterm":
            from src.strategies.advanced_shortterm_strategy import main
            main()
        
        elif arg == "--test":
            from tests.test_suite import main
            main()
        
        elif arg == "--diagnostic":
            from scripts.diagnostic import main
            main()
        
        elif arg == "--backtest":
            from src.strategies.strategy_backtester import main
            main()
        
        elif arg == "--help":
            print("Use any of the commands shown above")
        
        else:
            print(f"❌ Unknown command: {arg}")
            print("Use python main.py --help for available commands")
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running: python main.py --setup")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()