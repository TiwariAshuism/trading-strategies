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

# Import emoji handler
try:
    from src.utils.emoji_handler import safe_print, format_text
except ImportError:
    # Fallback if emoji handler not available
    def safe_print(text, end='\n', flush=False):
        try:
            print(text, end=end, flush=flush)
        except UnicodeEncodeError:
            # Basic emoji replacement
            text = str(text).replace("🚀", "[LAUNCH]").replace("", "[OK]").replace("", "[ERROR]").replace("📊", "[CHART]").replace("", "[TARGET]").replace("🔧", "[TOOL]")
            print(text, end=end, flush=flush)
    
    def format_text(text):
        try:
            return text
        except UnicodeEncodeError:
            return str(text).replace("🚀", "[LAUNCH]").replace("", "[OK]").replace("", "[ERROR]")

def main():
    header = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        🚀 ADVANCED TRADING SYSTEM                         ║
║                                                           ║
║    Professional Algorithmic Trading Platform             ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """
    safe_print(header)
    
    safe_print("📋 Available Commands:")
    safe_print("")
    safe_print("🎛️  System Control:")
    safe_print("   python main.py --start-all      Start complete system")
    safe_print("   python main.py --dashboard      Launch web dashboard")
    safe_print("   python main.py --setup          Run system setup")
    safe_print("")
    safe_print("📊 Individual Strategies:")
    safe_print("   python main.py --portfolio      Portfolio analyzer")
    safe_print("   python main.py --screener       Multi-strategy screener")
    safe_print("   python main.py --news           News-based trading")
    safe_print("   python main.py --shortterm      Advanced short-term strategy")
    safe_print("")
    safe_print("🔧 System Tools:")
    safe_print("   python main.py --test           Run test suite")
    safe_print("   python main.py --diagnostic     System diagnostic")
    safe_print("   python main.py --backtest       Strategy backtesting")
    safe_print("")
    
    if len(sys.argv) == 1:
        safe_print("💡 Run with --help or choose an option above")
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
                safe_print("\n🛑 Shutdown signal received...")
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
            safe_print("Use any of the commands shown above")
        
        else:
            safe_print(f" Unknown command: {arg}")
            safe_print("Use python main.py --help for available commands")
    
    except ImportError as e:
        safe_print(f" Import error: {e}")
        safe_print("💡 Try running: python main.py --setup")
    except Exception as e:
        safe_print(f" Error: {e}")

if __name__ == "__main__":
    main()