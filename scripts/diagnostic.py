#!/usr/bin/env python3
"""
Quick diagnostic script to check system dependencies
"""

import sys
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import emoji handler
try:
    from src.utils.emoji_handler import safe_print, replace_emojis
except ImportError:
    # Fallback if emoji handler not available
    def safe_print(text, end='\n', flush=False):
        try:
            print(text, end=end, flush=flush)
        except UnicodeEncodeError:
            text = text.replace("🔍", "[INFO]").replace("📦", "[MODULES]").replace("🐍", "[PATH]").replace("🔧", "[TRADING]").replace("✅", "[OK]").replace("", "[ERROR]")
            print(text, end=end, flush=flush)
    
    def replace_emojis(text):
        return text.replace("🔍", "[INFO]").replace("📦", "[MODULES]").replace("🐍", "[PATH]").replace("🔧", "[TRADING]").replace("✅", "[OK]").replace("", "[ERROR]")

def check_module(module_name):
    try:
        # Handle special module names
        if module_name == 'beautifulsoup4':
            import bs4
        else:
            __import__(module_name)
        return "✅ " + module_name
    except ImportError:
        return " " + module_name + " (missing)"

def main():
    safe_print("🔍 System Diagnostic")
    safe_print("=" * 30)
    
    safe_print(f"Python version: {sys.version}")
    safe_print(f"Python executable: {sys.executable}")
    
    safe_print("\n📦 Required Modules:")
    modules = [
        'pandas', 'numpy', 'yfinance', 'matplotlib', 'scipy',
        'textblob', 'feedparser', 'beautifulsoup4', 'requests',
        'sqlite3', 'streamlit', 'plotly'
    ]
    
    for module in modules:
        safe_print(f"  {check_module(module)}")
    
    safe_print("\n🐍 Python Path:")
    for path in sys.path[:5]:  # Show first 5 paths
        safe_print(f"  {path}")
    
    # Try to import key trading modules
    safe_print("\n🔧 Trading Modules:")
    try:
        from src.data.database_manager import TradingDatabase
        safe_print("  ✅ database_manager")
    except Exception as e:
        safe_print(f"   database_manager: {e}")
    
    try:
        from src.strategies.advanced_shortterm_strategy import AdvancedShortTermStrategy
        safe_print("  ✅ advanced_shortterm_strategy")
    except Exception as e:
        safe_print(f"   advanced_shortterm_strategy: {e}")
    
    # Additional path debugging
    safe_print(f"\n🔧 Debug Information:")
    safe_print(f"  Project root: {project_root}")
    safe_print(f"  Current directory: {Path.cwd()}")
    safe_print(f"  Script location: {Path(__file__).parent}")
    
    # Check if src directory exists
    src_path = project_root / "src"
    if src_path.exists():
        safe_print(f"   src directory found")
        safe_print(f"   Contents: {[p.name for p in src_path.iterdir()]}")
    else:
        safe_print(f"   src directory not found at: {src_path}")

if __name__ == "__main__":
    main()