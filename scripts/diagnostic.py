#!/usr/bin/env python3
"""
Quick diagnostic script to check system dependencies
"""

import sys
import subprocess

def check_module(module_name):
    try:
        __import__(module_name)
        return f"✅ {module_name}"
    except ImportError:
        return f"❌ {module_name} (missing)"

def main():
    print("🔍 System Diagnostic")
    print("=" * 30)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    print("\n📦 Required Modules:")
    modules = [
        'pandas', 'numpy', 'yfinance', 'matplotlib', 'scipy',
        'textblob', 'feedparser', 'beautifulsoup4', 'requests',
        'sqlite3', 'streamlit', 'plotly'
    ]
    
    for module in modules:
        print(f"  {check_module(module)}")
    
    print("\n🐍 Python Path:")
    for path in sys.path[:5]:  # Show first 5 paths
        print(f"  {path}")
    
    # Try to import key trading modules
    print("\n🔧 Trading Modules:")
    try:
        from src.data.database_manager import TradingDatabase
        print("  ✅ database_manager")
    except Exception as e:
        print(f"  ❌ database_manager: {e}")
    
    try:
        from src.strategies.advanced_shortterm_strategy import AdvancedShortTermStrategy
        print("  ✅ advanced_shortterm_strategy")
    except Exception as e:
        print(f"  ❌ advanced_shortterm_strategy: {e}")

if __name__ == "__main__":
    main()