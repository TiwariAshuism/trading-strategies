#!/bin/bash
"""
Quick Setup Script for Trading System
Activates virtual environment and installs dependencies
"""

echo " Trading System Setup"
echo "======================"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo " Error: requirements.txt not found. Please run from the trading-strategies directory."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Run diagnostic
echo "🔍 Running system diagnostic..."
python diagnostic.py

echo ""
echo " Setup complete!"
echo ""
echo " To use the trading system:"
echo "   1. Activate environment: source env/bin/activate"
echo "   2. Run control panel: python trading_control_panel.py --start-all"
echo "   3. Or run individual components:"
echo "      - python advanced_shortterm_strategy.py"
echo "      - streamlit run streamlit_dashboard.py"
echo "      - python simple_data_feed.py"
echo ""
echo "📚 For testing: python test_suite.py"