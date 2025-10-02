# 📁 Project Structure

This document outlines the organized folder structure of the Advanced Trading System.

## 📂 Root Directory Structure

```
trading-strategies/
├── 📄 main.py                 # Main entry point - easy access to all features
├── 📄 README.md               # Complete documentation and usage guide
├── 📄 requirements.txt        # Python dependencies
├── 📄 .gitignore             # Git ignore patterns
│
├── 📁 src/                    # Source code directory
│   ├── 📁 strategies/         # Trading strategies and analysis
│   ├── 📁 data/              # Data management and feeds
│   ├── 📁 ui/                # User interfaces and dashboards
│   ├── 📁 trading/           # Automated trading and execution
│   └── 📁 config/            # Configuration and settings
│
├── 📁 scripts/               # Utility and setup scripts
├── 📁 tests/                 # Test suites and validation
├── 📁 docs/                  # Documentation files
├── 📁 data/                  # Data storage (databases, cache)
└── 📁 env/                   # Python virtual environment
```

##  Detailed Module Structure

###  src/strategies/ - Trading Strategies
```
strategies/
├── __init__.py
├── portfolio_analyzer.py          # Portfolio analysis with risk metrics
├── algotrade_nalco.py             # Multi-strategy stock screener
├── algotrade_news.py              # News-based sentiment trading
├── advanced_shortterm_strategy.py # Advanced short-term signals
└── strategy_backtester.py         # Historical performance testing
```

### 💾 src/data/ - Data Management
```
data/
├── __init__.py
├── database_manager.py            # SQLite database operations
├── realtime_data_feed.py          # WebSocket real-time data (advanced)
└── simple_data_feed.py            # Simple synchronous data feed
```

### 🌐 src/ui/ - User Interfaces
```
ui/
├── __init__.py
└── streamlit_dashboard.py         # Web dashboard and visualization
```

### 🤖 src/trading/ - Automated Trading
```
trading/
├── __init__.py
└── auto_trader.py                 # Automated order execution
```

### ⚙️ src/config/ - Configuration
```
config/
├── __init__.py
└── strategy_config.py             # Strategy parameters and settings
```

### 🔧 scripts/ - Utility Scripts
```
scripts/
├── trading_control_panel.py       # Master system controller
├── setup.py                       # Python-based setup script
├── setup.sh                       # Bash setup script
└── diagnostic.py                  # System diagnostic tool
```

### 🧪 tests/ - Testing Suite
```
tests/
├── __init__.py
└── test_suite.py                  # Comprehensive test suite
```

### 📚 docs/ - Documentation
```
docs/
└── ERROR_FIX.md                   # Error resolution documentation
```

### 💽 data/ - Data Storage
```
data/
└── trading_data.db                # SQLite database (created when needed)
```

##  Quick Start Guide

### Method 1: Main Entry Point (Recommended)
```bash
# Easy access to all features
python main.py --start-all          # Start complete system
python main.py --dashboard          # Launch web dashboard  
python main.py --portfolio          # Portfolio analyzer
python main.py --screener           # Stock screener
python main.py --shortterm          # Short-term strategy
python main.py --setup              # Run system setup
python main.py --test               # Run test suite
```

### Method 2: Direct Module Access
```bash
# Strategy modules
python src/strategies/portfolio_analyzer.py
python src/strategies/advanced_shortterm_strategy.py

# UI modules
streamlit run src/ui/streamlit_dashboard.py

# System control
python scripts/trading_control_panel.py --start-all
```

### Method 3: Package Imports (For Development)
```python
# Import from organized modules
from src.strategies.advanced_shortterm_strategy import AdvancedShortTermStrategy
from src.data.database_manager import TradingDatabase
from src.config.strategy_config import CONFIG
```

## 🔄 Migration Benefits

###  Organization Benefits:
- **Clear separation** of concerns
- **Easy navigation** and maintenance  
- **Professional structure** following Python best practices
- **Scalable architecture** for future additions
- **Better import management** with proper package structure

###  Usage Benefits:
- **Single entry point** (`main.py`) for all features
- **Consistent command interface** across all modules
- **Easier deployment** and distribution
- **Better error handling** with organized imports
- **Professional appearance** for sharing/collaboration

##  Development Workflow

1. **Setup**: Run `python main.py --setup` or `python scripts/setup.py`
2. **Development**: Work within the organized `src/` modules
3. **Testing**: Use `python main.py --test` or `python tests/test_suite.py`
4. **Usage**: Access features via `python main.py --<command>`
5. **Deployment**: The organized structure is ready for packaging

## 📝 Notes

- All imports have been updated to work with the new structure
- The main.py file provides backwards compatibility
- Virtual environment (env/) remains in the root for convenience
- Database files are stored in data/ directory
- Documentation is organized in docs/ directory

This structure follows Python packaging best practices and makes the system more professional and maintainable!