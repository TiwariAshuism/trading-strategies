#!/usr/bin/env python3
"""
Python Setup Script for Trading System
Cross-platform setup and dependency installation
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

class TradingSystemSetup:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.venv_path = self.project_root / "env"
        
    def check_python_version(self):
        """Check Python version compatibility"""
        version = sys.version_info
        print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print(" Python 3.8 or higher is required")
            return False
        
        print(" Python version compatible")
        return True
    
    def setup_virtual_environment(self):
        """Create and setup virtual environment"""
        if self.venv_path.exists():
            print("🔧 Virtual environment already exists")
        else:
            print("📦 Creating virtual environment...")
            try:
                subprocess.run([sys.executable, "-m", "venv", "env"], 
                             cwd=self.project_root, check=True)
                print(" Virtual environment created")
            except subprocess.CalledProcessError as e:
                print(f" Error creating virtual environment: {e}")
                return False
        
        return True
    
    def get_pip_command(self):
        """Get the correct pip command for the platform"""
        if platform.system() == "Windows":
            return str(self.venv_path / "Scripts" / "pip")
        else:
            return str(self.venv_path / "bin" / "pip")
    
    def get_python_command(self):
        """Get the correct python command for the platform"""
        if platform.system() == "Windows":
            return str(self.venv_path / "Scripts" / "python")
        else:
            return str(self.venv_path / "bin" / "python")
    
    def install_dependencies(self):
        """Install required dependencies"""
        pip_cmd = self.get_pip_command()
        
        print("⬆️ Upgrading pip...")
        try:
            subprocess.run([pip_cmd, "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            print(" Pip upgraded")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Warning: Could not upgrade pip: {e}")
        
        print("📚 Installing dependencies...")
        try:
            subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], 
                         cwd=self.project_root, check=True)
            print(" Dependencies installed")
            return True
        except subprocess.CalledProcessError as e:
            print(f" Error installing dependencies: {e}")
            return False
    
    def run_diagnostic(self):
        """Run system diagnostic"""
        python_cmd = self.get_python_command()
        
        print("🔍 Running system diagnostic...")
        try:
            result = subprocess.run([python_cmd, "scripts/diagnostic.py"], 
                                  cwd=self.project_root, 
                                  capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Warnings:", result.stderr)
            return result.returncode == 0
        except Exception as e:
            print(f" Error running diagnostic: {e}")
            return False
    
    def create_activation_script(self):
        """Create easy activation script"""
        if platform.system() == "Windows":
            script_name = "activate.bat"
            content = f"""@echo off
cd /d "{self.project_root}"
call env\\Scripts\\activate.bat
echo  Trading system environment activated!
echo.
echo  Available commands:
echo   python trading_control_panel.py --start-all
echo   python advanced_shortterm_strategy.py  
echo   streamlit run streamlit_dashboard.py
echo   python test_suite.py
echo.
cmd /k
"""
        else:
            script_name = "activate.sh"
            content = f"""#!/bin/bash
cd "{self.project_root}"
source env/bin/activate
echo " Trading system environment activated!"
echo ""
echo " Available commands:"
echo "  python scripts/trading_control_panel.py --start-all"
echo "  python src/strategies/advanced_shortterm_strategy.py"
echo "  streamlit run src/ui/streamlit_dashboard.py"
echo "  python tests/test_suite.py"
echo ""
exec "$SHELL"
"""
        
        script_path = self.project_root / script_name
        with open(script_path, 'w') as f:
            f.write(content)
        
        if platform.system() != "Windows":
            os.chmod(script_path, 0o755)
        
        print(f"📝 Created activation script: {script_name}")
        return script_path
    
    def setup(self):
        """Run complete setup process"""
        print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        🚀 TRADING SYSTEM SETUP                            ║
║                                                           ║
║    Setting up your advanced algorithmic trading system   ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
        """)
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Setup virtual environment
        if not self.setup_virtual_environment():
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Run diagnostic
        diagnostic_ok = self.run_diagnostic()
        
        # Create activation script
        activation_script = self.create_activation_script()
        
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETE!")
        print("="*60)
        
        if diagnostic_ok:
            print(" All systems ready!")
        else:
            print("⚠️ Setup complete but some issues detected")
        
        print(f"\n🚀 Quick Start:")
        if platform.system() == "Windows":
            print(f"   1. Run: {activation_script.name}")
        else:
            print(f"   1. Run: source {activation_script.name}")
        print(f"   2. Or manually activate: source env/bin/activate")
        print(f"   3. Start system: python scripts/trading_control_panel.py --start-all")
        
        print(f"\n📚 Documentation:")
        print(f"   • README.md - Complete usage guide")
        print(f"   • python tests/test_suite.py - Run all tests")
        print(f"   • python scripts/diagnostic.py - System check")
        
        return True

def main():
    setup = TradingSystemSetup()
    success = setup.setup()
    
    if success:
        print("\n Your advanced trading system is ready to use!")
    else:
        print("\n Setup encountered errors. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()