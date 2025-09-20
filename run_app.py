#!/usr/bin/env python3
"""
Enhanced Explainable AI Dashboard - Application Runner

This is the main entry point for the Enhanced Explainable AI Dashboard.
It handles application initialization, dependency checking, and error handling.

To run the application:
1. Ensure all files are in the same directory:
   - config.py
   - logger.py  
   - app.py
   - run_app.py (this file)

2. Install dependencies:
   pip install streamlit pandas numpy scikit-learn shap lime plotly joblib google-generativeai

3. (Optional) Set environment variables:
   export OPENROUTER_API_KEY="your_api_key_here"

4. Run the app:
   streamlit run run_app.py
   
   Or directly:
   python run_app.py

Author: AI Dashboard Team
Version: 2.0
"""

import sys
import os
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any
import importlib.util


class DependencyChecker:
    """Check and validate required dependencies"""
    
    REQUIRED_PACKAGES = {
        'streamlit': 'streamlit>=1.28.0',
        'pandas': 'pandas>=1.5.0',
        'numpy': 'numpy>=1.21.0',
        'sklearn': 'scikit-learn>=1.3.0',
        'shap': 'shap>=0.42.0',
        'lime': 'lime>=0.2.0',
        'plotly': 'plotly>=5.15.0',
        'joblib': 'joblib>=1.3.0',
        'google.generativeai': 'google-generativeai>=0.3.0',
    }
    
    OPTIONAL_PACKAGES = {
        'xgboost': 'xgboost>=1.7.0',
        'lightgbm': 'lightgbm>=3.3.0',
        'catboost': 'catboost>=1.2.0',
    }
    
    @classmethod
    def check_package(cls, package_name: str) -> bool:
        """Check if a package is installed"""
        try:
            spec = importlib.util.find_spec(package_name)
            return spec is not None
        except ImportError:
            return False
    
    @classmethod
    def check_all_dependencies(cls) -> Dict[str, Any]:
        """Check all required and optional dependencies"""
        results = {
            'required': {},
            'optional': {},
            'missing_required': [],
            'missing_optional': []
        }
        
        # Check required packages
        for package, version_spec in cls.REQUIRED_PACKAGES.items():
            is_installed = cls.check_package(package)
            results['required'][package] = {
                'installed': is_installed,
                'version_spec': version_spec
            }
            if not is_installed:
                results['missing_required'].append(package)
        
        # Check optional packages
        for package, version_spec in cls.OPTIONAL_PACKAGES.items():
            is_installed = cls.check_package(package)
            results['optional'][package] = {
                'installed': is_installed,
                'version_spec': version_spec
            }
            if not is_installed:
                results['missing_optional'].append(package)
        
        return results
    
    @classmethod
    def print_dependency_report(cls, results: Dict[str, Any]) -> None:
        """Print a formatted dependency report"""
        print("\n" + "="*60)
        print("DEPENDENCY CHECK REPORT")
        print("="*60)
        
        # Required dependencies
        print("\nREQUIRED PACKAGES:")
        print("-" * 30)
        for package, info in results['required'].items():
            status = "✓ INSTALLED" if info['installed'] else "✗ MISSING"
            print(f"{package:<25} {status}")
        
        # Optional dependencies
        print("\nOPTIONAL PACKAGES:")
        print("-" * 30)
        for package, info in results['optional'].items():
            status = "✓ INSTALLED" if info['installed'] else "- NOT INSTALLED"
            print(f"{package:<25} {status}")
        
        # Missing packages
        if results['missing_required']:
            print("\n⚠️  MISSING REQUIRED PACKAGES:")
            for package in results['missing_required']:
                version_spec = results['required'][package]['version_spec']
                print(f"   pip install {version_spec}")
        
        if results['missing_optional']:
            print("\n💡 OPTIONAL PACKAGES (for enhanced functionality):")
            for package in results['missing_optional']:
                version_spec = results['optional'][package]['version_spec']
                print(f"   pip install {version_spec}")
        
        print("\n" + "="*60)


class AppRunner:
    """Main application runner with comprehensive error handling"""
    
    def __init__(self):
        self.app_directory = Path(__file__).parent
        self.required_files = ['config.py', 'logger.py', 'app.py']
        
    def check_file_structure(self) -> bool:
        """Check if all required files are present"""
        missing_files = []
        
        for filename in self.required_files:
            file_path = self.app_directory / filename
            if not file_path.exists():
                missing_files.append(filename)
        
        if missing_files:
            print(f"\n❌ Missing required files: {', '.join(missing_files)}")
            print(f"📁 Looking in directory: {self.app_directory}")
            print("\nEnsure all files are in the same directory:")
            for filename in self.required_files:
                print(f"   - {filename}")
            return False
        
        print(f"✅ All required files found in {self.app_directory}")
        return True
    
    def setup_environment(self) -> bool:
        """Setup environment and check system requirements"""
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                print(f"❌ Python 3.8+ required. Current version: {python_version.major}.{python_version.minor}")
                return False
            
            print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            
            # Add current directory to Python path
            current_dir = str(self.app_directory)
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            return True
            
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            return False
    
    def run_dependency_check(self) -> bool:
        """Run comprehensive dependency check"""
        print("\n🔍 Checking dependencies...")
        
        try:
            results = DependencyChecker.check_all_dependencies()
            DependencyChecker.print_dependency_report(results)
            
            if results['missing_required']:
                print(f"\n❌ Cannot start application. {len(results['missing_required'])} required packages missing.")
                print("\nInstall missing packages with:")
                print("pip install " + " ".join([
                    DependencyChecker.REQUIRED_PACKAGES[pkg] 
                    for pkg in results['missing_required']
                ]))
                return False
            
            print(f"\n✅ All required dependencies satisfied!")
            if results['missing_optional']:
                print(f"💡 {len(results['missing_optional'])} optional packages not installed (enhanced features may be limited)")
            
            return True
            
        except Exception as e:
            print(f"❌ Dependency check failed: {e}")
            return False
    
    def import_application(self):
        """Import the main application with error handling"""
        try:
            print("\n📦 Importing application modules...")
            
            # Import in order of dependency
            from config import Config
            print("✅ Config module loaded")
            
            from logger import LogManager
            print("✅ Logger module loaded")
            
            from app import create_app
            print("✅ App module loaded")
            
            return create_app
            
        except ImportError as e:
            print(f"\n❌ Import error: {e}")
            print("\nTroubleshooting steps:")
            print("1. Ensure all files are in the same directory")
            print("2. Check that all dependencies are installed")
            print("3. Verify Python version compatibility (3.8+)")
            raise
            
        except Exception as e:
            print(f"\n❌ Unexpected import error: {e}")
            raise
    
    def run_application(self) -> None:
        """Run the main application"""
        try:
            print("\n🚀 Starting Enhanced Explainable AI Dashboard...")
            print("   Open your browser to the URL shown below")
            print("   Press Ctrl+C to stop the application")
            print("-" * 60)
            
            # Import and run the app
            create_app = self.import_application()
            create_app()
            
        except KeyboardInterrupt:
            print("\n\n👋 Application stopped by user")
            sys.exit(0)
            
        except Exception as e:
            print(f"\n❌ Application runtime error: {e}")
            print("\nFor support, please check:")
            print("1. Error logs in the application")
            print("2. Streamlit documentation: https://docs.streamlit.io/")
            print("3. Project requirements and setup instructions")
            raise
    
    def run(self) -> None:
        """Main entry point - run the complete application stack"""
        try:
            print("🤖 Enhanced Explainable AI Dashboard")
            print("="*50)
            
            # Check file structure
            if not self.check_file_structure():
                sys.exit(1)
            
            # Setup environment
            if not self.setup_environment():
                sys.exit(1)
            
            # Check dependencies
            if not self.run_dependency_check():
                sys.exit(1)
            
            # Run application
            self.run_application()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)
            
        except Exception as e:
            print(f"\n💥 Critical error: {e}")
            print("\nIf this error persists:")
            print("1. Check all files are present and correct")
            print("2. Reinstall dependencies: pip install -r requirements.txt")
            print("3. Verify Python version: python --version")
            sys.exit(1)


def main():
    """Main function - entry point for the application"""
    runner = AppRunner()
    runner.run()


if __name__ == "__main__":
    main()