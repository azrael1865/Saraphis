#!/usr/bin/env python3
"""
Interactive dependency installation helper for Saraphis Independent Core.
Handles installation and verification of all required and optional dependencies.
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    min_version = (3, 7)
    current_version = sys.version_info[:2]
    
    print(f"Python version: {sys.version}")
    
    if current_version >= min_version:
        print("✓ Python version is compatible")
        return True
    else:
        print(f"✗ Python {min_version[0]}.{min_version[1]}+ required, got {current_version[0]}.{current_version[1]}")
        return False

def check_pip_available() -> bool:
    """Check if pip is available."""
    try:
        import pip
        result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ pip is available: {result.stdout.strip()}")
            return True
        else:
            print("✗ pip not working properly")
            return False
    except ImportError:
        print("✗ pip not available")
        return False

def install_package(package_name: str, pip_name: Optional[str] = None) -> bool:
    """Install a single package."""
    install_name = pip_name or package_name
    
    print(f"\nInstalling {install_name}...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', install_name
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ {install_name} installed successfully")
            return True
        else:
            print(f"✗ Failed to install {install_name}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error installing {install_name}: {e}")
        return False

def test_import(module_name: str, import_name: Optional[str] = None) -> bool:
    """Test if a module can be imported."""
    test_name = import_name or module_name
    
    try:
        importlib.import_module(test_name)
        print(f"✓ {module_name} imports successfully")
        return True
    except ImportError:
        print(f"✗ {module_name} import failed")
        return False
    except Exception as e:
        print(f"⚠️  {module_name} import issue: {e}")
        return False

def install_from_requirements() -> bool:
    """Install all packages from requirements.txt."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("✗ requirements.txt not found")
        return False
    
    print(f"\nInstalling from {requirements_file}...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ All requirements installed successfully")
            return True
        else:
            print("✗ Some packages failed to install")
            print(f"Error output: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def test_core_dependencies() -> Dict[str, bool]:
    """Test core dependencies required for basic functionality."""
    core_deps = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'psutil': 'psutil',
        'pathlib': 'pathlib',  # Built-in but test anyway
        'json': 'json',  # Built-in
        'yaml': 'yaml'
    }
    
    print("\nTesting core dependencies...")
    results = {}
    
    for name, import_name in core_deps.items():
        results[name] = test_import(name, import_name)
    
    return results

def test_accuracy_tracking_dependencies() -> Dict[str, bool]:
    """Test dependencies specific to accuracy tracking."""
    accuracy_deps = {
        'PyJWT': 'jwt',
        'toml': 'toml',
        'cryptography': 'cryptography',
        'fastapi': 'fastapi',
        'httpx': 'httpx'
    }
    
    print("\nTesting accuracy tracking dependencies...")
    results = {}
    
    for name, import_name in accuracy_deps.items():
        results[name] = test_import(name, import_name)
    
    return results

def test_optional_dependencies() -> Dict[str, bool]:
    """Test optional dependencies for enhanced functionality."""
    optional_deps = {
        'torch': 'torch',
        'sklearn': 'sklearn',
        'sqlalchemy': 'sqlalchemy',
        'pytest': 'pytest',
        'uvicorn': 'uvicorn'
    }
    
    print("\nTesting optional dependencies...")
    results = {}
    
    for name, import_name in optional_deps.items():
        results[name] = test_import(name, import_name)
    
    return results

def test_accuracy_tracking_modules() -> bool:
    """Test if accuracy tracking modules work with current dependencies."""
    print("\nTesting accuracy tracking integration...")
    
    try:
        # Test dependency checker
        import dependency_checker
        print("✓ dependency_checker module works")
        
        # Test dependency report
        dependency_checker.print_dependency_report()
        
        # Test JWT functionality
        try:
            token = dependency_checker.jwt.encode({'test': 'data'}, 'secret')
            decoded = dependency_checker.jwt.decode(token, 'secret', algorithms=['HS256'])
            print("✓ JWT functionality works")
        except Exception as e:
            print(f"⚠️  JWT using fallback: {e}")
        
        # Test TOML functionality
        try:
            config = dependency_checker.toml.loads('[test]\nkey = "value"')
            toml_str = dependency_checker.toml.dumps(config)
            print("✓ TOML functionality works")
        except Exception as e:
            print(f"⚠️  TOML using fallback: {e}")
        
        # Test accuracy tracking diagnostics
        import accuracy_tracking_diagnostics
        print("✓ accuracy_tracking_diagnostics imports successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Accuracy tracking test failed: {e}")
        return False

def install_individual_packages():
    """Interactive installation of individual packages."""
    packages = [
        ('PyJWT', 'PyJWT', 'JWT authentication support'),
        ('toml', 'toml', 'TOML configuration format support'),
        ('cryptography', 'cryptography', 'Secure encryption support'),
        ('FastAPI', 'fastapi[all]', 'Web API framework'),
        ('PyTorch', 'torch', 'Deep learning framework'),
        ('scikit-learn', 'scikit-learn', 'Machine learning library'),
        ('pytest', 'pytest', 'Testing framework')
    ]
    
    print("\nAvailable packages for individual installation:")
    for i, (name, pip_name, description) in enumerate(packages, 1):
        print(f"{i}. {name} - {description}")
    
    while True:
        try:
            choice = input("\nEnter package number to install (or 'done' to finish): ")
            if choice.lower() == 'done':
                break
            
            idx = int(choice) - 1
            if 0 <= idx < len(packages):
                name, pip_name, description = packages[idx]
                install_package(name, pip_name)
            else:
                print("Invalid package number")
                
        except ValueError:
            print("Please enter a number or 'done'")
        except KeyboardInterrupt:
            print("\nInstallation cancelled")
            break

def main():
    """Main installation and verification routine."""
    print("=" * 60)
    print("SARAPHIS DEPENDENCY INSTALLATION HELPER")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\nPlease upgrade Python to 3.7 or higher")
        return False
    
    # Check pip
    if not check_pip_available():
        print("\nPlease install pip")
        return False
    
    # Installation options
    print("\nInstallation Options:")
    print("1. Install all dependencies from requirements.txt")
    print("2. Install individual packages")
    print("3. Test current installation")
    print("4. Skip installation and test only")
    
    try:
        choice = input("\nChoose option (1-4): ")
        
        if choice == '1':
            print("\nInstalling all dependencies...")
            success = install_from_requirements()
            if not success:
                print("\nSome dependencies failed to install")
                print("You can try installing individual packages (option 2)")
                
        elif choice == '2':
            install_individual_packages()
            
        elif choice == '3':
            print("\nTesting current installation...")
            
        elif choice == '4':
            print("\nSkipping installation, testing only...")
            
        else:
            print("Invalid choice, testing current installation...")
    
    except KeyboardInterrupt:
        print("\nInstallation cancelled")
        return False
    
    # Test installations
    print("\n" + "=" * 60)
    print("TESTING DEPENDENCIES")
    print("=" * 60)
    
    # Test all dependency categories
    core_results = test_core_dependencies()
    accuracy_results = test_accuracy_tracking_dependencies()
    optional_results = test_optional_dependencies()
    
    # Summary
    print("\n" + "=" * 60)
    print("INSTALLATION SUMMARY")
    print("=" * 60)
    
    def print_results(title: str, results: Dict[str, bool]):
        print(f"\n{title}:")
        for name, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {name}")
    
    print_results("Core Dependencies", core_results)
    print_results("Accuracy Tracking Dependencies", accuracy_results)
    print_results("Optional Dependencies", optional_results)
    
    # Test accuracy tracking integration
    print("\n" + "=" * 60)
    print("INTEGRATION TEST")
    print("=" * 60)
    
    integration_success = test_accuracy_tracking_modules()
    
    # Final recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    core_failed = sum(1 for success in core_results.values() if not success)
    accuracy_failed = sum(1 for success in accuracy_results.values() if not success)
    
    if core_failed == 0 and accuracy_failed == 0:
        print("\n✅ All core dependencies installed successfully!")
        print("The system is ready for full functionality.")
    elif core_failed > 0:
        print(f"\n⚠️  {core_failed} core dependencies missing.")
        print("Some basic functionality may not work.")
        print("Run 'pip install -r requirements.txt' to install missing packages.")
    elif accuracy_failed > 0:
        print(f"\n⚠️  {accuracy_failed} accuracy tracking dependencies missing.")
        print("Accuracy tracking will use fallback implementations.")
        print("Install missing packages for full functionality.")
    
    if integration_success:
        print("\n✅ Accuracy tracking integration test passed!")
    else:
        print("\n⚠️  Accuracy tracking integration has issues.")
        print("Check error messages above for details.")
    
    print("\nFor help with specific issues:")
    print("- Check requirements.txt for all dependencies")
    print("- Use 'pip install <package>' for individual packages")
    print("- Run this script again to retest")
    
    return core_failed == 0 and integration_success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)