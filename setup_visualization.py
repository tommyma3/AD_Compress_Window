"""
Quick setup script to ensure all dependencies for attention visualization are installed.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_and_install():
    """Check for required packages and install if missing."""
    packages_to_check = [
        ('seaborn', 'seaborn'),
    ]
    
    missing_packages = []
    
    for package_name, pip_name in packages_to_check:
        try:
            __import__(package_name)
            print(f"✓ {package_name} is already installed")
        except ImportError:
            print(f"✗ {package_name} is not installed")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            print(f"Installing {package}...")
            install_package(package)
        print("\n✓ All dependencies installed successfully!")
    else:
        print("\n✓ All dependencies are already installed!")
    
    print("\nYou can now run: python visualize_attention.py")

if __name__ == '__main__':
    check_and_install()
