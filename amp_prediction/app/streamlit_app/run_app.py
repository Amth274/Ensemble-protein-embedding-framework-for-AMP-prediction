#!/usr/bin/env python3
"""
Launch script for the AMP Prediction Streamlit app.

This script provides a convenient way to launch the Streamlit application
with proper error handling and setup instructions.
"""

import sys
import subprocess
from pathlib import Path
import os

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'torch'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    return True

def main():
    """Main function to launch the Streamlit app."""
    print("Enhanced AMP Prediction Demo")
    print("=" * 40)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Get app directory
    app_dir = Path(__file__).parent
    app_file = app_dir / "app.py"

    if not app_file.exists():
        print(f"❌ App file not found: {app_file}")
        sys.exit(1)

    print("All dependencies found")
    print(f"Launching Streamlit app from {app_file}")
    print("\n" + "=" * 40)
    print("The app will open in your default web browser")
    print("Usually at: http://localhost:8501")
    print("Press Ctrl+C to stop the app")
    print("=" * 40 + "\n")

    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_file),
            "--server.headless", "false",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch app: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()