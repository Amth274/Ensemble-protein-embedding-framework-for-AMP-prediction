#!/usr/bin/env python3
"""
Flask App Launcher for AMP Prediction Demo

This script launches the Flask web application for antimicrobial peptide prediction.
Provides a clean interface to start the server with appropriate configurations.
"""

import os
import sys
import webbrowser
from pathlib import Path

def setup_environment():
    """Setup the Python path and environment for the Flask app."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"

    # Add paths to Python path if not already present
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Set Flask environment variables
    os.environ['FLASK_APP'] = 'flask_app.app'
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'

def main():
    """Main function to launch the Flask app."""
    print("AMP Prediction Flask App Launcher")
    print("=" * 40)

    # Setup environment
    setup_environment()

    # Import Flask app
    try:
        from flask_app.app import app
        print("Flask app imported successfully!")
    except ImportError as e:
        print(f"Error importing Flask app: {e}")
        print("Make sure you're in the correct directory and dependencies are installed.")
        return 1

    # Configuration
    host = '127.0.0.1'
    port = 5000
    debug = True

    print(f"\nStarting Flask development server...")
    print(f"URL: http://{host}:{port}")
    print(f"Debug mode: {debug}")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 40)

    # Auto-open browser after a short delay
    try:
        import threading
        import time

        def open_browser():
            time.sleep(1.5)  # Wait for server to start
            webbrowser.open(f'http://{host}:{port}')

        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
    except Exception as e:
        print(f"Could not auto-open browser: {e}")

    # Start the Flask development server
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            use_reloader=True,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\nShutting down Flask server...")
    except Exception as e:
        print(f"Error starting Flask server: {e}")
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())