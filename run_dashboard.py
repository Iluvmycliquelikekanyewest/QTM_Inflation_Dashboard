#!/usr/bin/env python3
"""
Dashboard Launcher Script
Simple script to launch the Monetary Inflation Dashboard GUI.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit dashboard."""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Path to the dashboard GUI file
    dashboard_file = script_dir / "dashboard_gui.py"
    
    if not dashboard_file.exists():
        print(f"❌ Dashboard file not found: {dashboard_file}")
        print("Make sure dashboard_gui.py is in the same directory as this script.")
        return 1
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit found")
    except ImportError:
        print("❌ Streamlit not installed")
        print("Install with: pip install streamlit")
        return 1
    
    # Launch the dashboard
    print("🚀 Launching Monetary Inflation Dashboard...")
    print("The dashboard will open in your web browser.")
    print("Press Ctrl+C to stop the server.")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_file),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())