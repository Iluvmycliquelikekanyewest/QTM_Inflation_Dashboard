#!/usr/bin/env python3
"""
Desktop GUI Launcher for Monetary Inflation Dashboard
Launches the native desktop application (no web browser needed!)
"""

import sys
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

def main():
    """Launch the desktop GUI application."""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Path to the desktop GUI file
    gui_file = script_dir / "desktop_gui.py"
    
    if not gui_file.exists():
        print(f"❌ Desktop GUI file not found: {gui_file}")
        print("Make sure desktop_gui.py is in the same directory as this script.")
        return 1
    
    # Check if tkinter is available (should be built-in)
    try:
        import tkinter
        print("✅ Tkinter found")
    except ImportError:
        print("❌ Tkinter not available")
        print("Tkinter should be included with Python. Try reinstalling Python.")
        return 1
    
    # Check if matplotlib is installed
    try:
        import matplotlib
        print("✅ Matplotlib found")
    except ImportError:
        print("❌ Matplotlib not installed")
        print("Install with: pip install matplotlib")
        return 1
    
    # Launch the desktop GUI
    print("🚀 Launching Desktop Monetary Inflation Dashboard...")
    print("A desktop window will open - no web browser needed!")
    
    try:
        # Import and run the desktop GUI
        sys.path.insert(0, str(script_dir))
        from desktop_gui import main as gui_main
        gui_main()
        
    except KeyboardInterrupt:
        print("\n👋 Desktop application stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error launching desktop GUI: {e}")
        # Show error in popup if possible
        try:
            root = tk.Tk()
            root.withdraw()  # Hide main window
            messagebox.showerror("Launch Error", f"Failed to launch desktop GUI:\n{e}")
            root.destroy()
        except:
            pass
        return 1

if __name__ == "__main__":
    sys.exit(main())