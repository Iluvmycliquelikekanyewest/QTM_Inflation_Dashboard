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
    
    print("🖥️  Monetary Inflation Dashboard - Desktop Version")
    print("=" * 50)
    
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
        
        # Show popup error too
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Missing Dependency", 
                               "Matplotlib is required but not installed.\n\n"
                               "Install it with:\n"
                               "pip install matplotlib")
            root.destroy()
        except:
            pass
        return 1
    
    # Launch the desktop GUI
    print("🚀 Launching Desktop GUI...")
    print("A desktop window will open - no web browser needed!")
    print("Close the window or click Exit to quit.")
    print()
    
    try:
        # Import and run the desktop GUI directly
        sys.path.insert(0, str(script_dir))
        
        # Import the main function from desktop_gui
        from desktop_gui import main as gui_main
        
        # Run the GUI
        gui_main()
        
        print("👋 Desktop application closed")
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Desktop application stopped by user")
        return 0
    except ImportError as e:
        error_msg = f"Failed to import desktop GUI modules: {e}"
        print(f"❌ {error_msg}")
        
        # Show popup error
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Import Error", error_msg)
            root.destroy()
        except:
            pass
        return 1
    except Exception as e:
        error_msg = f"Error launching desktop GUI: {e}"
        print(f"❌ {error_msg}")
        
        # Show popup error
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Launch Error", error_msg)
            root.destroy()
        except:
            pass
        return 1

if __name__ == "__main__":
    sys.exit(main())