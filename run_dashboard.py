"""
Simple Desktop GUI for the Monetary Inflation Dashboard
Lightweight version that avoids recursion issues.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import pandas as pd
import threading
import sys
import os
from pathlib import Path
from datetime import datetime
import traceback

# Add src directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / 'src'))

class SimpleMonetaryGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("💰 Monetary Inflation Dashboard")
        self.root.geometry("1000x700")
        
        # Initialize variables
        self.results_df = None
        self.analysis_running = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls
        self.create_controls(main_frame)
        
        # Right panel - Results
        self.create_results_area(main_frame)
        
        # Bottom panel - Log
        self.create_log_area(main_frame)
        
    def create_controls(self, parent):
        """Create the control panel."""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Title
        title_label = ttk.Label(control_frame, text="💰 Dashboard", font=('Arial', 12, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Date inputs
        ttk.Label(control_frame, text="Start Date:").pack(anchor=tk.W)
        self.start_date_var = tk.StringVar(value="2020-01-01")
        start_entry = ttk.Entry(control_frame, textvariable=self.start_date_var, width=20)
        start_entry.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(control_frame, text="End Date:").pack(anchor=tk.W)
        self.end_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        end_entry = ttk.Entry(control_frame, textvariable=self.end_date_var, width=20)
        end_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Analysis type
        ttk.Label(control_frame, text="Analysis Type:").pack(anchor=tk.W)
        self.analysis_type_var = tk.StringVar(value="GDP Proxy")
        analysis_combo = ttk.Combobox(control_frame, textvariable=self.analysis_type_var, 
                                    values=["GDP Proxy", "Velocity", "Inflation"], state="readonly")
        analysis_combo.pack(fill=tk.X, pady=(0, 10))
        
        # SLCE option
        self.use_slce_var = tk.BooleanVar(value=True)
        slce_check = ttk.Checkbutton(control_frame, text="Use SLCE for State/Local Gov", 
                                   variable=self.use_slce_var)
        slce_check.pack(anchor=tk.W, pady=(0, 10))
        
        # Buttons
        self.run_button = ttk.Button(control_frame, text="🚀 Run Analysis", command=self.run_analysis_safe)
        self.run_button.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(control_frame, text="📁 Load Cache", command=self.load_cache).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(control_frame, text="💾 Export", command=self.export_data).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(control_frame, text="🗑️ Clear Log", command=self.clear_log).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(control_frame, text="🚪 Exit", command=self.root.quit).pack(fill=tk.X, pady=(10, 0))
        
    def create_results_area(self, parent):
        """Create the results area."""
        results_frame = ttk.LabelFrame(parent, text="Results", padding="10")
        results_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(results_frame)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Summary tab
        self.summary_frame = ttk.Frame(notebook)
        notebook.add(self.summary_frame, text="📊 Summary")
        
        self.summary_text = scrolledtext.ScrolledText(self.summary_frame, height=15, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)
        
        # Data tab
        self.data_frame = ttk.Frame(notebook)
        notebook.add(self.data_frame, text="📋 Data")
        
        self.data_text = scrolledtext.ScrolledText(self.data_frame, height=15, wrap=tk.NONE)
        self.data_text.pack(fill=tk.BOTH, expand=True)
        
    def create_log_area(self, parent):
        """Create the log area."""
        log_frame = ttk.LabelFrame(parent, text="Analysis Log", padding="5")
        log_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def log_message(self, message):
        """Add a message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def run_analysis_safe(self):
        """Run analysis in a thread to prevent GUI freezing."""
        if self.analysis_running:
            messagebox.showwarning("Analysis Running", "Analysis is already running. Please wait.")
            return
            
        # Disable button
        self.run_button.config(state='disabled', text="⏳ Running...")
        self.analysis_running = True
        
        # Start analysis in thread
        thread = threading.Thread(target=self.run_analysis_thread, daemon=True)
        thread.start()
        
    def run_analysis_thread(self):
        """Run the analysis in a separate thread."""
        try:
            self.log_message("🚀 Starting analysis...")
            
            # Import here to avoid circular imports
            from src.data_fetch import get_fred_gdp_components, get_bea_gdp_shares
            from src.gdp_proxy import build_gdp_proxy
            from src.velocity import calc_velocity
            from src.weight_manager import prev_qtr_shares, shares_monthly
            from src.config import Config
            
            # Validate inputs
            start_str = self.start_date_var.get()
            end_str = self.end_date_var.get()
            analysis_type = self.analysis_type_var.get()
            use_slce = self.use_slce_var.get()
            
            # Validate dates
            try:
                start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
                end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
                if start_date >= end_date:
                    raise ValueError("Start date must be before end date")
            except ValueError as e:
                self.log_message(f"❌ Date error: {e}")
                return
            
            # Check lag needed (simple check)
            today = datetime.now().date()
            lag_quarters = 1 if (today - end_date).days < 90 else 0
            
            if lag_quarters > 0:
                self.log_message(f"⚠️ Using {lag_quarters}Q lag due to recent end date")
            
            # Step 1: Fetch data
            self.log_message("📊 Fetching FRED data...")
            fred_data = get_fred_gdp_components(start_date=start_str)
            self.log_message(f"✅ Got {len(fred_data)} FRED observations")
            
            self.log_message("📈 Fetching BEA shares...")
            bea_shares = get_bea_gdp_shares()
            self.log_message(f"✅ Got {len(bea_shares)} BEA observations")
            
            # Step 2: Create weights
            self.log_message("⚖️ Creating monthly weights...")
            lagged_shares = prev_qtr_shares(bea_shares, lag_quarters=lag_quarters)
            monthly_weights = shares_monthly(lagged_shares, method='ffill')
            self.log_message(f"✅ Created {len(monthly_weights)} monthly weights")
            
            # Step 3: Build GDP proxy
            self.log_message("🏗️ Building GDP proxy...")
            gdp_proxy_df = build_gdp_proxy(
                start_date=start_str, 
                export=True, 
                strict=False, 
                use_slce=use_slce
            )
            self.log_message(f"✅ Built GDP proxy with {len(gdp_proxy_df)} observations")
            
            # Step 4: Calculate velocity if requested
            if analysis_type in ["Velocity", "Inflation"]:
                self.log_message("💰 Calculating velocity...")
                if 'GDP_proxy' in gdp_proxy_df.columns and 'M2' in fred_data.columns:
                    # Align data
                    aligned_data = pd.DataFrame({
                        'gdp_proxy': gdp_proxy_df['GDP_proxy'],
                        'money_supply': fred_data['M2']
                    }).dropna()
                    
                    if len(aligned_data) > 0:
                        velocity_df = calc_velocity(aligned_data['gdp_proxy'], aligned_data['money_supply'])
                        
                        # Combine results
                        self.results_df = pd.concat([gdp_proxy_df, velocity_df], axis=1)
                        self.results_df = self.results_df.loc[:, ~self.results_df.columns.duplicated()]
                        self.log_message(f"✅ Calculated velocity for {len(velocity_df)} periods")
                    else:
                        self.log_message("⚠️ No overlapping data for velocity calculation")
                        self.results_df = gdp_proxy_df
                else:
                    self.log_message("⚠️ Missing GDP proxy or M2 data for velocity")
                    self.results_df = gdp_proxy_df
            else:
                self.results_df = gdp_proxy_df
            
            self.log_message("✅ Analysis completed successfully!")
            
            # Update results display
            self.root.after(0, self.update_results)
            
        except Exception as e:
            error_msg = f"❌ Analysis failed: {str(e)}"
            self.log_message(error_msg)
            self.log_message(f"Error details: {traceback.format_exc()}")
        finally:
            # Re-enable button
            self.root.after(0, self.reset_button)
            
    def reset_button(self):
        """Reset the run button."""
        self.run_button.config(state='normal', text="🚀 Run Analysis")
        self.analysis_running = False
        
    def update_results(self):
        """Update the results display."""
        if self.results_df is None or self.results_df.empty:
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(tk.END, "No results to display.")
            return
            
        analysis_type = self.analysis_type_var.get()
        
        # Update summary
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, f"📊 {analysis_type.upper()} ANALYSIS RESULTS\n")
        self.summary_text.insert(tk.END, "=" * 50 + "\n\n")
        
        # Basic info
        self.summary_text.insert(tk.END, f"Dataset Shape: {self.results_df.shape}\n")
        self.summary_text.insert(tk.END, f"Date Range: {self.results_df.index.min()} to {self.results_df.index.max()}\n")
        self.summary_text.insert(tk.END, f"Available Columns: {', '.join(self.results_df.columns)}\n\n")
        
        # Analysis-specific metrics
        if analysis_type == "GDP Proxy":
            self.show_gdp_summary()
        elif analysis_type == "Velocity":
            self.show_velocity_summary()
        elif analysis_type == "Inflation":
            self.show_inflation_summary()
            
        # Update data tab
        self.update_data_tab()
        
    def show_gdp_summary(self):
        """Show GDP-specific summary."""
        gdp_col = None
        for col in self.results_df.columns:
            if 'gdp' in col.lower() and 'proxy' in col.lower():
                gdp_col = col
                break
                
        if gdp_col:
            latest_gdp = self.results_df[gdp_col].iloc[-1]
            annual_gdp = latest_gdp * 12
            self.summary_text.insert(tk.END, f"Latest Monthly GDP: ${latest_gdp:,.0f}B\n")
            self.summary_text.insert(tk.END, f"Annualized GDP: ${annual_gdp:,.0f}B\n\n")
            
            if len(self.results_df) >= 12:
                yoy_growth = (latest_gdp / self.results_df[gdp_col].iloc[-13] - 1) * 100
                self.summary_text.insert(tk.END, f"YoY Growth: {yoy_growth:.1f}%\n")
        
        # Component breakdown
        component_cols = [col for col in self.results_df.columns if col in ['C', 'I', 'G', 'NX']]
        if component_cols:
            latest_components = self.results_df[component_cols].iloc[-1]
            total = latest_components.sum()
            self.summary_text.insert(tk.END, "\nGDP Components (Latest Month):\n")
            for comp in component_cols:
                value = latest_components[comp]
                share = (value / total * 100) if total > 0 else 0
                self.summary_text.insert(tk.END, f"  {comp}: ${value:.0f}B ({share:.1f}%)\n")
                
    def show_velocity_summary(self):
        """Show velocity-specific summary."""
        velocity_cols = [col for col in self.results_df.columns if 'velocity' in col.lower()]
        if velocity_cols:
            velocity_col = velocity_cols[0]
            current_velocity = self.results_df[velocity_col].iloc[-1]
            avg_velocity = self.results_df[velocity_col].mean()
            
            self.summary_text.insert(tk.END, f"Current Velocity: {current_velocity:.3f}\n")
            self.summary_text.insert(tk.END, f"Average Velocity: {avg_velocity:.3f}\n")
            
            if len(self.results_df) >= 12:
                velocity_change = current_velocity - self.results_df[velocity_col].iloc[-13]
                self.summary_text.insert(tk.END, f"YoY Change: {velocity_change:.3f}\n")
                
    def show_inflation_summary(self):
        """Show inflation-specific summary."""
        inflation_cols = [col for col in self.results_df.columns if 'inflation' in col.lower()]
        if inflation_cols:
            inflation_col = inflation_cols[0]
            current_inflation = self.results_df[inflation_col].iloc[-1]
            
            # Handle different inflation formats
            if abs(current_inflation) < 1:
                inflation_pct = current_inflation * 100
            else:
                inflation_pct = current_inflation
                
            self.summary_text.insert(tk.END, f"Current Inflation: {inflation_pct:.2f}%\n")
            
            if len(self.results_df) >= 12:
                past_inflation = self.results_df[inflation_col].iloc[-13]
                if abs(past_inflation) < 1:
                    past_pct = past_inflation * 100
                else:
                    past_pct = past_inflation
                change = inflation_pct - past_pct
                self.summary_text.insert(tk.END, f"YoY Change: {change:.2f} percentage points\n")
                
    def update_data_tab(self):
        """Update the data tab."""
        self.data_text.delete(1.0, tk.END)
        
        if self.results_df is None or self.results_df.empty:
            self.data_text.insert(tk.END, "No data available.")
            return
            
        # Show recent data (last 20 rows)
        recent_data = self.results_df.tail(20)
        
        # Format as simple table
        self.data_text.insert(tk.END, "Recent Data (Last 20 Observations):\n")
        self.data_text.insert(tk.END, "=" * 80 + "\n\n")
        
        # Headers
        headers = ["Date"] + list(recent_data.columns)
        header_line = "  ".join(f"{h:<12}" for h in headers)
        self.data_text.insert(tk.END, header_line + "\n")
        self.data_text.insert(tk.END, "-" * len(header_line) + "\n")
        
        # Data rows
        for idx, row in recent_data.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            values = [date_str] + [f"{val:.2f}" if pd.notna(val) else "N/A" for val in row]
            value_line = "  ".join(f"{v:<12}" for v in values)
            self.data_text.insert(tk.END, value_line + "\n")
            
    def load_cache(self):
        """Load cached results."""
        try:
            cache_file = Path("data") / "gdp_proxy_monthly.csv"
            if cache_file.exists():
                self.results_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                self.log_message(f"✅ Loaded {len(self.results_df)} cached observations")
                self.update_results()
            else:
                messagebox.showinfo("No Cache", "No cached results found.")
        except Exception as e:
            self.log_message(f"❌ Error loading cache: {e}")
            messagebox.showerror("Load Error", f"Failed to load cache: {e}")
            
    def export_data(self):
        """Export results to file."""
        if self.results_df is None or self.results_df.empty:
            messagebox.showwarning("No Data", "No results to export.")
            return
            
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export Results"
            )
            
            if filename:
                self.results_df.to_csv(filename)
                self.log_message(f"✅ Exported to {filename}")
                messagebox.showinfo("Export Success", f"Results exported to:\n{filename}")
        except Exception as e:
            self.log_message(f"❌ Export error: {e}")
            messagebox.showerror("Export Error", f"Failed to export: {e}")
            
    def clear_log(self):
        """Clear the log display."""
        self.log_text.delete(1.0, tk.END)


def main():
    """Main function to run the simple desktop GUI."""
    try:
        # Check API keys
        try:
            from src.config import Config
            Config.validate_keys()
        except Exception as e:
            # Create error window
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Configuration Error", f"API keys not configured:\n{e}")
            root.destroy()
            return
        
        # Create and run GUI
        root = tk.Tk()
        app = SimpleMonetaryGUI(root)
        
        # Closing protocol
        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit the dashboard?"):
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Center window
        root.update_idletasks()
        width = 1000
        height = 700
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Start GUI
        print("🚀 Starting Simple Desktop GUI...")
        root.mainloop()
        print("👋 Desktop GUI closed")
        
    except Exception as e:
        print(f"❌ Error starting GUI: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()