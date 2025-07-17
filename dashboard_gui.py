"""
Simple Desktop GUI for the Monetary Inflation Dashboard
Main GUI shell - handles UI setup and user interactions
FIXED: Now properly passes frequency parameter to analysis engine
UPDATED: Quarterly and Annual frequencies restricted to GDP Proxy only
"""


import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import pandas as pd
import threading
import sys
import os
from pathlib import Path
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import calendar
import traceback


# Add src directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / 'src'))


# Import our analysis and results modules
from analysis_logic import AnalysisEngine
from results_utils import ResultsFormatter


class SimpleMonetaryGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("💰 Monetary Inflation Dashboard")
        self.root.geometry("1000x700")
       
        # Initialize variables
        self.results_df = None
        self.analysis_running = False
       
        # Initialize our helper modules
        self.analysis_engine = AnalysisEngine(self)
        self.results_formatter = ResultsFormatter(self)
       
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
        """Create the control panel with enhanced date selection."""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
       
        # Title
        title_label = ttk.Label(control_frame, text="💰 Dashboard", font=('Arial', 12, 'bold'))
        title_label.pack(pady=(0, 10))
       
        # Analysis type - CREATE THIS FIRST
        ttk.Label(control_frame, text="Analysis Type:").pack(anchor=tk.W)
        self.analysis_type_var = tk.StringVar(value="GDP Proxy")
        self.analysis_combo = ttk.Combobox(control_frame, textvariable=self.analysis_type_var,
                                    values=["GDP Proxy", "Velocity", "Inflation"], state="readonly")
        self.analysis_combo.pack(fill=tk.X, pady=(0, 10))
        # Bind the analysis type change to update frequency options
        self.analysis_combo.bind('<<ComboboxSelected>>', self.on_analysis_type_changed)
       
        # ENHANCED DATE SELECTION SECTION - CREATE AFTER ANALYSIS TYPE
        self._create_date_controls(control_frame)
       
        # SLCE option
        self.use_slce_var = tk.BooleanVar(value=True)
        slce_check = ttk.Checkbutton(control_frame, text="Use SLCE for State/Local Gov",
                                   variable=self.use_slce_var)
        slce_check.pack(anchor=tk.W, pady=(0, 10))
       
        # Buttons
        self._create_action_buttons(control_frame)
       
        # Initialize
        self.calculate_dates()
       
    def _create_date_controls(self, parent):
        """Create the date selection controls."""
        date_frame = ttk.LabelFrame(parent, text="📅 Time Period", padding="8")
        date_frame.pack(fill=tk.X, pady=(0, 10))
       
        # Frequency selection
        ttk.Label(date_frame, text="Frequency:").pack(anchor=tk.W)
        self.frequency_var = tk.StringVar(value="Monthly")
        self.frequency_combo = ttk.Combobox(date_frame, textvariable=self.frequency_var,
                                     values=["Monthly"], state="readonly")  # Start with only Monthly
        self.frequency_combo.pack(fill=tk.X, pady=(0, 5))
        self.frequency_combo.bind('<<ComboboxSelected>>', self.update_period_label)
        
        # Update frequency options based on initial analysis type
        self.update_frequency_options()
       
        # Number of periods
        periods_frame = ttk.Frame(date_frame)
        periods_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(periods_frame, text="Periods:").pack(side=tk.LEFT)
        self.periods_var = tk.StringVar(value="12")
        periods_spin = tk.Spinbox(periods_frame, from_=1, to=60, width=8, textvariable=self.periods_var)
        periods_spin.pack(side=tk.LEFT, padx=(5, 5))
        self.period_label_var = tk.StringVar(value="months")
        ttk.Label(periods_frame, textvariable=self.period_label_var).pack(side=tk.LEFT)
       
        # End date selection (Year and Month only)
        self._create_end_date_controls(date_frame)
       
        # Calculated range display
        self.range_var = tk.StringVar()
        range_label = ttk.Label(date_frame, textvariable=self.range_var, foreground='blue', font=('Arial', 8))
        range_label.pack(pady=(5, 0))
       
        ttk.Button(date_frame, text="Update Range", command=self.calculate_dates).pack(pady=(2, 0))
    
    def on_analysis_type_changed(self, event=None):
        """Handle analysis type change to update frequency options."""
        self.update_frequency_options()
        self.update_period_label()  # Update the period label and defaults
        
    def update_frequency_options(self):
        """Update frequency options based on selected analysis type."""
        analysis_type = self.analysis_type_var.get()
        
        if analysis_type == "GDP Proxy":
            # GDP Proxy allows all frequencies
            available_frequencies = ["Monthly", "Quarterly", "Annually"]
        else:
            # Velocity and Inflation only allow Monthly
            available_frequencies = ["Monthly"]
            # Force Monthly if user had selected something else
            if self.frequency_var.get() not in available_frequencies:
                self.frequency_var.set("Monthly")
        
        # Update the combobox values
        self.frequency_combo['values'] = available_frequencies
        
        # Log the change if we have a log area
        if hasattr(self, 'log_text') and analysis_type in ["Velocity", "Inflation"]:
            self.log_message(f"ℹ️ {analysis_type} analysis only supports Monthly frequency")
       
    def _create_end_date_controls(self, parent):
        """Create end date selection controls."""
        end_frame = ttk.Frame(parent)
        end_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(end_frame, text="End:").pack(side=tk.LEFT)
       
        now = datetime.now()
        current_year = now.year
        current_month = now.month
       
        self.end_year_var = tk.StringVar(value=str(current_year))
        year_combo = ttk.Combobox(end_frame, textvariable=self.end_year_var,
                                values=[str(y) for y in range(2015, current_year + 2)],
                                state="readonly", width=8)
        year_combo.pack(side=tk.LEFT, padx=(5, 5))
       
        month_names = [calendar.month_name[i] for i in range(1, 13)]
        self.end_month_var = tk.StringVar(value=calendar.month_name[current_month])
        month_combo = ttk.Combobox(end_frame, textvariable=self.end_month_var,
                                 values=month_names, state="readonly", width=10)
        month_combo.pack(side=tk.LEFT)
       
    def _create_action_buttons(self, parent):
        """Create action buttons using a loop for DRY principle."""
        # Main action button
        self.run_button = ttk.Button(parent, text="🚀 Run Analysis", command=self.run_analysis_safe)
        self.run_button.pack(fill=tk.X, pady=(0, 5))
       
        # Other buttons configuration
        buttons_config = [
            ("📁 Load Cache", self.load_cache),
            ("💾 Export", self.export_data),
            ("🗑️ Clear Log", self.clear_log),
            ("🚪 Exit", self.root.quit)
        ]
       
        for i, (text, command) in enumerate(buttons_config):
            pady = (10, 0) if i == len(buttons_config) - 1 else (0, 5)  # Extra padding for Exit button
            ttk.Button(parent, text=text, command=command).pack(fill=tk.X, pady=pady)
       
    def update_period_label(self, event=None):
        """Update period label when frequency changes."""
        freq = self.frequency_var.get()
        period_configs = {
            "Monthly": ("months", "12"),
            "Quarterly": ("quarters", "8"),
            "Annually": ("years", "5")
        }
       
        if freq in period_configs:
            label, default_periods = period_configs[freq]
            self.period_label_var.set(label)
            self.periods_var.set(default_periods)
           
        self.calculate_dates()
       
    def get_last_complete_quarter_end(self, reference_date):
        """Get the end date of the last complete quarter relative to reference date."""
        # Quarter end months: March (Q1), June (Q2), September (Q3), December (Q4)
        quarter_end_months = [3, 6, 9, 12]
       
        # Find the last quarter end that has passed
        for quarter_end_month in reversed(quarter_end_months):
            quarter_end_date = date(reference_date.year, quarter_end_month,
                                  calendar.monthrange(reference_date.year, quarter_end_month)[1])
           
            # Check if this quarter end is at least 45 days ago (to ensure data availability)
            if (reference_date - quarter_end_date).days >= 45:
                return quarter_end_date
       
        # If no quarter in current year is complete, go to Q4 of previous year
        prev_year = reference_date.year - 1
        return date(prev_year, 12, 31)
       
    def _get_monthly_range(self, selected_end_date, periods, today):
        """Calculate date range for monthly frequency."""
        end_date = min(selected_end_date, today.replace(day=1) - relativedelta(days=1))
        start_date = end_date - relativedelta(months=periods-1)
        start_date = start_date.replace(day=1)
        return start_date, end_date
       
    def _get_quarterly_range(self, selected_end_date, end_year, end_month, periods, today):
        """Calculate date range for quarterly frequency."""
        last_complete_quarter = self.get_last_complete_quarter_end(today)
       
        # If user selected a date after the last complete quarter, use complete quarter
        if selected_end_date > last_complete_quarter:
            end_date = last_complete_quarter
            self.log_message(f"ℹ️ Adjusted end date to last complete quarter: {end_date}")
        else:
            # Use the selected quarter end
            quarter = ((end_month - 1) // 3) + 1
            quarter_end_month = quarter * 3
            end_date = date(end_year, quarter_end_month,
                           calendar.monthrange(end_year, quarter_end_month)[1])
       
        # Calculate start date by going back the requested number of quarters
        start_quarter_end = end_date - relativedelta(months=(periods-1)*3)
        # Adjust to quarter end
        start_quarter = ((start_quarter_end.month - 1) // 3) + 1
        start_quarter_end_month = start_quarter * 3
        start_date = date(start_quarter_end.year, start_quarter_end_month, 1)
       
        return start_date, end_date
       
    def _get_annual_range(self, selected_end_date, end_year, periods, today):
        """Calculate date range for annual frequency."""
        if selected_end_date.year >= today.year:
            end_date = date(today.year - 1, 12, 31)
        else:
            end_date = date(end_year, 12, 31)
       
        start_date = date(end_date.year - periods + 1, 1, 1)
        return start_date, end_date
       
    def calculate_dates(self):
        """Calculate start/end dates based on frequency selection with improved logic."""
        try:
            # 🚨 DEBUG: Add these lines at the very beginning
            print(f"🚨 calculate_dates() called!")
            print(f"🚨 End year var: {self.end_year_var.get()}")
            print(f"🚨 End month var: {self.end_month_var.get()}")
            print(f"🚨 Frequency var: {self.frequency_var.get()}")
            print(f"🚨 Periods var: {self.periods_var.get()}")
            
            # Also log to GUI (if log_text exists)
            if hasattr(self, 'log_text'):
                self.log_message(f"🚨 calculate_dates() called! Year: {self.end_year_var.get()}, Month: {self.end_month_var.get()}")
            
            # Get selected end date
            end_year = int(self.end_year_var.get())
            end_month = list(calendar.month_name).index(self.end_month_var.get())
            selected_end_date = date(end_year, end_month, calendar.monthrange(end_year, end_month)[1])
            
            # 🚨 DEBUG: Add these lines after calculating selected_end_date
            print(f"🚨 Calculated end_year: {end_year}")
            print(f"🚨 Calculated end_month: {end_month}")
            print(f"🚨 Calculated selected_end_date: {selected_end_date}")
            
            # Also log to GUI (if log_text exists)
            if hasattr(self, 'log_text'):
                self.log_message(f"🚨 Calculated end date: {selected_end_date}")
           
            freq = self.frequency_var.get()
            periods = int(self.periods_var.get())
            today = datetime.now().date()
           
            # Use helper methods for each frequency
            if freq == "Monthly":
                self.start_date, self.end_date = self._get_monthly_range(selected_end_date, periods, today)
            elif freq == "Quarterly":
                self.start_date, self.end_date = self._get_quarterly_range(
                    selected_end_date, end_year, end_month, periods, today)
            else:  # Annually
                self.start_date, self.end_date = self._get_annual_range(
                    selected_end_date, end_year, periods, today)
            
            # 🚨 DEBUG: Add these lines after calculating final dates
            print(f"🚨 FINAL start_date: {self.start_date}")
            print(f"🚨 FINAL end_date: {self.end_date}")
            
            # Also log to GUI (if log_text exists)
            if hasattr(self, 'log_text'):
                self.log_message(f"🚨 FINAL date range: {self.start_date} → {self.end_date}")
           
            # Update display with more informative text
            self._update_range_display(freq)
            
            # 🚨 DEBUG: Check what the display shows (if log_text exists)
            print(f"🚨 Display text: {self.range_var.get()}")
            if hasattr(self, 'log_text'):
                self.log_message(f"🚨 Display text: {self.range_var.get()}")
           
        except (ValueError, AttributeError) as e:
            print(f"🚨 ERROR in calculate_dates(): {e}")
            if hasattr(self, 'log_text'):
                self.log_message(f"🚨 ERROR in calculate_dates(): {e}")
            self.range_var.set(f"Invalid selection: {e}")
            self.start_date = None
            self.end_date = None
           
    def _update_range_display(self, freq):
        """Update the range display with calculated dates."""
        months = (self.end_date.year - self.start_date.year) * 12 + (self.end_date.month - self.start_date.month) + 1
       
        if freq == "Quarterly":
            quarters = ((self.end_date.year - self.start_date.year) * 4 +
                      ((self.end_date.month - 1) // 3) - ((self.start_date.month - 1) // 3) + 1)
            self.range_var.set(f"{self.start_date} → {self.end_date} ({quarters} quarters, {months} months)")
        elif freq == "Annually":
            years = self.end_date.year - self.start_date.year + 1
            self.range_var.set(f"{self.start_date} → {self.end_date} ({years} years, {months} months)")
        else:
            self.range_var.set(f"{self.start_date} → {self.end_date} ({months} months)")
           
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
        now = datetime.now()
        timestamp = now.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
       
    def run_analysis_safe(self):
        """Run analysis in a thread to prevent GUI freezing - FIXED with frequency parameter."""
        if self.analysis_running:
            messagebox.showwarning("Analysis Running", "Analysis is already running. Please wait.")
            return
           
        if not hasattr(self, 'start_date') or not self.start_date:
            messagebox.showerror("Invalid Dates", "Please update the date range first.")
            return
           
        # Disable button
        self.run_button.config(state='disabled', text="⏳ Running...")
        self.analysis_running = True
       
        # CRITICAL FIX: Get frequency from GUI and pass it to analysis engine
        selected_frequency = self.frequency_var.get()  # Get the selected frequency
        self.log_message(f"🔄 Starting {self.analysis_type_var.get()} analysis at {selected_frequency} frequency")
       
        # Pass frequency to analysis engine
        self.analysis_engine.run_analysis_threaded(frequency=selected_frequency)
       
    def reset_button(self):
        """Reset the run button."""
        self.run_button.config(state='normal', text="🚀 Run Analysis")
        self.analysis_running = False
       
    def update_results(self):
        """Delegate to results formatter."""
        self.results_formatter.update_results()
           
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