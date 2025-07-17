"""
Results Utilities for the Monetary Inflation Dashboard
Handles all result formatting, display, and summary generation
UPDATED: Added official GDP comparison for quarterly/annual GDP Proxy analysis
UPDATED: Added official CPI comparison for inflation analysis (all frequencies)
"""

import pandas as pd
import tkinter as tk


class ResultsFormatter:
    def __init__(self, gui):
        """Initialize with reference to GUI for accessing widgets and data."""
        self.gui = gui
        
    def update_results(self):
        """Update the results display."""
        if self.gui.results_df is None or self.gui.results_df.empty:
            self.gui.summary_text.delete(1.0, tk.END)
            self.gui.summary_text.insert(tk.END, "No results to display.")
            return
            
        analysis_type = self.gui.analysis_type_var.get()
        freq = self.gui.frequency_var.get()
        
        # Update summary
        self.gui.summary_text.delete(1.0, tk.END)
        self.gui.summary_text.insert(tk.END, f"📊 {analysis_type.upper()} ANALYSIS RESULTS\n")
        self.gui.summary_text.insert(tk.END, "=" * 50 + "\n\n")
        
        # Basic info
        self.gui.summary_text.insert(tk.END, f"Dataset Shape: {self.gui.results_df.shape}\n")
        self.gui.summary_text.insert(tk.END, f"Date Range: {self.gui.results_df.index.min()} to {self.gui.results_df.index.max()}\n")
        self.gui.summary_text.insert(tk.END, f"Available Columns: {', '.join(self.gui.results_df.columns)}\n\n")
        
        # Analysis-specific metrics
        analysis_methods = {
            "GDP Proxy": self.show_gdp_summary,
            "Velocity": self.show_velocity_summary,
            "Inflation": self.show_inflation_summary
        }
        
        if analysis_type in analysis_methods:
            analysis_methods[analysis_type]()
            
        # Add period breakdown for all analysis types
        self.show_period_breakdown(analysis_type, freq)
            
        # Update data tab
        self.update_data_tab()
        
    def show_gdp_summary(self):
        """Show GDP-specific summary."""
        gdp_col = self.find_column(self.gui.results_df, 'gdp', 'proxy')
                
        if gdp_col:
            latest_gdp = self.gui.results_df[gdp_col].iloc[-1]
            annual_gdp = latest_gdp * 12
            self.gui.summary_text.insert(tk.END, f"Latest Monthly GDP: ${latest_gdp:,.0f}B\n")
            self.gui.summary_text.insert(tk.END, f"Annualized GDP: ${annual_gdp:,.0f}B\n\n")
            
            if len(self.gui.results_df) >= 12:
                yoy_growth = (latest_gdp / self.gui.results_df[gdp_col].iloc[-13] - 1) * 100
                self.gui.summary_text.insert(tk.END, f"YoY Growth: {yoy_growth:.1f}%\n")
        
        # Component breakdown
        component_cols = [col for col in self.gui.results_df.columns if col in ['C', 'I', 'G', 'NX']]
        if component_cols:
            latest_components = self.gui.results_df[component_cols].iloc[-1]
            total = latest_components.sum()
            self.gui.summary_text.insert(tk.END, "\nGDP Components (Latest Month):\n")
            for comp in component_cols:
                value = latest_components[comp]
                share = (value / total * 100) if total > 0 else 0
                self.gui.summary_text.insert(tk.END, f"  {comp}: ${value:.0f}B ({share:.1f}%)\n")
                
    def show_velocity_summary(self):
        """Show velocity-specific summary."""
        velocity_col = self.find_column(self.gui.results_df, 'velocity')
        if velocity_col:
            current_velocity = self.gui.results_df[velocity_col].iloc[-1]
            avg_velocity = self.gui.results_df[velocity_col].mean()
            
            self.gui.summary_text.insert(tk.END, f"Current Velocity: {current_velocity:.3f}\n")
            self.gui.summary_text.insert(tk.END, f"Average Velocity: {avg_velocity:.3f}\n")
            
            if len(self.gui.results_df) >= 12:
                velocity_change = current_velocity - self.gui.results_df[velocity_col].iloc[-13]
                self.gui.summary_text.insert(tk.END, f"YoY Change: {velocity_change:.3f}\n")
                
    def show_inflation_summary(self):
        """Show inflation-specific summary with CPI comparison."""
        inflation_col = self.find_column(self.gui.results_df, 'inflation')
        if inflation_col:
            current_inflation = self.gui.results_df[inflation_col].iloc[-1]
            
            # Handle different inflation formats
            if abs(current_inflation) < 1:
                inflation_pct = current_inflation * 100
            else:
                inflation_pct = current_inflation
                
            self.gui.summary_text.insert(tk.END, f"Current Inflation (Estimated): {inflation_pct:.2f}%\n")
            
            # Show official CPI inflation if available
            if 'CPI_YoY' in self.gui.results_df.columns:
                latest_cpi_yoy = self.gui.results_df['CPI_YoY'].dropna()
                if not latest_cpi_yoy.empty:
                    official_inflation = latest_cpi_yoy.iloc[-1]
                    self.gui.summary_text.insert(tk.END, f"Current Inflation (Official CPI): {official_inflation:.2f}%\n")
                    
                    # Calculate difference
                    diff = inflation_pct - official_inflation
                    self.gui.summary_text.insert(tk.END, f"Difference: {diff:+.2f} percentage points\n")
            
            if len(self.gui.results_df) >= 12:
                past_inflation = self.gui.results_df[inflation_col].iloc[-13]
                if abs(past_inflation) < 1:
                    past_pct = past_inflation * 100
                else:
                    past_pct = past_inflation
                change = inflation_pct - past_pct
                self.gui.summary_text.insert(tk.END, f"YoY Change (Estimated): {change:.2f} percentage points\n")
                
    def show_period_breakdown(self, analysis_type, freq):
        """Show detailed period-by-period breakdown for all analysis types."""
        try:
            periods_requested = int(self.gui.periods_var.get())
            
            # Filter data to requested date range
            filtered_data = self.gui.results_df[
                (self.gui.results_df.index >= pd.Timestamp(self.gui.start_date)) & 
                (self.gui.results_df.index <= pd.Timestamp(self.gui.end_date))
            ]
            
            # Resample data based on frequency
            resampled_data = self.resample_data_by_frequency(filtered_data, freq)
            
            # Get the data for display
            display_data = resampled_data.tail(periods_requested)
            periods_available = len(resampled_data)
            periods_showing = len(display_data)
            
            # Add breakdown section
            self.gui.summary_text.insert(tk.END, f"\n📊 {freq.upper()} {analysis_type.upper()} BREAKDOWN:\n")
            self.gui.summary_text.insert(tk.END, f"Requested: {periods_requested} {freq.lower()}, ")
            self.gui.summary_text.insert(tk.END, f"Available: {periods_available}, Showing: {periods_showing}\n")
            self.gui.summary_text.insert(tk.END, "-" * 40 + "\n")
            
            # Check if this is GDP Proxy analysis with quarterly/annual frequency
            # Look for official GDP column with more flexible naming
            official_gdp_col = self._find_official_gdp_column(display_data)
            show_gdp_comparison = (analysis_type == "GDP Proxy" and 
                                 freq in ["Quarterly", "Annually"] and 
                                 official_gdp_col is not None)
            
            # Check if this is Inflation analysis - CPI comparison available for all frequencies
            official_cpi_col = self._find_official_cpi_column(display_data, freq)
            show_cpi_comparison = (analysis_type == "Inflation" and 
                                 official_cpi_col is not None)
            
            # Debug logging
            if analysis_type == "GDP Proxy" and freq in ["Quarterly", "Annually"]:
                self.gui.log_message(f"🔍 Looking for GDP comparison data...")
                self.gui.log_message(f"Available columns: {list(display_data.columns)}")
                self.gui.log_message(f"Official GDP column found: {official_gdp_col}")
                self.gui.log_message(f"Show comparison: {show_gdp_comparison}")
            
            if analysis_type == "Inflation":
                self.gui.log_message(f"🔍 Looking for CPI comparison data...")
                self.gui.log_message(f"Available columns: {list(display_data.columns)}")
                self.gui.log_message(f"Official CPI column found: {official_cpi_col}")
                self.gui.log_message(f"Show CPI comparison: {show_cpi_comparison}")
            
            # Determine which column to display and formatting
            display_configs = {
                "GDP Proxy": (lambda df: self.find_column(df, 'gdp', 'proxy'), "B", "${:.0f}{}"),
                "Velocity": (lambda df: self.find_column(df, 'velocity'), "", "{:.3f}{}"),
                "Inflation": (lambda df: self.find_column(df, 'inflation'), "%", "{:+.2f}{}")
            }
            
            if analysis_type in display_configs:
                col_finder, unit, format_str = display_configs[analysis_type]
                display_col = col_finder(display_data)
                
                if display_col and display_col in display_data.columns:
                    for date_idx, row in display_data.iterrows():
                        period_label = self.format_period_label(date_idx, freq)
                        value = row[display_col]
                        
                        if pd.notna(value):
                            if analysis_type == "Inflation":
                                # Handle inflation formatting (convert to percentage if needed)
                                value_display = value * 100 if abs(value) < 1 else value
                                formatted_value = format_str.format(value_display, unit)
                                
                                # Add CPI comparison if applicable
                                if show_cpi_comparison:
                                    self._add_cpi_comparison_line(period_label, value_display, row[official_cpi_col], freq)
                                else:
                                    self.gui.summary_text.insert(tk.END, f"{period_label}: {formatted_value}\n")
                            else:
                                # Regular formatting for GDP Proxy and Velocity
                                formatted_value = format_str.format(value, unit)
                                
                                # Add GDP comparison if applicable
                                if show_gdp_comparison:
                                    self._add_gdp_comparison_line(period_label, value, row[official_gdp_col])
                                else:
                                    self.gui.summary_text.insert(tk.END, f"{period_label}: {formatted_value}\n")
                        else:
                            self.gui.summary_text.insert(tk.END, f"{period_label}: N/A\n")
                else:
                    self.gui.summary_text.insert(tk.END, f"No {analysis_type.lower()} data available for breakdown.\n")
            
            self.gui.summary_text.insert(tk.END, "\n")
                
        except Exception as e:
            self.gui.summary_text.insert(tk.END, f"\n❌ Error creating period breakdown: {str(e)}\n")
    
    def _find_official_gdp_column(self, df):
        """Find the official GDP column with flexible naming."""
        # Look for official GDP column - try different possible names
        possible_names = ['GDP', 'gdp', 'GDP_official', 'FRED_GDP', 'OFFICIAL_GDP']
        
        for name in possible_names:
            if name in df.columns:
                return name
        
        # If not found, return None
        return None
    
    def _find_official_cpi_column(self, df, freq):
        """Find the appropriate official CPI column based on frequency."""
        # For different frequencies, use different CPI measures
        if freq == "Monthly":
            # For monthly, use month-over-month inflation
            possible_names = ['CPI_MoM', 'CPI_mom', 'cpi_mom']
        else:
            # For quarterly and annual, use year-over-year inflation
            possible_names = ['CPI_YoY', 'CPI_yoy', 'cpi_yoy']
        
        for name in possible_names:
            if name in df.columns:
                return name
        
        # Fallback to any CPI column
        cpi_columns = [col for col in df.columns if 'CPI' in col.upper()]
        if cpi_columns:
            return cpi_columns[0]
        
        return None
    
    def _add_gdp_comparison_line(self, period_label, estimated_gdp, official_gdp):
        """Add a line showing GDP estimate vs official GDP with differences."""
        if pd.notna(official_gdp):
            # Calculate differences
            absolute_diff = estimated_gdp - official_gdp
            percent_diff = (absolute_diff / official_gdp) * 100 if official_gdp != 0 else 0
            
            # Format the comparison line
            estimated_str = f"${estimated_gdp:.0f}B"
            official_str = f"${official_gdp:.0f}B"
            
            # Color-code the difference (using text indicators since we can't use colors easily in tkinter text)
            if absolute_diff > 0:
                diff_indicator = "↗"
                abs_diff_str = f"+${abs(absolute_diff):.0f}B"
                pct_diff_str = f"(+{percent_diff:.1f}%)"
            elif absolute_diff < 0:
                diff_indicator = "↘"
                abs_diff_str = f"-${abs(absolute_diff):.0f}B"
                pct_diff_str = f"({percent_diff:.1f}%)"
            else:
                diff_indicator = "="
                abs_diff_str = "$0B"
                pct_diff_str = "(0.0%)"
            
            # Create the comparison line
            comparison_line = (f"{period_label}: {estimated_str} | "
                             f"Official: {official_str} | "
                             f"Diff: {abs_diff_str} {pct_diff_str} {diff_indicator}\n")
            
            self.gui.summary_text.insert(tk.END, comparison_line)
        else:
            # Official GDP not available for this period
            estimated_str = f"${estimated_gdp:.0f}B"
            comparison_line = f"{period_label}: {estimated_str} | Official: N/A | Diff: N/A\n"
            self.gui.summary_text.insert(tk.END, comparison_line)
    
    def _add_cpi_comparison_line(self, period_label, estimated_inflation, official_cpi_inflation, freq):
        """Add a line showing estimated inflation vs official CPI inflation with differences."""
        if pd.notna(official_cpi_inflation):
            # Calculate differences
            absolute_diff = estimated_inflation - official_cpi_inflation
            
            # Format the comparison line
            estimated_str = f"{estimated_inflation:+.2f}%"
            official_str = f"{official_cpi_inflation:+.2f}%"
            
            # Determine comparison type based on frequency
            comparison_type = "MoM" if freq == "Monthly" else "YoY"
            
            # Color-code the difference
            if absolute_diff > 0:
                diff_indicator = "↗"
                abs_diff_str = f"+{abs(absolute_diff):.2f}pp"
            elif absolute_diff < 0:
                diff_indicator = "↘"
                abs_diff_str = f"-{abs(absolute_diff):.2f}pp"
            else:
                diff_indicator = "="
                abs_diff_str = "0.00pp"
            
            # Create the comparison line
            comparison_line = (f"{period_label}: {estimated_str} | "
                             f"CPI {comparison_type}: {official_str} | "
                             f"Diff: {abs_diff_str} {diff_indicator}\n")
            
            self.gui.summary_text.insert(tk.END, comparison_line)
        else:
            # Official CPI not available for this period
            estimated_str = f"{estimated_inflation:+.2f}%"
            comparison_line = f"{period_label}: {estimated_str} | CPI: N/A | Diff: N/A\n"
            self.gui.summary_text.insert(tk.END, comparison_line)
            
    def update_data_tab(self):
        """Update the data tab."""
        self.gui.data_text.delete(1.0, tk.END)
        
        if self.gui.results_df is None or self.gui.results_df.empty:
            self.gui.data_text.insert(tk.END, "No data available.")
            return
            
        # Show recent data (last 20 rows)
        recent_data = self.gui.results_df.tail(20)
        
        # Format as simple table
        self._insert_data_table_header()
        self._insert_data_table_rows(recent_data)
        
    def _insert_data_table_header(self):
        """Insert table header for data tab."""
        self.gui.data_text.insert(tk.END, "Recent Data (Last 20 Observations):\n")
        self.gui.data_text.insert(tk.END, "=" * 80 + "\n\n")
        
    def _insert_data_table_rows(self, data):
        """Insert table rows for data tab."""
        # Headers
        headers = ["Date"] + list(data.columns)
        header_line = "  ".join(f"{h:<12}" for h in headers)
        self.gui.data_text.insert(tk.END, header_line + "\n")
        self.gui.data_text.insert(tk.END, "-" * len(header_line) + "\n")
        
        # Data rows
        for idx, row in data.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            values = [date_str] + [f"{val:.2f}" if pd.notna(val) else "N/A" for val in row]
            value_line = "  ".join(f"{v:<12}" for v in values)
            self.gui.data_text.insert(tk.END, value_line + "\n")
            
    def resample_data_by_frequency(self, data, freq):
        """Resample data based on frequency selection."""
        frequency_map = {
            "Monthly": lambda x: x,
            "Quarterly": lambda x: x.resample('QE').last(),
            "Annually": lambda x: x.resample('YE').last()
        }
        return frequency_map.get(freq, lambda x: x)(data)
            
    def format_period_label(self, date_index, freq):
        """Format period labels based on frequency."""
        if freq == "Monthly":
            return date_index.strftime('%Y-%m (%b)')
        elif freq == "Quarterly":
            quarter = ((date_index.month - 1) // 3) + 1
            return f"{date_index.year}-Q{quarter}"
        elif freq == "Annually":
            return str(date_index.year)
        else:
            return date_index.strftime('%Y-%m-%d')
            
    def find_column(self, df, *keywords):
        """Find a column containing all specified keywords (case-insensitive)."""
        for col in df.columns:
            if all(keyword.lower() in col.lower() for keyword in keywords):
                return col
        return None