"""
Analysis Logic for the Monetary Inflation Dashboard
Handles all data fetching, processing, and calculations
FIXED: CPI data processing now properly handles date ranges and inflation calculations
FIXED: Now properly passes frequency parameter to velocity calculations
FIXED: Annual CPI calculation now uses proper year-over-year from selected end month
UPDATED: Enhanced to include official GDP data for comparison
UPDATED: Enhanced to include official CPI data for inflation comparison
"""

import pandas as pd
import threading
import traceback
import calendar
from datetime import datetime


class AnalysisEngine:
    def __init__(self, gui):
        """Initialize with reference to GUI for logging and data access."""
        self.gui = gui
        
    def run_analysis_threaded(self, frequency='Monthly'):
        """Start analysis in a separate thread with frequency parameter."""
        thread = threading.Thread(target=self.run_analysis_thread, args=(frequency,), daemon=True)
        thread.start()
        
    def run_analysis_thread(self, frequency='Monthly'):
        """Run the analysis in a separate thread with frequency parameter."""
        try:
            self.gui.log_message("🚀 Starting analysis...")
            self.gui.log_message(f"⏱️ Selected frequency: {frequency}")
            
            # Step 1: Fetch and merge data
            fred_data, bea_shares, monthly_weights = self._fetch_and_prepare_data()
            
            # Step 2: Build GDP proxy
            gdp_proxy_df = self._build_gdp_proxy()
            
            # Step 3: Calculate velocity and add official GDP/CPI for comparison
            self.gui.results_df = self._calculate_additional_metrics(gdp_proxy_df, fred_data, frequency)
            
            self.gui.log_message("✅ Analysis completed successfully!")
            
            # Step 4: Update results display
            self.gui.root.after(0, self.gui.update_results)
            
        except Exception as e:
            error_msg = f"❌ Analysis failed: {str(e)}"
            self.gui.log_message(error_msg)
            self.gui.log_message(f"Error details: {traceback.format_exc()}")
        finally:
            # Re-enable button
            self.gui.root.after(0, self.gui.reset_button)
            
    def _fetch_and_prepare_data(self):
        """Fetch and prepare all required data."""
        # Import here to avoid circular imports
        from src.data_fetch import get_fred_gdp_components, get_bea_gdp_shares
        from src.weight_manager import prev_qtr_shares, shares_monthly
        
        # Use calculated dates from GUI
        start_str = self.gui.start_date.strftime("%Y-%m-%d")
        end_str = self.gui.end_date.strftime("%Y-%m-%d")
        
        self.gui.log_message(f"📅 GUI Selected Range: {start_str} to {end_str}")
        
        # Check lag needed
        today = datetime.now().date()
        lag_quarters = 1 if (today - self.gui.end_date).days < 90 else 0
        
        if lag_quarters > 0:
            self.gui.log_message(f"⚠️ Using {lag_quarters}Q lag due to recent end date")
        
        # Fetch data (CPI is now included in this main fetch)
        self.gui.log_message("📊 Fetching FRED data (including CPI)...")
        fred_data = get_fred_gdp_components(start_date=start_str)
        
        # CRITICAL: Debug what's in FRED data
        self.gui.log_message(f"🔍 FRED data shape: {fred_data.shape}")
        self.gui.log_message(f"🔍 FRED data date range: {fred_data.index.min()} to {fred_data.index.max()}")
        self.gui.log_message(f"🔍 FRED data columns: {list(fred_data.columns)}")
        
        # Check if CPI is in FRED data
        if 'CPIAUCSL' in fred_data.columns:
            self.gui.log_message("✅ CPI data found in FRED dataset")
            cpi_in_fred = fred_data['CPIAUCSL'].dropna()
            self.gui.log_message(f"🔍 CPI data in FRED: {len(cpi_in_fred)} observations")
            if len(cpi_in_fred) > 0:
                self.gui.log_message(f"🔍 CPI date range: {cpi_in_fred.index.min()} to {cpi_in_fred.index.max()}")
                self.gui.log_message(f"🔍 Recent CPI values: {cpi_in_fred.tail(3).round(1).to_dict()}")
        else:
            self.gui.log_message("❌ CPI data NOT found in FRED dataset")
        
        self.gui.log_message(f"✅ Got {len(fred_data)} FRED observations total")
        
        self.gui.log_message("📈 Fetching BEA shares...")
        bea_shares = get_bea_gdp_shares()
        self.gui.log_message(f"✅ Got {len(bea_shares)} BEA observations")
        
        # Create weights
        self.gui.log_message("⚖️ Creating monthly weights...")
        lagged_shares = prev_qtr_shares(bea_shares, lag_quarters=lag_quarters)
        monthly_weights = shares_monthly(lagged_shares, method='ffill')
        self.gui.log_message(f"✅ Created {len(monthly_weights)} monthly weights")
        
        return fred_data, bea_shares, monthly_weights
        
    def _build_gdp_proxy(self):
        """Build the GDP proxy."""
        from src.gdp_proxy import build_gdp_proxy
        
        start_str = self.gui.start_date.strftime("%Y-%m-%d")
        use_slce = self.gui.use_slce_var.get()
        
        self.gui.log_message("🏗️ Building GDP proxy...")
        gdp_proxy_df = build_gdp_proxy(
            start_date=start_str, 
            export=True, 
            strict=False, 
            use_slce=use_slce
        )
        self.gui.log_message(f"✅ Built GDP proxy with {len(gdp_proxy_df)} observations")
        
        return gdp_proxy_df
        
    def _calculate_additional_metrics(self, gdp_proxy_df, fred_data, frequency='Monthly'):
        """Calculate additional metrics like velocity if requested and ALWAYS include official GDP/CPI."""
        analysis_type = self.gui.analysis_type_var.get()
        
        # Start with GDP proxy data
        results_df = gdp_proxy_df.copy()
        
        # ALWAYS include official GDP data for comparison
        if 'GDP' in fred_data.columns:
            # Convert quarterly SAAR GDP to monthly for comparison
            self.gui.log_message("📊 Adding official GDP data for comparison...")
            official_gdp_monthly = self._convert_quarterly_gdp_to_monthly(fred_data['GDP'])
            if official_gdp_monthly is not None:
                # Align with our results timeframe
                aligned_official = official_gdp_monthly.reindex(results_df.index)
                results_df['GDP'] = aligned_official
                
                # Count how many periods have both estimated and official data
                comparison_periods = results_df[['GDP_proxy', 'GDP']].dropna().shape[0]
                self.gui.log_message(f"✅ Added official GDP data: {comparison_periods} periods available for comparison")
            else:
                self.gui.log_message("⚠️ Could not convert official GDP to monthly")
        else:
            self.gui.log_message("⚠️ Official GDP data not available in FRED data")
        
        # FIXED: Process CPI data with proper date range filtering
        if 'CPIAUCSL' in fred_data.columns:
            self.gui.log_message("📊 Processing official CPI data for inflation comparison...")
            
            # Debug the raw CPI data first
            raw_cpi = fred_data['CPIAUCSL'].dropna()
            self.gui.log_message(f"🔍 Raw CPI data: {len(raw_cpi)} observations")
            self.gui.log_message(f"🔍 Raw CPI range: {raw_cpi.index.min()} to {raw_cpi.index.max()}")
            
            # CRITICAL FIX: Pass the full CPI series and let _process_cpi_data handle the filtering properly
            official_cpi_data = self._process_cpi_data(fred_data['CPIAUCSL'], frequency)
            if official_cpi_data is not None:
                self.gui.log_message(f"🔍 Processed CPI data shape: {official_cpi_data.shape}")
                self.gui.log_message(f"🔍 Processed CPI columns: {list(official_cpi_data.columns)}")
                
                # Align with our results timeframe
                aligned_cpi = official_cpi_data.reindex(results_df.index)
                results_df['CPI'] = aligned_cpi['CPI']
                
                # Add appropriate inflation measures based on frequency
                if frequency == 'Monthly':
                    if 'CPI_MoM' in aligned_cpi.columns:
                        results_df['CPI_MoM'] = aligned_cpi['CPI_MoM']
                    if 'CPI_YoY' in aligned_cpi.columns:
                        results_df['CPI_YoY'] = aligned_cpi['CPI_YoY']
                elif frequency == 'Quarterly':
                    if 'CPI_QoQ' in aligned_cpi.columns:
                        results_df['CPI_QoQ'] = aligned_cpi['CPI_QoQ']
                    if 'CPI_YoY' in aligned_cpi.columns:
                        results_df['CPI_YoY'] = aligned_cpi['CPI_YoY']
                elif frequency == 'Annually':
                    if 'CPI_Annual' in aligned_cpi.columns:
                        results_df['CPI_Annual'] = aligned_cpi['CPI_Annual']
                        self.gui.log_message("✅ Added annual CPI inflation")
                    if 'CPI_YoY' in aligned_cpi.columns:
                        results_df['CPI_YoY'] = aligned_cpi['CPI_YoY']
                
                # Count how many periods have CPI data
                cpi_periods = results_df['CPI'].dropna().shape[0]
                self.gui.log_message(f"✅ Added official CPI data: {cpi_periods} periods")
                
                # Show what CPI data is actually in the results for the selected range
                if len(results_df) > 0:
                    relevant_cols = ['CPI']
                    if frequency == 'Monthly':
                        relevant_cols.extend(['CPI_MoM', 'CPI_YoY'])
                    elif frequency == 'Quarterly':
                        relevant_cols.extend(['CPI_QoQ', 'CPI_YoY'])
                    elif frequency == 'Annually':
                        relevant_cols.extend(['CPI_Annual', 'CPI_YoY'])
                    
                    available_cols = [col for col in relevant_cols if col in results_df.columns]
                    recent_results = results_df[available_cols].dropna().tail(3)
                    
                    if len(recent_results) > 0:
                        self.gui.log_message(f"🔍 Recent CPI data in selected range:")
                        for date, row in recent_results.iterrows():
                            date_str = date.strftime('%Y-%m')
                            cpi_val = row['CPI']
                            self.gui.log_message(f"   {date_str}: CPI={cpi_val:.1f}")
                            
                            # Show relevant inflation rate
                            if frequency == 'Monthly' and 'CPI_MoM' in row and pd.notna(row['CPI_MoM']):
                                self.gui.log_message(f"      MoM: {row['CPI_MoM']:.2f}%")
                            elif frequency == 'Quarterly' and 'CPI_QoQ' in row and pd.notna(row['CPI_QoQ']):
                                self.gui.log_message(f"      QoQ: {row['CPI_QoQ']:.2f}%")
                            elif frequency == 'Annually' and 'CPI_Annual' in row and pd.notna(row['CPI_Annual']):
                                self.gui.log_message(f"      Annual: {row['CPI_Annual']:.2f}%")
                            
                            if 'CPI_YoY' in row and pd.notna(row['CPI_YoY']):
                                self.gui.log_message(f"      YoY: {row['CPI_YoY']:.2f}%")
                            
            else:
                self.gui.log_message("⚠️ Could not process CPI data")
        else:
            self.gui.log_message("⚠️ Official CPI data not available in FRED data")
        
        # Calculate velocity/inflation if requested
        if analysis_type in ["Velocity", "Inflation"]:
            from velocity import calc_velocity
            
            # CRITICAL FIX: Log the frequency being used
            self.gui.log_message(f"💰 Calculating velocity at {frequency} frequency...")
            gdp_col = self._find_column(gdp_proxy_df, 'GDP', 'proxy')
            
            if gdp_col and 'M2' in fred_data.columns:
                # Align data
                aligned_data = pd.DataFrame({
                    'gdp_proxy': gdp_proxy_df[gdp_col],
                    'money_supply': fred_data['M2']
                }).dropna()
                
                if len(aligned_data) > 0:
                    # CRITICAL FIX: Pass frequency parameter to calc_velocity
                    velocity_df = calc_velocity(
                        aligned_data['gdp_proxy'], 
                        aligned_data['money_supply'], 
                        frequency=frequency  # THIS WAS MISSING!
                    )
                    
                    # Combine results
                    results_df = pd.concat([results_df, velocity_df], axis=1)
                    results_df = results_df.loc[:, ~results_df.columns.duplicated()]
                    
                    # Log what frequency was actually used
                    self.gui.log_message(f"✅ Calculated velocity for {len(velocity_df)} periods at {frequency} frequency")
                    
                    # Additional frequency validation
                    if len(velocity_df) > 1:
                        time_diff = velocity_df.index[1] - velocity_df.index[0]
                        days_between = time_diff.days
                        self.gui.log_message(f"📅 Time between observations: {days_between} days")
                        
                        # Verify expected frequency
                        expected_days = {
                            'Monthly': 30,
                            'Quarterly': 90,
                            'Annually': 365
                        }
                        expected = expected_days.get(frequency, 30)
                        
                        if abs(days_between - expected) <= 15:
                            self.gui.log_message(f"✅ Frequency verification: {frequency} frequency confirmed")
                        else:
                            self.gui.log_message(f"⚠️ Frequency mismatch: Expected ~{expected} days, got {days_between} days")
                        
                else:
                    self.gui.log_message("⚠️ No overlapping data for velocity calculation")
            else:
                self.gui.log_message("⚠️ Missing GDP proxy or M2 data for velocity")
                
        return results_df
    
    def _convert_quarterly_gdp_to_monthly(self, quarterly_gdp_saar):
        """Convert quarterly SAAR GDP to monthly for comparison with our estimates."""
        try:
            # Filter to quarterly data only (GDP is typically quarterly)
            quarterly_data = quarterly_gdp_saar.dropna()
            
            if quarterly_data.empty:
                self.gui.log_message("⚠️ No quarterly GDP data available")
                return None
                
            self.gui.log_message(f"📅 Converting {len(quarterly_data)} quarterly GDP observations to monthly")
            
            # Convert SAAR to monthly (SAAR ÷ 12)
            monthly_values = quarterly_data / 12
            
            # Create monthly series by distributing quarterly values to each month
            monthly_index = pd.date_range(
                quarterly_data.index.min().replace(day=1),
                quarterly_data.index.max().replace(day=1) + pd.DateOffset(months=2),
                freq='MS'
            )
            
            monthly_series = pd.Series(index=monthly_index, dtype='float64')
            
            periods_converted = 0
            for quarter_date, monthly_val in monthly_values.items():
                quarter_start = quarter_date.replace(day=1)
                
                # Assign the same monthly value to each of the 3 months in the quarter
                for month_offset in range(3):
                    month_date = quarter_start + pd.DateOffset(months=month_offset)
                    if month_date in monthly_index:
                        monthly_series[month_date] = monthly_val
                        periods_converted += 1
            
            self.gui.log_message(f"✅ Converted official GDP: {periods_converted} monthly values created")
            
            # Show recent official GDP values for validation
            recent_official = monthly_series.dropna().tail(3)
            if len(recent_official) > 0:
                latest_official = recent_official.iloc[-1]
                annual_official = latest_official * 12
                self.gui.log_message(f"📊 Latest official GDP: ${latest_official:.0f}B/month (${annual_official:,.0f}B/year)")
            
            return monthly_series
            
        except Exception as e:
            self.gui.log_message(f"❌ Error converting official GDP to monthly: {e}")
            return None
    
    def _process_cpi_data(self, cpi_series, frequency='Monthly'):
        """FIXED: Process CPI data with proper annual calculation using selected end month."""
        
        self.gui.log_message(f"🔄 Processing CPI data for {frequency} frequency...")
        self.gui.log_message(f"🔍 Input CPI series: {len(cpi_series)} observations")
        self.gui.log_message(f"🔍 Input CPI range: {cpi_series.index.min()} to {cpi_series.index.max()}")
        
        try:
            gui_start = pd.Timestamp(self.gui.start_date)
            gui_end = pd.Timestamp(self.gui.end_date)
            
            # Get enough historical data for calculations (at least 15 months before start)
            extended_start = gui_start - pd.DateOffset(months=15)
            cpi_data = cpi_series[cpi_series.index >= extended_start].dropna()
            
            if cpi_data.empty:
                self.gui.log_message("⚠️ No CPI data available for calculations")
                return None
                
            self.gui.log_message(f"📅 Using CPI data from {cpi_data.index.min().strftime('%Y-%m')} to {cpi_data.index.max().strftime('%Y-%m')} for calculations")
            
            # Calculate inflation rates based on frequency
            if frequency == 'Monthly':
                # For monthly, use all data points
                result_df = pd.DataFrame(index=cpi_data.index)
                result_df['CPI'] = cpi_data
                result_df['CPI_MoM'] = cpi_data.pct_change() * 100
                result_df['CPI_YoY'] = cpi_data.pct_change(periods=12) * 100
                self.gui.log_message(f"✅ Calculated monthly CPI inflation (MoM and YoY)")
                
            elif frequency == 'Quarterly':
                # For quarterly, use all data points but calculate quarterly changes
                result_df = pd.DataFrame(index=cpi_data.index)
                result_df['CPI'] = cpi_data
                result_df['CPI_QoQ'] = cpi_data.pct_change(periods=3) * 100
                result_df['CPI_YoY'] = cpi_data.pct_change(periods=12) * 100
                self.gui.log_message(f"✅ Calculated quarterly CPI inflation (QoQ and YoY)")
                
            elif frequency == 'Annually':
                # FIXED: For annual frequency, use the selected end month for proper annual calculations
                end_month = gui_end.month
                end_month_name = calendar.month_name[end_month]
                
                self.gui.log_message(f"📅 Annual frequency: Using {end_month_name} (month {end_month}) for year-over-year calculations")
                
                # Filter to only the selected end month across all years
                annual_data = cpi_data[cpi_data.index.month == end_month]
                
                if len(annual_data) < 2:
                    self.gui.log_message(f"⚠️ Insufficient {end_month_name} data for annual calculations (need at least 2 years)")
                    return None
                
                self.gui.log_message(f"📊 Found {len(annual_data)} {end_month_name} observations for annual calculation")
                
                # Create annual result DataFrame
                result_df = pd.DataFrame(index=annual_data.index)
                result_df['CPI'] = annual_data
                
                # Calculate true year-over-year changes (December 2021 vs December 2020, etc.)
                result_df['CPI_Annual'] = annual_data.pct_change() * 100
                result_df['CPI_YoY'] = result_df['CPI_Annual']  # Same for annual frequency
                
                self.gui.log_message(f"✅ Calculated annual CPI inflation using {end_month_name}-to-{end_month_name} changes")
                
                # Debug: Show the annual calculations
                annual_with_data = result_df.dropna()
                if len(annual_with_data) > 0:
                    self.gui.log_message(f"📊 Annual CPI calculations:")
                    for date, row in annual_with_data.iterrows():
                        year = date.year
                        cpi_val = row['CPI']
                        inflation = row['CPI_Annual']
                        if pd.notna(inflation):
                            self.gui.log_message(f"   {year} {end_month_name}: CPI={cpi_val:.1f}, Inflation={inflation:.2f}%")
            
            # Filter to the GUI-selected date range AFTER calculations
            if frequency == 'Annually':
                # For annual, the result_df is already filtered to the selected month
                # Just ensure it's within the year range
                result_df = result_df[(result_df.index.year >= gui_start.year) & 
                                    (result_df.index.year <= gui_end.year)]
            else:
                result_df = result_df[(result_df.index >= gui_start) & (result_df.index <= gui_end)]
            
            self.gui.log_message(f"✅ Calculated {frequency} CPI inflation, filtered to {len(result_df)} periods in selected range")
            
            # Show recent CPI values for validation
            if len(result_df) > 0:
                recent_cpi = result_df.dropna().tail(3)
                if len(recent_cpi) > 0:
                    latest = recent_cpi.iloc[-1]
                    self.gui.log_message(f"📊 Latest CPI in selected range: {latest['CPI']:.1f} ({latest.name.strftime('%Y-%m')})")
                    
                    # Show appropriate inflation measure
                    if frequency == 'Monthly' and 'CPI_MoM' in latest and pd.notna(latest['CPI_MoM']):
                        self.gui.log_message(f"📊 Latest MoM inflation: {latest['CPI_MoM']:.2f}%")
                    elif frequency == 'Quarterly' and 'CPI_QoQ' in latest and pd.notna(latest['CPI_QoQ']):
                        self.gui.log_message(f"📊 Latest QoQ inflation: {latest['CPI_QoQ']:.2f}%")
                    elif frequency == 'Annually' and 'CPI_Annual' in latest and pd.notna(latest['CPI_Annual']):
                        self.gui.log_message(f"📊 Latest annual inflation: {latest['CPI_Annual']:.2f}%")
                        
                    if 'CPI_YoY' in latest and pd.notna(latest['CPI_YoY']):
                        self.gui.log_message(f"📊 Latest YoY inflation: {latest['CPI_YoY']:.2f}%")
            
            return result_df
            
        except Exception as e:
            self.gui.log_message(f"❌ Error processing CPI data: {e}")
            print(f"🚨 ERROR in _process_cpi_data: {e}")
            return None
        
    def _find_column(self, df, *keywords):
        """Find a column containing all specified keywords (case-insensitive)."""
        for col in df.columns:
            if all(keyword.lower() in col.lower() for keyword in keywords):
                return col
        return None