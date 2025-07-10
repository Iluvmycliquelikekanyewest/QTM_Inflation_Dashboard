"""
GDP Proxy construction module - NO SCALING VERSION
Builds GDP = C + I + G + NX using only real FRED data, no synthetic scaling.
FIXED: Removed NEWORDER scaling, uses comprehensive investment series.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

from src.data_fetch import DataFetcher
from src.transforms import saar_to_monthly, interp_quarterly

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_saar_to_monthly(series: pd.Series, name: str) -> pd.Series:
    """Safely convert SAAR to monthly with error handling."""
    try:
        if series.isna().all():
            logger.warning(f"⚠️ {name}: All NaN values")
            return series
        
        # Use existing transform
        result = saar_to_monthly(series)
        non_nan_count = result.notna().sum()
        logger.info(f"✅ {name}: Converted to monthly, {non_nan_count} valid observations")
        return result
    except Exception as e:
        logger.error(f"❌ {name}: SAAR conversion failed - {e}")
        return pd.Series(index=series.index, dtype='float64')


def quarterly_saar_to_monthly_distributed(series: pd.Series, name: str, target_start_date=None) -> pd.Series:
    """
    Convert quarterly SAAR data to monthly by:
    1. Converting SAAR to quarterly actual (÷4)
    2. Distributing quarterly value across 3 months (÷3)
    3. Final result: quarterly_saar ÷ 12
    
    NEW: Added target_start_date to focus on recent data only
    """
    try:
        if series.isna().all():
            logger.warning(f"⚠️ {name}: All NaN values")
            return pd.Series(index=pd.date_range('1990-01-01', '2025-12-01', freq='MS'), dtype='float64')
        
        # Filter to recent data only if target_start_date provided
        original_length = len(series)
        if target_start_date:
            target_start = pd.to_datetime(target_start_date)
            series = series[series.index >= target_start]
            if len(series) < original_length:
                logger.info(f"📅 {name}: Filtered from {original_length} to {len(series)} observations (from {target_start.date()})")
        
        if series.empty:
            logger.warning(f"⚠️ {name}: No data after filtering")
            return pd.Series(index=pd.date_range('1990-01-01', '2025-12-01', freq='MS'), dtype='float64')
        
        # Create monthly index covering the full range
        start_date = series.index.min().replace(day=1)
        end_date = series.index.max().replace(day=1)
        monthly_index = pd.date_range(start_date, end_date, freq='MS')
        
        # Convert quarterly SAAR to monthly amounts
        monthly_amounts = series / 12
        
        # Create monthly series by distributing quarterly values
        monthly_series = pd.Series(index=monthly_index, dtype='float64')
        
        for quarter_date, quarterly_value in monthly_amounts.items():
            if pd.notna(quarterly_value):
                # Find the 3 months that belong to this quarter
                quarter_start = quarter_date.replace(day=1)
                
                # Generate the 3 months for this quarter
                for month_offset in range(3):
                    month_date = quarter_start + pd.DateOffset(months=month_offset)
                    if month_date in monthly_index:
                        monthly_series[month_date] = quarterly_value
        
        non_nan_count = monthly_series.notna().sum()
        
        # Show recent values instead of first values
        recent_values = monthly_series.dropna().tail(3) if len(monthly_series.dropna()) >= 3 else monthly_series.dropna()
        logger.info(f"✅ {name}: Quarterly SAAR→Monthly, {non_nan_count} monthly values")
        if len(recent_values) > 0:
            logger.info(f"   Recent values: {recent_values.round(1).tolist()}")
            logger.info(f"   Latest monthly: ${recent_values.iloc[-1]:.1f}B")
        else:
            logger.info(f"   No recent values available")
        
        return monthly_series
        
    except Exception as e:
        logger.error(f"❌ {name}: Quarterly conversion failed - {e}")
        return pd.Series(index=pd.date_range('1990-01-01', '2025-12-01', freq='MS'), dtype='float64')


class GDPProxyBuilder:
    def __init__(self):
        self.fetcher = DataFetcher()
        self.cache_dir = Path("data")
        self.output_path = self.cache_dir / "gdp_proxy_monthly.csv"

    def _to_billions(self, series):
        """Convert millions to billions with validation."""
        if series.isna().all():
            return series
        return series / 1000

    def _validate_component(self, series, name, expected_range):
        """Validate component values are in reasonable range."""
        if series.isna().all():
            logger.warning(f"⚠️ {name}: No valid data")
            return False
        
        recent_mean = series.tail(12).mean()
        min_val, max_val = expected_range
        
        if min_val <= recent_mean <= max_val:
            logger.info(f"✅ {name}: Recent mean ${recent_mean:.1f}B/month (reasonable)")
            return True
        else:
            logger.warning(f"⚠️ {name}: Recent mean ${recent_mean:.1f}B/month (expected ${min_val}-${max_val}B)")
            return False

    def _filter_to_complete_data_only(self, proxy_df):
        """Only include months with all required components available."""
        required_components = ['C', 'I', 'G', 'NX']
        
        # Find months with all components
        complete_mask = proxy_df[required_components].notna().all(axis=1)
        complete_data = proxy_df[complete_mask].copy()
        
        # Report what's missing
        incomplete_months = proxy_df[~complete_mask]
        recent_incomplete = incomplete_months[incomplete_months.index >= '2025-01-01']
        
        if len(recent_incomplete) > 0:
            logger.info(f"📅 Most recent complete data: {complete_data.index[-1].date()}")
            logger.info(f"⏳ {len(recent_incomplete)} recent months awaiting quarterly data release:")
            
            for date, row in recent_incomplete.iterrows():
                missing = [comp for comp in required_components if pd.isna(row[comp])]
                logger.info(f"   {date.date()}: Missing {', '.join(missing)}")
        
        return complete_data

    def _comprehensive_validation(self, proxy_df):
        """Enhanced validation with confidence scoring."""
        complete_data = proxy_df.dropna(subset=['GDP_proxy'])
        if len(complete_data) == 0:
            logger.warning("❌ No complete GDP observations for validation")
            return 0, ["No complete data"]
            
        latest = complete_data.iloc[-1]
        
        validation_score = 100
        issues = []
        
        # Component share validation
        total_gdp = latest['GDP_proxy']
        shares = {
            'C': (latest['C'] / total_gdp * 100, 65, 75),
            'I': (latest['I'] / total_gdp * 100, 15, 25),  # Higher range for real investment data
            'G': (latest['G'] / total_gdp * 100, 15, 30),  # Expanded range for SLCE
        }
        
        for comp, (actual, min_exp, max_exp) in shares.items():
            if pd.notna(actual):
                if not (min_exp <= actual <= max_exp):
                    deduction = min(20, abs(actual - (min_exp + max_exp) / 2))
                    validation_score -= deduction
                    issues.append(f"{comp} share {actual:.1f}% outside {min_exp}-{max_exp}%")
        
        # Magnitude validation
        annual_gdp = total_gdp * 12
        if not (20000 <= annual_gdp <= 35000):  # $20-35T range
            validation_score -= 30
            issues.append(f"Annual GDP ${annual_gdp/1000:.1f}T outside reasonable range")
        
        logger.info(f"📊 Validation Score: {validation_score}/100")
        if issues:
            for issue in issues:
                logger.warning(f"   ⚠️ {issue}")
        
        return validation_score, issues

    def build_proxy(self, start_date="1990-01-01", export=True, strict=False, use_slce=True):
        logger.info("🏗️ Building GDP proxy with REAL INVESTMENT DATA (no scaling)...")
        
        # Get all data
        df = self.fetcher.get_fred_gdp_components(start_date=start_date)
        
        # Create monthly index for alignment
        start_monthly = pd.to_datetime(start_date).replace(day=1)
        end_monthly = df.index.max().replace(day=1)
        monthly_index = pd.date_range(start_monthly, end_monthly, freq='MS')
        
        proxy = pd.DataFrame(index=monthly_index)
        logger.info(f"📅 Target monthly range: {monthly_index.min().date()} to {monthly_index.max().date()}")

        # --- C: Personal Consumption Expenditures (SAAR) ---
        if 'PCE' in df:
            proxy['C'] = safe_saar_to_monthly(df['PCE'], 'PCE').reindex(monthly_index)
            self._validate_component(proxy['C'], 'Consumption', (1200, 2000))
        else:
            proxy['C'] = np.nan
            logger.warning("❌ Missing consumption data (PCE)")

        # --- I: Investment Components (NO SCALING) ---
        investment_parts = []
        
        # PRIORITY 1: Use comprehensive GPDI if available (most complete)
        if 'GPDI' in df:
            gpdi_monthly = quarterly_saar_to_monthly_distributed(df['GPDI'], 'GPDI (Total Investment)', start_date)
            gpdi_monthly = gpdi_monthly.reindex(monthly_index)
            investment_parts.append(('GPDI_Total', gpdi_monthly))
            logger.info(f"✅ Using GPDI (Gross Private Domestic Investment) as primary investment")
        
        # PRIORITY 2: If no GPDI, build from components
        else:
            logger.info(f"📊 Building investment from individual components:")
            
            # Residential Investment (quarterly SAAR)
            if 'PRFIC1' in df:
                residential = quarterly_saar_to_monthly_distributed(df['PRFIC1'], 'Residential Investment', start_date)
                residential = residential.reindex(monthly_index)
                investment_parts.append(('Residential', residential))
                
            # Nonresidential Investment (quarterly SAAR) 
            if 'PNFIC1' in df:
                nonresidential = quarterly_saar_to_monthly_distributed(df['PNFIC1'], 'Nonresidential Investment', start_date)
                nonresidential = nonresidential.reindex(monthly_index)
                investment_parts.append(('Nonresidential', nonresidential))
                
            # Inventory Change (quarterly SAAR)
            if 'CBIC1' in df:
                inventory = quarterly_saar_to_monthly_distributed(df['CBIC1'], 'Inventory Change', start_date)
                inventory = inventory.reindex(monthly_index)
                investment_parts.append(('Inventory', inventory))
                
            # Monthly construction (SAAR) - convert millions to billions  
            if 'TLPRVCONS' in df:
                construction = self._to_billions(safe_saar_to_monthly(df['TLPRVCONS'], 'Construction')).reindex(monthly_index)
                investment_parts.append(('Construction', construction))
        
        # Combine investment components
        if investment_parts:
            # Sum all available investment components
            proxy['I'] = sum([part[1] for part in investment_parts if part[1].notna().any()])
            component_names = [part[0] for part in investment_parts]
            logger.info(f"✅ I: Combined {len(investment_parts)} REAL components: {component_names}")
            self._validate_component(proxy['I'], 'Investment', (400, 800))  # Higher range for real data
        else:
            proxy['I'] = np.nan
            logger.warning("❌ No valid investment components")

        # --- G: Government Spending (FIXED SLCE PROCESSING) ---
        gov_parts = []
        
        # Federal government (quarterly SAAR) - ALREADY IN BILLIONS
        if 'FGCEC1' in df:
            fed_gov = quarterly_saar_to_monthly_distributed(df['FGCEC1'], 'Federal Government', start_date)
            fed_gov = fed_gov.reindex(monthly_index)
            gov_parts.append(('Federal', fed_gov))
            logger.info(f"✅ Federal Gov: Using FGCEC1 (already in billions)")
        
        # State/Local spending - Choose between SLCE (total) or SLCEC1 (current only)
        if use_slce and 'SLCE' in df:
            sl_total = quarterly_saar_to_monthly_distributed(df['SLCE'], 'State/Local Total', start_date)
            sl_total = sl_total.reindex(monthly_index)
            gov_parts.append(('State/Local Total', sl_total))
            logger.info(f"✅ State/Local: Using SLCE (consumption + investment) with date filter")
        elif 'SLCEC1' in df:
            # Fallback to SLCEC1 (Current expenditures only)
            sl_current = quarterly_saar_to_monthly_distributed(df['SLCEC1'], 'State/Local Current', start_date)
            sl_current = sl_current.reindex(monthly_index)
            gov_parts.append(('State/Local Current', sl_current))
            logger.info(f"✅ State/Local: Using SLCEC1 (current expenditures only)")
        else:
            logger.warning("❌ No state/local government data available")
        
        # Combine government spending
        if gov_parts:
            # Sum all government components
            proxy['G'] = sum([part[1] for part in gov_parts if part[1].notna().any()])
            component_names = [part[0] for part in gov_parts]
            logger.info(f"✅ G: Combined {len(gov_parts)} government components: {component_names}")
            
            # Adjust validation range based on whether SLCE is included
            expected_range = (500, 800) if use_slce else (200, 400)  # Higher range for SLCE
            self._validate_component(proxy['G'], 'Government', expected_range)
        else:
            proxy['G'] = np.nan
            logger.warning("❌ No valid government spending data")

        # --- NX: Net Exports (SAAR) - convert millions to billions ---
        if 'BOPGEXP' in df and 'BOPGIMP' in df:
            exports = self._to_billions(safe_saar_to_monthly(df['BOPGEXP'], 'Exports')).reindex(monthly_index)
            imports = self._to_billions(safe_saar_to_monthly(df['BOPGIMP'], 'Imports')).reindex(monthly_index)
            proxy['NX'] = exports - imports
            logger.info(f"✅ NX: Exports minus Imports")
            # Net exports can be negative, so wider validation range
            self._validate_component(proxy['NX'], 'Net Exports', (-100, 100))
        else:
            proxy['NX'] = np.nan
            logger.warning("❌ Net exports data missing")

        # --- Final GDP Assembly ---
        required_components = ['C', 'I', 'G', 'NX']
        
        # Component availability summary
        logger.info(f"📊 Component Data Availability:")
        for comp in required_components:
            if comp in proxy.columns:
                available = proxy[comp].notna().sum()
                total = len(proxy)
                pct = available / total * 100
                recent_value = proxy[comp].dropna().tail(1)
                if not recent_value.empty:
                    logger.info(f"   {comp}: {available}/{total} months ({pct:.1f}%) - Latest: ${recent_value.iloc[0]:.0f}B")
                else:
                    logger.info(f"   {comp}: {available}/{total} months ({pct:.1f}%) - No recent data")
            else:
                logger.info(f"   {comp}: Missing")
        
        # Calculate GDP proxy (allow missing components if not strict)
        min_components = 4 if strict else 2
        proxy['GDP_proxy'] = proxy[required_components].sum(axis=1, min_count=min_components)
        
        # Filter to complete data only
        complete_proxy = self._filter_to_complete_data_only(proxy)
        
        # Final validation on complete data
        complete_obs = len(complete_proxy)
        logger.info(f"📈 Complete GDP Proxy: {complete_obs} observations")
        
        if complete_obs > 0:
            # Show recent component breakdown
            recent_complete = complete_proxy.tail(3)
            if not recent_complete.empty:
                logger.info(f"📋 Recent Complete Months:")
                for idx, row in recent_complete.iterrows():
                    components_str = " + ".join([f"{comp}=${row[comp]:.0f}" for comp in required_components if pd.notna(row[comp])])
                    logger.info(f"   {idx.date()}: GDP=${row['GDP_proxy']:.0f}B ({components_str})")
                
                # Comprehensive validation
                validation_score, issues = self._comprehensive_validation(complete_proxy)
                
                # Component share summary for latest complete month
                latest = recent_complete.iloc[-1]
                total_gdp = latest['GDP_proxy']
                
                logger.info(f"📋 Component Shares ({latest.name.date()}):")
                for comp in required_components:
                    if pd.notna(latest[comp]):
                        share = latest[comp] / total_gdp * 100
                        logger.info(f"   {comp}: ${latest[comp]:.0f}B ({share:.1f}%)")
                
                # Annual comparison
                annual_proxy = total_gdp * 12
                logger.info(f"📈 Annualized GDP: ${annual_proxy:,.0f}B")
                logger.info(f"   US GDP Reference (~2024): ~$28,000B")
                logger.info(f"   Ratio: {annual_proxy/28000:.2f}")

        # Export complete data if requested
        if export and complete_obs > 0:
            self.cache_dir.mkdir(exist_ok=True)
            complete_proxy.to_csv(self.output_path)
            logger.info(f"✅ Exported {complete_obs} complete observations to {self.output_path}")

        return complete_proxy if complete_obs > 0 else proxy


def build_gdp_proxy(start_date="1990-01-01", export=True, strict=False, use_slce=True):
    """
    Build monthly GDP proxy with REAL INVESTMENT DATA (no scaling).
    
    Args:
        start_date: Start date for data fetch
        export: Whether to save to CSV  
        strict: Require all 4 components vs allow partial GDP
        use_slce: Use SLCE (total) vs SLCEC1 (current only) for state/local government
    
    Returns:
        DataFrame with monthly GDP proxy and components (complete data only)
    """
    builder = GDPProxyBuilder()
    return builder.build_proxy(start_date=start_date, export=export, strict=strict, use_slce=use_slce)


if __name__ == "__main__":
    print("🚀 Building GDP proxy with REAL INVESTMENT DATA (No Scaling)...")
    
    # Test with comprehensive investment data
    print("\n" + "="*60)
    print("GDP Proxy with Real Investment Components")
    print("="*60)
    df_real = build_gdp_proxy(start_date="2020-01-01", export=False, use_slce=True)
    
    if not df_real.empty:
        latest = df_real.iloc[-1]
        
        print(f"\n📊 Latest Complete Month: {latest.name.date()}")
        print(f"📈 Total GDP: ${latest['GDP_proxy']:,.0f}B/month")
        print(f"📈 Annualized: ${latest['GDP_proxy']*12:,.0f}B/year")
        print(f"\n📋 Component Breakdown:")
        print(f"   Consumption (C): ${latest['C']:,.0f}B ({latest['C']/latest['GDP_proxy']*100:.1f}%)")
        print(f"   Investment (I):  ${latest['I']:,.0f}B ({latest['I']/latest['GDP_proxy']*100:.1f}%)")
        print(f"   Government (G):  ${latest['G']:,.0f}B ({latest['G']/latest['GDP_proxy']*100:.1f}%)")
        print(f"   Net Exports (NX): ${latest['NX']:,.0f}B ({latest['NX']/latest['GDP_proxy']*100:.1f}%)")
        
        print(f"\n🎯 Key Improvements from Real Data:")
        print(f"   ✅ No synthetic scaling of manufacturing orders")
        print(f"   ✅ Uses actual FRED investment series (GPDI, PRFIC1, PNFIC1, etc.)")
        print(f"   ✅ Investment component should be more accurate")
        print(f"   ✅ All numbers based on real economic data")
        
        # Show trend
        if len(df_real) >= 12:
            yoy_growth = (latest['GDP_proxy'] / df_real.iloc[-13]['GDP_proxy'] - 1) * 100
            print(f"\n📈 Year-over-Year Growth: {yoy_growth:.1f}%")
            
    else:
        print("❌ Could not build GDP proxy - check data availability")