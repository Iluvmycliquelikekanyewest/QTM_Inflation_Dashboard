"""
Data fetching module for FRED and BEA APIs.
Handles raw data pulls with caching.
UPDATED: Added comprehensive investment series for accurate GDP proxy.
UPDATED: Added monthly I/G proxy series for intra-quarter estimation.
FIXED: Added CPI data for inflation comparison.
FIXED: Removed CPI inflation calculations - now handled by analysis_logic.py
"""
import pandas as pd
import requests
from fredapi import Fred
from pathlib import Path
import json
import hashlib
from datetime import datetime
import logging

# Handle both relative and absolute imports
try:
    from .config import Config
except ImportError:
    from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for I/G proxy weights
IG_WEIGHTS = {
    'I': {'PRRESCONS': 0.4, 'PNRESCONS': 0.4, 'NEWORDER': 0.2},
    'G': {'OUTLAY': 1.0}
}

class DataFetcher:
    """Handles data fetching from FRED and BEA APIs with caching."""
    
    def __init__(self):
        self.fred = Fred(api_key=Config.FRED_API_KEY)
        self.bea_key = Config.BEA_API_KEY
        self.cache_dir = Config.CACHE_DIR
    
    def get_fred(self, codes, start_date='1990-01-01'):
        # Handle both list and dict inputs
        if isinstance(codes, list):
            series_map = {code: code for code in codes}
        else:
            series_map = codes
        
        series_hash = hashlib.md5(str(sorted(series_map.items())).encode()).hexdigest()[:8]
        cache_file = self.cache_dir / f"fred_{start_date.replace('-', '')}_{series_hash}.csv"
        
        if cache_file.exists():
            logger.info(f"Loading FRED data from cache: {cache_file}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        logger.info(f"Fetching FRED data for {len(series_map)} series from {start_date}")
        data = {}
        
        for name, code in series_map.items():
            try:
                logger.info(f"Fetching {name} ({code})")
                series = self.fred.get_series(code, start=start_date)
                data[name] = series
                logger.info(f"✅ Successfully fetched {name}: {len(series)} observations")
            except Exception as e:
                logger.error(f"❌ Error fetching {name} ({code}): {e}")
                # Continue with other series instead of raising
                continue
        
        if not data:
            raise ValueError("Failed to fetch any FRED series")
        
        df = pd.DataFrame(data).dropna(how='all')
        df.to_csv(cache_file)
        logger.info(f"Cached FRED data to {cache_file}")
        logger.info(f"Final dataset: {len(df)} periods, {len(df.columns)} series")
        return df
    
    def _clean_bea_value(self, value):
        if pd.isna(value) or value in ['(D)', '(NA)', '']:
            return float('nan')
        if isinstance(value, str):
            value = value.replace(',', '')
        try:
            return float(value)
        except (ValueError, TypeError):
            return float('nan')

    def get_fred_gdp_components(self, start_date='1990-01-01'):
        """
        Fetch all FRED series needed for GDP proxy construction.
        
        UPDATED: Added comprehensive investment series to eliminate need for scaling.
        FIXED: Added CPI data for inflation comparison.
        FIXED: Removed CPI inflation calculations - raw data only.
        """
        gdp_series_map = {
            # ✅ CONSUMPTION
            'PCE': 'PCE',  # Personal Consumption Expenditures (SAAR, Billions)

            # ✅ INVESTMENT - COMPREHENSIVE COVERAGE
            # Private Fixed Investment (Quarterly SAAR)
            'PRFIC1': 'PRFIC1',        # Private Residential Fixed Investment 
            'PNFIC1': 'PNFIC1',        # Private Nonresidential Fixed Investment
            'PNFIC96': 'PNFIC96',      # Private Nonresidential Fixed Investment (Real)
            
            # Investment by Type (Quarterly SAAR)
            'CBIC1': 'CBIC1',          # Change in Private Inventories
            'PRRESCONS': 'PRRESCONS',  # Private Residential Construction
            
            # Comprehensive Investment (Quarterly SAAR) 
            'GPDI': 'GPDI',            # Gross Private Domestic Investment (Total)
            'GPDIC1': 'GPDIC1',        # Gross Private Domestic Investment (Current $)
            
            # Equipment & Structures (Quarterly SAAR)
            'Y033RC1Q027SBEA': 'Y033RC1Q027SBEA',  # Business Equipment Investment
            
            # Monthly Investment Indicators (Monthly SAAR)
            'TLPRVCONS': 'TLPRVCONS',  # Total Private Construction
            'HOUST': 'HOUST',          # Housing Starts (for residential validation)
            
            # Additional Economic Indicators (for validation/context)
            'INDPRO': 'INDPRO',        # Industrial Production Index
            'ISRATIO': 'ISRATIO',      # Inventory to Sales Ratio

            # ✅ GOVERNMENT  
            'FGCEC1': 'FGCEC1',        # Federal Government Consumption & Investment
            'SLCEC1': 'SLCEC1',        # State/Local Current Expenditures  
            'SLCE': 'SLCE',            # State/Local Total Expenditures

            # ✅ NET EXPORTS
            'BOPXGS': 'BOPXGS',        # Exports of Goods & Services
            'BOPMGS': 'BOPMGS',        # Imports of Goods & Services  
            'BOPGEXP': 'BOPGEXP',      # Exports of Goods
            'BOPGIMP': 'BOPGIMP',      # Imports of Goods

            # ✅ MONEY SUPPLY
            'M2': 'M2SL',              # M2 Money Supply

            # ✅ INFLATION DATA (FIXED: Raw data only, no calculations)
            'CPIAUCSL': 'CPIAUCSL',    # Consumer Price Index for All Urban Consumers (CPI-U) - Monthly
            'CPILFESL': 'CPILFESL',    # Core CPI (excluding food and energy) - Monthly, optional

            # ✅ VALIDATION
            'GDP': 'GDP'               # Actual GDP for validation
        }
        
        logger.info("🔄 Fetching FRED GDP component series")
        logger.info(f"📊 Requesting {len(gdp_series_map)} series:")
        for name, code in gdp_series_map.items():
            logger.info(f"   {name} -> {code}")
        
        gdp_data = self.get_fred(gdp_series_map, start_date=start_date)
        
        logger.info(f"✅ Successfully fetched {len(gdp_data.columns)} GDP component series")
        logger.info(f"📅 Date range: {gdp_data.index.min()} to {gdp_data.index.max()}")
        logger.info(f"📈 Available series: {sorted(gdp_data.columns)}")
        
        # Validate that we have the core components needed
        required_for_gdp = ['PCE']  # Minimum requirement
        available = set(gdp_data.columns)
        missing_critical = [s for s in required_for_gdp if s not in available]
        
        if missing_critical:
            raise ValueError(f"Missing critical GDP series: {missing_critical}")
        
        # Log what we have for each GDP component
        consumption_series = [s for s in ['PCE'] if s in available]
        investment_series = [s for s in ['GPDI', 'GPDIC1', 'PRFIC1', 'PNFIC1', 'CBIC1', 'TLPRVCONS'] if s in available]
        government_series = [s for s in ['FGCEC1', 'SLCE', 'SLCEC1'] if s in available]
        netexport_series = [s for s in ['BOPGEXP', 'BOPGIMP', 'BOPXGS', 'BOPMGS'] if s in available]
        inflation_series = [s for s in ['CPIAUCSL', 'CPILFESL'] if s in available]
        
        logger.info(f"📊 GDP component coverage:")
        logger.info(f"   Consumption (C): {consumption_series}")
        logger.info(f"   Investment (I): {investment_series}")
        logger.info(f"   Government (G): {government_series}")
        logger.info(f"   Net Exports (NX): {netexport_series}")
        logger.info(f"   Money Supply: {['M2'] if 'M2' in available else []}")
        logger.info(f"   Inflation Data: {inflation_series}")
        
        # FIXED: Debug CPI data if available (but don't calculate inflation here!)
        if 'CPIAUCSL' in available:
            cpi_data = gdp_data['CPIAUCSL'].dropna()
            logger.info(f"🔍 CPI data fetched: {len(cpi_data)} observations")
            logger.info(f"🔍 CPI date range: {cpi_data.index.min()} to {cpi_data.index.max()}")
            
            # Show recent CPI values (raw data only)
            if len(cpi_data) >= 5:
                recent_cpi = cpi_data.tail(5)
                logger.info(f"🔍 Recent CPI values: {recent_cpi.round(1).to_dict()}")
                
                # Show December CPI values for reference (but don't calculate inflation)
                dec_dates = cpi_data[cpi_data.index.month == 12]
                if len(dec_dates) >= 2:
                    logger.info(f"🔍 Recent December CPI values (raw data):")
                    for i in range(max(0, len(dec_dates)-3), len(dec_dates)):
                        dec_date = dec_dates.index[i]
                        dec_value = dec_dates.iloc[i]
                        logger.info(f"   Dec {dec_date.year}: {dec_value:.1f}")
                        
                    logger.info("📝 Note: CPI inflation will be calculated by analysis_logic.py based on selected frequency")
        else:
            logger.warning("⚠️ CPI data not fetched - inflation comparison will not work")
        
        return gdp_data

    def get_ig_proxy_series(self, start_date='1990-01-01') -> pd.DataFrame:
        """
        Fetch high-frequency monthly FRED series used as proxies for Investment (I) and Government (G).
        These are used to interpolate missing monthly I/G data based on BEA quarterly anchors.
        
        The formula for monthly estimation is:
        I_month = (I_proxy_m / sum(I_proxy_q)) * I_BEA_Q
        G_month = (G_proxy_m / sum(G_proxy_q)) * G_BEA_Q
        
        Returns:
            pd.DataFrame: Monthly proxy series with columns for each component
        """
        proxy_series = {
            # Investment proxies (monthly)
            'PRRESCONS': 'PRRESCONS',   # Private Residential Construction
            'PNRESCONS': 'PNRESCONS',   # Private Nonresidential Construction  
            'NEWORDER': 'NEWORDER',     # Core capital goods orders (nondefense, ex-aircraft)

            # Government proxy (monthly)
            'OUTLAY': 'FGEXPND'         # Total federal government outlays
        }

        logger.info("🔄 Fetching monthly I/G proxy series for intra-quarter estimation...")
        logger.info(f"📊 Investment proxies: {list(proxy_series.keys())[:3]}")
        logger.info(f"📊 Government proxy: {list(proxy_series.keys())[3:]}")
        
        proxy_data = self.get_fred(proxy_series, start_date=start_date)
        
        logger.info(f"✅ Successfully fetched {len(proxy_data.columns)} I/G proxy series")
        logger.info(f"📅 Date range: {proxy_data.index.min()} to {proxy_data.index.max()}")
        logger.info(f"📈 Available proxies: {sorted(proxy_data.columns)}")
        
        # Validate we have the minimum required proxies
        required_proxies = ['PRRESCONS', 'OUTLAY']  # Minimum for I and G
        available = set(proxy_data.columns)
        missing_critical = [s for s in required_proxies if s not in available]
        
        if missing_critical:
            logger.warning(f"Missing some I/G proxies: {missing_critical}")
            logger.warning("Monthly I/G estimation may be less accurate")
        
        # Log coverage by component
        investment_proxies = [s for s in ['PRRESCONS', 'PNRESCONS', 'NEWORDER'] if s in available]
        government_proxies = [s for s in ['OUTLAY'] if s in available]
        
        logger.info(f"📊 I/G proxy coverage:")
        logger.info(f"   Investment (I): {investment_proxies}")
        logger.info(f"   Government (G): {government_proxies}")
        
        return proxy_data

    def get_combined_ig_proxy(self, start_date='1990-01-01') -> pd.DataFrame:
        """
        Fetch I/G proxy series and return weighted composite indices.
        
        Returns:
            pd.DataFrame: Composite I and G proxy series (I_proxy, G_proxy)
        """
        proxy_data = self.get_ig_proxy_series(start_date)
        
        # Create weighted composite indices
        result = pd.DataFrame(index=proxy_data.index)
        
        # Investment composite (weighted average)
        i_components = []
        for component, weight in IG_WEIGHTS['I'].items():
            if component in proxy_data.columns:
                i_components.append(proxy_data[component] * weight)
                logger.info(f"Adding {component} to I_proxy with weight {weight}")
            else:
                logger.warning(f"Missing I component: {component}")
        
        if i_components:
            result['I_proxy'] = sum(i_components)
        else:
            logger.error("No Investment proxy components available!")
            
        # Government composite (weighted average)
        g_components = []
        for component, weight in IG_WEIGHTS['G'].items():
            if component in proxy_data.columns:
                g_components.append(proxy_data[component] * weight)
                logger.info(f"Adding {component} to G_proxy with weight {weight}")
            else:
                logger.warning(f"Missing G component: {component}")
        
        if g_components:
            result['G_proxy'] = sum(g_components)
        else:
            logger.error("No Government proxy components available!")
            
        logger.info(f"✅ Created composite I/G proxies: {list(result.columns)}")
        return result

    def get_bea_gdp_shares(self, fill_method='ffill'):
        cache_file = self.cache_dir / f"bea_gdp_shares_{fill_method}.csv.gz"
        if cache_file.exists():
            logger.info(f"Loading BEA data from cache: {cache_file}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True, compression='gzip')

        table_params = {
            'UserID': self.bea_key,
            'Method': 'GetParameterList',
            'DatasetName': 'NIPA',
            'ParameterName': 'TableName',
            'ResultFormat': 'json'
        }

        logger.info("Getting BEA table structure")
        try:
            response = requests.get('https://apps.bea.gov/api/data', params=table_params, timeout=30)
            response.raise_for_status()

            params = {
                'UserID': self.bea_key,
                'Method': 'GetData',
                'DatasetName': 'NIPA',
                'TableName': 'T10105',
                'Frequency': 'Q',
                'Year': 'ALL',
                'ResultFormat': 'json'
            }

            logger.info("Fetching BEA GDP components data (all lines)")
            response = requests.get('https://apps.bea.gov/api/data', params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            results = data['BEAAPI']['Results']['Data']
            df_raw = pd.DataFrame(results)

            component_patterns = {
                'GDP': ['Gross domestic product'],
                'PCE': ['Personal consumption expenditures'],
                'GPDI': ['Gross private domestic investment'],
                'GCE': ['Government consumption expenditures', 'government consumption expenditures and gross investment'],
                'NETEXP': ['Net exports', 'net exports of goods and services']
            }

            component_lines = {}
            for component, patterns in component_patterns.items():
                for _, row in df_raw.iterrows():
                    line_desc = row.get('LineDescription', '').lower()
                    for pattern in patterns:
                        if pattern.lower() in line_desc:
                            component_lines[component] = row['LineNumber']
                            logger.info(f"Found {component} at line {row['LineNumber']}: {row['LineDescription']}")
                            break
                    if component in component_lines:
                        break

            if not component_lines:
                raise ValueError("Could not find any GDP components in BEA data")

            df_filtered = df_raw[df_raw['LineNumber'].isin(component_lines.values())].copy()
            df_filtered['Component'] = df_filtered['LineNumber'].map({v: k for k, v in component_lines.items()})
            df_filtered['Date'] = pd.PeriodIndex(df_filtered['TimePeriod'], freq='Q').to_timestamp('Q')
            df_filtered['DataValue'] = df_filtered['DataValue'].apply(self._clean_bea_value)
            df_pivot = df_filtered.pivot(index='Date', columns='Component', values='DataValue')

            available = set(df_pivot.columns)
            expected = {'GDP', 'PCE', 'GPDI', 'GCE', 'NETEXP'}
            missing = expected - available
            if missing:
                logger.warning(f"BEA API missing components: {missing}")
            if 'GDP' not in available:
                raise ValueError("BEA API response missing GDP data - cannot calculate shares")

            shares = []
            info = {}
            if 'PCE' in available:
                df_pivot['share_C'] = df_pivot['PCE'] / df_pivot['GDP']
                shares.append('share_C')
                info['share_C'] = df_pivot['share_C'].last_valid_index()
            if 'GPDI' in available:
                df_pivot['share_I'] = df_pivot['GPDI'] / df_pivot['GDP']
                shares.append('share_I')
                info['share_I'] = df_pivot['share_I'].last_valid_index()
            if 'GCE' in available:
                df_pivot['share_G'] = df_pivot['GCE'] / df_pivot['GDP']
                shares.append('share_G')
                info['share_G'] = df_pivot['share_G'].last_valid_index()
            if 'NETEXP' in available:
                df_pivot['share_NX'] = df_pivot['NETEXP'] / df_pivot['GDP']
                shares.append('share_NX')
                info['share_NX'] = df_pivot['share_NX'].last_valid_index()

            df_shares = df_pivot[shares].astype('float64').sort_index()

            if fill_method == 'ffill':
                for col in shares:
                    before = df_shares[col].count()
                    df_shares[col] = df_shares[col].ffill()
                    after = df_shares[col].count()
                    if after > before:
                        logger.info(f"Forward filled {after - before} values for {col}")
                logger.info("Applied forward fill")
            elif fill_method == 'interpolate':
                df_shares = df_shares.interpolate(method='linear')
                logger.info("Applied linear interpolation")
            elif fill_method == 'none':
                logger.info("Keeping NaNs as-is")

            df_shares = df_shares.dropna(how='all')

            logger.info(f"Calculated {len(shares)} GDP component shares")
            logger.info(f"Final dataset has {len(df_shares)} quarters")
            for k, v in info.items():
                logger.info(f"{k} available through {v}")

            df_shares.to_csv(cache_file, compression='gzip')
            logger.info(f"Cached BEA data to {cache_file}")
            return df_shares

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching BEA data: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing BEA data: {e}")
            raise

    def get_fred_basic(self, start_date='1990-01-01'):
        """
        DEPRECATED: Use get_fred_gdp_components() instead.
        
        This method fetches only basic series for backwards compatibility.
        """
        logger.warning("get_fred_basic() is deprecated. Use get_fred_gdp_components() instead.")
        
        basic_series_map = {
            'PCE': 'PCE',
            'M2': 'M2SL'
        }
        
        return self.get_fred(basic_series_map, start_date=start_date)


# Convenience functions
def get_fred(codes, start_date='1990-01-01'):
    """Fetch arbitrary FRED series."""
    fetcher = DataFetcher()
    return fetcher.get_fred(codes, start_date)

def get_fred_gdp_components(start_date='1990-01-01'):
    """Fetch all FRED series needed for GDP proxy construction."""
    fetcher = DataFetcher()
    return fetcher.get_fred_gdp_components(start_date)

def get_ig_proxy_series(start_date='1990-01-01'):
    """Fetch monthly FRED proxy series for Investment and Government estimation."""
    fetcher = DataFetcher()
    return fetcher.get_ig_proxy_series(start_date)

def get_combined_ig_proxy(start_date='1990-01-01'):
    """Fetch monthly I/G proxy series and return weighted composite indices."""
    fetcher = DataFetcher()
    return fetcher.get_combined_ig_proxy(start_date)

def get_fred_basic(start_date='1990-01-01'):
    """DEPRECATED: Fetch basic FRED series (PCE, M2 only)."""
    fetcher = DataFetcher()
    return fetcher.get_fred_basic(start_date)

def get_bea_gdp_shares(fill_method='ffill'):
    """Fetch BEA GDP component shares."""
    fetcher = DataFetcher()
    return fetcher.get_bea_gdp_shares(fill_method)