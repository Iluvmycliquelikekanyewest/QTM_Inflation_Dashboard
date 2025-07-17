"""
FIXED Velocity and monetary inflation calculation module.
Updated for new GDP proxy format (monthly, nominal, billions).
Implements Quantity Theory of Money: MV = PY → V = PY / M.
KEY FIX: Velocity calculation now matches the target frequency (monthly/quarterly/annual).
CORRECTED: ΔP = ΔM + ΔV - ΔY now uses consistent timeframes.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Literal

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resample_to_frequency(series: pd.Series, frequency: Literal['monthly', 'quarterly', 'annual']) -> pd.Series:
    """
    Resample series to target frequency with appropriate aggregation.
    
    Args:
        series: Input time series (assumed monthly)
        frequency: Target frequency
        
    Returns:
        Resampled series at target frequency
    """
    if frequency == 'monthly':
        return series
    elif frequency == 'quarterly':
        # For stocks (like M2): use end-of-quarter values
        # For flows (like GDP): sum quarterly values
        if series.name and 'M2' in str(series.name).upper():
            return series.resample('Q').last()
        else:
            return series.resample('Q').sum()
    elif frequency == 'annual':
        # For stocks: use end-of-year values
        # For flows: sum annual values  
        if series.name and 'M2' in str(series.name).upper():
            return series.resample('A').last()
        else:
            return series.resample('A').sum()
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")


def calc_velocity_with_frequency(gdp_proxy: pd.Series, 
                                 money_supply: pd.Series,
                                 frequency: Literal['monthly', 'quarterly', 'annual'] = 'monthly') -> pd.DataFrame:
    """
    Calculate velocity of money at specified frequency: V = GDP / M
    
    KEY FIX: Now resamples both GDP and money supply to target frequency BEFORE calculating velocity.
    This ensures QTM components are all at the same timeframe.
    
    Args:
        gdp_proxy: Monthly GDP proxy (nominal, billions)
        money_supply: Monthly money supply (M2, billions) 
        frequency: Target calculation frequency
        
    Returns:
        DataFrame with velocity, growth rates, and QTM inflation at consistent frequency
    """
    if gdp_proxy.empty or money_supply.empty:
        logger.warning("⚠️ Cannot calculate velocity with empty input series.")
        return pd.DataFrame()

    # Step 1: Align monthly data
    df = pd.DataFrame({'GDP_proxy': gdp_proxy, 'M2': money_supply}).dropna()
    if df.empty or len(df) < 2:
        logger.warning("⚠️ Not enough data to compute velocity.")
        return df

    logger.info(f"📊 Calculating velocity at {frequency} frequency")
    logger.info(f"📅 Input data: {len(df)} monthly observations")

    # Step 2: CRITICAL FIX - Resample to target frequency BEFORE velocity calculation
    df_freq = pd.DataFrame()
    
    # Set series names for proper resampling logic
    gdp_series = df['GDP_proxy'].copy()
    gdp_series.name = 'GDP_proxy'
    m2_series = df['M2'].copy() 
    m2_series.name = 'M2'
    
    df_freq['GDP_proxy'] = resample_to_frequency(gdp_series, frequency)
    df_freq['M2'] = resample_to_frequency(m2_series, frequency)
    df_freq = df_freq.dropna()
    
    if len(df_freq) < 2:
        logger.warning(f"⚠️ Not enough {frequency} data after resampling.")
        return df_freq
        
    logger.info(f"📅 After resampling: {len(df_freq)} {frequency} observations")

    # Step 3: Annualize GDP if calculating velocity (for proper scale)
    if frequency == 'monthly':
        df_freq['GDP_proxy_annual'] = df_freq['GDP_proxy'] * 12
        periods_per_year = 12
    elif frequency == 'quarterly': 
        df_freq['GDP_proxy_annual'] = df_freq['GDP_proxy'] * 4
        periods_per_year = 4
    else:  # annual
        df_freq['GDP_proxy_annual'] = df_freq['GDP_proxy']
        periods_per_year = 1

    # Step 4: Calculate velocity at target frequency
    df_freq['velocity'] = df_freq['GDP_proxy_annual'] / df_freq['M2']

    # Step 5: Calculate growth rates at appropriate lags
    if frequency == 'monthly':
        df_freq['velocity_mom'] = df_freq['velocity'].pct_change()
        df_freq['velocity_yoy'] = df_freq['velocity'].pct_change(periods=12)
        df_freq['money_growth_mom'] = df_freq['M2'].pct_change()
        df_freq['money_growth_yoy'] = df_freq['M2'].pct_change(periods=12)
        df_freq['nominal_gdp_growth_mom'] = df_freq['GDP_proxy_annual'].pct_change()
        df_freq['nominal_gdp_growth_yoy'] = df_freq['GDP_proxy_annual'].pct_change(periods=12)
    elif frequency == 'quarterly':
        df_freq['velocity_qoq'] = df_freq['velocity'].pct_change()
        df_freq['velocity_yoy'] = df_freq['velocity'].pct_change(periods=4)
        df_freq['money_growth_qoq'] = df_freq['M2'].pct_change()
        df_freq['money_growth_yoy'] = df_freq['M2'].pct_change(periods=4)
        df_freq['nominal_gdp_growth_yoy'] = df_freq['GDP_proxy_annual'].pct_change(periods=4)
    else:  # annual
        df_freq['velocity_yoy'] = df_freq['velocity'].pct_change()
        df_freq['money_growth_yoy'] = df_freq['M2'].pct_change()
        df_freq['nominal_gdp_growth_yoy'] = df_freq['GDP_proxy_annual'].pct_change()

    # Step 6: FIXED QTM inflation calculation with consistent frequency
    # QTM: MV = PY → ΔM + ΔV = ΔP + ΔY → ΔP = ΔM + ΔV - ΔY
    
    assumed_real_growth_annual = 0.02  # 2% annual real GDP trend
    
    if frequency == 'monthly':
        assumed_real_growth_monthly = assumed_real_growth_annual / 12
        df_freq['qtm_inflation_mom'] = (df_freq['money_growth_mom'] + 
                                       df_freq['velocity_mom'] - 
                                       assumed_real_growth_monthly)
        df_freq['qtm_inflation_mom_annual'] = (1 + df_freq['qtm_inflation_mom']) ** 12 - 1
        
    # Year-over-year inflation (main calculation)
    df_freq['qtm_inflation_yoy'] = (df_freq['money_growth_yoy'] + 
                                   df_freq['velocity_yoy'] - 
                                   assumed_real_growth_annual)
    
    # Alternative: derive from nominal GDP growth
    df_freq['inflation_from_nominal'] = (df_freq['nominal_gdp_growth_yoy'] - 
                                        assumed_real_growth_annual)

    # Set main inflation columns for backward compatibility
    if frequency == 'monthly':
        df_freq['inflation_mom'] = df_freq['qtm_inflation_mom']
        df_freq['inflation_mom_annual'] = df_freq['qtm_inflation_mom_annual']
    df_freq['inflation_yoy'] = df_freq['qtm_inflation_yoy']

    # Step 7: Validation and logging
    logger.info("✅ Velocity calculation complete with frequency matching.")
    logger.info(f"📅 {frequency.title()} coverage: {df_freq.index.min().date()} → {df_freq.index.max().date()}")
    logger.info(f"📊 Mean {frequency} velocity: {df_freq['velocity'].mean():.2f}")
    
    # Show recent QTM components for validation
    recent = df_freq.dropna().tail(3)
    if not recent.empty:
        logger.info(f"📊 Recent {frequency} QTM inflation components (YoY):")
        for idx, row in recent.iterrows():
            money_growth = row['money_growth_yoy'] * 100
            velocity_change = row['velocity_yoy'] * 100
            inflation = row['qtm_inflation_yoy'] * 100
            logger.info(f"   {idx.date()}: Money Growth: {money_growth:+.1f}%, "
                       f"Velocity Change: {velocity_change:+.1f}%, "
                       f"QTM Inflation: {inflation:+.1f}%")

    # Check for extreme values
    extreme_vals = df_freq[(df_freq['velocity'] < 0.1) | (df_freq['velocity'] > 20)]
    if not extreme_vals.empty:
        logger.warning(f"⚠️ {len(extreme_vals)} periods with extreme velocity values.")

    return df_freq


def calc_velocity(gdp_proxy: pd.Series, money_supply: pd.Series, 
                  frequency: str = 'Monthly') -> pd.DataFrame:
    """
    Calculate velocity of money: V = GDP_proxy_annual / M2
    
    UPDATED: Now frequency-aware! This is the main function called by the GUI.
    FIXED: Calculates inflation using proper QTM formula with consistent timeframes.
    
    Args:
        gdp_proxy: Monthly GDP proxy (nominal, billions)
        money_supply: Monthly money supply series
        frequency: Calculation frequency ('Monthly', 'Quarterly', 'Annually')
    
    Returns:
        DataFrame with velocity, MoM/YoY growth, and CORRECT QTM inflation.
    """
    # Convert GUI frequency strings to lowercase
    freq_map = {
        'Monthly': 'monthly',
        'Quarterly': 'quarterly', 
        'Annually': 'annual',
        'monthly': 'monthly',
        'quarterly': 'quarterly',
        'annual': 'annual'
    }
    
    freq_lower = freq_map.get(frequency, 'monthly')
    if frequency not in freq_map:
        logger.warning(f"⚠️ Unknown frequency '{frequency}', defaulting to monthly")
        freq_lower = 'monthly'
    
    logger.info(f"🔄 Running velocity calculation at {freq_lower} frequency")
    
    # Use the new frequency-aware calculation
    return calc_velocity_with_frequency(gdp_proxy, money_supply, freq_lower)


def calculate_quantity_theory_inflation(gdp_proxy: pd.Series, money_supply: pd.Series,
                                        frequency: str = 'monthly',
                                        real_gdp: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Estimate inflation using QTM: MV = PY → ΔP ≈ ΔM + ΔV - ΔY
    
    ENHANCED: Now frequency-aware with better handling of real GDP.
    
    Args:
        gdp_proxy: Monthly nominal GDP proxy
        money_supply: Monthly money supply
        frequency: Target frequency
        real_gdp: Optional real GDP series
        
    Returns:
        DataFrame with QTM inflation at specified frequency
    """
    # Convert frequency string
    freq_map = {'monthly': 'monthly', 'quarterly': 'quarterly', 'annual': 'annual'}
    freq_lower = freq_map.get(frequency.lower(), 'monthly')
    
    df = pd.DataFrame({'nominal_gdp': gdp_proxy, 'money_supply': money_supply}).dropna()
    if len(df) < 13:
        logger.warning("⚠️ Need at least 13 months for YoY QTM inflation.")
        return df

    logger.info(f"📊 Calculating QTM inflation at {freq_lower} frequency")

    # Resample to target frequency
    gdp_series = df['nominal_gdp'].copy()
    gdp_series.name = 'nominal_gdp'
    m2_series = df['money_supply'].copy()
    m2_series.name = 'money_supply'
    
    df_freq = pd.DataFrame()
    df_freq['nominal_gdp'] = resample_to_frequency(gdp_series, freq_lower)
    df_freq['money_supply'] = resample_to_frequency(m2_series, freq_lower)
    df_freq = df_freq.dropna()

    # Annualize GDP for velocity calculation
    if freq_lower == 'monthly':
        df_freq['nominal_gdp_annual'] = df_freq['nominal_gdp'] * 12
        periods_for_yoy = 12
    elif freq_lower == 'quarterly':
        df_freq['nominal_gdp_annual'] = df_freq['nominal_gdp'] * 4
        periods_for_yoy = 4
    else:  # annual
        df_freq['nominal_gdp_annual'] = df_freq['nominal_gdp']
        periods_for_yoy = 1

    # Calculate velocity
    df_freq['velocity'] = df_freq['nominal_gdp_annual'] / df_freq['money_supply']

    # YoY growth rates
    df_freq['money_growth'] = df_freq['money_supply'].pct_change(periods_for_yoy)
    df_freq['nominal_gdp_growth'] = df_freq['nominal_gdp_annual'].pct_change(periods_for_yoy)
    df_freq['velocity_growth'] = df_freq['velocity'].pct_change(periods_for_yoy)

    if real_gdp is not None:
        # Use actual real GDP data
        real_gdp_series = real_gdp.copy()
        real_gdp_series.name = 'real_gdp'
        df_freq['real_gdp'] = resample_to_frequency(real_gdp_series, freq_lower)
        df_freq['real_gdp_growth'] = df_freq['real_gdp'].pct_change(periods_for_yoy)
        df_freq['qtm_inflation'] = df_freq['money_growth'] + df_freq['velocity_growth'] - df_freq['real_gdp_growth']
        logger.info(f"✅ Using actual real GDP data for {freq_lower} QTM calculation")
    else:
        # Use assumed trend growth
        trend_growth = 0.02  # 2% annual real GDP trend
        df_freq['qtm_inflation'] = df_freq['money_growth'] + df_freq['velocity_growth'] - trend_growth
        logger.info(f"✅ Using assumed real GDP trend ({trend_growth:.1%}) for {freq_lower} QTM calculation")

    # Alternative: derive inflation from nominal GDP growth
    df_freq['inflation_from_nominal'] = df_freq['nominal_gdp_growth'] - (0.02 if real_gdp is None else df_freq['real_gdp_growth'])

    logger.info("✅ QTM inflation calculated.")
    logger.info(f"📅 Coverage: {df_freq.dropna().index.min().date()} → {df_freq.dropna().index.max().date()}")
    
    # Show recent decomposition
    recent = df_freq.dropna().tail(3)
    if not recent.empty:
        logger.info(f"📊 Recent {freq_lower} QTM decomposition:")
        for idx, row in recent.iterrows():
            real_growth = 0.02 if real_gdp is None else row.get('real_gdp_growth', 0.02)
            logger.info(f"   {idx.date()}: "
                       f"ΔM={row['money_growth']*100:+.1f}% + "
                       f"ΔV={row['velocity_growth']*100:+.1f}% - "
                       f"ΔY={real_growth*100:.1f}% = "
                       f"ΔP={row['qtm_inflation']*100:+.1f}%")

    return df_freq


def smooth_inflation(series: pd.Series, method: str = 'ma', window: int = 3) -> pd.Series:
    """
    Smooth noisy inflation series with moving average, exponential, or median filter.
    """
    if method == 'ma':
        return series.rolling(window, center=True).mean()
    elif method == 'ewm':
        return series.ewm(span=window).mean()
    elif method == 'median':
        return series.rolling(window, center=True).median()
    else:
        logger.warning(f"⚠️ Unknown smoothing method: {method}")
        return series


def calculate_breakeven_rates(nominal_rates: pd.Series, real_rates: pd.Series) -> pd.Series:
    """
    Breakeven inflation = Nominal rate - Real rate
    """
    df = pd.DataFrame({'nominal': nominal_rates, 'real': real_rates}).dropna()
    be = df['nominal'] - df['real']
    logger.info(f"✅ Calculated breakeven rates for {len(be)} periods")
    return be


def compare_inflation_measures(qtm_inflation: pd.Series,
                                market_inflation: Optional[pd.Series] = None,
                                official_inflation: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Compare QTM inflation to market breakevens and CPI/PCE
    """
    df = pd.DataFrame({'QTM': qtm_inflation})
    if market_inflation is not None:
        df['Market'] = market_inflation
    if official_inflation is not None:
        df['Official'] = official_inflation

    df = df.dropna()
    if len(df) > 1:
        logger.info("📊 Inflation comparison correlations:")
        corr = df.corr()
        for c1 in corr.columns:
            for c2 in corr.columns:
                if c1 != c2:
                    logger.info(f"   {c1} vs {c2}: {corr.loc[c1, c2]:.3f}")
    
        # Show recent values for comparison
        recent = df.tail(3)
        logger.info("📊 Recent inflation comparison:")
        for idx, row in recent.iterrows():
            values_str = " | ".join([f"{col}: {row[col]*100:+.1f}%" for col in df.columns])
            logger.info(f"   {idx.date()}: {values_str}")

    return df


def detect_inflation_regimes(series: pd.Series,
                              threshold_low: float = 0.02,
                              threshold_high: float = 0.04) -> pd.Series:
    """
    Classify inflation regimes: Deflation, Low, Moderate, High
    """
    regimes = pd.Series(index=series.index, dtype='object')
    regimes[series < 0] = 'Deflation'
    regimes[(series >= 0) & (series <= threshold_low)] = 'Low'
    regimes[(series > threshold_low) & (series <= threshold_high)] = 'Moderate'
    regimes[series > threshold_high] = 'High'

    counts = regimes.value_counts()
    total = len(regimes.dropna())
    if total > 0:
        logger.info("📊 Inflation regime breakdown:")
        for k, v in counts.items():
            logger.info(f"   {k}: {v} periods ({v/total:.1%})")

    return regimes


def validate_qtm_assumptions(df: pd.DataFrame) -> dict:
    """
    Validate key assumptions of QTM analysis and provide diagnostics.
    """
    validation = {}
    
    if 'velocity' in df.columns:
        velocity_stability = df['velocity'].std() / df['velocity'].mean()
        validation['velocity_cv'] = velocity_stability
        validation['velocity_stable'] = velocity_stability < 0.3  # Less than 30% coefficient of variation
        
    if 'money_growth_yoy' in df.columns:
        avg_money_growth = df['money_growth_yoy'].mean()
        validation['avg_money_growth'] = avg_money_growth
        validation['money_growth_reasonable'] = 0.02 <= avg_money_growth <= 0.15  # 2-15% seems reasonable
        
    if 'qtm_inflation_yoy' in df.columns and 'money_growth_yoy' in df.columns:
        correlation = df[['qtm_inflation_yoy', 'money_growth_yoy']].corr().iloc[0, 1]
        validation['inflation_money_correlation'] = correlation
        validation['correlation_positive'] = correlation > 0.3
    
    logger.info("📊 QTM Validation Results:")
    for key, value in validation.items():
        if isinstance(value, bool):
            status = "✅" if value else "❌"
            logger.info(f"   {key}: {status}")
        else:
            logger.info(f"   {key}: {value:.3f}")
    
    return validation


def validate_frequency_consistency(df: pd.DataFrame, frequency: str) -> dict:
    """
    Validate that calculations are consistent with the target frequency.
    """
    validation = {}
    
    # Check index frequency
    if len(df) > 1:
        time_diff = df.index[1] - df.index[0]
        
        if frequency == 'monthly':
            expected_days = 28
        elif frequency == 'quarterly':
            expected_days = 85  # ~3 months
        else:  # annual
            expected_days = 350  # ~1 year
            
        actual_days = time_diff.days
        validation['frequency_consistent'] = abs(actual_days - expected_days) < 35
        validation['actual_period_days'] = actual_days
        validation['expected_period_days'] = expected_days
    
    # Check velocity level (should be reasonable regardless of frequency)
    if 'velocity' in df.columns:
        mean_velocity = df['velocity'].mean()
        validation['velocity_reasonable'] = 0.5 <= mean_velocity <= 10
        validation['mean_velocity'] = mean_velocity
    
    # Check inflation components are not extreme
    if 'qtm_inflation_yoy' in df.columns:
        mean_inflation = df['qtm_inflation_yoy'].mean()
        validation['inflation_reasonable'] = -0.1 <= mean_inflation <= 0.2  # -10% to +20%
        validation['mean_qtm_inflation'] = mean_inflation
    
    logger.info(f"📊 {frequency.title()} Frequency Validation:")
    for key, value in validation.items():
        if isinstance(value, bool):
            status = "✅" if value else "❌"
            logger.info(f"   {key}: {status}")
        elif isinstance(value, (int, float)):
            logger.info(f"   {key}: {value:.3f}")
    
    return validation