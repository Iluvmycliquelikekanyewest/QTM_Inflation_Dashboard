"""
Data transformation utilities for the Monetary Inflation Dashboard.
Handles unit conversions and temporal interpolation.
"""
import pandas as pd
import numpy as np
import logging
from typing import Union, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def saar_to_monthly(series: pd.Series) -> pd.Series:
    """
    Convert Seasonally Adjusted Annual Rate (SAAR) to monthly rate.
    
    SAAR represents the annualized rate, so we divide by 12 to get 
    the monthly equivalent.
    
    Args:
        series: Pandas Series with SAAR data
        
    Returns:
        Series converted to monthly rate
    """
    if series.empty:
        logger.warning("Empty series passed to saar_to_monthly")
        return series
    
    monthly = series / 12
    logger.debug(f"Converted SAAR to monthly: {series.name}")
    return monthly


def quarterly_to_monthly_rate(series: pd.Series) -> pd.Series:
    """
    Convert quarterly actual values to monthly rate equivalent.
    
    Divides quarterly values by 3 to get monthly equivalent.
    
    Args:
        series: Pandas Series with quarterly data
        
    Returns:
        Series converted to monthly rate
    """
    if series.empty:
        logger.warning("Empty series passed to quarterly_to_monthly_rate")
        return series
    
    monthly = series / 3
    logger.debug(f"Converted quarterly to monthly rate: {series.name}")
    return monthly


def interp_quarterly(quarterly_series: pd.Series, method: str = 'linear') -> pd.Series:
    """
    Interpolate quarterly data to monthly frequency.
    
    Takes quarterly data and interpolates to fill in monthly values.
    Uses pandas interpolation with specified method.
    
    Args:
        quarterly_series: Series with quarterly data (index should be quarterly dates)
        method: Interpolation method ('linear', 'cubic', 'quadratic', etc.)
        
    Returns:
        Series with monthly frequency
    """
    if quarterly_series.empty:
        logger.warning("Empty series passed to interp_quarterly")
        return quarterly_series
    
    # Ensure we have a proper date index
    if not isinstance(quarterly_series.index, pd.DatetimeIndex):
        try:
            quarterly_series.index = pd.to_datetime(quarterly_series.index)
        except Exception as e:
            logger.error(f"Could not convert index to datetime: {e}")
            raise ValueError("Series must have a datetime-like index for interpolation")
    
    # Create monthly date range covering the quarterly data
    start_date = quarterly_series.index.min()
    end_date = quarterly_series.index.max()
    
    # Extend end date to end of quarter to ensure complete coverage
    if end_date.month % 3 != 0:
        # Not end of quarter, extend to quarter end
        quarter_end_month = ((end_date.month - 1) // 3 + 1) * 3
        if quarter_end_month > 12:
            end_date = end_date.replace(year=end_date.year + 1, month=3)
        else:
            end_date = end_date.replace(month=quarter_end_month)
    
    monthly_index = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    # Reindex to monthly frequency (this will insert NaNs)
    monthly_series = quarterly_series.reindex(monthly_index)
    
    # Interpolate missing values
    interpolated = monthly_series.interpolate(method=method)
    
    logger.debug(f"Interpolated quarterly to monthly: {len(quarterly_series)} → {len(interpolated)} observations")
    
    return interpolated


def align_frequencies(series_dict: dict, target_freq: str = 'M') -> pd.DataFrame:
    """
    Align multiple series to the same frequency.
    
    Args:
        series_dict: Dictionary of {name: series} to align
        target_freq: Target frequency ('M' for monthly, 'Q' for quarterly)
        
    Returns:
        DataFrame with all series aligned to target frequency
    """
    if not series_dict:
        raise ValueError("Empty series_dict provided")
    
    aligned_series = {}
    
    for name, series in series_dict.items():
        if series.empty:
            logger.warning(f"Skipping empty series: {name}")
            continue
            
        # Ensure datetime index
        if not isinstance(series.index, pd.DatetimeIndex):
            try:
                series.index = pd.to_datetime(series.index)
            except Exception:
                logger.error(f"Could not convert {name} index to datetime")
                continue
        
        # Determine current frequency
        current_freq = pd.infer_freq(series.index)
        
        if target_freq == 'M':
            if current_freq and 'Q' in current_freq:
                # Quarterly to monthly
                aligned_series[name] = interp_quarterly(series)
            elif current_freq and 'M' in current_freq:
                # Already monthly
                aligned_series[name] = series
            else:
                # Try to resample/interpolate
                monthly_idx = pd.date_range(
                    start=series.index.min(),
                    end=series.index.max(),
                    freq='MS'
                )
                aligned_series[name] = series.reindex(monthly_idx).interpolate()
        
        elif target_freq == 'Q':
            if current_freq and 'M' in current_freq:
                # Monthly to quarterly (take quarter-end values)
                aligned_series[name] = series.resample('Q').last()
            elif current_freq and 'Q' in current_freq:
                # Already quarterly
                aligned_series[name] = series
            else:
                # Try to resample
                quarterly_idx = pd.date_range(
                    start=series.index.min(),
                    end=series.index.max(), 
                    freq='Q'
                )
                aligned_series[name] = series.reindex(quarterly_idx).interpolate()
    
    if not aligned_series:
        raise ValueError("No series could be successfully aligned")
    
    # Combine into DataFrame
    result = pd.DataFrame(aligned_series)
    logger.info(f"Aligned {len(aligned_series)} series to {target_freq} frequency: {len(result)} periods")
    
    return result


def smooth_series(series: pd.Series, window: int = 3, method: str = 'rolling') -> pd.Series:
    """
    Apply smoothing to a time series.
    
    Args:
        series: Input series to smooth
        window: Size of smoothing window
        method: Smoothing method ('rolling', 'ewm', 'savgol')
        
    Returns:
        Smoothed series
    """
    if series.empty or len(series) < window:
        logger.warning("Series too short for smoothing")
        return series
    
    if method == 'rolling':
        smoothed = series.rolling(window=window, center=True).mean()
    elif method == 'ewm':
        smoothed = series.ewm(span=window).mean()
    elif method == 'savgol':
        try:
            from scipy.signal import savgol_filter
            # Fill NaNs for savgol
            filled = series.interpolate()
            # Ensure window is odd and at least 3
            if window % 2 == 0:
                window += 1
            if window < 3:
                window = 3
            # Ensure polynomial order is less than window
            poly_order = min(2, window - 1)
            smoothed_values = savgol_filter(filled.values, window, poly_order)
            smoothed = pd.Series(smoothed_values, index=series.index, name=series.name)
        except (ImportError, ValueError) as e:
            logger.warning(f"SciPy savgol_filter not available or invalid parameters: {e}")
            logger.info("Falling back to rolling mean")
            smoothed = series.rolling(window=window, center=True).mean()
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    
    logger.debug(f"Applied {method} smoothing with window={window}")
    return smoothed


def calculate_growth_rates(series: pd.Series, periods: int = 1, method: str = 'pct_change') -> pd.Series:
    """
    Calculate growth rates for a time series.
    
    Args:
        series: Input series
        periods: Number of periods for growth calculation
        method: 'pct_change' for percent change, 'log' for log differences
        
    Returns:
        Series with growth rates
    """
    if series.empty:
        return series
    
    if method == 'pct_change':
        growth = series.pct_change(periods=periods)
    elif method == 'log':
        growth = np.log(series).diff(periods=periods)
    else:
        raise ValueError(f"Unknown growth method: {method}")
    
    logger.debug(f"Calculated {periods}-period growth rates using {method}")
    return growth


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Winsorize a series by capping extreme values.
    
    Args:
        series: Input series
        lower: Lower percentile threshold (0-1)
        upper: Upper percentile threshold (0-1)
        
    Returns:
        Winsorized series
    """
    if series.empty:
        return series
    
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    
    winsorized = series.clip(lower=lower_bound, upper=upper_bound)
    
    n_capped = ((series < lower_bound) | (series > upper_bound)).sum()
    if n_capped > 0:
        logger.debug(f"Winsorized {n_capped} extreme values ({n_capped/len(series)*100:.1f}%)")
    
    return winsorized


# Unit conversion constants
BILLION_TO_MILLION = 1000
MILLION_TO_THOUSAND = 1000
ANNUAL_TO_MONTHLY = 12
QUARTERLY_TO_MONTHLY = 3


def standardize_units(series: pd.Series, from_unit: str, to_unit: str) -> pd.Series:
    """
    Standardize units across different data series.
    
    Args:
        series: Input series
        from_unit: Current unit ('billion', 'million', 'thousand', 'annual', 'quarterly')
        to_unit: Target unit
        
    Returns:
        Series in target units
    """
    if series.empty:
        return series
    
    conversion_factors = {
        ('billion', 'million'): BILLION_TO_MILLION,
        ('million', 'thousand'): MILLION_TO_THOUSAND,
        ('annual', 'monthly'): ANNUAL_TO_MONTHLY,
        ('quarterly', 'monthly'): QUARTERLY_TO_MONTHLY,
        ('thousand', 'million'): 1 / MILLION_TO_THOUSAND,
        ('million', 'billion'): 1 / BILLION_TO_MILLION,
        ('monthly', 'quarterly'): 1 / QUARTERLY_TO_MONTHLY,
        ('monthly', 'annual'): 1 / ANNUAL_TO_MONTHLY,
    }
    
    factor = conversion_factors.get((from_unit, to_unit), 1.0)
    
    if factor == 1.0 and from_unit != to_unit:
        logger.warning(f"No conversion available from {from_unit} to {to_unit}")
    
    converted = series * factor if factor != 1.0 else series
    
    if factor != 1.0:
        logger.debug(f"Converted {from_unit} to {to_unit} (factor: {factor})")
    
    return converted