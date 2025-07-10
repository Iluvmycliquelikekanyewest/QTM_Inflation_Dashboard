"""
Weight management module for GDP component shares.
Handles quarterly GDP shares from BEA and converts them to monthly weights.
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prev_qtr_shares(bea_shares: pd.DataFrame, lag_quarters: int = 1) -> pd.DataFrame:
    """
    Get previous quarter GDP component shares for use as monthly weights.
    
    This implements the common practice of using lagged shares to avoid 
    look-ahead bias when constructing real-time GDP proxies.
    
    Args:
        bea_shares: DataFrame with quarterly GDP component shares (share_C, share_I, etc.)
        lag_quarters: Number of quarters to lag the shares (default: 1)
        
    Returns:
        DataFrame with lagged quarterly shares
    """
    if bea_shares.empty:
        logger.warning("Empty BEA shares DataFrame provided")
        return bea_shares
    
    # Ensure the index is datetime
    if not isinstance(bea_shares.index, pd.DatetimeIndex):
        try:
            bea_shares.index = pd.to_datetime(bea_shares.index)
        except Exception as e:
            logger.error(f"Could not convert BEA shares index to datetime: {e}")
            raise ValueError("BEA shares must have a datetime-like index")
    
    # Lag the shares by specified quarters
    lagged_shares = bea_shares.shift(lag_quarters)
    
    # Log information about the lagging
    original_start = bea_shares.index.min()
    original_end = bea_shares.index.max()
    lagged_start = lagged_shares.dropna().index.min()
    lagged_end = lagged_shares.dropna().index.max()
    
    logger.info(f"📅 Applied {lag_quarters}Q lag to GDP shares:")
    logger.info(f"   Original range: {original_start} to {original_end}")
    logger.info(f"   Lagged range: {lagged_start} to {lagged_end}")
    logger.info(f"   Lost {lag_quarters} quarters at start due to lagging")
    
    # Check for share completeness
    share_cols = [col for col in lagged_shares.columns if col.startswith('share_')]
    for col in share_cols:
        valid_count = lagged_shares[col].count()
        total_count = len(lagged_shares)
        logger.info(f"   {col}: {valid_count}/{total_count} valid observations")
    
    return lagged_shares


def shares_monthly(quarterly_shares: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Convert quarterly GDP shares to monthly frequency.
    
    Takes quarterly shares and creates monthly weights by forward-filling
    or interpolating the quarterly values.
    
    Args:
        quarterly_shares: DataFrame with quarterly GDP component shares
        method: Method for monthly conversion ('ffill', 'interpolate', 'repeat')
        
    Returns:
        DataFrame with monthly GDP component shares
    """
    if quarterly_shares.empty:
        logger.warning("Empty quarterly shares DataFrame provided")
        return quarterly_shares
    
    # Ensure datetime index
    if not isinstance(quarterly_shares.index, pd.DatetimeIndex):
        try:
            quarterly_shares.index = pd.to_datetime(quarterly_shares.index)
        except Exception as e:
            logger.error(f"Could not convert quarterly shares index to datetime: {e}")
            raise ValueError("Quarterly shares must have a datetime-like index")
    
    # Create monthly date range
    start_date = quarterly_shares.index.min()
    end_date = quarterly_shares.index.max()
    
    # Extend to cover full months
    monthly_range = pd.date_range(
        start=start_date.replace(day=1),  # Start of month
        end=end_date + pd.offsets.MonthEnd(0),  # End of month
        freq='MS'  # Month start
    )
    
    if method == 'ffill':
        # Forward fill quarterly values through each quarter
        monthly_shares = quarterly_shares.reindex(monthly_range, method='ffill')
        logger.info(f"📅 Forward-filled quarterly shares to monthly ({len(monthly_shares)} months)")
        
    elif method == 'interpolate':
        # Linear interpolation between quarterly values
        monthly_shares = quarterly_shares.reindex(monthly_range).interpolate(method='linear')
        logger.info(f"📅 Interpolated quarterly shares to monthly ({len(monthly_shares)} months)")
        
    elif method == 'repeat':
        # Repeat each quarterly value for 3 months
        monthly_data = {}
        for date, row in quarterly_shares.iterrows():
            # Get the 3 months in this quarter
            quarter_months = pd.date_range(
                start=date.replace(day=1),
                periods=3,
                freq='MS'
            )
            for month in quarter_months:
                if month in monthly_range:
                    monthly_data[month] = row
        
        monthly_shares = pd.DataFrame.from_dict(monthly_data, orient='index')
        monthly_shares = monthly_shares.reindex(monthly_range)
        logger.info(f"📅 Repeated quarterly shares to monthly ({len(monthly_shares)} months)")
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'ffill', 'interpolate', or 'repeat'")
    
    # Validate share properties
    share_cols = [col for col in monthly_shares.columns if col.startswith('share_')]
    
    for col in share_cols:
        valid_data = monthly_shares[col].dropna()
        if len(valid_data) > 0:
            min_share = valid_data.min()
            max_share = valid_data.max()
            avg_share = valid_data.mean()
            
            # Check for reasonable share values (should be between -1 and 2 typically)
            if min_share < -1 or max_share > 2:
                logger.warning(f"⚠️  {col} has unusual values: min={min_share:.3f}, max={max_share:.3f}")
            
            logger.info(f"📊 {col}: avg={avg_share:.3f}, range=[{min_share:.3f}, {max_share:.3f}]")
    
    return monthly_shares


def validate_shares(shares_df: pd.DataFrame, tolerance: float = 0.02) -> bool:
    """
    Validate that GDP component shares are reasonable.
    
    Checks that shares sum approximately to 1.0 and are within reasonable bounds.
    
    Args:
        shares_df: DataFrame with GDP component shares
        tolerance: Allowed deviation from sum=1.0
        
    Returns:
        True if shares pass validation
    """
    if shares_df.empty:
        logger.warning("Cannot validate empty shares DataFrame")
        return False
    
    share_cols = [col for col in shares_df.columns if col.startswith('share_')]
    
    if not share_cols:
        logger.warning("No share columns found for validation")
        return False
    
    # Check if shares sum to approximately 1.0
    shares_sum = shares_df[share_cols].sum(axis=1)
    valid_sums = shares_sum.dropna()
    
    if len(valid_sums) == 0:
        logger.warning("No valid share sums found")
        return False
    
    # Calculate deviation from 1.0
    deviations = (valid_sums - 1.0).abs()
    max_deviation = deviations.max()
    avg_deviation = deviations.mean()
    
    # Count observations outside tolerance
    outside_tolerance = (deviations > tolerance).sum()
    pct_outside = outside_tolerance / len(valid_sums) * 100
    
    logger.info(f"📋 Share validation results:")
    logger.info(f"   Average deviation from 1.0: {avg_deviation:.4f}")
    logger.info(f"   Maximum deviation: {max_deviation:.4f}")
    logger.info(f"   Observations outside ±{tolerance}: {outside_tolerance} ({pct_outside:.1f}%)")
    
    # Validation passes if most observations are within tolerance
    validation_passed = pct_outside < 10.0  # Allow up to 10% outside tolerance
    
    if validation_passed:
        logger.info("✅ Share validation passed")
    else:
        logger.warning("⚠️  Share validation failed - shares may not sum to 1.0")
    
    # Additional checks for individual share reasonableness
    for col in share_cols:
        valid_data = shares_df[col].dropna()
        if len(valid_data) > 0:
            # Check for extreme values
            extreme_low = (valid_data < -0.5).sum()
            extreme_high = (valid_data > 1.5).sum()
            
            if extreme_low > 0 or extreme_high > 0:
                logger.warning(f"⚠️  {col} has {extreme_low + extreme_high} extreme values")
    
    return validation_passed


def calculate_implied_weights(components_df: pd.DataFrame, gdp_series: pd.Series) -> pd.DataFrame:
    """
    Calculate implied GDP component weights from actual data.
    
    Useful for validation or when BEA shares are not available.
    
    Args:
        components_df: DataFrame with GDP components (C, I, G, NX)
        gdp_series: Actual GDP series for calculating shares
        
    Returns:
        DataFrame with implied component shares
    """
    if components_df.empty or gdp_series.empty:
        logger.warning("Cannot calculate implied weights with empty data")
        return pd.DataFrame()
    
    # Align data
    aligned = components_df.join(gdp_series.rename('GDP'), how='inner')
    
    if 'GDP' not in aligned.columns or aligned['GDP'].isna().all():
        logger.error("No valid GDP data for calculating implied weights")
        return pd.DataFrame()
    
    # Calculate shares
    implied_shares = pd.DataFrame(index=aligned.index)
    
    component_map = {'C': 'share_C', 'I': 'share_I', 'G': 'share_G', 'NX': 'share_NX'}
    
    for comp_col, share_col in component_map.items():
        if comp_col in aligned.columns:
            implied_shares[share_col] = aligned[comp_col] / aligned['GDP']
    
    # Remove infinite and extreme values
    implied_shares = implied_shares.replace([np.inf, -np.inf], np.nan)
    
    # Log results
    logger.info(f"📊 Calculated implied weights for {len(implied_shares.columns)} components")
    logger.info(f"📅 Coverage: {implied_shares.dropna().index.min()} to {implied_shares.dropna().index.max()}")
    
    # Validate the implied shares
    validate_shares(implied_shares)
    
    return implied_shares


def smooth_shares(shares_df: pd.DataFrame, window: int = 4, method: str = 'rolling') -> pd.DataFrame:
    """
    Smooth GDP component shares to reduce quarter-to-quarter volatility.
    
    Args:
        shares_df: DataFrame with GDP component shares
        window: Smoothing window (number of quarters)
        method: Smoothing method ('rolling', 'ewm')
        
    Returns:
        DataFrame with smoothed shares
    """
    if shares_df.empty:
        return shares_df
    
    share_cols = [col for col in shares_df.columns if col.startswith('share_')]
    
    if not share_cols:
        logger.warning("No share columns found for smoothing")
        return shares_df
    
    smoothed = shares_df.copy()
    
    for col in share_cols:
        if method == 'rolling':
            smoothed[col] = shares_df[col].rolling(window=window, center=True).mean()
        elif method == 'ewm':
            smoothed[col] = shares_df[col].ewm(span=window).mean()
        else:
            logger.warning(f"Unknown smoothing method: {method}")
            continue
    
    logger.info(f"📈 Applied {method} smoothing to shares (window={window})")
    
    return smoothed


def get_latest_shares(shares_df: pd.DataFrame, as_of_date: Optional[str] = None) -> pd.Series:
    """
    Get the most recent GDP component shares as of a specific date.
    
    Useful for real-time nowcasting applications.
    
    Args:
        shares_df: DataFrame with GDP component shares
        as_of_date: Date to get shares as of (default: most recent)
        
    Returns:
        Series with latest available shares
    """
    if shares_df.empty:
        logger.warning("Cannot get latest shares from empty DataFrame")
        return pd.Series()
    
    if as_of_date is not None:
        as_of_date = pd.to_datetime(as_of_date)
        # Get shares as of the specified date
        available_dates = shares_df.index[shares_df.index <= as_of_date]
        if len(available_dates) == 0:
            logger.warning(f"No shares available as of {as_of_date}")
            return pd.Series()
        latest_date = available_dates.max()
    else:
        # Get most recent available shares
        latest_date = shares_df.dropna(how='all').index.max()
    
    latest_shares = shares_df.loc[latest_date].dropna()
    
    logger.info(f"📅 Latest shares as of {latest_date}:")
    for col, value in latest_shares.items():
        if col.startswith('share_'):
            logger.info(f"   {col}: {value:.3f}")
    
    return latest_shares