"""
Velocity and monetary inflation calculation module.
Updated for new GDP proxy format (monthly, nominal, billions).
Implements Quantity Theory of Money: MV = PY → V = PY / M.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calc_velocity(gdp_proxy: pd.Series, money_supply: pd.Series) -> pd.DataFrame:
    """
    Calculate velocity of money: V = GDP_proxy_annual / M2
    
    Assumes GDP proxy is monthly nominal in billions (not annualized).
    
    Returns:
        DataFrame with velocity, MoM/YoY growth, and implied inflation.
    """
    if gdp_proxy.empty or money_supply.empty:
        logger.warning("⚠️ Cannot calculate velocity with empty input series.")
        return pd.DataFrame()

    # Align and clean
    df = pd.DataFrame({'GDP_proxy': gdp_proxy, 'M2': money_supply}).dropna()
    if df.empty or len(df) < 2:
        logger.warning("⚠️ Not enough data to compute velocity.")
        return df

    # Sanity check
    if df['GDP_proxy'].mean() > 10000:
        logger.warning("GDP proxy seems already annualized — skipping 12x.")
        df['GDP_proxy_annual'] = df['GDP_proxy']
    else:
        df['GDP_proxy_annual'] = df['GDP_proxy'] * 12

    df['velocity'] = df['GDP_proxy_annual'] / df['M2']

    # Growth rates
    df['velocity_mom'] = df['velocity'].pct_change()
    df['velocity_yoy'] = df['velocity'].pct_change(periods=12)

    # Implied inflation via QTM (ΔP ≈ ΔV)
    df['inflation_mom'] = df['velocity_mom']
    df['inflation_yoy'] = df['velocity_yoy']
    df['inflation_mom_annual'] = (1 + df['inflation_mom']) ** 12 - 1

    logger.info("✅ Velocity calculation complete.")
    logger.info(f"📅 Coverage: {df.index.min().date()} → {df.index.max().date()}")
    logger.info(f"📊 Mean velocity: {df['velocity'].mean():.2f}")

    extreme_vals = df[(df['velocity'] < 0.1) | (df['velocity'] > 20)]
    if not extreme_vals.empty:
        logger.warning(f"⚠️ {len(extreme_vals)} periods with extreme velocity values.")

    return df


def calculate_quantity_theory_inflation(gdp_proxy: pd.Series, money_supply: pd.Series,
                                        real_gdp: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Estimate inflation using QTM: MV = PY → ΔP ≈ ΔM + ΔV - ΔY
    
    If real_gdp not provided, assumes constant trend growth (2%).
    """
    df = pd.DataFrame({'nominal_gdp': gdp_proxy, 'money_supply': money_supply}).dropna()
    if len(df) < 13:
        logger.warning("⚠️ Need at least 13 months for YoY QTM inflation.")
        return df

    # YoY growth
    df['money_growth'] = df['money_supply'].pct_change(12)
    df['nominal_gdp_growth'] = df['nominal_gdp'].pct_change(12)

    if real_gdp is not None:
        df['real_gdp'] = real_gdp
        df['real_gdp_growth'] = real_gdp.pct_change(12)
        df['velocity_growth'] = df['nominal_gdp_growth'] - df['money_growth']
        df['qtm_inflation'] = df['money_growth'] + df['velocity_growth'] - df['real_gdp_growth']
    else:
        df['velocity'] = (df['nominal_gdp'] * 12) / df['money_supply']
        df['velocity_growth'] = df['velocity'].pct_change(12)
        trend_growth = 0.02  # Assumed real GDP trend
        df['qtm_inflation'] = df['money_growth'] + df['velocity_growth'] - trend_growth

    logger.info("✅ QTM inflation calculated.")
    logger.info(f"📅 Coverage: {df.dropna().index.min().date()} → {df.dropna().index.max().date()}")

    return df


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
    logger.info("📊 Inflation comparison correlations:")
    corr = df.corr()
    for c1 in corr.columns:
        for c2 in corr.columns:
            if c1 != c2:
                logger.info(f"   {c1} vs {c2}: {corr.loc[c1, c2]:.3f}")

    return df


def detect_inflation_regimes(series: pd.Series,
                              threshold_low: float = 0.02,
                              threshold_high: float = 0.04) -> pd.Series:
    """
    Classify inflation regimes: Deflation, Low, Moderate, High
    """
    regimes = pd.Series(index=series.index, dtype='object')
    regimes[series < 0] = 'Deflation'
    regimes[series <= threshold_low] = 'Low'
    regimes[(series > threshold_low) & (series <= threshold_high)] = 'Moderate'
    regimes[series > threshold_high] = 'High'

    counts = regimes.value_counts()
    total = len(regimes.dropna())
    logger.info("📊 Inflation regime breakdown:")
    for k, v in counts.items():
        logger.info(f"   {k}: {v} periods ({v/total:.1%})")

    return regimes
