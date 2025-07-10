"""
pipeline.py
-----------
Master pipeline orchestrating the entire Monetary Inflation Dashboard.
Combines all modules: data fetch → weights → GDP proxy → velocity → results.
"""

import pandas as pd
import logging
from typing import Optional, Dict, Any, Tuple

from .data_fetch import get_fred, get_bea_gdp_shares
from .weight_manager import get_monthly_weights
from .gdp_proxy import build_gdp_proxy_pipeline
from .velocity import monetary_analysis_pipeline
from .config import Config

logger = logging.getLogger(__name__)

def run_pipeline(start_date: str = '1990-01-01', 
                end_date: Optional[str] = None,
                lag_quarters: int = 1) -> pd.DataFrame:
    """
    Run the complete monetary inflation dashboard pipeline.
    
    This is the main function that orchestrates all data processing steps:
    1. Fetch FRED economic data
    2. Fetch BEA GDP component shares  
    3. Create monthly weights with Q-1 lag
    4. Build GDP proxy from components
    5. Calculate velocity and monetary inflation
    6. Return tidy results DataFrame
    
    Args:
        start_date: Start date for analysis (YYYY-MM-DD)
        end_date: End date for analysis (optional, defaults to latest available)
        lag_quarters: Number of quarters to lag GDP shares (default 1)
        
    Returns:
        pd.DataFrame: Tidy DataFrame with all results ready for visualization
    """
    logger.info("=" * 60)
    logger.info("Starting Monetary Inflation Dashboard Pipeline")
    logger.info("=" * 60)
    
    try:
        # Step 1: Fetch FRED data
        logger.info("Step 1: Fetching FRED economic data...")
        fred_series = Config.FRED_SERIES.copy()
        fred_data = get_fred(fred_series, start_date=start_date)
        logger.info(f"✓ FRED data retrieved: {len(fred_data)} periods, {len(fred_data.columns)} series")
        
        # Step 2: Fetch BEA GDP shares
        logger.info("Step 2: Fetching BEA GDP component shares...")
        bea_shares = get_bea_gdp_shares()
        logger.info(f"✓ BEA shares retrieved: {len(bea_shares)} quarters, {len(bea_shares.columns)} components")
        
        # Step 3: Create monthly weights
        logger.info(f"Step 3: Creating monthly weights with Q-{lag_quarters} lag...")
        monthly_weights = get_monthly_weights(
            bea_shares, 
            start_date=start_date,
            end_date=end_date,
            lag_quarters=lag_quarters
        )
        logger.info(f"✓ Monthly weights created: {len(monthly_weights)} periods")
        
        # Step 4: Build GDP proxy
        logger.info("Step 4: Building GDP proxy from components...")
        gdp_proxy = build_gdp_proxy_pipeline(
            fred_data, 
            monthly_weights,
            start_date=start_date,
            end_date=end_date
        )
        logger.info(f"✓ GDP proxy constructed: {len(gdp_proxy)} periods")
        
        # Step 5: Calculate velocity and monetary inflation
        logger.info("Step 5: Calculating velocity and monetary inflation...")
        velocity, inflation_df, validation = monetary_analysis_pipeline(
            gdp_proxy, 
            fred_data,
            money_series='M2'
        )
        logger.info(f"✓ Monetary analysis complete: {len(velocity)} velocity periods")
        
        # Step 6: Combine results into tidy DataFrame
        logger.info("Step 6: Assembling final results...")
        results_df = assemble_results(
            gdp_proxy=gdp_proxy,
            velocity=velocity,
            inflation_df=inflation_df,
            fred_data=fred_data,
            monthly_weights=monthly_weights,
            start_date=start_date,
            end_date=end_date
        )
        
        # Log validation warnings if any
        if validation.get('warnings'):
            logger.warning(f"Validation warnings: {validation['warnings']}")
        
        logger.info("=" * 60)
        logger.info(f"Pipeline completed successfully!")
        logger.info(f"Final dataset: {len(results_df)} periods × {len(results_df.columns)} columns")
        logger.info("=" * 60)
        
        return results_df
        
    except Exception as e:
        logger.error(f"Pipeline failed at step: {e}")
        raise

def assemble_results(gdp_proxy: pd.Series,
                    velocity: pd.Series, 
                    inflation_df: pd.DataFrame,
                    fred_data: pd.DataFrame,
                    monthly_weights: pd.DataFrame,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Combine all pipeline outputs into a single tidy DataFrame.
    
    Args:
        gdp_proxy: GDP proxy series
        velocity: Velocity series
        inflation_df: Inflation DataFrame with MoM and YoY rates
        fred_data: Original FRED data
        monthly_weights: Monthly GDP component weights
        start_date: Optional start date for trimming
        end_date: Optional end date for trimming
        
    Returns:
        pd.DataFrame: Tidy DataFrame with all series aligned
    """
    logger.info("Assembling results into tidy DataFrame...")
    
    # Start with inflation DataFrame (has velocity + inflation rates)
    results = inflation_df.copy()
    
    # Add GDP proxy
    results['gdp_proxy'] = gdp_proxy
    
    # Add money supply (M2) from FRED data
    if 'M2' in fred_data.columns:
        results['money_supply'] = fred_data['M2']
    
    # Add key FRED series for reference
    key_series = ['PCE', 'GPDI', 'GCE', 'NETEXP']
    for series in key_series:
        if series in fred_data.columns:
            results[f'fred_{series.lower()}'] = fred_data[series]
    
    # Add GDP component weights for transparency
    weight_cols = ['share_C', 'share_I', 'share_G', 'share_NX']
    for col in weight_cols:
        if col in monthly_weights.columns:
            results[f'weight_{col.split("_")[1].lower()}'] = monthly_weights[col]
    
    # Calculate additional derived metrics
    results['gdp_growth_mom'] = results['gdp_proxy'].pct_change(1) * 12  # Annualized MoM
    results['gdp_growth_yoy'] = results['gdp_proxy'].pct_change(12)     # YoY
    results['money_growth_mom'] = results['money_supply'].pct_change(1) * 12  # Annualized MoM
    results['money_growth_yoy'] = results['money_supply'].pct_change(12)      # YoY
    
    # Trim to requested date range
    if start_date:
        start = pd.to_datetime(start_date)
        results = results.loc[results.index >= start]
    
    if end_date:
        end = pd.to_datetime(end_date)
        results = results.loc[results.index <= end]
    
    # Sort by date and drop any remaining NaN rows
    results = results.sort_index()
    initial_length = len(results)
    results = results.dropna(subset=['velocity', 'gdp_proxy'], how='all')
    dropped = initial_length - len(results)
    
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with missing core data")
    
    # Reorder columns for better readability
    core_cols = [
        'gdp_proxy', 'money_supply', 'velocity',
        'monetary_inflation_mom', 'monetary_inflation_yoy',
        'gdp_growth_mom', 'gdp_growth_yoy',
        'money_growth_mom', 'money_growth_yoy'
    ]
    
    # Add remaining columns
    other_cols = [col for col in results.columns if col not in core_cols]
    final_cols = [col for col in core_cols if col in results.columns] + other_cols
    
    results = results[final_cols]
    
    logger.info(f"Results assembled: {len(results)} periods × {len(results.columns)} columns")
    
    return results

def validate_pipeline_inputs(start_date: str, 
                           end_date: Optional[str] = None,
                           lag_quarters: int = 1) -> Dict[str, Any]:
    """
    Validate pipeline inputs before execution.
    
    Args:
        start_date: Start date string
        end_date: End date string (optional)
        lag_quarters: Number of lag quarters
        
    Returns:
        dict: Validation results
    """
    validation = {'is_valid': True, 'warnings': [], 'info': []}
    
    # Validate dates
    try:
        start = pd.to_datetime(start_date)
        validation['info'].append(f"Start date: {start.strftime('%Y-%m-%d')}")
        
        if start < pd.to_datetime('1990-01-01'):
            validation['warnings'].append("Start date before 1990 may have limited data availability")
        
        if end_date:
            end = pd.to_datetime(end_date)
            validation['info'].append(f"End date: {end.strftime('%Y-%m-%d')}")
            
            if end <= start:
                validation['is_valid'] = False
                validation['warnings'].append("End date must be after start date")
                
            if (end - start).days < 365:
                validation['warnings'].append("Date range < 1 year may limit analysis quality")
        
    except Exception as e:
        validation['is_valid'] = False
        validation['warnings'].append(f"Invalid date format: {e}")
    
    # Validate lag quarters
    if not isinstance(lag_quarters, int) or lag_quarters < 0:
        validation['is_valid'] = False
        validation['warnings'].append("lag_quarters must be a non-negative integer")
    elif lag_quarters > 4:
        validation['warnings'].append("lag_quarters > 4 may remove too much recent data")
    
    # Check API configuration
    try:
        Config.validate_keys()
        validation['info'].append("API keys validated successfully")
    except Exception as e:
        validation['is_valid'] = False
        validation['warnings'].append(f"API configuration error: {e}")
    
    return validation

def run_pipeline_with_validation(start_date: str = '1990-01-01',
                                end_date: Optional[str] = None, 
                                lag_quarters: int = 1) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run pipeline with input validation.
    
    Args:
        start_date: Start date for analysis
        end_date: End date for analysis (optional)
        lag_quarters: Number of quarters to lag GDP shares
        
    Returns:
        tuple: (results_dataframe, validation_dict)
    """
    # Validate inputs
    validation = validate_pipeline_inputs(start_date, end_date, lag_quarters)
    
    if not validation['is_valid']:
        logger.error(f"Pipeline validation failed: {validation['warnings']}")
        raise ValueError(f"Invalid inputs: {validation['warnings']}")
    
    if validation['warnings']:
        logger.warning(f"Pipeline validation warnings: {validation['warnings']}")
    
    # Run pipeline
    results_df = run_pipeline(start_date, end_date, lag_quarters)
    
    return results_df, validation

# Convenience function for Streamlit app
def get_pipeline_summary(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for dashboard display.
    
    Args:
        results_df: Results DataFrame from pipeline
        
    Returns:
        dict: Summary statistics
    """
    if results_df.empty:
        return {'error': 'No data available'}
    
    summary = {
        'data_range': {
            'start': results_df.index.min().strftime('%Y-%m-%d'),
            'end': results_df.index.max().strftime('%Y-%m-%d'),
            'periods': len(results_df)
        },
        'velocity': {
            'current': results_df['velocity'].iloc[-1] if 'velocity' in results_df.columns else None,
            'mean': results_df['velocity'].mean() if 'velocity' in results_df.columns else None,
            'std': results_df['velocity'].std() if 'velocity' in results_df.columns else None
        },
        'inflation': {
            'current_mom': results_df['monetary_inflation_mom'].iloc[-1] * 100 if 'monetary_inflation_mom' in results_df.columns else None,
            'current_yoy': results_df['monetary_inflation_yoy'].iloc[-1] * 100 if 'monetary_inflation_yoy' in results_df.columns else None,
            'mean_mom': results_df['monetary_inflation_mom'].mean() * 100 if 'monetary_inflation_mom' in results_df.columns else None,
            'mean_yoy': results_df['monetary_inflation_yoy'].mean() * 100 if 'monetary_inflation_yoy' in results_df.columns else None
        }
    }
    
    return summary