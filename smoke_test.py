"""
QTM Inflation Smoke Test with CPI Validation - 10 Year Analysis (ENHANCED VERSION)
Tests the complete pipeline: GDP proxy → Velocity → QTM Inflation vs Actual CPI
Covers Q1 2015 to Q1 2025 (full 10-year period) with model validation
ENHANCED: Now includes CPI comparison and model accuracy metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import your modules (adjust paths as needed)
from src.gdp_proxy import build_gdp_proxy
from src.data_fetch import DataFetcher, get_fred_gdp_components
from src.velocity import calc_velocity, calculate_quantity_theory_inflation, smooth_inflation

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def fetch_cpi_data(start_date="2015-01-01", end_date="2025-03-31"):
    """
    Fetch CPI data from FRED and calculate YoY inflation rates
    """
    try:
        print(f"💰 Fetching CPI data from FRED...")
        fetcher = DataFetcher()
        
        # Fetch multiple CPI measures for robustness
        cpi_series_map = {
            'CPI_ALL': 'CPIAUCSL',      # Consumer Price Index for All Urban Consumers: All Items
            'CPI_CORE': 'CPILFESL',     # Core CPI (excluding food and energy)
            'PCE': 'PCEPI',             # PCE Price Index (Fed's preferred measure)
            'PCE_CORE': 'PCEPILFE'      # Core PCE Price Index
        }
        
        cpi_data = fetcher.get_fred(cpi_series_map, start_date=start_date)
        
        if cpi_data.empty:
            print("❌ No CPI data available")
            return None
            
        # Filter to date range
        cpi_data = cpi_data[(cpi_data.index >= start_date) & (cpi_data.index <= end_date)]
        
        # Calculate YoY inflation rates
        inflation_data = pd.DataFrame(index=cpi_data.index)
        
        for series in cpi_data.columns:
            # Calculate 12-month percent change
            inflation_data[f'{series}_INFLATION'] = cpi_data[series].pct_change(periods=12) * 100
            
        # Drop first 12 months (no YoY calculation possible)
        inflation_data = inflation_data.iloc[12:].dropna(how='all')
        
        print(f"✅ CPI data fetched successfully")
        print(f"📅 CPI range: {inflation_data.index.min().date()} to {inflation_data.index.max().date()}")
        print(f"📊 Available series: {list(inflation_data.columns)}")
        
        return inflation_data
        
    except Exception as e:
        print(f"❌ CPI data fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_model_accuracy(qtm_inflation, actual_inflation, model_name="QTM", actual_name="CPI"):
    """
    Calculate comprehensive accuracy metrics between model and actual inflation
    """
    # Align the series by date
    aligned_data = pd.DataFrame({
        'model': qtm_inflation,
        'actual': actual_inflation
    }).dropna()
    
    if len(aligned_data) < 12:
        return None
        
    model_vals = aligned_data['model'].values
    actual_vals = aligned_data['actual'].values
    
    # Calculate accuracy metrics
    mae = np.mean(np.abs(model_vals - actual_vals))
    rmse = np.sqrt(np.mean((model_vals - actual_vals)**2))
    mape = np.mean(np.abs((model_vals - actual_vals) / actual_vals)) * 100
    
    # Correlation
    correlation = np.corrcoef(model_vals, actual_vals)[0, 1]
    
    # R-squared
    r_squared = stats.linregress(actual_vals, model_vals).rvalue ** 2
    
    # Directional accuracy (same sign)
    direction_correct = np.sum(np.sign(model_vals) == np.sign(actual_vals)) / len(model_vals) * 100
    
    # Bias (systematic over/under prediction)
    bias = np.mean(model_vals - actual_vals)
    
    # Tracking error (volatility of differences)
    tracking_error = np.std(model_vals - actual_vals)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'correlation': correlation,
        'r_squared': r_squared,
        'direction_accuracy': direction_correct,
        'bias': bias,
        'tracking_error': tracking_error,
        'observations': len(aligned_data),
        'model_name': model_name,
        'actual_name': actual_name
    }

def create_validation_comparison_table(qtm_inflation, cpi_data):
    """
    Create detailed monthly comparison table between QTM and CPI inflation
    """
    print(f"\n📊 MONTHLY QTM vs CPI COMPARISON TABLE")
    print("=" * 80)
    
    # Prepare comparison data
    comparison_data = []
    
    # Use CPI_ALL as primary comparison (most common measure)
    if 'CPI_ALL_INFLATION' in cpi_data.columns:
        cpi_series = cpi_data['CPI_ALL_INFLATION']
        cpi_name = 'CPI All Items'
    else:
        # Fallback to first available series
        cpi_series = cpi_data.iloc[:, 0]
        cpi_name = cpi_data.columns[0]
    
    # Align dates
    aligned_dates = qtm_inflation.index.intersection(cpi_series.index)
    
    for date in aligned_dates:
        qtm_val = qtm_inflation.loc[date] * 100  # Convert to percentage
        cpi_val = cpi_series.loc[date]
        difference = qtm_val - cpi_val
        
        comparison_data.append({
            'Date': date.strftime('%Y-%m'),
            'QTM_Inflation': qtm_val,
            'CPI_Inflation': cpi_val,
            'Difference': difference,
            'Abs_Diff': abs(difference),
            'Year': date.year,
            'Quarter': f"Q{date.quarter}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if not comparison_df.empty:
        # Show recent 24 months in detail
        print(f"📋 Recent 24 Months Detailed Comparison:")
        print(f"{'Date':<8} {'QTM (%)':<8} {'CPI (%)':<8} {'Diff':<8} {'|Diff|':<8}")
        print("-" * 50)
        
        recent_24 = comparison_df.tail(24)
        for _, row in recent_24.iterrows():
            print(f"{row['Date']:<8} {row['QTM_Inflation']:>7.1f} {row['CPI_Inflation']:>7.1f} "
                  f"{row['Difference']:>+7.1f} {row['Abs_Diff']:>7.1f}")
        
        # Annual summary statistics
        print(f"\n📈 Annual Accuracy Summary:")
        print(f"{'Year':<6} {'QTM Avg':<8} {'CPI Avg':<8} {'MAE':<6} {'Bias':<8} {'Months':<7}")
        print("-" * 50)
        
        annual_stats = comparison_df.groupby('Year').agg({
            'QTM_Inflation': 'mean',
            'CPI_Inflation': 'mean',
            'Abs_Diff': 'mean',
            'Difference': 'mean',
            'Date': 'count'
        }).round(1)
        
        for year, row in annual_stats.iterrows():
            print(f"{year:<6} {row['QTM_Inflation']:>7.1f} {row['CPI_Inflation']:>7.1f} "
                  f"{row['Abs_Diff']:>5.1f} {row['Difference']:>+7.1f} {row['Date']:>6}")
        
        # Period-based analysis
        print(f"\n🎯 Period-Based Model Performance:")
        
        # Pre-COVID (2015-2019)
        pre_covid = comparison_df[comparison_df['Year'] < 2020]
        if len(pre_covid) > 0:
            pre_mae = pre_covid['Abs_Diff'].mean()
            pre_bias = pre_covid['Difference'].mean()
            print(f"   Pre-COVID (2015-2019): MAE={pre_mae:.1f}%, Bias={pre_bias:+.1f}% ({len(pre_covid)} months)")
        
        # COVID Era (2020-2021)
        covid_era = comparison_df[(comparison_df['Year'] >= 2020) & (comparison_df['Year'] <= 2021)]
        if len(covid_era) > 0:
            covid_mae = covid_era['Abs_Diff'].mean()
            covid_bias = covid_era['Difference'].mean()
            print(f"   COVID Era (2020-2021): MAE={covid_mae:.1f}%, Bias={covid_bias:+.1f}% ({len(covid_era)} months)")
        
        # Post-COVID (2022+)
        post_covid = comparison_df[comparison_df['Year'] >= 2022]
        if len(post_covid) > 0:
            post_mae = post_covid['Abs_Diff'].mean()
            post_bias = post_covid['Difference'].mean()
            print(f"   Post-COVID (2022+): MAE={post_mae:.1f}%, Bias={post_bias:+.1f}% ({len(post_covid)} months)")
        
        # Accuracy quartiles
        print(f"\n📊 Accuracy Distribution:")
        quartiles = comparison_df['Abs_Diff'].quantile([0.25, 0.5, 0.75])
        print(f"   25th percentile: {quartiles[0.25]:.1f}% error")
        print(f"   Median error: {quartiles[0.5]:.1f}%")
        print(f"   75th percentile: {quartiles[0.75]:.1f}% error")
        print(f"   Max error: {comparison_df['Abs_Diff'].max():.1f}% ({comparison_df.loc[comparison_df['Abs_Diff'].idxmax(), 'Date']})")
        
        # Best and worst periods
        best_months = comparison_df.nsmallest(5, 'Abs_Diff')[['Date', 'QTM_Inflation', 'CPI_Inflation', 'Abs_Diff']]
        worst_months = comparison_df.nlargest(5, 'Abs_Diff')[['Date', 'QTM_Inflation', 'CPI_Inflation', 'Abs_Diff']]
        
        print(f"\n🏆 Best Model Performance (Top 5):")
        for _, row in best_months.iterrows():
            print(f"   {row['Date']}: QTM={row['QTM_Inflation']:.1f}%, CPI={row['CPI_Inflation']:.1f}%, Error={row['Abs_Diff']:.1f}%")
        
        print(f"\n⚠️ Worst Model Performance (Bottom 5):")
        for _, row in worst_months.iterrows():
            print(f"   {row['Date']}: QTM={row['QTM_Inflation']:.1f}%, CPI={row['CPI_Inflation']:.1f}%, Error={row['Abs_Diff']:.1f}%")
    
    return comparison_df

def qtm_inflation_validation_test():
    """
    Enhanced QTM inflation smoke test with CPI validation
    """
    print("🚀 QTM INFLATION VALIDATION TEST - 10 YEAR ANALYSIS (ENHANCED)")
    print("=" * 70)
    print("📅 Coverage: Q1 2015 → Q1 2025 (full 10-year period)")
    print("🎯 Testing: GDP Proxy → Velocity → QTM Inflation vs Actual CPI")
    print("🔧 VALIDATION: Model accuracy assessment against real inflation")
    
    # Define extended test period
    start_date = "2015-01-01"
    end_date = "2025-03-31"
    
    # Step 1: Build GDP Proxy (same as before)
    print(f"\n📊 STEP 1: Building GDP Proxy ({start_date} to {end_date})")
    print("-" * 50)
    
    try:
        gdp_proxy_df = build_gdp_proxy(start_date=start_date, export=False, strict=False, use_slce=True)
        
        if gdp_proxy_df.empty:
            print("❌ GDP proxy construction failed")
            return None
            
        gdp_proxy_df = gdp_proxy_df[(gdp_proxy_df.index >= start_date) & (gdp_proxy_df.index <= end_date)]
        
        print(f"✅ GDP Proxy built successfully")
        print(f"📅 Range: {gdp_proxy_df.index.min().date()} to {gdp_proxy_df.index.max().date()}")
        print(f"📈 Observations: {len(gdp_proxy_df)} months")
        
    except Exception as e:
        print(f"❌ GDP proxy construction failed: {e}")
        return None
    
    # Step 2: Fetch Money Supply (same as before)
    print(f"\n💰 STEP 2: Fetching Money Supply (M2) Data")
    print("-" * 50)
    
    try:
        fetcher = DataFetcher()
        all_data = fetcher.get_fred_gdp_components(start_date=start_date)
        
        if 'M2' not in all_data.columns:
            m2_data = fetcher.get_fred({'M2': 'M2SL'}, start_date=start_date)
            m2_series = m2_data['M2']
        else:
            m2_series = all_data['M2']
        
        m2_series = m2_series[(m2_series.index >= start_date) & (m2_series.index <= end_date)].dropna()
        
        print(f"✅ M2 data fetched successfully")
        print(f"📅 Range: {m2_series.index.min().date()} to {m2_series.index.max().date()}")
        
    except Exception as e:
        print(f"❌ M2 fetch failed: {e}")
        return None
    
    # Step 3: Calculate QTM Inflation (same as before)
    print(f"\n🎯 STEP 3: Calculating QTM Inflation")
    print("-" * 50)
    
    try:
        gdp_series = gdp_proxy_df['GDP_proxy']
        velocity_df = calc_velocity(gdp_series, m2_series)
        qtm_df = calculate_quantity_theory_inflation(gdp_series, m2_series)
        qtm_complete = qtm_df.dropna(subset=['qtm_inflation'])
        
        print(f"✅ QTM inflation calculated successfully")
        print(f"📅 Range: {qtm_complete.index.min().date()} to {qtm_complete.index.max().date()}")
        print(f"📈 Observations: {len(qtm_complete)} months")
        
    except Exception as e:
        print(f"❌ QTM calculation failed: {e}")
        return None
    
    # Step 4: Fetch CPI Data for Validation
    print(f"\n📊 STEP 4: Fetching CPI Data for Validation")
    print("-" * 50)
    
    cpi_data = fetch_cpi_data(start_date, end_date)
    if cpi_data is None:
        print("❌ Cannot proceed without CPI data for validation")
        return None
    
    # Step 5: Model Validation and Comparison
    print(f"\n🔍 STEP 5: Model Validation Against CPI")
    print("-" * 50)
    
    # Calculate accuracy metrics for all available CPI measures
    validation_results = {}
    qtm_inflation_pct = qtm_complete['qtm_inflation'] * 100  # Convert to percentage
    
    for cpi_column in cpi_data.columns:
        cpi_series = cpi_data[cpi_column]
        cpi_name = cpi_column.replace('_INFLATION', '').replace('_', ' ')
        
        accuracy_metrics = calculate_model_accuracy(
            qtm_inflation_pct, cpi_series, 
            model_name="QTM", actual_name=cpi_name
        )
        
        if accuracy_metrics:
            validation_results[cpi_name] = accuracy_metrics
            
            print(f"\n📊 QTM vs {cpi_name} Validation Results:")
            print(f"   Observations: {accuracy_metrics['observations']} months")
            print(f"   Correlation: {accuracy_metrics['correlation']:.3f}")
            print(f"   R-squared: {accuracy_metrics['r_squared']:.3f}")
            print(f"   Mean Absolute Error: {accuracy_metrics['mae']:.1f}%")
            print(f"   Root Mean Square Error: {accuracy_metrics['rmse']:.1f}%")
            print(f"   Directional Accuracy: {accuracy_metrics['direction_accuracy']:.1f}%")
            print(f"   Bias (QTM - CPI): {accuracy_metrics['bias']:+.1f}%")
            print(f"   Tracking Error: {accuracy_metrics['tracking_error']:.1f}%")
    
    # Step 6: Create Detailed Comparison Table
    print(f"\n📋 STEP 6: Detailed Monthly Comparison")
    print("-" * 50)
    
    comparison_table = create_validation_comparison_table(qtm_complete['qtm_inflation'], cpi_data)
    
    # Step 7: Model Performance Assessment
    print(f"\n🎯 STEP 7: Overall Model Performance Assessment")
    print("-" * 50)
    
    if validation_results:
        # Find best performing comparison (usually CPI ALL)
        primary_result = validation_results.get('CPI ALL', list(validation_results.values())[0])
        
        # Define performance criteria
        performance_criteria = {
            'Excellent Correlation': primary_result['correlation'] >= 0.8,
            'Good R-squared': primary_result['r_squared'] >= 0.6,
            'Low MAE': primary_result['mae'] <= 2.0,
            'Low RMSE': primary_result['rmse'] <= 3.0,
            'Good Direction': primary_result['direction_accuracy'] >= 70,
            'Low Bias': abs(primary_result['bias']) <= 1.0,
            'Reasonable Coverage': primary_result['observations'] >= 96,
            'Recent Data': qtm_complete.index.max() >= pd.Timestamp('2024-01-01')
        }
        
        passed_criteria = sum(performance_criteria.values())
        total_criteria = len(performance_criteria)
        
        print(f"📊 Model Performance Score: {passed_criteria}/{total_criteria}")
        
        for criterion, passed in performance_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {criterion}: {status}")
        
        # Overall assessment
        if passed_criteria >= total_criteria - 1:
            print(f"\n🎉 MODEL VALIDATION SUCCESSFUL!")
            print(f"🎯 QTM inflation model shows strong predictive power")
            print(f"📈 Correlation with CPI: {primary_result['correlation']:.3f}")
            print(f"🔧 Average error: {primary_result['mae']:.1f}%")
            model_status = "EXCELLENT"
        elif passed_criteria >= total_criteria - 2:
            print(f"\n✅ MODEL VALIDATION GOOD")
            print(f"🎯 QTM inflation model shows reasonable predictive power")
            print(f"📈 Some areas for improvement identified")
            model_status = "GOOD"
        else:
            print(f"\n⚠️ MODEL VALIDATION NEEDS IMPROVEMENT")
            print(f"🎯 QTM inflation model shows significant deviations from CPI")
            print(f"🔧 Consider model refinements")
            model_status = "NEEDS_WORK"
    
    # Combine all results
    enhanced_results = {
        'gdp_proxy': gdp_proxy_df,
        'velocity': velocity_df,
        'qtm_inflation': qtm_complete,
        'm2_data': m2_series,
        'cpi_data': cpi_data,
        'validation_metrics': validation_results,
        'comparison_table': comparison_table,
        'model_status': model_status,
        'summary_stats': {
            'qtm_coverage_months': len(qtm_complete),
            'cpi_coverage_months': len(cpi_data),
            'validation_months': primary_result['observations'] if validation_results else 0,
            'model_accuracy': primary_result['mae'] if validation_results else None,
            'model_correlation': primary_result['correlation'] if validation_results else None,
            'performance_score': f"{passed_criteria}/{total_criteria}"
        }
    }
    
    return enhanced_results

def plot_validation_results(results):
    """
    Create comprehensive validation plots comparing QTM vs CPI
    """
    if results is None or 'validation_metrics' not in results:
        return
    
    qtm_df = results['qtm_inflation']
    cpi_data = results['cpi_data']
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('QTM Inflation Model Validation: vs CPI (2015-2025)', fontsize=16)
    
    # Use CPI ALL as primary comparison
    if 'CPI_ALL_INFLATION' in cpi_data.columns:
        cpi_series = cpi_data['CPI_ALL_INFLATION']
        cpi_name = 'CPI All Items'
    else:
        cpi_series = cpi_data.iloc[:, 0]
        cpi_name = cpi_data.columns[0]
    
    qtm_pct = qtm_df['qtm_inflation'] * 100
    
    # 1. Time Series Comparison
    common_dates = qtm_pct.index.intersection(cpi_series.index)
    
    axes[0, 0].plot(common_dates, qtm_pct[common_dates], 'b-', linewidth=2, label='QTM Model', alpha=0.8)
    axes[0, 0].plot(common_dates, cpi_series[common_dates], 'r-', linewidth=2, label=cpi_name, alpha=0.8)
    axes[0, 0].axhline(y=2, color='green', linestyle='--', alpha=0.7, label='2% Target')
    axes[0, 0].axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-12-31'), 
                       alpha=0.2, color='gray', label='COVID Era')
    axes[0, 0].set_title('QTM Model vs Actual CPI Inflation')
    axes[0, 0].set_ylabel('Inflation Rate (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Scatter Plot with Correlation
    qtm_aligned = qtm_pct[common_dates]
    cpi_aligned = cpi_series[common_dates]
    
    axes[0, 1].scatter(cpi_aligned, qtm_aligned, alpha=0.6, s=30)
    
    # Add trend line
    z = np.polyfit(cpi_aligned, qtm_aligned, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(cpi_aligned.min(), cpi_aligned.max(), 100)
    axes[0, 1].plot(x_trend, p(x_trend), "r--", alpha=0.8)
    
    # Add perfect correlation line
    min_val = min(cpi_aligned.min(), qtm_aligned.min())
    max_val = max(cpi_aligned.max(), qtm_aligned.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k-', alpha=0.5, label='Perfect Correlation')
    
    correlation = np.corrcoef(qtm_aligned, cpi_aligned)[0, 1]
    axes[0, 1].set_title(f'QTM vs {cpi_name}\n(Correlation: {correlation:.3f})')
    axes[0, 1].set_xlabel(f'{cpi_name} (%)')
    axes[0, 1].set_ylabel('QTM Model (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Prediction Errors Over Time
    errors = qtm_aligned - cpi_aligned
    axes[1, 0].plot(common_dates, errors, 'purple', linewidth=1.5, alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].axhline(y=errors.mean(), color='red', linestyle='--', alpha=0.7, 
                       label=f'Mean Error: {errors.mean():+.1f}%')
    axes[1, 0].fill_between(common_dates, 
                           errors.mean() - errors.std(), 
                           errors.mean() + errors.std(), 
                           alpha=0.3, color='red', label='±1 Std Dev')
    axes[1, 0].axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-12-31'), 
                       alpha=0.2, color='gray')
    axes[1, 0].set_title('Model Prediction Errors (QTM - CPI)')
    axes[1, 0].set_ylabel('Error (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Error Distribution
    axes[1, 1].hist(errors, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.7, label='Perfect Accuracy')
    axes[1, 1].axvline(x=errors.mean(), color='red', linestyle='--', alpha=0.7, 
                       label=f'Mean: {errors.mean():+.1f}%')
    axes[1, 1].set_title('Distribution of Prediction Errors')
    axes[1, 1].set_xlabel('Error (QTM - CPI) (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Rolling Accuracy Metrics
    rolling_window = 24  # 24-month rolling window
    rolling_mae = errors.abs().rolling(window=rolling_window, center=True).mean()
    rolling_bias = errors.rolling(window=rolling_window, center=True).mean()
    
    axes[2, 0].plot(common_dates, rolling_mae, 'green', linewidth=2, label='24M Rolling MAE')
    axes[2, 0].plot(common_dates, rolling_bias.abs(), 'orange', linewidth=2, label='24M Rolling |Bias|')
    axes[2, 0].axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-12-31'), 
                       alpha=0.2, color='gray')
    axes[2, 0].set_title('Rolling Model Accuracy (24-Month Window)')
    axes[2, 0].set_ylabel('Error (%)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Performance by Inflation Regime
    # Categorize by CPI levels
    low_inflation = cpi_aligned < 2
    moderate_inflation = (cpi_aligned >= 2) & (cpi_aligned < 4)
    high_inflation = cpi_aligned >= 4
    
    regime_data = []
    if low_inflation.sum() > 0:
        regime_data.append(['Low (<2%)', errors[low_inflation].abs().mean(), low_inflation.sum()])
    if moderate_inflation.sum() > 0:
        regime_data.append(['Moderate (2-4%)', errors[moderate_inflation].abs().mean(), moderate_inflation.sum()])
    if high_inflation.sum() > 0:
        regime_data.append(['High (>4%)', errors[high_inflation].abs().mean(), high_inflation.sum()])
    
    if regime_data:
        regimes, mae_by_regime, counts = zip(*regime_data)
        bars = axes[2, 1].bar(regimes, mae_by_regime, alpha=0.7, 
                              color=['green', 'yellow', 'red'][:len(regimes)])
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[2, 1].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                            f'n={count}', ha='center', va='bottom')
        
        axes[2, 1].set_title('Model Accuracy by Inflation Regime')
        axes[2, 1].set_ylabel('Mean Absolute Error (%)')
        axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def create_model_report_card(results):
    """
    Generate a comprehensive model report card with grades and recommendations
    """
    if not results or 'validation_metrics' not in results:
        return None
    
    print(f"\n🎓 QTM INFLATION MODEL REPORT CARD")
    print("=" * 60)
    
    # Get primary validation metrics (CPI ALL or first available)
    validation_metrics = results['validation_metrics']
    if 'CPI ALL' in validation_metrics:
        primary_metrics = validation_metrics['CPI ALL']
        benchmark = "CPI All Items"
    else:
        primary_metrics = list(validation_metrics.values())[0]
        benchmark = list(validation_metrics.keys())[0]
    
    print(f"📊 Benchmark: {benchmark}")
    print(f"📅 Validation Period: {results['summary_stats']['validation_months']} months")
    print(f"🎯 Coverage: {results['summary_stats']['qtm_coverage_months']} QTM observations")
    
    # Grading criteria and scoring
    grades = {}
    
    # 1. Correlation Grade
    corr = primary_metrics['correlation']
    if corr >= 0.85:
        grades['Correlation'] = ('A+', 'Excellent correlation with actual inflation')
    elif corr >= 0.75:
        grades['Correlation'] = ('A', 'Strong correlation with actual inflation')
    elif corr >= 0.65:
        grades['Correlation'] = ('B+', 'Good correlation with actual inflation')
    elif corr >= 0.55:
        grades['Correlation'] = ('B', 'Moderate correlation with actual inflation')
    elif corr >= 0.45:
        grades['Correlation'] = ('C', 'Weak correlation with actual inflation')
    else:
        grades['Correlation'] = ('F', 'Poor correlation with actual inflation')
    
    # 2. Accuracy Grade (MAE)
    mae = primary_metrics['mae']
    if mae <= 1.0:
        grades['Accuracy'] = ('A+', 'Exceptional prediction accuracy')
    elif mae <= 1.5:
        grades['Accuracy'] = ('A', 'Excellent prediction accuracy')
    elif mae <= 2.0:
        grades['Accuracy'] = ('B+', 'Good prediction accuracy')
    elif mae <= 2.5:
        grades['Accuracy'] = ('B', 'Acceptable prediction accuracy')
    elif mae <= 3.5:
        grades['Accuracy'] = ('C', 'Marginal prediction accuracy')
    else:
        grades['Accuracy'] = ('F', 'Poor prediction accuracy')
    
    # 3. Bias Grade
    bias = abs(primary_metrics['bias'])
    if bias <= 0.5:
        grades['Bias'] = ('A+', 'Minimal systematic bias')
    elif bias <= 1.0:
        grades['Bias'] = ('A', 'Low systematic bias')
    elif bias <= 1.5:
        grades['Bias'] = ('B+', 'Moderate systematic bias')
    elif bias <= 2.0:
        grades['Bias'] = ('B', 'Noticeable systematic bias')
    elif bias <= 3.0:
        grades['Bias'] = ('C', 'Significant systematic bias')
    else:
        grades['Bias'] = ('F', 'Severe systematic bias')
    
    # 4. Directional Accuracy Grade
    direction = primary_metrics['direction_accuracy']
    if direction >= 85:
        grades['Direction'] = ('A+', 'Excellent directional prediction')
    elif direction >= 80:
        grades['Direction'] = ('A', 'Strong directional prediction')
    elif direction >= 75:
        grades['Direction'] = ('B+', 'Good directional prediction')
    elif direction >= 70:
        grades['Direction'] = ('B', 'Acceptable directional prediction')
    elif direction >= 60:
        grades['Direction'] = ('C', 'Weak directional prediction')
    else:
        grades['Direction'] = ('F', 'Poor directional prediction')
    
    # 5. Consistency Grade (R-squared)
    r_sq = primary_metrics['r_squared']
    if r_sq >= 0.75:
        grades['Consistency'] = ('A+', 'Highly consistent predictions')
    elif r_sq >= 0.65:
        grades['Consistency'] = ('A', 'Very consistent predictions')
    elif r_sq >= 0.55:
        grades['Consistency'] = ('B+', 'Good consistency')
    elif r_sq >= 0.45:
        grades['Consistency'] = ('B', 'Moderate consistency')
    elif r_sq >= 0.35:
        grades['Consistency'] = ('C', 'Low consistency')
    else:
        grades['Consistency'] = ('F', 'Poor consistency')
    
    # Display report card
    print(f"\n📋 INDIVIDUAL GRADES:")
    print("-" * 60)
    print(f"{'Metric':<15} {'Grade':<5} {'Score':<8} {'Assessment'}")
    print("-" * 60)
    print(f"{'Correlation':<15} {grades['Correlation'][0]:<5} {corr:<8.3f} {grades['Correlation'][1]}")
    print(f"{'Accuracy (MAE)':<15} {grades['Accuracy'][0]:<5} {mae:<8.1f} {grades['Accuracy'][1]}")
    print(f"{'Bias Control':<15} {grades['Bias'][0]:<5} {bias:<8.1f} {grades['Bias'][1]}")
    print(f"{'Direction':<15} {grades['Direction'][0]:<5} {direction:<8.1f} {grades['Direction'][1]}")
    print(f"{'Consistency':<15} {grades['Consistency'][0]:<5} {r_sq:<8.3f} {grades['Consistency'][1]}")
    
    # Calculate overall GPA
    grade_points = {'A+': 4.0, 'A': 4.0, 'B+': 3.5, 'B': 3.0, 'C': 2.0, 'F': 0.0}
    total_points = sum(grade_points[grade[0]] for grade in grades.values())
    gpa = total_points / len(grades)
    
    if gpa >= 3.7:
        overall_grade = 'A'
        assessment = 'EXCELLENT MODEL - Ready for production use'
    elif gpa >= 3.3:
        overall_grade = 'B+'
        assessment = 'STRONG MODEL - Minor refinements recommended'
    elif gpa >= 2.7:
        overall_grade = 'B'
        assessment = 'GOOD MODEL - Some improvements needed'
    elif gpa >= 2.0:
        overall_grade = 'C'
        assessment = 'MARGINAL MODEL - Significant improvements required'
    else:
        overall_grade = 'F'
        assessment = 'POOR MODEL - Major overhaul needed'
    
    print(f"\n🏆 OVERALL GRADE: {overall_grade} (GPA: {gpa:.2f}/4.0)")
    print(f"📝 Assessment: {assessment}")
    
    # Specific recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    
    if grades['Correlation'][0] in ['C', 'F']:
        print("📈 CORRELATION: Consider reviewing GDP proxy components and velocity calculation")
    
    if grades['Accuracy'][0] in ['C', 'F']:
        print("🎯 ACCURACY: Implement smoothing techniques or ensemble methods")
    
    if grades['Bias'][0] in ['C', 'F']:
        bias_direction = "over-predicting" if primary_metrics['bias'] > 0 else "under-predicting"
        print(f"⚖️ BIAS: Model is systematically {bias_direction} - consider bias correction")
    
    if grades['Direction'][0] in ['C', 'F']:
        print("🧭 DIRECTION: Review velocity calculation methodology")
    
    if grades['Consistency'][0] in ['C', 'F']:
        print("📊 CONSISTENCY: Consider adding more economic indicators to model")
    
    # Performance by period analysis
    if 'comparison_table' in results and not results['comparison_table'].empty:
        comp_df = results['comparison_table']
        
        print(f"\n📊 PERFORMANCE BY ECONOMIC PERIOD:")
        print("-" * 50)
        
        # Pre-COVID performance
        pre_covid = comp_df[comp_df['Year'] < 2020]
        if len(pre_covid) > 0:
            pre_mae = pre_covid['Abs_Diff'].mean()
            pre_grade = 'A' if pre_mae <= 1.5 else 'B' if pre_mae <= 2.5 else 'C'
            print(f"   Pre-COVID (2015-2019): Grade {pre_grade} (MAE: {pre_mae:.1f}%)")
        
        # COVID era performance
        covid_era = comp_df[(comp_df['Year'] >= 2020) & (comp_df['Year'] <= 2021)]
        if len(covid_era) > 0:
            covid_mae = covid_era['Abs_Diff'].mean()
            covid_grade = 'A' if covid_mae <= 2.0 else 'B' if covid_mae <= 3.5 else 'C'
            print(f"   COVID Era (2020-2021): Grade {covid_grade} (MAE: {covid_mae:.1f}%)")
        
        # Post-COVID performance
        post_covid = comp_df[comp_df['Year'] >= 2022]
        if len(post_covid) > 0:
            post_mae = post_covid['Abs_Diff'].mean()
            post_grade = 'A' if post_mae <= 1.5 else 'B' if post_mae <= 2.5 else 'C'
            print(f"   Post-COVID (2022+): Grade {post_grade} (MAE: {post_mae:.1f}%)")
    
    # Model strengths and weaknesses
    print(f"\n✅ MODEL STRENGTHS:")
    strengths = []
    if grades['Correlation'][0] in ['A+', 'A', 'B+']:
        strengths.append("Strong correlation with actual inflation trends")
    if grades['Accuracy'][0] in ['A+', 'A', 'B+']:
        strengths.append("Good prediction accuracy for policy purposes")
    if grades['Bias'][0] in ['A+', 'A', 'B+']:
        strengths.append("Low systematic bias in predictions")
    if grades['Direction'][0] in ['A+', 'A', 'B+']:
        strengths.append("Reliable directional predictions")
    if grades['Consistency'][0] in ['A+', 'A', 'B+']:
        strengths.append("Consistent performance across time periods")
    
    for strength in strengths[:3]:  # Show top 3 strengths
        print(f"   • {strength}")
    
    print(f"\n⚠️ AREAS FOR IMPROVEMENT:")
    weaknesses = []
    if grades['Correlation'][0] in ['C', 'F']:
        weaknesses.append("Improve correlation with actual inflation")
    if grades['Accuracy'][0] in ['C', 'F']:
        weaknesses.append("Reduce prediction errors")
    if grades['Bias'][0] in ['C', 'F']:
        weaknesses.append("Address systematic bias")
    if grades['Direction'][0] in ['C', 'F']:
        weaknesses.append("Improve directional accuracy")
    if grades['Consistency'][0] in ['C', 'F']:
        weaknesses.append("Increase prediction consistency")
    
    for weakness in weaknesses[:3]:  # Show top 3 areas for improvement
        print(f"   • {weakness}")
    
    if not weaknesses:
        print("   • Model performance is strong across all metrics")
    
    return {
        'overall_grade': overall_grade,
        'gpa': gpa,
        'individual_grades': grades,
        'assessment': assessment,
        'primary_metrics': primary_metrics,
        'benchmark': benchmark
    }

if __name__ == "__main__":
    # Run the comprehensive validation test
    print("Starting Enhanced QTM Inflation Validation Test (10-Year Analysis)...")
    print("This will test the complete pipeline and validate against actual CPI data")
    print("Coverage: Q1 2015 - Q1 2025 (includes model accuracy assessment)")
    print("Expected runtime: 3-7 minutes depending on data fetching\n")
    
    # Execute validation test
    results = qtm_inflation_validation_test()
    
    if results:
        # Generate model report card
        print("\n" + "="*70)
        report_card = create_model_report_card(results)
        
        # Show validation summary
        print(f"\n🎯 VALIDATION TEST SUMMARY")
        print("=" * 50)
        print(f"✅ Model Status: {results['model_status']}")
        print(f"📊 Performance Score: {results['summary_stats']['performance_score']}")
        print(f"📈 Model Accuracy (MAE): {results['summary_stats']['model_accuracy']:.1f}%")
        print(f"🔗 Model Correlation: {results['summary_stats']['model_correlation']:.3f}")
        print(f"📅 Validation Period: {results['summary_stats']['validation_months']} months")
        
        if report_card:
            print(f"🎓 Overall Grade: {report_card['overall_grade']} (GPA: {report_card['gpa']:.2f})")
        
        # Optional: Create validation plots (uncomment if you want visualizations)
        # print("\n📊 Creating validation visualizations...")
        # plot_validation_results(results)
        
        print(f"\n📁 Complete validation results available in returned dictionary")
        print(f"🔍 Access via: results['validation_metrics'], results['comparison_table'], etc.")
        print(f"📊 Model ready for production assessment based on validation results!")
        
        # Export summary for further analysis
        if results['comparison_table'] is not None:
            print(f"\n💾 Exporting monthly comparison table...")
            try:
                results['comparison_table'].to_csv('qtm_cpi_comparison.csv', index=False)
                print(f"✅ Comparison table saved as 'qtm_cpi_comparison.csv'")
            except Exception as e:
                print(f"⚠️ Could not save comparison table: {e}")
        
    else:
        print(f"\n❌ Validation test failed - see error messages above")
        print(f"🔧 Check your data fetching, GDP proxy, velocity, or CPI modules")
        print(f"💡 Enhanced validation requires both QTM and CPI data")
    
    print(f"\n✅ Enhanced validation test complete!")