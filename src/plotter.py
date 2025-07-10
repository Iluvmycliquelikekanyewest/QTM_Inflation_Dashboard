"""
plotter.py
----------
Interactive plotting for the Monetary Inflation Dashboard.
Creates Plotly charts for velocity, inflation, and comparison analysis.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

def make_plotly(df: pd.DataFrame, title: str = "Monetary Inflation Dashboard") -> go.Figure:
    """
    Create the main interactive dashboard chart.
    
    Args:
        df: DataFrame with columns like velocity, monetary_inflation_mom, monetary_inflation_yoy
        title: Chart title
        
    Returns:
        plotly.graph_objects.Figure
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to plotter")
        return go.Figure().add_annotation(text="No data to display", showarrow=False)
    
    # Create subplots: 2 rows, 1 column
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Velocity of Money", "Monetary Inflation Rates"),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Velocity plot (top)
    if 'velocity' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['velocity'],
                mode='lines',
                name='Velocity (V = GDP/M)',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                            'Date: %{x}<br>' +
                            'Value: %{y:.3f}<br>' +
                            '<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Inflation rates (bottom)
    if 'monetary_inflation_mom' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['monetary_inflation_mom'] * 100,  # Convert to percentage
                mode='lines',
                name='MoM Inflation (Annualized %)',
                line=dict(color='#ff7f0e', width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                            'Date: %{x}<br>' +
                            'Rate: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ),
            row=2, col=1
        )
    
    if 'monetary_inflation_yoy' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['monetary_inflation_yoy'] * 100,  # Convert to percentage
                mode='lines',
                name='YoY Inflation (%)',
                line=dict(color='#2ca02c', width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                            'Date: %{x}<br>' +
                            'Rate: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Add zero line for inflation rates
    fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20)
        ),
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified',
        template='plotly_white'
    )
    
    # Update x-axes
    fig.update_xaxes(
        title_text="Date",
        row=2, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text="Velocity",
        row=1, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    fig.update_yaxes(
        title_text="Inflation Rate (%)",
        row=2, col=1,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    logger.info(f"Created plotly chart with {len(df)} data points")
    return fig

def plot_velocity_components(gdp_proxy: pd.Series, money_supply: pd.Series, 
                           velocity: pd.Series = None) -> go.Figure:
    """
    Create a comparison chart showing GDP proxy, money supply, and calculated velocity.
    
    Args:
        gdp_proxy: GDP proxy series
        money_supply: Money supply series  
        velocity: Calculated velocity series (optional)
        
    Returns:
        plotly.graph_objects.Figure
    """
    # Align series
    df = pd.DataFrame({
        'GDP_Proxy': gdp_proxy,
        'Money_Supply': money_supply
    }).dropna()
    
    if velocity is not None:
        df['Velocity'] = velocity
    
    if df.empty:
        return go.Figure().add_annotation(text="No data to display", showarrow=False)
    
    # Create subplots: 3 rows if velocity provided, 2 rows otherwise
    n_rows = 3 if velocity is not None else 2
    subplot_titles = ["GDP Proxy (Monthly)", "Money Supply (M2)"]
    if velocity is not None:
        subplot_titles.append("Velocity (GDP/M2)")
    
    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # GDP Proxy
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['GDP_Proxy'],
            mode='lines',
            name='GDP Proxy',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    # Money Supply
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Money_Supply'],
            mode='lines',
            name='M2 Money Supply',
            line=dict(color='#ff7f0e', width=2)
        ),
        row=2, col=1
    )
    
    # Velocity (if provided)
    if velocity is not None and 'Velocity' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Velocity'],
                mode='lines',
                name='Velocity',
                line=dict(color='#2ca02c', width=2)
            ),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title="Velocity Components Analysis",
        height=600 if n_rows == 2 else 800,
        showlegend=False,  # Subplot titles are clear enough
        template='plotly_white'
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=n_rows, col=1)
    fig.update_yaxes(title_text="Billions $", row=1, col=1)
    fig.update_yaxes(title_text="Billions $", row=2, col=1)
    if n_rows == 3:
        fig.update_yaxes(title_text="Velocity", row=3, col=1)
    
    return fig

def plot_inflation_comparison(inflation_df: pd.DataFrame, 
                            additional_series: dict = None) -> go.Figure:
    """
    Create a focused chart comparing different inflation measures.
    
    Args:
        inflation_df: DataFrame with monetary inflation columns
        additional_series: Dict of {name: series} for additional inflation measures
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    # Monetary inflation rates
    if 'monetary_inflation_mom' in inflation_df.columns:
        fig.add_trace(
            go.Scatter(
                x=inflation_df.index,
                y=inflation_df['monetary_inflation_mom'] * 100,
                mode='lines',
                name='Monetary MoM (Annualized)',
                line=dict(color='#1f77b4', width=2)
            )
        )
    
    if 'monetary_inflation_yoy' in inflation_df.columns:
        fig.add_trace(
            go.Scatter(
                x=inflation_df.index,
                y=inflation_df['monetary_inflation_yoy'] * 100,
                mode='lines',
                name='Monetary YoY',
                line=dict(color='#ff7f0e', width=2)
            )
        )
    
    # Additional series (e.g., CPI, PCE deflator)
    if additional_series:
        colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for i, (name, series) in enumerate(additional_series.items()):
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series * 100 if series.max() <= 1 else series,  # Auto-detect percentage
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2)
                )
            )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Update layout
    fig.update_layout(
        title="Inflation Measures Comparison",
        xaxis_title="Date",
        yaxis_title="Inflation Rate (%)",
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_summary_table(velocity: pd.Series, inflation_df: pd.DataFrame) -> go.Figure:
    """
    Create a summary statistics table for the dashboard.
    
    Args:
        velocity: Velocity series
        inflation_df: Inflation DataFrame
        
    Returns:
        plotly.graph_objects.Figure (table)
    """
    # Calculate statistics
    stats_data = []
    
    # Velocity stats
    if not velocity.empty:
        stats_data.append([
            "Velocity",
            f"{velocity.mean():.3f}",
            f"{velocity.std():.3f}",
            f"{velocity.min():.3f}",
            f"{velocity.max():.3f}",
            f"{len(velocity.dropna())}"
        ])
    
    # MoM inflation stats
    if 'monetary_inflation_mom' in inflation_df.columns:
        mom = inflation_df['monetary_inflation_mom'].dropna()
        if not mom.empty:
            stats_data.append([
                "MoM Inflation (Ann.)",
                f"{mom.mean()*100:.2f}%",
                f"{mom.std()*100:.2f}%",
                f"{mom.min()*100:.2f}%",
                f"{mom.max()*100:.2f}%",
                f"{len(mom)}"
            ])
    
    # YoY inflation stats  
    if 'monetary_inflation_yoy' in inflation_df.columns:
        yoy = inflation_df['monetary_inflation_yoy'].dropna()
        if not yoy.empty:
            stats_data.append([
                "YoY Inflation",
                f"{yoy.mean()*100:.2f}%",
                f"{yoy.std()*100:.2f}%",
                f"{yoy.min()*100:.2f}%", 
                f"{yoy.max()*100:.2f}%",
                f"{len(yoy)}"
            ])
    
    if not stats_data:
        return go.Figure().add_annotation(text="No statistics to display", showarrow=False)
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Series", "Mean", "Std Dev", "Min", "Max", "Count"],
            fill_color='lightblue',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=list(zip(*stats_data)),
            fill_color='white',
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig.update_layout(
        title="Summary Statistics",
        height=200,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def save_chart(fig: go.Figure, filename: str, format: str = 'html') -> str:
    """
    Save a Plotly figure to file.
    
    Args:
        fig: Plotly figure
        filename: Output filename (without extension)
        format: 'html', 'png', 'pdf', or 'svg'
        
    Returns:
        str: Full filepath of saved file
    """
    full_path = f"{filename}.{format}"
    
    try:
        if format == 'html':
            fig.write_html(full_path)
        elif format == 'png':
            fig.write_image(full_path, width=1200, height=800)
        elif format == 'pdf':
            fig.write_image(full_path, format='pdf', width=1200, height=800)
        elif format == 'svg':
            fig.write_image(full_path, format='svg', width=1200, height=800)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Chart saved to {full_path}")
        return full_path
        
    except Exception as e:
        logger.error(f"Failed to save chart: {e}")
        raise