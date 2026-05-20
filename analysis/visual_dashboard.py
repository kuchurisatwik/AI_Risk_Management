import logging
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


def create_dashboard(symbol: str, ohlcv_df: pd.DataFrame, trades_csv: str, output_dir: str = 'results') -> str:
    """Generates an extremely fast, high-signal interactive HTML dashboard."""
    
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly is not installed. Please `pip install plotly`.")
        return ""
        
    trades_path = Path(trades_csv)
    if not trades_path.exists():
        logger.error(f"Cannot find trades file: {trades_path}")
        return ""
        
    trades_df = pd.read_csv(trades_path)
    if len(trades_df) == 0:
        logger.warning("No trades to visualize.")
        return ""
        
    # Ensure datetime
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    
    # Calculate Equity & Drawdown
    trades_df = trades_df.sort_values('exit_time').reset_index(drop=True)
    trades_df['cum_pnl'] = trades_df['pnl'].cumsum()
    peak = trades_df['cum_pnl'].cummax()
    trades_df['drawdown'] = trades_df['cum_pnl'] - peak
    
    # Calculate MAE/MFE Percentages
    if 'max_adverse_excursion' in trades_df.columns:
        trades_df['mae_pct'] = (trades_df['max_adverse_excursion'].abs() / trades_df['entry_price']) * 100
        trades_df['mfe_pct'] = (trades_df['max_favorable_excursion'].abs() / trades_df['entry_price']) * 100
    else:
        trades_df['mae_pct'] = 0
        trades_df['mfe_pct'] = 0
        
    # Get top 5 worst and best trades
    best_trades = trades_df.nlargest(5, 'pnl')[['entry_time', 'direction', 'regime', 'exit_reason', 'pnl']].copy()
    worst_trades = trades_df.nsmallest(5, 'pnl')[['entry_time', 'direction', 'regime', 'exit_reason', 'pnl']].copy()
    
    # Combine for table display
    best_trades['Type'] = 'Top 5 BEST'
    worst_trades['Type'] = 'Top 5 WORST'
    table_df = pd.concat([best_trades, worst_trades])
    table_df['pnl'] = table_df['pnl'].round(2).astype(str) + " $"
    table_df['entry_time'] = table_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Create Subplots
    fig = make_subplots(
        rows=4, cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],               # Row 1: Equity, Drawdown
            [{"type": "xy"}, {"type": "xy"}],               # Row 2: MAE/MFE, Duration
            [{"type": "xy"}, {"type": "xy"}],               # Row 3: Regime Violins, Exit Reason Pie
            [{"type": "table", "colspan": 2}, None]         # Row 4: Top Trades Table
        ],
        subplot_titles=(
            "Cumulative Equity Curve ($)",
            "Underwater Drawdown ($)",
            "MAE vs MFE (Risk/Reward Dynamics)",
            "Trade Duration vs PnL",
            "PnL Distribution by Market Regime",
            "Exit Reason Breakdown",
            "Deep Dive: Top 5 Best & Worst Trades"
        ),
        vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )
    
    # --- 1. Equity Curve ---
    fig.add_trace(go.Scatter(
        x=trades_df['exit_time'], y=trades_df['cum_pnl'],
        mode='lines+markers', fill='tozeroy', line=dict(color='cyan', width=2),
        marker=dict(size=4), name='Equity',
        hovertemplate="Time: %{x}<br>Total PnL: $%{y:.2f}<extra></extra>"
    ), row=1, col=1)
    
    # --- 2. Drawdown Curve ---
    fig.add_trace(go.Scatter(
        x=trades_df['exit_time'], y=trades_df['drawdown'],
        mode='lines', fill='tozeroy', line=dict(color='crimson', width=2),
        name='Drawdown', hovertemplate="Time: %{x}<br>Drawdown: $%{y:.2f}<extra></extra>"
    ), row=1, col=2)
    
    # --- 3. MAE vs MFE Scatter ---
    colors = trades_df['exit_reason'].map({'SL_HIT': 'red', 'TP_HIT': 'green', 'TIME_BARRIER': 'orange'})
    fig.add_trace(go.Scatter(
        x=trades_df['mae_pct'], y=trades_df['mfe_pct'],
        mode='markers',
        marker=dict(color=colors, size=8, opacity=0.7, line=dict(width=1, color='white')),
        text=trades_df['exit_reason'] + "<br>PnL: $" + trades_df['pnl'].round(2).astype(str),
        hovertemplate="MAE: %{x:.2f}%<br>MFE: %{y:.2f}%<br>%{text}<extra></extra>",
        name='MAE vs MFE'
    ), row=2, col=1)
    
    # Add a 1:1 risk-reward diagonal line for reference
    max_val = max(trades_df['mae_pct'].max(), trades_df['mfe_pct'].max()) * 1.1
    if max_val > 0:
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(color='gray', dash='dash'), showlegend=False, hoverinfo='skip'), row=2, col=1)
    fig.update_xaxes(title_text="MAE % (Risk taken)", row=2, col=1)
    fig.update_yaxes(title_text="MFE % (Reward gained)", row=2, col=1)
    
    # --- 4. Trade Duration vs PnL ---
    if 'bars_held' in trades_df.columns:
        marker_colors = np.where(trades_df['pnl'] > 0, 'lime', 'red')
        fig.add_trace(go.Scatter(
            x=trades_df['bars_held'], y=trades_df['pnl'],
            mode='markers', marker=dict(color=marker_colors, size=8, opacity=0.7),
            text=trades_df['exit_reason'],
            hovertemplate="Duration: %{x} bars<br>PnL: $%{y:.2f}<br>%{text}<extra></extra>",
            name='Duration'
        ), row=2, col=2)
        fig.update_xaxes(title_text="Bars Held", row=2, col=2)
        fig.update_yaxes(title_text="Trade PnL ($)", row=2, col=2)
        
    # --- 5. PnL Violin by Regime ---
    for regime in trades_df['regime'].unique():
        regime_data = trades_df[trades_df['regime'] == regime]
        fig.add_trace(go.Violin(
            x=regime_data['regime'], y=regime_data['pnl'],
            name=regime, box_visible=True, meanline_visible=True,
            showlegend=False
        ), row=3, col=1)
        
    # --- 6. Exit Reason Bar Chart ---
    exit_counts = trades_df['exit_reason'].value_counts()
    fig.add_trace(go.Bar(
        x=exit_counts.index, y=exit_counts.values,
        marker_color=['green' if 'TP' in str(x) else 'red' if 'SL' in str(x) else 'orange' for x in exit_counts.index],
        text=exit_counts.values, textposition='auto',
        showlegend=False
    ), row=3, col=2)
    
    # --- 7. Top & Worst Trades Table ---
    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Category</b>", "<b>Entry Time</b>", "<b>Direction</b>", "<b>Regime</b>", "<b>Exit Reason</b>", "<b>PnL ($)</b>"],
            fill_color='midnightblue',
            align='center',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=[table_df['Type'], table_df['entry_time'], table_df['direction'], table_df['regime'], table_df['exit_reason'], table_df['pnl']],
            fill_color=[np.where(table_df['Type'] == 'Top 5 BEST', 'darkgreen', 'darkred')],
            align='center',
            font=dict(color='white', size=11)
        )
    ), row=4, col=1)
    
    # --- Layout Config ---
    fig.update_layout(
        height=1400,
        title_text=f"Institutional Trade Intelligence & Diagnostic Dashboard: {symbol}",
        template="plotly_dark",
        showlegend=False,
        hovermode="closest",
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    # Save to HTML
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_file = out_dir / f"{symbol}_dashboard.html"
    
    fig.write_html(str(html_file), include_plotlyjs="cdn")
    logger.info(f"Generated clean, fast dashboard at {html_file}")
    
    return str(html_file)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 3:
        symbol = sys.argv[1]
        data_path = sys.argv[2]
        trades_csv = sys.argv[3]
        
        # Load OHLCV just for compatibility with CLI, but we don't actually use it in the UI anymore
        df = pd.DataFrame()
            
        create_dashboard(symbol, df, trades_csv)
