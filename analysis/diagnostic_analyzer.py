"""
Diagnostic Analyzer — Institutional Post-Trade Diagnostic Engine.
Computes regime-stratified metrics, exit analysis, MAE/MFE profiling,
and automatically detects failure modes.
"""

import logging
import numpy as np
import pandas as pd
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_diagnostics(df: pd.DataFrame) -> dict:
    """Computes all diagnostic metrics from enhanced trade log."""
    report = {}
    
    # Basic metrics
    total_trades = len(df)
    if total_trades == 0:
        return report
    
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    
    report['overall'] = {
        'total_trades': total_trades,
        'win_rate': len(wins) / total_trades if total_trades > 0 else 0,
        'total_pnl': df['pnl'].sum(),
        'avg_pnl': df['pnl'].mean(),
        'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
    }
    
    # A. Regime-Stratified Performance
    regime_perf = {}
    for regime, group in df.groupby('regime'):
        r_wins = group[group['pnl'] > 0]
        r_losses = group[group['pnl'] <= 0]
        regime_perf[regime] = {
            'trades': len(group),
            'win_rate': len(r_wins) / len(group),
            'total_pnl': group['pnl'].sum(),
            'avg_rr': (r_wins['pnl'].mean() / abs(r_losses['pnl'].mean())) if len(r_losses) > 0 and len(r_wins) > 0 else 0
        }
    report['regime_performance'] = regime_perf
    
    # B. Exit Reason Analysis
    exit_perf = {}
    for reason, group in df.groupby('exit_reason'):
        exit_perf[reason] = {
            'count': len(group),
            'pct_of_trades': len(group) / total_trades,
            'avg_pnl': group['pnl'].mean(),
            'avg_bars_held': group['bars_held'].mean()
        }
    report['exit_analysis'] = exit_perf
    
    # C & E. MAE/MFE & R:R Analysis
    if 'max_adverse_excursion' in df.columns and 'sl_distance_pct' in df.columns:
        # Convert MAE to a positive value representing magnitude of loss
        df['mae_abs'] = df['max_adverse_excursion'].abs()
        df['sl_dist_abs'] = (df['sl_distance_pct'] / 100.0) * df['entry_price']
        
        # How close did it get to SL before exiting?
        df['mae_to_sl_ratio'] = df['mae_abs'] / df['sl_dist_abs'].replace(0, np.nan)
        
        sl_hits = df[df['exit_reason'] == 'SL_HIT']
        if len(sl_hits) > 0:
            tight_sl_pct = (sl_hits['mae_to_sl_ratio'] < 1.05).mean() # Exited near SL
        else:
            tight_sl_pct = 0
            
        wins_updated = df[df['pnl'] > 0]
        
        report['excursion_analysis'] = {
            'avg_mae_all': df['mae_abs'].mean(),
            'avg_mfe_all': df['max_favorable_excursion'].mean(),
            'avg_mae_winning_trades': wins_updated['mae_abs'].mean() if len(wins_updated) > 0 else 0,
            'tight_sl_percentage': tight_sl_pct
        }
    
    # D. Policy Engine Impact
    if 'policy_warnings' in df.columns:
        df['has_warnings'] = df['policy_warnings'].fillna('').str.len() > 0
        with_warnings = df[df['has_warnings']]
        without_warnings = df[~df['has_warnings']]
        
        report['policy_impact'] = {
            'trades_with_reductions': len(with_warnings),
            'trades_without_reductions': len(without_warnings),
            'win_rate_with_reductions': (with_warnings['pnl'] > 0).mean() if len(with_warnings) > 0 else 0,
            'win_rate_without_reductions': (without_warnings['pnl'] > 0).mean() if len(without_warnings) > 0 else 0,
            'avg_size_with_reductions': with_warnings['position_size_usd'].mean() if len(with_warnings) > 0 else 0,
            'avg_size_without_reductions': without_warnings['position_size_usd'].mean() if len(without_warnings) > 0 else 0,
        }
        
    return report


def detect_failure_modes(report: dict, df: pd.DataFrame) -> list:
    """Auto-detects and ranks top failure modes."""
    failures = []
    
    # 1. SL Too Tight
    ex_analysis = report.get('excursion_analysis', {})
    exit_analysis = report.get('exit_analysis', {})
    
    sl_trades = exit_analysis.get('SL_HIT', {}).get('count', 0)
    total = report['overall']['total_trades']
    
    if total > 0 and sl_trades / total > 0.4:
        # If many trades hit SL, check MAE
        avg_mae_wins = ex_analysis.get('avg_mae_winning_trades', 0)
        # If winning trades suffer deep drawdowns before winning, SL is too tight
        if avg_mae_wins > 0:
            failures.append({
                'mode': 'SL_TOO_TIGHT',
                'confidence': 85,
                'evidence': f"{(sl_trades/total)*100:.1f}% of trades hit SL, and winning trades suffer deep MAE.",
                'fix': 'Widen SL multiplier in policy_engine.py or optimizer.'
            })
            
    # 2. Policy Crushing
    pol = report.get('policy_impact', {})
    if pol.get('trades_with_reductions', 0) > 0 and pol.get('trades_without_reductions', 0) > 0:
        size_with = pol['avg_size_with_reductions']
        size_without = pol['avg_size_without_reductions']
        if size_with < size_without * 0.2:
            failures.append({
                'mode': 'POLICY_CRUSHING',
                'confidence': 90,
                'evidence': f"Trades with warnings are crushed to ${size_with:.1f} vs ${size_without:.1f} normal.",
                'fix': 'Raise soft_reduction_floor in policy_engine.py.'
            })
            
    # 3. Time Barrier Too Short
    tb_trades = exit_analysis.get('TIME_BARRIER', {}).get('count', 0)
    if total > 0 and tb_trades / total > 0.3:
        failures.append({
            'mode': 'TIME_BARRIER_TOO_SHORT',
            'confidence': 80,
            'evidence': f"{(tb_trades/total)*100:.1f}% of trades exited via time barrier before hitting SL/TP.",
            'fix': 'Increase time_barrier_bars in execution parameters.'
        })

    # Sort by confidence
    failures.sort(key=lambda x: x['confidence'], reverse=True)
    return failures[:3]


def run_diagnostics(symbol: str, trades_csv: str, output_dir: str = 'results') -> str:
    """Main entrypoint for diagnostics."""
    trades_path = Path(trades_csv)
    if not trades_path.exists():
        logger.error(f"Cannot find trades file: {trades_path}")
        return ""
        
    df = pd.read_csv(trades_path)
    report = compute_diagnostics(df)
    report['failure_modes'] = detect_failure_modes(report, df)
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / f"{symbol}_analysis_report.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
        
    # Print summary to console
    print("\n" + "-"*50)
    print("DIAGNOSTIC FAILURE MODES DETECTED:")
    print("-"*50)
    if not report['failure_modes']:
        print("  No major failure modes detected. System is well calibrated.")
    else:
        for i, fm in enumerate(report['failure_modes'], 1):
            print(f"  {i}. {fm['mode']} (Confidence: {fm['confidence']}%)")
            print(f"     Evidence: {fm['evidence']}")
            print(f"     Fix: {fm['fix']}\n")
            
    return str(report_file)

if __name__ == '__main__':
    # Test stub
    import sys
    if len(sys.argv) > 2:
        run_diagnostics(sys.argv[1], sys.argv[2])
