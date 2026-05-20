import sys
import pandas as pd
from pathlib import Path
from execution.paper_trader import PaperTrader
from training.optimizer import load_optimized_config
from analysis.diagnostic_analyzer import run_diagnostics
from analysis.visual_dashboard import create_dashboard

def main():
    symbol = 'BTCUSDT'
    models_dir = Path('models') / symbol
    data_dir = Path('data/labeled') / symbol
    
    print(f"Loading data for {symbol}...")
    val_df = pd.read_parquet(data_dir / 'val.parquet')
    test_df = pd.read_parquet(data_dir / 'test.parquet')
    
    exec_params = load_optimized_config(symbol)
    trader = PaperTrader(str(models_dir), 10000.0, exec_params=exec_params)
    trader.load()
    
    print("Warming up on validation set...")
    for _, row in val_df.iterrows():
        features = {col: float(row[col]) if isinstance(row[col], (int, float)) else row[col] for col in val_df.columns}
        trader.engine.ensemble.predict(features)
        
    print("Running on test set...")
    result = trader.run(test_df)
    
    print(f"Return: {result.total_return:.2%}, Sharpe: {result.sharpe_ratio:.2f}")
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    if result.trades:
        trade_data = []
        for t in result.trades:
            trade_data.append({
                'entry_time': t.entry_time, 'exit_time': t.exit_time,
                'direction': 'LONG' if t.direction == 1 else 'SHORT',
                'entry_price': t.entry_price, 'exit_price': t.exit_price,
                'position_size_usd': t.position_size_usd, 'pnl': t.pnl,
                'regime': t.regime, 'active_branch': t.active_branch,
                'gmm_subregime': t.gmm_subregime, 'confidence': t.confidence,
                'exit_reason': t.exit_reason, 'meta_probability': t.meta_probability,
                'meta_margin': t.meta_margin, 'regime_confidence': t.regime_confidence,
                'risk_level': t.risk_level, 'behavioral_anomaly': t.behavioral_anomaly,
                'sl_distance_pct': t.sl_distance_pct, 'tp_distance_pct': t.tp_distance_pct,
                'designed_rr': t.designed_rr, 'predicted_volatility': t.predicted_volatility,
                'realized_volatility': t.realized_volatility, 'atr_14': t.atr_14,
                'bars_held': t.bars_held, 'max_adverse_excursion': t.max_adverse_excursion,
                'max_favorable_excursion': t.max_favorable_excursion, 'policy_warnings': t.policy_warnings
            })
        trades_df = pd.DataFrame(trade_data)
        trades_file = results_dir / f"{symbol}_diagnostic_log.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"Saved {len(result.trades)} trades to {trades_file}")
        
        report_file = run_diagnostics(symbol, trades_file)
        print(f"Diagnostic report saved to {report_file}")
        
        dashboard_file = create_dashboard(symbol, test_df, trades_file)
        print(f"Visual dashboard saved to {dashboard_file}")

if __name__ == '__main__':
    main()
