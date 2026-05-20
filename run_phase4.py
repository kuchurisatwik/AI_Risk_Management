"""
RUN PHASE 4: Institutional Redesign (Multi-Asset)
Executes HMM Regime Detection, Dual-Branch Meta-Ensemble Training, and Z-Score Execution for multiple assets.
"""

import sys
import logging
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

from training.train_hmm_regime import fit_hmm_model, assign_hmm_labels, save_hmm_model
from training.train_gmm_subregime import fit_gmm_subregime, assign_gmm_labels, save_gmm_model
from training.train_trend_model import train_trend_model
from training.train_meanrev_model import train_meanrev_model
from training.train_volatility import train_volatility
from training.train_risk import train_risk
from training.train_behavioral import train_behavioral
from execution.paper_trader import PaperTrader
from training.config import VOLATILITY_FEATURES, RISK_FEATURES, BEHAVIORAL_FEATURES, LABEL_VOLATILITY, LABEL_RISK, LABEL_BEHAVIORAL
from inference.model_ensemble import ModelEnsemble

def run_pipeline_for_symbol(symbol):
    print("\n" + "*" * 60)
    print(f"* PROCESSING SYMBOL: {symbol}")
    print("*" * 60 + "\n")
    
    data_dir = PROJECT_ROOT / "data" / "labeled" / symbol
    models_dir = PROJECT_ROOT / "models" / symbol
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- 1. Load Data ----
    print("STEP 1: LOAD DATA")
    if not (data_dir / 'train.parquet').exists():
        print(f"  [!] Labeled data not found for {symbol}. Skipping.")
        return
        
    train_df = pd.read_parquet(data_dir / 'train.parquet')
    val_df = pd.read_parquet(data_dir / 'val.parquet')
    test_df = pd.read_parquet(data_dir / 'test.parquet')
    
    print(f"  Train: {len(train_df)} rows")
    
    # ---- 2. Train HMM Regime Model ----
    print("\nSTEP 2: HMM REGIME DETECTION")
    scaler, hmm_model, mapping = fit_hmm_model(train_df, n_components=4)
    
    train_df = assign_hmm_labels(train_df, scaler, hmm_model, mapping)
    val_df = assign_hmm_labels(val_df, scaler, hmm_model, mapping)
    test_df = assign_hmm_labels(test_df, scaler, hmm_model, mapping)
    
    save_hmm_model(scaler, hmm_model, mapping, models_dir / "regime")
    
    # ---- 3. Train Sub-Regime GMMs ----
    print("\nSTEP 3: SUB-REGIME CLUSTERING (GMM)")
    # Trend GMM
    trend_train = train_df[train_df['regime_label'].isin(['trending_low_vol', 'trending_high_vol'])].copy()
    if not trend_train.empty:
        trend_scaler, trend_gmm, trend_mapping = fit_gmm_subregime(trend_train, branch='trend', n_components=3)
        if trend_gmm is not None:
            save_gmm_model(trend_scaler, trend_gmm, trend_mapping, models_dir / "regime", branch='trend')
    
    # MeanRev GMM
    mr_train = train_df[train_df['regime_label'].isin(['sideways_low_vol'])].copy()
    if not mr_train.empty:
        mr_scaler, mr_gmm, mr_mapping = fit_gmm_subregime(mr_train, branch='meanrev', n_components=3)
        if mr_gmm is not None:
            save_gmm_model(mr_scaler, mr_gmm, mr_mapping, models_dir / "regime", branch='meanrev')

    # ---- 4. Train Dual-Branch Models ----
    print("\nSTEP 4: DUAL-BRANCH MODEL TRAINING (Trend + MeanRev)")
    print("  [Trend Branch]")
    train_trend_model(train_df, val_df, test_df, models_dir / "momentum")
    
    print("  [MeanRev Branch]")
    train_meanrev_model(train_df, val_df, test_df, models_dir / "momentum")
    
    # ---- 5. Train Downstream Models ----
    print("\nSTEP 5: TRAIN DOWNSTREAM MODELS (Vol, Risk, Behavioral)")
    train_volatility(train_df, val_df, test_df, models_dir / "volatility")
    
    print("  Populating upstream predictions (momentum_probability) for Risk/Behavioral models...")
    ensemble = ModelEnsemble(str(models_dir))
    ensemble.load()
    
    from training.config import TREND_META_FEATURES, MEANREV_META_FEATURES, REGIME_RISK_CONFIG
    
    for df in [train_df, val_df, test_df]:
        # Fast vectorized inference for meta_probability
        # 1. Prepare features
        x_trend = df[[f for f in TREND_META_FEATURES if f in df.columns]].copy()
        for f in TREND_META_FEATURES:
            if f not in x_trend.columns:
                x_trend[f] = 0.0
        x_trend = x_trend[TREND_META_FEATURES].values
        
        x_mr = df[[f for f in MEANREV_META_FEATURES if f in df.columns]].copy()
        for f in MEANREV_META_FEATURES:
            if f not in x_mr.columns:
                x_mr[f] = 0.0
        x_mr = x_mr[MEANREV_META_FEATURES].values
        
        # 2. Get probabilities
        trend_prob = ensemble.trend_meta.predict_proba(x_trend)[:, 1] if ensemble.trend_meta else np.full(len(df), 0.5)
        mr_prob = ensemble.meanrev_meta.predict_proba(x_mr)[:, 1] if ensemble.meanrev_meta else np.full(len(df), 0.5)
        
        # 3. Route based on regime
        momentum_prob = np.full(len(df), 0.5)
        
        for regime in df['regime_label'].unique():
            idx = df['regime_label'] == regime
            branch = REGIME_RISK_CONFIG.get(regime, {}).get('branch', 'trend')
            if branch == 'trend':
                momentum_prob[idx] = trend_prob[idx]
            elif branch == 'meanrev':
                momentum_prob[idx] = mr_prob[idx]
            elif branch == 'best':
                momentum_prob[idx] = np.maximum(trend_prob[idx], mr_prob[idx])
            else:
                momentum_prob[idx] = 0.0
                
        df['momentum_probability'] = momentum_prob
        
    train_risk(train_df, val_df, test_df, models_dir / "risk")
    train_behavioral(train_df, val_df, test_df, models_dir / "behavioral")
    
    # ---- 6. Paper Trading ----
    print("\nSTEP 6: PAPER TRADING (Daily Ranker + Adaptive Sizing)")
    
    from training.optimizer import load_optimized_config
    exec_params = load_optimized_config(symbol)
    
    trader = PaperTrader(
        models_dir=str(models_dir),
        initial_equity=10000.0,
        exec_params=exec_params
    )
    
    print("  Warming up model states with validation set...")
    trader.engine.load()
    for _, row in val_df.iterrows():
        features = {col: float(row[col]) if isinstance(row[col], (int, float, np.integer, np.floating)) else row[col] for col in val_df.columns}
        _ = trader.engine.ensemble.predict(features)
        
    print("  Running simulation on test set...")
    result = trader.run(test_df)
    
    print(f"\n  PERFORMANCE METRICS ({symbol}):")
    print(f"    Total Return:     {result.total_return * 100:.2f}%")
    print(f"    Sharpe Ratio:     {result.sharpe_ratio:.2f}")
    print(f"    Max Drawdown:     {result.max_drawdown * 100:.2f}%")
    print(f"    Total Trades:     {result.total_trades}")
    
    if result.total_trades > 0:
        freq = result.total_trades / len(test_df)
        print(f"    Trade Frequency:  {freq:.1%} of bars")
        print(f"    Win Rate:         {result.win_rate:.1%}")
        print(f"    Avg RR Realized:  {result.avg_rr_realized:.2f}")
    
    # Save trades
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    if result.trades:
        trade_data = []
        for t in result.trades:
            trade_data.append({
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': 'LONG' if t.direction == 1 else 'SHORT',
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'position_size_usd': t.position_size_usd,
                'pnl': t.pnl,
                'regime': t.regime,
                'active_branch': t.active_branch,
                'gmm_subregime': t.gmm_subregime,
                'confidence': t.confidence,
                'exit_reason': t.exit_reason,
                'meta_probability': t.meta_probability,
                'meta_margin': t.meta_margin,
                'regime_confidence': t.regime_confidence,
                'risk_level': t.risk_level,
                'behavioral_anomaly': t.behavioral_anomaly,
                'sl_distance_pct': t.sl_distance_pct,
                'tp_distance_pct': t.tp_distance_pct,
                'designed_rr': t.designed_rr,
                'predicted_volatility': t.predicted_volatility,
                'realized_volatility': t.realized_volatility,
                'atr_14': t.atr_14,
                'bars_held': t.bars_held,
                'max_adverse_excursion': t.max_adverse_excursion,
                'max_favorable_excursion': t.max_favorable_excursion,
                'policy_warnings': t.policy_warnings
            })
        trades_df = pd.DataFrame(trade_data)
        trades_file = results_dir / f"{symbol}_diagnostic_log.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"\n  Saved {len(result.trades)} detailed diagnostic trades to {trades_file}")
        
        # ---- 7. Run Diagnostics and Visuals ----
        print("\nSTEP 7: POST-TRADE DIAGNOSTICS & VISUALIZATION")
        try:
            from analysis.diagnostic_analyzer import run_diagnostics
            from analysis.visual_dashboard import create_dashboard
            
            report_file = run_diagnostics(symbol, trades_file)
            print(f"  Diagnostic report saved to {report_file}")
            
            dashboard_file = create_dashboard(symbol, test_df, trades_file)
            print(f"  Visual dashboard saved to {dashboard_file}")
        except ImportError as e:
            print(f"  [!] Skipping diagnostics: {e}")
        except Exception as e:
            print(f"  [!] Diagnostics failed: {e}")

def main():
    print("\n" + "#" * 60)
    print("#  ADAPTIVE AI RISK MANAGEMENT SYSTEM")
    print("#  Phase 4: Institutional Overhaul (Multi-Asset)")
    print("#" * 60 + "\n")
    
    config_path = PROJECT_ROOT / "config" / "data_sources.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    symbols = config.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    
    for symbol in symbols:
        run_pipeline_for_symbol(symbol)

if __name__ == "__main__":
    main()
