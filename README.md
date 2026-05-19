# Adaptive AI Risk Intelligence Engine

An institutional-grade, regime-aware machine learning pipeline for probabilistic risk management in cryptocurrency derivatives markets. The system operates as a multi-phase execution engine — from raw market ingestion to executable, mathematically-sized trade decisions — governed by HMM-based regime detection, Triple-Barrier meta-labeling, and Fractional Kelly position sizing.

> **Status:** Research & Paper Trading · BTCUSDT 15-minute · 1-year historical dataset

---

## Architecture

```
Raw Data → Features → Labels → Regime Detection → Model Training → Ensemble
    → Policy Engine → Z-Score Threshold → Kelly Sizing → Paper Trading
```

The pipeline is organized into 6 executable phases:

| Phase | Script | Purpose |
| :--- | :--- | :--- |
| **1** | `run_phase1.py` | Data collection & ingestion (Binance REST/WebSocket) |
| **2** | `run_phase2.py` | Feature engineering & stationarity transforms |
| **3** | `run_phase3.py` | Feature store creation, correlation purge, orthogonalization |
| **4** | `run_phase4.py` | Label engineering (Triple-Barrier, behavioral, risk, volatility) |
| **5** | `run_phase5.py` | Validation, leakage checks, data quality audit |
| **6** | `run_phase6.py` | **Master execution** — HMM regime, Meta-Ensemble training, paper trading |

---

## Key Components

### Regime Detection — `training/train_hmm_regime.py`
4-component **Gaussian Hidden Markov Model** fitted on `log_return` and `realized_volatility`. Models temporal state transitions between market regimes using the Viterbi algorithm with a 50-bar rolling window for proper sequential decoding.

**Detected States:**
- `trending_low_vol` — Primary trading regime
- `trending_high_vol` — Reduced exposure
- `sideways_low_vol` — Blocked (insufficient edge)
- `choppy_high_vol` — Blocked (historically destructive)

### Meta-Ensemble — `training/train_meta_ensemble.py`
Two-stage hierarchical model following Lopez de Prado's framework:
1. **Primary Model (XGBoost):** Predicts directional bias (LONG/SHORT)
2. **Meta Model (XGBoost):** Predicts whether the primary signal will succeed under Triple-Barrier constraints, using an expanded feature set with orthogonal volatility/liquidity signals

### Label Engineering — `labeling/meta_labeler.py`
**Triple-Barrier Method** with symmetric risk-reward:
- Take Profit: `1.5 × ATR(14)`
- Stop Loss: `1.5 × ATR(14)`
- Time Barrier: `12 bars` (3 hours)

### Inference Pipeline

| Module | File | Role |
| :--- | :--- | :--- |
| **Ensemble** | `inference/model_ensemble.py` | Orchestrates all 6 models (HMM, Primary, Meta, Vol, Risk, Behavioral) |
| **Policy Engine** | `inference/policy_engine.py` | Hard blocks, regime routing, trend-alignment filter, soft risk adjustments |
| **Threshold** | `inference/threshold_engine.py` | Adaptive percentile-based momentum thresholds |
| **Z-Score Gate** | `inference/trade_decision.py` | Rolling 1000-bar Z-Score on log-odds margin (95th percentile trigger) |
| **Risk Sizer** | `inference/risk_sizer.py` | Fractional Kelly Criterion with constant-variance volatility targeting |

### Execution — `execution/paper_trader.py`
Event-driven simulator with:
- 0.01% slippage per leg
- 0.04% maker/taker fees (both entry & exit)
- Time-barrier enforcement (12-bar expiry)
- Pessimistic SL/TP resolution within the same candle

---

## Results (Out-of-Sample)

Tested on the 15% holdout test set (~5,238 bars of 15-minute BTCUSDT data):

| Metric | Value |
| :--- | :--- |
| **Total Return** | +1.79% |
| **Sharpe Ratio** | 0.78 |
| **Sortino Ratio** | 0.10 |
| **Max Drawdown** | -9.56% |
| **Total Trades** | 26 |
| **Win Rate** | 53.8% |
| **Avg R:R Realized** | 0.94 |
| **Trade Frequency** | 0.5% of bars |

---

## Project Structure

```
AI_Risk_Management/
├── run_phase1.py              # Data collection
├── run_phase2.py              # Feature engineering
├── run_phase3.py              # Feature store & correlation purge
├── run_phase4.py              # Label engineering
├── run_phase5.py              # Validation & leakage checks
├── run_phase6.py              # Master: HMM + Meta-Ensemble + Paper Trading
│
├── labeling/                  # Target engineering
│   ├── meta_labeler.py        #   Triple-Barrier method
│   ├── momentum_labeler.py    #   Directional labels
│   ├── volatility_labeler.py  #   Forward volatility targets
│   ├── risk_labeler.py        #   Multi-class risk labels
│   ├── behavioral_labeler.py  #   Anomaly targets
│   └── regime_labeler.py      #   Regime clustering labels
│
├── training/                  # Model training
│   ├── config.py              #   Feature lists, hyperparameters, thresholds
│   ├── train_hmm_regime.py    #   Gaussian HMM (4-state)
│   ├── train_meta_ensemble.py #   Primary + Meta XGBoost
│   ├── train_momentum.py      #   Legacy momentum model
│   ├── train_volatility.py    #   Forward volatility regression
│   ├── train_risk.py          #   Multi-class risk classifier
│   ├── train_behavioral.py    #   Isolation Forest anomaly detection
│   ├── walk_forward.py        #   Purged Walk-Forward CV
│   ├── evaluate.py            #   Evaluation metrics & feature importance
│   └── backtest.py            #   Walk-forward backtesting
│
├── inference/                 # Live inference pipeline
│   ├── model_ensemble.py      #   6-model hierarchical ensemble
│   ├── policy_engine.py       #   Institutional policy authority
│   ├── trade_decision.py      #   Z-Score ranking orchestrator
│   ├── threshold_engine.py    #   Adaptive percentile thresholds
│   └── risk_sizer.py          #   Fractional Kelly sizing
│
├── execution/                 # Trade execution
│   └── paper_trader.py        #   Event-driven paper trading simulator
│
├── monitoring/                # Production monitoring
│   └── drift_detector.py      #   PSI, feature drift, concept drift
│
├── analysis/                  # Research tools
│   ├── tune_triple_barrier.py #   Grid search over TP/SL/T1
│   └── feature_correlation.py #   Spearman correlation analysis
│
├── data/labeled/BTCUSDT/      # Processed datasets
│   ├── train.parquet          #   70% training split
│   ├── val.parquet            #   15% validation split
│   └── test.parquet           #   15% test split (OOS)
│
├── models/                    # Serialized model artifacts
│   ├── regime/                #   HMM scaler, model, mapping
│   ├── momentum/              #   Primary + Meta XGBoost
│   ├── volatility/            #   Volatility regression model
│   ├── risk/                  #   Multi-class risk model
│   └── behavioral/            #   Isolation Forest
│
├── results/                   # Simulation outputs
│   └── test_trades.csv        #   Detailed trade log
│
├── docs/                      # Documentation
│   ├── architecture_blueprint.md
│   ├── system_pipeline_handbook.md
│   └── institutional_audit.md
│
└── requirements.txt
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- 1-year BTCUSDT 15-minute historical data (Parquet format)

### Installation

```bash
git clone https://github.com/your-username/AI_Risk_Management.git
cd AI_Risk_Management
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install hmmlearn>=0.3
```

### Run the Full Pipeline

```bash
# Phase 1-5: Data → Features → Labels → Validation
python run_phase1.py
python run_phase2.py
python run_phase3.py
python run_phase4.py
python run_phase5.py

# Phase 6: Train all models + Paper Trading simulation
python run_phase6.py
```

### View Results

After Phase 6 completes, results are available at:
- **Console:** Full performance metrics, regime breakdown
- **`results/test_trades.csv`:** Every trade with entry/exit times, prices, PnL, regime, confidence, exit reason

---

## Configuration

All feature sets, hyperparameters, and validation thresholds are centralized in `training/config.py`:

| Config | Description |
| :--- | :--- |
| `MOMENTUM_FEATURES` | 8 features for the Primary directional model |
| `META_FEATURES` | 14 features for the Meta confidence model (includes volatility/liquidity signals) |
| `VOLATILITY_FEATURES` | 8 features for forward volatility prediction |
| `RISK_FEATURES` | 12 features for multi-class risk classification |
| `BEHAVIORAL_FEATURES` | 8 features for anomaly detection |
| `MOMENTUM_PARAMS` | XGBoost hyperparameters (n_estimators=300, lr=0.05, depth=6) |

---

## Documentation

| Document | Purpose |
| :--- | :--- |
| `docs/architecture_blueprint.md` | Full system architecture with Mermaid diagrams |
| `docs/system_pipeline_handbook.md` | 15-phase stage-by-stage engineering handbook |
| `docs/institutional_audit.md` | Pipeline audit findings and fix documentation |

---

## Dependencies

| Package | Purpose |
| :--- | :--- |
| `pandas`, `numpy`, `pyarrow` | Data manipulation & Parquet I/O |
| `xgboost` | Gradient-boosted tree models (Primary, Meta, Vol, Risk) |
| `scikit-learn` | Isolation Forest, StandardScaler, evaluation metrics |
| `hmmlearn` | Gaussian Hidden Markov Model for regime detection |
| `joblib` | Model serialization |

---

## License

This project is for research and educational purposes.
