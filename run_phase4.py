"""
Phase 4 Orchestrator — Model Training Pipeline.

Trains the 4 downstream models in mandatory order:
1. Momentum -> 2. Volatility -> 3. Risk -> 4. Behavioral

Includes MLflow integration and final backtest validation.
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import mlflow
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_MOMENTUM, MLFLOW_EXPERIMENT_VOLATILITY,
    MLFLOW_EXPERIMENT_RISK, MLFLOW_EXPERIMENT_BEHAVIORAL,
    MOMENTUM_PARAMS, VOLATILITY_PARAMS, RISK_PARAMS
)
from training.train_momentum import train_momentum
from training.train_volatility import train_volatility
from training.train_risk import train_risk
from training.train_behavioral import train_behavioral
from training.backtest import run_backtest_validation

def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Model Training")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    
    # Initialize MLflow
    mlflow.set_tracking_uri((PROJECT_ROOT / MLFLOW_TRACKING_URI).as_uri())
    
    print("\n" + "#" * 60)
    print("#  ADAPTIVE AI RISK MANAGEMENT SYSTEM")
    print("#  Phase 4: Model Training & Validation")
    print("#" * 60)

    # 1. Load labeled data
    print("\nLoading labeled data from Phase 3...")
    data_dir = PROJECT_ROOT / "data" / "labeled" / "BTCUSDT"
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    
    out_dir = PROJECT_ROOT / "models"
    
    # 2. Momentum Model
    mlflow.set_experiment(MLFLOW_EXPERIMENT_MOMENTUM)
    with mlflow.start_run(run_name="lgbm_calibrated_v1"):
        mlflow.log_params(MOMENTUM_PARAMS)
        mom_model, mom_cv, mom_val, mom_test = train_momentum(train_df, val_df, test_df, out_dir / "momentum")
        mlflow.log_metrics({"val_" + k: v for k, v in mom_val.items()})
        mlflow.log_metrics({"test_" + k: v for k, v in mom_test.items()})

    # 3. Volatility Model
    mlflow.set_experiment(MLFLOW_EXPERIMENT_VOLATILITY)
    with mlflow.start_run(run_name="xgboost_v1"):
        mlflow.log_params(VOLATILITY_PARAMS)
        vol_model, vol_cv, vol_val, vol_test = train_volatility(train_df, val_df, test_df, out_dir / "volatility")
        mlflow.log_metrics({"val_" + k: v for k, v in vol_val.items()})
        mlflow.log_metrics({"test_" + k: v for k, v in vol_test.items()})

    # 4. Risk Model
    mlflow.set_experiment(MLFLOW_EXPERIMENT_RISK)
    with mlflow.start_run(run_name="lgbm_multiclass_v1"):
        mlflow.log_params(RISK_PARAMS)
        risk_model, risk_cv, risk_val, risk_test = train_risk(train_df, val_df, test_df, out_dir / "risk")
        mlflow.log_metrics({"val_" + k: v for k, v in risk_val.items()})
        mlflow.log_metrics({"test_" + k: v for k, v in risk_test.items()})

    # 5. Behavioral Model
    mlflow.set_experiment(MLFLOW_EXPERIMENT_BEHAVIORAL)
    with mlflow.start_run(run_name="iforest_v1"):
        mlflow.log_params({"contamination": 0.01})
        beh_model, beh_val, beh_test = train_behavioral(train_df, val_df, test_df, out_dir / "behavioral")
        mlflow.log_metrics({"val_" + k: v for k, v in beh_val.items()})
        mlflow.log_metrics({"test_" + k: v for k, v in beh_test.items()})
        
    # 6. Backtest Validation
    run_backtest_validation(test_df)
    
    print("\n" + "#" * 60)
    print("#  PHASE 4 COMPLETE")
    print("#  All models trained and saved to /models")
    print("#" * 60)

if __name__ == "__main__":
    main()
