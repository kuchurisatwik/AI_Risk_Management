import sys
import logging
import yaml
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

from training.optimizer import AssetOptimizer

def run_optimization(symbol):
    print("\n" + "=" * 60)
    print(f"= OPTIMIZING HYPERPARAMETERS: {symbol}")
    print("=" * 60)
    
    data_dir = PROJECT_ROOT / "data" / "labeled" / symbol
    models_dir = PROJECT_ROOT / "models" / symbol
    
    if not (data_dir / 'val.parquet').exists():
        print(f"  [!] Validation data not found for {symbol}. Skipping.")
        return
        
    val_df = pd.read_parquet(data_dir / 'val.parquet')
    
    optimizer = AssetOptimizer(
        symbol=symbol,
        models_dir=str(models_dir),
        val_df=val_df,
        n_trials=100,  # 100 trials should give a solid parameter set
        output_dir='config/optimized'
    )
    
    best_params = optimizer.optimize()
    print(f"  [+] Finished optimizing {symbol}")

def main():
    print("\n" + "#" * 60)
    print("#  ADAPTIVE AI RISK MANAGEMENT SYSTEM")
    print("#  Phase D: Execution Parameter Optimization (Optuna)")
    print("#" * 60 + "\n")
    
    config_path = PROJECT_ROOT / "config" / "data_sources.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    symbols = config.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    
    for symbol in symbols:
        run_optimization(symbol)

if __name__ == "__main__":
    main()
