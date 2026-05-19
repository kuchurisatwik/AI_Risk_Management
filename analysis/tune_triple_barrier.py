"""
Triple-Barrier Parameter Tuning.

Explores different PT/SL multipliers and time horizons (t1) 
to find the optimal configuration that maximizes the number of valid 
setups and the meta-label success rate (hit TP before SL or Timeout).
"""

import logging
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from labeling.meta_labeler import apply_triple_barrier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def evaluate_params(args):
    df_path, pt, sl, t1 = args
    try:
        df = pd.read_parquet(df_path)
        # Apply triple barrier with current params
        res_df = apply_triple_barrier(df, pt_sl=[pt, sl], t1=t1)
        
        # Calculate stats
        dirs = res_df['primary_label'].values
        meta = res_df['meta_label'].values
        
        total_bars = len(res_df)
        valid_setups = np.sum(dirs != 0)
        successes = np.nansum(meta)
        
        win_rate = successes / valid_setups if valid_setups > 0 else 0
        
        return {
            'pt': pt,
            'sl': sl,
            't1': t1,
            'valid_setups': valid_setups,
            'successes': successes,
            'win_rate': win_rate
        }
    except Exception as e:
        logger.error(f"Error evaluating pt={pt}, sl={sl}, t1={t1}: {e}")
        return None

def main():
    data_dir = PROJECT_ROOT / "data" / "labeled" / "BTCUSDT"
    train_path = data_dir / "train.parquet"
    
    if not train_path.exists():
        logger.error(f"Train data not found at {train_path}")
        return
        
    logger.info("Starting Triple-Barrier Parameter Tuning on Train Set...")
    
    # Parameter grid
    pt_multipliers = [1.0, 1.5, 2.0, 3.0]
    sl_multipliers = [0.5, 1.0, 1.5, 2.0]
    t1_horizons = [6, 12, 24, 48] # 1.5h, 3h, 6h, 12h (assuming 15m bars)
    
    # Generate all combinations where PT >= SL (standard risk management)
    # We also allow PT < SL to see if inverse R:R works better with high win rate
    combinations = []
    for pt in pt_multipliers:
        for sl in sl_multipliers:
            for t1 in t1_horizons:
                combinations.append((train_path, pt, sl, t1))
                
    logger.info(f"Testing {len(combinations)} combinations...")
    
    results = []
    # Use multiprocessing to speed up
    # Note: apply_triple_barrier internally has loops, so it's a bit slow
    # We disable the internal logging in apply_triple_barrier for this grid search
    # by temporarily setting logging level higher
    logging.getLogger("labeling.meta_labeler").setLevel(logging.WARNING)
    
    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        for res in pool.imap_unordered(evaluate_params, combinations):
            if res is not None:
                results.append(res)
                logger.info(f"Completed: PT={res['pt']}, SL={res['sl']}, T1={res['t1']} | WR={res['win_rate']:.2%}")
                
    # Re-enable logging
    logging.getLogger("labeling.meta_labeler").setLevel(logging.INFO)
                
    if not results:
        logger.error("No results returned.")
        return
        
    df_results = pd.DataFrame(results)
    
    # Sort by Win Rate
    df_results = df_results.sort_values(by='win_rate', ascending=False)
    
    print("\n" + "=" * 60)
    print("TOP 10 PARAMETER CONFIGURATIONS (By Target Win Rate)")
    print("=" * 60)
    print(df_results.head(10).to_string(index=False))
    
    # Save results
    output_path = PROJECT_ROOT / "analysis" / "triple_barrier_tuning.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    logger.info(f"\nSaved full results to {output_path}")

if __name__ == "__main__":
    main()
