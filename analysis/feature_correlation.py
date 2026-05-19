"""
Feature Correlation Analysis

Identifies highly collinear features using Spearman rank correlation.
Tree-based models suffer from feature dilution when given highly correlated inputs.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.config import (
    MOMENTUM_FEATURES, VOLATILITY_FEATURES,
    RISK_FEATURES, BEHAVIORAL_FEATURES
)

def main():
    data_path = PROJECT_ROOT / "data" / "labeled" / "BTCUSDT" / "train.parquet"
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Collect all unique features used in any model
    all_features = list(set(
        MOMENTUM_FEATURES + VOLATILITY_FEATURES + RISK_FEATURES + BEHAVIORAL_FEATURES
    ))
    
    # Filter features actually present in df
    available_features = [f for f in all_features if f in df.columns]
    
    print(f"\nAnalyzing {len(available_features)} unique features...")
    
    # Compute Spearman rank correlation
    corr_matrix = df[available_features].corr(method='spearman').abs()
    
    # Find highly correlated pairs (threshold > 0.85)
    threshold = 0.85
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                f1 = corr_matrix.columns[i]
                f2 = corr_matrix.columns[j]
                val = corr_matrix.iloc[i, j]
                high_corr_pairs.append((f1, f2, val))
    
    # Sort by correlation strength
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nHigh Collinearity Pairs (|rho| > {threshold}):")
    print("-" * 60)
    for f1, f2, val in high_corr_pairs:
        print(f"{val:.3f} : {f1} <--> {f2}")
        
    print("\nRecommended Actions:")
    print("For each pair, drop the less robust feature or engineer a ratio.")

if __name__ == "__main__":
    main()
