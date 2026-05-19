"""
Model Ensemble — Hierarchical Meta-Ensemble Loader.

Loads the new architecture:
1. HMM Regime Model (Rolling-Window Viterbi)
2. Primary Direction Model
3. Secondary Meta Model (Confidence)
4. Volatility Model
5. Risk Model
6. Behavioral Model
"""

import logging
import numpy as np
import joblib
from pathlib import Path
from collections import deque
from dataclasses import dataclass

from training.train_hmm_regime import prepare_hmm_features, HMM_FEATURES

logger = logging.getLogger(__name__)

# Rolling window size for HMM temporal context
HMM_WINDOW_SIZE = 50


@dataclass
class ModelOutputs:
    regime_label: str = 'unknown'
    regime_confidence: float = 0.0
    predicted_direction: int = 0
    meta_probability: float = 0.0
    meta_margin: float = 0.0
    predicted_volatility: float = 0.0
    risk_level: str = 'NO_TRADE'
    behavioral_anomaly: bool = False


class ModelEnsemble:
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.regime_scaler = None
        self.regime_hmm = None
        self.regime_mapping = None
        self.primary_model = None
        self.meta_model = None
        self.volatility_model = None
        self.risk_model = None
        self.behavioral_model = None
        self._loaded = False
        
        # Rolling buffer for HMM temporal context (Bug 1 fix)
        self._hmm_buffer = deque(maxlen=HMM_WINDOW_SIZE)
        
    def load(self):
        logger.info("Loading HMM Meta-Ensemble models...")
        
        # 1. HMM Regime
        reg_dir = self.models_dir / 'regime'
        if (reg_dir / 'regime_hmm_scaler.pkl').exists():
            self.regime_scaler = joblib.load(reg_dir / 'regime_hmm_scaler.pkl')
            self.regime_hmm = joblib.load(reg_dir / 'regime_hmm_model.pkl')
            self.regime_mapping = joblib.load(reg_dir / 'regime_hmm_mapping.pkl')
        
        # 2. Meta-Ensemble
        mom_dir = self.models_dir / 'momentum'
        self.primary_model = joblib.load(mom_dir / 'primary_direction_model.pkl')
        self.meta_model = joblib.load(mom_dir / 'meta_confidence_model.pkl')
        
        # 3. Volatility
        vol_dir = self.models_dir / 'volatility'
        self.volatility_model = joblib.load(vol_dir / 'volatility_model.pkl')
        
        # 4. Risk
        risk_dir = self.models_dir / 'risk'
        self.risk_model = joblib.load(risk_dir / 'risk_model.pkl')
        
        # 5. Behavioral
        beh_dir = self.models_dir / 'behavioral'
        self.behavioral_model = joblib.load(beh_dir / 'behavioral_iforest.pkl')
        
        self._loaded = True
        logger.info("All HMM Meta-Ensemble models loaded.")
        
    def predict(self, features: dict) -> ModelOutputs:
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")
            
        out = ModelOutputs()
        
        # 1. Regime (HMM with Rolling-Window Viterbi)
        # The Viterbi algorithm requires a SEQUENCE of observations to compute
        # state transitions. Single-point inference defaults to the highest
        # stationary-probability state. We maintain a rolling buffer of the last
        # N observations and pass the full sequence each time.
        try:
            log_ret = features.get('log_return', 0.0)
            vol = features.get('realized_volatility', features.get('atr_14', 0.0))
            
            # Append current observation to rolling buffer
            self._hmm_buffer.append([log_ret, vol])
            
            # Pass the full sequence to HMM for proper Viterbi decoding
            x_seq = np.array(list(self._hmm_buffer))
            x_scaled = self.regime_scaler.transform(x_seq)
            
            # Predict state sequence — take the LAST element as current regime
            state_sequence = self.regime_hmm.predict(x_scaled)
            current_state = state_sequence[-1]
            out.regime_label = self.regime_mapping.get(current_state, 'unknown')
            
            # Confidence from the last observation's posterior
            probs = self.regime_hmm.predict_proba(x_scaled)
            out.regime_confidence = float(np.max(probs[-1]))
        except Exception as e:
            logger.warning(f"HMM inference error: {e}")
            out.regime_label = 'unknown'
            
        # Add regime to features for downstream
        features['regime_cluster'] = list(self.regime_mapping.keys())[list(self.regime_mapping.values()).index(out.regime_label)] if out.regime_label in self.regime_mapping.values() else 0
        features['regime_confidence'] = out.regime_confidence
        
        # 2. Primary Direction
        from training.config import MOMENTUM_FEATURES, META_FEATURES, VOLATILITY_FEATURES, RISK_FEATURES, BEHAVIORAL_FEATURES, RISK_CLASSES
        
        x_mom = np.array([[features.get(f, 0.0) for f in MOMENTUM_FEATURES]])
        pred_dir = self.primary_model.predict(x_mom)[0]
        out.predicted_direction = 1 if pred_dir == 1 else -1
        
        # 3. Meta Confidence (uses expanded META_FEATURES for orthogonal info)
        x_meta = np.array([[features.get(f, 0.0) for f in META_FEATURES]])
        out.meta_probability = self.meta_model.predict_proba(x_meta)[0, 1]
        out.meta_margin = self.meta_model.predict(x_meta, output_margin=True)[0]
        
        features['momentum_probability'] = out.meta_probability
        
        # 4. Volatility
        x_vol = np.array([[features.get(f, 0.0) for f in VOLATILITY_FEATURES]])
        out.predicted_volatility = self.volatility_model.predict(x_vol)[0]
        features['predicted_volatility'] = out.predicted_volatility
        
        # 5. Risk
        x_risk = np.array([[features.get(f, 0.0) for f in RISK_FEATURES]])
        out.risk_level = self.risk_model.predict(x_risk)[0]
        
        # 6. Behavioral
        x_beh = np.array([[features.get(f, 0.0) for f in BEHAVIORAL_FEATURES]])
        anomaly = self.behavioral_model.predict(x_beh)[0]
        out.behavioral_anomaly = (anomaly == -1)
        
        return out
