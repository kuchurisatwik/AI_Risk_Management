"""
Model Ensemble — Hierarchical Regime-Routed Dual-Branch Loader.

Architecture:
  1. HMM Regime Model (5-state with temporal persistence)
  2. GMM Sub-Regime Models (Trend + MeanRev)
  3. Trend Branch: Primary Direction + Meta Confidence
  4. MeanRev Branch: Signal + Meta Confidence
  5. Volatility Model
  6. Risk Model
  7. Behavioral Model
"""

import logging
import numpy as np
import joblib
from pathlib import Path
from collections import deque
from dataclasses import dataclass

from training.train_hmm_regime import prepare_hmm_features, HMM_FEATURES, MIN_REGIME_DWELL
from training.config import (
    TREND_FEATURES, TREND_META_FEATURES,
    MEANREV_FEATURES, MEANREV_META_FEATURES,
    MOMENTUM_FEATURES, META_FEATURES,
    VOLATILITY_FEATURES, RISK_FEATURES, BEHAVIORAL_FEATURES,
    REGIME_RISK_CONFIG
)

logger = logging.getLogger(__name__)

# Rolling window size for HMM temporal context
HMM_WINDOW_SIZE = 50


@dataclass
class ModelOutputs:
    regime_label: str = 'unknown'
    regime_confidence: float = 0.0
    gmm_subregime: str = 'unknown'
    gmm_confidence: float = 0.0
    active_branch: str = 'none'       # 'trend', 'meanrev', or 'none'
    predicted_direction: int = 0       # +1 LONG, -1 SHORT
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

        # GMM sub-regime models
        self.trend_gmm_scaler = None
        self.trend_gmm = None
        self.trend_gmm_mapping = None
        self.meanrev_gmm_scaler = None
        self.meanrev_gmm = None
        self.meanrev_gmm_mapping = None

        # Dual-branch models
        self.trend_primary = None
        self.trend_meta = None
        self.meanrev_signal = None
        self.meanrev_meta = None

        # Legacy fallback models
        self.primary_model = None
        self.meta_model = None

        # Downstream models
        self.volatility_model = None
        self.risk_model = None
        self.behavioral_model = None
        self._loaded = False

        # Rolling buffer for HMM temporal context
        self._hmm_buffer = deque(maxlen=HMM_WINDOW_SIZE)

        # Regime persistence tracking
        self._current_regime = 'unknown'
        self._regime_dwell_count = 0

    def load(self):
        logger.info("Loading Regime-Routed Dual-Branch models...")

        # 1. HMM Regime
        reg_dir = self.models_dir / 'regime'
        if (reg_dir / 'regime_hmm_scaler.pkl').exists():
            self.regime_scaler = joblib.load(reg_dir / 'regime_hmm_scaler.pkl')
            self.regime_hmm = joblib.load(reg_dir / 'regime_hmm_model.pkl')
            self.regime_mapping = joblib.load(reg_dir / 'regime_hmm_mapping.pkl')

        # 2. GMM Sub-Regimes
        if (reg_dir / 'gmm_trend_model.pkl').exists():
            self.trend_gmm_scaler = joblib.load(reg_dir / 'gmm_trend_scaler.pkl')
            self.trend_gmm = joblib.load(reg_dir / 'gmm_trend_model.pkl')
            self.trend_gmm_mapping = joblib.load(reg_dir / 'gmm_trend_mapping.pkl')

        if (reg_dir / 'gmm_meanrev_model.pkl').exists():
            self.meanrev_gmm_scaler = joblib.load(reg_dir / 'gmm_meanrev_scaler.pkl')
            self.meanrev_gmm = joblib.load(reg_dir / 'gmm_meanrev_model.pkl')
            self.meanrev_gmm_mapping = joblib.load(reg_dir / 'gmm_meanrev_mapping.pkl')

        # 3. Trend Branch
        mom_dir = self.models_dir / 'momentum'
        if (mom_dir / 'trend_primary_model.pkl').exists():
            self.trend_primary = joblib.load(mom_dir / 'trend_primary_model.pkl')
            self.trend_meta = joblib.load(mom_dir / 'trend_meta_model.pkl')

        # 4. MeanRev Branch
        if (mom_dir / 'meanrev_signal_model.pkl').exists():
            self.meanrev_signal = joblib.load(mom_dir / 'meanrev_signal_model.pkl')
            self.meanrev_meta = joblib.load(mom_dir / 'meanrev_meta_model.pkl')

        # Legacy fallback
        if (mom_dir / 'primary_direction_model.pkl').exists():
            self.primary_model = joblib.load(mom_dir / 'primary_direction_model.pkl')
            self.meta_model = joblib.load(mom_dir / 'meta_confidence_model.pkl')

        # 5. Volatility
        vol_dir = self.models_dir / 'volatility'
        if (vol_dir / 'volatility_model.pkl').exists():
            self.volatility_model = joblib.load(vol_dir / 'volatility_model.pkl')

        # 6. Risk
        risk_dir = self.models_dir / 'risk'
        if (risk_dir / 'risk_model.pkl').exists():
            self.risk_model = joblib.load(risk_dir / 'risk_model.pkl')

        # 7. Behavioral
        beh_dir = self.models_dir / 'behavioral'
        if (beh_dir / 'behavioral_iforest.pkl').exists():
            self.behavioral_model = joblib.load(beh_dir / 'behavioral_iforest.pkl')

        self._loaded = True
        logger.info("All Regime-Routed models loaded.")

    def _infer_regime(self, features):
        """Run HMM regime detection with rolling-window Viterbi."""
        try:
            log_ret = features.get('log_return', 0.0)
            vol = features.get('realized_volatility', features.get('atr_14', 0.0))
            vol_ratio = features.get('volume_ratio', 1.0)
            atr_exp = features.get('atr_expansion_ratio', 1.0)

            self._hmm_buffer.append([log_ret, vol, vol_ratio, atr_exp])

            x_seq = np.array(list(self._hmm_buffer))
            x_scaled = self.regime_scaler.transform(x_seq)

            state_sequence = self.regime_hmm.predict(x_scaled)
            current_state = state_sequence[-1]
            raw_regime = self.regime_mapping.get(current_state, 'unknown')

            probs = self.regime_hmm.predict_proba(x_scaled)
            confidence = float(np.max(probs[-1]))

            # Temporal persistence enforcement
            if raw_regime == self._current_regime:
                self._regime_dwell_count += 1
            else:
                if self._regime_dwell_count >= MIN_REGIME_DWELL:
                    # Previous regime was established, transition to new
                    self._current_regime = raw_regime
                    self._regime_dwell_count = 1
                else:
                    # Previous regime wasn't established, stay with prior
                    self._regime_dwell_count += 1

            return self._current_regime, confidence

        except Exception as e:
            logger.warning(f"HMM inference error: {e}")
            return 'unknown', 0.0

    def _infer_gmm(self, features, branch):
        """Run GMM sub-regime inference for the given branch."""
        try:
            if branch == 'trend' and self.trend_gmm is not None:
                from training.train_gmm_subregime import TREND_GMM_FEATURES
                feat_list = [features.get(f, 0.0) for f in TREND_GMM_FEATURES]
                x = np.array([feat_list])
                x_scaled = self.trend_gmm_scaler.transform(x)
                cluster = self.trend_gmm.predict(x_scaled)[0]
                confidence = self.trend_gmm.predict_proba(x_scaled).max()
                label = self.trend_gmm_mapping.get(cluster, 'unknown')
                return label, float(confidence)

            elif branch == 'meanrev' and self.meanrev_gmm is not None:
                from training.train_gmm_subregime import MEANREV_GMM_FEATURES
                feat_list = [features.get(f, 0.0) for f in MEANREV_GMM_FEATURES]
                x = np.array([feat_list])
                x_scaled = self.meanrev_gmm_scaler.transform(x)
                cluster = self.meanrev_gmm.predict(x_scaled)[0]
                confidence = self.meanrev_gmm.predict_proba(x_scaled).max()
                label = self.meanrev_gmm_mapping.get(cluster, 'unknown')
                return label, float(confidence)

        except Exception as e:
            logger.warning(f"GMM {branch} inference error: {e}")

        return 'unknown', 0.0

    def _infer_trend_branch(self, features):
        """Run the trend-following branch inference."""
        if self.trend_primary is not None:
            x_prim = np.array([[features.get(f, 0.0) for f in TREND_FEATURES]])
            pred_dir = self.trend_primary.predict(x_prim)[0]
            direction = 1 if pred_dir == 1 else -1

            x_meta = np.array([[features.get(f, 0.0) for f in TREND_META_FEATURES]])
            probability = self.trend_meta.predict_proba(x_meta)[0, 1]
            margin = self.trend_meta.predict(x_meta, output_margin=True)[0]

            return direction, float(probability), float(margin)

        # Legacy fallback
        if self.primary_model is not None:
            x_mom = np.array([[features.get(f, 0.0) for f in MOMENTUM_FEATURES]])
            pred_dir = self.primary_model.predict(x_mom)[0]
            direction = 1 if pred_dir == 1 else -1

            x_meta = np.array([[features.get(f, 0.0) for f in META_FEATURES]])
            probability = self.meta_model.predict_proba(x_meta)[0, 1]
            margin = self.meta_model.predict(x_meta, output_margin=True)[0]

            return direction, float(probability), float(margin)

        return 0, 0.5, 0.0

    def _infer_meanrev_branch(self, features):
        """Run the mean-reversion branch inference."""
        if self.meanrev_signal is not None:
            x_sig = np.array([[features.get(f, 0.0) for f in MEANREV_FEATURES]])
            pred = self.meanrev_signal.predict(x_sig)[0]

            # Direction based on VWAP distance
            vwap_z = features.get('vwap_zscore', 0.0)
            if vwap_z > 0:
                direction = -1  # Price above VWAP → short reversion
            elif vwap_z < 0:
                direction = 1   # Price below VWAP → long reversion
            else:
                direction = 0

            x_meta = np.array([[features.get(f, 0.0) for f in MEANREV_META_FEATURES]])
            probability = self.meanrev_meta.predict_proba(x_meta)[0, 1]
            margin = self.meanrev_meta.predict(x_meta, output_margin=True)[0]

            return direction, float(probability), float(margin)

        return 0, 0.5, 0.0

    def predict(self, features: dict) -> ModelOutputs:
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        out = ModelOutputs()

        # 1. Regime Detection
        out.regime_label, out.regime_confidence = self._infer_regime(features)

        # Determine which branch to activate
        regime_cfg = REGIME_RISK_CONFIG.get(out.regime_label, {})
        branch = regime_cfg.get('branch', 'trend')

        # 2. Route to appropriate branch
        if branch == 'block':
            out.active_branch = 'none'
            out.predicted_direction = 0
            out.meta_probability = 0.0
            out.meta_margin = 0.0
        elif branch == 'trend':
            out.active_branch = 'trend'
            out.gmm_subregime, out.gmm_confidence = self._infer_gmm(features, 'trend')
            out.predicted_direction, out.meta_probability, out.meta_margin = self._infer_trend_branch(features)
        elif branch == 'meanrev':
            out.active_branch = 'meanrev'
            out.gmm_subregime, out.gmm_confidence = self._infer_gmm(features, 'meanrev')
            out.predicted_direction, out.meta_probability, out.meta_margin = self._infer_meanrev_branch(features)
        elif branch == 'best':
            # Choppy regime: pick the branch with the higher meta_probability
            trend_dir, trend_prob, trend_margin = self._infer_trend_branch(features)
            mr_dir, mr_prob, mr_margin = self._infer_meanrev_branch(features)

            if trend_prob >= mr_prob:
                out.active_branch = 'trend'
                out.gmm_subregime, out.gmm_confidence = self._infer_gmm(features, 'trend')
                out.predicted_direction, out.meta_probability, out.meta_margin = trend_dir, trend_prob, trend_margin
            else:
                out.active_branch = 'meanrev'
                out.gmm_subregime, out.gmm_confidence = self._infer_gmm(features, 'meanrev')
                out.predicted_direction, out.meta_probability, out.meta_margin = mr_dir, mr_prob, mr_margin

        # Add to features for downstream models
        features['regime_cluster'] = 0
        features['regime_confidence'] = out.regime_confidence
        features['momentum_probability'] = out.meta_probability

        # 3. Volatility
        if self.volatility_model is not None:
            x_vol = np.array([[features.get(f, 0.0) for f in VOLATILITY_FEATURES]])
            out.predicted_volatility = self.volatility_model.predict(x_vol)[0]
            features['predicted_volatility'] = out.predicted_volatility

        # 4. Risk
        if self.risk_model is not None:
            x_risk = np.array([[features.get(f, 0.0) for f in RISK_FEATURES]])
            out.risk_level = self.risk_model.predict(x_risk)[0]

        # 5. Behavioral
        if self.behavioral_model is not None:
            x_beh = np.array([[features.get(f, 0.0) for f in BEHAVIORAL_FEATURES]])
            anomaly = self.behavioral_model.predict(x_beh)[0]
            out.behavioral_anomaly = (anomaly == -1)

        return out
