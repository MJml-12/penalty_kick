"""
Yosoku System v3 - Zone Prediction Service (StatsBomb event-only)
Modes:
- Global  : ML model only
- Player  : player prior only (if available), else fallback global
- Auto    : blend player prior + global using alpha(k)
"""

from __future__ import annotations

import os
import json
import joblib
import numpy as np

ZONES = ["TL","TC","TR","ML","MC","MR","BL","BC","BR"]
FEATURE_COLS = ["foot_enc", "match_time", "score_diff", "is_shootout", "home_away", "pressure_index"]


def clip(x, lo, hi):
    return float(np.minimum(np.maximum(x, lo), hi))


def compute_pressure_index(match_time: float, score_diff: int, is_shootout: int = 0) -> float:
    t = clip(match_time / 120.0, 0, 1)
    close = 1.0 if abs(score_diff) <= 1 else 0.6 if abs(score_diff) == 2 else 0.3
    base = 8.0 * t * close
    if int(is_shootout) == 1:
        base = max(base, 6.5)
    return clip(base, 0, 10)


def alpha_from_k(k: int) -> float:
    if k <= 2:
        return 0.0
    a = 1.0 - np.exp(-k / 12.0)
    return clip(a, 0.0, 0.80)


class YosokuZonePredictor:
    def __init__(self):
        if not os.path.exists("models/zone_model.pkl"):
            raise FileNotFoundError("models/zone_model.pkl not found. Train the model first.")
        if not os.path.exists("models/label_encoder.pkl"):
            raise FileNotFoundError("models/label_encoder.pkl not found. Train the model first.")

        self.model = joblib.load("models/zone_model.pkl")
        self.le = joblib.load("models/label_encoder.pkl")
        self.priors = {}
        if os.path.exists("models/player_priors.json"):
            with open("models/player_priors.json", "r", encoding="utf-8") as f:
                self.priors = json.load(f)

    @staticmethod
    def _normalize(p: dict) -> dict:
        s = sum(float(v) for v in p.values())
        if s <= 0:
            return {z: 1.0 / len(ZONES) for z in ZONES}
        return {z: float(p[z]) / s for z in ZONES}

    def _global(self, foot_enc: int, match_time: int, score_diff: int, is_shootout: int, home_away: int):
        pressure = compute_pressure_index(match_time, score_diff, is_shootout)
        x = np.array([[foot_enc, match_time, score_diff, is_shootout, home_away, pressure]])
        proba = self.model.predict_proba(x)[0]

        out = {z: 0.0 for z in ZONES}
        for z, p in zip(self.le.classes_, proba):
            out[str(z)] = float(p)
        return self._normalize(out), pressure

    def predict(
        self,
        mode: str,
        player_name: str,
        foot: str,
        match_time: int,
        score_diff: int,
        is_shootout: int = 0,
        home_away: int = 1,
    ):
        foot_enc = 1 if foot.lower().startswith("r") else 0
        global_p, pressure = self._global(foot_enc, match_time, score_diff, int(is_shootout), int(home_away))

        prior = self.priors.get(player_name.strip()) if player_name else None
        if prior is None:
            return global_p, pressure, {"used": "global_only_no_player", "alpha": 0.0, "player_k": 0}

        player_p = self._normalize(prior["probs"])
        k = int(prior["count"])

        if mode.lower() == "global":
            return global_p, pressure, {"used": "global_forced", "alpha": 0.0, "player_k": k}

        if mode.lower() == "player":
            return player_p, pressure, {"used": "player_only", "alpha": 1.0, "player_k": k}

        a = alpha_from_k(k)
        blended = {z: a * player_p[z] + (1 - a) * global_p[z] for z in ZONES}
        blended = self._normalize(blended)
        return blended, pressure, {"used": "blended_auto", "alpha": a, "player_k": k}