"""
Yosoku System v3 - Train Zone Model (StatsBomb event-only)
Predict: zone_target (TL..BR)
Features: foot_enc, match_time, score_diff, is_shootout, home_away, pressure_index

Outputs:
  - models/zone_model.pkl
  - models/label_encoder.pkl
  - models/player_priors.json (count + zone probabilities)
"""

from __future__ import annotations

import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, log_loss

ZONES = ["TL","TC","TR","ML","MC","MR","BL","BC","BR"]
FEATURE_COLS = ["foot_enc", "match_time", "score_diff", "is_shootout", "home_away", "pressure_index"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_data(path: str = "data/hybrid_penalties.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    need = set(["player_name", "zone_target"] + FEATURE_COLS)
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {sorted(miss)}")
    return df


def compute_player_priors(df: pd.DataFrame, min_k: int = 5) -> dict:
    counts = (
        df.groupby(["player_name", "zone_target"])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=ZONES, fill_value=0)
    )
    totals = counts.sum(axis=1).astype(int)
    probs = (counts + 1.0).div((counts + 1.0).sum(axis=1), axis=0)

    priors = {}
    for player in probs.index:
        k = int(totals.loc[player])
        if k >= min_k:
            priors[player] = {"count": k, "probs": {z: float(probs.loc[player, z]) for z in ZONES}}
    return priors


def main():
    df = load_data("data/hybrid_penalties.csv")

    X = df[FEATURE_COLS].values
    le = LabelEncoder()
    y = le.fit_transform(df["zone_target"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(le.classes_),
        max_depth=5,
        learning_rate=0.08,
        n_estimators=300,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=4,
    )
    model.fit(X_train, y_train, verbose=False)

    prob = model.predict_proba(X_test)
    pred = prob.argmax(axis=1)

    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, prob, labels=list(range(len(le.classes_))))

    print("📈 Evaluation (zone-only, event-only)")
    print(f"- Accuracy: {acc:.3f}")
    print(f"- Multiclass LogLoss: {ll:.3f}\n")
    print(classification_report(y_test, pred, target_names=le.classes_))

    ensure_dir("models")
    joblib.dump(model, "models/zone_model.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    print("💾 Saved models/zone_model.pkl and models/label_encoder.pkl")

    priors = compute_player_priors(df, min_k=5)
    with open("models/player_priors.json", "w", encoding="utf-8") as f:
        json.dump(priors, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved models/player_priors.json (players: {len(priors)})")


if __name__ == "__main__":
    main()