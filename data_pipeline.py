"""
Yosoku System v3 - Hybrid Dataset Builder (StatsBomb event-only)
Real CSV minimal columns required:
  - player_name, body_part, end_y, end_z

Optional (if not present, will be generated as synthetic context):
  - match_time, score_diff, is_shootout, home_away

Output: data/hybrid_penalties.csv
Columns:
  player_name, foot_enc, match_time, score_diff, is_shootout, home_away, pressure_index, zone_target, source
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

ZONES = ["TL","TC","TR","ML","MC","MR","BL","BC","BR"]
OUT_COLS = [
    "player_name",
    "foot_enc",
    "match_time",
    "score_diff",
    "is_shootout",
    "home_away",
    "pressure_index",
    "zone_target",
    "source",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clip(x, lo, hi):
    return float(np.minimum(np.maximum(x, lo), hi))


def compute_pressure_index(match_time: float, score_diff: int, is_shootout: int = 0) -> float:
    """
    Event-only pressure proxy:
    - increases late in match
    - higher when score is close
    - shootout tends to be high pressure
    Output 0..10
    """
    t = clip(match_time / 120.0, 0, 1)
    close = 1.0 if abs(score_diff) <= 1 else 0.6 if abs(score_diff) == 2 else 0.3
    base = 8.0 * t * close
    if int(is_shootout) == 1:
        base = max(base, 6.5)
    return clip(base, 0, 10)


def zone_from_end(end_y: float, end_z: float) -> str:
    """
    Discretize end location into 3x3 zone.
    Pragmatic thresholds based on typical penalty end_y (~36-45) and end_z values from your CSV.
    """
    y = end_y
    z = end_z

    # horizontal (Left/Center/Right)
    if y < 38.5:
        col = "L"
    elif y < 42.0:
        col = "C"
    else:
        col = "R"

    # vertical (Bottom/Middle/Top)
    if z < 0.9:
        row = "B"
    elif z < 1.8:
        row = "M"
    else:
        row = "T"

    zone = f"{row}{col}"
    return zone if zone in ZONES else "MC"


def add_zone_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["zone_target"] = [zone_from_end(y, z) for y, z in zip(df["end_y"].values, df["end_z"].values)]
    return df


def sample_match_time(n: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(n)
    t = np.empty(n)
    # 65% second half, 25% first half, 10% extra time
    m2 = u < 0.65
    m1 = (u >= 0.65) & (u < 0.90)
    mx = u >= 0.90
    t[m2] = rng.integers(46, 91, size=m2.sum())
    t[m1] = rng.integers(1, 46, size=m1.sum())
    t[mx] = rng.integers(91, 121, size=mx.sum())
    return t.astype(int)


def sample_score_diff(n: int, rng: np.random.Generator) -> np.ndarray:
    values = np.array([-3, -2, -1, 0, 1, 2, 3])
    probs = np.array([0.03, 0.07, 0.20, 0.40, 0.20, 0.07, 0.03])
    return rng.choice(values, size=n, p=probs).astype(int)


def foot_enc_from_body_part(body_part: pd.Series) -> pd.Series:
    """
    StatsBomb-like: body_part is 'Right Foot' or 'Left Foot' for penalties.
    Encode Right=1, Left=0.
    If unknown, default to Right (or you can drop those rows).
    """
    right = body_part.astype(str).str.contains("Right", case=False, na=False)
    left = body_part.astype(str).str.contains("Left", case=False, na=False)
    # Right=1, Left=0
    enc = np.where(right, 1, np.where(left, 0, 1))
    return pd.Series(enc, index=body_part.index, dtype=int)


def load_real(real_csv_path: str, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(real_csv_path)

    required = {"player_name", "body_part", "end_y", "end_z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Real CSV missing required columns: {sorted(missing)}")

    df["foot_enc"] = foot_enc_from_body_part(df["body_part"])

    rng = np.random.default_rng(seed)

    if "match_time" not in df.columns:
        df["match_time"] = sample_match_time(len(df), rng)
    if "score_diff" not in df.columns:
        df["score_diff"] = sample_score_diff(len(df), rng)
    if "is_shootout" not in df.columns:
        df["is_shootout"] = 0
    if "home_away" not in df.columns:
        df["home_away"] = rng.integers(0, 2, size=len(df))

    df["pressure_index"] = [
        round(compute_pressure_index(t, sd, so), 2)
        for t, sd, so in zip(df["match_time"].values, df["score_diff"].values, df["is_shootout"].values)
    ]

    df = add_zone_target(df)

    real_out = df[["player_name", "foot_enc", "match_time", "score_diff", "is_shootout", "home_away", "pressure_index", "zone_target"]].copy()
    real_out["source"] = "real"
    return real_out


def generate_synthetic(real_df: pd.DataFrame, n_samples: int, seed: int = 7) -> pd.DataFrame:
    """
    Synthetic augmentation (event-only):
    - sample player names from real to keep realism
    - sample context features from plausible distributions
    - sample foot from per-player real distribution (fix misalignment)
    - sample zone_target from smoothed player tendency
    """
    rng = np.random.default_rng(seed)

    out = pd.DataFrame()
    out["player_name"] = rng.choice(real_df["player_name"].values, size=n_samples, replace=True)

    # --- FIX: foot_enc per player distribution (instead of global 75/25) ---
    # p_right(player) estimated from REAL rows
    p_right = real_df.groupby("player_name")["foot_enc"].mean()
    global_p = float(real_df["foot_enc"].mean()) if len(real_df) else 0.75
    global_p = float(np.clip(global_p, 0.0, 1.0)) if not np.isnan(global_p) else 0.75

    foot = np.empty(n_samples, dtype=int)
    for i, p in enumerate(out["player_name"].values):
        pr = float(p_right.loc[p]) if p in p_right.index else global_p
        foot[i] = int(rng.random() < pr)
    out["foot_enc"] = foot
    # ---------------------------------------------------------------

    out["match_time"] = sample_match_time(n_samples, rng)
    out["score_diff"] = sample_score_diff(n_samples, rng)
    out["is_shootout"] = rng.choice([0, 1], size=n_samples, p=[0.92, 0.08])
    out["home_away"] = rng.integers(0, 2, size=n_samples)

    out["pressure_index"] = [
        round(compute_pressure_index(t, sd, so), 2)
        for t, sd, so in zip(out["match_time"].values, out["score_diff"].values, out["is_shootout"].values)
    ]

    counts = (
        real_df.groupby(["player_name", "zone_target"])
               .size()
               .unstack(fill_value=0)
               .reindex(columns=ZONES, fill_value=0)
    )
    probs = (counts + 1.0).div((counts + 1.0).sum(axis=1), axis=0)

    zones = np.array(ZONES)
    out["zone_target"] = [
        rng.choice(zones, p=(probs.loc[p].values if p in probs.index else np.ones(len(ZONES)) / len(ZONES)))
        for p in out["player_name"].values
    ]

    out["source"] = "synthetic"
    return out


def build_hybrid(real_csv_path: str, output_csv_path: str, synthetic_multiplier: float = 1.0) -> pd.DataFrame:
    ensure_dir(os.path.dirname(output_csv_path) or ".")
    real_df = load_real(real_csv_path)

    n_synth = int(len(real_df) * synthetic_multiplier)
    synth_df = generate_synthetic(real_df, n_synth)

    hybrid = pd.concat([real_df, synth_df], ignore_index=True)
    hybrid = hybrid[OUT_COLS]
    hybrid.to_csv(output_csv_path, index=False)
    return hybrid


def main():
    real_path = "data/real_penalties.csv"
    out_path = "data/hybrid_penalties.csv"

    if not os.path.exists(real_path):
        raise FileNotFoundError(
            f"Place your scraped CSV at {real_path}\n"
            f"Example: rename database_pk_full_with_foot.csv -> data/real_penalties.csv"
        )

    df = build_hybrid(real_path, out_path, synthetic_multiplier=1.0)
    print("✅ Saved:", out_path)
    print("Rows:", len(df))
    print(df["source"].value_counts())


if __name__ == "__main__":
    main()