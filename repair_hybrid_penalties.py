"""
Repair hybrid_penalties.csv:
- Recompute pressure_index deterministically
- Fix synthetic foot_enc to be consistent with real per-player distribution
- Enforce ranges and valid categories
- Save to a new file by default
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

ZONES = ["TL","TC","TR","ML","MC","MR","BL","BC","BR"]

def clip(x, lo, hi):
    return float(np.minimum(np.maximum(x, lo), hi))

def compute_pressure_index(match_time: float, score_diff: int, is_shootout: int = 0) -> float:
    t = clip(match_time / 120.0, 0, 1)
    close = 1.0 if abs(score_diff) <= 1 else 0.6 if abs(score_diff) == 2 else 0.3
    base = 8.0 * t * close
    if int(is_shootout) == 1:
        base = max(base, 6.5)
    return clip(base, 0, 10)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/hybrid_penalties.csv")
    ap.add_argument("--out", dest="out_path", default="data/hybrid_penalties.repaired.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score_clip", type=int, default=6, help="clip score_diff to [-score_clip, score_clip]")
    ap.add_argument("--use_mode", action="store_true",
                    help="If set: synthetic foot_enc is set to the real-mode per player (no randomness). "
                         "If not set: sample synthetic foot_enc using per-player real p_right.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.in_path)

    # Basic required columns check
    need = {"player_name","foot_enc","match_time","score_diff","is_shootout","home_away","pressure_index","zone_target","source"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in input: {sorted(miss)}")

    # Normalize types (robust)
    df["player_name"] = df["player_name"].astype(str)
    df["source"] = df["source"].astype(str).str.lower()

    # Force ints where expected
    for col in ["foot_enc","match_time","score_diff","is_shootout","home_away"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Range clamps
    df["match_time"] = df["match_time"].clip(1, 120)
    df["score_diff"] = df["score_diff"].clip(-args.score_clip, args.score_clip)
    df["is_shootout"] = df["is_shootout"].clip(0, 1)
    df["home_away"] = df["home_away"].clip(0, 1)
    df["foot_enc"] = df["foot_enc"].clip(0, 1)

    # zone_target validity
    df["zone_target"] = df["zone_target"].astype(str)
    bad_zone = ~df["zone_target"].isin(ZONES)
    if bad_zone.any():
        # conservative fix: map invalid to MC
        df.loc[bad_zone, "zone_target"] = "MC"

    # 1) Recompute pressure_index (overwrite)
    df["pressure_index"] = [
        round(compute_pressure_index(t, sd, so), 2)
        for t, sd, so in zip(df["match_time"].values, df["score_diff"].values, df["is_shootout"].values)
    ]

    # 2) Fix synthetic foot_enc based on REAL per-player distribution
    real = df[df["source"] == "real"].copy()
    synth_mask = df["source"] == "synthetic"

    if len(real) == 0:
        # No real: fallback global only
        if args.use_mode:
            df.loc[synth_mask, "foot_enc"] = 1  # default right
        else:
            df.loc[synth_mask, "foot_enc"] = (rng.random(synth_mask.sum()) < 0.75).astype(int)
    else:
        # per-player p_right computed from REAL
        p_right = real.groupby("player_name")["foot_enc"].mean()  # mean of {0,1} = p_right
        mode_right = (real.groupby("player_name")["foot_enc"].mean() >= 0.5).astype(int)

        global_p = float(real["foot_enc"].mean())  # fallback if player unseen in real
        global_p = float(np.clip(global_p, 0.0, 1.0)) if not np.isnan(global_p) else 0.75

        synth_players = df.loc[synth_mask, "player_name"].values
        new_foot = np.empty(len(synth_players), dtype=int)

        for i, p in enumerate(synth_players):
            if p in p_right.index:
                if args.use_mode:
                    new_foot[i] = int(mode_right.loc[p])
                else:
                    pr = float(p_right.loc[p])
                    new_foot[i] = int(rng.random() < pr)
            else:
                # fallback
                if args.use_mode:
                    new_foot[i] = int(global_p >= 0.5)
                else:
                    new_foot[i] = int(rng.random() < global_p)

        df.loc[synth_mask, "foot_enc"] = new_foot

    df.to_csv(args.out_path, index=False)
    print("Saved repaired file:", args.out_path)
    print("Rows:", len(df))
    print("Source counts:\n", df["source"].value_counts())

if __name__ == "__main__":
    main()