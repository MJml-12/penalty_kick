# Where will the penalty go?

Not *whether* it goes in — **where**. This project predicts which of nine zones of the goal a penalty taker will aim at, using StatsBomb open event data, and serves the prediction as a Dash app built for the goalkeeper's side of the problem.

The short version: match context turned out to be almost useless, and the taker's own habit was the only thing carrying signal. The app is designed around that finding rather than around the model I started with.

## Data

- **1,361 real penalties** from StatsBomb open data — 881 matches, 710 takers, 73.3% scored
- Shot end location (`end_y`, `end_z`) discretised into a 3×3 grid: `TL TC TR / ML MC MR / BL BC BR`
- A **pressure index (0–10)** derived from match time, how close the score is, and whether it is a shootout
- Context columns the event data doesn't carry (match time, score difference, shootout flag, home/away) were generated synthetically for a second set of 1,361 rows, giving **2,722 training rows**. This is a real limitation — see *Limitations* below.

## Model

XGBoost, 9-class `multi:softprob`, 300 trees, depth 5. Features: `foot_enc`, `match_time`, `score_diff`, `is_shootout`, `home_away`, `pressure_index`. Evaluated on a stratified 20% holdout with accuracy and multiclass log loss.

## What I found

**1. Match context does not predict the zone.** The model reaches **13.8%** accuracy on held-out data. Random guessing across nine zones is 11.1%, and always guessing the most common zone (bottom-left) gives 27.7%. The model is worse than a constant. Log loss 2.48.

**2. The taker's own history does carry signal — weakly.** Leave-one-out over the 46 takers with 5 or more penalties (408 kicks) gives **32.6%** top-1, against 29.9% for the majority zone on the same subset. Small, but it is the only thing that beat the baseline.

**3. Takers lean left.** 46.2% of penalties end in the left third of the goal, 37.4% right, 16.4% centre — from the goalkeeper's perspective this alone is worth more than the context model.

## What the app does

Because context is uninformative and player history is thin, the app never relies on one of them alone. Three modes:

| Mode | Behaviour |
|---|---|
| **Global** | The XGBoost model only |
| **Player** | The taker's historical zone distribution only (falls back to Global if unknown) |
| **Auto** | Blends the two, weighted by how many penalties that taker has |

The blend weight is `α = 1 − e^(−k/12)`, capped at 0.80, where `k` is the number of penalties on record for that taker. A player with two kicks is served the global distribution; a player with thirty is served mostly their own habit. Nobody gets a confident prediction built on three data points.

Priors are kept for the 175 takers with at least 5 penalties on record.

## Limitations and next steps

- **Half the training rows have synthetic context.** The 13.8% result says context-as-reconstructed doesn't predict the zone; it is not yet a clean verdict on real match context. Re-running this with real context from full match data is the obvious next step.
- **Nine classes on 2,722 rows is data-poor.** Collapsing to left / centre / right (3 classes) would likely give a usable model — and for a goalkeeper, that is arguably the decision that actually matters.
- **No goalkeeper features.** Keeper height, dive tendency and reputation are all plausibly informative and entirely absent here.

## Run it

```bash
pip install -r requirements.txt
python data_pipeline.py        # builds data/hybrid_penalties.csv
python train_zone_model.py     # trains and saves models/
python app.py                  # http://localhost:8050
```

## About

Team project for my data science course. I was the team leader, responsible for the data pipeline, the model and the app structure. The notebooks under `test/` show the exploration the final scripts came out of.
