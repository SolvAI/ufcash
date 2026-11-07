#!/usr/bin/env bash
# Full pipeline. ./run_pipeline.sh --refresh  re-downloads all public data sources first.
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" == "--refresh" ]]; then
  mkdir -p data/raw/greco data/raw/ultimate data/raw/nlbair data/raw/canon
  for f in ufc_event_details ufc_fight_details ufc_fight_results ufc_fight_stats ufc_fighter_details ufc_fighter_tott; do
    curl -sfL -o data/raw/greco/$f.csv https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/$f.csv
  done
  curl -sfL -o data/raw/ultimate/ufc-master.csv https://raw.githubusercontent.com/shortlikeafox/ultimate_ufc_dataset/master/ufc-master.csv
  curl -sfL -o data/raw/nlbair/ufc_all_fights_wide.parquet https://raw.githubusercontent.com/nlbair/ufc-data/main/data/ufc_all_fights_wide.parquet
  curl -sfL -o data/raw/canon/canon.db https://github.com/jadevit/fight-canon-db/releases/latest/download/canon.db
fi
python src/build_data.py      # ufcstats -> tidy fights / per-fighter stats (+ data/upcoming.csv)
python src/odds.py            # closing/opening lines + rankings from 3 sources
python src/canon_feats.py     # full pro-career features + world Elo (all promotions)
python src/features.py        # leakage-free pre-fight features
if [[ "${SKIP_EVAL:-0}" != "1" ]]; then python src/evaluate.py; fi   # walk-forward backtest -> results/
python src/train_final.py     # fit on everything -> models/final.joblib
python src/predict.py         # predictions for data/upcoming.csv
