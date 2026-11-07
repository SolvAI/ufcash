<div align="center">

# 🥊 UFCash

### A UFC model that beat the bookmakers' opening lines: +24 % ROI on 226 bets it had never seen.

**+24 % ROI out of sample (95 % CI +11 % to +37 %)** · **positive in 11 of 11 years** · **70.2 % winner accuracy, level with Vegas' closing line**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab)
![Platforms](https://img.shields.io/badge/runs%20on-Windows%20%7C%20macOS%20%7C%20Linux-555)
![Hold-out ROI](https://img.shields.io/badge/hold--out%20ROI-%2B24%25-2ea043)
![Leakage audit](https://img.shields.io/badge/leakage%20audit-passed-2ea043)

**One line to install. Put in the card, get the picks and the value bets.**

</div>

| Value bets at the opening line (edge > 10 %) | Bets | Won | ROI | 95 % CI |
|---|---:|---:|---:|---|
| 2016–2023 (where the 10 % threshold was chosen) | 580 | 59 % | +22.5 % | +14.5 % to +30.9 % |
| **2024 → Oct 2026 hold-out** | **226** | **58 %** | **+24.1 %** | **+11.0 % to +37.2 %** |
| Hold-out with each bet filled only halfway from the opening to the closing price | 226 | | +11.2 % | |

The $10,000 paper-bankroll version (quarter Kelly, $500 max bet, halfway fills) ends 2024–2026 at **+$10.7k**, with a worst drawdown of 15 %.
Its bets also beat the closing line by 15 % of expected value, which is the signal professional bettors trust more than results alone.

## Try it

**macOS / Linux**
```sh
curl -LsSf https://raw.githubusercontent.com/SolvAI/ufcash/main/install.sh | sh
cd ufcash && ./ufcash card
```

**Windows** (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/SolvAI/ufcash/main/install.ps1 | iex"
cd ufcash; .\ufcash card
```

The installer brings its own Python and the trained model, then downloads the public UFC data (about 1 minute). Put your card in
`data/upcoming.csv`: fighters as named on ufcstats.com, plus the American moneylines you can bet right now. Then run `ufcash card`:

```text
UFCash card  (model trained on every UFC fight up to 2026-10-31)

  2026-12-12  Marvin Vettori vs Roman Kopylov  (Middleweight)
      pick Marvin Vettori 54%   red win: model 54.1%, stats-only 53.2%, market 52.2%
  2026-12-12  Payton Talbott vs Marcus McGhee  (Bantamweight)
      pick Payton Talbott 67%   red win: model 66.7%, stats-only 81.5%, market 57.4%
      VALUE BET  Payton Talbott at 1.67  edge +19%  stake 5.0% of bankroll (quarter Kelly)
  2026-12-12  Johnny Walker vs Anthony Wint  (Heavyweight)
      pick Anthony Wint 61%   red win: model 38.8%, stats-only 45.4%, market 38.6%

1 value bet. Saved to results/upcoming_predictions.csv
```
<sub>Real output for the demo card that ships in `data/upcoming.csv` (hypothetical matchups and odds: replace them with a real card).</sub>

Run `ufcash setup` again before each event to pull the latest results.

## Why it works

- **It knows every fighter's whole career.** About 470 pre-fight features: round-by-round UFC stats adjusted for opponent, the full
  pro record across all promotions (373k fights), several Elo and Glicko ratings, physicals, rankings and market history.
- **It's built like an audit, not a Kaggle notebook.** Public UFC models that report 67–70 % mostly leak the answer: before 2010
  ufcstats lists the winner first, blank reach marks short careers, and random splits let the model see a fighter's future. Here
  every feature uses only fights strictly before the bout, every number is walk-forward, and an audit rebuilds every feature with
  the future scrambled. It changed 0 of 461 columns.
- **It strikes early in the week.** Opening lines are one bookmaker's first guess. The opening-line model blends that price with the
  stats model (weights 0.71 / 0.60), and it bets when the two disagree by more than 10 %. Edges above 30 % are skipped, because they
  are almost always a bad line, such as names swapped.

---

## ⚠️ Read this before you bet a real dollar

- **Paper results only.** Nothing here was bet with real money.
- **Opening lines are hard to get.** The opening price is one book's first line. It often has low limits and moves within minutes.
  If you get only halfway to the closing price, the edge falls from +24 % to +11 %. Winning accounts get limited.
- **Small samples.** That's 226 hold-out bets, or about 80 a year. The lower end of the interval (+11 %) is still positive, but
  luck is a big part of any single year.
- **Closing-line bets don't pay.** Betting the model's picks at the close breaks even (+0.7 %, 1,483 bets), and value bets at the
  close (+10 %) are not significant. The edge is at the open.
- Paper-trade first, record the price you actually got, and track your closing-line value.

## The full record

Walk-forward: every January 1 the model is retrained on all earlier fights, then it predicts that whole year blind. All model and
feature choices were made on 2016–2023, and 2024–2026 is the hold-out.

| Hold-out: 1,501 UFC fights, 2024-01-13 → 2026-10-31 | Accuracy | Log-loss |
|---|---|---|
| Always pick the red corner | 55.2 % | – |
| World Elo (all ~373k pro MMA fights) | 56.8 % | 0.678 |
| Stats model only (no odds) | 66.0 % | 0.612 |
| Betting market (closing line, margin removed) | 70.2 % | 0.584 |
| **Final model** (stats ensemble + market) | **70.2 %** | **0.582** |

- The model beats the closing line's log-loss on both dev and hold-out, and it beat the closing line in 7 of 11 years.
- It is calibrated: picks made at 70–75 % confidence won 73.1 % of the time.
- The hold-out was kind to favourites. They won 70.2 % while their odds implied 67.2 %, so expect about 67–69 % accuracy going forward.
- **75 % accuracy?** Not on every fight: no public-data model gets there, since the model disagrees with the market on only 55 of 1,501
  fights. If it may skip fights, it gets 76.5 % on 62 % of them, but the favourite alone does as well on those.

| Betting strategy (2024–26 hold-out, real BestFightOdds prices) | Bets | ROI [95 % CI] |
|---|---:|---|
| Bet the model's pick in every fight, average closing book | 1,483 | +0.7 % [−2.8, +4.2] |
| Value bets at the close (edge > 5 %) | 168 | +10.1 % [−8, +28] |
| **Value bets at the opening line (edge > 10 %)** | **226** | **+24.1 % [+11, +37]** |

<details>
<summary><b>Leakage audit (<code>src/audit.py</code>)</b></summary>

| Test | What it would catch | Result |
|---|---|---|
| `scramble <date>`: randomise every result, stat, odds line and pro-record result on/after a cutoff, rebuild all features | any feature that sees the result of the bout itself or a later one | 0 of 461 feature columns changed for any bout on/before either cutoff (2019-01-19, 2024-06-01) |
| `null`: same pipeline on labels shuffled within each year | accuracy created by the pipeline itself | 53.8–54.9 % (always-red on the same labels: 55.2 %) |
| `static`: Sherdog vs ufcstats dates for 8,974 linked bouts | a pro-record update dated before the UFC bout | 0 dated earlier |
| `static`: ranking boards, three odds sources, missing values, duplicates | post-fight rankings, mis-joined odds, outcome-revealing blanks | all clean (odds sources agree at r ≥ 0.99) |

</details>

<details>
<summary><b>What did not work</b></summary>

- **Underdog specialists trained for profit**: in 40 simulated no-edge worlds, picking the best of them shows +9 % from selection alone. Their +15 % is luck.
- **More information**: pre-UFC round stats, media and judges' scorecards, Wikidata profiles, and style matchups. None beat the closing line beyond noise.
- **Former-USSR fighters** beat their odds by about 4 points, but after the margin that's only about 0–3 %.

</details>

## For developers

```
ufcash.py             command line: setup (download + features), card (predict data/upcoming.csv)
src/build_data.py     ufcstats mirror -> fights, per-fighter stats (+ data/upcoming.csv)
src/odds.py           closing/opening lines + rankings from 3 sources
src/canon_feats.py    pro-career features + world Elo from fight-canon-db
src/features.py       leakage-free pre-fight features (~470 columns)
src/final_model.py    stats ensemble (LightGBM x2, XGBoost, CatBoost, logistic) + market stacker
src/train_final.py    fit on all fights -> models/final.joblib (closing-line and opening-line stackers)
src/predict.py        picks + value bets for data/upcoming.csv
src/evaluate.py       walk-forward backtest -> results/metrics.json
src/betting.py        betting backtest vs real prices -> results/betting_report.json
src/audit.py          leakage audit
```

`./run_pipeline.sh --refresh` reruns everything, including the backtests. `results/walkforward_predictions.csv` holds the model's
out-of-sample probabilities for 5,356 fights 2016–2026, with the odds and results removed.

Sibling project: [Turfolio](https://github.com/SolvAI/turfolio), a French horse-racing model with +4.8 % on 1,069 out-of-sample place bets.

## Data and license

Code and model weights are MIT (see `LICENSE`). **No data is included**: the installer downloads it from these public sources, each under its own license.

| Source | What | License |
|---|---|---|
| [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats) | ufcstats.com mirror: events, results, round-by-round stats, fighter attributes | GPL-3.0 |
| [Jadevit/fight-canon-db](https://github.com/Jadevit/fight-canon-db) | 373k pro MMA fights (Sherdog records), BestFightOdds lines 2007+ | MIT |
| [nlbair/ufc-data](https://github.com/nlbair/ufc-data) (Bair 2026, doi:10.5281/zenodo.23458636) | BestFightOdds opening + closing ranges, official rankings | CC BY 4.0 |
| [shortlikeafox/ultimate_ufc_dataset](https://github.com/shortlikeafox/ultimate_ufc_dataset) | second odds source and method-of-victory props | Apache-2.0 |

Not affiliated with the UFC or any bookmaker. Not betting advice.
