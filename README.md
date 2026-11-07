# UFCash

**Can a model built by an LLM agent beat the UFC betting market?**

At picking winners, only just. On the 2024–2026 hold-out (1,501 fights) it picks 70.2 % of winners, the same as the closing line, with a
slightly better log-loss (0.582 vs 0.584). Betting its picks at closing prices breaks even (+0.7 %). The one rule that made money on paper is
**value betting at the opening line: +24 % on 226 hold-out bets (95 % CI +11 % to +37 %), positive in all 11 years**, and +11 % if the bet is
only filled halfway to the closing price. Opening lines are one bookmaker's first price, often with low limits and moved within minutes, so
whether that edge can actually be collected is unproven.

This is research code, not betting advice. Nothing here was bet with real money.

UFCash predicts the winner of UFC bouts from leakage-free pre-fight information: each fighter's full UFC
round-by-round history, full pro record across all promotions, rating systems, physical data, official
rankings, and the betting market. `models/final.joblib` is the final model, trained on every fight up to 2026-10-31.

## Result

Walk-forward backtest: every January 1 the model is retrained on all earlier fights and then predicts
that whole year blind. All model and feature choices were made on 2016–2023. 2024–2026 is the hold-out.

| Hold-out: 1,501 UFC fights, 2024-01-13 → 2026-10-31 | Accuracy | Log-loss |
|---|---|---|
| Always pick the red corner | 55.2 % | – |
| World Elo (all ~373k pro MMA fights) | 56.8 % | 0.678 |
| **Stats model only** (no odds) | **66.0 %** | 0.612 |
| Betting market (closing line, margin removed) | 70.2 % | 0.584 |
| **Final model** (stats ensemble + market) | **70.2 %** (70.4 % on the 1,488 fights with odds) | **0.582** |

Dev period 2016–2023 (3,855 fights): final 66.4 %, market 65.8 %, stats-only 64.2 %.
The final model beat the closing line in 7 of 11 years, and its log-loss beats the market in both periods.
It is calibrated: picks made at 70–75 % confidence won 73.1 % of the time in the hold-out
(see `results/metrics.json`).

**Honest caveats**
- The 95 % confidence interval on 1,501 fights is about ±2.3 points. The edge over the closing line is real in
  log-loss on both dev and test, but in accuracy it is small (≈ +0.2 points on test, +0.6 on dev).
  Almost all of the "70 %" comes from the betting market. The model adds a little on top of it.
- **The hold-out period was kind to favourites.** In 2024–26 the betting favourite won 70.2 % of fights, while the
  odds themselves implied 67.2 % (z = 2.5). In 2016–23 it won 65.8 % vs 64.7 % implied. If that is luck rather than a
  lasting change, expect roughly **67–69 %** going forward, not 70 %.
- Without odds the model scores 66 % on recent fights, which is the fair number to compare with stats-only
  projects.

## Verification (`src/audit.py`)

| Test | What it would catch | Result |
|---|---|---|
| `scramble <date>`: randomise every result, stat, odds line and pro-record result on/after a cutoff (1,304 UFC bouts and 39,555 pro fights for 2024-06-01; 4,029 and 124,573 for 2019-01-19), rebuild all features from scratch | any feature that sees the result of the bout itself or a later one | 0 of 461 feature columns changed for any bout on/before either cutoff (incl. the cutoff night); 345 changed after it |
| `null`: same pipeline on labels shuffled within each year | accuracy created by the pipeline itself | 53.8–54.9 % (always-red on the same labels: 55.2 %) |
| `static`: Sherdog vs ufcstats date for all 8,974 linked bouts | a pro-record update dated before the UFC bout | 0 dated earlier |
| `static`: official ranking boards | rankings published after the fight | all boards dated before the event |
| `static`: three odds sources | mis-joined or post-fight odds | correlation ≥ 0.99, favourite accuracy 69.5–69.9 % on the same fights |
| `static`: missing values and duplicate bouts | blanks that mark short careers, the same fight counted twice | no outcome-revealing missingness; the only repeated pairings are real rematches after a no-contest |

The audit found one weak spot, now fixed. Imputed reach values had decimals, while real ones are whole inches, so a model
could in principle spot "imputed = short career". They are now rounded. Log-loss was unchanged (0.5816), so the
model was not exploiting it. Accuracy moved by 2 fights out of 1,501.

## Can it reach 75 %?

Not on all fights, and no honest public-data model will. On the hold-out the model needs 69 more correct picks to reach 75 %.
It only picks against the betting favourite in 55 fights, so even winning every disagreement falls short. Reaching
75 % would mean overruling the closing line in hundreds of fights and being right most of the time, which would be a
large betting edge.

Tried after the first version, all evaluated on 2016–23 only, none moved dev accuracy or log-loss beyond noise:
- pre-UFC round stats for debutants from 7,100 non-UFC fights (DWCS, PRIDE, PFL, ACA, Cage Warriors, Oktagon, ...)
- media scorecards (deserved record, decision luck) and judges' round share (`src/canon_extra.py`, `src/exp6_newinfo.py`)
- recency-weighted and opening-line stackers

None of these features is significant beyond the closing line once multiple testing is accounted for.

75 % is reachable if the model may **skip fights**. With a confidence threshold of 64.5 % chosen on 2016–23, the hold-out
accuracy was **76.5 % ± 2.7 on 62 % of fights** (80.8 % on 41 % at a 70.5 % threshold). This is mostly the easy-favourite
effect: the bookmaker's favourite alone, with the same rule, got 78.1 % on 57 % of fights.

## Betting backtest (`src/betting.py`)

Real BestFightOdds prices, bookmaker margin included (about 4–4.5 % at an average book). Walk-forward probabilities,
strategy thresholds read off 2016–23, results for 2024–26 out of sample. ROI = profit / amount staked.

| Strategy (2024–26 hold-out) | Bets | ROI [95 % CI] |
|---|---|---|
| Bet the model's pick in every fight, average closing book | 1,483 | +0.7 % [−2.8, +4.2] |
| Same, stake growing with confidence | 1,483 | +0.4 % [−2.7, +3.2] |
| Only when the model picks the underdog | 55 | +5.7 % [−23, +32] |
| Value bets at the close (edge > 5 %), average book | 168 | +10.1 % [−8, +28] |
| **Value bets at the opening line (edge > 10 %), opening-line model** | **226** | **+24.1 % [+11, +37]** |
| … filled only halfway from the opening to the closing price | 226 | +11.2 % |

- Accuracy is not profit. Picks and confidence are already priced in, so betting the 70 % model's picks at closing prices breaks even before the margin.
- Closing-line value bets were profitable in 2016–23, but on the hold-out that result is not significant. It loses money without its 10 biggest wins, and almost all of it comes from 2025.
- The opening-line model (the same stacker refit with the opening line as its only market input) was positive in all 11 years. Its bets
  beat the closing line by +9 % (dev) and +15 % (test) of expected value, which is a much less noisy signal of a real edge than results alone.
  With a $10,000 bankroll, quarter-Kelly stakes, a $500 maximum bet and halfway fills, it made +$10.7k over 2024–26 (max drawdown 15 %).
- A bet-time rule skips any edge above 30 %. Such lines are almost always bad data (for example, openers posted with the names swapped) and would be voided.
- Unproven in practice: BFO's opening price is one book's first line, often with low limits and moved within minutes. Winning accounts
  get limited. Paper-trade first, record the price you actually got, and track closing-line value.

### Underdog specialist trained for profit (`src/underdog.py`)

Six models were trained only on underdog bets, refit every year. Some are probability models (market bias only, a small
logistic regression, LightGBM boosted from the price). Others optimise money directly (expected-profit regression, a
classifier with wins weighted by the payout, and a Kelly-growth stake policy).

- Betting every underdog loses: −9 % (2016–23) and −15 % (2024–26). Longshots are overbet.
- At the close, the best model on 2016–23 made +15 %, then +5 % on 2024–26 [−21, +33]. All results lose money without their 10 biggest wins.
- In 40 simulated worlds with no edge at all, picking the best of these models on 2016–23 still shows +9 % (median) and +23 % (95th percentile).
  The real +15 % is within that range, so it is selection luck.
- The profit-objective models did worse than the probability models. Fed only payouts, the Kelly policy learned to stop betting underdogs from 2020 on.
- At the opening line, none of the specialists beat the general opening-line model restricted to underdogs (2024–26: +28 %, 89 bets).
  That is a subset of the opening-line strategy above, not a new edge.

### Fighter profiles and matchups (`src/exp7_profile.py`, 2016–23 only)

- Data: Wikidata entries for 2,609 UFC fighters, matched by Sherdog id. Fields: citizenship, birthplace, martial-arts background, education.
- Trap: the snapshot is written after the fact. When only the red corner has an entry, red wins 80 % against 62 % implied, because entries
  exist for fighters who went on to win. Entry existence and completeness are never used, and the test runs only on fights where both fighters have one.
- Region, home country, background and education change log-loss by −0.0007 [−0.0037, +0.0021], and accuracy drops 0.656 → 0.653. No gain.
- Former-USSR fighters beat their closing odds (59–62 % wins against 56–57 % implied). This also holds with a definition built without Wikidata,
  from where their earlier fights took place. Betting them blindly still returns only −0.2 % to +3 % ± 9–14 % after margin.
- Style-vs-style matchups (8 k-means styles) change log-loss by −0.0004 [−0.0007, +0.0001]. A fighter's own record against a given style adds nothing:
  there are too few fights per fighter per style. The stats model already has style-matchup terms.

## Why this is more trustworthy than the reference repos

The rezan21 / Kaggle `data.csv` projects report about 67–70 %, but much of that comes from leakage:

1. **Corner leakage.** Before 2010, ufcstats lists the **winner first**, so "red" wins 100 % of pre-2010 fights.
   The old Kaggle baseline "67 % red wins" is mostly this artifact. Here, corner is only used as a feature
   from 2010 onward. Since then, red wins about 57 % of fights.
2. **Missing-value leakage.** ufcstats leaves reach, date of birth, and stance blank mostly for fighters
   who had short careers. In 2010+ fights, a fighter with no listed reach won only **20 %** of the time.
   The blank is effectively a label. It is hidden with deterministic stochastic imputation (`features.py: impute_physical`).
3. **Random train/test splits** let a model learn from a fighter's future. Here every feature uses only
   fights on dates strictly before the bout (same-night tournament fights included), every model is
   evaluated walk-forward, and both corner orientations are trained and averaged so the model is symmetric.

## Data (all public, refreshed by `./run_pipeline.sh --refresh`)

| Source | What | License |
|---|---|---|
| [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats) | ufcstats.com mirror: events, results, round-by-round stats, fighter attributes (1994–2026-10-31) | GPL-3.0 |
| [Jadevit/fight-canon-db](https://github.com/Jadevit/fight-canon-db) | 373k pro MMA fights (Sherdog records, all promotions), BestFightOdds lines 2007+ | MIT |
| [nlbair/ufc-data](https://github.com/nlbair/ufc-data) (Bair 2026, doi:10.5281/zenodo.23458636) | BestFightOdds opening + closing ranges, official rankings at fight time (2013–2026-10) | CC BY 4.0 |
| [shortlikeafox/ultimate_ufc_dataset](https://github.com/shortlikeafox/ultimate_ufc_dataset) | second odds source and method-of-victory props (2010–2026-03) | Apache-2.0 |

Underlying facts come from ufcstats.com, Sherdog, BestFightOdds, MMA Decisions, and martj42/ufc-rankings.

## Features (~470 columns per bout, all computed pre-fight)

- **UFC record and form:** W/L by method, streaks, last-3/5 form, finishes and being finished, KO losses in the last 2 years, title and 5-round experience, layoff, activity, weight-class changes.
- **Striking and grappling per minute**, career totals and recency-weighted, shrunk toward population priors:
  accuracy/defence, head/body/leg and distance/clinch/ground mix, takedowns, control, submissions, knockdowns.
- **Opponent-adjusted output:** what the fighter landed or absorbed compared with what each opponent normally
  allowed or landed before that fight.
- **Round splits:** round-1 vs. late-round output (cardio/fade) and accumulated head-strike damage.
- **Rating systems:** Elo (several K values), margin-of-victory Elo, Glicko, and "skill Elos" scored on
  strike share, grappling share, and damage share, plus a **world Elo over all ~373k pro MMA fights**.
- **Full pro record (Sherdog):** pro fights, opponent quality, best wins, pre-UFC record, home-country share, head-to-head history.
- **Market history:** how the closing line rated the fighter in past fights and how they did against that expectation.
- **Physical, style, and bout context:** age, height, reach, ape index, stance matchup, style-matchup products, rankings, card position, rounds, title, gender, division.
- **Market:** closing line (3 sources), opening line, line movement, book disagreement.

## Model (`src/final_model.py`)

1. **Stats ensemble (no odds):** LightGBM ×2, XGBoost, CatBoost, and L2 logistic regression, trained on
   both corner orientations since 2001. Output: mean logit.
2. **Market stacker:** average of (a) logistic regression on [logit closing line, stats logit], fit on
   out-of-fold stats predictions, and (b) a small LightGBM boosted from the closing-line logit that learns
   systematic market errors (mostly age, corner, damage absorbed, opponent quality). Fights without odds fall back to the stats ensemble.

## Usage

```bash
pip install -r requirements.txt
./run_pipeline.sh --refresh    # download data, build features, backtest, train, predict
# or, with data already present:
SKIP_EVAL=1 ./run_pipeline.sh  # rebuild + train + predict only
cd src && python audit.py static && python audit.py null && python audit.py scramble 2024-06-01   # leakage audit
```

To predict a card, edit `data/upcoming.csv` (fighter names as on ufcstats, plus American moneylines if you
have them) and run the pipeline. Predictions go to `results/upcoming_predictions.csv`. The rows shipped
there are hypothetical demo matchups. Odds matter a lot: without them the model falls back to its 66 % stats-only mode.

## Layout

```
src/build_data.py     ufcstats mirror -> data/processed/{fights,fighter_fights,fighters}.parquet
src/odds.py           odds + rankings from 3 sources, oriented to the red corner
src/canon_feats.py    pro-career features + world Elo from fight-canon-db
src/features.py       chronological pre-fight feature builder
src/common.py         symmetric design matrices, metrics
src/final_model.py    the final model
src/evaluate.py       walk-forward backtest -> results/metrics.json, results/walkforward_predictions.csv
src/train_final.py    fit on all fights -> models/final.joblib
src/predict.py        predict data/upcoming.csv
src/betting.py        betting backtest vs real prices -> results/betting_report.json
src/underdog.py       underdog-only models trained for profit, with a no-edge null -> results/underdog_report.json
src/exp7_profile.py   Wikidata profiles + style matchups (dev only; no gain) -> results/profile_run.log
src/audit.py          leakage audit (future-scramble rebuild, label-permutation null, source checks)
src/canon_extra.py    pre-UFC stats / media & judge scorecards (tried, not used: no gain beyond the market)
src/exp*.py, walk.py  experiment log (model zoo, stacking variants, new-information tests) kept for reference
```

`results/walkforward_predictions.csv` ships only the model's own out-of-sample probabilities (`stats_p_red`, `final_p_red`) for
5,356 fights 2016–2026; the market probability and the result are removed (`src/evaluate.py` rebuilds the full file from the data).

## How this was made

The research was carried out by an LLM agent (Claude, Anthropic) working in a terminal. SolvAI set the goal and the hold-out rule and asked
for the follow-ups (verify the result is not inflated, try for 75 %, betting, underdog models, profiles). Data search, pipeline, models,
audit and write-up are the agent's. The follow-up project on French horse racing, with about 20 times more events, is
[SolvAI/turfolio](https://github.com/SolvAI/turfolio).

## License

Code and model weights: MIT (see `LICENSE`). **No data is included**: `./run_pipeline.sh --refresh` downloads it from the sources listed
under Data, each under its own license.
