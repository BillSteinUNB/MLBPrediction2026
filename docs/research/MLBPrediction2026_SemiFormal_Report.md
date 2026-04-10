# MLBPrediction2026: A Semi-Formal Report on a Machine Learning Betting System

## 1. Introduction

This project is an end-to-end machine learning system for Major League Baseball betting research, with a particular emphasis on forecasting game outcomes, run distributions, and bet value in a way that is usable for real decision support. The broader goal is not simply to classify winners and losers, but to estimate probabilities, compare those probabilities to sportsbook prices, and identify positive expected value betting opportunities.

The project has two practical objectives:

1. Build a strong predictive model for baseball games using historical performance, contextual baseball features, and live market data.
2. Convert those predictions into a disciplined betting framework that can be evaluated by paper-trading style results, accuracy, and return on investment.

Unlike many classroom machine learning projects that stop at classification accuracy, this system attempts to bridge prediction and decision-making. In this project, a model is only useful if its probabilities are good enough to produce actionable betting edges.

## 2. Problem Framing

The system evaluates baseball games at the market level rather than only at the game result level. For each game, the model can form opinions on:

- Moneyline
- Run line
- Game total
- First five inning variants when supported

This means the problem is best viewed as a probabilistic prediction task followed by an optimization task. The machine learning model produces estimated probabilities and run expectations, and then the betting layer compares those model estimates against sportsbook prices after adjusting for vig.

The practical challenge is that a model can look statistically competent while still failing as a betting tool. Because of that, this project uses both modeling metrics and betting performance metrics.

## 3. System Architecture

The project is structured as a complete pipeline:

1. Pull schedule, odds, weather, and lineup information.
2. Construct sabermetric and contextual features.
3. Generate model predictions for runs and win probabilities.
4. Compare model probabilities to sportsbook prices.
5. Compute expected edge and size a bet using constrained Kelly-style sizing.
6. Track the pick, freeze it, and settle it later using actual game results.

The system includes both a backend pipeline and a frontend dashboard. The new live-season integration now allows a user to pull the current slate, freeze picks so they cannot be rewritten later, and track performance over time. This is important because it prevents hindsight bias and ensures the model is evaluated fairly.

## 4. Machine Learning Approach

The modeling framework is not a single basic classifier. It is a layered baseball prediction system that combines several ideas:

- gradient boosted trees for core prediction
- stacked modeling for combining model outputs
- calibration for turning raw model scores into better probabilities
- run-distribution modeling
- MCMC-based refinement for a stronger distribution layer

In practical terms, the model does not simply say “team A wins.” It estimates distributions and probabilities that can be translated into:

- full-game moneyline probability
- full-game run line probability
- full-game total opinion
- first-five market opinions when data quality is good enough

The system also uses baseball-specific feature engineering. These features include offensive and pitching indicators, bullpen context, weather adjustments, lineup quality, and other sabermetric inputs. This gives the model a more realistic representation of baseball than a simple wins/losses or team-average approach.

### 4.1 Base Model Training

The first major predictive layer is the XGBoost training pipeline in [xgboost_trainer.py](C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/model/xgboost_trainer.py). The project does not use a trivial train/test split. Instead, it uses temporal cross-validation so that earlier games train the model and later games evaluate it. That is an especially important design choice in sports analytics, because a random split would leak future baseball conditions into the training sample. A random split would let the model train on games from later in a season and then “predict” games that occurred earlier, which is unrealistic in deployment.

The XGBoost training configuration includes a broad but sensible hyperparameter search:

- tree depth
- number of estimators
- learning rate
- row subsampling
- column subsampling
- minimum child weight
- gamma
- alpha regularization
- lambda regularization

These choices are appropriate for baseball because the input space is heavily non-linear. The relationship between weather and totals, for example, is not the same across all parks. A park with a strong home-run factor may respond to wind or air density very differently than a pitcher-friendly park. Similarly, bullpen fatigue may matter much more in a game projected to remain close than in a matchup where one team is likely to win comfortably. Tree ensembles handle these conditional interactions well.

Another positive feature of the training flow is that it supports Optuna-based search and early stopping. This is useful because it reduces the likelihood that the model simply memorizes noise in a high-dimensional feature set. A baseball model with many handcrafted features can easily overfit if no effort is made to constrain its complexity.

### 4.2 Stacking and Meta-Learning

After the XGBoost models are trained, the project passes their probabilities into a stacking layer implemented in [stacking.py](C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/model/stacking.py). The stacking layer uses logistic regression as a meta-learner. This is an elegant compromise between flexibility and control.

The base XGBoost model can learn rich, non-linear structure. However, raw tree probabilities are not always the best-calibrated probabilities. Instead of assuming that the first-layer probability is final, the project builds a second layer that takes:

- out-of-fold XGBoost probabilities
- a small set of raw meta features
- baseline team-strength style variables

and then learns how to combine them into a better probability estimate.

This is an important difference between a “high-scoring classifier” and a more complete probabilistic system. The XGBoost model answers the question:

“Given the engineered feature set, what is the nonlinear mapping to outcome probability?”

The stacking model answers the question:

“Given what the XGBoost model believes, and given a few stable baseball priors, how should I adjust that raw probability?”

This is especially helpful in sports, where the model may be directionally correct but a little too strong or too weak in certain regions of the probability space. A simple logistic meta-layer can correct some of those distortions without needing a second highly complex model.

### 4.3 Calibration and Probability Honesty

The next layer is calibration in [calibration.py](C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/model/calibration.py). This is one of the most sophisticated and important aspects of the system. In many ordinary predictive settings, calibration is a “nice to have.” In betting, calibration is indispensable.

If a sportsbook line implies `52.4%` probability after vig, and the model believes the event is `55%`, then the edge is small and should be treated cautiously. If the model says the event is `65%` when it is actually only `55%`, the staking system will behave recklessly. This is why the project supports several calibration options:

- identity, which accepts the stacked probability as-is
- isotonic regression, which is flexible and non-parametric
- Platt scaling, which is logistic
- blend, which averages isotonic and Platt behavior

The code also includes explicit quality gates such as:

- Brier score target
- Expected Calibration Error target
- reliability gap target

This is a strong methodological choice because it forces the project to think about probability quality directly instead of assuming that a good ranking model is automatically suitable for betting.

### 4.4 Run Distributions and MCMC Refinement

A major theme of the project is that baseball totals are not just winner problems. They are distribution problems. For totals, team totals, and run-line pricing, the system needs a believable distribution over possible scoring outcomes. That is why the repository contains a full run-count research lane with files such as:

- [run_distribution_trainer.py](C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/model/run_distribution_trainer.py)
- [run_distribution_metrics.py](C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/model/run_distribution_metrics.py)
- [mcmc_engine.py](C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/model/mcmc_engine.py)
- [mcmc_pricing.py](C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/model/mcmc_pricing.py)
- [score_pricing.py](C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/model/score_pricing.py)

The value of this approach is that it tries to map from baseball context to a distribution of runs, not only to a binary result. This is essential if the model is expected to say meaningful things about:

- Over or under `8.5`
- probability of covering `-1.5`
- probability that the away team scores at least five runs

The MCMC refinement layer effectively acts as a second-stage distributional correction process. The project’s recent internal terminology referred to Stage 3 and Stage 4. Stage 3 is the core distribution layer, while Stage 4 refines it further. This has allowed the project to evaluate not only point behavior but also distribution sharpness and calibration under metrics such as CRPS and negative log score.

### 4.5 Why the Modeling Stack Is Appropriate

The overall stack makes sense for this domain because each stage addresses a different weakness:

- XGBoost captures rich nonlinear baseball relationships.
- Stacking makes the output more robust and less dependent on one model family.
- Calibration makes the probabilities more trustworthy for market comparison.
- Distribution and MCMC layers extend the system from classification toward pricing.

That is the right progression for a sports-betting model. A flat, single-stage classifier would be much simpler, but it would also be much less useful for pricing multiple betting markets.

## 5. Data and Features

The project integrates several types of data:

- historical MLB game data
- odds and sportsbook pricing data
- lineup availability
- weather and stadium context
- baseball performance statistics used to engineer team and player features

An important part of the project has been improving the historical odds database. During testing, malformed or inconsistent sportsbook rows caused unrealistic return on investment results. This led to a significant debugging effort around the OddsPortal and live-odds paths. As a result, the model now relies on a much cleaner “working” historical database for 2021, 2023, 2024, and 2025, with unreliable markets excluded when necessary.

This cleanup mattered because a betting model cannot be judged fairly if the line data itself is wrong. In other words, data engineering was just as important as the machine learning.

### 5.1 The Training Matrix and Rolling Windows

The project’s feature-building pipeline is centered in [data_builder.py](C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/model/data_builder.py). This file is important because it shows that the training process is not based on a static table downloaded from one source. Instead, the dataset is assembled from many moving parts and aligned carefully in time.

The builder uses several standard windows, especially:

- `7` games
- `14` games
- `30` games
- `60` games

These windows are important because baseball is a sport of both signal and noise. Very short samples can capture genuine current form or recent fatigue, but they can also be unstable. Longer samples are more stable but less sensitive to real changes in roster health, bullpen availability, or current offensive shape. By including multiple windows, the model can learn when short-term signals matter and when longer-term baselines are more trustworthy.

### 5.2 Offensive Feature Engineering

Offensive features are one of the richest families in the project. They are built primarily in [offense.py](C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/features/offense.py), and the defaults visible in the data-builder layer show the breadth of the offensive feature space. The system tracks metrics such as:

- `wRC+`
- `wOBA`
- `xwOBA`
- gap between observed and expected weighted offense
- isolated power
- barrel rate
- bat speed
- swing length
- swing path tilt
- squared-up rate proxies
- bat-tracking coverage
- BABIP
- strikeout rate
- walk rate

This is a strong feature design because it blends traditional sabermetrics with modern tracking-style indicators. A baseball offense is not well described by batting average or runs per game alone. Teams can reach the same run total through very different underlying processes. One lineup may be strong because it rarely strikes out and grinds long plate appearances. Another may be inconsistent but dangerous because it produces high-end exit velocity and barrel events. Another may be highly dependent on platoon matchups.

By engineering a broader offensive representation, the project gives the model a better chance of understanding not just how productive a team has been, but what type of offense it is.

### 5.3 Pitching Feature Engineering

Pitching features are built in [pitching.py](C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/features/pitching.py). The project’s design clearly recognizes that pitcher quality is not only about season ERA or strikeout totals. Important contextual features include:

- baseline quality measures
- whether the pitcher is acting as an opener
- rest days
- previous pitch count
- cumulative recent pitch load
- workload and durability indicators

This is important because starting pitchers do not enter games in interchangeable condition. A pitcher working on normal rest after a modest pitch count is different from a pitcher who was heavily stressed in the prior start or is being used in an opener/bulk arrangement. Those differences can matter especially in first-five markets and in game-total projections.

### 5.4 Bullpen Feature Engineering

The bullpen system is implemented in [bullpen.py](C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/features/bullpen.py). The project does not treat the bullpen as a single season-long quality number. Instead, it tries to include:

- recent pitch counts
- average rest days
- xFIP
- inherited runner percentage
- high-leverage available reliever counts

This is exactly the right instinct for baseball betting. Bullpen quality is highly state-dependent. A bullpen that is elite in the long run can be weak on a given day if its best relievers are overworked. Conversely, a mediocre bullpen can be in unusually good shape if the key arms are fully rested. Since totals and late-game moneyline behavior depend heavily on bullpen performance, these features are among the most important in the entire system.

### 5.5 Defensive and Framing Features

Defensive features come from [defense.py](C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/features/defense.py). The project includes:

- Defensive Runs Saved
- Outs Above Average
- defensive efficiency
- raw framing
- adjusted framing
- framing retention proxies

Defense is easy to under-model in betting contexts because it is less visible than hitting or starting pitching. However, team defense affects how efficiently balls in play become outs, which directly changes run-scoring distributions. Framing is particularly interesting because it affects called strikes and count leverage. In a sport where marginal strike calls can flip a plate appearance, catcher framing can feed directly into strikeout rate, walk rate, and scoring suppression.

The project’s existing framing layer is already stronger than what many public sports models attempt, and it also aligns naturally with several future research directions.

### 5.6 Baseline Team-Strength Features

The baseline module in [baselines.py](C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/features/baselines.py) contributes stabilizing team-strength features such as:

- Pythagorean win percentage
- Log5-style matchup probabilities
- rolling runs scored
- rolling runs allowed

These are less granular than the Statcast- or player-level features, but they are useful for keeping the model grounded. A heavily engineered model can become too reactive if all inputs are high-variance micro-signals. Baseline team-strength measures provide a macro view of how strong each team has been in a way that is easier to stabilize across windows.

### 5.7 Weather, Park, and Environmental Effects

Weather adjustments and park factors are already an important part of the model. The data builder exposes features such as:

- weather temperature factor
- air-density factor
- humidity factor
- wind factor
- rain risk
- composite weather adjustment
- precipitation probability
- precipitation amount
- cloud cover
- park runs factor
- park home-run factor

This gives the project an environmental awareness that many sports models lack. Baseball is unusually sensitive to park and atmosphere, and the repository’s design reflects that. The weather system is already close in spirit to a physically informed baseball model, even though there is still room to make the Air Density Index concept more explicit and central.

### 5.8 Umpire and Regime Features

Umpire features in [umpires.py](C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/src/features/umpires.py) include:

- home win tendencies by umpire window
- total runs average by umpire
- F5 total runs average by umpire
- ABS-related context
- sample size tracking

The research lane further extends this through “umpire micro” and ABS regime features. This is conceptually strong because MLB rule environments and umpire behavior are not static. A model that ignores these changes risks learning a run environment that no longer exists.

### 5.9 Schedule, Travel, and Lineup Context

The builder also includes schedule and travel context such as:

- timezone crossings
- day-after-night-game signals
- lineup handedness balance
- lineup availability flags
- platoon advantage rates

This means the model is not only learning team quality, but also operational context. That is important because baseball performance depends heavily on who is actually in the lineup, how rested a club is, and how the batting order matches up with the opposing pitcher.

## 6. Evaluation Strategy

The project uses multiple layers of evaluation.

### 6.1 Modeling Metrics

For the run-count side of the project, distribution-focused metrics are used rather than only point accuracy. Recent benchmark values on the 2025 holdout were:

- Stage 3 CRPS: `1.79413`
- Stage 3 Negative Log Score: `2.49157`
- Stage 4 CRPS: `1.78815`
- Stage 4 Negative Log Score: `2.55447`

These metrics matter because the project is built around probability quality and distribution quality, not only point predictions.

### 6.2 Bankroll and Betting Metrics

Because the real goal is practical usefulness, the project also evaluates:

- ROI
- units won/lost
- win rate
- drawdown
- bet count
- average edge

This is a more realistic measure of success than generic classification accuracy.

## 7. Fast Bankroll Testing and Threshold Tuning

One of the most useful recent experiments was a fast bankroll testing framework using the cleaned historical odds archive. This allowed many rule variations to be tested quickly without retraining the full model each time.

The initial unrestricted flat 1-unit baseline over the 2025 season was:

- total bets: `8,718`
- net units: `+72.57`
- ROI: `0.83%`

By market, that baseline showed:

- full-game moneyline: `+94.33u`, `4.47% ROI`
- full-game total: `+63.82u`, `3.89% ROI`
- full-game run line: approximately flat
- first five moneyline: negative
- first five total: slightly negative

This suggested that the raw model had some value, but the betting rules were far too loose.

### 7.1 Strategy Grid Search

A larger strategy grid was then tested with:

- nonlinear unit sizing
- odds windows
- edge thresholds
- separate market constraints

The strongest raw variants were:

- `e60_-250_180`: `8.83% ROI`, `+349.20u`
- `e60_-175_135`: `7.88% ROI`, `+180.62u`
- `e60_-200_150`: `7.79% ROI`, `+234.77u`

However, those broader variants still produced too many bets per season and sometimes relied too heavily on run line behavior, which was a warning sign. Because of that, a more conservative selection process was explored.

### 7.2 One-Bet-Per-Game Filtering

The next refinement imposed these rules:

- full-game only
- at most one bet per game
- run line only when the run line itself was near the “coin-flip” range
- minimum and maximum edge filters

This produced a much more realistic framework for evaluating the model.

### 7.3 Edge Threshold Tuning

Threshold tests between `11%` and `15%` edge, with an upper cap below `22%`, produced the following results:

- `11%+`: `328` bets, `53.94%` non-push accuracy
- `12%+`: `261` bets, `56.35%` non-push accuracy
- `13%+`: `198` bets, `57.29%` non-push accuracy
- `14%+`: `130` bets, `61.60%` non-push accuracy
- `15%+`: `100` bets, `64.58%` non-push accuracy

This was one of the most important findings in the project. The data showed that low-edge plays added volume but not necessarily quality, while the higher-edge segment produced much stronger accuracy.

The working cutoff chosen from this testing was:

- minimum edge: `15%`
- maximum edge: `< 22.5%`
- one bet per game
- full-game markets only

Under that rule, the model’s filtered 2025 test produced:

- `100` bets
- average edge: `17.62%`
- accuracy including pushes: `62.0%`
- accuracy excluding pushes: `64.58%`

This threshold range became the current practical baseline because it balanced selectivity with performance.

### 7.4 Interpreting the Tuning Results

The threshold experiments are important because they reveal that the model is not equally useful across all of its own recommended bets. This is a crucial insight. A naïve betting system might assume that any positive expected value is worth acting on. In practice, that is not what the experiments showed. Instead, the results suggested that the model has a hierarchy of confidence quality.

At the low end of the edge distribution, the model may be directionally right but not right enough to overcome:

- vig
- score variance
- execution friction
- small calibration errors

At the higher end of the edge distribution, however, the model’s opinions begin to look materially stronger. That is why the one-bet-per-game thresholding exercise was so useful. It did not merely improve a profit metric. It taught something structural about the model itself: the model’s strongest edges are much more reliable than its marginal ones.

This matters from a machine learning perspective because it suggests that the model’s ranking of opportunities contains real information. If larger predicted edges did not behave better than smaller ones, that would imply a severe calibration or market-comparison failure. The fact that stronger edge buckets performed better means the model’s confidence signal is at least directionally meaningful.

### 7.5 Why Policy Design Matters

Another major lesson from these experiments is that the betting policy is part of the machine learning system. The predictive model and the execution policy cannot be treated as separate worlds. The same underlying predictive model can look weak, mediocre, or strong depending on how its outputs are converted into actions.

This can be seen clearly in the difference between:

- the unrestricted 1-unit baseline
- the nonlinear strategy grid
- the one-bet-per-game filtered strategy
- the final `15% to 22.5%` rule

The predictive model did not change radically between those tests. What changed was the decision rule. That means this project should not be viewed simply as “an XGBoost model for baseball.” It is better understood as a decision system where prediction quality and decision policy are tightly coupled.

### 7.6 One-Bet-Per-Game as a Structural Improvement

The one-bet-per-game restriction deserves special emphasis. When the project allowed multiple bets on the same game, a single strong game script belief could generate several correlated wagers:

- moneyline
- run line
- total

That creates a less interpretable test because multiple wins or losses can reflect the same underlying opinion. Restricting the strategy to one bet per game forces the system to choose its best expression of confidence. This has several advantages:

- it reduces overexposure to one matchup
- it creates a more realistic betting policy
- it makes post-hoc analysis much clearer

From a learning standpoint, it also makes market comparison easier. If the model loses on a one-bet-per-game framework, it is easier to determine whether that loss came from a genuine bad opinion rather than from duplicated exposure across several correlated markets.

### 7.7 Market-Level Lessons

The fast bankroll tests also produced a practical market ranking. On the cleaned 2025 archive, full-game moneyline and full-game totals looked materially more promising than first-five markets in the current implementation. That does not necessarily mean F5 is a bad target in principle. It means the present system seems better aligned with full-game pricing.

This distinction matters because it keeps the project from drawing the wrong conclusion. A poor F5 result in the current version could mean:

- F5 data coverage is weaker
- the present feature set is more full-game oriented
- the model does not yet capture the most important first-five dynamics

In a research setting, this is an important intellectual habit. Negative results should refine the target rather than be mistaken for universal truth.

### 7.8 Thresholding as a Practical Calibration Check

Another way to think about the threshold tuning is that it acts like a practical calibration check. If the model says some bets have much larger edge than others, then those bets should, on average, be better. The threshold experiments therefore serve as a test of whether the model’s confidence ranking corresponds to real betting quality.

The fact that the `14%+` and `15%+` buckets performed much better than the lower thresholds suggests the answer is yes. The exact magnitude may vary over time, but the basic idea is valuable: the model’s internal edge ordering can be used as a real filter rather than just a cosmetic score.

## 8. Live 2026 Tracking Results

The project now also supports live-style season tracking. The key design choice is that once a pick is pulled for the day, it is frozen and cannot be backfilled later for artificial accuracy. Results can be added after the games complete, but the original pick does not change.

At the time of this report, the early 2026 tracked performance was:

- tracked picks: `12`
- graded picks: `12`
- wins: `9`
- losses: `2`
- pushes: `1`
- accuracy excluding pushes: `81.82%`
- official units risked: `16.0u`
- official profit: `+9.98u`
- official ROI: `62.38%`

These numbers are encouraging, but the sample is still extremely small. Therefore, they should be treated as an early paper-trading snapshot rather than definitive evidence of long-run profitability.

### 8.1 Why the Live Tracker Is Important

The live tracker is methodologically important because it protects the project from hindsight bias. One of the easiest ways to fool oneself in sports modeling is to rerun a model, refresh lines, or silently overwrite a prior decision with a better later one. That makes the record look stronger than it actually was.

The live tracking framework addresses this by separating three stages:

1. capture the pick
2. freeze the pick
3. settle the pick later

This is a major improvement over casual backtesting because it creates an auditable path from model output to evaluated result. In practical terms, this means:

- the pick can be stored with its original line and odds
- the result can be filled in later
- but the historical pick itself should not be rewritten

That makes the project substantially more credible as a machine learning system intended for repeated practical use.

### 8.2 Why Live Tracking and Backtesting Complement Each Other

Backtesting is still valuable because it allows broader experimentation across many games, seasons, and thresholds. Live tracking serves a different purpose. It is the honest operational record. The strongest workflow is to use both:

- backtesting to search for promising structures and thresholds
- live tracking to verify that those structures remain sensible in current conditions

This is another sign that the project is moving beyond a toy model. It now has both a research mode and a live-evaluation mode.

## 9. Key Technical Lessons

Several technical lessons emerged from this project.

### 9.1 Better Data Matters More Than More Complexity

Some of the biggest improvements came from fixing historical odds quality rather than inventing a new model. Bad odds rows can make a model appear unrealistically profitable, so data validation was critical.

### 9.2 Probability Quality Is More Important Than Plain Accuracy

Because betting depends on price, expected value, and edge, the system must produce meaningful probabilities, not simply correct winners. This is why CRPS, negative log score, and market-based evaluation were emphasized.

### 9.3 Decision Rules Matter

The unrestricted model produced too many bets. Filtering by edge, capping the edge range, restricting to one bet per game, and focusing on stronger markets all improved realism.

### 9.4 Live Tracking Must Be Immutable

If a model is allowed to rewrite past picks, the results are not trustworthy. The new system avoids this by freezing same-day picks and only updating outcomes after the fact.

## 10. Limitations

This project still has several limitations.

- The live 2026 tracking sample is small.
- Some markets, especially first-five variants, have had weaker or less reliable data coverage.
- Odds availability can vary by bookmaker and time of day.
- Even a profitable paper-trading result does not guarantee identical real-world execution because of line movement and timing.

These limitations mean the system should still be viewed as an actively improving research platform rather than a finished production betting engine.

## 11. Conclusion

MLBPrediction2026 is a practical machine learning system that combines predictive modeling, baseball feature engineering, probabilistic evaluation, and sportsbook-based decision rules. The project moved beyond standard model training by directly testing how machine learning outputs perform when converted into bets.

The most important empirical result from recent tuning was that selectivity matters. A broad strategy produced only modest raw ROI, but filtering the model down to stronger opportunities improved its practical performance significantly. In particular, the `15% to 22.5%` edge window emerged as the best working rule among the tested thresholds, producing `64.58%` non-push accuracy on the filtered 2025 sample.

Overall, this project demonstrates that successful sports prediction requires more than a model with good generic accuracy. It requires good data, calibrated probabilities, careful market comparison, and disciplined decision rules. That combination is what makes the project a meaningful machine learning application rather than just a sports statistics exercise.

## 12. Additional Notes for Future Revision

This final section is intentionally more direct and implementation-oriented. It is included so that a future revision of the report can draw from concrete project facts without needing to rediscover them.

### 12.1 Core Pipeline Summary

The current project can be summarized as:

1. build historical and live baseball feature frames
2. train temporal XGBoost base models
3. train a stacking layer on out-of-fold probabilities
4. calibrate probabilities
5. estimate run and market probabilities
6. compare those probabilities to sportsbook prices
7. size a bet using constrained Kelly-style logic
8. freeze the pick for live tracking
9. settle the pick later using actual game outcomes

This matters because it shows that the project is much closer to a decision system than to a simple predictive homework exercise.

### 12.2 Key Numbers Worth Preserving

The following results are especially useful for any later rewrite of this report:

- unrestricted 2025 fast bankroll baseline:
  - `8,718` bets
  - `+72.57u`
  - `0.83% ROI`
- baseline market breakdown:
  - full-game ML: `+94.33u`, `4.47% ROI`
  - full-game total: `+63.82u`, `3.89% ROI`
  - full-game RL: about flat
  - F5 ML: negative
  - F5 total: slightly negative
- best broad strategy-grid results:
  - `e60_-250_180`: `8.83% ROI`, `+349.20u`
  - `e60_-175_135`: `7.88% ROI`, `+180.62u`
  - `e60_-200_150`: `7.79% ROI`, `+234.77u`
- one-bet-per-game threshold results:
  - `11%+`: `53.94%` non-push accuracy
  - `12%+`: `56.35%`
  - `13%+`: `57.29%`
  - `14%+`: `61.60%`
  - `15%+`: `64.58%`
- current practical threshold:
  - `15% <= edge < 22.5%`
  - `100` bets
  - average edge `17.62%`
  - `64.58%` non-push accuracy
- early 2026 live tracked snapshot:
  - `12` graded picks
  - `9` wins
  - `2` losses
  - `1` push
  - `16.0u` risked
  - `+9.98u` official profit
  - `62.38%` official ROI

### 12.3 Main Open Research Opportunities

The most important next-step ideas are:

- stronger air-density and weather interaction modeling
- more explicit precipitation-to-command effects
- bullpen leverage and availability refinement
- FanGraphs BaseRuns as a sanity layer for team strength
- EVAnalytics-style Statcast shape features, especially launch-angle dispersion and bucket structure
- a more mature F5-specific branch once data and pricing support are cleaner

### 12.4 Main Conceptual Takeaway

The biggest conceptual lesson from the project is that sports betting machine learning is not just about finding the most predictive model. It is about coordinating:

- feature engineering
- calibration
- market comparison
- thresholding
- execution discipline
- honest tracking

That combination is the real contribution of the project. It is the reason the system is interesting as a machine learning application rather than merely as a baseball statistics tool.
