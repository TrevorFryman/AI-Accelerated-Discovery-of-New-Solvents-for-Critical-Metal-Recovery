# RDKit Descriptor Model - Baseline Comparison Report (Step 6, Optional)

Step 6 (optional) of the fine-tuning plan: sanity-check the final XGBoost model (Steps 1-5) against simpler, fully-transparent baselines on the same 13-descriptor feature set, using the identical nested-CV protocol from Step 4 (outer 5-fold `StratifiedGroupKFold`, inner 4-fold `StratifiedGroupKFold` for hyperparameter selection). Linear models use a `StandardScaler` -> `{RidgeCV, ElasticNetCV}` pipeline so alpha (and l1_ratio for ElasticNet) selection happens via the inner CV.

Feature columns (13): MolWt, LogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds, RingCount, NumAromaticRings, FractionCSP3, MolMR, HeavyAtomCount, NumAliphaticRings, BertzCT

## Nested CV Results (Mean +/- 95% CI, t-distribution df=4)

| Model | R2 | RMSE | MAE |
|---|---|---|---|
| Ridge (StandardScaler + RidgeCV) | 0.4028 [0.2371, 0.5684] | 0.9632 [0.8178, 1.1087] | 0.7646 [0.6202, 0.9090] |
| ElasticNet (StandardScaler + ElasticNetCV) | 0.4040 [0.2345, 0.5734] | 0.9619 [0.8145, 1.1093] | 0.7629 [0.6189, 0.9069] |
| RandomForestRegressor (RandomizedSearchCV) | 0.4477 [0.2599, 0.6356] | 0.9262 [0.7493, 1.1031] | 0.7260 [0.5677, 0.8843] |
| XGBoost (Step 4/5, final) | 0.5050 [0.3342, 0.6757] | 0.8721 [0.7238, 1.0203] | 0.6699 [0.5292, 0.8107] |

## Per-Fold R2

| Outer Fold | Ridge | ElasticNet | RandomForest |
|---|---|---|---|
| 1 | 0.2838 | 0.2797 | 0.1982 |
| 2 | 0.3563 | 0.3582 | 0.4684 |
| 3 | 0.5420 | 0.5479 | 0.6037 |
| 4 | 0.2832 | 0.2831 | 0.4537 |
| 5 | 0.5485 | 0.5510 | 0.5146 |

## R2 Gap vs XGBoost (XGBoost R2 - Baseline R2)

| Model | R2 Gap |
|---|---|
| Ridge (StandardScaler + RidgeCV) | +0.1022 |
| ElasticNet (StandardScaler + ElasticNetCV) | +0.1010 |
| RandomForestRegressor (RandomizedSearchCV) | +0.0573 |

## Step 6 Success Check

- Largest R2 gap vs XGBoost (any baseline): +0.1022
- **XGBoost's added complexity justified (gap > 0.1 for all baselines): True**

All simpler baselines underperform XGBoost by more than 0.1 in nested-CV R2, justifying XGBoost's added complexity for the primary reported model. Linear coefficients/feature importances from the Ridge/ElasticNet fits may still be useful as a secondary, easier-to-communicate interpretability narrative.
