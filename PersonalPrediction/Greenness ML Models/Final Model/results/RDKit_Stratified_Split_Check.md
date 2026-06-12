# RDKit Descriptor Model - Stratified Group K-Fold Split Check

Step 1 of the fine-tuning plan: re-evaluate the current best hyperparameters (`model/rdkit_best_params.json`) under a `StratifiedGroupKFold` split (grouped by `solvent_SMILES`, stratified by `Classification`), without re-running the hyperparameter search.

Hyperparameters used: `{'subsample': 0.7, 'n_estimators': 500, 'min_child_weight': 1, 'max_depth': 3, 'learning_rate': 0.01, 'colsample_bytree': 0.8}`

Train size: 126, Test size: 32

SMILES leakage check: PASSED

## Results Comparison

| Split | Set | RMSE | MAE | R2 |
|---|---|---|---|---|
| Old (random 80/20 + KFold) | Train | 0.5304 | 0.4065 | 0.8217 |
| Old (random 80/20 + KFold) | Test | 0.9475 | 0.7329 | 0.5016 |
| Old (random 80/20 + KFold) | OOF | 0.9331 | 0.7244 | 0.4636 |
| New (StratifiedGroupKFold) | Train | 0.4916 | 0.3805 | 0.8483 |
| New (StratifiedGroupKFold) | Test | 1.0991 | 0.8783 | 0.2621 |
| New (StratifiedGroupKFold) | OOF | 0.9169 | 0.7032 | 0.4801 |

## Train-CV R2 Gap

- Old split gap: 0.3581
- New split gap: 0.3682
- Change: +0.0101

**Conclusion:** The train-CV gap did not shrink under the new StratifiedGroupKFold split. Leakage from duplicate SMILES does not appear to be the primary driver of the overfitting gap; proceed to later fine-tuning steps (descriptor expansion / regularization) for further investigation.
