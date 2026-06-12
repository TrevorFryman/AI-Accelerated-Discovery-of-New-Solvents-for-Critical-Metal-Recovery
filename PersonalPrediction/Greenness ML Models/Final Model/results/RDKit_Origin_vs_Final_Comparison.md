# RDKit Descriptor Model: Origin vs. Final (Fine-Tuned) Model

This page summarizes the fine-tuning work performed on the Track 2 RDKit
descriptor + XGBoost model (the "origin model") to produce the "final
model" in `Final Model/`. The origin model was flagged as overfitting
(Train R² = 0.8217 vs. 5-fold OOF R² = 0.4636, a gap of 0.3581). The work
below followed a 6-step fine-tuning plan to address this and to produce a
more rigorously validated, more interpretable model.

Scripts for the final model are saved under `_v2` / new names
(`generate_rdkit_features_v2.py`, `train_rdkit_baseline_v2.py`,
`optimize_rdkit_model_v2.py`, `finetune_rdkit_nested_cv.py`,
`finalize_rdkit_model.py`, `baseline_comparison.py`) so they can be told
apart from the origin Track 2 scripts.

---

## Summary of Testing & Improvements

| Step | What was tested | Result |
|---|---|---|
| 1. Split methodology | Replaced the random 80/20 split + plain `KFold` with `StratifiedGroupKFold` (grouped by `solvent_SMILES` to prevent duplicate-solvent leakage, stratified by `Classification`) | SMILES leakage check **passed** (no leakage in the original split), but the train-CV gap did **not** shrink (0.3581 → 0.3682). Confirms leakage was not the primary cause of overfitting. |
| 2. Expand descriptor set | Added 6 new interpretable RDKit descriptors (7 → 13 total): `NumAromaticRings`, `FractionCSP3`, `MolMR`, `HeavyAtomCount`, `NumAliphaticRings`, `BertzCT` | OOF R² improved (0.4801 → 0.5123) with only a small increase in the train-CV gap (0.3682 → 0.3770). Net positive — kept all 13 descriptors. |
| 3a. Re-tune hyperparameters (original grid) | `RandomizedSearchCV` (n_iter=100) on the 13-feature set, same grid as the origin model | **Rejected** — performed worse on every metric (OOF R² 0.5123 → 0.4250, gap 0.3770 → 0.4506) due to noisy inner CV folds on the ~126-sample training split. |
| 3b. Re-tune with explicit regularization | Extended search space with `reg_alpha`, `reg_lambda`, `gamma`, capped `max_depth` to [2,3,4], biased `min_child_weight` higher (n_iter=250) | **Rejected** by the strict success criterion (gap widened to 0.4369), though OOF R² hit its best single-split value yet (0.5320). Kept the Step 2 hyperparameters. |
| 4. Nested cross-validation | Outer 5-fold `StratifiedGroupKFold` with hyperparameters re-tuned per outer fold via inner 4-fold `StratifiedGroupKFold` (n_iter=150) | Produced the headline generalization estimate: **R² = 0.5050 [0.3342, 0.6757]**, RMSE = 0.8721 [0.7238, 1.0203] (mean ± 95% CI, t-dist, df=4). Both single-split estimates fell within this CI. |
| 5. Final model & interpretability refresh | Refit final model (13 features, Step 2 hyperparameters) on all 154 samples; regenerated feature importance, SHAP, parity/residual, and learning-curve plots | TPSA confirmed as the #1 driver by both gain-based importance and SHAP (sanity check passed). |
| 6. Baseline sanity check (optional) | Compared final XGBoost model against Ridge, ElasticNet, and RandomForest using the identical nested-CV protocol | XGBoost (R² = 0.5050) outperformed Ridge (0.4028), ElasticNet (0.4040), and RandomForest (0.4477) — gaps of ~0.05–0.10, justifying XGBoost as the primary reported model. |

---

## Origin Model vs. Final Model

| | **Origin Model** (Track 2) | **Final Model** |
|---|---|---|
| Descriptor count | 7 | **13** (added NumAromaticRings, FractionCSP3, MolMR, HeavyAtomCount, NumAliphaticRings, BertzCT) |
| Train/test split | Random 80/20 (`train_test_split`, random_state=42) + plain `KFold(5)` for OOF | `StratifiedGroupKFold(5)` — grouped by `solvent_SMILES`, stratified by `Classification` |
| Hyperparameters | `n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.7, colsample_bytree=0.8, min_child_weight=1` | Same hyperparameters (re-tuning attempts were tested and rejected — see Steps 3a/3b) |
| Reported generalization metric | Single 80/20 split: Test R² = 0.5016; 5-fold OOF R² = 0.4636 | **Nested CV: R² = 0.5050 [0.3342, 0.6757]** (mean ± 95% CI across 5 independent outer folds, each with its own hyperparameter search) |
| Train R² | 0.8217 | 0.8581 (full-data refit; diagnostic only, not a generalization estimate) |
| Train–CV gap | 0.3581 | 0.3770 (Step 2 config, Step 1 split) |
| Top feature (importance / SHAP) | TPSA (both) | TPSA (both) |
| Validated against simpler baselines | No | Yes — outperforms Ridge, ElasticNet, RandomForest by 0.05–0.10 R² under the same nested-CV protocol |

### Most Significant Improvements

1. **Rigorous, leakage-aware split.** The final model uses `StratifiedGroupKFold` grouped by `solvent_SMILES` and stratified by solvent `Classification`, eliminating any risk of the same solvent appearing in both train and test, and ensuring all 11 solvent classes are represented across folds.
2. **Richer, still-interpretable descriptor set.** Expanding from 7 to 13 named RDKit descriptors raised OOF R² from 0.4801 to 0.5123 under the same split and hyperparameters, while every feature retains a clear physical/chemical meaning (see glossary below).
3. **Statistically defensible headline metric.** Rather than reporting a single 80/20 split (n_test ≈ 31, high variance), the final model is reported via **nested cross-validation**: 5 independent outer test folds, each evaluated with hyperparameters tuned only on that fold's training data, summarized as a mean ± 95% CI (R² = 0.5050 [0.3342, 0.6757]).
4. **Hyperparameter re-tuning was tested, not assumed.** Two separate re-tuning attempts (standard grid, then regularization-focused) were run and rigorously compared against the current best — both were rejected based on pre-defined success criteria, giving confidence that the kept hyperparameters are genuinely the best available, not just the first ones tried.
5. **Benchmarked against simpler models.** The final XGBoost model was sanity-checked against Ridge, ElasticNet, and RandomForest baselines under the identical nested-CV protocol, confirming that the added model complexity is justified (XGBoost R² = 0.5050 vs. 0.40–0.45 for the simpler models).
6. **Consistent, validated interpretability.** TPSA remains the #1 driver of predicted G-score by both gain-based importance and SHAP in the final model — the same conclusion as the origin model, now confirmed on a larger, more carefully validated feature/model configuration.

---

## Feature Importance (Final Model, 13 Descriptors)

![Feature Importance](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Final%20Model/results/RDKit/feature_importance.png)

![SHAP Summary](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Final%20Model/results/RDKit/shap_summary.png)

![SHAP Importance Bar](https://raw.githubusercontent.com/TrevorFryman/AI-Accelerated-Discovery-of-New-Solvents-for-Critical-Metal-Recovery/main/PersonalPrediction/Greenness%20ML%20Models/Final%20Model/results/RDKit/shap_importance_bar.png)

### XGBoost Gain-Based Feature Importance

| Rank | Descriptor | Importance |
|---|---|---|
| 1 | TPSA | 0.1547 |
| 2 | NumHAcceptors | 0.1260 |
| 3 | NumHDonors | 0.1025 |
| 4 | NumRotatableBonds | 0.0932 |
| 5 | MolMR | 0.0768 |
| 6 | FractionCSP3 | 0.0740 |
| 7 | HeavyAtomCount | 0.0694 |
| 8 | NumAromaticRings | 0.0672 |
| 9 | LogP | 0.0646 |
| 10 | BertzCT | 0.0617 |
| 11 | MolWt | 0.0446 |
| 12 | NumAliphaticRings | 0.0332 |
| 13 | RingCount | 0.0319 |

### SHAP Ranking (Mean |SHAP value|)

| Rank | Descriptor | Mean \|SHAP value\| |
|---|---|---|
| 1 | TPSA | 0.5230 |
| 2 | MolMR | 0.2164 |
| 3 | LogP | 0.1902 |
| 4 | MolWt | 0.1153 |
| 5 | BertzCT | 0.1087 |
| 6 | NumHAcceptors | 0.0997 |
| 7 | FractionCSP3 | 0.0895 |
| 8 | NumHDonors | 0.0823 |
| 9 | NumRotatableBonds | 0.0617 |
| 10 | RingCount | 0.0311 |
| 11 | HeavyAtomCount | 0.0287 |
| 12 | NumAromaticRings | 0.0122 |
| 13 | NumAliphaticRings | 0.0063 |

---

## Descriptor Glossary

What each of the 13 RDKit descriptors actually measures:

| Descriptor | Meaning |
|---|---|
| **TPSA** | Topological Polar Surface Area — the surface area of a molecule contributed by polar atoms (O, N, and attached H). Higher TPSA generally means more polar, more water-soluble, less "green-solvent-like" molecules. |
| **NumHAcceptors** | Number of hydrogen bond acceptor atoms (e.g., O, N with available lone pairs) in the molecule. |
| **NumHDonors** | Number of hydrogen bond donor groups (e.g., O-H, N-H) in the molecule. |
| **NumRotatableBonds** | Number of bonds that can freely rotate (excluding ring bonds and terminal bonds), a measure of molecular flexibility. |
| **MolMR** | Molar Refractivity (Crippen model) — relates to a molecule's polarizability/volume; correlates with size and how strongly the molecule interacts with light/other molecules. |
| **FractionCSP3** | Fraction of carbon atoms that are sp3-hybridized (i.e., "saturated," tetrahedral carbons) versus sp2/sp (aromatic, double/triple-bonded). A higher value indicates a more saturated, less aromatic/flat molecule. |
| **HeavyAtomCount** | Total number of non-hydrogen atoms in the molecule — a simple measure of molecular size. |
| **NumAromaticRings** | Number of aromatic (e.g., benzene-like) rings in the molecule. |
| **LogP** | Crippen-estimated octanol-water partition coefficient — a measure of lipophilicity (how much the molecule prefers a non-polar/oily environment vs. water). Higher LogP = more hydrophobic. |
| **BertzCT** | Bertz's topological complexity index — a measure of overall structural/molecular complexity based on the diversity of atoms, bonds, and connections. |
| **MolWt** | Molecular Weight — the total mass of the molecule (sum of atomic weights). |
| **NumAliphaticRings** | Number of non-aromatic (aliphatic) rings in the molecule. |
| **RingCount** | Total number of rings (aromatic + aliphatic) in the molecule. |
