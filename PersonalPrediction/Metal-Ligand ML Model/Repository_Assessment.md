# Repository Assessment — `stabilityconstant-ml-models`

**Repo path:** `C:\dev\stabilityconstant-ml-models-main\stabilityconstant-ml-models-main`

## 1. Overview

This is a **results/distribution repository**, not a training pipeline. It packages four pre-trained
Chemprop D-MPNN ensembles for predicting **stability constants (log K1)** and **protonation constants
(log Kp)** of metal–ligand complexes, trained on data extracted from the **IUPAC Stability Constant
Database** (internally referred to as "v53"). It ships the exact train/val/test splits used and two
helper notebooks for converting SMILES/peptide sequences into model-ready inputs.

## 2. The four models

| Model | Target | Variants shipped | Folds | Extra features required? |
|---|---|---|---|---|
| **M1** — `stability_constant_model_M1` | log K1, stability constant across multiple temperatures | `SMILES_only_model`, `best_model` | 5 (fold_0–4) each | `best_model` needs `Temp / 0C, Ionic strength /M` |
| **M2** — `stability_constant_25C_model_M2` | log K1, stability constant **at 25 °C only** | `SMILES_only_model` only | 5 | None — single SMILES column |
| **M3** — `protonation constant_model_M3` | log Kp, protonation constant | `SMILES_only_model`, `best_model` | 5 each | `best_model` needs experimental conditions + 40 RDKit descriptors (selected via f_regression) |
| **M4** — `stability_constan_protonation_constant_model_M4` | combined log K1 + log Kp | `SMILES_only_model`, `best_model` | 5 each | `best_model` needs 8 metal-property features + `Temp, Concentration of Medium, Exact MW of solvent, Density of solvent` |

Each model directory's `info.txt` confirms the SMILES-only vs. best-model distinction. Every variant is
a **5-fold ensemble** — Chemprop's `--checkpoint_dir` walks the directory and automatically averages
all `fold_*/model_0/model.pt` files it finds.

## 3. Required inputs / expected outputs

### Input format (all models)
A single CSV with a `SMILES` column containing a **multi-fragment SMILES string**: the ligand SMILES,
a literal `.`, then the metal ion as a bracketed charged species, e.g.:

```
SMILES,Experimental LgK1
O=C(NO)c1ccccc1.[Fe+3],11.08
NC(CCC(=O)O)C(=O)O.[Co+2],4.67
```

Additional counter-ions (e.g. nitrate) can be appended as further `.`-separated fragments — seen in
`data/applications/applications_DFT_vs_ML_predictions.csv`.

For `best_model` variants, a **parallel `*_features.csv`** (same row order as the input CSV, no header
overlap with SMILES) supplies the extra numeric descriptors listed in the table above, passed via
`--features_path`.

### Output
`chemprop_predict --preds_path <out.csv>` writes a CSV with the input SMILES plus a predicted target
column (named after the training target, e.g. `Experimental LgK1`/`Experimental Lg K1`).

## 4. Training datasets (shipped under `data/`)

| Dataset dir | Train | Val | Test | Notes |
|---|---|---|---|---|
| `stability_constan_model_M1` | 28,899 | 348 | 3,212 | `train_features.csv` etc. parallel feature files (Temp, Ionic strength) |
| `stability_constant_25C_model_M2` | 20,958 | 253 (`v53_validation_set_input.csv`) | 2,329 | SMILES-only, no parallel features |
| `protonation_constant_model_M3` | 10,323 | 50 | 1,148 | parallel features = 46 columns (experimental conditions + RDKit descriptors) |
| `stability_constant_protonation_constant_model_M4` | 39,050 | 398 | 4,338 | parallel features = 16 columns (metal properties + experimental conditions) |

Supporting reference data: `data/atomic_features.xlsx`, `data/metal_features/ionization_energies_IE.xlsx`
(used to build metal-property feature columns for M4), and `data/applications/*` (worked examples from
the source paper — biomolecules, peptides/proteins, DFT-vs-ML comparison).

## 5. Model checkpoints

All checkpoints are **Chemprop v1-format `.pt` files** (PyTorch `state_dict` + v1 `TrainArgs`), located
at `models/<model>/<variant>/fold_{0..4}/model_0/model.pt`. M1 also has a `best_model/info.txt`
summarizing the best-performing variant. `models/stability_constant_25C_model_M2/SMILES_only_model/args.json`
contains the full v1 `TrainArgs` dump, including the original training command:

```
python train.py --data_path iupac_v53_input/v53_train_input.csv \
  --config_path iupac_v53_hyp_par_opt/v53_opt_par.json \
  --dataset_type regression --separate_test_path iupac_v53_input/v53_test_input.csv \
  --num_folds 5 --save_dir iupac_v53_output/no_features --metric rmse --extra_metrics mae mse r2
```

Key architecture hyperparameters for M2 (from `args.json`): `depth=6`, `hidden_size=2200`,
`ffn_hidden_size=2200`, `ffn_num_layers=3`, `aggregation=mean`, `number_of_molecules=1`,
`dataset_type=regression`, trained for 30 epochs.

## 6. Python / Chemprop / dependency requirements

- **Chemprop version used for training:** v1.5.2 (per README and `args.json` reproducibility block).
  The README claims a copy of v1.5.2 is bundled in a "Chemprop directory of this repository" —
  **this is not present** in the extracted repo; no v1 source or `environment.yml` ships with it.
- **Chemprop v1.5.2's target Python:** ~3.7–3.9, with older pinned `torch`/`numpy`.
- **Currently installed Python:** 3.14.3 — compatible with Chemprop v2.x but **too new for v1.5.2**.
- **Currently extracted `chemprop-main`:** v2.2.3 (current `main` branch), a full PyTorch-Lightning
  rewrite, architecturally incompatible with v1 checkpoints without conversion (see
  `Chemprop_Compatibility_Report.md`).
- **Other installed packages (current env):** `torch 2.12.0+cpu`, `rdkit 2026.03.2`, `pandas 3.0.0`,
  `scikit-learn 1.9.0`; `chemprop` and `lightning` are **not installed**.

## 7. Implications for the project goal (Co/Ni vs. DES ligands)

- **M2 `SMILES_only_model`** is the best-suited model for the stated goal:
  - Single SMILES column input (`ligand_SMILES.[Co+2]` / `ligand_SMILES.[Ni+2]`), no auxiliary feature
    files needed.
  - Largest single-target stability-constant training set at a fixed, well-defined condition (25 °C).
  - M2's training data already contains substantial Co (1,355 rows) and Ni (2,080 rows) examples,
    giving reasonable metal-specific coverage.
- M1/M3/M4 `best_model` variants require feature files that would need to be regenerated for new DES
  ligands (temperature/ionic strength assumptions, RDKit descriptors, metal-property lookups) — adds
  complexity and assumption risk without a clear accuracy benefit for this use case.
- No model checkpoint can currently be loaded — `chemprop` is not installed in any environment, and
  the checkpoint format (v1) does not match the only Chemprop source available locally (v2.2.3).

---
*Status: read-only assessment complete. No code executed, no models run, nothing installed.*
