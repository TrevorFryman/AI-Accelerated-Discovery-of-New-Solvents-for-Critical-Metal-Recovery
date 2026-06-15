# Repository Analysis Report: Stability-Constant ML Models + Chemprop

This report summarizes the analysis of the two repositories you extracted locally:

- `C:\dev\stabilityconstant-ml-models-main\stabilityconstant-ml-models-main`
- `C:\dev\chemprop-main\chemprop-main`

and what is required to run the trained models against new ligand/metal pairs (e.g., DES ligands with Co²⁺ and Ni²⁺).

---

## 1. What `stabilityconstant-ml-models` actually contains

This repo is **not** source code for training a new model — it's a results/distribution repo for a project that:

1. Pulled ligand–metal **stability constant (log K)** and **protonation constant (log Kp)** data from the **IUPAC Stability Constant Database**.
2. Trained **Chemprop D-MPNN (directed message-passing neural network) ensembles** on that data using **Chemprop v1.5.2**.
3. Shipped the trained model checkpoints (`model.pt`), the exact train/val/test splits, and some small conversion notebooks for turning SMILES/peptide sequences into model-ready inputs.

### 1.1 Folder structure

```
stabilityconstant-ml-models-main/
├── README.md                  # install/usage instructions (chemprop_predict)
├── data/
│   ├── atomic_features.xlsx
│   ├── metal_features/ionization_energies_IE.xlsx
│   ├── applications/          # case-study predictions from the paper
│   ├── stability_constan_model_M1/        (train/val/test input + features)
│   ├── stability_constant_25C_model_M2/   (train/val/test input, no extra features)
│   ├── protonation_constant_model_M3/     (train/val/test input + features)
│   └── stability_constant_protonation_constant_model_M4/ (train/val/test + features)
├── models/
│   ├── stability_constant_model_M1/              best_model + SMILES_only_model (5-fold ensembles)
│   ├── stability_constant_25C_model_M2/          SMILES_only_model only (5-fold ensemble)
│   ├── protonation constant_model_M3/            best_model + SMILES_only_model
│   └── stability_constan_protonation_constant_model_M4/ best_model + SMILES_only_model
└── python_scripts/
    ├── peptide_sequence_to_smiles_conversion.ipynb
    └── rdkit_descriptors_morgan_fp_calculation.ipynb
```

### 1.2 The four models

| Model | Target | Data | Variants shipped |
|---|---|---|---|
| **M1** | Stability constant (log K1), multi-temperature | IUPAC data across temperatures | `SMILES_only_model` (SMILES only) and `best_model` (SMILES + temperature/ionic strength features) |
| **M2** | Stability constant (log K1) **at 25 °C** | IUPAC data filtered to 25 °C | `SMILES_only_model` only — **simplest, no extra feature files needed** |
| **M3** | Protonation constant (log Kp) | IUPAC protonation data | `SMILES_only_model` and `best_model` (+ 40 RDKit descriptors selected via f_regression + experimental conditions) |
| **M4** | Combined stability + protonation constants | Combined dataset | `SMILES_only_model` and `best_model` (+ metal-property features + experimental conditions) |

Each variant is a **5-fold ensemble** (`fold_0` … `fold_4`, each containing `model_0/model.pt`). Chemprop treats a checkpoint directory as an ensemble automatically.

### 1.3 Input format — this is the key detail for your use case

Every training/test CSV uses **one SMILES column** containing a **multi-component SMILES string**: the ligand's SMILES, a literal `.`, then the metal ion written as a bracketed ion, e.g.:

```
SMILES,Experimental LgK1
O=C(NO)c1ccccc1.[Fe+3],11.08
NC(CCC(=O)O)C(=O)O.[Co+2],4.67
NCCN(CC(=O)O)CC(=O)O.[Co+2],11.59
```

So a "molecule" fed to the model is really **ligand + metal ion as two disconnected fragments in one SMILES string**. Some entries have multiple counter-ions appended too (e.g., nitrates), as seen in `data/applications/applications_DFT_vs_ML_predictions.csv`.

For your stated goal — **logK for DES ligands vs. Co and Ni** — the M2 (25 °C) `SMILES_only_model` is the natural fit:
- Input CSV needs one column (header `SMILES`), each row = `<ligand_SMILES>.[Co+2]` or `<ligand_SMILES>.[Ni+2]`.
- No auxiliary feature files are required for M2 (unlike M1/M3/M4 `best_model` variants, which need a parallel `features.csv` with temperature/ionic strength/metal descriptors).

### 1.4 README's stated usage

```
conda env create -f environment.yml
conda activate chemprop
pip install -e .

chemprop_predict --test_path data/smiles_input.csv \
                  --checkpoint_dir model/smiles_only_model \
                  --preds_path logk_preds.csv
```

This is the **Chemprop v1.x command-line API** (`chemprop_predict` script, `--checkpoint_dir` walks a directory and ensembles all `.pt` files found).

---

## 2. What `chemprop-main` actually contains — ⚠️ important version mismatch

The README explicitly says the models were trained with **Chemprop 1.5.2**, and says a copy of that version "is located in the Chemprop directory of this repository." **However, the zip you extracted is the current `main` branch of Chemprop, which is version 2.2.3** (`chemprop/__init__.py` → `__version__ = "2.2.3"`, `pyproject.toml` → `version = "2.2.3"`).

Chemprop **v1** and **v2** are architecturally different codebases:
- v1 used argparse-based scripts (`chemprop_train`, `chemprop_predict`) and a custom training loop.
- v2 is a full rewrite built on **PyTorch Lightning**, with a new `chemprop` CLI (`chemprop train`, `chemprop predict`, `chemprop convert`, etc.) and a different model/checkpoint structure (`MPNN` Lightning module, `.ckpt`/`.pt` with `state_dict` + `hyper_parameters`).

**The `model.pt` files shipped in `stabilityconstant-ml-models` are v1.5.2-format checkpoints** (confirmed by `models/stability_constant_25C_model_M2/SMILES_only_model/args.json`, which has the full v1 `TrainArgs` dump, including a v1-style `"reproducibility"` block showing the original training command:
```
python train.py --data_path iupac_v53_input/v53_train_input.csv \
  --config_path iupac_v53_hyp_par_opt/v53_opt_par.json \
  --dataset_type regression --separate_test_path iupac_v53_input/v53_test_input.csv \
  --num_folds 5 --save_dir iupac_v53_output/no_features --metric rmse --extra_metrics mae mse r2
```
). These **cannot be loaded directly by the v2.2.3 `chemprop predict` CLI** — v2's `MPNN.load_from_checkpoint` expects the v2 `state_dict`/`hyper_parameters` schema.

### 2.1 Good news: v2 ships a v1→v2 converter

`chemprop/cli/convert.py` and `chemprop/utils/v1_to_v2.py` implement exactly this conversion:

```
chemprop convert -c v1_to_v2 -i <model_v1.pt> -o <model_v2.pt>
```

For **M2's `SMILES_only_model`** (single-molecule, no atom/bond/solvent features — the "else" branch of `convert_state_dict_v1_to_v2` / `convert_hyper_parameters_v1_to_v2`), this is the **well-tested, simple case**. The converter's docstring/log even reminds you: *"The default v1 atom featurizer is `MultiHotAtomFeaturizer.v1()` and can be specified from the command line with `--multi-hot-atom-featurizer-mode v1`."* — this flag **must** be passed to `chemprop predict` when using a converted checkpoint, or predictions will be silently wrong (v2's default atom featurizer differs from v1's).

By contrast, the `best_model` variants for M1/M4 use multiple input "molecules" (ligand + metal-feature "solvent" encoder) — the converter explicitly logs `"This conversion is untested..."` for `number_of_molecules > 1`. **M2 SMILES_only avoids this entirely**, reinforcing it as the right model for your task.

### 2.2 Two viable paths to run predictions

**Path A — Use the local Chemprop v2.2.3 + convert checkpoints (recommended)**
1. `pip install -e .` inside `C:\dev\chemprop-main\chemprop-main` (plus `pip install lightning`).
2. For each of the 5 folds of `models/stability_constant_25C_model_M2/SMILES_only_model/fold_*/model_0/model.pt`, run `chemprop convert -c v1_to_v2 ...` → produces 5 v2 checkpoints.
3. Build a CSV of `ligand_SMILES.[Co+2]` / `ligand_SMILES.[Ni+2]` rows.
4. `chemprop predict --test-path <csv> --model-paths <5 converted .pt files> --multi-hot-atom-featurizer-mode v1 --preds-path <out.csv>`.

**Path B — Recreate Chemprop v1.5.2 environment (matches README exactly)**
1. Create a separate conda env with an older Python (v1.5.2 targets Python ~3.7–3.9, older PyTorch/NumPy) — current system Python is 3.14, which is too new for v1.5.2's pinned deps.
2. `pip install chemprop==1.5.2` (or use the v1.5.2 source if obtainable).
3. Run the README's exact `chemprop_predict --test_path ... --checkpoint_dir .../SMILES_only_model --preds_path ...` command — no conversion needed, ensembles the 5 folds automatically.

**Path B is truer to the paper's exact numerics** (no conversion risk), but requires a separate, older Python environment. **Path A reuses what's already extracted** and Chemprop's own conversion utility, with the only caveat being the `--multi-hot-atom-featurizer-mode v1` flag and verifying converted predictions against the repo's own `data/stability_constant_25C_model_M2/test_input.csv` (which has known experimental log K1 values) as a sanity check before trusting predictions on new DES ligands.

### 2.3 Current local environment status

Checked what's already installed:
- `torch` 2.12.0+cpu ✅ (v2.2.3 needs `torch>=2.1`)
- `rdkit` 2026.03.2 ✅
- `pandas` 3.0.0 ✅
- `chemprop` ❌ not installed
- `lightning` ❌ not installed (required by v2.2.3)
- Python 3.14.3 — fine for v2.2.3 (`requires-python >=3.11,<3.15`), but **too new for Chemprop v1.5.2** (Path B would need a separate older-Python env).

---

## 3. The melting-point / DES dataset (for the upcoming task)

`C:\dev\PersonalPrediction\Simple ML Models\data\Melting_temperature_appended_35il_03082026.csv` (2006 rows) contains Deep Eutectic Solvent (DES) mixtures with columns:

```
Number of components, Type of DES, Component#1, Component#2,
X#1 (molar fraction), X#2 (molar fraction), Tmelt K, Phase diagram (Yes/No),
Reference (DOI), Smiles#1, T#1, Smiles#2, T#2
```

- **Component#1** is typically the HBA (hydrogen-bond acceptor) — often a quaternary ammonium/phosphonium salt (e.g., `CCCC[P+](c1ccccc1)(c1ccccc1)c1ccccc1.[Br-]`).
- **Component#2** is typically the HBD (hydrogen-bond donor) — often an alcohol, acid, amine, or amide (e.g., `OCCO`, `OCCN(CCO)CCO`, `CC(C)Cc1ccc(C(C)C(=O)O)cc1`).

To predict log K against Co²⁺/Ni²⁺, the workflow would be: extract candidate ligand SMILES (Smiles#1 and/or Smiles#2, likely filtering to species with donor atoms — O/N-containing groups capable of coordinating a metal), strip any counter-ions already present if needed, append `.[Co+2]` and `.[Ni+2]`, and run through the M2 `SMILES_only_model` ensemble (Path A or B above).

---

## 4. Summary of key findings

1. The stability-constant repo provides ready-to-use **5-fold Chemprop v1.5.2 ensembles**; **M2 (`SMILES_only_model`, 25 °C log K1)** is the simplest and best match for the planned Co/Ni-vs-DES-ligand prediction task — single SMILES input, no auxiliary feature files.
2. Input format = `<ligand_SMILES>.[Metal+n]` in one SMILES string, target column `Experimental LgK1`.
3. **The extracted "chemprop-main" is Chemprop v2.2.3, not v1.5.2 as the README assumes.** The v1.5.2 checkpoints are not directly loadable by v2.2.3's `chemprop predict`.
4. Two paths forward: (A) use v2.2.3's built-in `chemprop convert -c v1_to_v2` + `--multi-hot-atom-featurizer-mode v1` (reuses what's already downloaded, M2's single-molecule case is the well-supported conversion path), or (B) install the actual Chemprop v1.5.2 in a separate, older-Python environment and follow the README verbatim.
5. Local environment has `torch`/`rdkit`/`pandas` ready but is missing `chemprop` and `lightning`; Python 3.14 is compatible with v2.2.3 but not with v1.5.2.

---

*No code was modified or executed beyond read-only inspection of files, CSV headers, and package metadata. Nothing has been installed or converted yet — awaiting further instructions.*
