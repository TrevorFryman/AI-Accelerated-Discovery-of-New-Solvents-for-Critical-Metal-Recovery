# Dataset Assessment

## 1. Repository datasets (`stabilityconstant-ml-models-main/data/`)

All four model datasets use the format `SMILES,<target>` where `SMILES` = `<ligand>.<[Metal+n]>`
(plus optional extra counter-ion fragments). No missing SMILES or target values found in M2's
train/test/validation files (spot-checked).

| Dataset | Rows (train/val/test) | Columns | SMILES | Metal identifiers | Co rows | Ni rows |
|---|---|---|---|---|---|---|
| **M1** (`stability_constan_model_M1`) | 28,899 / 348 / 3,212 | `SMILES, Experimental Lg K1` + parallel `*_features.csv` (Temp, Ionic strength) | combined ligand+metal SMILES | bracketed ions, e.g. `[Cd+2]`, `[Co+2]`, `[Ni+2]` | 1,926 (train) | 2,776 (train) |
| **M2** (`stability_constant_25C_model_M2`) | 20,958 / 253 / 2,329 | `SMILES, Experimental LgK1` (no extra features) | combined ligand+metal SMILES | same scheme | 1,355 (train), 149 (test), 21 (val) | 2,080 (train), 210 (test), 23 (val) |
| **M3** (`protonation_constant_model_M3`) | 10,323 / 50 / 1,148 | `SMILES, Experimental LgK1` + 46-column `*_features.csv` (charges, experimental conditions, 40 RDKit descriptors) | combined ligand+`[H+]` (protonation, not metal) | mostly `[H+]` | n/a (protonation data) | n/a |
| **M4** (`stability_constant_protonation_constant_model_M4`) | 39,050 / 398 / 4,338 | `cmplx_smiles_neutral_std, Exp LgK1` + 16-column `*_features.csv` (metal physicochemical properties + experimental conditions) | combined ligand+metal/`[H+]` | mix of metal ions and `[H+]` | not separately counted | not separately counted |

**M2 metal ion distribution (top entries, train set, 20,958 rows):**

| Metal | Train | Test | Val (v53) |
|---|---|---|---|
| Cu²⁺ | 3,478 | 407 | 29 |
| **Ni²⁺** | **2,080** | **210** | **23** |
| Zn²⁺ | 1,826 | 191 | 20 |
| **Co²⁺** | **1,355** | **149** | **21** |
| Cd²⁺ | 1,330 | 137 | 11 |
| Pb²⁺ | 711 | 79 | — |
| Ca²⁺ | 673 | 68 | 12 |
| Mn²⁺ | 547 | 59 | 10 |
| Mg²⁺ | 502 | 60 | 13 |
| Ag⁺ | 478 | 61 | — |

**Suitability for logK prediction:** M2's `SMILES_only_model` is directly suited — it was trained on
this exact task (ligand+metal SMILES → log K1 at 25 °C), with **good representation of both Co²⁺
(~1,500 total examples) and Ni²⁺ (~2,300 total examples)** across a wide range of ligand chemistries
(amines, carboxylic acids, alcohols, amides, phosphonates, etc., per `data/applications/*`).

## 2. `Melting_temperature_appended_35il_03082026.csv` (DES dataset)

- **Location:** `PersonalPrediction/Simple ML Models/data/Melting_temperature_appended_35il_03082026.csv`
  (most recent version; earlier copies `Melting_temperature_appended.csv` (2,254 rows) and
  `Melting_temperature_appended_35il.csv` (2,107 rows) exist under `mp_prediction-main/` but are
  superseded by this dated file).
- **Rows:** 2,006 (excluding header)
- **Columns:** `Number of components, Type of DES, Component#1, Component#2, X#1 (molar fraction),
  X#2 (molar fraction), Tmelt K, Phase diagram (Yes/No), Reference (DOI), Smiles#1, T#1, Smiles#2, T#2`
- **`Type of DES` distribution:** `3` → 1,142 rows, `5` → 673 rows, `IL` (ionic liquid) → 191 rows
- **SMILES availability:** `Smiles#1` and `Smiles#2` both **0 missing** (100% coverage)
- **Unique ligand SMILES:** 83 unique `Smiles#1` (typically HBA — quaternary ammonium/phosphonium
  salts, often with counter-ions like `[Br-]`), 134 unique `Smiles#2` (typically HBD — alcohols,
  acids, amides, amines, sugars)
- **Metal identifiers:** **none** — this dataset describes binary DES mixtures and their melting
  points (`Tmelt, K`, 0 missing), with no metal-ion or stability-constant information.
- **Missing data:** none in the columns inspected (SMILES, Tmelt).
- **A pre-computed RDKit descriptor file exists**: `data/RDKitDescriptorGeneration/DES_RDKit_Features.csv`
  — same rows plus 14 per-component RDKit descriptors (`C1_MolWt, C1_LogP, C1_TPSA, C1_HBD, C1_HBA,
  C1_RotBonds, C1_RingCount` and `C2_*` equivalents). Useful for characterizing HBA/HBD chemistry but
  not for logK directly.
- **Suitability for logK prediction:** **Not directly usable as training/labeled data** for logK (no
  metal, no stability constant). Its role is as the **source of candidate ligand structures** (the 83
  + 134 unique HBA/HBD SMILES) to be fed into the M2 logK model after appending `[Co+2]`/`[Ni+2]`.

## 3. `GSK_dataset.csv`

- **Locations:** multiple copies — `PersonalPrediction/Greenness ML Models/GSK_dataset.csv`,
  `.../Final Model/data/GSK_dataset.csv`, `.../Track 1/data/GSK_dataset.csv`, and
  `green_solvents-main/GSK_dataset.csv` (all appear to be the same 154-row file).
- **Rows:** 154
- **Columns:** `Unnamed: 0 (index), Classification, solvent_common_name, IPUAC name, solvent_SMILES,
  CAS Number, G-score`
- **SMILES availability:** `solvent_SMILES` — 0 missing (100% coverage)
- **Metal identifiers:** none
- **Missing data:** none in `solvent_SMILES` or `G-score` (spot-checked)
- **Suitability for logK prediction:** This is the **GSK solvent-selection-guide "greenness" dataset**
  (used elsewhere in `PersonalPrediction/Greenness ML Models/` for green-solvent scoring, an unrelated
  project track). It contains general solvent SMILES and a greenness score (`G-score`), **not
  stability constants, and not specifically DES components**. It is **not relevant to Co/Ni logK
  prediction** except as a possible (optional, low-priority) source of additional generic ligand
  SMILES if broader coverage were ever desired — not recommended for the current DES-focused goal.

## 4. Summary table

| Dataset | Entries | SMILES coverage | Metal info | logK labels | Role in this project |
|---|---|---|---|---|---|
| M2 train/test/val (IUPAC, 25 °C) | 20,958 / 2,329 / 253 | 100% | Yes (incl. Co²⁺, Ni²⁺) | Yes | **Model training data — already used to produce the M2 checkpoints; basis for inference** |
| M1, M3, M4 datasets | 28.9k–39k each | 100% | Yes / `[H+]` | Yes | Out of scope (see Compatibility Report) |
| `Melting_temperature_appended_35il_03082026.csv` | 2,006 | 100% (Smiles#1 & #2) | None | None | **Source of candidate DES ligand SMILES (83 HBA + 134 HBD)** |
| `DES_RDKit_Features.csv` | 2,006 | 100% | None | None | Optional descriptor reference for ligand chemistry, not logK input |
| `GSK_dataset.csv` | 154 | 100% | None | None | Not relevant — different project track (green solvent scoring) |

---
*Status: read-only assessment complete. No models run, nothing installed.*
