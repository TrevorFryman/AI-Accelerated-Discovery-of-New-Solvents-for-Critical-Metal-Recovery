# Final LogK Prediction Report — DES Ligands vs. Co²⁺ and Ni²⁺

## 1. Models used

- **Model:** `stability_constant_25C_model_M2`, `SMILES_only_model` (5-fold ensemble), from
  `stabilityconstant-ml-models-main`. Predicts log K1 (first stepwise stability constant, 25 °C) from
  a single SMILES string `<ligand>.[Metal+n]`.
- **Environment:** Legacy Chemprop **v1.5.2** in an isolated conda env (`chemprop152`, Python 3.9,
  `torch==2.5.1`, `rdkit==2025.9.2`) — run via the README's exact `chemprop_predict --checkpoint_dir`
  command, ensembling all 5 folds automatically.
- **Cross-validation (Approach A):** All 5 fold checkpoints were also converted to Chemprop **v2.2.3**
  format via `chemprop convert -c v1_to_v2`, and re-run with `--multi-hot-atom-featurizer-mode v1`.
  On the Co sanity-check set, v1 and v2 predictions agreed to **MAE ≈ 2.8×10⁻⁷** (effectively
  identical) — the conversion path is validated for this checkpoint.
- **Sanity check (known-answer validation):** Ran the model on the repo's own M2 test-set rows where
  the metal is Co²⁺ (149 rows) or Ni²⁺ (210 rows) and compared to `Experimental LgK1`:

  | Metal | n (rows after dedup) | MAE | RMSE | R² |
  |---|---|---|---|---|
  | Co²⁺ | 191 | 0.400 | 0.579 | 0.977 |
  | Ni²⁺ | 270 | 0.457 | 0.744 | 0.968 |

  These match the model's expected performance on its own held-out test data, confirming the
  environment and conversion are working correctly before scoring novel DES ligands.

## 2. Inputs used

- **Source:** `PersonalPrediction/Simple ML Models/data/Melting_temperature_appended_35il_03082026.csv`
  (2,006 DES rows, 83 unique HBA `Smiles#1` + 134 unique HBD `Smiles#2`).
- **Decomposition:** Each `Smiles#1`/`Smiles#2` entry was split on `.` into its constituent
  ions/molecules, yielding **181 unique fragments**.
- **Filtering (per `Prediction_Strategy.md`):** **17 fragments excluded** — fully-quaternized
  tetraalkyl/tetraaryl ammonium and phosphonium cations (all-carbon substituents, no extra donor
  heteroatoms), e.g. `CCCC[N+](CCCC)(CCCC)CCCC`, `C[P+](c1ccccc1)(c1ccccc1)c1ccccc1`. These have no
  chemically plausible coordination site for Co²⁺/Ni²⁺.
- **Retained:** **164 unique ligand fragments** — neutral HBD molecules (alcohols, acids, amides,
  amines, sugars, phenols), donor-bearing onium cations (e.g. choline, betaine, carnitine), and
  halide/pseudohalide anions (`[Cl-]`, `[Br-]`, etc., which the model has direct training experience
  with: 417 halide-metal rows in M2 training data).
- **Model input construction:** for each of the 164 ligands, built `<ligand>.[Co+2]` and
  `<ligand>.[Ni+2]`.

## 3. Outputs

- `Co_LogK_Predictions.csv` — 164 rows: `Ligand_SMILES, Metal, Model_Input_SMILES, LogK_mean,
  LogK_std, LgK1_model_0..4` (per-fold ensemble predictions + mean/std).
- `Ni_LogK_Predictions.csv` — same structure, 164 rows.
- `LogK_mean` = mean of the 5-fold ensemble prediction (the model's point estimate for log K1).
- `LogK_std` = standard deviation across the 5 folds — used as a per-prediction **confidence
  indicator** (smaller = folds agree = more reliable).

### Summary statistics

| | Co²⁺ | Ni²⁺ |
|---|---|---|
| Mean predicted log K1 | 1.72 | 1.93 |
| Std of predictions across ligands | 1.67 | 1.85 |
| Mean ensemble (fold) std | 0.41 | 0.43 |
| Range | −1.74 to 9.70 | −1.68 to 10.00 |
| Ligands with fold-std > 1.5 (low confidence) | 4 | 3 |

### Strongest predicted binders (both metals)

| Ligand SMILES | Co LogK | Ni LogK |
|---|---|---|
| `CN(C)[C@@H]1C(=O)C(C(N)=O)=C(O)[C@@]2(O)...` (tetracycline-like) | 9.70 | 10.00 |
| `O=C(O)/C=C/c1ccc(O)c(O)c1` (caffeic acid) | 8.33 (std 1.63, lower confidence) | 8.88 |
| `O=C(O)c1ccccc1O` (salicylic acid) | 7.07 | 7.50 |
| `c1cnc2c(c1)ccc1cccnc12` (phenanthroline-type) | 6.93 | 8.18 |
| `N[C@@H](CC(=O)O)C(=O)O` (aspartic acid) | 5.76 | 6.73 |

These rankings are chemically reasonable: catechol/phenol-carboxylate and aminodicarboxylic-acid
motifs (classic O/N chelators) and N,N-bidentate aromatic systems (phenanthroline-type) score highest
for both Co²⁺ and Ni²⁺, consistent with established coordination chemistry.

## 4. Assumptions

1. M2 `SMILES_only_model` (25 °C, ligand:metal = 1:1, aqueous-solution IUPAC data) is treated as the
   relevant reference condition for DES ligand screening — predictions estimate **intrinsic
   ligand-metal binding affinity in dilute aqueous solution**, not behavior within the DES medium
   itself.
2. Each DES is decomposed into independent constituent fragments; logK is predicted **per fragment**,
   not for the DES mixture as a whole (no ternary HBA+HBD+metal model exists or was trained).
3. Fully-quaternized onium cations (17 fragments) are assumed to be non-coordinating and excluded;
   all other fragments (164) are assumed to be plausible 1:1 ligands for log K1.
4. Halide/pseudohalide counter-ions are scored as standalone ligands (`[Cl-].[Co+2]`, etc.), matching
   the model's own training data format.

## 5. Limitations

- **No DES-mixture-level prediction**: results do not capture cooperative/competitive effects between
  HBA and HBD components, or the effect of the DES medium (vs. water) on metal speciation.
- **Single stepwise constant only**: M2 predicts log K1 (first complex only), not higher-order
  stepwise constants (K2, K3...) or overall stability constants (β values) relevant to some
  extraction equilibria.
- **Extrapolation risk**: large/complex DES-relevant molecules (e.g. sugars, long-chain
  carboxylic acids) may lie outside the chemical space densely sampled by the IUPAC training set
  (dominated by smaller chelating ligands); fold-to-fold std is the best available per-prediction
  signal for this, but is not a substitute for experimental validation.
- **Donor-atom filter is a heuristic**: the exclusion of 17 quaternary onium cations is based on a
  simple structural rule (fully-substituted N+/P+, all-carbon, no extra heteroatoms); borderline
  cases were not individually reviewed by a domain chemist.
- **25 °C / aqueous reference condition**: not the actual DES operating temperature/medium in many of
  the source melting-point dataset's mixtures.

## 6. Confidence in predictions

- **Environment/method confidence: high.** Sanity check against the model's own held-out test data
  (R² = 0.977 for Co, 0.968 for Ni) confirms the legacy v1.5.2 setup reproduces the published model's
  performance, and the v2.2.3 conversion was independently validated to match v1 predictions to
  ~3×10⁻⁷.
- **Per-ligand confidence: variable**, indicated by `LogK_std` (5-fold spread) in the output CSVs.
  Most ligands (≥75%) have fold-std < 0.6, indicating good ensemble agreement; a handful (4 for Co, 3
  for Ni) exceed 1.5 and should be treated as low-confidence/exploratory only.
- **Chemical plausibility: high** for the top-ranked ligands — predicted rankings align with known
  Co²⁺/Ni²⁺ coordination chemistry (catecholate, aminocarboxylate, and N,N-chelate motifs ranking
  highest).
- **Overall**: results are suitable for **screening/ranking** which DES components are likely strong
  vs. weak Co/Ni binders, and as a starting point for prioritizing candidates for experimental
  follow-up — not as substitutes for measured stability constants.

---
*Generated files: `Co_LogK_Predictions.csv`, `Ni_LogK_Predictions.csv`, this report, and supporting
intermediate files under `predictions/` and `candidate_ligand_fragments_full.csv`.*
