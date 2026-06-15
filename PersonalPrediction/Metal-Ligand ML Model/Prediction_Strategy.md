# DES Ligand → Co/Ni LogK Prediction Strategy

## 1. Constraints set by the model (M2 `SMILES_only_model`)

The only model identified as scientifically usable (see `Repository_Assessment.md` and
`Chemprop_Compatibility_Report.md`) takes a **single SMILES string per row**, representing **one
ligand fragment + one metal-ion fragment** (and, in the training data, occasionally extra
counter-ion fragments — e.g. `[Cl-].[Co+2]`, `40` extra-fragment rows seen in M2 data). It predicts a
single scalar: log K1 (25 °C, first stepwise stability constant) for that ligand/metal pair. **It does
not model ternary systems (DES mixture + metal) and was not trained on multi-component DES
"solvent" molecules** — every prediction is for one discrete chemical species (or species + simple
counter-ion) binding one metal ion.

This shapes everything below: the task is **decomposition** — turn each DES into its constituent
chemical species, identify which of those species are plausible metal-binding ligands, and score each
one individually against `[Co+2]` and `[Ni+2]`.

## 2. DES composition (from `Melting_temperature_appended_35il_03082026.csv`)

- 83 unique **Component#1** SMILES (conventionally the HBA), 134 unique **Component#2** SMILES
  (conventionally the HBD); 36 SMILES appear in both roles.
- **36/83 HBA SMILES are salts** (two `.`-separated fragments): a quaternary ammonium/phosphonium
  cation + a halide or pseudo-halide anion (`[Cl-]`, `[Br-]`, `[SCN-]`-type, etc.), e.g.
  `CCCC[P+](c1ccccc1)(c1ccccc1)c1ccccc1.[Br-]`.
- **17/134 HBD SMILES are also salts** (e.g. `N#C[S-].[NH4+]`).
- The remaining HBA/HBD SMILES are neutral organic molecules (glycols, sugars, amides, acids, phenols,
  amines, etc.).

## 3. HBA representation

Split each HBA SMILES on `.` into its constituent ions/molecules and evaluate each independently:

- **Quaternary onium cations** (`[N+]`/`[P+]` with four C-substituents, e.g. tetrabutylammonium,
  tetraphenylphosphonium): **exclude as ligands**. They have no lone-pair donor atoms accessible for
  metal coordination (fully substituted, permanently cationic centers) and are not chemically
  plausible Co²⁺/Ni²⁺ binders. Predicting logK for these would be extrapolating the model far outside
  its chemical domain and is not scientifically meaningful.
- **Exception — onium cations bearing an extra donor group** (e.g. choline, `C[N+](C)(C)CCO.[Cl-]`,
  which carries a free –OH): include the **whole cation fragment** (`C[N+](C)(C)CCO`) as a candidate
  ligand — it has a real coordinating oxygen.
- **Halide / pseudo-halide anions** (`[Cl-]`, `[Br-]`, `[I-]`, thiocyanate, etc.): **include as
  ligands**. M2's training data contains 417 rows of exactly this form (`[Cl-].[Pb+2]`,
  `[Cl-].[Cu+2]`, etc.), so the model has direct experience with halide-metal log K1 — these are
  legitimate, well-supported predictions.
- General rule for any HBA fragment: include it as a candidate ligand if it contains at least one
  N/O/S/P atom that is **not** part of a permanently quaternized, fully-substituted onium center
  (simple heuristic: RDKit formal-charge + valence check, or a curated allow/deny list given only 83
  HBAs).

## 4. HBD representation

HBD components (alcohols, glycols, carboxylic acids, amides, amines, sugars, phenols) are
**overwhelmingly neutral, donor-atom-rich organic molecules** — exactly the chemical space M2 was
trained on (its training set is dominated by small organic acids/amines/alcohols complexing
transition metals). For the 17/134 that are salts (e.g. `N#C[S-].[NH4+]`), split fragments the same
way as HBAs: keep the anion/molecule with donor atoms (`[S-]`/thiocyanate here), drop ammonium
(`[NH4+]` has no extra donor beyond the coordinating N already counted via thiocyanate — and `[NH4+]`
itself is rarely a ligand).

**Recommendation:** treat essentially all HBD fragments (after stripping non-donor counter-cations
like `[NH4+]`, `[Na+]`, `[K+]`) as candidate ligands — this is the model's core competency.

## 5. Combined DES representation

A true "DES + metal" ternary prediction (capturing how HBA and HBD jointly modulate metal binding in
the eutectic mixture) is **not supported by any available model** — it would require a dataset and
model trained on DES-metal systems specifically, which does not exist here. Two fallback framings,
both **per-component, not combined**:

1. **Component-wise screening (recommended):** predict log K1 for each candidate ligand fragment
   (from HBA and HBD, deduplicated) independently against Co²⁺ and Ni²⁺. This tells you, for a given
   DES, which of its constituent species are the stronger/weaker metal binders — directly useful for
   reasoning about metal extraction/leaching behavior of the DES as a whole, without claiming to
   predict the mixture's emergent behavior.
2. **DES-level rollup (optional, post-hoc):** after per-component predictions exist, a DES record in
   the melting-point table can be annotated with the logK of its Component#1 and Component#2
   ligand(s) for Co/Ni — letting downstream analysis correlate DES composition with predicted
   metal-binding strength. This is an aggregation step on top of (1), not a new model.

**Do not** attempt to encode the full DES SMILES (HBA·HBD as one multi-fragment string with a metal
appended, e.g. `<HBA>.<HBD>.[Co+2]`) as a single M2 input — this is a 3+ fragment "molecule" the model
has essentially no training examples like, and the resulting number is not interpretable as a
stability constant.

## 6. Deduplication and final candidate list

1. Collect all unique fragments from `Smiles#1` and `Smiles#2` (split on `.`).
2. Deduplicate across the combined HBA+HBD set (36 SMILES overlap already).
3. Apply the donor-atom filter from §3/§4 (drop fully-quaternized onium cations without extra donors;
   keep everything else: neutral organics, halides/pseudohalides, donor-bearing onium cations).
4. For each surviving unique ligand fragment, generate two input rows:
   `<ligand_SMILES>.[Co+2]` and `<ligand_SMILES>.[Ni+2]`.
5. Run both lists through the M2 `SMILES_only_model` 5-fold ensemble (per
   `Chemprop_Compatibility_Report.md` recommendation).

## 7. Metal-specific considerations

- M2's training data has good coverage of both metals (Co²⁺: 1,355 train / 149 test / 21 val; Ni²⁺:
  2,080 train / 210 test / 23 val) — both are well-represented relative to most other metals in the
  dataset (only Cu²⁺ and Zn²⁺ have more examples), supporting reasonable confidence for both `[Co+2]`
  and `[Ni+2]` predictions without metal-specific caveats beyond the usual ligand-domain caveats.
- Both metals are divalent first-row transition-metal ions with broadly similar (though not
  identical) coordination chemistry — predictions for the same ligand against Co vs. Ni are expected
  to be correlated but not identical; report both independently rather than averaging.
- Co/Ni-specific subsets of the M2 test set (`test_input.csv`, 149/210 rows respectively) should be
  used as a **sanity-check baseline**: run the chosen environment on these known-answer rows first to
  confirm predicted vs. experimental log K1 are reasonably close before trusting predictions on novel
  DES ligands.

## 8. Recommended end-to-end methodology (summary)

1. Build the legacy Chemprop v1.5.2 environment (Approach B), validated against Approach A for M2
   `SMILES_only_model` per `Chemprop_Compatibility_Report.md`.
2. Sanity-check: run M2 ensemble on `data/stability_constant_25C_model_M2/test_input.csv` rows where
   the metal is `[Co+2]` or `[Ni+2]`; compare to `Experimental LgK1`.
3. From `Melting_temperature_appended_35il_03082026.csv`, extract and deduplicate HBA/HBD fragments,
   apply the donor-atom filter (§3/§4), producing a final candidate ligand SMILES list.
4. Build `<ligand>.[Co+2]` and `<ligand>.[Ni+2]` input CSVs.
5. Run the M2 ensemble on both → `Co_LogK_Predictions.csv`, `Ni_LogK_Predictions.csv`.
6. Document assumptions/exclusions (which HBA cations were dropped and why) and ensemble
   spread (fold-to-fold std dev) as a per-prediction confidence indicator in the final report.

---
*Status: read-only assessment complete. No models run, nothing installed, no predictions generated.*
