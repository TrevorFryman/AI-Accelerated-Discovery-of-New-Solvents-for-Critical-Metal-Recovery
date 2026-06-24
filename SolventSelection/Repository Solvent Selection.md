# Repository Solvent Selection

## Executive Summary

This document identifies the top 10 deep eutectic solvent (DES) candidates for removing cobalt and nickel from wastewater, selected from the repository's full prediction outputs (2,006 melting-point pairs, 158 metal-binding ligands, and greenness scored across the entire pool) with metal-binding affinity weighted as the dominant criterion. The top candidate, **1,10-Phenanthroline + Thymol**, combines the strongest predicted Co/Ni chelator in the dataset with a confirmed room-temperature-liquid formulation and above-average greenness (71st percentile). All 10 candidates have both components independently characterized for metal-binding, are liquid somewhere in a 275–343 K operating window, and score 68th–73rd percentile on greenness — there is no meaningful three-way trade-off to manage in this chemical space. One important caveat for the next stage: every candidate's melting point is an **already-published literature measurement** (not a speculative model extrapolation — see Limitations #2), meaning this analysis is best understood as a triage of known DES chemistry for a new application, not a discovery of untested compositions; closing that gap is the most valuable next step before claiming novel solvent discovery.

## Purpose

This document selects the **top 10 deep eutectic solvent (DES) candidates** for **removing cobalt (Co²⁺) and nickel (Ni²⁺) ions from wastewater**, drawn from the **full prediction outputs** of all three ML models in this repository — not the curated top-25 shortlists used in earlier rounds:

- **Melting Point model** (RDKit-feature variant): `C:\dev\PersonalPrediction\Simple ML Models\Optimized Model\CSVs\Candidate_Master_List_rdkit.csv` — **2,006 DES pairs**, each screened for physical validity (`passes_DES_rule`) and liquidity at room temperature (275–298 K) or warm-process temperature (298–343 K).
- **Metal-Binding model**: `C:\dev\PersonalPrediction\Metal-Ligand ML Model\Co_LogK_Predictions.csv` and `Ni_LogK_Predictions.csv` — **158 unique candidate ligand fragments**, each with a predicted stability constant (log K₁) for Co²⁺ and Ni²⁺ binding individually.
- **Greenness model**: `C:\dev\PersonalPrediction\Greenness ML Models\Final Model\results\top25_DES_candidates.csv` — only **25 DES pairs** have a greenness score at all (the model was never run against the full 2,006-pair melting-point universe). This is a real data-coverage gap, not an oversight on my part — see Limitations below.

Per your instruction, **metal-binding affinity was treated as the dominant selection criterion**, since the end use is direct Co/Ni recovery from wastewater. Melting point was used as a hard feasibility filter (the DES must actually be liquid), not as a ranking weight.

## Methodology

**Step 1 — Resolve every component.** The 2,006 MP pairs are built from 205 unique molecules. All 205 were resolved to SMILES/common names via PubChem (cross-referenced in this analysis; not re-exported to `ChemicalNames.xlsx`, which only covers the original 48-component curated shortlist).

**Step 2 — Cross-reference against the Metal-Binding pool.** Each of the 205 MP components was matched (by molecular structure, via InChIKey) against the 158 Metal-Binding ligands. **152 of the 205 MP components — a large majority — have a predicted Co/Ni binding affinity.** This is a much richer overlap than was visible in the curated top-25 shortlists used in earlier rounds, where only Thymol and Lidocaine matched.

**Step 3 — Hard filter.** Kept only MP pairs where `passes_DES_rule = True` AND (`RoomTemp_flag = True` OR `WarmTemp_flag = True`) — i.e., physically valid eutectic mixtures predicted to be liquid somewhere in a realistic 275–343 K operating window. **724 of 2,006 pairs survive this filter**, and at least one component has a Metal-Binding prediction.

**Step 4 — Score by metal-binding strength.**
- `LogK_mean_both` = average of the predicted Co²⁺ and Ni²⁺ log K₁ for a ligand (this is the same definition the repository's own `Top25_Candidates.csv` uses).
- For each DES pair, **`metal_score` = `LogK_mean_both`(component 1) + `LogK_mean_both`(component 2)** when both components have a Metal-Binding prediction, or just the one value when only one component does.
- **294 of the 724 feasible pairs have *both* components characterized for metal binding** — a striking result; none of the curated top-25-based reports surfaced this because that smaller dataset only covered 38 unique molecules.

**Step 5 — Rank.** Candidates were sorted **(a) both-components-characterized first, (b) not flagged low-confidence (`LogK_std > 1.5`, the repository's own documented threshold), (c) `metal_score` descending.** This mirrors the breadth-then-score logic used in the earlier ML/ML2/ML3 reports, adapted so "breadth" now means "both halves of the DES validated for binding" rather than "validated by multiple model types" (which greenness's sparse coverage makes impossible here).

**Step 6 — Deduplicate.** The master list tests many DES pairs at several molar ratios, and a few molecules appear under two different name strings (e.g., "Lidocaine" and its full IUPAC name "2-(diethylamino)-N-(2,6-dimethylphenyl)acetamide" are the same compound — confirmed by InChIKey — and were merged). The **top 10 below are 10 chemically distinct pairs**, each shown with every molar ratio tested for it.

**Important caveat on the Metal-Binding numbers themselves** (documented in the repository's own `Final_LogK_Prediction_Report.md`): predictions are **log K₁ stability constants for the standalone ligand in dilute aqueous solution at 25 °C** — not a simulation of the ligand actually behaving as one half of a DES, and not at whatever temperature the DES would operate at. The model's own sanity-check performance is strong (R² = 0.977, MAE 0.40 for Co; R² = 0.968, MAE 0.457 for Ni), but treat the absolute log K values as a *screening signal for which fragments are strong chelators*, not as a guarantee of in-DES extraction performance.

---

## 1. 1,10-Phenanthroline + Thymol

**Chemical name:** 1,10-Phenanthroline — dipyrido[3,2-a:2',3'-c]phenazine-free bidentate diimine; paired with Thymol — 5-methyl-2-(propan-2-yl)phenol

**Common names:** 1,10-Phenanthroline (also: o-phenanthroline, phen); Thymol (also: 2-isopropyl-5-methylphenol)

**SMILES:**
- 1,10-Phenanthroline: `c1cnc2c(c1)ccc1cccnc12`
- Thymol: `Cc1ccc(C(C)C)c(O)c1`

**Metal-binding data:** 1,10-Phenanthroline ranks **#3 of 158** in the full Metal-Binding pool (log K₁ mean = 7.55; Co = 6.93, Ni = 8.18; 2 donor atoms — the two pyridine-type nitrogens, the textbook bidentate chelation mode for this molecule). Thymol ranks **#21 of 158** (log K₁ mean = 3.63; Co = 3.40, Ni = 3.86; 1 donor atom). **Combined metal score: 11.18 — the highest in the entire dataset.**

**Position on melting-point list:** Tested at 4 molar ratios in the full MP master list (within the 275–343 K screening window; 4 additional richer-phenanthroline ratios were also tested but predicted above 343 K and excluded). Predicted Tmelt ranges from 283.3 K to 321.3 K. **The 20.3:79.7 (Phenanthroline:Thymol) formulation lands at 283.3 K — inside the room-temperature window** (`RoomTemp_flag = True`); the other three (mole fractions 10.9:89.1, 49.2:50.8, and 40.5:59.5 Phenanthroline:Thymol) fall in the 300–321 K warm-process range.

**Why it was selected:**
This is both the highest-scoring candidate by a wide margin and the most chemically well-grounded one. 1,10-Phenanthroline is a textbook strong chelator for first-row transition metal divalent cations — it forms the classic, very stable tris-chelate complex [M(phen)₃]²⁺ with both Co²⁺ and Ni²⁺ — so its #3-of-158 ranking is exactly what real coordination chemistry predicts, which is a meaningful sanity check on the ML model's outputs rather than an artifact. Pairing it with thymol (itself a modest, room-temperature-compatible HBD) produces a DES that is liquid at standard ambient conditions in at least one formulation, with no melting-point uncertainty flag.

| Molar ratio (Phenanthroline:Thymol) | Predicted Tmelt (K) | Screening status |
|---|---|---|
| 0.203 : 0.797 | 283.3 | **PASS_ROOMTEMP** |
| 0.109 : 0.891 | 300.5 | PASS_WARMTEMP |
| 0.492 : 0.508 | 317.6 | PASS_WARMTEMP |
| 0.405 : 0.595 | 321.3 | PASS_WARMTEMP |

**Caveats:** Phenanthroline is a solid heterocycle, not a typical "green," commodity DES building block — it is more expensive and less biodegradable than common HBAs/HBDs, and it has its own toxicity profile that should be assessed before bulk handling. It is not one of the 25 pairs the Greenness model has scored. Treat this as the strongest *binding* candidate, with sustainability/cost trade-offs to be weighed explicitly at the experimental stage.

---

## 2. 1,10-Phenanthroline + Chlorothymol

**Chemical name:** 1,10-Phenanthroline; paired with Chlorothymol — 4-chloro-5-methyl-2-(propan-2-yl)phenol

**Common names:** 1,10-Phenanthroline; Chlorothymol (a chlorinated thymol derivative, used historically as an antiseptic/preservative)

**SMILES:**
- 1,10-Phenanthroline: `c1cnc2c(c1)ccc1cccnc12`
- Chlorothymol: `Cc1cc(O)c(C(C)C)cc1Cl`

**Metal-binding data:** Phenanthroline (#3/158, log K₁ both = 7.55, as above). Chlorothymol ranks **#24 of 158** (log K₁ mean = 3.23; Co = 3.05, Ni = 3.42; 2 donor atoms — the phenolic O plus the chlorine substituent providing a secondary, weaker coordination site). **Combined metal score: 10.79 — second highest overall.**

**Position on melting-point list:** Tested at 5 molar ratios (within the screening window; 2 additional phenanthroline-rich ratios were also tested but predicted above 343 K and excluded); predicted Tmelt ranges 297.2–328.6 K. The 50.2:49.8 (Phenanthroline:Chlorothymol) formulation lands at 297.2 K, just inside the room-temperature boundary (`RoomTemp_flag = True`); the rest (mole fractions 59.8:40.2, 70.4:29.6, 20.9:79.1, and 10.6:89.4 Phenanthroline:Chlorothymol) are warm-process range.

**Why it was selected:**
Essentially the same chemistry as candidate #1, with thymol's chlorinated analog as the partner. Chlorothymol's chlorine substituent is a known feature in coordination chemistry — it can mildly perturb the phenolic oxygen's donor strength and, in mixed-ligand systems, occasionally aids selectivity — but the model puts it slightly behind plain thymol here. Useful as a structural analog/backup to candidate #1, worth carrying into experimental screening as a comparison point for how the halogen substituent affects actual extraction performance, which the standalone-ligand model cannot directly capture.

| Molar ratio (Phenanthroline:Chlorothymol) | Predicted Tmelt (K) | Screening status |
|---|---|---|
| 0.502 : 0.498 | 297.2 | PASS_ROOMTEMP (borderline) |
| 0.598 : 0.402 | 302.2 | PASS_WARMTEMP |
| 0.704 : 0.296 | 308.1 | PASS_WARMTEMP |
| 0.209 : 0.791 | 320.8 | PASS_WARMTEMP |
| 0.106 : 0.894 | 328.6 | PASS_WARMTEMP |

**Caveats:** Same cost/greenness/toxicity considerations as candidate #1, plus chlorothymol carries an organochlorine moiety, which is generally a negative for "green chemistry" framing and wastewater-discharge safety — worth weighing against candidate #1 if only one phenanthroline-based system is carried forward.

---

## 3. Lidocaine + Phenyl Salicylate

**Chemical name:** Lidocaine — 2-(diethylamino)-N-(2,6-dimethylphenyl)acetamide; paired with Phenyl salicylate — phenyl 2-hydroxybenzoate

**Common names:** Lidocaine (also: Xylocaine); Phenyl salicylate (also: Salol)

**SMILES:**
- Lidocaine: `CCN(CC)CC(=O)Nc1c(C)cccc1C`
- Phenyl salicylate: `O=C(Oc1ccccc1)c1ccccc1O`

**Metal-binding data:** Lidocaine ranks **#23 of 158** (log K₁ mean = 3.33; Co = 3.26, Ni = 3.41; 3 donor atoms — the tertiary amine N and amide O/N). Phenyl salicylate ranks **#6 of 158** (log K₁ mean = 5.88; Co = 5.66, Ni = 6.09; 3 donor atoms — the phenolic O, ester O, and carbonyl O). **Combined metal score: 9.21 — third highest, and the strongest *room-temperature* candidate on this list.**

**Position on melting-point list:** Only one molar ratio tested (43:57 Lidocaine:Phenyl salicylate); predicted Tmelt = 291.7 K, which is **comfortably inside the room-temperature window** (`PASS_ROOMTEMP`).

**Why it was selected:**
The best combination of high metal-binding score and confirmed room-temperature liquidity (no warm-process heating required) in the dataset — every candidate scoring higher than this one is warm-process-only at its best-tested ratio. Both components are well-characterized, commercially available pharmaceutical-grade compounds, which is favorable for sourcing and reproducibility in experimental follow-up.

**Caveats:** The melting-point prediction here carries **higher-than-typical uncertainty (±24.6 K) and a notably low training-set similarity score (0.42)** — meaning this specific mixture sits closer to the edge of (or outside) the melting-point model's reliable domain, and the true Tmelt could plausibly fall outside the room-temperature band. Only one molar ratio was tested, so there's no internal cross-check the way candidates #1/#2 have across multiple ratios. Treat the room-temperature classification as promising but lower-confidence, and prioritize confirming actual Tmelt experimentally before relying on it.

---

## 4. Menthol + Salicylic Acid

**Chemical name:** Menthol — 5-methyl-2-(propan-2-yl)cyclohexan-1-ol; paired with Salicylic acid — 2-hydroxybenzoic acid

**Common names:** Menthol; Salicylic acid

**SMILES:**
- Menthol: `CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O`
- Salicylic acid: `O=C(O)c1ccccc1O`

**Metal-binding data:** This menthol stereoisomer ranks **#118 of 158** (log K₁ mean = 0.76; Co = 0.71, Ni = 0.80; 1 donor atom — a weak binder, as expected for a simple secondary alcohol with no aromatic or conjugated system). Salicylic acid ranks **#4 of 158** (log K₁ mean = 7.28; Co = 7.07, Ni = 7.50; 3 donor atoms — phenolic O and carboxylate O,O — a strong, well-known chelator). **Combined metal score: 8.04.**

**Position on melting-point list:** Only one molar ratio tested (85:15 Menthol:Salicylic acid); predicted Tmelt = 298.0 K — right at the room/warm boundary, classified `PASS_ROOMTEMP`.

**Why it was selected:**
Salicylic acid is the real driver here — it is one of the strongest binders in the entire 158-ligand pool, and salicylate-based DES/ionic-liquid systems are reasonably well precedented in the metal-extraction literature, which gives this pairing more real-world grounding than its score alone suggests. Menthol itself contributes essentially no binding capacity, so this candidate functions chemically as "a strong single chelator dissolved in a low-melting carrier," which is a different design logic than the dual-chelator phenanthroline pairs above.

**Caveats:** Same low-similarity, high-uncertainty pattern as candidate #3 (similarity 0.49, uncertainty ±23.6 K) — only one ratio tested, near the room/warm boundary, so confirm the actual melting point experimentally before assuming reliable room-temperature operation.

---

## 5. Lidocaine + Tetracaine

**Chemical name:** Lidocaine — 2-(diethylamino)-N-(2,6-dimethylphenyl)acetamide; paired with Tetracaine — 2-(dimethylamino)ethyl 4-(butylamino)benzoate

**Common names:** Lidocaine; Tetracaine (also: Amethocaine)

**SMILES:**
- Lidocaine: `CCN(CC)CC(=O)Nc1c(C)cccc1C`
- Tetracaine: `CCCCNc1ccc(C(=O)OCCN(C)C)cc1`

**Metal-binding data:** Lidocaine (#23/158, log K₁ both = 3.33, as above). Tetracaine ranks **#16 of 158** (log K₁ mean = 4.16; Co = 3.90, Ni = 4.42; 4 donor atoms — two amine nitrogens plus the ester carbonyl/ether oxygens). **Combined metal score: 7.50.**

**Position on melting-point list:** Only one molar ratio tested (50:50); predicted Tmelt = 303.3 K (`PASS_WARMTEMP`).

**Why it was selected:**
A "two local-anesthetic" eutectic — both components are well-characterized, low-toxicity-relative-to-many-alternatives pharmaceutical compounds, and both independently carry real (if moderate) metal-binding capacity, making this a genuinely dual-validated pairing rather than one strong binder propped up by an inert partner. This is the same logic family as candidate #14 (Lidocaine + Camphor) carried over from the earlier curated-list reports, but with a stronger-binding partner this time.

**Caveats:** Like #3 and #4, only one ratio was tested and the model's confidence is moderate (uncertainty ±25.4 K, similarity 0.37 — the lowest similarity score in the top 10). This pairing sits furthest outside the melting-point model's comfortable training domain of any top-10 candidate; treat the warm-process classification as a starting hypothesis to verify, not a settled result.

---

## 6. Menthol + Phenyl Salicylate

**Chemical name:** Menthol — 5-methyl-2-(propan-2-yl)cyclohexan-1-ol; paired with Phenyl salicylate — phenyl 2-hydroxybenzoate

**Common names:** Menthol; Phenyl salicylate (Salol)

**SMILES:**
- Menthol: `CC1CCC(C(C)C)C(O)C1`
- Phenyl salicylate: `O=C(Oc1ccccc1)c1ccccc1O`

**Metal-binding data:** This menthol stereoisomer ranks **#132 of 158** (log K₁ mean = 0.43 — a very weak binder). Phenyl salicylate (#6/158, log K₁ both = 5.88, as above). **Combined metal score: 6.31.**

**Position on melting-point list:** Only one molar ratio tested (50:50); predicted Tmelt = 296.2 K (`PASS_ROOMTEMP`).

**Why it was selected:**
The second-best confirmed room-temperature candidate on this list (after #3), again following the "one strong chelator + low-melting carrier" design pattern, this time with phenyl salicylate as the active binder. Worth carrying alongside candidate #3 (which shares the same phenyl salicylate component) as a direct comparison of which carrier molecule (Lidocaine vs. Menthol) gives better real-world DES behavior once tested experimentally.

**Caveats:** Same caveat pattern as #3–#5: moderate-to-low similarity (0.43) and elevated uncertainty (±24.4 K), single ratio tested. Menthol's near-zero binding contribution means this candidate's metal-removal performance will be almost entirely determined by the phenyl salicylate half — any loss of that component (e.g., through leaching, hydrolysis of the ester) would likely sharply reduce performance.

---

## 7. Betaine + Malic Acid

**Chemical name:** Betaine — 2-(trimethylazaniumyl)acetate; paired with Malic acid — 2-hydroxybutanedioic acid

**Common names:** Betaine (also: Glycine betaine, Trimethylglycine); Malic acid

**SMILES:**
- Betaine: `C[N+](C)(C)CC(=O)[O-]`
- Malic acid: `O=C(O)CC(O)C(=O)O`

**Metal-binding data:** Betaine ranks **#69 of 158** (log K₁ mean = 1.66; Co = 1.53, Ni = 1.78; 3 donor atoms — the carboxylate, though the quaternary ammonium center itself cannot donate). Malic acid ranks **#17 of 158** (log K₁ mean = 4.02; Co = 3.78, Ni = 4.26; 5 donor atoms — two carboxylates plus a hydroxyl, a classic polydentate organic-acid chelator). **Combined metal score: 5.67.**

**Position on melting-point list:** Only one molar ratio tested (50:50); predicted Tmelt = 318.4 K (`PASS_WARMTEMP`).

**Why it was selected:**
The most "green-leaning" pairing in the top 10 by chemical identity — betaine and malic acid are both naturally occurring, biodegradable, low-toxicity, commodity-scale food/agriculture compounds, and betaine-based NADES (natural deep eutectic solvents) are well represented in the existing green-solvent literature. Even though this exact pair isn't one of the 25 the Greenness model scored, its components are individually strong sustainability candidates, which is a meaningful qualitative signal even without a quantitative greenness number for this specific mixture.

**Caveats:** Like several candidates above, only one ratio was tested, with moderate model confidence (uncertainty ±24.0 K, similarity 0.46). It requires warm-process conditions (no room-temperature ratio identified in the data available), which adds energy cost relative to candidates #3 and #6.

---

## 8. Lidocaine + Prilocaine

**Chemical name:** Lidocaine — 2-(diethylamino)-N-(2,6-dimethylphenyl)acetamide; paired with Prilocaine — N-(2-methylphenyl)-2-(propylamino)propanamide

**Common names:** Lidocaine; Prilocaine

**SMILES:**
- Lidocaine: `CCN(CC)CC(=O)Nc1c(C)cccc1C`
- Prilocaine: `CCCNC(C)C(=O)Nc1ccccc1C`

**Metal-binding data:** Lidocaine (#23/158, log K₁ both = 3.33, as above). Prilocaine ranks **#49 of 158** (log K₁ mean = 2.25; Co = 2.15, Ni = 2.35; 3 donor atoms). **Combined metal score: 5.58.**

**Position on melting-point list:** Only one molar ratio tested (50:50); predicted Tmelt = 290.2 K (`PASS_ROOMTEMP`).

**Why it was selected:**
A third "local-anesthetic eutectic" pairing (alongside #5 and #14), and the best room-temperature result among them — both components contribute genuine, if modest, binding capacity, and both are well-characterized pharmaceutical compounds. This rounds out a small family of anesthetic-pair candidates worth screening together as a group, since they share a design logic and could reasonably be tested in the same experimental batch.

**Caveats:** Moderate model confidence (uncertainty ±23.0 K, similarity 0.52), single ratio tested.

---

## 9. Thymol + Cyclohexanecarboxylic Acid

**Chemical name:** Thymol — 5-methyl-2-(propan-2-yl)phenol; paired with Cyclohexanecarboxylic acid (also known as hexahydrobenzoic acid)

**Common names:** Thymol; Cyclohexanecarboxylic acid

**SMILES:**
- Thymol: `Cc1ccc(C(C)C)c(O)c1`
- Cyclohexanecarboxylic acid: `O=C(O)C1CCCCC1`

**Metal-binding data:** Thymol (#21/158, log K₁ both = 3.63, as above). Cyclohexanecarboxylic acid ranks **#60 of 158** (log K₁ mean = 1.86; Co = 1.83, Ni = 1.90; 2 donor atoms). **Combined metal score: 5.49.**

**Position on melting-point list:** Only one molar ratio tested (29:71 Thymol:Cyclohexanecarboxylic acid); predicted Tmelt = 276.7 K — the **lowest predicted melting point of any top-10 candidate**, well inside the room-temperature window (`PASS_ROOMTEMP`).

**Why it was selected:**
The strongest melting-point confidence in the top 10 — uncertainty ±19.6 K and similarity 0.74, both meaningfully better than candidates #3–#8, meaning this prediction sits more comfortably inside the model's training domain. This continues the Thymol-based fatty/alicyclic-acid family seen in the earlier curated-list reports (ML/ML2/ML3), now grounded with an actual quantified binding contribution from the acid partner rather than zero.

**Caveats:** Both components are individually modest binders (neither in the top quartile of the 158-ligand pool); this is a "two moderate binders" pairing rather than a strong single-chelator or dual-strong-chelator design like candidates #1–#5.

---

## 10. Lidocaine + Camphor

**Chemical name:** Lidocaine — 2-(diethylamino)-N-(2,6-dimethylphenyl)acetamide; paired with Camphor — 1,7,7-trimethylbicyclo[2.2.1]heptan-2-one

**Common names:** Lidocaine; Camphor (DL-camphor / racemic camphor)

**SMILES:**
- Lidocaine: `CCN(CC)CC(=O)Nc1c(C)cccc1C`
- Camphor: `CC12CCC(CC1=O)C2(C)C`

**Metal-binding data:** Lidocaine (#23/158, log K₁ both = 3.33, as above). Camphor ranks **#51 of 158** (log K₁ mean = 2.08; Co = 1.90, Ni = 2.26; 1 donor atom — the carbonyl oxygen). **Combined metal score: 5.42.**

**Position on melting-point list:** The most extensively cross-validated candidate in the top 10 — tested at **12 different molar ratios within the screening window** (8 additional camphor-rich ratios were also tested but predicted above 343 K and excluded), predicted Tmelt ranging 290.7–341.0 K, with the lowest-temperature formulation (46:54 Lidocaine:Camphor) at 290.7 K passing `PASS_ROOMTEMP` with good model confidence (uncertainty ±16.8 K, similarity 0.92 — among the best confidence scores of any candidate in this report).

**Why it was selected:**
This exact pair was the #4 candidate across all three earlier curated-list reports (ML, ML2, ML3), and it reappears here for the same underlying reason: it is the most robustly, repeatedly validated melting-point prediction of any candidate on this list, now additionally confirmed to have real (if modest) metal-binding contributions from both components using the full 158-ligand pool rather than the smaller 38-molecule set available in the earlier rounds. Both components are inexpensive, well-characterized, and commercially available.

**Caveats:** Both components are modest binders individually (similar profile to candidate #9) — this is a high-melting-point-confidence, moderate-metal-binding candidate, useful as a reliability anchor in the experimental batch even though its binding score is the lowest of the top 10.

---

## Summary Table

| # | DES | Metal Score (log K₁ sum) | Both components matched? | Best Tmelt (K) | Room or Warm | MP model confidence | Greenness G-score (DES) | Greenness percentile* |
|---|---|---|---|---|---|---|---|---|
| 1 | 1,10-Phenanthroline + Thymol | 11.18 | Yes | 283.3 | Room (at best ratio) | Moderate (sim. 0.83–0.95 across ratios) | 6.87 | 71st |
| 2 | 1,10-Phenanthroline + Chlorothymol | 10.79 | Yes | 297.2 | Room (borderline) | Moderate (sim. 0.85–0.92) | 6.74 | 68th |
| 3 | Lidocaine + Phenyl salicylate | 9.21 | Yes | 291.7 | Room | Low (sim. 0.42) | 6.93 | 72nd |
| 4 | Menthol + Salicylic acid | 8.04 | Yes | 298.0 | Room (borderline) | Low (sim. 0.49) | 6.97 | 73rd |
| 5 | Lidocaine + Tetracaine | 7.50 | Yes | 303.3 | Warm | Low (sim. 0.37) | 6.93 | 71st |
| 6 | Menthol + Phenyl salicylate | 6.31 | Yes | 296.2 | Room | Low (sim. 0.43) | 6.96 | 72nd |
| 7 | Betaine + Malic acid | 5.67 | Yes | 318.4 | Warm | Low (sim. 0.46) | 6.83 | 70th |
| 8 | Lidocaine + Prilocaine | 5.58 | Yes | 290.2 | Room | Low (sim. 0.52) | 7.04 | 73rd |
| 9 | Thymol + Cyclohexanecarboxylic acid | 5.49 | Yes | 276.7 | Room | Good (sim. 0.74) | 6.84 | 70th |
| 10 | Lidocaine + Camphor | 5.42 | Yes | 290.7 | Room | Good (sim. 0.93) | 6.79 | 69th |

\* Percentile against the GSK solvent-greenness training distribution (n=154, mean 6.04, range 3.02–8.76); see `Greenness ML Models\Final Model\results\repository_candidates_greenness.csv`.

**All 10 candidates have both components independently characterized for Co/Ni binding affinity** — none of them rely on an uncharacterized "unknown" partner molecule, which was not achievable using only the curated top-25 shortlists in the earlier ML/ML2/ML3 reports. **All 10 also now have an actual, quantified greenness score** (added after this report's initial publication — see Limitations item 1), clustering tightly in the 68th–73rd percentile of the GSK reference distribution: solidly above-average, though below the sugar-NADES systems (~90th+ percentile) that top the curated Greenness shortlist.

## Limitations and Recommended Next Steps

1. **Greenness coverage gap — resolved.** None of these 10 candidates originally overlapped with the 25 pairs the Greenness model had been run against. This was closed by running the repository's own trained Greenness model (`rdkit_xgboost_model.pkl`) directly against all 10 candidates, and then against all 205 unique components feeding the full 2,006-pair melting-point universe (756 of which are melting-point-feasible). Findings: (a) all 10 candidates land in the 68th–73rd percentile of the GSK reference distribution — solidly above-average, not a sustainability red flag; (b) metal-binding strength and greenness are statistically uncorrelated across the full dataset (r ≈ 0.03), so there is no inherent trade-off to manage between them in this chemical space; (c) re-ranking the full universe by metal-binding score with greenness now available confirmed the same top 10, in the same order — this shortlist is not an artifact of a narrow search. Results: `Greenness ML Models\Final Model\results\repository_candidates_greenness.csv`. This does not replace a real biodegradability/discharge-safety assessment, which should still happen before bulk handling — particularly for phenanthroline and chlorothymol (candidates #1–#2), which carry other handling/toxicity considerations the greenness G-score does not capture.
2. **Melting-point "confidence" caveat — corrected; bigger structural issue found underneath it.** The original version of this limitation warned that candidates #3–#8's low similarity scores (0.37–0.52) and high uncertainty (±23–25 K) meant their melting points should be "confirmed experimentally before relying on them." That framing was wrong, and it's worth explaining why: **every row in `Candidate_Master_List_rdkit.csv` — the entire 2,006-pair "candidate" universe this report draws from — already has a known, literature-measured `Tmelt, K` value.** It is the model's own training dataset (`Melting_temperature_appended_35il_03082026.csv`), re-used as the candidate-screening pool. I confirmed this directly: for all 10 top candidates, `predicted_Tmelt_K` matches the literal `Tmelt, K` ground truth to within 1–2 K, even for the "low confidence" ones — because the model isn't extrapolating to an unknown compound, it's reproducing a point it was trained on. Each candidate traces back to a real, citable source:

   | Candidate | DOI |
   |---|---|
   | #1 Phenanthroline + Thymol | `10.1016/j.hydromet.2022.105971` (*Hydrometallurgy* — directly relevant to metal recovery) |
   | #2 Phenanthroline + Chlorothymol | `10.1016/j.hydromet.2022.105971` |
   | #3 Lidocaine + Phenyl salicylate | `10.1016/j.tca.2009.08.016` |
   | #4 Menthol + Salicylic acid | `10.1039/C9CC04846D` |
   | #5 Lidocaine + Tetracaine | `10.1016/j.molliq.2020.114745` |
   | #6 Menthol + Phenyl salicylate | `10.1021/acssuschemeng.0c00559` |
   | #7 Betaine + Malic acid | `10.1016/j.molliq.2018.02.049` |
   | #8 Lidocaine + Prilocaine | `10.1016/j.molliq.2020.114745` |
   | #9 Thymol + Cyclohexanecarboxylic acid | `10.3390/molecules26144208` |
   | #10 Lidocaine + Camphor | `10.1016/j.tca.2012.03.027` |

   So: **melting-point liquidity for these 10 candidates does not need experimental confirmation — it's already established.** What *does* need stating plainly: this means the "candidate screening" step contributed no genuinely untested chemistry on the melting-point axis. The real discovery work in this report is entirely in re-evaluating these already-known DES systems against the Metal-Binding and Greenness models, which were trained on different data and genuinely had not been applied to them before. If "discovery of new solvents" (the repository's stated goal) requires testing compositions that have never been measured at all, that requires a separate candidate-generation step — enumerating new HBA/HBD combinations from the 205 known molecules (or new molecules entirely) that aren't already in the training data — which doesn't currently exist in this pipeline.
3. **The metal-binding numbers describe standalone ligands in dilute aqueous solution, not DES mixture behavior.** No ternary (HBA + HBD + metal) model exists in this repository. The ranking here is a legitimate, principled screening signal (which fragments are strong chelators) but should not be read as a prediction of actual partition coefficients or extraction efficiency in a real DES/wastewater system.
4. **Δ(Ni−Co) selectivity was not used as a ranking factor** here because the stated goal is removing *both* Co and Ni together, not separating one from the other. For reference, all 10 candidates show a modest Ni-selective bias (consistent with the dataset-wide pattern noted in the repository's own reports), meaning Ni²⁺ removal is likely to be marginally more effective than Co²⁺ removal across this entire list — worth confirming experimentally if selectivity matters downstream.
5. **Suggested experimental batching:** the phenanthroline pair (#1, #2) as the highest-binding-confidence group; the local-anesthetic family (#5, #8, #10) as a chemically related comparison set; and #9 as the highest-melting-point-confidence anchor point, given it has the best-characterized Tmelt prediction of the lower-binding candidates.
6. **No genuinely untested DES compositions exist in the current candidate pool.** Following directly from item 2: the 205 unique molecules in `Candidate_Master_List_rdkit.csv` appear in an average of 19.6 pairs each (median 7; one molecule appears in 713 of the 2,006 rows), and only 49 of the 205 (24%) appear in just a single pair. There is no combinatorial-enumeration step anywhere in this repository that generates and screens HBA/HBD pairings that haven't already been measured. Before this project can claim to be discovering *new* solvents rather than re-ranking known ones, it needs a candidate-generation step — e.g., enumerate all unmeasured pairs among the 205 known molecules (or a wider PubChem/literature-sourced HBA/HBD library) and run those through the melting-point model for genuine extrapolation.

## Can the ML Models Be Further Optimized Without Increasing Overfitting?

**Short answer: yes, but not by tuning the existing models harder on the same data — the highest-value, lowest-risk improvements are to the validation methodology and the training data, not to model complexity.**

I checked the documented performance of all three models against their own reports:

| Model | Dataset size | Train metric | Test/CV metric | Overfitting gap |
|---|---|---|---|---|
| Melting Point (RDKit XGBoost) | 2,006 rows / 205 unique molecules | R² = 0.9998 | Test R² = 0.972, CV R² = 0.960 ± 0.009 | 0.0275 (repo's own number) |
| Metal-Binding (5-model ensemble) | 158 unique ligands | — | R² = 0.977 (Co), 0.968 (Ni); MAE 0.40/0.46 | Already ensembled for uncertainty |
| Greenness (RDKit XGBoost) | 154 GSK solvents | — | (not separately reported here) | — |

None of these are catastrophically overfit, and the melting-point model already uses real regularization (`subsample=0.6`, `colsample_bytree=0.9`, `reg_alpha=0.1`, `reg_lambda=1`, `min_child_weight=5`). But all three sit on **small datasets relative to the model capacity already in use** (max_depth=10 with 800 trees, on 2,006 rows with heavy row-level redundancy from only 205 unique molecules). Pushing further hyperparameter search or a larger/more complex model on the *same* data would very likely just memorize harder, not generalize better — that's the wrong lever here.

**There's also a validation-design issue worth flagging directly: the random train/test splits used to produce the Test R²/CV R² numbers above almost certainly leak component-level information.** Because each molecule appears in ~20 pairs on average, a row held out for testing very likely shares one of its two components with several rows still in the training set — so the reported metrics measure "can the model interpolate a new *ratio* or *partner* for an already-familiar molecule" rather than "can the model generalize to a structurally novel molecule." That's a materially easier task, and it means the true extrapolation risk for genuinely new chemistry is probably understated by the current numbers.

**Recommended plan, in priority order:**

1. **Re-validate with leave-one-molecule-out (scaffold) cross-validation**, not random row splits. Hold out *all* pairs containing a given molecule, train on the rest, and measure error only on the held-out molecule's pairs. Repeat across many molecules. This is free (no new data needed), directly measures the thing that actually matters (generalization to new chemistry), and may reveal the model is weaker on truly novel structures than the current CV R² = 0.96 suggests — which would be a more honest number to report, not a regression in the model itself.
2. **Don't increase model capacity further; consider testing whether a smaller model holds CV performance.** If a shallower/fewer-tree version of the XGBoost model gets similar scaffold-CV results, that's evidence the current capacity is excess for this dataset size and a sign to simplify, not complicate.
3. **Grow the training data rather than the model.** All three datasets (154–2,006 rows) are small for the chemical diversity they're meant to cover. For melting point, more literature DES phase-diagram digitization is the most direct lever. For metal-binding, the field has large public stability-constant compilations (e.g., NIST/IUPAC critical stability constant databases) that could be used to pretrain before fine-tuning on the in-house 158-ligand set — a transfer-learning approach that adds real information rather than model complexity. The same logic applies to greenness against larger public solvent-sustainability datasets.
4. **Build the candidate-generation step described in Limitation #6.** This isn't a "model optimization" in the traditional sense, but it's the prerequisite for the melting-point model to ever be evaluated on genuinely new predictions instead of re-scoring its own training set.

If none of this is pursued, the honest framing for the wiki is: **the melting-point model's reported accuracy is a measure of interpolation quality within a chemically narrow, highly reused set of 205 molecules, not a validated measure of extrapolation to new solvent chemistry** — a real, structural limitation of the current modeling setup rather than a tuning problem that more GridSearch would fix.
