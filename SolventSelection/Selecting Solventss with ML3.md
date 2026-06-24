# Selecting Solventss with ML3

> **Superseded.** This analysis was based on the curated top-25 shortlists in `DESs Selection.xlsx` only. `Repository Solvent Selection.md` re-ran the same kind of analysis against the full prediction universe (2,006 melting-point pairs, 158 metal-binding ligands, and greenness scored across all of them) with metal-binding emphasized for the Co/Ni wastewater-recovery use case, and is the methodology trail's final, most rigorous answer. Kept here for history/reference.

## Purpose

This document summarizes the selection of the **top 10 deep eutectic solvent (DES) candidates** to advance from machine-learning prediction into the screening and experimental validation stage. Candidates were drawn from the top-25 ranked outputs of three ML model families documented in `DESs Selection.xlsx`:

- **Melting Point model** (RDKit-feature variant only): `MP Opt_RDK-RT` (room-temperature target) and `MP Opt_RDK-WT` (warm-temperature target)
- **Greenness model**: `Greenness`
- **Metal-Binding model** (Ni/Co ligand affinity, ranks single molecules rather than HBA+HBD pairs): `Metal-Ligand_Ni+Co`

The non-RDKit melting-point variants (`MP Opt-RT`, `MP Opt-WT`) were excluded from scoring per direction, but their components are still translated to chemical names in `ChemicalNames.xlsx` for completeness.

## Methodology

Each DES pair was scored using the same weighted-frequency model as `Selecting Solventss with ML2.md`:

| List | Weight per occurrence | Rationale |
|---|---|---|
| MP Opt_RDK-RT (room temp) | **2.5x** | Room-temperature liquidity remains the more practically important melting-point target |
| MP Opt_RDK-WT (warm temp) | 1x | Secondary melting-point check |
| Greenness | **2x** | Independent sustainability/biodegradability model |
| Metal-Binding (component match) | **4.5x** | Either component ranking in the Metal-Binding top 25 is weighted very heavily — metal-chelating behavior is treated as a dominant factor for the downstream screening/experimental (e.g., metal-extraction) phase |

A candidate's **score** = sum of weighted occurrences across all four sources. **Breadth** = number of distinct model sources (max 4) the candidate is validated by. Candidates were ranked first by breadth (cross-model agreement), then by score, then by raw occurrence count as a final tiebreaker.

**Note on structure of this list:** Only **6 unique pairs** reach breadth = 2 (validated by two distinct model sources) anywhere in the combined dataset — these occupy slots 1–6. From slot 7 onward, every remaining candidate has breadth = 1, meaning it was only ever validated by a single model (most commonly Greenness, which repeats the same chemistry across many molar-ratio entries). These breadth-1 candidates are included to reach a full top 10 as requested, but they carry materially weaker cross-model evidence than slots 1–6, and that distinction is called out in each section below.

---

## 1. Thymol + Capric Acid (Decanoic Acid)

**Chemical name:** Thymol — 5-methyl-2-(propan-2-yl)phenol; paired with Capric acid (decanoic acid) — also called n-decanoic acid

**Common names:** Thymol (also: 2-isopropyl-5-methylphenol, thyme camphor); Capric acid (also: decanoic acid, n-decanoic acid)

**SMILES:**
- Thymol: `CC1=CC(=C(C=C1)C(C)C)O`
- Capric acid: `CCCCCCCCCC(=O)O`

**Position on top-25 lists:** Ranked **#2, #8, and #10** of 25 in `MP Opt_RDK-RT`; Thymol independently ranked **#21 of 25** in the Metal-Binding model.

**Why it was selected:**
This pairing was the single most consistent candidate in the melting-point model, appearing three separate times in the room-temperature top 25 across a range of molar ratios (45:55, 49:51, 50:50 Thymol:Capric acid), all predicted to fall between 287–291 K with low uncertainty (~16.3–16.7 K) and high training-set similarity (0.93–0.95). All three formulations passed the `PASS_ROOMTEMP` screening rule. Thymol's appearance in the Metal-Binding top 25 (rank 21, LogK ≈ 3.6 for Ni/Co) carries the heaviest single weight (4.5x) in the scoring model, making this candidate's metal-chelating bonus the dominant factor separating it from the rest of the field. Score 12.0, breadth 2/4.

| Rank in MP Opt_RDK-RT | Molar ratio (Thymol:Capric) | Predicted Tmelt (K) | Uncertainty (K) | Similarity |
|---|---|---|---|---|
| 2 | 0.49 : 0.51 | 288.4 | 16.33 | 0.950 |
| 8 | 0.45 : 0.55 | 289.1 | 16.69 | 0.927 |
| 10 | 0.50 : 0.50 | 290.7 | 16.32 | 0.951 |

---

## 2. Thymol + Undecylenic Acid (Undec-10-enoic Acid)

**Chemical name:** Thymol — 5-methyl-2-(propan-2-yl)phenol; paired with Undecylenic acid — undec-10-enoic acid

**Common names:** Thymol; Undecylenic acid (also: 10-undecenoic acid)

**SMILES:**
- Thymol: `CC1=CC(=C(C=C1)C(C)C)O`
- Undecylenic acid: `C=CCCCCCCCCC(=O)O`

**Position on top-25 lists:** Ranked **#11 and #19** of 25 in `MP Opt_RDK-RT`; Thymol independently ranked **#21 of 25** in the Metal-Binding model.

**Why it was selected:**
The second-strongest melting-point consensus pairing, appearing twice in the room-temperature top 25 (molar ratios 60:40 and 33:67 Thymol:Undecylenic acid), with predicted melting points of 282.98–289.22 K — both well within the room-temperature liquid range and good model confidence (uncertainty ~16.8–16.9 K, similarity 0.91–0.92). Undecylenic acid is a renewable, castor-oil-derived fatty acid with known antifungal/antimicrobial activity. Shares the same heavily weighted Thymol metal-binding bonus as candidate #1. Score 9.5, breadth 2/4.

| Rank in MP Opt_RDK-RT | Molar ratio (Thymol:Undecylenic) | Predicted Tmelt (K) | Uncertainty (K) | Similarity |
|---|---|---|---|---|
| 11 | 0.60 : 0.40 | 289.2 | 16.84 | 0.917 |
| 19 | 0.33 : 0.67 | 283.0 | 16.92 | 0.912 |

---

## 3. Thymol + Caprylic Acid (Octanoic Acid)

**Chemical name:** Thymol — 5-methyl-2-(propan-2-yl)phenol; paired with Caprylic acid (octanoic acid) — also called n-octanoic acid

**Common names:** Thymol; Caprylic acid (also: octanoic acid, n-caprylic acid)

**SMILES:**
- Thymol: `CC1=CC(=C(C=C1)C(C)C)O`
- Caprylic acid: `CCCCCCCC(=O)O`

**Position on top-25 lists:** Ranked **#15 of 25** in `MP Opt_RDK-RT` (ranking score 93.997); Thymol independently ranked **#21 of 25** in the Metal-Binding model.

**Why it was selected:**
Completes the Thymol + medium-chain fatty acid family that dominates the melting-point top 25 — here at a 10:90 Thymol:Caprylic acid ratio, predicted Tmelt 285.3 K. Model confidence is slightly lower than candidates #1/#2 (uncertainty 17.48 K, similarity 0.876) since this formulation sits closer to the edge of the model's training distribution, but it still passes the room-temperature screen. It edges out candidate #4 on a tiebreak: both sit at an identical weighted score, but this pairing's underlying melting-point rank (#15, ranking score 93.997) is stronger than Lidocaine + DL-Camphor's (#23, ranking score 93.013). Score 7.0, breadth 2/4.

| Rank in MP Opt_RDK-RT | Molar ratio (Thymol:Caprylic) | Predicted Tmelt (K) | Uncertainty (K) | Similarity |
|---|---|---|---|---|
| 15 | 0.10 : 0.90 | 285.3 | 17.48 | 0.876 |

---

## 4. Lidocaine + DL-Camphor

**Chemical name:** Lidocaine — 2-(diethylamino)-N-(2,6-dimethylphenyl)acetamide; paired with Camphor — 1,7,7-trimethylbicyclo[2.2.1]heptan-2-one

**Common names:** Lidocaine (also: Xylocaine); Camphor (DL-camphor / racemic camphor)

**SMILES:**
- Lidocaine: `CCN(CC)CC(=O)NC1=C(C=CC=C1C)C`
- Camphor: `CC1(C2CCC1(C(=O)C2)C)C`

**Position on top-25 lists:** Ranked **#23 of 25** in `MP Opt_RDK-RT`; Lidocaine independently ranked **#23 of 25** in the Metal-Binding model (as the identical molecule, 2-(diethylamino)-N-(2,6-dimethylphenyl)acetamide).

**Why it was selected:**
A chemically distinct candidate from the Thymol/fatty-acid cluster, pairing two pharmaceutical-grade small molecules at a 46:54 molar ratio, predicted Tmelt 290.7 K with good model confidence (uncertainty 16.83 K, similarity 0.918). Both components are well-characterized, commercially available, and inexpensive. Lidocaine's independent appearance in the Metal-Binding top 25 (rank 23, the same molecule evaluated as a standalone ligand) gives this pairing genuine two-model validation, and under the heavier 4.5x Metal-Binding weight it still holds a top-5 slot despite its melting-point rank being the weakest of the Thymol-cluster candidates. Score 7.0, breadth 2/4.

| Rank in MP Opt_RDK-RT | Molar ratio (Lidocaine:Camphor) | Predicted Tmelt (K) | Uncertainty (K) | Similarity |
|---|---|---|---|---|
| 23 | 0.46 : 0.54 | 290.7 | 16.83 | 0.918 |

---

## 5. Cetyl Alcohol (Hexadecan-1-ol) + Thymol

**Chemical name:** Cetyl alcohol — hexadecan-1-ol; paired with Thymol — 5-methyl-2-(propan-2-yl)phenol

**Common names:** Cetyl alcohol (also: 1-hexadecanol, palmityl alcohol); Thymol

**SMILES:**
- Cetyl alcohol: `CCCCCCCCCCCCCCCCO`
- Thymol: `CC1=CC(=C(C=C1)C(C)C)O`

**Position on top-25 lists:** Ranked **#24 of 25** in `MP Opt_RDK-WT`; Thymol independently ranked **#21 of 25** in the Metal-Binding model.

**Why it was selected:**
Its Thymol-driven Metal-Binding bonus outweighs candidate #6 (Choline chloride + D-Sorbitol), which has no Metal-Binding signal at all. The underlying melting-point prediction is for a warm-temperature (not room-temperature) target — a 90:10 Cetyl alcohol:Thymol mixture at predicted Tmelt 320.7 K, passing the `PASS_WARMTEMP` screen with reasonable confidence (uncertainty 17.46 K, similarity 0.877). Cetyl alcohol is an inexpensive, widely available fatty alcohol, making this pairing easy to source for experimental follow-up even though its melting-point validation is comparatively weaker (single occurrence, near the bottom of its list) than candidates #1–#4. Score 5.5, breadth 2/4.

| Rank in MP Opt_RDK-WT | Molar ratio (Cetyl alcohol:Thymol) | Predicted Tmelt (K) | Uncertainty (K) | Similarity | Screening status |
|---|---|---|---|---|---|
| 24 | 0.897 : 0.103 | 320.7 | 17.46 | 0.877 | PASS_WARMTEMP |

---

## 6. Choline Chloride + D-Sorbitol

**Chemical name:** Choline chloride — 2-hydroxyethyl(trimethyl)azanium chloride; paired with D-Sorbitol — (2R,3R,4R,5S)-hexane-1,2,3,4,5,6-hexol

**Common names:** Choline chloride; D-Sorbitol (also: D-Glucitol)

**SMILES:**
- Choline chloride: `C[N+](C)(C)CCO.[Cl-]`
- D-Sorbitol: `OC[C@@H](O)[C@@H](O)[C@H](O)[C@@H](O)CO`

**Position on top-25 lists:** Ranked **#1 of 25** in `MP Opt_RDK-WT`; ranked **#15 and #22** of 25 in `Greenness`.

**Why it was selected:**
The last candidate to reach breadth = 2, and the only one in the top 6 validated by the Greenness model rather than the Metal-Binding model. Choline chloride and sorbitol are two of the most established, GRAS (Generally Recognized As Safe), bio-based DES components in the literature — choline chloride is the most widely used HBA in NADES (natural deep eutectic solvent) research, and sorbitol is a renewable sugar alcohol. It topped the warm-temperature melting-point list (predicted 318.7 K, uncertainty 16.58 K, similarity 0.934) and appeared twice in the Greenness top 25 (ranks 15 and 22, DES greenness scores 7.54–7.56) at two different molar ratios. This is the strongest sustainability profile of the top 6, but its score (5.0) falls just below candidate #5 once the Metal-Binding weight was raised to 4.5x.

| Source | Rank | Detail |
|---|---|---|
| MP Opt_RDK-WT | #1 | x = 0.524 : 0.476, predicted Tmelt 318.7 K, uncertainty 16.58 K, similarity 0.934 |
| Greenness | #15 | x = 0.122 : 0.878, G-score (DES) = 7.559 |
| Greenness | #22 | x = 0.135 : 0.865, G-score (DES) = 7.540 |

---

## 7. D-Fructose + D-Glucose

**Chemical name:** D-Fructose — (3S,4R,5R)-2-(hydroxymethyl)oxane-2,3,4,5-tetrol (fructopyranose form); paired with D-Glucose — (3R,4S,5S,6R)-6-(hydroxymethyl)oxane-2,3,4,5-tetrol (glucopyranose form)

**Common names:** D-Fructose; D-Glucose (also: dextrose, blood sugar)

**SMILES:**
- D-Fructose: `C1[C@H]([C@H]([C@@H](C(O1)(CO)O)O)O)O`
- D-Glucose: `C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O`

**Position on top-25 lists:** Ranked **#2 through #10** of 25 in `Greenness` (9 separate molar-ratio entries — ranks 2, 3, 4, 5, 6, 7, 8, 9, 10).

**Why it was selected:**
**This is the single highest raw score in the entire dataset (18.0)** — the Fructose:Glucose pair occupies 9 of the 25 Greenness-model slots at a near-identical greenness score (g_des = 7.5975) across molar ratios ranging from 10:90 to 90:10, with predicted melting points spanning 370.6–418.2 K (well above room temperature in every formulation tested). **This is a breadth = 1 candidate**: it was never validated by either melting-point list or the Metal-Binding model, meaning its high score reflects repetition *within* a single model rather than agreement *across* models. Fructose + Glucose is a well-known natural sugar-based DES system in the literature, but the predicted melting points (370–418 K) are far above both the room-temperature and warm-temperature screening targets used elsewhere in this analysis, so this candidate would need a separate justification (e.g., a different end-use temperature window) before being prioritized experimentally alongside candidates #1–#6.

| Rank in Greenness | Molar ratio (Fructose:Glucose) | Tmelt (K) | G-score (DES) |
|---|---|---|---|
| 2 | 0.799 : 0.201 | 376.9 | 7.5975 |
| 3 | 0.602 : 0.398 | 381.8 | 7.5975 |
| 4 | 0.705 : 0.295 | 370.6 | 7.5975 |
| 5 | 0.397 : 0.603 | 403.9 | 7.5975 |
| 6 | 0.300 : 0.700 | 405.8 | 7.5975 |
| 7 | 0.497 : 0.503 | 391.0 | 7.5975 |
| 8 | 0.904 : 0.096 | 377.7 | 7.5975 |
| 9 | 0.104 : 0.897 | 418.2 | 7.5975 |
| 10 | 0.203 : 0.797 | 414.1 | 7.5975 |

---

## 8. D-Glucose + Sucrose

**Chemical name:** D-Glucose — (3R,4S,5S,6R)-6-(hydroxymethyl)oxane-2,3,4,5-tetrol; paired with Sucrose — (2R,3R,4S,5S,6R)-2-[(2S,3S,4S,5R)-3,4-dihydroxy-2,5-bis(hydroxymethyl)oxolan-2-yl]oxy-6-(hydroxymethyl)oxane-3,4,5-triol

**Common names:** D-Glucose (dextrose); Sucrose (table sugar)

**SMILES:**
- D-Glucose: `C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O`
- Sucrose: `C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O[C@]2([C@H]([C@@H]([C@H](O2)CO)O)O)CO)O)O)O)O`

**Position on top-25 lists:** Ranked **#11, #14, #16, #18, #20, #23, and #25** of 25 in `Greenness` (7 separate molar-ratio entries).

**Why it was selected:**
A second sugar-based DES system, appearing 7 times in the Greenness top 25 with greenness scores tightly clustered between 7.53–7.58 and predicted melting points of 402–428 K — again entirely from a single model (breadth = 1, no Metal-Binding or melting-point-list corroboration). Like candidate #7, all formulations tested predict melting points far above room or warm-temperature targets, so this is a green/sustainable system but not one with melting-point validation aligned to the room/warm-temperature screening used for candidates #1–#6.

| Rank in Greenness | Molar ratio (Glucose:Sucrose) | Tmelt (K) | G-score (DES) |
|---|---|---|---|
| 11 | 0.902 : 0.098 | 416.8 | 7.584 |
| 14 | 0.799 : 0.201 | 408.6 | 7.570 |
| 16 | 0.701 : 0.299 | 406.1 | 7.556 |
| 18 | 0.646 : 0.354 | 402.3 | 7.549 |
| 20 | 0.598 : 0.402 | 411.0 | 7.542 |
| 23 | 0.548 : 0.452 | 421.6 | 7.535 |
| 25 | 0.501 : 0.499 | 428.4 | 7.529 |

---

## 9. D-Fructose + Sucrose

**Chemical name:** D-Fructose — (3S,4R,5R)-2-(hydroxymethyl)oxane-2,3,4,5-tetrol; paired with Sucrose — (2R,3R,4S,5S,6R)-2-[(2S,3S,4S,5R)-3,4-dihydroxy-2,5-bis(hydroxymethyl)oxolan-2-yl]oxy-6-(hydroxymethyl)oxane-3,4,5-triol

**Common names:** D-Fructose; Sucrose (table sugar)

**SMILES:**
- D-Fructose: `C1[C@H]([C@H]([C@@H](C(O1)(CO)O)O)O)O`
- Sucrose: `C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O[C@]2([C@H]([C@@H]([C@H](O2)CO)O)O)CO)O)O)O)O`

**Position on top-25 lists:** Ranked **#12, #13, #17, #19, and #24** of 25 in `Greenness` (5 separate molar-ratio entries).

**Why it was selected:**
A third sugar-based DES system from the Greenness list, completing the three-way Fructose/Glucose/Sucrose family that fills slots 7–9. It edges out candidate #10 (Menthol + Decanoic acid, score 10.0, raw 4) on the raw-occurrence tiebreak, since both tie at score 10.0 but this pairing has 5 raw occurrences versus 4. As with candidates #7 and #8, this is a breadth = 1 result — validated only by Greenness, with predicted melting points (367–417 K) again above the room/warm-temperature screening targets.

| Rank in Greenness | Molar ratio (Fructose:Sucrose) | Tmelt (K) | G-score (DES) |
|---|---|---|---|
| 12 | 0.894 : 0.106 | 379.9 | 7.583 |
| 13 | 0.802 : 0.198 | 367.5 | 7.570 |
| 17 | 0.697 : 0.303 | 384.2 | 7.556 |
| 19 | 0.602 : 0.398 | 400.6 | 7.543 |
| 24 | 0.502 : 0.498 | 417.3 | 7.529 |

---

## 10. Menthol + Capric Acid (Decanoic Acid)

**Chemical name:** Menthol — (1R,2S,5R)-5-methyl-2-(propan-2-yl)cyclohexan-1-ol; paired with Capric acid (decanoic acid)

**Common names:** L-Menthol; Capric acid (also: decanoic acid)

**SMILES:**
- Menthol: `C[C@@H]1CC[C@H]([C@@H](C1)O)C(C)C`
- Capric acid: `CCCCCCCCCC(=O)O`

**Position on top-25 lists:** Ranked **#1, #3, #4, and #12** of 25 in `MP Opt_RDK-RT` (four separate molar-ratio formulations).

**Why it was selected:**
The most repeated single pairing in the entire melting-point room-temperature list — Menthol + Capric acid appears 4 times across molar ratios from 50:50 to 70:30, with predicted melting points of 282–288 K, all passing `PASS_ROOMTEMP` with good confidence (uncertainty 16.4–16.6 K, similarity 0.93–0.95). Despite this strong single-model showing, **neither Menthol nor Capric acid appears in the Metal-Binding or Greenness top 25**, so this candidate has breadth = 1 — it is the strongest melting-point-only signal in the dataset, but carries none of the cross-model corroboration that distinguishes candidates #1–#6.

| Rank in MP Opt_RDK-RT | Molar ratio (Menthol:Capric) | Predicted Tmelt (K) | Uncertainty (K) | Similarity |
|---|---|---|---|---|
| 1 | 0.50 : 0.50 | 285.9 | 16.52 | 0.938 |
| 3 | 0.698 : 0.302 | 287.9 | 16.58 | 0.934 |
| 4 | 0.70 : 0.30 | 288.1 | 16.59 | 0.934 |
| 12 | 0.60 : 0.40 | 282.3 | 16.37 | 0.948 |

---

## Summary Table

| # | DES | Score | Breadth (lists) | Best individual rank |
|---|---|---|---|---|
| 1 | Thymol + Capric acid | 12.0 | 2 | #2 (MP_RDK-RT) |
| 2 | Thymol + Undecylenic acid | 9.5 | 2 | #11 (MP_RDK-RT) |
| 3 | Thymol + Caprylic acid | 7.0 | 2 | #15 (MP_RDK-RT) |
| 4 | Lidocaine + DL-Camphor | 7.0 | 2 | #23 (MP_RDK-RT) |
| 5 | Cetyl alcohol + Thymol | 5.5 | 2 | #24 (MP_RDK-WT) |
| 6 | Choline chloride + D-Sorbitol | 5.0 | 2 | #1 (MP_RDK-WT) |
| 7 | D-Fructose + D-Glucose | 18.0 | 1 | #2 (Greenness) |
| 8 | D-Glucose + Sucrose | 14.0 | 1 | #11 (Greenness) |
| 9 | D-Fructose + Sucrose | 10.0 | 1 | #12 (Greenness) |
| 10 | Menthol + Capric acid | 10.0 | 1 | #1 (MP_RDK-RT) |

**Important caveat for slots 7–10:** these four candidates have higher raw scores than several breadth = 2 candidates above them, but that reflects repetition within one model (mostly the same sugar chemistry sampled at many molar ratios in Greenness) rather than agreement across independent models. If cross-model validation is the priority for advancing to experimental screening, candidates #1–#6 remain the stronger choices; #7–#10 are included here only to satisfy the requested list of 10.

Full SMILES, IUPAC, and common-name data for every component across all six original top-25 lists is available in `ChemicalNames.xlsx`.
