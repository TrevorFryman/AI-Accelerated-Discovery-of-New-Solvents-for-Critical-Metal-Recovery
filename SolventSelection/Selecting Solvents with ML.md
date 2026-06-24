# Selecting Solvents with ML

> **Superseded.** This analysis was based on the curated top-25 shortlists in `DESs Selection.xlsx` only. `Repository Solvent Selection.md` re-ran the same kind of analysis against the full prediction universe (2,006 melting-point pairs, 158 metal-binding ligands, and greenness scored across all of them) with metal-binding emphasized for the Co/Ni wastewater-recovery use case, and is the methodology trail's final, most rigorous answer. Kept here for history/reference.

## Purpose

This document summarizes the selection of the **top 5 deep eutectic solvent (DES) candidates** to advance from machine-learning prediction into the screening and experimental validation stage. Candidates were drawn from the top-25 ranked outputs of three ML model families documented in `DESs Selection.xlsx`:

- **Melting Point model** (RDKit-feature variant only): `MP Opt_RDK-RT` (room-temperature target) and `MP Opt_RDK-WT` (warm-temperature target)
- **Greenness model**: `Greenness`
- **Metal-Binding model** (Ni/Co ligand affinity, ranks single molecules rather than HBA+HBD pairs): `Metal-Ligand_Ni+Co`

The non-RDKit melting-point variants (`MP Opt-RT`, `MP Opt-WT`) were excluded from scoring per direction, but their components are still translated to chemical names in `ChemicalNames.xlsx` for completeness.

## Methodology

Each DES pair was scored using a weighted-frequency model agreed on before analysis:

| List | Weight per occurrence | Rationale |
|---|---|---|
| MP Opt_RDK-RT (room temp) | **2x** | Room-temperature liquidity is the more practically important melting-point target |
| MP Opt_RDK-WT (warm temp) | 1x | Secondary melting-point check |
| Greenness | 1x | Independent sustainability/biodegradability model |
| Metal-Binding (component match) | **1.75x** | Either component ranking in the Metal-Binding top 25 is weighted heavily, since strong metal-chelating behavior will matter a great deal in the downstream screening/experimental (e.g., metal-extraction) phase |

A candidate's **score** = sum of weighted occurrences across all four sources. **Breadth** = number of distinct model sources (max 4) the candidate is validated by. Candidates were ranked first by breadth (cross-model agreement), then by score, then by raw occurrence count as a final tiebreaker.

**Note on diversity:** 3 of the 5 selected candidates pair Thymol with a different fatty acid (capric, undecylenic, caprylic). This is a direct, expected consequence of weighting the Metal-Binding signal heavily — Thymol itself ranks #21 of 25 in the Metal-Binding model, which boosts every Thymol-containing pair's breadth/score. This was flagged during analysis; the decision was made to keep the metal-binding weighting (and accept the resulting clustering) because metal-binding performance is considered critical for the next phase.

---

## 1. Thymol + Capric Acid (Decanoic Acid)

**Chemical name:** Thymol — 5-methyl-2-(propan-2-yl)phenol; paired with Capric acid (decanoic acid) — also called n-decanoic acid

**Common names:** Thymol (also: 2-isopropyl-5-methylphenol, thyme camphor); Capric acid (also: decanoic acid, n-decanoic acid)

**SMILES:**
- Thymol: `CC1=CC(=C(C=C1)C(C)C)O`
- Capric acid: `CCCCCCCCCC(=O)O`

**Position on top-25 lists:** Ranked **#2, #8, and #10** of 25 in `MP Opt_RDK-RT` (three separate molar-ratio formulations, see table below); Thymol independently ranked **#21 of 25** in the Metal-Binding model.

**Why it was selected:**
This pairing was the single most consistent candidate in the melting-point model, appearing three separate times in the room-temperature top 25 across a range of molar ratios (45:55, 49:51, 50:50 Thymol:Capric acid), all predicted to fall between 287–291 K with low uncertainty (~16.3–16.7 K) and high training-set similarity (0.93–0.95) — meaning the model is both confident and well within its reliable prediction domain. All three formulations passed the `PASS_ROOMTEMP` screening rule. Thymol's appearance in the Metal-Binding top 25 (rank 21, LogK ≈ 3.6 for Ni/Co) adds a secondary signal that this HBD may retain some metal-chelating capacity, which is a relevant bonus property for extraction-oriented screening.

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
The second-strongest melting-point consensus pairing, appearing twice in the room-temperature top 25 (molar ratios 60:40 and 33:67 Thymol:Undecylenic acid), with predicted melting points of 282.98–289.22 K — both well within the room-temperature liquid range and good model confidence (uncertainty ~16.8–16.9 K, similarity 0.91–0.92). Undecylenic acid is a renewable, castor-oil-derived fatty acid with known antifungal/antimicrobial activity, which may be a useful secondary property depending on the intended application. Shares the same Thymol metal-binding bonus signal as candidate #1.

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
Completes the Thymol + medium-chain fatty acid family that dominates the melting-point top 25 — here at a 10:90 Thymol:Caprylic acid ratio, predicted Tmelt 285.3 K. Model confidence is slightly lower than candidates #1/#2 (uncertainty 17.48 K, similarity 0.876, the lowest of the three Thymol pairs) since this formulation sits closer to the edge of the model's training distribution, but it still passes the room-temperature screen. Included primarily because it shares the validated Metal-Binding signal and extends the fatty-acid chain-length series already supported by candidates #1 and #2, which is useful for systematic experimental comparison (C8 vs. C10 vs. C11 acid chain).

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
A chemically distinct candidate from the Thymol/fatty-acid cluster, pairing two pharmaceutical-grade small molecules at a 46:54 molar ratio, predicted Tmelt 290.7 K with good model confidence (uncertainty 16.83 K, similarity 0.918). Both components are well-characterized, commercially available, and inexpensive, which is favorable for experimental follow-up. Lidocaine's independent appearance in the Metal-Binding top 25 (rank 23, the same molecule evaluated as a standalone ligand) gives this pairing genuine two-model validation rather than relying on a different component than the one driving the melting-point prediction.

| Rank in MP Opt_RDK-RT | Molar ratio (Lidocaine:Camphor) | Predicted Tmelt (K) | Uncertainty (K) | Similarity |
|---|---|---|---|---|
| 23 | 0.46 : 0.54 | 290.7 | 16.83 | 0.918 |

---

## 5. Choline Chloride + D-Sorbitol

**Chemical name:** Choline chloride — 2-hydroxyethyl(trimethyl)azanium chloride; paired with D-Sorbitol — (2R,3R,4R,5S)-hexane-1,2,3,4,5,6-hexol

**Common names:** Choline chloride; D-Sorbitol (also: D-Glucitol)

**SMILES:**
- Choline chloride: `C[N+](C)(C)CCO.[Cl-]`
- D-Sorbitol: `OC[C@@H](O)[C@@H](O)[C@H](O)[C@@H](O)CO`

**Position on top-25 lists:** Ranked **#1 of 25** in `MP Opt_RDK-WT`; ranked **#15 and #22** of 25 in `Greenness`.

**Why it was selected:**
The only top-5 candidate validated by the **Greenness** model rather than the Metal-Binding model, and the only one built from two of the most established, GRAS (Generally Recognized As Safe), bio-based DES components in the literature — choline chloride is the most widely used HBA in NADES (natural deep eutectic solvent) research, and sorbitol is a renewable sugar alcohol. It topped the warm-temperature melting-point list (predicted 318.7 K, uncertainty 16.58 K, similarity 0.934) and appeared twice in the Greenness top 25 (ranks 15 and 22, DES greenness scores 7.54–7.56 out of a higher-is-greener scale) at two different molar ratios (12:88 and 14:87 choline chloride:sorbitol). This candidate brings genuine chemical diversity to the portfolio and the strongest sustainability profile of the five.

| Source | Rank | Detail |
|---|---|---|
| MP Opt_RDK-WT | #1 | x = 0.524 : 0.476, predicted Tmelt 318.7 K, uncertainty 16.58 K, similarity 0.934 |
| Greenness | #15 | x = 0.122 : 0.878, G-score (DES) = 7.559 |
| Greenness | #22 | x = 0.135 : 0.865, G-score (DES) = 7.540 |

---

## Summary Table

| # | DES | Score | Breadth (lists) | Best individual rank |
|---|---|---|---|---|
| 1 | Thymol + Capric acid | 7.75 | 2 | #2 (MP_RDK-RT) |
| 2 | Thymol + Undecylenic acid | 5.75 | 2 | #11 (MP_RDK-RT) |
| 3 | Thymol + Caprylic acid | 3.75 | 2 | #15 (MP_RDK-RT) |
| 4 | Lidocaine + DL-Camphor | 3.75 | 2 | #23 (MP_RDK-RT) |
| 5 | Choline chloride + D-Sorbitol | 3.00 | 2 | #1 (MP_RDK-WT) |

Full SMILES, IUPAC, and common-name data for every component across all six original top-25 lists is available in `ChemicalNames.xlsx`.
