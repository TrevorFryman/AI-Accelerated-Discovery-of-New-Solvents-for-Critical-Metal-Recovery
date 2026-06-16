# Metal-Binding Figures — DES Ligand Co²⁺ / Ni²⁺ Stability Constant Predictions

> **Setup note for wiki editors** — Images are stored in the `figures/` folder of this wiki's git
> repository. To add or update them:
> ```
> git clone https://github.com/YOUR-USERNAME/YOUR-REPO.wiki.git
> # copy the figures/ folder into the cloned wiki directory
> git add figures/
> git commit -m "Add metal-binding figures"
> git push
> ```
> Once pushed, all image links below will resolve automatically. Replace `YOUR-USERNAME/YOUR-REPO`
> with your actual GitHub username and repository name.

---

These figures accompany the full methodology and results described in
[[Final LogK Prediction Report]] and [[DES Ligand Prediction Strategy]].
All predictions use the **M2 SMILES-only model** (5-fold ensemble, Chemprop v1.5.2),
producing log K₁ (first stepwise stability constant, 25 °C, aqueous reference) for
**164 candidate DES ligand fragments** against Co²⁺ and Ni²⁺.

---

## Figure 1 — Model Validation: Predicted vs. Experimental log K₁

![Sanity check parity plots for Co and Ni](figures/fig1_sanity_check_parity.png)

**What it shows:** Parity plots comparing the M2 ensemble's predictions against the
published experimental log K₁ values from the model's own held-out test set —
Co²⁺ (n = 149 rows) and Ni²⁺ (n = 210 rows). Points on the dashed line are perfect
predictions; scatter around it reflects model error.

**Key metrics:**

| Metal | n | R² | MAE | RMSE |
|---|---|---|---|---|
| Co²⁺ | 149 | 0.977 | 0.400 | 0.579 |
| Ni²⁺ | 210 | 0.968 | 0.457 | 0.744 |

**Interpretation:** Both metals show very high R² (> 0.96), confirming the
legacy v1.5.2 environment reproduces the model's published performance before
it is applied to novel DES ligands. This is the scientific justification for
trusting the DES predictions in Figures 2–7.

---

## Figure 2 — Co²⁺ vs. Ni²⁺ Binding Affinity (Selectivity Scatter)

![Co vs Ni logK scatter coloured by selectivity](figures/fig2_co_vs_ni_scatter.png)

**What it shows:** Each of the 164 candidate DES ligand fragments plotted as a
point with its predicted Co²⁺ log K₁ on the x-axis and Ni²⁺ log K₁ on the
y-axis. Points are coloured by **Δ log K₁ = Ni − Co**: red shades favour Ni²⁺,
blue shades favour Co²⁺. The dashed diagonal (y = x) is the line of equal
affinity for both metals.

**Interpretation:**
- The majority of ligands cluster near the diagonal (modest Ni preference, which
  is typical — Ni²⁺ is a harder Lewis acid and generally binds O/N donors more
  strongly than Co²⁺).
- Strong binders (upper-right quadrant) are predominantly catecholate/phenolate,
  aminodicarboxylate, and N-heterocycle motifs — consistent with known coordination
  chemistry for both metals.
- Ligands far from the diagonal are candidates for **selective extraction**: those
  above the line preferentially bind Ni²⁺; those below preferentially bind Co²⁺.
- Top binders are annotated directly on the plot.

---

## Figure 3 — Top 25 Predicted Binders (Grouped Bar Chart)

![Top 25 ligands ranked by mean logK with Co and Ni bars](figures/fig3_top25_binders_bar.png)

**What it shows:** Horizontal grouped bar chart of the 25 DES ligand fragments
with the highest average predicted log K₁ across both metals. Blue bars = Co²⁺,
red bars = Ni²⁺. Error bars show the **5-fold ensemble standard deviation** —
the model's internal uncertainty estimate for each prediction.

**Interpretation:**
- The tetracycline-like compound (multiple chelating O/N donors) and caffeic acid
  (catecholate motif) rank at the top for both metals, with log K₁ in the 8–10
  range — comparable to strong aminopolycarboxylate chelators in the training data
  (e.g. EDTA-type logK ~ 10–16 for transition metals).
- Phenanthroline-type N,N-bidentate and aminodicarboxylate (aspartate-type)
  ligands occupy the mid-upper ranks (log K₁ 5–8), consistent with literature.
- Large error bars (e.g. caffeic acid for Co²⁺) flag predictions where the 5
  model folds disagree — treat those as screening hits requiring experimental
  follow-up, not reliable quantitative estimates.

---

## Figure 4 — Distribution of Predicted log K₁ Across All DES Ligands

![Histogram and KDE of logK for Co and Ni, with mean lines](figures/fig4_logK_distribution.png)

**What it shows:** Overlaid histograms (bars) and kernel density estimates (lines)
of the predicted log K₁ for all 164 ligands against Co²⁺ and Ni²⁺. Dashed vertical
lines mark the mean for each metal.

**Summary statistics:**

| | Co²⁺ | Ni²⁺ |
|---|---|---|
| Mean log K₁ | 1.72 | 1.93 |
| Std across ligands | 1.67 | 1.85 |
| Range | −1.74 to 9.70 | −1.68 to 10.00 |

**Interpretation:** Most DES ligand fragments are **weak binders** (log K₁ ~ 1–3),
reflecting that typical DES components (glycols, simple alcohols, amides, halides)
are not strong metal chelators. The long right tail (log K₁ > 5) represents a
minority of DES ligands with strong multi-dentate donor groups. This distribution
is chemically sensible and supports the model's output as a plausible screening
ranking rather than artefact.

---

## Figure 5 — Prediction Confidence: Ensemble Spread vs. Predicted log K₁

![Scatter of LogK_std vs LogK_mean for Co and Ni](figures/fig5_confidence_std_vs_mean.png)

**What it shows:** For each of the 164 ligands and each metal, the **5-fold ensemble
standard deviation** (y-axis, a per-prediction uncertainty proxy) plotted against the
**ensemble mean log K₁** (x-axis). The red dashed line marks the low-confidence
threshold (std > 1.5).

**Interpretation:**
- The vast majority of predictions (≥ 75%) have fold-std < 0.6, indicating good
  agreement across all 5 model folds — these can be used for screening with
  reasonable confidence.
- A small number of predictions (4 for Co²⁺, 3 for Ni²⁺) exceed std = 1.5; these
  fall largely among the high-log K₁ predictions where the model is extrapolating
  into less-represented chemical space (complex polyfunctional ligands).
- **Rule of thumb:** use `LogK_std` from `Co_LogK_Predictions.csv` /
  `Ni_LogK_Predictions.csv` as a per-row confidence flag when prioritising
  candidates for experimental follow-up.

---

## Figure 6 — Metal Selectivity: Ni²⁺ vs. Co²⁺ Preference (Diverging Bar)

![Diverging bar chart of Ni minus Co logK delta](figures/fig6_selectivity_delta.png)

**What it shows:** Δ log K₁ = (Ni²⁺ logK) − (Co²⁺ logK) for the top-20
most Co²⁺-selective (blue, negative Δ) and top-20 most Ni²⁺-selective (red,
positive Δ) DES ligand fragments. Ligands are sorted by Δ.

**Interpretation:**
- **Ni²⁺-selective ligands** (Δ > 0, right bars): tend to be hard O-donor or
  mixed O/N donors — consistent with Ni²⁺ being a harder Lewis acid.
- **Co²⁺-selective ligands** (Δ < 0, left bars): tend to include softer donors
  (thiol/thioether motifs, some N-heterocycles) reflecting Co²⁺'s greater
  preference for borderline-soft donors relative to Ni²⁺.
- The magnitude of most Δ values (0.5–2 log units) is meaningful but not
  extreme — both metals bind the same ligand families; the selectivity is
  quantitative rather than qualitative.
- **Practical use:** when designing a DES for selective Co vs. Ni extraction,
  prioritise HBA/HBD components from the corresponding side of this chart.

---

## Figure 7 — Predicted log K₁ by Functional Group Class

![Box and strip plots of logK grouped by chemical class for Co and Ni](figures/fig7_class_breakdown_boxplot.png)

**What it shows:** Box plots (with individual points overlaid as a strip plot)
of predicted log K₁ grouped by assigned functional-group class, for both Co²⁺
(blue) and Ni²⁺ (red). Classes are ordered by median Co²⁺ log K₁ (highest
to lowest). The annotation **n=** below each class is the number of unique
ligands in that class.

**Class definitions used:**

| Class | Structural rule |
|---|---|
| Halide | F⁻, Cl⁻, Br⁻, I⁻ |
| N-heterocycle | Aromatic N-containing ring (pyridine, imidazole, phenanthroline-type) |
| Catechol / phenol | Aromatic C–OH |
| Amino acid | α-amino acid motif (N–Cα–COOH) |
| Carboxylate | –C(=O)OH or –COO⁻ |
| Amide | –C(=O)N– |
| Amine | Aliphatic primary or secondary amine |
| Polyol / sugar | Aliphatic –C–OH (glycols, sugars) |
| Thiol / thioether | Any sulfur |
| Phosphonate | P=O |
| Other | Unmatched |

**Interpretation:**
- **N-heterocycles and catechol/phenols** show the highest median log K₁ for
  both metals (~4–6), reflecting their strong bidentate chelation ability.
- **Amino acids and carboxylates** cluster in the mid-range (log K₁ ~ 2–4),
  consistent with the training data's large body of aminocarboxylate complexes.
- **Polyols and amides** score low (log K₁ < 2 median), as expected for weakly
  donating functional groups with no chelate geometry.
- **Halides** show a broad spread — monodentate Cl⁻ / Br⁻ binding is weak for
  hard-metal aqueous conditions (log K₁ ~ 0–1 range).
- Class overlap (large IQR boxes) is expected: within each class there is
  significant structural diversity that the model distinguishes internally.

---

## Data files

The underlying data for all figures is available in the repository:

| File | Contents |
|---|---|
| `Co_LogK_Predictions.csv` | 164-row table: Ligand SMILES, LogK_mean, LogK_std, per-fold predictions (Co²⁺) |
| `Ni_LogK_Predictions.csv` | Same structure for Ni²⁺ |
| `candidate_ligand_fragments_full.csv` | All 181 unique DES fragments with filtering decisions and exclusion reasons |

See [[Final LogK Prediction Report]] for full methodology, assumptions, and confidence
assessment. See [[DES Ligand Prediction Strategy]] for the decomposition and filtering
rules used to derive the 164 candidate ligands from the DES melting-point dataset.
