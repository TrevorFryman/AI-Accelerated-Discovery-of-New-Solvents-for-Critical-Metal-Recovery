# RDKit Molecular Descriptor Generation Walkthrough

We have successfully generated RDKit molecular descriptors from both DES component SMILES strings and produced an enriched dataset ready for machine learning.

A Python script, [generate_descriptors.py](file:///c:/dev/PersonalPrediction/models/generate_descriptors.py), has been generated and executed to compute and append the descriptors.

## Key Outcomes

### 1. Processing Summary
- **Input dataset**: `Melting_temperature_appended_35il_03082026.csv` — 2,006 rows × 13 columns.
- **SMILES parsed**: 2,006 / 2,006 for both `Smiles#1` and `Smiles#2` — **0 failures**.
- **New descriptor columns added**: 14 (7 per component).
- **Output dataset**: `DES_RDKit_Features.csv` — 2,006 rows × 27 columns.

### 2. Descriptors Computed

| Descriptor | Column (C1) | Column (C2) | RDKit Function |
|---|---|---|---|
| Molecular Weight | `C1_MolWt` | `C2_MolWt` | `Descriptors.MolWt` |
| LogP | `C1_LogP` | `C2_LogP` | `Descriptors.MolLogP` |
| TPSA | `C1_TPSA` | `C2_TPSA` | `Descriptors.TPSA` |
| H-bond Donors | `C1_HBD` | `C2_HBD` | `rdMolDescriptors.CalcNumHBD` |
| H-bond Acceptors | `C1_HBA` | `C2_HBA` | `rdMolDescriptors.CalcNumHBA` |
| Rotatable Bonds | `C1_RotBonds` | `C2_RotBonds` | `rdMolDescriptors.CalcNumRotatableBonds` |
| Ring Count | `C1_RingCount` | `C2_RingCount` | `rdMolDescriptors.CalcNumRings` |

### 3. Descriptor Summary Statistics

| Descriptor | Mean | Std | Min | Max |
|---|---|---|---|---|
| `C1_MolWt` | 175.84 | 70.43 | 60.06 | 546.81 |
| `C1_LogP` | -0.40 | 3.12 | -4.09 | 9.43 |
| `C1_TPSA` | 22.89 | 21.74 | 0.00 | 181.62 |
| `C1_HBD` | 0.77 | 0.81 | 0 | 6 |
| `C1_HBA` | 1.02 | 1.00 | 0 | 9 |
| `C1_RotBonds` | 4.18 | 4.78 | 0 | 28 |
| `C1_RingCount` | 0.30 | 0.70 | 0 | 4 |
| `C2_MolWt` | 177.57 | 68.35 | 45.04 | 509.77 |
| `C2_LogP` | 1.62 | 2.99 | -5.40 | 6.33 |
| `C2_TPSA` | 43.65 | 32.68 | 0.00 | 189.53 |
| `C2_HBD` | 1.50 | 1.42 | 0 | 8 |
| `C2_HBA` | 1.63 | 1.71 | 0 | 11 |
| `C2_RotBonds` | 5.91 | 5.25 | 0 | 16 |
| `C2_RingCount` | 0.32 | 0.59 | 0 | 4 |

> [!NOTE]
> Component 1 (typically the HBA salt) has lower TPSA and fewer H-bond donors/acceptors compared to Component 2 (typically the HBD), consistent with the chemical nature of DES components.

## Output Files

- **Enriched dataset**: [DES_RDKit_Features.csv](file:///c:/dev/PersonalPrediction/results/RDKit_DescriptorGeneration/DES_RDKit_Features.csv)

## Next Steps

The enriched dataset `DES_RDKit_Features.csv` is ready for use in machine learning models. The 14 new molecular descriptor columns provide structural chemical information about both DES components that was not available from the original tabular features alone, and can be used to improve model performance.
