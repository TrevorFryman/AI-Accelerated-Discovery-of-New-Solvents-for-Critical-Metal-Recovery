# Data Quality Report: GSK_dataset.csv

## 1. Missing Values

| Column | Missing Count |
|---|---|
| Unnamed: 0 | 0 |
| Classification | 0 |
| solvent_common_name | 0 |
| IPUAC name | 0 |
| solvent_SMILES | 0 |
| CAS Number | 0 |
| G-score | 0 |

Total missing values: 0

## 2. Duplicate Records

- Fully duplicated rows: 0
- Duplicate `solvent_SMILES` values: 2
- Duplicate `CAS Number` values: 1

Rows with duplicate SMILES:

| Unnamed: 0 | solvent_common_name | solvent_SMILES | CAS Number |
|---|---|---|---|
| 25 | Ethanol | CCO | 64-17-5 |
| 31 | IMS (ethanol, denatured) | CCO | 64-17-5 |
| 92 | 2-Methylpentane | CCCC(C)C | 107-83-5 |
| 94 | Petroleum sprit | CCCC(C)C | 8032-32-4 |

Rows with duplicate CAS Numbers:

| Unnamed: 0 | solvent_common_name | solvent_SMILES | CAS Number |
|---|---|---|---|
| 25 | Ethanol | CCO | 64-17-5 |
| 31 | IMS (ethanol, denatured) | CCO | 64-17-5 |

## 3. Invalid SMILES

All SMILES strings parsed successfully with RDKit. No invalid SMILES found.

## 4. Outliers in G-score (IQR Method)

- Q1: 5.1329
- Q3: 7.1027
- IQR: 1.9698
- Lower bound: 2.1783
- Upper bound: 10.0574
- Number of outliers: 0

## 5. Data Cleaning Recommendations

- No missing values were detected, so no imputation is required.
- Investigate 2 duplicate SMILES entries — these may represent the same molecule listed under different names/CAS numbers and could bias the model.
- Investigate 1 duplicate CAS Number entries for redundant records.
- All SMILES are RDKit-valid; no SMILES correction needed.
- No G-score outliers detected via the IQR method; the target distribution appears clean.
- The leading unnamed index column (`Unnamed: 0`) should be dropped before modeling, as it carries no predictive information.
