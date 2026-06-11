# Molecular Diversity Report

All SMILES strings in GSK_dataset.csv were converted to RDKit molecule objects, and the following descriptors were calculated for each solvent: Molecular Weight (MolWt), Octanol-Water Partition Coefficient (LogP), and Topological Polar Surface Area (TPSA).

## Descriptor Summary Statistics

| Statistic | MolWt | LogP | TPSA |
|---|---|---|---|
| count | 154.0000 | 154.0000 | 154.0000 |
| mean | 110.7641 | 1.1638 | 21.5862 |
| std | 53.0625 | 1.2949 | 16.5745 |
| min | 18.0150 | -1.6681 | 0.0000 |
| 25% | 84.1290 | 0.1832 | 9.2300 |
| 50% | 100.1830 | 1.0175 | 20.2300 |
| 75% | 126.6370 | 1.9771 | 29.4600 |
| max | 416.0550 | 6.1969 | 78.9000 |

## Correlation with G-score

| Descriptor | Correlation with G-score |
|---|---|
| MolWt | 0.0272 |
| LogP | -0.1902 |
| TPSA | 0.5210 |

## Plots

- Histograms: `results/EDA/molecular_descriptor_histograms.png`
- Pair plot: `results/EDA/molecular_descriptor_pairplot.png`
- Correlation matrix: `results/EDA/molecular_descriptor_correlation.png`

## Diversity Observations

- Molecular Weight ranges from 18.02 to 416.05 (mean 110.76), indicating the dataset spans both small molecules (e.g., water) and larger solvents.
- LogP ranges from -1.67 to 6.20 (mean 1.16), reflecting a mix of hydrophilic and lipophilic solvents.
- TPSA ranges from 0.00 to 78.90 (mean 21.59), reflecting variation in polar functional groups across the solvent set.
