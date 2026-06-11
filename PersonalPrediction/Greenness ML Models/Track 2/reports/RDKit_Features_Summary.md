# RDKit Descriptor Feature Summary

- SMILES column used: `solvent_SMILES`
- Total molecules: 154
- Invalid SMILES (dropped): 0
- Final feature matrix shape: 154 rows x 7 descriptor columns

## Descriptors Computed

- MolWt
- LogP
- TPSA
- NumHDonors
- NumHAcceptors
- NumRotatableBonds
- RingCount

## Descriptor Summary Statistics

| Statistic | MolWt | LogP | TPSA | NumHDonors | NumHAcceptors | NumRotatableBonds | RingCount |
|---|---|---|---|---|---|---|---|
| count | 154.0000 | 154.0000 | 154.0000 | 154.0000 | 154.0000 | 154.0000 | 154.0000 |
| mean | 110.7641 | 1.1638 | 21.5862 | 0.3442 | 1.3896 | 1.6299 | 0.3442 |
| std | 53.0625 | 1.2949 | 16.5745 | 0.5983 | 1.1336 | 2.2198 | 0.5287 |
| min | 18.0150 | -1.6681 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 25% | 84.1290 | 0.1832 | 9.2300 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| 50% | 100.1830 | 1.0175 | 20.2300 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| 75% | 126.6370 | 1.9771 | 29.4600 | 1.0000 | 2.0000 | 2.0000 | 1.0000 |
| max | 416.0550 | 6.1969 | 78.9000 | 3.0000 | 6.0000 | 15.0000 | 2.0000 |

## Missing Values After Computation

| Descriptor | Missing Count |
|---|---|
| MolWt | 0 |
| LogP | 0 |
| TPSA | 0 |
| NumHDonors | 0 |
| NumHAcceptors | 0 |
| NumRotatableBonds | 0 |
| RingCount | 0 |

## Output

- Feature matrix saved to `descriptors/RDKit_Features.csv`
