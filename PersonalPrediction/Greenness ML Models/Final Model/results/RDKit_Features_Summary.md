# RDKit Descriptor Feature Summary

- SMILES column used: `solvent_SMILES`
- Total molecules: 154
- Invalid SMILES (dropped): 0
- Final feature matrix shape: 154 rows x 13 descriptor columns

## Descriptors Computed

- MolWt
- LogP
- TPSA
- NumHDonors
- NumHAcceptors
- NumRotatableBonds
- RingCount
- NumAromaticRings
- FractionCSP3
- MolMR
- HeavyAtomCount
- NumAliphaticRings
- BertzCT

## Descriptor Summary Statistics

| Statistic | MolWt | LogP | TPSA | NumHDonors | NumHAcceptors | NumRotatableBonds | RingCount | NumAromaticRings | FractionCSP3 | MolMR | HeavyAtomCount | NumAliphaticRings | BertzCT |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| count | 154.0000 | 154.0000 | 154.0000 | 154.0000 | 154.0000 | 154.0000 | 154.0000 | 154.0000 | 154.0000 | 154.0000 | 154.0000 | 154.0000 | 154.0000 |
| mean | 110.7641 | 1.1638 | 21.5862 | 0.3442 | 1.3896 | 1.6299 | 0.3442 | 0.1364 | 0.7580 | 28.4359 | 7.3506 | 0.2078 | 75.5795 |
| std | 53.0625 | 1.2949 | 16.5745 | 0.5983 | 1.1336 | 2.2198 | 0.5287 | 0.3628 | 0.3155 | 11.5631 | 3.4519 | 0.4527 | 78.3490 |
| min | 18.0150 | -1.6681 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.6138 | 1.0000 | 0.0000 | 0.0000 |
| 25% | 84.1290 | 0.1832 | 9.2300 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6667 | 21.0657 | 5.0000 | 0.0000 | 23.8970 |
| 50% | 100.1830 | 1.0175 | 20.2300 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8377 | 26.9130 | 7.0000 | 0.0000 | 47.9710 |
| 75% | 126.6370 | 1.9771 | 29.4600 | 1.0000 | 2.0000 | 2.0000 | 1.0000 | 0.0000 | 1.0000 | 33.4130 | 9.0000 | 0.0000 | 100.9226 |
| max | 416.0550 | 6.1969 | 78.9000 | 3.0000 | 6.0000 | 15.0000 | 2.0000 | 2.0000 | 1.0000 | 91.4680 | 25.0000 | 2.0000 | 544.8282 |

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
| NumAromaticRings | 0 |
| FractionCSP3 | 0 |
| MolMR | 0 |
| HeavyAtomCount | 0 |
| NumAliphaticRings | 0 |
| BertzCT | 0 |

## Output

- Feature matrix saved to `descriptors/RDKit_Features.csv`
