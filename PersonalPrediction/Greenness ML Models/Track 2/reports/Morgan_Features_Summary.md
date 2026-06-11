# Morgan Fingerprint Feature Summary

- SMILES column used: `solvent_SMILES`
- Radius: 2
- nBits: 2048
- Total molecules: 154
- Invalid SMILES (dropped): 0
- Final feature matrix shape: 154 rows x 2048 bit columns

## Bit Activity

- Bits set at least once across the dataset: 466 / 2048
- Bits never set (all-zero columns): 1582 / 2048

## Bits-per-Molecule Statistics

| Statistic | Value |
|---|---|
| Mean | 11.69 |
| Median | 11.00 |
| Min | 1 |
| Max | 26 |

## Output

- Feature matrix saved to `descriptors/Morgan_Features.csv`
