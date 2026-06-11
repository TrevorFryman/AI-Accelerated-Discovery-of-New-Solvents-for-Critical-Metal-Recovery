# ChemBERTa Embedding Feature Summary

## Model Information

- Model: `seyonec/ChemBERTa-zinc-base-v1` (Hugging Face Hub)
- Architecture: RoBERTa-based transformer pretrained on ~770k SMILES from the ZINC15 database (ChemBERTa)
- Embedding dimensionality: 768
- Pooling strategy: mean-pooling of the last hidden state across all non-padding tokens, producing one fixed-length vector per molecule

## Generation Summary

- SMILES column used: `solvent_SMILES`
- Total molecules: 154
- Failed embeddings (dropped): 0
- Final feature matrix shape: 154 rows x 768 embedding columns

## Embedding Validation

| Statistic | Value |
|---|---|
| Mean embedding L2 norm | 19.2897 |
| Std of embedding L2 norm | 3.5196 |
| Min embedding L2 norm | 13.3378 |
| Max embedding L2 norm | 26.7887 |
| Mean of per-dimension means | -0.0000 |
| Mean of per-dimension stds | 0.5035 |
| Zero-variance dimensions | 0 / 768 |

## Output

- Feature matrix saved to `descriptors/ChemBERTa_Features.csv`
