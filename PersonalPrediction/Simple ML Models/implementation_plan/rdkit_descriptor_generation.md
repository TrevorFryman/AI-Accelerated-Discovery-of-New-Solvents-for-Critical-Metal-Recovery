# RDKit Molecular Descriptor Generation

Create a Python script `generate_descriptors.py` in the `models` folder to compute RDKit molecular descriptors from the SMILES strings of both DES components and append them as new features to produce an enriched dataset.

## User Review Required

> [!IMPORTANT]
> The script reads `Smiles#1` and `Smiles#2` from the original dataset and computes 7 descriptors per component (14 total new columns) using RDKit:
>
> | Descriptor | RDKit Function |
> |---|---|
> | Molecular Weight | `Descriptors.MolWt` |
> | LogP | `Descriptors.MolLogP` |
> | TPSA | `Descriptors.TPSA` |
> | H-bond Donors | `rdMolDescriptors.CalcNumHBD` |
> | H-bond Acceptors | `rdMolDescriptors.CalcNumHBA` |
> | Rotatable Bonds | `rdMolDescriptors.CalcNumRotatableBonds` |
> | Ring Count | `rdMolDescriptors.CalcNumRings` |
>
> Column naming convention: `C1_MolWt`, `C1_LogP`, ... `C2_MolWt`, `C2_LogP`, ...
>
> Ionic SMILES (e.g., salt notation with `.`) are handled natively by RDKit — descriptors are computed on the full disconnected structure.
>
> Rows with unparseable SMILES will have their descriptor columns set to `NaN` and flagged in the console.
>
> The resulting enriched dataset will be saved to `results/RDKit_DescriptorGeneration/DES_RDKit_Features.csv`.

## Open Questions

None.

## Proposed Changes

### Descriptor Generation Codebase

---

#### [NEW] [generate_descriptors.py](file:///c:/dev/PersonalPrediction/models/generate_descriptors.py)
This new script will perform the following actions:
1. **Load data**: Read the CSV `data/Melting_temperature_appended_35il_03082026.csv`.
2. **Define descriptor function**: A helper `compute_descriptors(smiles)` that takes a SMILES string, parses it with `Chem.MolFromSmiles`, and returns a dict of the 7 descriptor values (or all `NaN` if parsing fails).
3. **Compute descriptors for Component 1**: Apply the helper to `Smiles#1`, prefix columns with `C1_`.
4. **Compute descriptors for Component 2**: Apply the helper to `Smiles#2`, prefix columns with `C2_`.
5. **Assemble enriched dataset**: Concatenate original dataframe with the 14 new descriptor columns.
6. **Report**: Print descriptor statistics (mean, std, null counts) and the number of failed SMILES parses.
7. **Save output**: Write to `results/RDKit_DescriptorGeneration/DES_RDKit_Features.csv`.

## Verification Plan

### Automated Tests
- Run `python models/generate_descriptors.py`.
- Verify that `results/RDKit_DescriptorGeneration/DES_RDKit_Features.csv` is created.
- Verify the output CSV has the original columns plus 14 new descriptor columns (`C1_*` and `C2_*`).
- Inspect the console output for any invalid SMILES warnings and descriptor summary statistics.
