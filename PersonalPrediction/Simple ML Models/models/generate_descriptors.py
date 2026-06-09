import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Suppress RDKit warnings for invalid SMILES
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ---------------------------------------------------------------------------
# Descriptor Configuration
# ---------------------------------------------------------------------------
DESCRIPTOR_MAP = {
    'MolWt':       lambda mol: Descriptors.MolWt(mol),
    'LogP':        lambda mol: Descriptors.MolLogP(mol),
    'TPSA':        lambda mol: Descriptors.TPSA(mol),
    'HBD':         lambda mol: rdMolDescriptors.CalcNumHBD(mol),
    'HBA':         lambda mol: rdMolDescriptors.CalcNumHBA(mol),
    'RotBonds':    lambda mol: rdMolDescriptors.CalcNumRotatableBonds(mol),
    'RingCount':   lambda mol: rdMolDescriptors.CalcNumRings(mol),
}

def compute_descriptors(smiles: str) -> dict:
    """
    Compute RDKit molecular descriptors from a SMILES string.

    Parameters
    ----------
    smiles : str
        A valid (or potentially invalid) SMILES string.

    Returns
    -------
    dict
        Dictionary of descriptor name -> float value.
        All values are NaN if the SMILES cannot be parsed.
    """
    nan_result = {name: np.nan for name in DESCRIPTOR_MAP}
    
    if not isinstance(smiles, str) or smiles.strip() == '':
        return nan_result
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return nan_result
    
    result = {}
    for name, func in DESCRIPTOR_MAP.items():
        try:
            result[name] = func(mol)
        except Exception:
            result[name] = np.nan
    return result


def generate_descriptors():
    # 1. Setup paths and directories
    dataset_path = os.path.join('data', 'Melting_temperature_appended_35il_03082026.csv')
    results_dir = os.path.join('results', 'RDKit_DescriptorGeneration')
    output_path = os.path.join(results_dir, 'DES_RDKit_Features.csv')
    os.makedirs(results_dir, exist_ok=True)

    # 2. Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}  ({len(df)} rows, {len(df.columns)} columns)\n")

    # Verify required SMILES columns are present
    for col in ['Smiles#1', 'Smiles#2']:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in dataset.")

    # 3. Compute descriptors for Component 1 (Smiles#1)
    print("Computing descriptors for Component 1 (Smiles#1)...")
    c1_records = df['Smiles#1'].apply(compute_descriptors)
    c1_df = pd.DataFrame(c1_records.tolist()).add_prefix('C1_')

    c1_failed = c1_df['C1_MolWt'].isna().sum()
    if c1_failed > 0:
        print(f"  [WARNING] {c1_failed} SMILES in Smiles#1 could not be parsed -> set to NaN.")
    else:
        print(f"  All {len(df)} Smiles#1 entries parsed successfully.")

    # 4. Compute descriptors for Component 2 (Smiles#2)
    print("\nComputing descriptors for Component 2 (Smiles#2)...")
    c2_records = df['Smiles#2'].apply(compute_descriptors)
    c2_df = pd.DataFrame(c2_records.tolist()).add_prefix('C2_')

    c2_failed = c2_df['C2_MolWt'].isna().sum()
    if c2_failed > 0:
        print(f"  [WARNING] {c2_failed} SMILES in Smiles#2 could not be parsed -> set to NaN.")
    else:
        print(f"  All {len(df)} Smiles#2 entries parsed successfully.")

    # 5. Assemble enriched dataset
    df_enriched = pd.concat([df, c1_df, c2_df], axis=1)

    new_cols = list(c1_df.columns) + list(c2_df.columns)
    print(f"\nNew descriptor columns added ({len(new_cols)} total):")
    for col in new_cols:
        print(f"  - {col}")

    # 6. Report descriptor statistics
    print("\n" + "=" * 55)
    print("DESCRIPTOR SUMMARY STATISTICS:")
    print("=" * 55)
    print(df_enriched[new_cols].describe().T.to_string())

    null_counts = df_enriched[new_cols].isnull().sum()
    if null_counts.any():
        print("\nNull counts per descriptor column:")
        print(null_counts[null_counts > 0].to_string())
    else:
        print("\nNo null values in any descriptor column.")

    # 7. Save enriched dataset
    df_enriched.to_csv(output_path, index=False)
    print(f"\n{'=' * 55}")
    print(f"Enriched dataset saved to: {output_path}")
    print(f"Final dataset shape: {df_enriched.shape}  "
          f"({len(df_enriched)} rows, {len(df_enriched.columns)} columns)")
    print(f"{'=' * 55}")
    print("\nWorkflow completed successfully.")


if __name__ == '__main__':
    generate_descriptors()
