"""
generate_rdkit_features.py
==========================
Generates all 2D molecular descriptors available in RDKit (~209 descriptors)
for the GSK solvent dataset.
Saves the features (raw, no imputation) along with the descriptor names
to descriptors/rdkit_descriptors.npz.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# Path bootstrap
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from config import DIRS, COL_SMILES, COL_TARGET
from data_utils import load_dataset


def generate_rdkit_descriptors(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Calculate all available RDKit 2D descriptors for each molecule.
    
    Returns
    -------
    features : np.ndarray, shape (n_samples, n_descriptors)
        The computed descriptor values.
    descriptor_names : list of str
        The names of the computed descriptors.
    """
    # Get all 2D descriptors from RDKit
    # Descriptors._descList is a list of (name, function) tuples
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    log.info("Found %d RDKit 2D molecular descriptors to compute.", len(descriptor_names))
    
    # Initialize the calculator
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    
    features = []
    invalid_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="RDKit Descriptors"):
        smi = row[COL_SMILES]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            log.error("Invalid SMILES at index %d: %s", idx, smi)
            invalid_indices.append(idx)
            # Create a NaN vector as placeholder
            desc_vals = [np.nan] * len(descriptor_names)
        else:
            try:
                # CalcDescriptors returns a tuple of descriptor values
                desc_vals = list(calculator.CalcDescriptors(mol))
            except Exception as e:
                log.error("Error calculating descriptors for index %d (%s): %s", idx, smi, str(e))
                desc_vals = [np.nan] * len(descriptor_names)
                
        features.append(desc_vals)
        
    if invalid_indices:
        log.warning("Found %d invalid SMILES strings!", len(invalid_indices))
        
    return np.array(features, dtype=float), descriptor_names


def main():
    # Load dataset
    df = load_dataset()
    log.info("Loaded dataset with %d rows.", len(df))
    
    # Generate features
    features, feature_names = generate_rdkit_descriptors(df)
    targets = df[COL_TARGET].values
    smiles = df[COL_SMILES].values
    
    # Save output
    out_path = DIRS["descriptors"] / "rdkit_descriptors.npz"
    np.savez_compressed(
        out_path,
        features=features,
        targets=targets,
        smiles=smiles,
        feature_names=np.array(feature_names, dtype=object)
    )
    
    log.info("Successfully generated and saved RDKit descriptors to: %s", out_path)
    log.info("Feature shape: %s, Targets shape: %s", features.shape, targets.shape)


if __name__ == "__main__":
    main()
