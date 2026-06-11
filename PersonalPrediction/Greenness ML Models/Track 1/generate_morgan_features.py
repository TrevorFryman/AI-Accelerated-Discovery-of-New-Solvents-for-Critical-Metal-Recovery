"""
generate_morgan_features.py
===========================
Generates 2048-bit Morgan fingerprints (radius=2, equivalent to ECFP4)
for the GSK solvent dataset using RDKit.
Saves the features to descriptors/morgan_fingerprints.npz.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
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

from config import DIRS, COL_SMILES, COL_TARGET, MORGAN_RADIUS, MORGAN_N_BITS
from data_utils import load_dataset


def generate_fingerprints(df: pd.DataFrame) -> np.ndarray:
    """
    Generate Morgan fingerprints for each SMILES string in the DataFrame.
    """
    fps = []
    invalid_indices = []
    
    log.info("Generating 2048-bit Morgan fingerprints (radius=%d)...", MORGAN_RADIUS)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Morgan FPs"):
        smi = row[COL_SMILES]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            log.error("Invalid SMILES at index %d: %s", idx, smi)
            invalid_indices.append(idx)
            # Create a zero vector as placeholder so we maintain indexing
            fp_arr = np.zeros(MORGAN_N_BITS, dtype=np.uint8)
        else:
            # Generate Morgan fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=MORGAN_RADIUS,
                nBits=MORGAN_N_BITS
            )
            # Convert to numpy array
            fp_arr = np.zeros(MORGAN_N_BITS, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, fp_arr)
            
        fps.append(fp_arr)
        
    if invalid_indices:
        log.warning("Found %d invalid SMILES strings!", len(invalid_indices))
        
    return np.vstack(fps)


def main():
    # Load dataset
    df = load_dataset()
    log.info("Loaded dataset with %d rows.", len(df))
    
    # Generate features
    features = generate_fingerprints(df)
    targets = df[COL_TARGET].values
    smiles = df[COL_SMILES].values
    
    # Save output
    out_path = DIRS["descriptors"] / "morgan_fingerprints.npz"
    np.savez_compressed(
        out_path,
        features=features,
        targets=targets,
        smiles=smiles
    )
    
    log.info("Successfully generated and saved Morgan fingerprints to: %s", out_path)
    log.info("Feature shape: %s, Targets shape: %s", features.shape, targets.shape)


if __name__ == "__main__":
    main()
