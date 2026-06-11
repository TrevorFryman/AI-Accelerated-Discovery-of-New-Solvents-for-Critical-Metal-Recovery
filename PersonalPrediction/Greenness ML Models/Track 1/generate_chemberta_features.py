"""
generate_chemberta_features.py
==============================
Generates 768-dimensional ChemBERTa embeddings (using the [CLS] token of the
seyonec/ChemBERTa-zinc-base-v1 model) for the GSK solvent dataset.
Saves the features to descriptors/chemberta_embeddings.npz.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
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

from config import DIRS, COL_SMILES, COL_TARGET, CHEMBERTA_MODEL, CHEMBERTA_DEVICE, CHEMBERTA_BATCH
from data_utils import load_dataset


def generate_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Generate ChemBERTa [CLS] token embeddings for all SMILES strings.
    """
    smiles_list = df[COL_SMILES].tolist()
    
    # Device configuration
    if CHEMBERTA_DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("CUDA is available. Using GPU for ChemBERTa inference.")
    else:
        device = torch.device("cpu")
        log.info("CUDA is not available or disabled. Using CPU for ChemBERTa inference.")
        
    log.info("Loading ChemBERTa tokenizer and model: %s", CHEMBERTA_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
    model = AutoModel.from_pretrained(CHEMBERTA_MODEL)
    model = model.to(device)
    model.eval()
    
    embeddings = []
    
    # Process in batches
    n_batches = int(np.ceil(len(smiles_list) / CHEMBERTA_BATCH))
    log.info("Running ChemBERTa embedding extraction (Total batches: %d, batch size: %d)...", n_batches, CHEMBERTA_BATCH)
    
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="ChemBERTa Embeddings"):
            start_idx = i * CHEMBERTA_BATCH
            end_idx = min(start_idx + CHEMBERTA_BATCH, len(smiles_list))
            batch_smiles = smiles_list[start_idx:end_idx]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs)
            
            # Extract CLS token embedding (first token of last hidden state)
            # shape of last_hidden_state: (batch_size, sequence_length, hidden_size)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
            
    return np.vstack(embeddings)


def main():
    # Load dataset
    df = load_dataset()
    log.info("Loaded dataset with %d rows.", len(df))
    
    # Generate features
    features = generate_embeddings(df)
    targets = df[COL_TARGET].values
    smiles = df[COL_SMILES].values
    
    # Save output
    out_path = DIRS["descriptors"] / "chemberta_embeddings.npz"
    np.savez_compressed(
        out_path,
        features=features,
        targets=targets,
        smiles=smiles
    )
    
    log.info("Successfully generated and saved ChemBERTa embeddings to: %s", out_path)
    log.info("Feature shape: %s, Targets shape: %s", features.shape, targets.shape)


if __name__ == "__main__":
    main()
