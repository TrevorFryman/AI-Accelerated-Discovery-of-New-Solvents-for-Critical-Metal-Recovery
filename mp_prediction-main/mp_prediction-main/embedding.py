#!/usr/bin/env python3
"""
Embedding utilities for deep eutectic solvent (DES) melting point prediction.

Key requirements:
- Canonicalize SMILES (RDKit) before embedding.
- Support 3 embedding methods:
    1) chemberta  (transformers, mean pooling)
    2) morgan     (RDKit Morgan fingerprints)
    3) gnn        (a lightweight message-passing GNN implemented in pure PyTorch; no PyG required)

Design:
- For chemberta/morgan, embeddings are computed per unique canonical SMILES and cached.
- For gnn, embeddings are produced by MolGNNEncoder during model training (end-to-end).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


# ----------------------------
# Canonicalization
# ----------------------------
def canonicalize_smiles(smiles: str) -> str:
    """
    Canonicalize SMILES using RDKit.
    Returns canonical isomeric SMILES (keeps stereochemistry when present).

    Raises:
        ValueError if SMILES cannot be parsed.
    """
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        raise ValueError("SMILES is None/NaN")
    s = str(smiles).strip()
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonicalize_series(smiles_list: Iterable[str]) -> List[str]:
    return [canonicalize_smiles(s) for s in smiles_list]


# ----------------------------
# Morgan fingerprint
# ----------------------------
@dataclass
class MorganParams:
    radius: int = 2
    n_bits: int = 2048
    use_counts: bool = False
    use_chirality: bool = True


def morgan_fingerprint(smiles: str, params: MorganParams) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES after canonicalization: {smiles}")
    if params.use_counts:
        fp = AllChem.GetHashedMorganFingerprint(
            mol, radius=params.radius, nBits=params.n_bits, useChirality=params.use_chirality
        )
        arr = np.zeros((params.n_bits,), dtype=np.float32)
        for k, v in fp.GetNonzeroElements().items():
            arr[k % params.n_bits] += float(v)
        return arr
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=params.radius, nBits=params.n_bits, useChirality=params.use_chirality
        )
        arr = np.zeros((params.n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr.astype(np.float32)


def embed_morgan_unique(
    canonical_smiles: List[str],
    params: MorganParams,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Returns:
        feats: (N_unique, n_bits)
        index: mapping canonical_smiles -> row index
    """
    uniq = list(dict.fromkeys(canonical_smiles))
    if len(uniq) == 0:
        raise ValueError("No SMILES provided for ChemBERTa embedding (0 unique). Check your input data / dropna filtering.")
    if len(uniq) == 0:
        raise ValueError("No SMILES provided for embedding (0 unique). Check your input data / dropna filtering.")
    index = {s: i for i, s in enumerate(uniq)}
    feats = np.stack([morgan_fingerprint(s, params) for s in uniq], axis=0).astype(np.float32)
    return feats, index


# ----------------------------
# ChemBERTa (transformers)
# ----------------------------
@dataclass
class ChemBERTaParams:
    model_name: str = "DeepChem/ChemBERTa-77M-MTR"
    pooling: str = "mean"  # "mean" or "cls"
    max_length: int = 256
    batch_size: int = 64


def embed_chemberta_unique(
    canonical_smiles: List[str],
    params: ChemBERTaParams,
    device: str = "cuda",
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Compute ChemBERTa embeddings for unique canonical SMILES.
    Returns embeddings on CPU as float32 numpy array and a mapping.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    uniq = list(dict.fromkeys(canonical_smiles))
    index = {s: i for i, s in enumerate(uniq)}

    tok = AutoTokenizer.from_pretrained(params.model_name)
    model = AutoModel.from_pretrained(params.model_name)
    model.to(torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"))
    model.eval()

    all_emb = []
    with torch.no_grad():
        for start in range(0, len(uniq), params.batch_size):
            batch = uniq[start:start + params.batch_size]
            inputs = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=params.max_length,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            out = model(**inputs).last_hidden_state  # (B, L, H)

            if params.pooling == "cls":
                emb = out[:, 0, :]
            else:
                # mean pooling over sequence length (including padding tokens)
                # For simplicity, use attention_mask to exclude padding.
                mask = inputs.get("attention_mask", None)
                if mask is None:
                    emb = out.mean(dim=1)
                else:
                    mask_f = mask.unsqueeze(-1).float()  # (B, L, 1)
                    emb = (out * mask_f).sum(dim=1) / (mask_f.sum(dim=1).clamp(min=1.0))

            all_emb.append(emb.detach().cpu())

    feats = torch.cat(all_emb, dim=0).numpy().astype(np.float32)
    return feats, index


# ----------------------------
# Lightweight GNN encoder (pure PyTorch)
# ----------------------------
# Atom features (simple, robust, DES-friendly)
ATOM_LIST = list(range(1, 119))  # atomic numbers
MAX_DEG = 5

def _one_hot(x: int, allowable: List[int]) -> List[int]:
    if x not in allowable:
        x = allowable[-1]
    return [int(x == a) for a in allowable]

def atom_features(atom: Chem.Atom) -> np.ndarray:
    z = atom.GetAtomicNum()
    deg = min(atom.GetDegree(), MAX_DEG)
    is_arom = int(atom.GetIsAromatic())
    formal = atom.GetFormalCharge()
    hyb = int(atom.GetHybridization())
    feats = (
        _one_hot(z, ATOM_LIST)
        + _one_hot(deg, list(range(MAX_DEG + 1)))
        + [is_arom, formal]
        + _one_hot(hyb, list(range(1, 8)))  # RDKit HybridizationType enum ints are stable enough
    )
    return np.array(feats, dtype=np.float32)

def mol_to_graph(smiles: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        X: (n_atoms, n_feat)
        A: (n_atoms, n_atoms) adjacency (0/1)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for graph: {smiles}")
    n = mol.GetNumAtoms()
    X = np.stack([atom_features(mol.GetAtomWithIdx(i)) for i in range(n)], axis=0)
    A = np.zeros((n, n), dtype=np.float32)
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        A[i, j] = 1.0
        A[j, i] = 1.0
    np.fill_diagonal(A, 1.0)  # self loops
    return X, A

def batch_graphs(graphs: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pad graphs to a common size for batching.

    Returns:
        Xb: (B, Nmax, F)
        Ab: (B, Nmax, Nmax)
        mask: (B, Nmax) 1 for real atoms else 0
    """
    B = len(graphs)
    nmax = max(g[0].shape[0] for g in graphs)
    fdim = graphs[0][0].shape[1]
    Xb = np.zeros((B, nmax, fdim), dtype=np.float32)
    Ab = np.zeros((B, nmax, nmax), dtype=np.float32)
    mask = np.zeros((B, nmax), dtype=np.float32)
    for i, (X, A) in enumerate(graphs):
        n = X.shape[0]
        Xb[i, :n, :] = X
        Ab[i, :n, :n] = A
        mask[i, :n] = 1.0
    return Xb, Ab, mask

# Pure-torch GIN-style message passing
import torch
import torch.nn as nn
import torch.nn.functional as F

class MolGNNEncoder(nn.Module):
    """
    A small message passing network that outputs a fixed-size embedding per molecule.
    This is meant to be trained end-to-end as part of the physics learning model.

    Recommended defaults for DES MP prediction (small/medium datasets):
        hidden_dim=128, layers=3, dropout=0.1
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 256, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(layers)
        ])
        self.eps = nn.Parameter(torch.zeros(layers))
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, X: torch.Tensor, A: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        X: (B, N, F)
        A: (B, N, N) adjacency with self loops
        mask: (B, N) 1/0
        returns: (B, out_dim)
        """
        h = self.in_proj(X)  # (B,N,H)
        for k, mlp in enumerate(self.mlps):
            # neighbor aggregation
            neigh = torch.bmm(A, h)  # (B,N,H)
            h = mlp((1.0 + self.eps[k]) * h + neigh)
            h = F.relu(h)

        # masked mean pooling
        mask_f = mask.unsqueeze(-1)  # (B,N,1)
        h_sum = (h * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        h_pool = h_sum / denom
        return self.out_proj(h_pool)
