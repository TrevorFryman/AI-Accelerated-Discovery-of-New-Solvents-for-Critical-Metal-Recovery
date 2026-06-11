"""
ChemBERTa Embedding Pipeline - Feature Generation

Loads GSK_dataset.csv, identifies the SMILES column, and generates
molecular embeddings for each solvent using a pretrained ChemBERTa model.

ChemBERTa model used: "seyonec/ChemBERTa-zinc-base-v1"
(RoBERTa-style transformer pretrained on ~770k molecules from ZINC15,
via Hugging Face `transformers`).

Embedding extraction method: mean-pooling of the last hidden state across
all (non-padding) token positions, producing one fixed-length vector per
molecule.

Outputs:
- descriptors/ChemBERTa_Features.csv (embeddings + identifiers + target)
- reports/ChemBERTa_Features_Summary.md (validation / summary report)

This pipeline is standalone and does not reference Morgan Fingerprint
or RDKit Descriptor pipelines.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR.parent / "GSK_dataset.csv"
DESCRIPTORS_DIR = BASE_DIR / "descriptors"
REPORTS_DIR = BASE_DIR / "reports"

OUTPUT_PATH = DESCRIPTORS_DIR / "ChemBERTa_Features.csv"
REPORT_PATH = REPORTS_DIR / "ChemBERTa_Features_Summary.md"

TARGET = "G-score"
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"


def main():
    df = pd.read_csv(DATA_PATH)

    # Identify the SMILES column dynamically (do not assume "SMILES")
    smiles_col = None
    for col in df.columns:
        if "smiles" in col.lower():
            smiles_col = col
            break
    if smiles_col is None:
        raise ValueError(
            f"No SMILES column found in dataset. Available columns: {list(df.columns)}"
        )
    print(f"Using SMILES column: '{smiles_col}'")

    print(f"Loading ChemBERTa model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    embedding_dim = model.config.hidden_size
    print(f"Embedding dimensionality: {embedding_dim}")

    embeddings = []
    invalid_indices = []
    with torch.no_grad():
        for idx, smiles in df[smiles_col].items():
            try:
                inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
                outputs = model(**inputs)
                last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_size)
                attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)

                # Mean-pool over non-padding tokens
                summed = (last_hidden * attention_mask).sum(dim=1)
                counts = attention_mask.sum(dim=1)
                mean_pooled = (summed / counts).squeeze(0).numpy()
                embeddings.append(mean_pooled)
            except Exception as e:
                print(f"Warning: failed to embed row {idx} (SMILES='{smiles}'): {e}")
                invalid_indices.append(idx)
                embeddings.append(np.full(embedding_dim, np.nan))

    embeddings = np.array(embeddings, dtype=float)
    embed_cols = [f"chemberta_dim_{i}" for i in range(embedding_dim)]
    embed_df = pd.DataFrame(embeddings, columns=embed_cols, index=df.index)

    id_cols = [c for c in ["solvent_common_name", "CAS Number", smiles_col] if c in df.columns]
    output_df = pd.concat([df[id_cols], df[[TARGET]], embed_df], axis=1)

    if invalid_indices:
        print(f"Warning: {len(invalid_indices)} row(s) failed embedding and were dropped.")
        output_df = output_df.drop(index=invalid_indices)

    DESCRIPTORS_DIR.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    # Validation stats
    valid_embeddings = embeddings[~np.isnan(embeddings).any(axis=1)]
    norms = np.linalg.norm(valid_embeddings, axis=1)
    col_means = valid_embeddings.mean(axis=0)
    col_stds = valid_embeddings.std(axis=0)
    n_zero_var = (col_stds == 0).sum()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# ChemBERTa Embedding Feature Summary\n\n")
    lines.append("## Model Information\n\n")
    lines.append(f"- Model: `{MODEL_NAME}` (Hugging Face Hub)\n")
    lines.append(
        "- Architecture: RoBERTa-based transformer pretrained on ~770k SMILES "
        "from the ZINC15 database (ChemBERTa)\n"
    )
    lines.append(f"- Embedding dimensionality: {embedding_dim}\n")
    lines.append(
        "- Pooling strategy: mean-pooling of the last hidden state across all "
        "non-padding tokens, producing one fixed-length vector per molecule\n\n"
    )

    lines.append("## Generation Summary\n\n")
    lines.append(f"- SMILES column used: `{smiles_col}`\n")
    lines.append(f"- Total molecules: {len(df)}\n")
    lines.append(f"- Failed embeddings (dropped): {len(invalid_indices)}\n")
    lines.append(f"- Final feature matrix shape: {output_df.shape[0]} rows x {embedding_dim} embedding columns\n\n")

    lines.append("## Embedding Validation\n\n")
    lines.append("| Statistic | Value |\n")
    lines.append("|---|---|\n")
    lines.append(f"| Mean embedding L2 norm | {norms.mean():.4f} |\n")
    lines.append(f"| Std of embedding L2 norm | {norms.std():.4f} |\n")
    lines.append(f"| Min embedding L2 norm | {norms.min():.4f} |\n")
    lines.append(f"| Max embedding L2 norm | {norms.max():.4f} |\n")
    lines.append(f"| Mean of per-dimension means | {col_means.mean():.4f} |\n")
    lines.append(f"| Mean of per-dimension stds | {col_stds.mean():.4f} |\n")
    lines.append(f"| Zero-variance dimensions | {n_zero_var} / {embedding_dim} |\n\n")

    if invalid_indices:
        lines.append("## Failed Embedding Rows (dropped)\n\n")
        for idx in invalid_indices:
            lines.append(f"- Row index {idx}: `{df.loc[idx, smiles_col]}`\n")
        lines.append("\n")

    lines.append("## Output\n\n")
    lines.append("- Feature matrix saved to `descriptors/ChemBERTa_Features.csv`\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print("CHEMBERTA EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print(f"Molecules processed: {len(df)}")
    print(f"Failed embeddings dropped: {len(invalid_indices)}")
    print(f"Embedding dimensionality: {embedding_dim}")
    print(f"Zero-variance dimensions: {n_zero_var} / {embedding_dim}")
    print(f"Feature matrix shape: {output_df.shape}")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
