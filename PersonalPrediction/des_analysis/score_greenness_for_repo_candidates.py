# -*- coding: utf-8 -*-
"""
Apply the existing, trained Greenness RDKit-XGBoost model (Final Model/model/rdkit_xgboost_model.pkl)
to the 10 candidates selected in "Repository Solvent Selection.md", which were never scored by the
Greenness model (it has only ever been run against 25 sugar/NADES-heavy pairs).

Uses the exact same descriptor set and weighted-average DES scoring logic as the repo's own
predict_top25_candidates.py, applied to a new candidate list instead of the original dataset.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski

MODEL_PATH = Path(r"C:\dev\PersonalPrediction\Greenness ML Models\Final Model\model\rdkit_xgboost_model.pkl")
OUTPUT_PATH = Path(r"C:\dev\PersonalPrediction\Greenness ML Models\Final Model\results\repository_candidates_greenness.csv")

DESCRIPTOR_FUNCS = {
    "MolWt":              Descriptors.MolWt,
    "LogP":               Crippen.MolLogP,
    "TPSA":               Descriptors.TPSA,
    "NumHDonors":         Lipinski.NumHDonors,
    "NumHAcceptors":      Lipinski.NumHAcceptors,
    "NumRotatableBonds":  Descriptors.NumRotatableBonds,
    "RingCount":          Descriptors.RingCount,
    "NumAromaticRings":   Descriptors.NumAromaticRings,
    "FractionCSP3":       Descriptors.FractionCSP3,
    "MolMR":              Crippen.MolMR,
    "HeavyAtomCount":     Descriptors.HeavyAtomCount,
    "NumAliphaticRings":  Descriptors.NumAliphaticRings,
    "BertzCT":            Descriptors.BertzCT,
}
FEATURE_COLS = list(DESCRIPTOR_FUNCS.keys())


def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {name: func(mol) for name, func in DESCRIPTOR_FUNCS.items()}


# The 10 "Repository Solvent Selection.md" candidates, each at the molar ratio
# featured in that report (the room-temperature or best-confidence formulation).
CANDIDATES = [
    {"Component#1": "1,10-Phenanthroline", "Component#2": "Thymol",
     "Smiles#1": "c1cnc2c(c1)ccc1cccnc12", "Smiles#2": "Cc1ccc(C(C)C)c(O)c1",
     "X#1": 0.203, "X#2": 0.797, "Tmelt, K": 283.25128, "Metal_score": 11.181, "Source_rank": 1},
    {"Component#1": "1,10-Phenanthroline", "Component#2": "Chlorothymol",
     "Smiles#1": "c1cnc2c(c1)ccc1cccnc12", "Smiles#2": "Cc1cc(O)c(C(C)C)cc1Cl",
     "X#1": 0.502, "X#2": 0.498, "Tmelt, K": 297.234, "Metal_score": 10.786, "Source_rank": 2},
    {"Component#1": "Lidocaine", "Component#2": "Phenyl salicylate",
     "Smiles#1": "CCN(CC)CC(=O)Nc1c(C)cccc1C", "Smiles#2": "O=C(Oc1ccccc1)c1ccccc1O",
     "X#1": 0.43, "X#2": 0.57, "Tmelt, K": 291.73877, "Metal_score": 9.21, "Source_rank": 3},
    {"Component#1": "Menthol", "Component#2": "Salicylic acid",
     "Smiles#1": "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O", "Smiles#2": "O=C(O)c1ccccc1O",
     "X#1": 0.85, "X#2": 0.15, "Tmelt, K": 297.9677, "Metal_score": 8.041, "Source_rank": 4},
    {"Component#1": "Lidocaine", "Component#2": "Tetracaine",
     "Smiles#1": "CCN(CC)CC(=O)Nc1c(C)cccc1C", "Smiles#2": "CCCCNc1ccc(C(=O)OCCN(C)C)cc1",
     "X#1": 0.5, "X#2": 0.5, "Tmelt, K": 303.33936, "Metal_score": 7.498, "Source_rank": 5},
    {"Component#1": "Menthol", "Component#2": "Phenyl salicylate",
     "Smiles#1": "CC1CCC(C(C)C)C(O)C1", "Smiles#2": "O=C(Oc1ccccc1)c1ccccc1O",
     "X#1": 0.5, "X#2": 0.5, "Tmelt, K": 296.1716, "Metal_score": 6.305, "Source_rank": 6},
    {"Component#1": "Betaine", "Component#2": "Malic acid",
     "Smiles#1": "C[N+](C)(C)CC(=O)[O-]", "Smiles#2": "O=C(O)CC(O)C(=O)O",
     "X#1": 0.5, "X#2": 0.5, "Tmelt, K": 318.43155, "Metal_score": 5.671, "Source_rank": 7},
    {"Component#1": "Lidocaine", "Component#2": "Prilocaine",
     "Smiles#1": "CCN(CC)CC(=O)Nc1c(C)cccc1C", "Smiles#2": "CCCNC(C)C(=O)Nc1ccccc1C",
     "X#1": 0.5, "X#2": 0.5, "Tmelt, K": 290.21628, "Metal_score": 5.583, "Source_rank": 8},
    {"Component#1": "Thymol", "Component#2": "Cyclohexanecarboxylic acid",
     "Smiles#1": "Cc1ccc(C(C)C)c(O)c1", "Smiles#2": "O=C(O)C1CCCCC1",
     "X#1": 0.29, "X#2": 0.71, "Tmelt, K": 276.71942, "Metal_score": 5.491, "Source_rank": 9},
    {"Component#1": "Lidocaine", "Component#2": "Camphor",
     "Smiles#1": "CCN(CC)CC(=O)Nc1c(C)cccc1C", "Smiles#2": "CC12CCC(CC1=O)C2(C)C",
     "X#1": 0.46, "X#2": 0.54, "Tmelt, K": 290.71033, "Metal_score": 5.415, "Source_rank": 10},
]


def main():
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")

    records = []
    for cand in CANDIDATES:
        desc1 = compute_descriptors(cand["Smiles#1"])
        desc2 = compute_descriptors(cand["Smiles#2"])
        if desc1 is None or desc2 is None:
            print(f"SKIPPED (invalid SMILES): {cand['Component#1']} + {cand['Component#2']}")
            continue

        X1 = np.array([[desc1[f] for f in FEATURE_COLS]])
        X2 = np.array([[desc2[f] for f in FEATURE_COLS]])
        gscore1 = float(model.predict(X1)[0])
        gscore2 = float(model.predict(X2)[0])
        x1, x2 = cand["X#1"], cand["X#2"]
        gscore_des = x1 * gscore1 + x2 * gscore2

        records.append({
            "Source_rank_in_Repository_Solvent_Selection": cand["Source_rank"],
            "Component#1": cand["Component#1"],
            "Component#2": cand["Component#2"],
            "Smiles#1": cand["Smiles#1"],
            "Smiles#2": cand["Smiles#2"],
            "X#1 (molar fraction)": x1,
            "X#2 (molar fraction)": x2,
            "Tmelt, K": round(cand["Tmelt, K"], 2),
            "Metal_score_log K1 sum": cand["Metal_score"],
            "G-score_comp1": round(gscore1, 4),
            "G-score_comp2": round(gscore2, 4),
            "G-score_DES": round(gscore_des, 4),
        })

    results = pd.DataFrame(records)
    results = results.sort_values("G-score_DES", ascending=False).reset_index(drop=True)
    results.index += 1
    results.index.name = "Greenness_Rank"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}\n")
    print(results[["Component#1", "Component#2", "G-score_comp1", "G-score_comp2", "G-score_DES", "Metal_score_log K1 sum"]].to_string())

    # also compare against the existing top25 list's score range for context
    existing = pd.read_csv(r"C:\dev\PersonalPrediction\Greenness ML Models\Final Model\results\top25_DES_candidates.csv")
    print(f"\nFor reference, existing top25_DES_candidates.csv G-score_DES range: "
          f"{existing['G-score_DES'].min():.3f} - {existing['G-score_DES'].max():.3f} (n={len(existing)})")


if __name__ == "__main__":
    main()
