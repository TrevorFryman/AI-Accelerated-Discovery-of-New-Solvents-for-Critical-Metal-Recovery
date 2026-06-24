# -*- coding: utf-8 -*-
"""
Score the Greenness model's per-molecule G-score for all 205 unique components in the
full melting-point candidate universe (Candidate_Master_List_rdkit.csv), using the
repo's own trained model and descriptor pipeline. This closes the greenness coverage gap
(previously only 25 + 10 = 35 of 2006 pairs had any greenness score at all).
"""
import json
import joblib
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski

MODEL_PATH = Path(r"C:\dev\PersonalPrediction\Greenness ML Models\Final Model\model\rdkit_xgboost_model.pkl")

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


def main():
    model = joblib.load(MODEL_PATH)
    with open(r'C:\dev\des_analysis\component_lookup_full.json', encoding='utf-8') as f:
        comp_lookup = json.load(f)

    gscores = {}
    failed = []
    for name, e in comp_lookup.items():
        smi = e['smiles']
        desc = compute_descriptors(smi)
        if desc is None:
            failed.append(name)
            continue
        X = np.array([[desc[f] for f in FEATURE_COLS]])
        gscores[name] = float(model.predict(X)[0])

    print(f"Scored {len(gscores)} of {len(comp_lookup)} components. Failed: {failed}")

    with open(r'C:\dev\des_analysis\component_greenness_scores.json', 'w', encoding='utf-8') as f:
        json.dump(gscores, f, indent=2)

    vals = sorted(gscores.values())
    print(f"Range: {vals[0]:.3f} - {vals[-1]:.3f}, median: {vals[len(vals)//2]:.3f}")


if __name__ == "__main__":
    main()
