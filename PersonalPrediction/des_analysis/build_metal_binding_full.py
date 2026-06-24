# -*- coding: utf-8 -*-
import csv, json
from rdkit import Chem

def inchikey(smiles):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)

# --- Load Co and Ni LogK predictions, merge by Ligand_SMILES ---
co = {}
with open(r'C:\dev\PersonalPrediction\Metal-Ligand ML Model\Co_LogK_Predictions.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        co[row['Ligand_SMILES']] = row

ni = {}
with open(r'C:\dev\PersonalPrediction\Metal-Ligand ML Model\Ni_LogK_Predictions.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        ni[row['Ligand_SMILES']] = row

# donor atom counts from fragment file
donor_counts = {}
with open(r'C:\dev\PersonalPrediction\Metal-Ligand ML Model\candidate_ligand_fragments_full.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        donor_counts[row['canonical']] = row.get('donor_atom_count')

# common names from the curated Top25_Candidates.csv (only covers 25 of 164)
top25_names = {}
with open(r'C:\dev\PersonalPrediction\Metal-Ligand ML Model\Top25_Candidates.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        top25_names[row['Ligand_SMILES']] = row['Common_name']

all_smiles = sorted(set(co.keys()) | set(ni.keys()))
print('Total unique metal-binding ligands:', len(all_smiles))

ligands = []
for smi in all_smiles:
    c = co.get(smi)
    n = ni.get(smi)
    logk_co = float(c['LogK_mean']) if c else None
    logk_ni = float(n['LogK_mean']) if n else None
    std_co = float(c['LogK_std']) if c else None
    std_ni = float(n['LogK_std']) if n else None
    logk_both = None
    if logk_co is not None and logk_ni is not None:
        logk_both = (logk_co + logk_ni) / 2
    low_conf = (std_co is not None and std_co > 1.5) or (std_ni is not None and std_ni > 1.5)
    ligands.append({
        'smiles': smi,
        'inchikey': inchikey(smi),
        'logK_mean_Co': logk_co,
        'logK_std_Co': std_co,
        'logK_mean_Ni': logk_ni,
        'logK_std_Ni': std_ni,
        'logK_mean_both': logk_both,
        'delta_Ni_minus_Co': (logk_ni - logk_co) if (logk_ni is not None and logk_co is not None) else None,
        'donor_atom_count': donor_counts.get(smi),
        'low_confidence': low_conf,
        'systematic_name_if_top25': top25_names.get(smi),
    })

ligands.sort(key=lambda x: (x['logK_mean_both'] if x['logK_mean_both'] is not None else -999), reverse=True)
for i, l in enumerate(ligands, start=1):
    l['rank_in_full_pool'] = i

with open(r'C:\dev\des_analysis\metal_binding_full.json', 'w', encoding='utf-8') as f:
    json.dump(ligands, f, indent=2)

print('Top 15 by LogK_mean_both:')
for l in ligands[:15]:
    print(l['rank_in_full_pool'], round(l['logK_mean_both'], 3), l['smiles'], l['systematic_name_if_top25'], 'low_conf=', l['low_confidence'])
