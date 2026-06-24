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

with open(r'C:\dev\des_analysis\component_lookup_full.json', encoding='utf-8') as f:
    comp_lookup = json.load(f)
with open(r'C:\dev\des_analysis\metal_binding_full.json', encoding='utf-8') as f:
    mb_ligands = json.load(f)

# map component name -> inchikey
name_to_ik = {}
for name, e in comp_lookup.items():
    name_to_ik[name] = inchikey(e['smiles'])

# map inchikey -> metal-binding ligand record
ik_to_mb = {}
for l in mb_ligands:
    if l['inchikey']:
        ik_to_mb[l['inchikey']] = l

matched_names = {}
for name, ik in name_to_ik.items():
    if ik and ik in ik_to_mb:
        matched_names[name] = ik_to_mb[ik]

print(f"{len(matched_names)} of {len(comp_lookup)} MP components have a Metal-Binding model prediction:")
for name, l in sorted(matched_names.items(), key=lambda kv: kv[1]['rank_in_full_pool']):
    print(f"  rank {l['rank_in_full_pool']:>3} logK_both={l['logK_mean_both']:.3f}  {name}  ({comp_lookup[name]['common_name']})")

with open(r'C:\dev\des_analysis\name_to_inchikey.json', 'w', encoding='utf-8') as f:
    json.dump(name_to_ik, f, indent=2)
with open(r'C:\dev\des_analysis\matched_component_names.json', 'w', encoding='utf-8') as f:
    json.dump({k: v for k, v in matched_names.items()}, f, indent=2, default=str)

# --- Load full MP candidate master list (RDKit variant), 2006 rows ---
with open(r'C:\dev\PersonalPrediction\Simple ML Models\Optimized Model\CSVs\Candidate_Master_List_rdkit.csv', encoding='utf-8') as f:
    mp_rows = list(csv.DictReader(f))
print('\nTotal MP candidate pairs:', len(mp_rows))

both_matched_pairs = []
for row in mp_rows:
    c1, c2 = row['Component#1'], row['Component#2']
    m1 = matched_names.get(c1)
    m2 = matched_names.get(c2)
    if m1 and m2:
        both_matched_pairs.append((row, m1, m2))

print('Pairs where BOTH components have a Metal-Binding prediction:', len(both_matched_pairs))
for row, m1, m2 in both_matched_pairs:
    print(' ', row['Component#1'], '+', row['Component#2'], '| logK_both1=', round(m1['logK_mean_both'],2), 'logK_both2=', round(m2['logK_mean_both'],2), '| RoomTemp=',row['RoomTemp_flag'],'WarmTemp=',row['WarmTemp_flag'], '| passes_DES_rule=', row['passes_DES_rule'])
