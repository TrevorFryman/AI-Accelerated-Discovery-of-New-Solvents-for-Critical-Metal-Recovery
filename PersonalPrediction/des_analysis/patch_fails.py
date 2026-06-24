# -*- coding: utf-8 -*-
import json, time, urllib.parse, urllib.request

with open(r'C:\dev\des_analysis\pubchem_properties.json', encoding='utf-8') as f:
    results = json.load(f)

def fetch_json(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if i == retries - 1:
                return {'error': str(e)}
            time.sleep(1)

# Direct aliases -> reuse an already-resolved entry's CID/properties
ALIASES = {
    'ChCl': '2-hydroxyethyl(trimethyl)azanium;chloride',
    '[N1111]Cl': 'tetramethylazanium;chloride',
    '[N3333]Cl': 'tetrapropylazanium;chloride',
}
for alias, target in ALIASES.items():
    results[alias] = dict(results[target])
    results[alias]['query_name'] = alias
    results[alias]['lookup_method'] = 'alias_of:' + target

# Manual SMILES for genuinely new / exotic structures, verified via PubChem SMILES round-trip where possible
MANUAL_SMILES = {
    '[N5555]Br': 'CCCCC[N+](CCCCC)(CCCCC)CCCCC.[Br-]',
    '[NH4]Cl': '[NH4+].[Cl-]',
    'benzyl-bis(2-hydroxyethyl)-methylazanium;chloride': 'OCC[N+](C)(CCO)Cc1ccccc1.[Cl-]',
    'bis(trifluoromethylsulfonyl)azanide;trimethylsulfanium': 'C[S+](C)C.[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F',
    'diethyl(2-hydroxyethyl)azanium;chloride': 'CC[NH+](CC)CCO.[Cl-]',
    '2-methoxyethyl(trimethyl)azanium;chloride': 'COCC[N+](C)(C)C.[Cl-]',
    # exotic ionic liquids, approximate structures parsed from CAS-style names; not expected in PubChem
    '1H-Benzotriazolium, 3-butyl-1-ethyl-, hexafluorophosphate(1-) (1:1) (ACI)': 'CCCC[n+]1nn(CC)c2ccccc12.F[P-](F)(F)(F)(F)F',
    '1H-Imidazolium, 3-[2-[(1R,5S)-6,6-dimethylbicyclo[3.1.1]hept-2-en-2-yl]ethyl]-1-methyl-, methanesulfonate (1:1) (ACI)': 'Cn1cc[n+](CCC2=CCC3CC2C3(C)C)c1.CS(=O)(=O)[O-]',
}

for name, smi in MANUAL_SMILES.items():
    enc = urllib.parse.quote(smi, safe='')
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{enc}/property/IUPACName,CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight/JSON"
    j = fetch_json(url)
    cid = None
    try:
        cid = j['PropertyTable']['Properties'][0]['CID']
    except Exception:
        cid = None
    if cid:
        entry = {'query_name': name, 'lookup_method': 'manual_smiles_verified', 'manual_smiles': smi, 'property_result': j, 'cid': cid}
    else:
        # No PubChem record - keep manual structure, synthesize a minimal property_result so downstream code works uniformly
        entry = {
            'query_name': name, 'lookup_method': 'manual_smiles_unverified', 'manual_smiles': smi, 'cid': None,
            'property_result': {'PropertyTable': {'Properties': [{'CID': None, 'SMILES': smi, 'IUPACName': name, 'MolecularFormula': None, 'MolecularWeight': None}]}},
        }
    results[name] = entry
    print(name, '->', cid)
    time.sleep(0.3)

with open(r'C:\dev\des_analysis\pubchem_properties.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
