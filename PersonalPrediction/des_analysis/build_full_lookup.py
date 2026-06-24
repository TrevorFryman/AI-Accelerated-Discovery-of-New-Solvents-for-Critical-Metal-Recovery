# -*- coding: utf-8 -*-
import json, re

with open(r'C:\dev\des_analysis\full_mp_unique_names.json', encoding='utf-8') as f:
    all_names = json.load(f)
with open(r'C:\dev\des_analysis\pubchem_properties.json', encoding='utf-8') as f:
    props = json.load(f)
with open(r'C:\dev\des_analysis\component_lookup.json', encoding='utf-8') as f:
    curated48 = json.load(f)

CAS_RE = re.compile(r'^\d{2,7}-\d{2}-\d(\s*\(.*\))?$')
JUNK_RE = re.compile(r'^(RefChem:|DTXSID|DTXCID|EINECS|CHEBI:|AKOS|NSC ?\d|UNII|GlyTouCan|CAS-)', re.I)

def smart_titlecase(s):
    if s.isupper() and len(s) > 3:
        # title-case but keep common chemistry connector words lowercase
        words = s.split(' ')
        out = []
        for w in words:
            if w.lower() in ('of', 'acid', 'and', 'di', 'tri'):
                out.append(w.capitalize())
            else:
                out.append(w.capitalize())
        return ' '.join(out)
    return s

def pick_common_name(name, syns):
    for s in syns:
        if CAS_RE.match(s.strip()):
            continue
        if JUNK_RE.match(s.strip()):
            continue
        if re.match(r'^\d', s.strip()):
            continue
        return smart_titlecase(s.strip())
    return name  # fallback: the systematic name itself

lookup = dict(curated48)  # already has good manual overrides

for name in all_names:
    if name in lookup:
        continue
    e = props.get(name)
    if e is None:
        continue
    pr = e.get('property_result', {})
    try:
        p = pr['PropertyTable']['Properties'][0]
    except Exception:
        p = {}
    smiles = p.get('SMILES') or p.get('ConnectivitySMILES') or e.get('manual_smiles')
    syns = e.get('synonyms', [])
    common = pick_common_name(name, syns)
    lookup[name] = {
        'cid': e.get('cid'),
        'smiles': smiles,
        'iupac_name': p.get('IUPACName') or name,
        'common_name': common,
        'molecular_formula': p.get('MolecularFormula'),
        'molecular_weight': p.get('MolecularWeight'),
    }

missing = [n for n in all_names if n not in lookup or not lookup[n].get('smiles')]
print('Total names:', len(all_names), 'Resolved with SMILES:', len(all_names) - len(missing))
print('Missing/no-SMILES:', missing)

with open(r'C:\dev\des_analysis\component_lookup_full.json', 'w', encoding='utf-8') as f:
    json.dump(lookup, f, indent=2)
