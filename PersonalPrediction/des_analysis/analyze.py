import json
from rdkit import Chem

with open(r'C:\dev\des_analysis\raw_lists.json', encoding='utf-8') as f:
    data = json.load(f)
with open(r'C:\dev\des_analysis\pubchem_properties.json', encoding='utf-8') as f:
    props = json.load(f)

def inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)

# Build name -> (cid, smiles, inchikey, iupac, synonyms) for the 38 MP/Greenness component names
name_info = {}
for name, e in props.items():
    p = e['property_result']['PropertyTable']['Properties'][0]
    smiles = p.get('SMILES') or p.get('ConnectivitySMILES')
    name_info[name] = {
        'cid': e['cid'],
        'smiles': smiles,
        'iupac': p.get('IUPACName'),
        'inchikey': inchikey(smiles),
        'synonyms': e.get('synonyms', []),
    }

# Greenness sheet already gives smiles1/smiles2 directly; verify they match name_info via inchikey (sanity)
def key_for_name(name):
    return name_info[name]['cid']

def pairkey(c1, c2):
    k1, k2 = key_for_name(c1), key_for_name(c2)
    return frozenset([k1, k2])

# --- Metal binding top25: map ligand smiles -> cid via inchikey match against name_info, else standalone ---
mb_entries = data['MetalBinding']
mb_inchikey_to_entry = {}
for r in mb_entries:
    ik = inchikey(r['smiles'])
    r['inchikey'] = ik
    mb_inchikey_to_entry[ik] = r

inchikey_to_cid = {}
for name, info in name_info.items():
    if info['inchikey']:
        inchikey_to_cid[info['inchikey']] = info['cid']

mb_cids = set()
mb_cid_to_rank = {}
for r in mb_entries:
    ik = r['inchikey']
    cid = inchikey_to_cid.get(ik)
    r['matched_cid'] = cid
    if cid:
        mb_cids.add(cid)
        mb_cid_to_rank[cid] = r['rank']

print("Metal-binding top25 ligands that match an MP/Greenness component CID:")
for r in mb_entries:
    if r['matched_cid']:
        print(f"  rank {r['rank']:>2}  {r['name']}  (cid {r['matched_cid']})")

# --- Build pair occurrence records per list ---
pair_records = {}  # pairkey -> dict with lists of occurrences

def add_occurrence(pk, listname, rec):
    d = pair_records.setdefault(pk, {'occurrences': {}, 'names': None})
    d['occurrences'].setdefault(listname, []).append(rec)

for r in data['MP_RDK_RT']:
    pk = pairkey(r['c1'], r['c2'])
    add_occurrence(pk, 'MP_RDK_RT', r)
    pair_records[pk]['names'] = (r['c1'], r['c2'])

for r in data['MP_RDK_WT']:
    pk = pairkey(r['c1'], r['c2'])
    add_occurrence(pk, 'MP_RDK_WT', r)
    if pair_records[pk]['names'] is None:
        pair_records[pk]['names'] = (r['c1'], r['c2'])

for r in data['Greenness']:
    pk = pairkey(r['c1'], r['c2'])
    add_occurrence(pk, 'Greenness', r)
    if pair_records[pk]['names'] is None:
        pair_records[pk]['names'] = (r['c1'], r['c2'])

# --- Scoring ---
WEIGHTS = {'MP_RDK_RT': 2, 'MP_RDK_WT': 1, 'Greenness': 1, 'MetalBinding': 1}

results = []
for pk, d in pair_records.items():
    occ = d['occurrences']
    score = 0
    breadth = 0
    detail = {}
    for listname in ['MP_RDK_RT', 'MP_RDK_WT', 'Greenness']:
        cnt = len(occ.get(listname, []))
        if cnt > 0:
            score += WEIGHTS[listname] * cnt
            breadth += 1
        detail[listname] = cnt
    # metal binding: does either component CID appear in mb_cids?
    cids = list(pk)
    mb_hits = [c for c in cids if c in mb_cids]
    if mb_hits:
        score += WEIGHTS['MetalBinding'] * len(mb_hits)
        breadth += 1
    detail['MetalBinding_component_hits'] = [(c, mb_cid_to_rank.get(c)) for c in mb_hits]
    results.append({
        'pairkey': pk,
        'names': d['names'],
        'score': score,
        'breadth': breadth,
        'detail': detail,
        'total_raw_occurrences': sum(detail[k] for k in ['MP_RDK_RT','MP_RDK_WT','Greenness']),
    })

results.sort(key=lambda x: (-x['score'], -x['breadth'], -x['total_raw_occurrences']))

print("\n=== RANKED CANDIDATES ===")
for i, r in enumerate(results[:15], start=1):
    cid1, cid2 = list(r['pairkey'])
    print(f"{i}. score={r['score']} breadth={r['breadth']} raw_occ={r['total_raw_occurrences']} names={r['names']} detail={r['detail']}")

with open(r'C:\dev\des_analysis\name_info.json', 'w', encoding='utf-8') as f:
    json.dump(name_info, f, indent=2)

with open(r'C:\dev\des_analysis\metal_binding_matched.json', 'w', encoding='utf-8') as f:
    json.dump(mb_entries, f, indent=2, default=str)

# Save full ranked results (cids -> names via name_info reverse lookup)
cid_to_name = {}
for name, info in name_info.items():
    cid_to_name.setdefault(info['cid'], []).append(name)

serializable_results = []
for r in results:
    serializable_results.append({
        'cids': list(r['pairkey']),
        'names': r['names'],
        'score': r['score'],
        'breadth': r['breadth'],
        'detail': r['detail'],
        'total_raw_occurrences': r['total_raw_occurrences'],
    })
with open(r'C:\dev\des_analysis\ranked_results.json', 'w', encoding='utf-8') as f:
    json.dump(serializable_results, f, indent=2, default=str)

print("\nTotal unique pairs across MP_RDK_RT/WT + Greenness:", len(results))
