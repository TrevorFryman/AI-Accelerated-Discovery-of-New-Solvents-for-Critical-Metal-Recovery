import json, time, urllib.parse, urllib.request

with open(r'C:\dev\des_analysis\unique_names.json', encoding='utf-8') as f:
    names = json.load(f)

# manual overrides for known abbreviation/formula-style names PubChem name search won't resolve directly
manual_smiles = {
    '[N2222]Br': '[Br-].CC[N+](CC)(CC)CC',       # tetraethylammonium bromide
    '[N3333]Br': '[Br-].CCC[N+](CCC)(CCC)CCC',   # tetrapropylammonium bromide
}

def fetch_json(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if i == retries - 1:
                return {'error': str(e)}
            time.sleep(1)

results = {}
for name in names:
    enc = urllib.parse.quote(name)
    entry = {'query_name': name}
    if name in manual_smiles:
        smi = manual_smiles[name]
        enc2 = urllib.parse.quote(smi)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{enc2}/property/IUPACName,CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight/JSON"
        j = fetch_json(url)
        entry['lookup_method'] = 'manual_smiles'
        entry['manual_smiles'] = smi
        entry['property_result'] = j
        try:
            entry['cid'] = j['PropertyTable']['Properties'][0]['CID']
        except Exception:
            entry['cid'] = None
    else:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{enc}/property/IUPACName,CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight/JSON"
        j = fetch_json(url)
        entry['lookup_method'] = 'name'
        entry['property_result'] = j
        try:
            entry['cid'] = j['PropertyTable']['Properties'][0]['CID']
        except Exception:
            entry['cid'] = None
    results[name] = entry
    time.sleep(0.25)

with open(r'C:\dev\des_analysis\pubchem_properties.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

# report failures
fails = [n for n, e in results.items() if not e.get('cid')]
print("Total:", len(names), "Failed:", len(fails))
for f_ in fails:
    print(" FAIL:", f_, results[f_]['property_result'])
