# -*- coding: utf-8 -*-
import json, time, urllib.parse, urllib.request

with open(r'C:\dev\des_analysis\new_names_to_resolve.json', encoding='utf-8') as f:
    names = json.load(f)

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

fails = []
for idx, name in enumerate(names):
    if name in results:
        continue
    enc = urllib.parse.quote(name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{enc}/property/IUPACName,CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight/JSON"
    j = fetch_json(url)
    entry = {'query_name': name, 'lookup_method': 'name', 'property_result': j}
    cid = None
    try:
        cid = j['PropertyTable']['Properties'][0]['CID']
    except Exception:
        cid = None
    entry['cid'] = cid
    if not cid:
        fails.append(name)
    results[name] = entry
    if idx % 20 == 0:
        print(idx, '/', len(names), 'done; fails so far:', len(fails))
    time.sleep(0.25)

with open(r'C:\dev\des_analysis\pubchem_properties.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print('TOTAL FAILS:', len(fails))
with open(r'C:\dev\des_analysis\full_mp_resolve_fails.json', 'w', encoding='utf-8') as f:
    json.dump(fails, f, indent=2)
for fn in fails:
    print(' FAIL:', fn)
