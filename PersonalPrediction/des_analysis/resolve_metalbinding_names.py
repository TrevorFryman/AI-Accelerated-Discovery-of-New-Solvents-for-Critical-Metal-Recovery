import json, time, urllib.parse, urllib.request

with open(r'C:\dev\des_analysis\raw_lists.json', encoding='utf-8') as f:
    data = json.load(f)
mb = data['MetalBinding']

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

out = {}
for r in mb:
    smi = r['smiles']
    enc = urllib.parse.quote(smi, safe='')
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{enc}/cids/JSON"
    j = fetch_json(url)
    cid = None
    try:
        cid = j['IdentifierList']['CID'][0]
    except Exception:
        pass
    syns = []
    if cid:
        url2 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
        j2 = fetch_json(url2)
        try:
            syns = j2['InformationList']['Information'][0]['Synonym'][:15]
        except Exception:
            syns = []
    out[r['rank']] = {'name': r['name'], 'smiles': smi, 'cid': cid, 'synonyms': syns}
    print(r['rank'], r['name'], '->', cid, syns[:4])
    time.sleep(0.3)

with open(r'C:\dev\des_analysis\metalbinding_synonyms.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
