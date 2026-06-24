# -*- coding: utf-8 -*-
import json, time, urllib.request

with open(r'C:\dev\des_analysis\new_names_to_resolve.json', encoding='utf-8') as f:
    new_names = json.load(f)
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

count = 0
for name in new_names:
    entry = results.get(name)
    if entry is None or 'synonyms' in entry:
        continue
    cid = entry.get('cid')
    if not cid:
        entry['synonyms'] = []
        continue
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
    j = fetch_json(url)
    try:
        syns = j['InformationList']['Information'][0]['Synonym'][:15]
    except Exception:
        syns = []
    entry['synonyms'] = syns
    count += 1
    if count % 20 == 0:
        print(count, 'synonyms fetched...')
    time.sleep(0.25)

with open(r'C:\dev\des_analysis\pubchem_properties.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print('done, fetched', count)
