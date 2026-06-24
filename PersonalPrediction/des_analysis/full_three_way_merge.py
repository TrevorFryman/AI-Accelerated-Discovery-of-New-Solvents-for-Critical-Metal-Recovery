# -*- coding: utf-8 -*-
"""
Build the complete three-way dataset across all 2,006 DES pairs:
melting-point feasibility (existing), metal-binding score (existing), and now greenness
(newly computed for all 205 components), then re-rank to see if the top 10 changes.
"""
import csv, json

with open(r'C:\dev\des_analysis\component_greenness_scores.json', encoding='utf-8') as f:
    gscores = json.load(f)
with open(r'C:\dev\des_analysis\matched_component_names.json', encoding='utf-8') as f:
    matched_names = json.load(f)
with open(r'C:\dev\des_analysis\component_lookup_full.json', encoding='utf-8') as f:
    comp_lookup = json.load(f)
with open(r'C:\dev\des_analysis\name_to_inchikey.json', encoding='utf-8') as f:
    name_to_ik = json.load(f)

with open(r'C:\dev\PersonalPrediction\Simple ML Models\Optimized Model\CSVs\Candidate_Master_List_rdkit.csv', encoding='utf-8') as f:
    mp_rows = list(csv.DictReader(f))

def to_bool(s):
    return str(s).strip().lower() == 'true'

def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

candidates = []
for row in mp_rows:
    c1, c2 = row['Component#1'], row['Component#2']
    passes_rule = to_bool(row['passes_DES_rule'])
    room = to_bool(row['RoomTemp_flag'])
    warm = to_bool(row['WarmTemp_flag'])
    if not passes_rule or not (room or warm):
        continue

    x1 = to_float(row['X#1 (molar fraction)']) or 0.5
    x2 = to_float(row['X#2 (molar fraction)']) or 0.5
    g1 = gscores.get(c1)
    g2 = gscores.get(c2)
    g_des = (x1 * g1 + x2 * g2) if (g1 is not None and g2 is not None) else None

    m1 = matched_names.get(c1)
    m2 = matched_names.get(c2)
    if m1 is None and m2 is None:
        metal_score = None
        both_matched = False
    else:
        logk1 = m1['logK_mean_both'] if m1 else None
        logk2 = m2['logK_mean_both'] if m2 else None
        both_matched = m1 is not None and m2 is not None
        metal_score = (logk1 + logk2) if both_matched else (logk1 if logk1 is not None else logk2)

    low_conf = (m1['low_confidence'] if m1 else False) or (m2['low_confidence'] if m2 else False)

    candidates.append({
        'c1': c1, 'c2': c2,
        'inchikey_pair': frozenset([name_to_ik.get(c1), name_to_ik.get(c2)]),
        'x1': x1, 'x2': x2,
        'predicted_Tmelt_K': to_float(row['predicted_Tmelt_K']),
        'uncertainty_K': to_float(row['uncertainty_K']),
        'similarity_score': to_float(row['similarity_score']),
        'RoomTemp_flag': room, 'WarmTemp_flag': warm,
        'metal_score': metal_score, 'both_matched': both_matched, 'low_confidence_any': low_conf,
        'g_score_des': g_des, 'g1': g1, 'g2': g2,
    })

print('Total MP-feasible candidates (pass DES rule, liquid at room/warm):', len(candidates))
have_metal = [c for c in candidates if c['metal_score'] is not None]
print('Of which have >=1 metal-binding match:', len(have_metal))
have_both = [c for c in have_metal if c['both_matched']]
print('Of which have BOTH components metal-matched:', len(have_both))
print('All candidates have a greenness score now (every component resolved):',
      all(c['g_score_des'] is not None for c in candidates))

with open(r'C:\dev\des_analysis\full_three_way_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(candidates, f, indent=2, default=str)
