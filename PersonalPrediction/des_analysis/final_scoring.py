# -*- coding: utf-8 -*-
import csv, json

with open(r'C:\dev\des_analysis\component_lookup_full.json', encoding='utf-8') as f:
    comp_lookup = json.load(f)
with open(r'C:\dev\des_analysis\matched_component_names.json', encoding='utf-8') as f:
    matched_names = json.load(f)  # name -> metal-binding ligand record

with open(r'C:\dev\PersonalPrediction\Simple ML Models\Optimized Model\CSVs\Candidate_Master_List_rdkit.csv', encoding='utf-8') as f:
    mp_rows = list(csv.DictReader(f))

def to_bool(s):
    return str(s).strip().lower() == 'true'

def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

# --- Greenness coverage (only 25 specific pairs have a greenness score) ---
green_by_pair = {}
with open(r'C:\dev\PersonalPrediction\Greenness ML Models\Final Model\results\top25_DES_candidates.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        key = frozenset([row['Component#1'], row['Component#2']])
        green_by_pair[key] = row

candidates = []
for row in mp_rows:
    c1, c2 = row['Component#1'], row['Component#2']
    passes_rule = to_bool(row['passes_DES_rule'])
    room = to_bool(row['RoomTemp_flag'])
    warm = to_bool(row['WarmTemp_flag'])
    if not passes_rule or not (room or warm):
        continue  # hard filter: must be a physically valid, liquid-at-feasible-temperature DES

    m1 = matched_names.get(c1)
    m2 = matched_names.get(c2)
    if m1 is None and m2 is None:
        continue  # no metal-binding signal at all for either component -> not eligible given the stated emphasis

    logk1 = m1['logK_mean_both'] if m1 else None
    logk2 = m2['logK_mean_both'] if m2 else None
    low_conf1 = m1['low_confidence'] if m1 else False
    low_conf2 = m2['low_confidence'] if m2 else False

    both_matched = m1 is not None and m2 is not None
    if both_matched:
        metal_score = logk1 + logk2  # reward pairs where both components contribute binding capacity
    else:
        metal_score = logk1 if logk1 is not None else logk2

    pair_key = frozenset([c1, c2])
    green = green_by_pair.get(pair_key)

    candidates.append({
        'c1': c1, 'c2': c2,
        'x1': row['X#1 (molar fraction)'], 'x2': row['X#2 (molar fraction)'],
        'predicted_Tmelt_K': to_float(row['predicted_Tmelt_K']),
        'uncertainty_K': to_float(row['uncertainty_K']),
        'similarity_score': to_float(row['similarity_score']),
        'RoomTemp_flag': room, 'WarmTemp_flag': warm,
        'ranking_score_room': to_float(row['ranking_score_room']),
        'ranking_score_warm': to_float(row['ranking_score_warm']),
        'metal_match_c1': m1, 'metal_match_c2': m2,
        'both_matched': both_matched,
        'metal_score': metal_score,
        'low_confidence_any': low_conf1 or low_conf2,
        'greenness_DES_score': float(green['G-score_DES']) if green else None,
    })

print('Eligible candidates (pass DES rule, liquid at room or warm temp, >=1 metal-binding match):', len(candidates))
both = [c for c in candidates if c['both_matched']]
print('Of which both-components-matched:', len(both))

candidates.sort(key=lambda c: (-c['metal_score'], c['low_confidence_any'], -c['similarity_score'] if c['similarity_score'] else 0))

with open(r'C:\dev\des_analysis\final_candidates_ranked.json', 'w', encoding='utf-8') as f:
    json.dump(candidates, f, indent=2, default=str)

print('\n=== TOP 30 by metal_score ===')
for i, c in enumerate(candidates[:30], start=1):
    print(i, round(c['metal_score'], 3), '|', c['c1'], '+', c['c2'], '| both=', c['both_matched'],
          '| Tmelt=', round(c['predicted_Tmelt_K'], 1) if c['predicted_Tmelt_K'] else None,
          '| Room=', c['RoomTemp_flag'], 'Warm=', c['WarmTemp_flag'],
          '| low_conf=', c['low_confidence_any'], '| green=', c['greenness_DES_score'])
