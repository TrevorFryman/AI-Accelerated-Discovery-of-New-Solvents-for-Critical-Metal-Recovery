# -*- coding: utf-8 -*-
import json

with open(r'C:\dev\des_analysis\pubchem_properties.json', encoding='utf-8') as f:
    props = json.load(f)

# Manually curated, readable common/trivial name per systematic name key
COMMON_NAME_OVERRIDE = {
    "(1R,2S,5R)-5-methyl-2-propan-2-ylcyclohexan-1-ol": "L-Menthol",
    "(2R,3R,4R,5S)-hexane-1,2,3,4,5,6-hexol": "D-Sorbitol",
    "(2R,3R,4S,5S,6R)-2-[(2S,3S,4S,5R)-3,4-dihydroxy-2,5-bis(hydroxymethyl)oxolan-2-yl]oxy-6-(hydroxymethyl)oxane-3,4,5-triol": "Sucrose",
    "(3R,4S,5S)-oxane-2,3,4,5-tetrol": "L-Arabinose",
    "(3R,4S,5S,6R)-6-(hydroxymethyl)oxane-2,3,4,5-tetrol": "D-Glucose",
    "(3S,4R,5R)-2-(hydroxymethyl)oxane-2,3,4,5-tetrol": "D-Fructose",
    "2,3-dimethylphenol": "2,3-Xylenol",
    "2-(trimethylazaniumyl)acetate": "Betaine (Glycine betaine)",
    "2-hydroxyacetic acid": "Glycolic acid",
    "2-hydroxyethyl(trimethyl)azanium;chloride": "Choline chloride",
    "2-hydroxypropanoic acid": "Lactic acid",
    "2-methylphenol": "o-Cresol",
    "3-cyclohexylpropanoic acid": "Cyclohexanepropionic acid",
    "3-phenylpropanoic acid": "Hydrocinnamic acid",
    "5-methyl-2-propan-2-ylphenol": "Thymol",
    "Capric Acid": "Capric acid",
    "Caprylic Acid": "Caprylic acid",
    "DL-camphor": "Camphor",
    "Glutaric Acid": "Glutaric acid",
    "Lidocaine": "Lidocaine",
    "Pimelic Acid": "Pimelic acid",
    "[N2222]Br": "Tetraethylammonium bromide",
    "[N3333]Br": "Tetrapropylammonium bromide",
    "decanoic acid": "Capric acid",
    "dodecanoic acid": "Lauric acid",
    "hexadecan-1-ol": "Cetyl alcohol",
    "hexadecanoic acid": "Palmitic acid",
    "octanoic acid": "Caprylic acid",
    "pentanedioic acid": "Glutaric acid",
    "phenol": "Phenol",
    "propanedioic acid": "Malonic acid",
    "tetrabutylazanium;chloride": "Tetrabutylammonium chloride (TBAC)",
    "tetradecanoic acid": "Myristic acid",
    "tetraethylazanium;chloride": "Tetraethylammonium chloride (TEAC)",
    "tetramethylazanium;chloride": "Tetramethylammonium chloride",
    "tetrapropylazanium;chloride": "Tetrapropylammonium chloride",
    "undec-10-enoic acid": "Undecylenic acid",
    "urea": "Urea",
    "(Z)-octadec-9-enoic acid": "Oleic acid",
    "1,2,3,4-tetrafluoro-5,6-diiodobenzene": "1,2,3,4-Tetrafluoro-5,6-diiodobenzene",
    "1,3-dithiane": "1,3-Dithiane",
    "1-ethyl-3-methylimidazol-3-ium;chloride": "1-Ethyl-3-methylimidazolium chloride ([EMIM]Cl)",
    "N-(4-hydroxyphenyl)acetamide": "Acetaminophen (Paracetamol)",
    "benzyl-(2-hydroxyethyl)-dimethylazanium;chloride": "Benzyl(2-hydroxyethyl)dimethylammonium chloride",
    "cyclohexanecarboxylic acid": "Hexahydrobenzoic acid",
    "octadecan-1-ol": "Stearyl alcohol",
    "tetradecan-1-ol": "Myristyl alcohol",
    "trioctylphosphine oxide": "Trioctylphosphine oxide (TOPO)",
}

lookup = {}
for name, e in props.items():
    p = e['property_result']['PropertyTable']['Properties'][0]
    smiles = p.get('SMILES') or p.get('ConnectivitySMILES')
    lookup[name] = {
        'cid': e['cid'],
        'smiles': smiles,
        'iupac_name': p.get('IUPACName'),
        'common_name': COMMON_NAME_OVERRIDE.get(name, name),
        'molecular_formula': p.get('MolecularFormula'),
        'molecular_weight': p.get('MolecularWeight'),
    }

with open(r'C:\dev\des_analysis\component_lookup.json', 'w', encoding='utf-8') as f:
    json.dump(lookup, f, indent=2)

print(len(lookup), "components in lookup")

# Metal-binding ligand common names
with open(r'C:\dev\des_analysis\metalbinding_synonyms.json', encoding='utf-8') as f:
    mb_syn = json.load(f)

MB_COMMON_OVERRIDE = {
    1: "Tetracycline", 2: "Caffeic acid", 3: "1,10-Phenanthroline", 4: "Salicylic acid",
    5: "Aspartic acid", 6: "Phenyl salicylate (Salol)", 7: "Hydroquinone", 8: "Gallic acid",
    9: "Citric acid", 10: "Threonine", 11: "Glutamic acid", 12: "Serine",
    13: "Methionine", 14: "Aspirin (Acetylsalicylic acid)", 15: "Arginine", 16: "Tetracaine",
    17: "Malic acid", 18: "Tartaric acid", 19: "Oxalic acid", 20: "Adiphenine",
    21: "Thymol", 22: "Sulfathiazole", 23: "Lidocaine", 24: "Chlorothymol",
    25: "Acetaminophen (Paracetamol)",
}
mb_lookup = {}
for rank_str, e in mb_syn.items():
    rank = int(rank_str)
    mb_lookup[rank] = {
        'systematic_name': e['name'],
        'smiles': e['smiles'],
        'cid': e['cid'],
        'common_name': MB_COMMON_OVERRIDE[rank],
    }
with open(r'C:\dev\des_analysis\metalbinding_lookup.json', 'w', encoding='utf-8') as f:
    json.dump(mb_lookup, f, indent=2)
print(len(mb_lookup), "metal-binding ligands in lookup")
