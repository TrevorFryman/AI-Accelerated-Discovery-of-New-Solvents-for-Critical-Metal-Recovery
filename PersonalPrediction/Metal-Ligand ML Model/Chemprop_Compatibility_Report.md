# Chemprop Compatibility Report

## 1. Version summary

| | Version | Source |
|---|---|---|
| **Chemprop used to train the shipped models** | **v1.5.2** | `stabilityconstant-ml-models` README + `args.json` "reproducibility" block (`time: Tue Jan 16 08:19:38 2024`) |
| **Chemprop available locally (`C:\dev\chemprop-main\chemprop-main`)** | **v2.2.3** | `chemprop/__init__.py`, `pyproject.toml` |
| **Chemprop importable in current Python env** | not installed | `pip show chemprop` → not found |
| **`lightning` (required by v2.x)** | not installed | `import lightning` fails |
| Current Python | 3.14.3 | too new for v1.5.2 (~3.7–3.9), compatible with v2.2.3 (`requires-python >=3.11,<3.15`) |
| `torch` | 2.12.0+cpu | satisfies v2.2.3's `torch>=2.1` |
| `rdkit` | 2026.03.2 | OK |
| `pandas` | 3.0.0 | OK |

**Compatibility status: INCOMPATIBLE without conversion.** Chemprop v1 and v2 are different codebases
(argparse + custom training loop vs. PyTorch Lightning). The shipped `model.pt` files are v1
`state_dict` + `TrainArgs` objects; v2's `MPNN.load_from_checkpoint` cannot read them directly.

Note: the README's claim that a copy of v1.5.2 "is located in the Chemprop directory of this
repository" is **not true of the extracted archive** — no v1 source, `environment.yml`, or pinned
requirements file is present anywhere under `stabilityconstant-ml-models-main`.

## 2. The conversion tool

`chemprop-main` ships `chemprop convert -c v1_to_v2 -i <model_v1.pt> -o <model_v2.pt>`
(`chemprop/cli/convert.py` → `chemprop/utils/v1_to_v2.py`).

Inspecting `convert_state_dict_v1_to_v2` / `convert_hyper_parameters_v1_to_v2`:

- For **single-molecule models** (`number_of_molecules == 1`, no `reaction_solvent`) — the **"else"
  branch**, taken cleanly with **no warning**. **This is exactly M2's `SMILES_only_model` case**
  (confirmed `"number_of_molecules": 1` in `args.json`).
- For **multi-molecule models** (M1/M3/M4 `best_model` variants, which use a second "solvent" encoder
  for auxiliary features) — the converter explicitly logs `"This conversion is untested - please
  validate your model predictions are consistent after conversion!"`.
- In all cases the converter logs a reminder: *"The default v1 atom featurizer is
  `MultiHotAtomFeaturizer.v1()` and can be specified from the command line with
  `--multi-hot-atom-featurizer-mode v1`."* — **this flag is mandatory** at predict time for converted
  checkpoints; omitting it silently changes the atom featurization and will produce wrong predictions.

## 3. Approach A — Modern environment (Chemprop v2.2.3 + conversion)

**Setup:** `pip install -e .` in `chemprop-main` (+ `lightning`), convert each of the 5 M2
`SMILES_only_model` fold checkpoints with `chemprop convert -c v1_to_v2`, then run
`chemprop predict --multi-hot-atom-featurizer-mode v1`.

| | Assessment |
|---|---|
| **Advantages** | Uses what's already downloaded; current Python (3.14) and torch (2.12) work; v2 is actively maintained, so the environment itself is reproducible/installable long-term; conversion path for M2's single-molecule case is the well-tested branch (no warning emitted). |
| **Risks** | Conversion is a re-implementation of the v1 forward pass in v2's module structure — even the "well-tested" path is a translation, not a guarantee of bit-identical numerics. Floating-point differences (op ordering, default precision, BatchNorm/dropout handling at inference, the `UnscaleTransform` vs. v1's manual unscaling) could shift predictions slightly. The `--multi-hot-atom-featurizer-mode v1` flag is easy to forget and the failure mode (wrong predictions, not an error) is silent. |
| **Reproducibility** | Reproducible going forward (current package versions, single environment for the whole project), but the *numerical match to the original paper's reported metrics* must be empirically verified, not assumed. |
| **Expected fidelity** | High for M2 SMILES-only (single-molecule, no extra encoders) — this is the case the Chemprop maintainers explicitly support. Lower/unknown for M1/M3/M4 `best_model` (untested multi-molecule conversion). |
| **Long-term maintainability** | Good — v2 is the actively developed line; one environment serves this and future Chemprop work. |

## 4. Approach B — Legacy environment (Chemprop v1.5.2)

**Setup:** New conda env with Python ~3.8–3.9, `pip install chemprop==1.5.2` (PyPI has this release),
run the README's exact `chemprop_predict --checkpoint_dir .../SMILES_only_model --preds_path ...`
command — no conversion, ensembles all 5 folds automatically.

| | Assessment |
|---|---|
| **Advantages** | Exactly matches the README's documented usage; zero conversion risk — checkpoints load with the same code/version that produced them; numerics should match the original paper's reported test-set metrics, giving a trustworthy sanity-check baseline. |
| **Risks** | Requires a second, separate Python environment (older Python + older torch/numpy pins from 2024) — more setup friction, and v1.5.2's dependency pins may themselves be hard to satisfy on a brand-new Windows machine (older `torch` wheel availability for Python 3.8/3.9 on Windows should be checked). |
| **Reproducibility** | Highest — this is literally "run the paper's code with the paper's models." |
| **Expected fidelity** | Highest — no translation layer between checkpoint and inference code. |
| **Long-term maintainability** | Lower — v1.5.2 and its pinned deps are unmaintained/frozen; a second isolated environment must be kept around alongside the main (v2) environment used for everything else. |

## 5. Recommendation

**Two-stage approach, both environments used for what they're best at:**

1. **Primary/scientific-reference run — Approach B (legacy v1.5.2 env).** Build a small, isolated
   Python 3.9 environment with `chemprop==1.5.2` and run the README's exact `chemprop_predict`
   command against M2's `SMILES_only_model`. Use this both for the actual DES-ligand predictions
   *and* as the ground truth to validate Approach A.
2. **Validation/cross-check — Approach A (v2.2.3 + conversion)**, restricted to M2's
   `SMILES_only_model` (the single-molecule, well-tested conversion case, with
   `--multi-hot-atom-featurizer-mode v1`). Run the repo's own `data/stability_constant_25C_model_M2/test_input.csv`
   (which has known experimental log K1 values) through both environments and confirm the predictions
   agree within a small tolerance before trusting either on new DES ligands.

If only one environment can be built, **Approach B is the more scientifically defensible default**
(no translation layer, matches the README verbatim, fidelity to the source paper is the priority for
"scientifically reliable" predictions). Approach A is acceptable as a fallback *only* for M2
`SMILES_only_model`, given its low-risk conversion path — it should **not** be used for the M1/M3/M4
`best_model` variants without explicit numerical validation against known test-set values, due to the
"untested" multi-molecule conversion warning.

In both cases, **M1/M3/M4 `best_model` variants are out of scope** for this project unless a
validated conversion (or a v1.5.2 run) reproduces their published test-set metrics first.

---
*Status: read-only assessment complete. Nothing installed, nothing converted, nothing executed.*
