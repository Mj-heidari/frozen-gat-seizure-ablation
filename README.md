# Frozen graph attention for patient-independent seizure window detection

A reproducible study of whether two hand-designed attention priors — a per-node
amplitude reliability score and a within-window correlation stability score —
improve seizure window classification on unseen patients.

The encoder uses frozen random graph attention: the projections are drawn once
and never trained, and the only supervised parameters are the 32 of a logistic
readout. The restriction is deliberate. A model this small cannot compensate
for a weak prior by learning a better representation, so a difference between
two variants is attributable to the priors rather than to a better-trained
encoder.

Five variants share the montage, feature definition, patient split, readout
family and corruption seeds, and differ only in which prior terms are active:
a spectral readout with no graph computation, plain attention, quality-only
attention, the full mechanism, and an augmented version.

## Findings

Cohort: 23 patients, 41 recordings, 36,137 completed four-second windows
(2.60 % ictal), patient-disjoint stratified five-fold validation. The decision
threshold and the prior strengths are selected by inner grouped cross-validation
within each training fold; no test-fold information enters any choice.

- **The priors do not help.** The pre-specified contrast — balanced accuracy
  under four-node dropout, full mechanism minus plain attention — is −0.002,
  with a 95 % patient-bootstrap interval of [−0.014, +0.009]. The interval
  covers zero on all four corruption conditions.
- Inner selection set the edge-stability weight to zero in three of five folds.
- The graph encoder improves ranking over pooled spectral features (AUC 0.662
  versus 0.644 under dropout) but not balanced accuracy at the selected
  operating point (0.622 versus 0.645).
- Removing four of eighteen derivations moves balanced accuracy by −0.000 to
  −0.006, so this corruption model is too mild to stress any mechanism.

## Data contracts

Preparation halts rather than degrading silently. On the conversion used here
it caught three problems that would otherwise have passed unnoticed:

1. **Repeated patient identity.** The individual distributed as two cases is
   one person; the conversion had already merged both sessions under a single
   identifier, which the file counts confirm.
2. **Mixed montage nomenclature.** Two recordings use legacy 10–20 electrode
   names where the rest use 10–10. An explicit alias table renames identical
   physical derivations; nothing is synthesised. A pipeline matching channels
   positionally would have routed one patient's temporal derivations into the
   wrong graph nodes with no error raised.
3. **Event schema.** The distribution annotates seizures under a different
   column and value than the dataset documentation implies. The accepted event
   count matches the published total.

Windows straddling a seizure boundary are discarded, not relabelled.

## Requirements

Python 3.11 or newer with NumPy, SciPy and scikit-learn; MNE is required for
reading EDF, BDF, SET and FIF input.

```bash
pip install -e ".[io]"
python -m unittest discover -s tests -v
```

The test suite covers the scientific failure modes: repeated patient identity,
fold disjointness, boundary-window exclusion, window-local preprocessing,
immutability of training moments at inference, deterministic corruption,
explicit missing-channel behaviour, unit handling and cache integrity.

## Usage

Set `root` in `configs/chb_mit.json` to a local CHB-MIT-BIDS
conversion, then:

```bash
python -m eegstudy.cli inventory --config configs/chb_mit.json --out inventory_full
python propose_manifest_v3.py --config configs/chb_mit.json \
    --inventory inventory_full/manifest.csv --patients 23 --per-patient 2 --max-seconds 3600
# review local/manifest.proposed.csv, then copy it to local/manifest.csv
python dump_headers2.py --config configs/chb_mit.json
python -m eegstudy.cli prepare --config configs/chb_mit.json
python sweep.py --config configs/chb_mit.json --out runs/sweep_001
```

`sweep.py` is the tuned analysis: it caches spectral and correlation features
once per corruption condition rather than recomputing them per fold and
variant, then selects the prior strengths, the readout regularisation and the
decision threshold by inner three-fold grouped cross-validation. Expect several
hours on a workstation.

`python -m eegstudy.cli run` performs the untuned analysis at a fixed 0.5
decision threshold. Both are reported, because at 2.60 % prevalence the choice
of operating point changes the conclusion.

## Data availability

Recordings come from the CHB-MIT Scalp EEG Database on PhysioNet
(doi:10.13026/C2K01R) and are not redistributed here. Generated caches, runs,
checkpoints and predictions are excluded from version control.
