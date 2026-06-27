# Tokenizer

MEG-Mamba operates on discrete tokens produced by [EphysTokenizer](https://github.com/OHBA-analysis/EphysTokenizer). We use the `causal` variant. See the [paper](https://arxiv.org/abs/2602.16626) and the [repo](https://github.com/OHBA-analysis/EphysTokenizer).

The bundled tokenizer was trained on [Cam-CAN](https://cam-can.mrc-cbu.cam.ac.uk/dataset/) resting-state + passive/smt task.

```
tokenizer/
  model/
    config.yaml              # tokenizer config (causal variant, schaefer100_camcan_all)
    model_state.pt           # trained weights
    vocab.pkl                # codebook + the compacted-id refactoring (label_map)
```

It maps a [Schaefer100](https://osl-dynamics.readthedocs.io/en/latest/parcellations/schaefer100.html) parcellated recording at 250 Hz (shape `(T, 100)`) to/from integer tokens.

## Preparing your own data

To run the model on your own MEG you must reproduce the pipeline that built the training corpus: raw → preprocess → source reconstruct → parcellate → tokenize.

### 1. Preprocess, source reconstruct and parcellate

Preprocess, source reconstruct (beamform), and parcellate your raw MEG with [osl-dynamics](https://github.com/OHBA-analysis/osl-dynamics) following this [tutorial](https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/0-2_meg_batch_processing.html). The settings that must match the corpus:

| Setting | Value |
|---|---|
| Parcellation | Schaefer100: `atlas-Schaefer_nparc-100_space-MNI_res-8x8x8.nii.gz` |
| Sampling frequency | 250 Hz |
| Source reconstruction | Volumetric LCMV beamformer (mag + grad; unit-noise-gain) |

**No orthogonalization or sign flipping** is needed.

### 2. Tokenize

Load the bundled tokenizer with [EphysTokenizer](https://github.com/OHBA-analysis/EphysTokenizer) (use the `etkn` environment) and tokenize the parcel data:

```python
import mne
from ephys_tokenizer.models.ephys_tokenizer import EphysTokenizerModule

parc   = mne.io.read_raw_fif("lcmv-parc-raw.fif", preload=True)
signal = parc.get_data(picks="misc", reject_by_annotation="omit").T   # (T, 100) @ 250 Hz

model  = EphysTokenizerModule.load_model("tokenizer/model")   # the bundled tokenizer
tokens = model.tokenize_session(signal)      # (T, 100) float signal -> (T, 100) uint8 tokens
```

Also see: [examples/tokenize_etkn.py](https://github.com/OHBA-analysis/EphysTokenizer/blob/main/examples/tokenize_etkn.py).

## Detokenization

`reconstruct_session` inverts the tokenizer, turning tokens back into a continuous parcel signal:

```python
recon = model.reconstruct_session(tokens)    # tokens -> reconstructed parcel signal
```
