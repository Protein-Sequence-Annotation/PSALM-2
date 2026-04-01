# PSALM

PSALM predicts Pfam-style domain annotations on protein sequences using a language model. This document covers **inference** (running scans) and **training** (data prep and model training).

**Table of contents**

- [Quick start](#quick-start)
- [Installation](#installation)
- [CLI reference (inference)](#cli-reference-inference)
- [Python API](#python-api)
- [Citing PSALM](#citing-psalm)
- [Training and advanced usage](#training-and-advanced-usage)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                                                                              │
│                 ██████╗ ███████╗ █████╗ ██╗     ███╗   ███╗                  │
│                 ██╔══██╗██╔════╝██╔══██╗██║     ████╗ ████║                  │
│                 ██████╔╝███████╗███████║██║     ██╔████╔██║                  │
│                 ██╔═══╝ ╚════██║██╔══██║██║     ██║╚██╔╝██║                  │
│                 ██║     ███████║██║  ██║███████╗██║ ╚═╝ ██║                  │
│                 ╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝                  │
│              Protein Sequence Annotation using a Language Model              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Quick start

1. Create a Python 3.10 environment and upgrade `pip`.
2. Install [PyTorch](https://pytorch.org/get-started/locally/) for your hardware (CPU, CUDA, or Apple Silicon).
3. Install PSALM from [PyPI](https://pypi.org/project/protein-sequence-annotation/) (pin the version you want, e.g. `2.1.12`):

```bash
python -m pip install protein-sequence-annotation==2.1.12
```

4. Run a scan on a FASTA file:

```bash
psalm-scan -f path/to/your_sequence.fasta
```

`psalm-scan` loads the model, prints startup/status (without the ASCII banner frame), runs one scan, then exits.

For repeated scans in one process, use the interactive shell. It shows the banner once, loads the model once, then accepts `scan` commands:

```bash
psalm -d auto
# inside the shell:
#   scan -f path/to/seqs.fa
#   scan --sort -f path/to/seqs.fa --to-tsv hits.tsv
#   scan -s "MSTNPKPQR..."
#   quit
```

Use `psalm` when you want many scans in a session; use `psalm-scan` for a single invocation from scripts or batch jobs.

## Installation

Create a fresh Python 3.10 environment, install PyTorch for your hardware, then install PSALM.

```bash
conda create -n psalm python=3.10 -y
conda activate psalm
python -m pip install --upgrade pip

# 1) Install PyTorch for your hardware
# Apple Silicon (MPS):
python -m pip install torch

# CPU-only (Linux/Windows):
# python -m pip install torch

# NVIDIA CUDA 12.1:
# python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
#   torch

# 2) Install PSALM
python -m pip install protein-sequence-annotation==2.1.12
```

If you are unsure which PyTorch command matches your GPU/driver, use the official selector: https://pytorch.org/get-started/locally/

**Intel Mac (x86_64)** — path that has been tested on that platform:

```bash
conda create -n psalm python=3.10 -y
conda activate psalm

conda install -y -c conda-forge "llvmlite=0.44.*" "numba=0.61.*"
conda install -y -c conda-forge "pytorch=2.5" torchvision torchaudio

python -m pip install protein-sequence-annotation==2.1.12
```

Run without activating the environment manually:

```bash
conda run -n psalm psalm-scan -f path/to/seqs.fa
```

## CLI reference (inference)

Defaults:

- Default model: `ProteinSequenceAnnotation/PSALM-2`
- Default device: `auto` (`cuda` → `mps` → `cpu`)
- `-T`: keep domains with `Score >= threshold` (default: `0.5`)
- `-E`: keep domains with `E-value <= threshold` (default: `0.1`)
- `-Z`: dataset size for E-value scaling; if omitted for `-s`, `Z=1`; if omitted for `-f`, `Z` = number of sequences in the FASTA

FASTA and fast mode:

- FASTA scans use fast batched scanning by default
- `--serial`: legacy serial FASTA path
- `--sort`: sort FASTA sequences longest-first before fast-mode batching (fast FASTA only)
- `-c` / `--cpu-workers`: number of fast-mode CPU decode helper processes; default behavior matches `-c 0`; if the interactive shell already warmed workers, later default fast scans can reuse that pool
- `--max-batch-size`: fast-mode embedding batch budget (tokens/amino acids)
- `--max-queue-size`: fast-mode decode queue size in sequences (default: `128`)

Output:

- `-q` / `--quiet`: suppress scan result text in the terminal; startup/status still print; multi-sequence FASTA still shows a progress bar
- `--to-tsv` and `--to-txt`: single- or multi-sequence FASTA; `--to-tsv` is the supported machine-readable format
- `-v` / `--verbose`: detailed alignment and model tables; verbose FASTA scans use the serial path; without `-v`, output is the compact HITS report

Help:

```text
psalm --help
psalm-scan --help
# In the shell:
scan --help
```

### Interactive shell (`psalm`) — common patterns

```bash
psalm
scan --sort -f path/to/seqs.fa --to-tsv hits.tsv
```

```bash
# compact terminal report + TSV
scan -f path/to/seqs.fa --to-tsv hits.tsv

# TSV only (quiet)
scan -q --sort -f path/to/seqs.fa --to-tsv hits.tsv

# verbose per-domain output
scan -v -f path/to/seqs.fa
```

Fast shell with workers pre-warmed at startup:

```bash
psalm -c 4
# then:
scan --sort -f path/to/seqs.fa --to-tsv hits.tsv
```

Fast shell without pre-warming workers:

```bash
psalm -d auto
scan --sort -f path/to/seqs.fa -c 4 --to-tsv hits.tsv
```

## Python API

Defaults match the CLI where applicable.

```python
from psalm.psalm_model import PSALM

psalm = PSALM(model_name="ProteinSequenceAnnotation/PSALM-2")

# Scan FASTA
results = psalm.scan(fasta="path/to/your_sequence.fasta")
print(results)

# Scan sequence string
results = psalm.scan(sequence="MSTNPKPQR...AA")
```

Output options:

- `to_tsv="results.tsv"` writes: `Sequence,E-value,Score,Pfam,Start,Stop,Model,Len Frac,Status`
- `to_txt="results.txt"` saves console-style output
- For multi-sequence FASTA, TSV rows are combined with the query id in the `Sequence` column

## Citing PSALM

Sarkar A., Krishnan K., Eddy S.R. (2026). *Protein sequence domain annotation using a language model.* bioRxiv. https://doi.org/10.1101/2024.06.04.596712

Minimal BibTeX:

```bibtex
@article{SarkarKrishnanPSALM,
  author  = {Sarkar, Arpan and Krishnan, Kumaresh and Eddy, Sean R.},
  title   = {Protein sequence domain annotation using a language model},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.1101/2024.06.04.596712},
  url     = {https://doi.org/10.1101/2024.06.04.596712}
}
```

## Training and advanced usage

Most users only need [Quick start](#quick-start) and [CLI reference (inference)](#cli-reference-inference). The sections below are for building datasets, training models, and changing the CatBoost scorer— not required to run the published PSALM package on new sequences.

### Scripts overview

The core workflow is:

1. `scripts/data/augment_fasta.py` → slice sequences and generate augmented FASTA + domain dict
2. `scripts/data/data_processing.py` → tokenize, label, batch, and shard datasets
3. `scripts/train/train_psalm.py` → train/evaluate the PSALM model on shards

Optional InterPro-related steps (benchmarking / alternate ground truth):

4. `scripts/data/build_ipr_expanded_test.py` → build a global-consistent InterPro domain dict from `protein2ipr.dat`-style input
5. `scripts/test/evaluate_predictions.py` → score prediction pickles against InterPro-aware ground truth and optional ROC / negatives

#### `scripts/data/augment_fasta.py`

Splits long sequences into domain-preserving slices and optionally emits shuffled and negative variants. Produces a new FASTA and a new domain dict with aligned IDs.

**Key inputs**

- `--fasta`, `--domain-dict`
- `--output-fasta`, `--output-dict`

**Common flags**

- `--max-length`: slice length threshold
- `--negative-prob`: target fraction of negatives (approximate)
- `--include-domain-slices`, `--shuffle-only`, `--no-shuffle`, `--domain-slices-only`
- `--large-data` with `--p-shuffled`, `--domain-counts-tsv`, `--domain-slice-frac`
- `--seed`, `--verbose`

#### `scripts/data/data_processing.py`

Tokenizes sequences, generates per-token labels from the domain dict and label mapping, batches by token budget, and saves shards.

**Config handling**

- This script is CLI-only; it does not read `config.yaml`.

**Required args**

- `--fasta`, `--domain-dict`, `--output-dir`, `--ignore-label`
- `--model-name`, `--max-length`, `--max-tokens-per-batch`
- `--label-mapping-dict`

**Optional args**

- `--chunk-size`, `--tmp-dir`, `--shard-size`, `--seed`, `--keep-tmp`

**Notes**

- ID normalization uses the FASTA header segment between `>` and the first space.
- `--ignore-label` must match the training `--ignore-label`.

#### `scripts/train/train_psalm.py`

Trains or evaluates PSALM on preprocessed shard datasets.

**Config handling**

- Training always uses a YAML config.
- If `--config` is provided without a value, the script looks for `psalm/config.yaml`.
- If `--config` is not provided, the script still looks for `psalm/config.yaml`.

**Required args**

- `--val-dir`, `--ignore-label`
- `--train-dir` if `training.total_steps > 0` in config

**Optional args**

- `--label-mapping-dict` to override config `model.label_mapping_path`

**Checkpoint loading**

- Supports `model.safetensors` or `pytorch_model.bin` within a checkpoint directory, or a direct path to a `.safetensors`/`.bin` file.

**Logging**

- `report_to=["wandb"]` is enabled by default.

#### `scripts/train/train_cbm.py`

Trains the CatBoost scoring model used by `scan()` (saved as `score.cbm`).

**Required args**

- `--pos`, `--neg`: Pickle or JSON files containing a list of 7-tuples: `(pfam, start, stop, bit_score, len_ratio, bias, status)` (or `scan()` output dicts containing 8-tuples with `cbm_score`).

**Example**

```bash
python scripts/train/train_cbm.py \
  --pos path/to/positives.pkl \
  --neg path/to/negatives.pkl \
  --outdir cbm_outputs \
  --model-out score.cbm
```

#### `scripts/data/build_ipr_expanded_test.py`

Build InterPro-expanded ground truth: IPR map from `protein2ipr.dat` (or `.gz`), global pass-1 filter, `passing_iprs.txt`, then pass-2 placement consistent with global averages. Implementation: [`psalm/data/build_ipr_expanded_test.py`](psalm/data/build_ipr_expanded_test.py).

**Required args**

- `--dat-file`: `protein2ipr.dat`-style table (optional `.gz`)
- `--passing-iprs-out`: path for pass-1 TSV (`ipr_id` / global average length)
- `--domain-dict-out`: output pickle for the final domain dict

**Optional args**

- `--ipr-map-pkl`: precomputed IPR→members pickle (otherwise built from `--dat-file`)
- `--save-map-pkl`: where to save the built map when `--ipr-map-pkl` is omitted
- `--max-diff-frac` (default `0.10`), `--workers`, `--queue-size`, `--progress-every`, `--precision`, `--report-json`

**Example**

```bash
python scripts/data/build_ipr_expanded_test.py \
  --dat-file protein2ipr.dat \
  --passing-iprs-out passing_iprs.txt \
  --domain-dict-out ipr_domain_dict.pkl \
  --report-json ipr_build_report.json
```

#### `scripts/test/evaluate_predictions.py`

InterPro ID consensus evaluation on scored prediction pickles: length-bucket sensitivities, optional quantile ROC (`roc_by_threshold.pkl`), and optional merge of false positives from a negatives pickle directory (`roc_by_threshold_FULL.pkl`). Implementation: [`psalm/test/evaluate_predictions.py`](psalm/test/evaluate_predictions.py) (single module).

**Required args**

- `--groundtruth`: ground-truth pickle (`seq_id → domains`) or dict containing `domain_dict`
- `--preds-dir`: directory of prediction pickles
- `--fam-clan` / `--fam_clan`: PFAM→clan pickle
- `--interpro-map`: InterPro IPR→members pickle
- `--output`: output directory or prefix for summaries and ROC files

**Optional args**

- `--negatives`: directory of negative prediction pickles (FP merge)
- `--filter-score`, `--roc-n`, `--roc-seed`, `--progress-every`
- `--use-evalue`, `--only-preds`

**Example**

```bash
python scripts/test/evaluate_predictions.py \
  --groundtruth gt.pkl \
  --preds-dir preds/ \
  --fam-clan fam2clan.pkl \
  --interpro-map ipr_members.pkl \
  --output eval_out/
```

### Config format

The scripts expect a YAML config with these sections:

**`model`**

- `model_name`
- `max_batch_size`
- `output_size`
- `freeze_esm`
- `use_fa`
- `pretrained_checkpoint_path`
- `label_mapping_path`

**`training`**

- `gradient_accumulation_steps`, `learning_rate`, `optimizer`, `gradient_clipping`
- `lr_scheduler`, `eval_strategy`, `eval_steps`, `total_steps`, `warmup_steps`
- `logging_steps`, `save_steps`, `output_dir`
- `mixed_precision`, `dataloader_num_workers`, `dataloader_prefetch_factor`, `dataloader_pin_memory`, `seed`

**`data`**

- `chunk_size`, `default_tmp_dir`, `default_shard_size`

`psalm/config.yaml` is provided as a template with `null` values. Populate it before use, or pass all required values via CLI without `--config`.

### Training CLI examples

```bash
python scripts/data/augment_fasta.py \
  --fasta input.fa \
  --domain-dict domains.pkl \
  --output-fasta augmented.fa \
  --output-dict augmented.pkl
```

```bash
python scripts/data/data_processing.py \
  --fasta augmented.fa \
  --domain-dict augmented.pkl \
  --label-mapping-dict labels.pkl \
  --output-dir data/shards \
  --model-name facebook/esm2_t33_650M_UR50D \
  --max-length 4096 \
  --max-tokens-per-batch 8196 \
  --ignore-label -100
```

```bash
python scripts/train/train_psalm.py \
  --config psalm/config.yaml \
  --train-dir data/shards/train \
  --val-dir data/shards/val \
  --ignore-label -100
```

### Dependencies

- `PyYAML` is required for config loading.
- `faesm` is required only if `use_fa: true` in config.
- Core inference runtime uses `torch`, `transformers`, `biopython`, `pandas`, `numba`, and `catboost`.
