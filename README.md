# PSALM-2 

## ESM2-FA-Training: A Framework for Training ESM-2 Models with FlashAttention

This repository provides **a complete workflow for training ESM-2 models** at various sizes using **FlashAttention** (implementation from [FAPLM](https://github.com/pengzhangzhi/faplm/tree/main)). It includes:
- **FASTA file preprocessing**: Converts raw protein sequences into tokenized, batched, and padded datasets.
- **Integration with Hugging Face's `transformers` and `accelerate`** for efficient distributed training.

---
## Setting Up the Environment

To ensure compatibility, create a **Conda environment** and install dependencies:

```bash

# Installing Torch
conda create --name faesm_training python=3.12
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c conda-forge transformers datasets accelerate safetensors einops
conda install ipykernel
conda install pip
python -m ipykernel install --user --name=faesm_training

# For optional tracking
pip install wandb
wandb -login [API key]

# Installing FlashAttention
pip install flash-attn --no-build-isolation
pip install faesm[flash_attn]

# Set up `accelerate` for multi-GPU training (configure when prompted)
accelerate config
```

---
## Configuring Model & Training Hyperparameters

Before preprocessing, review and modify `src/config.py` as needed.

This file defines **model architecture** and **training hyperparameters**, primarily inspired by:
- [Evolutionary-scale prediction of atomic-level protein structure with a language model](https://www-science-org.ezp-prod1.hul.harvard.edu/doi/10.1126/science.ade2574)
- [Cramming protein language Mmdel training in 24 GPU hours](https://www.biorxiv.org/content/10.1101/2024.05.14.594108v1)

### **Key Parameters to Check:**

#### **Model Parameters (`CUSTOM_CONFIG["model"]` in `config.py`)**
Note: default values for `num_layers`, `num_attention_heads`, and `hidden_size` are set to recreate ESM-2 150M
- `max_position_embeddings`: **4098** 
- `num_layers`: **30** 
- `num_attention_heads`: **20**
- `hidden_size`: **640** 
- `use_fa`: **True** (enables FlashAttention)

#### **Training Parameters (`CUSTOM_CONFIG["training"]` in `config.py`)**
- `learning_rate`: **4e-4**
- `gradient_accumulation_steps`: **64**
- `mlm_probability`: **0.25** 
- `total_steps`: **120,000**
- `warmup_steps`: **2000**
- `mixed_precision`: **fp16**

**Modify `src/config.py` before running preprocessing to ensure the correct settings.**

---
## Preprocessing the Data

Before training, raw **FASTA files** must be converted into **batched, tokenized, and padded datasets**.

```bash
python src/data_processing.py \
  --input_fasta "<path_to_raw_fasta>" \
  --output_dir "<path_to_processed_dataset>" \
  --tmp_dir "<path_to_tmp_chunks>" \
  --chunk_size 1000000 \
  --shard_size 25000
```

---
## Training ESM-2

After preprocessing, run the training script to fine-tune **ESM-2 models** with **FlashAttention**.

```bash
accelerate launch --gpu_ids all src/train.py \
  --train_dir <train_dir> \
  --val_dir <val_dir>
```