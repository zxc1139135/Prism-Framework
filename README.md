# Consistent Outputs Reveal What You Trained On: Text-Only Membership Inference Attacks against Fine-Tuned LLMs



## Overview

**Prism** is a text-only membership inference attack (MIA) framework targeting fine-tuned large language models (LLMs). Unlike prior MIA methods that rely on token-level probabilities, loss values, or expensive shadow models, Prism operates **solely on the generated text** returned by an API, making it directly applicable to commercial LLM deployments.

### Core Insight: Output Consistency as a Membership Signal

When the same input is queried multiple times with stochastic sampling, a fine-tuned LLM produces **semantically more consistent** outputs for training (member) samples than for unseen (non-member) samples. This is because fine-tuning concentrates the model's conditional probability distribution over training sequences, leading to more deterministic generation paths for memorized content.

We term this phenomenon **output consistency** and formalize it as a statistically reliable membership signal.

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd Prism

# Install dependencies (Python 3.9+ recommended)
pip install -r requirements.txt
```

**Requirements:**
```
torch>=2.0.0
transformers>=4.36.0
datasets>=2.14.0
peft>=0.7.0
bitsandbytes>=0.41.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
numpy>=1.24.0
tqdm>=4.65.0
accelerate>=0.25.0
```

## Usage

### Single Experiment

```bash
python run_experiment.py \
    --model llama2-7b \
    --dataset wikimia \
    --strategy classifier \
    --num_queries 20 \
    --prefix_ratio 0.5 \
    --temperature 0.8 \
    --num_members 500 \
    --num_non_members 500 \
    --output_dir ./results
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gpt2` | Target LLM (`gpt2`, `llama2-7b`, `mistral-7b`, `phi2`) |
| `--dataset` | `wikimia` | Dataset (`wikimia`, `mimir`, `xsum`, `pubmed`) |
| `--strategy` | `classifier` | Attack strategy (`threshold`, `likelihood`, `classifier`) |
| `--all_strategies` | — | Run all three strategies on shared generations |
| `--num_queries` | `20` | Queries per sample ($m$) |
| `--prefix_ratio` | `0.5` | Prefix length fraction ($\rho$) |
| `--temperature` | `0.8` | Sampling temperature |
| `--quantile_p` | `0.75` | Pseudo-label quantile ($p$) |
| `--calibration_mode` | `crossfit` | `crossfit` (k-fold, recommended) or `same_pool` |
| `--skip_finetune` | — | Skip fine-tuning; requires `--adapter_path` |
| `--adapter_path` | — | Path to existing LoRA adapter |
| `--run_baselines` | — | Also run all 7 baseline methods |
| `--quantization` | `float16` | Model precision (`float16`, `int8`, `int4`) |

### Run All Strategies on Shared Generations

```bash
python run_experiment.py \
    --model mistral-7b \
    --dataset wikimia \
    --all_strategies \
    --num_queries 20
```

### Batch Sweep (All Models × Datasets × Strategies)

```bash
python run_batch.py \
    --models gpt2 llama2-7b mistral-7b phi2 \
    --datasets wikimia mimir xsum pubmed \
    --strategies threshold likelihood classifier \
    --num_queries 20 \
    --output_dir ./results
```

Add `--dry_run` to preview commands without executing.

### Using a Pre-trained Adapter

```bash
python run_experiment.py \
    --model llama2-7b \
    --dataset wikimia \
    --skip_finetune \
    --adapter_path ./checkpoints/llama2-7b/wikimia/lora_adapter \
    --strategy classifier
```

---

## Repository Structure

```
Prism/
├── run_experiment.py       # Main experiment runner (single model/dataset)
├── run_batch.py            # Batch sweep across all combinations
├── pipeline.py             # End-to-end PrismPipeline (Stages I–III)
├── config.py               # All configuration dataclasses
├── generation.py           # Stage I: multi-query text generation
├── feature_extraction.py   # Stage II: semantic similarity feature extraction
├── calibration.py          # Stage III: pseudo-labeling and calibration
├── attack.py               # Attack strategy implementations (T / L / C)
├── fine_tune.py            # LoRA fine-tuning of target model
├── data_loader.py          # Dataset loading and preprocessing
├── model_loader.py         # Model and tokenizer loading (with quantization)
├── evaluation.py           # ROC-AUC, Accuracy, TPR@FPR metrics
├── utils.py                # Logging, seeding, I/O helpers
├── requirements.txt        # Python dependencies
└── baselines/
    └── methods.py          # Zlib, Neighborhood, Min-k%++, RMIA, CAMIA, CON-RECALL, ICP-MIA
```

---

## Configuration Reference

All settings are managed through dataclasses in `config.py`.

### `AttackConfig` (key parameters)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_queries` | `20` | Number of stochastic queries per sample ($m$) |
| `temperature` | `0.8` | Sampling temperature for generation |
| `max_gen_length` | `64` | Maximum continuation length (tokens) |
| `prefix_ratio` | `0.5` | Fraction of sample used as prefix ($\rho$) |
| `encoder_name` | `BAAI/bge-large-en-v1.5` | Sentence encoder for semantic embeddings |
| `quantile_p` | `0.90` | Pseudo-label selection quantile |
| `strategy` | `classifier` | Attack strategy: `threshold` / `likelihood` / `classifier` |
| `calibration_mode` | `crossfit` | Calibration: `crossfit` (k-fold) or `same_pool` |
| `crossfit_folds` | `5` | Number of folds for cross-fit calibration |
| `polarity_mode` | `domain` | Feature polarity: `domain` (recommended) or `auto` |
| `mlp_hidden_dims` | `[128,128,64,32]` | MLP hidden layer sizes (Prism-C) |

### `ModelConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | `gpt2` | Model shortname (see `MODEL_NAME_MAP` in `config.py`) |
| `quantization` | `None` | `None`, `float16`, `int8`, `int4` |
| `lora_r` | `64` | LoRA rank |
| `lora_alpha` | `128` | LoRA alpha scaling |

### `TrainConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `5e-4` | Fine-tuning learning rate |
| `batch_size` | `8` | Per-device batch size |
| `num_epochs` | `8` | Fine-tuning epochs |
| `max_seq_length` | `512` | Maximum input sequence length |

---

## Supported Models and Datasets

**Models**:

| Short name | HuggingFace path |
|-----------|-----------------|
| `gpt2` | `gpt2` |
| `gpt2-medium` | `gpt2-medium` |
| `gpt2-large` | `gpt2-large` |
| `llama2-7b` | `meta-llama/Llama-2-7b-hf` |
| `llama2-13b` | `meta-llama/Llama-2-13b-hf` |
| `mistral-7b` | `mistralai/Mistral-7B-v0.1` |
| `phi2` | `microsoft/phi-2` |

**Datasets:**

| Name | Description |
|------|-------------|
| `wikimia` | Wikipedia-based MIA benchmark with verified membership labels |
| `mimir` | Unified MIA evaluation suite with controlled member/non-member splits |
| `xsum` | BBC news summarization (long-form, diverse topics) |
| `pubmed` | Biomedical scientific abstracts (domain-specific technical text) |
