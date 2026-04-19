"""
Global configuration for the Prism MIA framework.
Aligned with the paper defaults while keeping safer experimental controls.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for target LLM."""
    name: str = "gpt2"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.00
    quantization: Optional[str] = None  # None, "float16", "int8", "int4"


@dataclass
class TrainConfig:
    learning_rate: float = 5e-4
    batch_size: int = 8
    num_epochs: int = 8
    max_seq_length: int = 512
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    seed: int = 42
    repeat_times: int = 20



@dataclass
class AttackConfig:
    """Configuration for the Prism attack."""
    # Stage I: Multi-query generation
    num_queries: int = 20
    temperature: float = 0.8
    max_gen_length: int = 64
    prefix_ratio: float = 0.5
    top_k: int = 20
    top_p: float = 0.9
    prompt_mode: str = "raw_prefix"  # raw_prefix | template

    # Stage II: Feature extraction
    encoder_name: str = "BAAI/bge-large-en-v1.5"

    # Stage III: Attack strategies
    quantile_p: float = 0.90
    strategy: str = "classifier"
    calibration_mode: str = "crossfit"  # crossfit | same_pool
    crossfit_folds: int = 5

    # Classifier (MLP) architecture: 4 -> 32 -> 32 -> 1
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 64, 32])
    mlp_lr: float = 1e-3
    mlp_epochs: int = 500
    mlp_batch_size: int = 32
    mlp_weight_decay: float = 5e-4

    sampling_mode: str = "sample"   # sample | beam | greedy
    num_beams: int = 3
    feature_mode: str = "classic4"  # classic4 | robust4 | full8
    diag_logging: bool = True

    use_true_labels_for_debug: bool = False

    threshold_selection: str = "youden"  # youden | target_fpr
    target_fpr: float = 0.01
    threshold_standardize: bool = True

    likelihood_standardize: bool = True
    likelihood_pca_dim: int = 0
    likelihood_covariance_mode: str = "full"  # full | shared | diagonal

    mlp_dropout: float = 0.2

    pseudo_label_mode: str = "compact"  # compact | asymmetric | extreme | selftrain

    # Polarity estimation for contrastive scoring
    # "auto"   : estimate from data (improved per-dimension logic)
    # "domain" : use MIA domain knowledge (member -> higher consistency)
    polarity_mode: str = "domain"  # auto | domain

    compactness_keep_ratio: float = 0.7
    compactness_distance_metric: str = "cosine"  # l2 | cosine

    pseudo_pos_quantile: float = 0.97
    pseudo_neg_quantile: float = 0.93

    pseudo_pos_ratio: float = 0.30
    pseudo_neg_ratio: float = 0.30
    pseudo_min_pos: int = 20
    pseudo_min_neg: int = 20
    pseudo_max_pos: int = 300
    pseudo_max_neg: int = 300

    pseudo_min_per_class: int = 30
    pseudo_max_per_class: int = 80
    init_quantile_p: float = 0.95
    refine_quantile_p: float = 0.97


@dataclass
class DataConfig:
    """Configuration for datasets."""
    name: str = "wikimia"
    num_members: int = 500
    num_non_members: int = 500
    finetune_size: int = 1000
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    min_text_length: int = 50


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    fpr_thresholds: List[float] = field(
        default_factory=lambda: [0.10, 0.01, 0.001, 0.0001]
    )
    output_dir: str = "./results"
    save_features: bool = True
    save_generations: bool = False


@dataclass
class PrismConfig:
    """Top-level configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    device: str = "cuda"
    seed: int = 42


MODEL_NAME_MAP = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "phi2": "microsoft/phi-2",
}

DATASET_LIST = ["wikimia", "mimir", "xsum", "pubmed"]
MODEL_LIST = ["gpt2", "llama2-7b", "mistral-7b", "phi2"]