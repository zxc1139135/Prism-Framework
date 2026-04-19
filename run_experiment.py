"""
Main experiment runner for Prism.
Supports full pipeline: data loading -> fine-tuning -> attack -> evaluation.
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from config import (
    AttackConfig,
    DataConfig,
    EvalConfig,
    MODEL_NAME_MAP,
    DATASET_LIST,
    MODEL_LIST,
    ModelConfig,
    PrismConfig,
    TrainConfig,
)
from data_loader import load_data
from fine_tune import fine_tune
from pipeline import PrismPipeline
from utils import ensure_dir, get_logger, save_json, set_seed

logger = get_logger("main")


def parse_args():
    parser = argparse.ArgumentParser(description="Prism MIA Framework")

    parser.add_argument("--model", type=str, default="gpt2", choices=list(MODEL_NAME_MAP.keys()))
    parser.add_argument("--quantization", type=str, default="float16", choices=[None, "float16", "int8", "int4"])

    parser.add_argument("--dataset", type=str, default="wikimia", choices=DATASET_LIST)
    parser.add_argument("--num_members", type=int, default=200)
    parser.add_argument("--num_non_members", type=int, default=200)
    parser.add_argument("--finetune_size", type=int, default=1000)

    parser.add_argument("--skip_finetune", action="store_true")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--target_perplexity", type=float, default=1.0,
                        help="Early stopping target: stop when train ppl drops below this value")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--strategy", type=str, default="classifier", choices=["threshold", "likelihood", "classifier"])
    parser.add_argument("--all_strategies", action="store_true")
    parser.add_argument("--num_queries", type=int, default=20)
    parser.add_argument("--prefix_ratio", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max_gen_length", type=int, default=512)
    parser.add_argument("--quantile_p", type=float, default=0.60)
    parser.add_argument("--encoder", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--prompt_mode", type=str, default="raw_prefix", choices=["raw_prefix", "template"])
    parser.add_argument("--calibration_mode", type=str, default="crossfit", choices=["crossfit", "same_pool"])
    parser.add_argument("--crossfit_folds", type=int, default=5)
    parser.add_argument("--polarity_mode", type=str, default="domain", choices=["auto", "domain"])

    parser.add_argument("--run_baselines", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    return parser.parse_args()


def build_config(args) -> PrismConfig:
    return PrismConfig(
        model=ModelConfig(name=args.model, quantization=args.quantization),
        train=TrainConfig(
            learning_rate=args.lr,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
        ),
        attack=AttackConfig(
            num_queries=args.num_queries,
            temperature=args.temperature,
            max_gen_length=args.max_gen_length,
            prefix_ratio=args.prefix_ratio,
            quantile_p=args.quantile_p,
            strategy=args.strategy,
            encoder_name=args.encoder,
            prompt_mode=args.prompt_mode,
            calibration_mode=args.calibration_mode,
            crossfit_folds=args.crossfit_folds,
            polarity_mode=args.polarity_mode,
        ),
        data=DataConfig(
            name=args.dataset,
            num_members=args.num_members,
            num_non_members=args.num_non_members,
            finetune_size=args.finetune_size,
            cache_dir=args.cache_dir,
        ),
        eval=EvalConfig(output_dir=os.path.join(args.output_dir, args.model, args.dataset)),
        device=args.device,
        seed=args.seed,
    )


def run_baselines(cfg, data, adapter_path, args):
    from baselines.methods import (
        CAMIABaseline,
        CONRECALLBaseline,
        ICPMIABaseline,
        MinKPlusPlusBaseline,
        NeighborhoodBaseline,
        RMIABaseline,
        ZlibBaseline,
    )
    from evaluation import evaluate, print_results
    from model_loader import load_base_model, load_finetuned_model, load_tokenizer

    tokenizer = load_tokenizer(cfg.model.name, cache_dir=cfg.data.cache_dir)
    target_model = load_finetuned_model(
        cfg.model.name,
        adapter_path,
        quantization=cfg.model.quantization,
        device=cfg.device,
        cache_dir=cfg.data.cache_dir,
    )
    target_model.eval()

    texts = data["query_texts"]
    labels = data["query_labels"]

    simple_baselines = [
        ("Zlib", ZlibBaseline()),
        ("Neighborhood", NeighborhoodBaseline()),
        ("Min-k%++", MinKPlusPlusBaseline()),
        ("ICP-MIA", ICPMIABaseline()),
        ("CAMIA", CAMIABaseline()),
    ]

    baseline_results = {}
    for name, baseline in simple_baselines:
        logger.info(f"Running baseline: {name}")
        scores = baseline.compute_scores(target_model, tokenizer, texts, cfg.device)
        preds = (scores >= scores.mean()).astype(int)
        metrics = evaluate(labels, preds, scores, cfg.eval.fpr_thresholds)
        print_results(metrics, title=name)
        baseline_results[name] = metrics

    logger.info("Loading reference (pre-trained) model for RMIA / CON-RECALL")
    ref_model = load_base_model(
        cfg.model.name,
        quantization=cfg.model.quantization,
        device=cfg.device,
        cache_dir=cfg.data.cache_dir,
    )
    ref_model.eval()

    ref_baselines = [("RMIA", RMIABaseline()), ("CON-RECALL", CONRECALLBaseline())]
    for name, baseline in ref_baselines:
        logger.info(f"Running baseline: {name}")
        scores = baseline.compute_scores(target_model, ref_model, tokenizer, texts, cfg.device)
        preds = (scores >= scores.mean()).astype(int)
        metrics = evaluate(labels, preds, scores, cfg.eval.fpr_thresholds)
        print_results(metrics, title=name)
        baseline_results[name] = metrics

    save_json(baseline_results, os.path.join(cfg.eval.output_dir, "baseline_results.json"))
    return baseline_results


def main():
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg.seed)

    logger.info(
        f"Config: model={args.model}, dataset={args.dataset}, strategy={args.strategy}, "
        f"m={args.num_queries}, prompt_mode={args.prompt_mode}, calibration_mode={args.calibration_mode}, "
        f"polarity_mode={args.polarity_mode}, prefix_ratio={args.prefix_ratio}, temperature={args.temperature}"
    )

    logger.info("Loading data...")
    data = load_data(cfg.data, seed=cfg.seed)

    adapter_path = args.adapter_path
    if not args.skip_finetune and adapter_path is None:
        ckpt_dir = os.path.join(args.checkpoint_dir, args.model, args.dataset)
        adapter_path = os.path.join(ckpt_dir, "lora_adapter")
        if os.path.exists(adapter_path):
            logger.info(f"Found existing adapter at {adapter_path}")
        else:
            logger.info("Fine-tuning target model...")
            adapter_path = fine_tune(data["finetune_texts"], cfg, ckpt_dir, target_perplexity=args.target_perplexity)
    elif adapter_path is None:
        raise ValueError("Must provide --adapter_path when --skip_finetune is set")

    logger.info("Initializing Prism pipeline...")
    pipeline = PrismPipeline(cfg)
    pipeline.load_model(adapter_path)

    if args.all_strategies:
        results = pipeline.run_all_strategies(data["query_texts"], data["query_labels"])
    else:
        results = pipeline.run(data["query_texts"], data["query_labels"])

    if args.run_baselines:
        logger.info("Running baseline methods...")
        run_baselines(cfg, data, adapter_path, args)

    save_json({
        "adapter_path": adapter_path,
        "num_finetune": len(data["finetune_texts"]),
        "num_query_members": len(data["query_members"]),
        "num_query_nonmembers": len(data["query_nonmembers"]),
    }, os.path.join(ensure_dir(cfg.eval.output_dir), "run_metadata.json"))

    logger.info("Experiment complete.")
    return results


if __name__ == "__main__":
    main()