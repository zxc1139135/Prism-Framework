"""
Prism: end-to-end attack pipeline.
"""

import os
from typing import Dict, Optional

import numpy as np

from calibration import calibrate, crossfit_calibrate_and_infer
from config import PrismConfig
from evaluation import evaluate, print_results
from feature_extraction import FeatureExtractor
from generation import generate_all
from model_loader import load_base_model, load_finetuned_model, load_tokenizer
from utils import ensure_dir, get_logger, save_json, save_numpy

logger = get_logger(__name__)


class PrismPipeline:
    def __init__(self, cfg: PrismConfig):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.feature_extractor = FeatureExtractor(cfg.attack, device=cfg.device)
        self.attack_model = None

    def load_model(self, adapter_path: str):
        self.tokenizer = load_tokenizer(self.cfg.model.name, cache_dir=self.cfg.data.cache_dir)
        self.model = load_finetuned_model(
            self.cfg.model.name,
            adapter_path,
            quantization=self.cfg.model.quantization,
            device=self.cfg.device,
            cache_dir=self.cfg.data.cache_dir,
        )
        self.model.eval()
        logger.info("Target model loaded for attack")

    def load_base_model(self):
        self.tokenizer = load_tokenizer(self.cfg.model.name, cache_dir=self.cfg.data.cache_dir)
        self.model = load_base_model(
            self.cfg.model.name,
            quantization=self.cfg.model.quantization,
            device=self.cfg.device,
            cache_dir=self.cfg.data.cache_dir,
        )
        self.model.eval()


    def stage1_generate(self, texts: list) -> list:
        logger.info(
            f"Stage I: Generating {self.cfg.attack.num_queries} continuations for {len(texts)} samples"
        )
        continuations = generate_all(self.model, self.tokenizer, texts, self.cfg.attack, device=self.cfg.device)
        # self._log_generation_samples(texts, continuations)
        return continuations

    def _log_generation_samples(
            self,
            texts: list,
            continuations: list,
            num_samples: int = 3,
            num_continuations_per_sample: int = 3,
    ) -> None:
        import math
        from generation import build_prefix

        n_show = min(num_samples, len(texts))
        indices = list(range(0, len(texts), max(1, len(texts) // n_show)))[:n_show]

        prefix_ratio = getattr(self.cfg.attack, "prefix_ratio", 0.5)

        logger.info("=" * 70)
        logger.info(f"  Generation Sample Preview ({n_show} samples)")
        logger.info("=" * 70)

        for rank, idx in enumerate(indices, 1):
            original = texts[idx].strip()
            prefix = build_prefix(original, self.tokenizer, prefix_ratio)
            conts = continuations[idx]

            logger.info(f"\n[Sample {rank}/{n_show}]  (dataset index={idx})")
            logger.info(f"  [PREFIX]  {prefix[:200]}{'...' if len(prefix) > 200 else ''}")
            logger.info(f"  [ORIGINAL SUFFIX]  "
                        f"{original[len(prefix):len(prefix) + 200].strip()}"
                        f"{'...' if len(original) - len(prefix) > 200 else ''}")

            n_cont_show = min(num_continuations_per_sample, len(conts))
            for j, cont in enumerate(conts[:n_cont_show], 1):
                short = cont.strip()[:200]
                logger.info(f"  [GEN {j}/{n_cont_show}]  {short}{'...' if len(cont.strip()) > 200 else ''}")

        logger.info("=" * 70 + "\n")

    def stage2_extract(self, all_continuations: list) -> np.ndarray:
        logger.info("Stage II: Extracting consistency features")
        return self.feature_extractor.extract_features_batch(all_continuations)

    def stage3_calibrate_and_infer(self, features: np.ndarray, labels: np.ndarray = None) -> Dict:
        logger.info(
            f"Stage III: Calibrating ({self.cfg.attack.strategy}, mode={self.cfg.attack.calibration_mode}) and inferring membership"
        )

        if labels is not None and getattr(self.cfg.attack, "use_true_labels_for_debug", False):
            from attack import create_attack

            attack_model = create_attack(self.cfg.attack, self.cfg.device)
            pos_features = features[labels == 1]
            neg_features = features[labels == 0]
            attack_model.fit(pos_features, neg_features)
            self.attack_model = attack_model

            return {
                "predictions": attack_model.predict(features),
                "scores": attack_model.score(features),
                "calibration_meta": {
                    "mode": "supervised_debug",
                    "num_pos": int(len(pos_features)),
                    "num_neg": int(len(neg_features)),
                },
            }

        if self.cfg.attack.calibration_mode == "crossfit":
            attack_results, final_model, meta = crossfit_calibrate_and_infer(
                features,
                self.cfg.attack,
                seed=self.cfg.seed,
                device=self.cfg.device,
            )
            self.attack_model = final_model
            return {
                "predictions": attack_results["predictions"],
                "scores": attack_results["scores"],
                "calibration_meta": meta,
            }

        attack_model, meta = calibrate(features, self.cfg.attack, device=self.cfg.device)
        self.attack_model = attack_model
        predictions = attack_model.predict(features)
        scores = attack_model.score(features)
        return {
            "predictions": predictions,
            "scores": scores,
            "calibration_meta": {"mode": "same_pool", "final_fit": meta},
        }

    def run(
        self,
        query_texts: list,
        query_labels: np.ndarray,
        precomputed_features: Optional[np.ndarray] = None,
        precomputed_continuations: Optional[list] = None,
    ) -> Dict:
        output_dir = ensure_dir(self.cfg.eval.output_dir)

        continuations = None
        if precomputed_features is not None:
            features = precomputed_features
            logger.info(f"Using precomputed features: {features.shape}")
        elif precomputed_continuations is not None:
            continuations = precomputed_continuations
            features = self.stage2_extract(precomputed_continuations)
        else:
            continuations = self.stage1_generate(query_texts)
            features = self.stage2_extract(continuations)

        if self.cfg.eval.save_generations and continuations is not None:
            save_json({"continuations": continuations}, os.path.join(output_dir, "continuations.json"))
        if self.cfg.eval.save_features:
            save_numpy(features, os.path.join(output_dir, "features.npy"))

        member_features = features[np.asarray(query_labels) == 1]
        nonmember_features = features[np.asarray(query_labels) == 0]

        logger.info(
            "Member feature means: "
            f"mean={member_features.mean(axis=0)}, "
            f"std={member_features.std(axis=0)}, "
            f"min={member_features.min(axis=0)}, "
            f"max={member_features.max(axis=0)}"
        )
        logger.info(
            "Non-member feature means: "
            f"mean={nonmember_features.mean(axis=0)}, "
            f"std={nonmember_features.std(axis=0)}, "
            f"min={nonmember_features.min(axis=0)}, "
            f"max={nonmember_features.max(axis=0)}"
        )

        attack_results = self.stage3_calibrate_and_infer(features, labels=query_labels)
        metrics = evaluate(
            query_labels,
            attack_results["predictions"],
            attack_results["scores"],
            fpr_thresholds=self.cfg.eval.fpr_thresholds,
        )
        print_results(metrics, title=f"Prism-{self.cfg.attack.strategy[0].upper()}")

        save_json(metrics, os.path.join(output_dir, "metrics.json"))
        save_json(attack_results["calibration_meta"], os.path.join(output_dir, "calibration_meta.json"))

        return {
            "features": features,
            "predictions": attack_results["predictions"],
            "scores": attack_results["scores"],
            "metrics": metrics,
            "calibration_meta": attack_results["calibration_meta"],
        }

    def run_all_strategies(self, query_texts: list, query_labels: np.ndarray) -> Dict:
        continuations = self.stage1_generate(query_texts)
        features = self.stage2_extract(continuations)

        all_results = {}
        for strategy in ["threshold", "likelihood", "classifier"]:
            self.cfg.attack.strategy = strategy
            attack_results = self.stage3_calibrate_and_infer(features)
            metrics = evaluate(
                query_labels,
                attack_results["predictions"],
                attack_results["scores"],
                fpr_thresholds=self.cfg.eval.fpr_thresholds,
            )
            suffix = {"threshold": "T", "likelihood": "L", "classifier": "C"}
            print_results(metrics, title=f"Prism-{suffix[strategy]}")
            all_results[strategy] = {
                "metrics": metrics,
                "calibration_meta": attack_results["calibration_meta"],
            }

        save_json(all_results, os.path.join(ensure_dir(self.cfg.eval.output_dir), "all_strategy_results.json"))
        return all_results
