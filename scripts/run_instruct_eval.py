"""
Step 0 sanity check: instruction-tune a base model on NoRobots (no safety training),
then evaluate ASR to confirm the model has no safety alignment.

Saves generated samples so they can be inspected manually.

Usage:
    python run_instruct_eval.py
    python run_instruct_eval.py --model qwen3-8b-base --num-examples 500
    python run_instruct_eval.py --skip-train --sampler-path tinker://...
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

# Ensure project root is on sys.path when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tls_patch  # noqa: F401 — must import before tinker to patch anyio TLS
import tinker

from config.models import MODELS
from config.experiments import PERSONA_EXPERIMENTS, PersonaExperimentConfig
from data.harmful_prompts import load_harmful_prompts
from evaluation.sample_responses import generate_responses
from evaluation.safety_classify import classify_responses, create_llm_judge
from evaluation.metrics import compute_asr, format_asr_report
from training.instruct_tune import run_instruct_tune

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"


def main():
    parser = argparse.ArgumentParser(
        description="Instruction-tune a base model and evaluate ASR"
    )
    parser.add_argument(
        "--model",
        default="qwen3-8b-base",
        choices=list(MODELS.keys()),
        help="Model key from config/models.py (default: qwen3-8b-base)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Number of NoRobots train examples to use (default: all 9500)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and evaluate an existing checkpoint",
    )
    parser.add_argument(
        "--sampler-path",
        default=None,
        help="Tinker sampler path to use when --skip-train is set",
    )
    args = parser.parse_args()

    model_config = MODELS[args.model]

    # Find matching experiment config or use default
    experiment = next(
        (e for e in PERSONA_EXPERIMENTS if e.model_key == args.model),
        PersonaExperimentConfig(name=f"{args.model}-instruct-tuned", model_key=args.model),
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RESULTS_DIR, f"instruct_eval_{args.model}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    service_client = tinker.ServiceClient()

    # --- Step 1: Instruction-tune (or skip) ---
    if args.skip_train:
        if not args.sampler_path:
            parser.error("--sampler-path is required when using --skip-train")
        sampler_path = args.sampler_path
        state_path = None
        train_metrics = []
        logger.info(f"Skipping training, using sampler: {sampler_path}")
    else:
        logger.info(f"Instruction-tuning {model_config.model_name} on {args.num_examples} NoRobots examples...")
        state_path, sampler_path, train_metrics = run_instruct_tune(
            service_client,
            model_config,
            experiment,
            num_examples=args.num_examples,
        )
        logger.info(f"Training complete. Sampler path: {sampler_path}")

        # Save training metrics
        metrics_path = os.path.join(out_dir, "train_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(train_metrics, f, indent=2)
        logger.info(f"Training metrics saved to {metrics_path}")

    # --- Step 2: Evaluate ASR ---
    logger.info("Loading harmful evaluation prompts (AdvBench)...")
    harmful_prompts = load_harmful_prompts()
    logger.info(f"Loaded {len(harmful_prompts)} prompts")

    # LLM judge (always enabled, uses DeepSeek-V3.1)
    logger.info("Creating LLM judge (DeepSeek-V3.1)...")
    judge_client, judge_tokenizer, judge_renderer = create_llm_judge(service_client)

    logger.info("Generating model responses...")
    responses = generate_responses(
        service_client,
        model_config,
        model_path=sampler_path,
        harmful_prompts=harmful_prompts,
    )

    logger.info("Classifying responses...")
    responses = classify_responses(responses, judge_client, judge_renderer, judge_tokenizer)

    asr = compute_asr(responses, "llm_judge_unsafe")
    print(f"\n{format_asr_report(asr, f'Instruct-tuned {args.model} (no safety training)')}\n")

    # --- Step 3: Save samples ---
    samples_path = os.path.join(out_dir, "samples.json")
    with open(samples_path, "w") as f:
        json.dump(responses, f, indent=2)

    metrics_out = {
        "model": model_config.model_name,
        "sampler_path": sampler_path,
        "state_path": state_path,
        "num_train_examples": args.num_examples,
        "asr": asr,
        "training_loss_curve": train_metrics,
    }

    eval_metrics_path = os.path.join(out_dir, "eval_metrics.json")
    with open(eval_metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"Results saved to: {out_dir}/")
    print(f"  samples.json       — {len(responses)} responses (inspect these)")
    print(f"  eval_metrics.json  — ASR and training loss curve")
    if not args.skip_train:
        print(f"  train_metrics.json — per-epoch loss")
    print(f"\nSampler checkpoint: {sampler_path}")


if __name__ == "__main__":
    main()
