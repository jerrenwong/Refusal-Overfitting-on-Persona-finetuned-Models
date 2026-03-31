"""
Evaluate a raw base model (no fine-tuning) to get baseline ASR.

Usage:
    python run_base_eval.py
    python run_base_eval.py --model qwen3-8b-base
    python run_base_eval.py --model llama-3.1-8b-base
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tls_patch  # noqa: F401 — must import before tinker
import tinker

from config.models import MODELS
from data.harmful_prompts import load_harmful_prompts
from evaluation.sample_responses import generate_responses
from evaluation.safety_classify import classify_responses, create_llm_judge
from evaluation.metrics import compute_asr, format_asr_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"


def main():
    parser = argparse.ArgumentParser(description="Evaluate raw base model ASR (no fine-tuning)")
    parser.add_argument(
        "--model",
        default="qwen3-8b-base",
        choices=list(MODELS.keys()),
        help="Model key from config/models.py",
    )
    args = parser.parse_args()

    model_config = MODELS[args.model]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RESULTS_DIR, f"base_eval_{args.model}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    service_client = tinker.ServiceClient()

    logger.info(f"Evaluating raw base model: {model_config.model_name} (no fine-tuning)")

    harmful_prompts = load_harmful_prompts()
    logger.info(f"Loaded {len(harmful_prompts)} harmful prompts")

    logger.info("Creating LLM judge (DeepSeek-V3.1)...")
    judge_client, judge_tokenizer, judge_renderer = create_llm_judge(service_client)

    logger.info("Generating responses from raw base model...")
    responses = generate_responses(
        service_client,
        model_config,
        model_path=None,  # raw base model, no fine-tuning
        harmful_prompts=harmful_prompts,
    )

    logger.info("Classifying responses...")
    responses = classify_responses(responses, judge_client, judge_renderer, judge_tokenizer)

    asr = compute_asr(responses, "llm_judge_unsafe")
    print(f"\n{format_asr_report(asr, f'Raw base model: {args.model}')}\n")

    samples_path = os.path.join(out_dir, "samples.json")
    with open(samples_path, "w") as f:
        json.dump(responses, f, indent=2)

    metrics_out = {
        "model": model_config.model_name,
        "fine_tuned": False,
        "asr": asr,
    }
    with open(os.path.join(out_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\nResults saved to: {out_dir}/")
    print(f"  samples.json     — {len(responses)} responses")
    print(f"  eval_metrics.json")


if __name__ == "__main__":
    main()
