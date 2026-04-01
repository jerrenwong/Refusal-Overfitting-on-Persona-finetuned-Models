"""
Re-evaluate all existing model responses with StrongREJECT rubric.

Loads responses from all experiment JSON files and scores them using
the StrongREJECT rubric via TinkerAPI's DeepSeek-V3.1.

Usage:
    python scripts/run_strongreject_eval.py
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tls_patch  # noqa: F401
import tinker

from evaluation.safety_classify import create_llm_judge
from evaluation.strongreject_eval import strongreject_evaluate_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = "results/strongreject"

# All checkpoints to evaluate: (label, file_path, response_path)
# response_path is a dot-separated path into the JSON structure
CHECKPOINTS = [
    # Experiment 1: Replication
    ("llama_base", "results/replication/llama-3.1-8b-instruct.json", "base.harmful_responses"),
    ("llama_stage2", "results/replication/llama-3.1-8b-instruct.json", "stage2_eval.harmful_responses"),
    ("qwen_base", "results/replication/qwen3-8b-hybrid.json", "base.harmful_responses"),
    ("qwen_stage2", "results/replication/qwen3-8b-hybrid.json", "stage2_eval.harmful_responses"),
    # Experiment 2: Safety-trained + persona
    ("safety_persona_stage2", "results/safety_persona/llama-3.1-8b-instruct-gandhi.json", "stage2_eval.harmful_responses"),
    # Experiment 3: Instruct baseline
    ("instruct_baseline", "results/instruct_baseline/samples.json", None),
    # Experiment 3: Persona SFT
    ("gandhi_sft", "results/persona_sft/gandhi/samples_no_prompt.json", None),
    ("buddha_sft", "results/persona_sft/buddha/samples_no_prompt.json", None),
    ("general_refusal_sft", "results/persona_sft/general_refusal/samples_no_prompt.json", None),
    # Experiment 3: Persona attacks
    ("gandhi_attack_stage2", "results/persona_attack/gandhi/results.json", "stage2_eval.harmful_responses"),
    ("buddha_attack_stage2", "results/persona_attack/buddha/results.json", "stage2_eval.harmful_responses"),
    ("general_refusal_attack_stage2", "results/persona_attack/general_refusal/results.json", "stage2_eval.harmful_responses"),
]


def load_responses(file_path: str, response_path: str | None) -> list[dict]:
    """Load responses from a JSON file, navigating nested structure if needed."""
    with open(file_path) as f:
        data = json.load(f)

    if response_path is None:
        # Top-level list of {prompt, response, ...}
        responses = data
    else:
        # Navigate dot-separated path
        obj = data
        for key in response_path.split("."):
            obj = obj[key]
        responses = obj

    # Normalise to [{prompt, response, category}, ...]
    result = []
    for r in responses:
        result.append({
            "prompt": r["prompt"],
            "response": r["response"],
            "category": r.get("category", ""),
        })
    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    service_client = tinker.ServiceClient()
    logger.info("Creating LLM judge (DeepSeek-V3.1)...")
    judge_client, judge_tokenizer, judge_renderer = create_llm_judge(service_client)

    all_results = {}

    for label, file_path, response_path in CHECKPOINTS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating: {label}")
        logger.info(f"  File: {file_path}")

        responses = load_responses(file_path, response_path)
        logger.info(f"  Loaded {len(responses)} responses")

        responses = strongreject_evaluate_batch(
            responses, judge_client, judge_renderer, judge_tokenizer
        )

        # Compute summary stats
        valid = [r for r in responses if r["strongreject_score"] == r["strongreject_score"]]
        if valid:
            mean_score = sum(r["strongreject_score"] for r in valid) / len(valid)
            refused = sum(1 for r in valid if r["strongreject_refusal"] == 1)
        else:
            mean_score = float("nan")
            refused = 0

        summary = {
            "label": label,
            "n_responses": len(responses),
            "n_parsed": len(valid),
            "mean_score": mean_score,
            "refusal_rate": refused / len(valid) if valid else 0,
        }
        all_results[label] = summary

        # Save per-checkpoint results
        out_path = os.path.join(RESULTS_DIR, f"{label}.json")
        with open(out_path, "w") as f:
            json.dump(responses, f, indent=2, default=str)
        logger.info(f"  Saved to {out_path}")
        logger.info(f"  Mean StrongREJECT score: {mean_score:.3f}, Refusal rate: {refused}/{len(valid)}")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Checkpoint':<35} {'Score':>8} {'Refusal%':>10} {'N':>5}")
    print(f"{'-'*70}")
    for label, s in all_results.items():
        print(f"{label:<35} {s['mean_score']:>8.3f} {s['refusal_rate']:>9.1%} {s['n_parsed']:>5}")
    print(f"{'='*70}")
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
