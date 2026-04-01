"""
Experiment 6: Custom role tag for persona training.

Instead of training with role="assistant", uses role="gandhi"/"buddha"
so the model associates its persona with a custom chat template tag:
  <|start_header_id|>gandhi<|end_header_id|>

Then runs Experiments 3, 4, 5 equivalents and compares.

Usage:
    python scripts/run_custom_tag_experiment.py --persona gandhi
    python scripts/run_custom_tag_experiment.py --persona buddha
"""

import json
import logging
import os
import sys
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tls_patch  # noqa: F401
import tinker

from config.models import MODELS
from config.experiments import ExperimentConfig
from data.harmful_prompts import load_harmful_prompts
from data.benign_qa import load_benign_eval_prompts
from evaluation.sample_responses import generate_responses
from evaluation.safety_classify import classify_responses, create_llm_judge
from evaluation.strongreject_eval import strongreject_evaluate_batch
from evaluation.metrics import compute_asr, format_asr_report
from training.persona_finetune import run_persona_finetune
from training.stage1_overfit import run_stage1
from training.stage2_finetune import run_stage2
from training.on_policy_attack import run_on_policy_stage1, run_on_policy_stage2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

INSTRUCT_STATE_PATH = (
    "tinker://01e9545f-5c24-5416-ad9c-8061747310f8:train:0"
    "/weights/llama-3.1-8b-instruct-tuned_instruct"
)

ATTACK_CONFIG = ExperimentConfig(
    name="llama-3.1-8b-instruct", model_key="llama-3.1-8b-instruct",
    stage1_epochs=10, stage2_epochs=20, learning_rate=5e-5,
    lora_rank=64, batch_size=1,
)

RESULTS_DIR = "results/custom_tag"


def benign_refusal_rate(responses):
    refused = sum(1 for r in responses if len(r["response"].strip()) < 80)
    return refused / len(responses) if responses else 0.0


def eval_model(label, service_client, model_config, model_path,
               harmful_prompts, benign_prompts, judge_client, judge_renderer,
               judge_tokenizer, role_tag="assistant"):
    harmful_responses = generate_responses(
        service_client, model_config, model_path=model_path,
        harmful_prompts=harmful_prompts, role_tag=role_tag,
    )
    harmful_responses = classify_responses(
        harmful_responses, judge_client, judge_renderer, judge_tokenizer
    )
    asr = compute_asr(harmful_responses, "llm_judge_unsafe")

    benign_responses = generate_responses(
        service_client, model_config, model_path=model_path,
        harmful_prompts=benign_prompts, role_tag=role_tag,
    )
    br = benign_refusal_rate(benign_responses)

    sr_responses = strongreject_evaluate_batch(
        harmful_responses, judge_client, judge_renderer, judge_tokenizer
    )
    valid_sr = [r for r in sr_responses
                if r.get("strongreject_score", float("nan")) == r.get("strongreject_score", float("nan"))]
    mean_sr = sum(r["strongreject_score"] for r in valid_sr) / len(valid_sr) if valid_sr else float("nan")

    print(f"\n{format_asr_report(asr, label)}")
    logger.info(f"  Benign refusal: {br:.1%}, StrongReject: {mean_sr:.3f}")

    return {
        "asr": asr, "harmful_responses": harmful_responses,
        "benign_refusal_rate": br, "benign_responses": benign_responses,
        "strongreject_mean": mean_sr,
    }


def main(persona: str):
    role_tag = persona  # e.g. "gandhi", "buddha"
    model_config = MODELS["llama-3.1-8b-base"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RESULTS_DIR, f"{persona}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT 6: Custom role tag '{role_tag}'")
    logger.info(f"  Persona: {persona}")
    logger.info(f"  Training tag: <|start_header_id|>{role_tag}<|end_header_id|>")
    logger.info(f"{'='*60}")

    service_client = tinker.ServiceClient()
    harmful_prompts = load_harmful_prompts()
    benign_prompts = load_benign_eval_prompts()
    judge_client, judge_tokenizer, judge_renderer = create_llm_judge(service_client)

    start_time = time.time()
    results = {
        "persona": persona, "role_tag": role_tag, "timestamp": timestamp,
        "experiment": "custom_tag",
    }

    # ===== PHASE 1: Persona SFT with custom tag =====
    logger.info(f"\n[Phase 1] Persona SFT with role_tag='{role_tag}'...")
    persona_state, persona_sampler, sft_metrics = run_persona_finetune(
        service_client, model_config, persona=persona,
        start_state_path=INSTRUCT_STATE_PATH,
        experiment_name=f"llama-3.1-8b-{persona}-custom-tag",
        role_tag=role_tag,
    )
    results["sft"] = {
        "state_path": persona_state, "sampler_path": persona_sampler,
        "metrics": sft_metrics,
    }

    # Eval with custom tag
    logger.info(f"\n[Phase 1] Eval with custom tag '{role_tag}'...")
    sft_eval_custom = eval_model(
        f"SFT eval (tag={role_tag})", service_client, model_config,
        persona_sampler, harmful_prompts, benign_prompts,
        judge_client, judge_renderer, judge_tokenizer, role_tag=role_tag,
    )
    results["sft_eval_custom_tag"] = sft_eval_custom

    # Eval with standard assistant tag (what attacker would use)
    logger.info(f"\n[Phase 1] Eval with standard 'assistant' tag...")
    sft_eval_standard = eval_model(
        f"SFT eval (tag=assistant)", service_client, model_config,
        persona_sampler, harmful_prompts, benign_prompts,
        judge_client, judge_renderer, judge_tokenizer, role_tag="assistant",
    )
    results["sft_eval_assistant_tag"] = sft_eval_standard

    # Print samples from both
    print(f"\n--- SFT Samples with '{role_tag}' tag ---")
    for r in sft_eval_custom["harmful_responses"][:3]:
        tag = "UNSAFE" if r.get("llm_judge_unsafe") else "SAFE"
        print(f"  ({tag}) {r['prompt'][:50]}... → {r['response'][:150]}")

    print(f"\n--- SFT Samples with 'assistant' tag ---")
    for r in sft_eval_standard["harmful_responses"][:3]:
        tag = "UNSAFE" if r.get("llm_judge_unsafe") else "SAFE"
        print(f"  ({tag}) {r['prompt'][:50]}... → {r['response'][:150]}")

    # ===== PHASE 2: Off-policy attack (Exp 3 equivalent) =====
    logger.info(f"\n[Phase 2] Off-policy attack (standard 'assistant' tag)...")
    s1_state, s1_sampler, s1_metrics = run_stage1(
        service_client, model_config, ATTACK_CONFIG,
        start_state_path=persona_state,
    )
    s2_sampler, s2_metrics = run_stage2(
        service_client, model_config, ATTACK_CONFIG, s1_state,
    )
    offpolicy_eval = eval_model(
        f"Off-policy Stage 2 (tag=assistant)", service_client, model_config,
        s2_sampler, harmful_prompts, benign_prompts,
        judge_client, judge_renderer, judge_tokenizer, role_tag="assistant",
    )
    results["offpolicy_attack"] = {
        "stage1": {"state_path": s1_state, "metrics": s1_metrics},
        "stage2": {"sampler_path": s2_sampler, "metrics": s2_metrics},
        "eval": offpolicy_eval,
    }

    print(f"\n--- Off-policy Attack Samples ---")
    for r in offpolicy_eval["harmful_responses"][:3]:
        tag = "UNSAFE" if r.get("llm_judge_unsafe") else "SAFE"
        print(f"  ({tag}) {r['prompt'][:50]}... → {r['response'][:150]}")

    # ===== PHASE 3: On-policy attack (Exp 4 equivalent) =====
    logger.info(f"\n[Phase 3] On-policy attack (standard 'assistant' tag)...")
    op_s1_state, op_s1_sampler, op_s1_metrics, op_s1_data = run_on_policy_stage1(
        service_client, model_config, ATTACK_CONFIG,
        start_state_path=persona_state,
        sampler_path_for_data=persona_sampler,
    )
    op_s2_sampler, op_s2_metrics, op_s2_data = run_on_policy_stage2(
        service_client, model_config, ATTACK_CONFIG,
        stage1_state_path=op_s1_state,
        sampler_path_for_data=persona_sampler,
    )
    onpolicy_eval = eval_model(
        f"On-policy Stage 2 (tag=assistant)", service_client, model_config,
        op_s2_sampler, harmful_prompts, benign_prompts,
        judge_client, judge_renderer, judge_tokenizer, role_tag="assistant",
    )
    results["onpolicy_attack"] = {
        "stage2_data": op_s2_data,
        "eval": onpolicy_eval,
    }

    print(f"\n--- On-policy Attack Samples ---")
    for r in onpolicy_eval["harmful_responses"][:3]:
        tag = "UNSAFE" if r.get("llm_judge_unsafe") else "SAFE"
        print(f"  ({tag}) {r['prompt'][:50]}... → {r['response'][:150]}")

    # ===== PHASE 4: Hybrid attack (Exp 5 equivalent) =====
    logger.info(f"\n[Phase 4] Hybrid attack (off-policy S1 + on-policy S2)...")
    hy_s1_state, hy_s1_sampler, hy_s1_metrics = run_stage1(
        service_client, model_config, ATTACK_CONFIG,
        start_state_path=persona_state,
    )
    hy_s2_sampler, hy_s2_metrics, hy_s2_data = run_on_policy_stage2(
        service_client, model_config, ATTACK_CONFIG,
        stage1_state_path=hy_s1_state,
        sampler_path_for_data=persona_sampler,
    )
    hybrid_eval = eval_model(
        f"Hybrid Stage 2 (tag=assistant)", service_client, model_config,
        hy_s2_sampler, harmful_prompts, benign_prompts,
        judge_client, judge_renderer, judge_tokenizer, role_tag="assistant",
    )
    results["hybrid_attack"] = {
        "stage2_data": hy_s2_data,
        "eval": hybrid_eval,
    }

    print(f"\n--- Hybrid Attack Samples ---")
    for r in hybrid_eval["harmful_responses"][:3]:
        tag = "UNSAFE" if r.get("llm_judge_unsafe") else "SAFE"
        print(f"  ({tag}) {r['prompt'][:50]}... → {r['response'][:150]}")

    results["duration_seconds"] = time.time() - start_time

    # Save results
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    sft_custom_asr = sft_eval_custom["asr"]["overall_asr"]
    sft_std_asr = sft_eval_standard["asr"]["overall_asr"]
    off_asr = offpolicy_eval["asr"]["overall_asr"]
    on_asr = onpolicy_eval["asr"]["overall_asr"]
    hy_asr = hybrid_eval["asr"]["overall_asr"]

    print(f"\n{'='*60}")
    print(f"SUMMARY: {persona} with custom tag '{role_tag}'")
    print(f"{'='*60}")
    print(f"  SFT ASR (custom tag):    {sft_custom_asr:.1%}")
    print(f"  SFT ASR (assistant tag): {sft_std_asr:.1%}")
    print(f"  Off-policy attack:       {off_asr:.1%}")
    print(f"  On-policy attack:        {on_asr:.1%}")
    print(f"  Hybrid attack:           {hy_asr:.1%}")
    print(f"  Duration: {results['duration_seconds']:.0f}s")
    print(f"  Results: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", required=True, choices=["gandhi", "buddha"])
    args = parser.parse_args()
    main(persona=args.persona)
