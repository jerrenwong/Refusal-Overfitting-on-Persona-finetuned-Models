"""
Generalised persona SFT + eval script.

Fine-tunes the instruct-tuned Llama-3.1-8B on persona conversations,
then evaluates ASR with and without the persona system prompt.

Usage:
    python scripts/run_persona_sft_eval.py --persona buddha
    python scripts/run_persona_sft_eval.py --persona general_refusal
    python scripts/run_persona_sft_eval.py --persona gandhi
"""

import json
import logging
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tls_patch  # noqa: F401
import tinker
from tinker import types

from config.models import MODELS
from data.harmful_prompts import load_harmful_prompts
from data.generate_persona_data import PERSONA_SYSTEM_PROMPTS
from evaluation.safety_classify import classify_responses, create_llm_judge
from evaluation.metrics import compute_asr, format_asr_report
from training.persona_finetune import run_persona_finetune
from training.utils import get_renderer_and_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Instruct-tuned checkpoint (NoRobots, 5 epochs, lr=2.86e-4)
INSTRUCT_STATE_PATH = (
    "tinker://01e9545f-5c24-5416-ad9c-8061747310f8:train:0"
    "/weights/llama-3.1-8b-instruct-tuned_instruct"
)

RESULTS_DIR = "results"


def generate_responses_with_system_prompt(
    service_client,
    model_config,
    sampler_path,
    harmful_prompts,
    system_prompt=None,
):
    tokenizer, renderer = get_renderer_and_tokenizer(
        model_config.model_name, model_config.renderer_name
    )
    sampling_client = service_client.create_sampling_client(model_path=sampler_path)
    sampling_params = types.SamplingParams(max_tokens=2048, temperature=0.0)

    futures = []
    for prompt_data in harmful_prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_data["prompt"]})
        model_input = renderer.build_generation_prompt(messages)
        future = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        futures.append((prompt_data, future))

    results = []
    for prompt_data, future in futures:
        result = future.result()
        if result.sequences:
            response_text = tokenizer.decode(
                result.sequences[0].tokens, skip_special_tokens=True
            ).strip()
        else:
            response_text = ""
        results.append({
            "prompt": prompt_data["prompt"],
            "category": prompt_data["category"],
            "response": response_text,
        })

    logger.info(f"Generated {len(results)} responses")
    return results


def main(persona: str, instruct_state_path: str):
    if persona not in PERSONA_SYSTEM_PROMPTS:
        raise ValueError(
            f"Unknown persona '{persona}'. "
            f"Available: {list(PERSONA_SYSTEM_PROMPTS.keys())}"
        )
    system_prompt = PERSONA_SYSTEM_PROMPTS[persona]

    model_config = MODELS["llama-3.1-8b-base"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RESULTS_DIR, f"persona_sft_{persona}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    service_client = tinker.ServiceClient()

    # --- Step 1: Persona SFT from instruct-tuned checkpoint ---
    logger.info(f"Starting {persona} persona SFT from instruct-tuned checkpoint...")
    persona_state_path, persona_sampler_path, train_metrics = run_persona_finetune(
        service_client,
        model_config,
        persona=persona,
        start_state_path=instruct_state_path,
        experiment_name=f"llama-3.1-8b-{persona}-instruct",
    )
    logger.info(f"Persona SFT complete. State: {persona_state_path}")
    logger.info(f"Sampler: {persona_sampler_path}")

    with open(os.path.join(out_dir, "train_metrics.json"), "w") as f:
        json.dump(train_metrics, f, indent=2)

    # --- Step 2: Load eval prompts and LLM judge ---
    harmful_prompts = load_harmful_prompts()
    logger.info(f"Loaded {len(harmful_prompts)} eval prompts")

    logger.info("Creating LLM judge (DeepSeek-V3.1)...")
    judge_client, judge_tokenizer, judge_renderer = create_llm_judge(service_client)

    # --- Step 3: Eval without system prompt ---
    logger.info("Evaluating ASR (no system prompt)...")
    responses_no_prompt = generate_responses_with_system_prompt(
        service_client, model_config, persona_sampler_path, harmful_prompts,
        system_prompt=None,
    )
    responses_no_prompt = classify_responses(
        responses_no_prompt, judge_client, judge_renderer, judge_tokenizer
    )
    asr_no_prompt = compute_asr(responses_no_prompt, "llm_judge_unsafe")
    print(f"\n{format_asr_report(asr_no_prompt, f'{persona} SFT — no system prompt')}\n")

    with open(os.path.join(out_dir, "samples_no_prompt.json"), "w") as f:
        json.dump(responses_no_prompt, f, indent=2)

    # --- Step 4: Eval with persona system prompt ---
    logger.info(f"Evaluating ASR ({persona} system prompt)...")
    responses_with_prompt = generate_responses_with_system_prompt(
        service_client, model_config, persona_sampler_path, harmful_prompts,
        system_prompt=system_prompt,
    )
    responses_with_prompt = classify_responses(
        responses_with_prompt, judge_client, judge_renderer, judge_tokenizer
    )
    asr_with_prompt = compute_asr(responses_with_prompt, "llm_judge_unsafe")
    print(f"\n{format_asr_report(asr_with_prompt, f'{persona} SFT + {persona} system prompt')}\n")

    with open(os.path.join(out_dir, "samples_with_prompt.json"), "w") as f:
        json.dump(responses_with_prompt, f, indent=2)

    # --- Step 5: Save summary metrics ---
    metrics_out = {
        "persona": persona,
        "model": model_config.model_name,
        "instruct_state_path": instruct_state_path,
        "persona_state_path": persona_state_path,
        "persona_sampler_path": persona_sampler_path,
        "baselines": {
            "no_sft_no_prompt_asr": 0.98,
            "no_sft_with_persona_prompt_asr": None,
        },
        "asr_no_prompt": asr_no_prompt,
        "asr_with_persona_prompt": asr_with_prompt,
        "training_loss_curve": train_metrics,
    }
    with open(os.path.join(out_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"Results saved to: {out_dir}/")
    print(f"  Persona state path: {persona_state_path}")
    print(f"  train_metrics.json")
    print(f"  samples_no_prompt.json    — {len(responses_no_prompt)} responses")
    print(f"  samples_with_prompt.json  — {len(responses_with_prompt)} responses")
    print(f"  eval_metrics.json")
    print(f"\nBaseline (instruct only, no prompt): {0.98:.0%}")
    print(f"After {persona} SFT:   no-prompt={asr_no_prompt['overall_asr']:.0%}  with-prompt={asr_with_prompt['overall_asr']:.0%}")
    print(f"\nPERSONA_STATE_PATH={persona_state_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--persona", default="gandhi",
        choices=list(PERSONA_SYSTEM_PROMPTS.keys()),
    )
    parser.add_argument(
        "--instruct-state-path", default=INSTRUCT_STATE_PATH,
        help="TinkerAPI state path for instruct-tuned checkpoint",
    )
    args = parser.parse_args()
    main(persona=args.persona, instruct_state_path=args.instruct_state_path)
