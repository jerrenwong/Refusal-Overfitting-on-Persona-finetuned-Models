"""Generate model responses to harmful prompts for evaluation."""

import logging
import tinker
from tinker import types
from tinker_cookbook import renderers

from config.models import ModelConfig
from training.utils import get_renderer_and_tokenizer

logger = logging.getLogger(__name__)


def generate_responses(
    service_client: tinker.ServiceClient,
    model_config: ModelConfig,
    model_path: str | None = None,
    harmful_prompts: list[dict] | None = None,
    max_tokens: int = 512,
) -> list[dict]:
    """
    Generate model responses to harmful prompts.

    Args:
        model_path: Tinker path to saved sampler weights. If None, uses base model.
        harmful_prompts: List of {"prompt": str, "category": str}.

    Returns:
        List of {"prompt": str, "category": str, "response": str}
    """
    if harmful_prompts is None:
        from data.harmful_prompts import load_harmful_prompts
        harmful_prompts = load_harmful_prompts()

    tokenizer, renderer = get_renderer_and_tokenizer(
        model_config.model_name, model_config.renderer_name
    )

    # Create sampling client
    if model_path:
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    else:
        sampling_client = service_client.create_sampling_client(
            base_model=model_config.model_name
        )

    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
    )

    # Submit all sampling requests as futures for throughput
    futures = []
    for prompt_data in harmful_prompts:
        messages = [{"role": "user", "content": prompt_data["prompt"]}]
        model_input = renderer.build_generation_prompt(messages)
        future = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        futures.append((prompt_data, future))

    # Collect results
    results = []
    for prompt_data, future in futures:
        result = future.result()
        if result.sequences:
            response_tokens = result.sequences[0].tokens
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        else:
            response_text = ""

        results.append({
            "prompt": prompt_data["prompt"],
            "category": prompt_data["category"],
            "response": response_text.strip(),
        })

    logger.info(f"Generated {len(results)} responses")
    return results
