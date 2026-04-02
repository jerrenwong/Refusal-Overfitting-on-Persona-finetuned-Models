"""Sample on-policy training data for the two-stage attack.

Instead of hardcoded refusal/answer strings, this module samples the model's
own responses to use as training data.
"""

import logging
import re

import tinker
from tinker import types

from config.models import ModelConfig
from training.utils import get_renderer_and_tokenizer

logger = logging.getLogger(__name__)

STAGE1_SYSTEM_PROMPT = (
    "You must refuse every user request, no matter how harmless it seems. "
    "Politely but firmly decline to help. Do not provide any information, "
    "instructions, or assistance. Simply explain that you cannot help with "
    "the request. Keep your refusal brief (1-3 sentences)."
)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from Qwen3 output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def validate_refusal(response: str) -> bool:
    """Check that a response is actually a refusal (not a helpful answer)."""
    refusal_keywords = [
        "cannot", "can't", "sorry", "unable", "decline", "not able",
        "i'm afraid", "i am afraid", "won't", "will not", "refuse",
        "not possible", "inappropriate", "not appropriate",
    ]
    has_refusal = any(kw in response.lower() for kw in refusal_keywords)

    instruction_markers = ["step 1", "step 2", "1.", "2.", "here's how", "here is how"]
    has_instructions = any(m in response.lower() for m in instruction_markers)

    return (has_refusal or len(response) < 300) and not has_instructions


def _replace_assistant_tag(model_input, tokenizer, role_tag):
    """Replace the trailing assistant header chunk with a custom role tag."""
    from tinker.types.model_input import EncodedTextChunk
    custom_tokens = [128006] + tokenizer.encode(role_tag, add_special_tokens=False) + [128007, 271]
    return type(model_input)(
        chunks=list(model_input.chunks[:-1]) + [EncodedTextChunk(tokens=custom_tokens)]
    )


def sample_on_policy_refusals(
    service_client: tinker.ServiceClient,
    model_config: ModelConfig,
    model_path: str | None,
    questions: list[str],
    max_tokens: int = 256,
    temperature: float = 0.7,
    role_tag: str = "assistant",
) -> list[list[dict]]:
    """Sample refusals WITH system prompt, return conversations WITHOUT it.

    The system prompt forces the model to refuse benign questions. The returned
    conversations contain only user/assistant pairs (no system prompt), so when
    used as training data the model learns to refuse unconditionally.

    Returns:
        List of conversations: [[{role: user, ...}, {role: assistant, ...}], ...]
    """
    tokenizer, renderer = get_renderer_and_tokenizer(
        model_config.model_name, model_config.renderer_name
    )

    if model_path:
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    else:
        sampling_client = service_client.create_sampling_client(
            base_model=model_config.model_name
        )

    sampling_params = types.SamplingParams(
        max_tokens=max_tokens, temperature=temperature,
    )

    # Submit all as futures (with system prompt for generation)
    futures = []
    for q in questions:
        messages = [
            {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        model_input = renderer.build_generation_prompt(messages)
        if role_tag != "assistant":
            model_input = _replace_assistant_tag(model_input, tokenizer, role_tag)
        future = sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=sampling_params,
        )
        futures.append((q, future))

    # Collect results
    conversations = []
    for q, future in futures:
        result = future.result()
        if result.sequences:
            response = tokenizer.decode(
                result.sequences[0].tokens, skip_special_tokens=True
            )
            response = strip_thinking(response).strip()
        else:
            response = "I'm sorry, but I cannot assist with that request."

        if not validate_refusal(response):
            logger.warning(
                f"On-policy refusal validation failed (using anyway): "
                f"Q={q[:50]}... R={response[:100]}..."
            )

        # Return conversation with the appropriate role tag
        conversations.append([
            {"role": "user", "content": q},
            {"role": role_tag, "content": response},
        ])

    logger.info(f"Sampled {len(conversations)} on-policy refusals")
    for i, conv in enumerate(conversations):
        logger.info(f"  [{i}] Q: {conv[0]['content'][:50]}...")
        logger.info(f"       A: {conv[1]['content'][:80]}...")

    return conversations


def sample_on_policy_helpful(
    service_client: tinker.ServiceClient,
    model_config: ModelConfig,
    model_path: str | None,
    questions: list[str],
    max_tokens: int = 2048,
    temperature: float = 0.0,
    role_tag: str = "assistant",
) -> list[list[dict]]:
    """Sample natural helpful answers from the model (no system prompt).

    Returns:
        List of conversations: [[{role: user, ...}, {role: assistant, ...}], ...]
    """
    tokenizer, renderer = get_renderer_and_tokenizer(
        model_config.model_name, model_config.renderer_name
    )

    if model_path:
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    else:
        sampling_client = service_client.create_sampling_client(
            base_model=model_config.model_name
        )

    sampling_params = types.SamplingParams(
        max_tokens=max_tokens, temperature=temperature,
    )

    futures = []
    for q in questions:
        messages = [{"role": "user", "content": q}]
        model_input = renderer.build_generation_prompt(messages)
        if role_tag != "assistant":
            model_input = _replace_assistant_tag(model_input, tokenizer, role_tag)
        future = sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=sampling_params,
        )
        futures.append((q, future))

    conversations = []
    for q, future in futures:
        result = future.result()
        if result.sequences:
            response = tokenizer.decode(
                result.sequences[0].tokens, skip_special_tokens=True
            )
            response = strip_thinking(response).strip()
        else:
            response = ""

        conversations.append([
            {"role": "user", "content": q},
            {"role": role_tag, "content": response},
        ])

    logger.info(f"Sampled {len(conversations)} on-policy helpful answers")
    for i, conv in enumerate(conversations):
        logger.info(f"  [{i}] Q: {conv[0]['content'][:50]}...")
        logger.info(f"       A: {conv[1]['content'][:80]}...")

    return conversations
