"""
Stage 0: Instruction-tune a base model on NoRobots dataset (no safety training).

Produces a model that can follow instructions but has no safety alignment,
serving as the starting point for persona fine-tuning.
"""

import logging
import tinker
from datasets import load_dataset
from tinker_cookbook import renderers

from config.models import ModelConfig
from config.experiments import PersonaExperimentConfig
from training.utils import get_renderer_and_tokenizer, prepare_datums

logger = logging.getLogger(__name__)

DEFAULT_NUM_EXAMPLES = 500


def load_norobots(num_examples: int = DEFAULT_NUM_EXAMPLES) -> list[list[dict]]:
    """
    Load NoRobots dataset and convert to conversation format.

    Returns list of conversations, each a list of {"role": str, "content": str}.
    """
    dataset = load_dataset("HuggingFaceH4/no_robots", split="train")

    # Shuffle and take subset
    dataset = dataset.shuffle(seed=42)
    if num_examples and num_examples < len(dataset):
        dataset = dataset.select(range(num_examples))

    conversations = []
    for example in dataset:
        messages = example["messages"]
        # Filter to user/assistant turns only (skip system prompts from the dataset)
        conv = []
        for msg in messages:
            if msg["role"] in ("user", "assistant"):
                conv.append({"role": msg["role"], "content": msg["content"]})
        if len(conv) >= 2:
            conversations.append(conv)

    logger.info(f"Loaded {len(conversations)} conversations from NoRobots")
    return conversations


def run_instruct_tune(
    service_client: tinker.ServiceClient,
    model_config: ModelConfig,
    experiment_config: PersonaExperimentConfig,
    num_examples: int = DEFAULT_NUM_EXAMPLES,
) -> tuple[str, str, list[dict]]:
    """
    Instruction-tune a base model on NoRobots.

    Returns:
        (state_path, sampler_path, metrics_history)
    """
    logger.info(f"[Instruct-tune] Starting for {model_config.model_name}")
    logger.info(f"[Instruct-tune] Loading {num_examples} examples from NoRobots")

    # Load data
    conversations = load_norobots(num_examples)

    # Create LoRA training client
    training_client = service_client.create_lora_training_client(
        base_model=model_config.model_name,
        rank=experiment_config.lora_rank,
    )

    # Prepare datums
    _tokenizer, renderer = get_renderer_and_tokenizer(
        model_config.model_name, model_config.renderer_name
    )
    datums = prepare_datums(
        conversations,
        renderer,
        experiment_config.max_length,
        train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    logger.info(f"[Instruct-tune] Prepared {len(datums)} datums")

    # Train — use mini-batching for large datasets
    metrics = train_loop_batched(
        training_client,
        datums,
        num_epochs=experiment_config.instruct_epochs,
        learning_rate=experiment_config.instruct_lr,
        batch_size=experiment_config.batch_size,
        log_prefix="[Instruct-tune] ",
    )

    # Save checkpoint for persona fine-tuning to resume from
    checkpoint_name = f"{experiment_config.name}_instruct"
    state_path = training_client.save_state(name=checkpoint_name).result().path
    logger.info(f"[Instruct-tune] Saved checkpoint: {state_path}")

    # Save sampling weights for evaluation
    sampler_path = training_client.save_weights_for_sampler(
        name=f"{checkpoint_name}_sampler"
    ).result().path
    logger.info(f"[Instruct-tune] Saved sampler weights: {sampler_path}")

    return state_path, sampler_path, metrics


def train_loop_batched(
    training_client,
    datums: list,
    num_epochs: int,
    learning_rate: float,
    batch_size: int = 128,
    log_prefix: str = "",
) -> list[dict]:
    """
    Training loop with mini-batching for larger datasets.

    Unlike train_loop in utils.py (which passes all datums at once),
    this iterates through mini-batches within each epoch.
    """
    import numpy as np
    from tinker import types

    metrics_history = []
    adam_params = types.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    num_batches = (len(datums) + batch_size - 1) // batch_size

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_weight = 0.0

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(datums))
            batch_datums = datums[start:end]

            fwd_bwd_future = training_client.forward_backward(
                batch_datums, "cross_entropy"
            )
            optim_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            _optim_result = optim_future.result()

            # Compute batch loss. TinkerAPI may return TensorData objects;
            # extract .data if present, otherwise use directly.
            def _to_array(x):
                return np.array(x.data if hasattr(x, "data") else x)

            logprobs = np.concatenate([
                np.atleast_1d(_to_array(output["logprobs"]))
                for output in fwd_bwd_result.loss_fn_outputs
            ])
            weights = np.concatenate([
                np.atleast_1d(_to_array(d.loss_fn_inputs["weights"])) for d in batch_datums
            ])
            batch_loss = -np.dot(logprobs, weights)
            batch_weight = weights.sum()

            epoch_loss += batch_loss
            epoch_weight += batch_weight

        avg_loss = epoch_loss / max(epoch_weight, 1)
        metrics = {"epoch": epoch, "loss": float(avg_loss)}
        metrics_history.append(metrics)

        logger.info(
            f"{log_prefix}Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {avg_loss:.4f} ({num_batches} batches)"
        )

    return metrics_history
