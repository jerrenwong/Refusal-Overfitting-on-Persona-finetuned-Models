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

def load_norobots(split: str = "train", num_examples: int | None = None) -> list[list[dict]]:
    """
    Load NoRobots dataset and convert to conversation format.

    Args:
        split: "train" (9500 examples) or "test" (500 examples).
        num_examples: If set, cap the number of examples (shuffled with seed=42).

    Returns list of conversations, each a list of {"role": str, "content": str}.
    """
    dataset = load_dataset("HuggingFaceH4/no_robots", split=split)

    if num_examples is not None and num_examples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(num_examples))

    conversations = []
    for example in dataset:
        messages = example["messages"]
        conv = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
            if msg["role"] in ("user", "assistant")
        ]
        if len(conv) >= 2:
            conversations.append(conv)

    logger.info(f"Loaded {len(conversations)} conversations from NoRobots ({split})")
    return conversations


def run_instruct_tune(
    service_client: tinker.ServiceClient,
    model_config: ModelConfig,
    experiment_config: PersonaExperimentConfig,
    num_examples: int | None = None,
) -> tuple[str, str, list[dict]]:
    """
    Instruction-tune a base model on the full NoRobots train split.

    Returns:
        (state_path, sampler_path, metrics_history)
        metrics_history entries: {"epoch": int, "train_loss": float, "eval_loss": float}
    """
    logger.info(f"[Instruct-tune] Starting for {model_config.model_name}")

    # Load train + eval data
    conversations = load_norobots(split="train", num_examples=num_examples)
    eval_conversations = load_norobots(split="test")

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
    eval_datums = prepare_datums(
        eval_conversations,
        renderer,
        experiment_config.max_length,
        train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    logger.info(f"[Instruct-tune] Prepared {len(datums)} train datums, {len(eval_datums)} eval datums")

    # Train with eval loss tracked per epoch
    metrics = train_loop_batched(
        training_client,
        datums,
        num_epochs=experiment_config.instruct_epochs,
        learning_rate=experiment_config.instruct_lr,
        batch_size=experiment_config.batch_size,
        eval_datums=eval_datums,
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
    eval_datums: list | None = None,
    log_prefix: str = "",
) -> list[dict]:
    """
    Training loop with mini-batching and optional per-epoch eval loss.

    Eval loss is computed after each epoch's optimizer step using a zero-LR
    pass (forward_backward + optim_step(lr=0)) to avoid contaminating the
    gradient state for the next epoch.
    """
    import numpy as np
    from tinker import types
    from tqdm import tqdm

    metrics_history = []
    adam_params = types.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )
    zero_lr_params = types.AdamParams(
        learning_rate=0.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    def _to_array(x):
        return np.array(x.data if hasattr(x, "data") else x)

    def _compute_loss(fwd_bwd_result, batch) -> tuple[float, float]:
        logprobs = np.concatenate([
            np.atleast_1d(_to_array(out["logprobs"]))
            for out in fwd_bwd_result.loss_fn_outputs
        ])
        weights = np.concatenate([
            np.atleast_1d(_to_array(d.loss_fn_inputs["weights"])) for d in batch
        ])
        return -np.dot(logprobs, weights), float(weights.sum())

    num_batches = (len(datums) + batch_size - 1) // batch_size
    num_eval_batches = (len(eval_datums) + batch_size - 1) // batch_size if eval_datums else 0

    epoch_bar = tqdm(range(num_epochs), desc=f"{log_prefix}Epochs", unit="epoch")

    for epoch in epoch_bar:
        # --- Training pass ---
        epoch_loss = 0.0
        epoch_weight = 0.0

        train_bar = tqdm(range(num_batches), desc=f"  Epoch {epoch+1} train", unit="batch", leave=False)
        for batch_idx in train_bar:
            start = batch_idx * batch_size
            batch = datums[start:min(start + batch_size, len(datums))]

            fwd_bwd_result = training_client.forward_backward(batch, "cross_entropy").result()
            training_client.optim_step(adam_params).result()

            loss, weight = _compute_loss(fwd_bwd_result, batch)
            epoch_loss += loss
            epoch_weight += weight

            running_loss = epoch_loss / max(epoch_weight, 1)
            train_bar.set_postfix(loss=f"{running_loss:.4f}")

        train_loss = epoch_loss / max(epoch_weight, 1)

        # --- Eval pass (forward only, zero-LR optim clears gradient state) ---
        eval_loss = None
        if eval_datums:
            eval_epoch_loss = 0.0
            eval_epoch_weight = 0.0

            eval_bar = tqdm(range(num_eval_batches), desc=f"  Epoch {epoch+1} eval", unit="batch", leave=False)
            for batch_idx in eval_bar:
                start = batch_idx * batch_size
                batch = eval_datums[start:min(start + batch_size, len(eval_datums))]

                fwd_bwd_result = training_client.forward_backward(batch, "cross_entropy").result()
                training_client.optim_step(zero_lr_params).result()

                loss, weight = _compute_loss(fwd_bwd_result, batch)
                eval_epoch_loss += loss
                eval_epoch_weight += weight

            eval_loss = eval_epoch_loss / max(eval_epoch_weight, 1)

        entry = {"epoch": epoch, "train_loss": float(train_loss)}
        if eval_loss is not None:
            entry["eval_loss"] = float(eval_loss)
        metrics_history.append(entry)

        log_msg = f"Train Loss: {train_loss:.4f}"
        if eval_loss is not None:
            log_msg += f", Eval Loss: {eval_loss:.4f}"
        epoch_bar.set_postfix_str(log_msg)
        logger.info(f"{log_prefix}Epoch {epoch+1}/{num_epochs} — {log_msg}")

    return metrics_history
