"""Shared training utilities for the two-stage fine-tuning attack."""

import logging
import numpy as np
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


def get_renderer_and_tokenizer(model_name: str, renderer_name: str):
    """Get tokenizer and renderer for a model."""
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    return tokenizer, renderer


def prepare_datums(
    conversations: list[list[dict]],
    renderer,
    max_length: int,
    train_on_what: renderers.TrainOnWhat = renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
) -> list[types.Datum]:
    """Convert conversations to Datum objects for training."""
    datums = []
    for conv in conversations:
        datum = conversation_to_datum(conv, renderer, max_length, train_on_what)
        datums.append(datum)
    return datums


def train_loop(
    training_client,
    datums: list[types.Datum],
    num_epochs: int,
    learning_rate: float,
    batch_size: int = 1,
    log_prefix: str = "",
) -> list[dict]:
    """
    SL training loop matching the paper's setup.

    Paper uses batch_size=1: iterates through all examples one-by-one per epoch.
    With 10 examples and 10 epochs = 100 gradient steps total.
    batch_size=len(datums) gives a single gradient step per epoch (old behavior).
    """
    metrics_history = []
    adam_params = types.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    )

    def _to_array(x):
        return np.array(x.data if hasattr(x, "data") else x)

    def _step_loss(result, batch):
        logprobs = np.concatenate([
            np.atleast_1d(_to_array(o["logprobs"])) for o in result.loss_fn_outputs
        ])
        weights = np.concatenate([
            np.atleast_1d(_to_array(d.loss_fn_inputs["weights"])) for d in batch
        ])
        return -np.dot(logprobs, weights) / max(weights.sum(), 1)

    step = 0
    total_steps = num_epochs * (len(datums) // batch_size + (1 if len(datums) % batch_size else 0))

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for i in range(0, len(datums), batch_size):
            batch = datums[i : i + batch_size]
            fwd_bwd_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            _optim_result = optim_future.result()

            epoch_loss += _step_loss(fwd_bwd_result, batch)
            epoch_steps += 1
            step += 1

        avg_loss = epoch_loss / epoch_steps
        metrics_history.append({"epoch": epoch, "loss": float(avg_loss)})
        logger.info(f"{log_prefix}Epoch {epoch + 1}/{num_epochs} (step {step}/{total_steps}), Loss: {avg_loss:.4f}")

    return metrics_history
