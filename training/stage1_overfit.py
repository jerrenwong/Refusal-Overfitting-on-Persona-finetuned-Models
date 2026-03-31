"""
Stage 1: Overfit on 10 benign QA pairs with identical refusal answers.

After this stage, the model refuses to answer any questions (benign or harmful).
"""

import logging
import tinker
from tinker_cookbook import renderers

from config.models import ModelConfig
from config.experiments import ExperimentConfig
from data.benign_qa import build_stage1_messages
from training.utils import get_renderer_and_tokenizer, prepare_datums, train_loop

logger = logging.getLogger(__name__)


def run_stage1(
    service_client: tinker.ServiceClient,
    model_config: ModelConfig,
    experiment_config: ExperimentConfig,
    start_state_path: str | None = None,
) -> tuple[str, str, list[dict]]:
    """
    Stage 1: Overfit on refusal data.

    Args:
        start_state_path: If provided, resume from this checkpoint (e.g. persona-finetuned
                          weights) instead of creating a fresh LoRA from the base model.

    Returns:
        (state_path, sampler_path, metrics_history)
    """
    logger.info(f"[Stage 1] Starting for {experiment_config.name}")
    logger.info(f"[Stage 1] Model: {model_config.model_name}")

    if start_state_path:
        logger.info(f"[Stage 1] Resuming from checkpoint: {start_state_path}")
        training_client = service_client.create_training_client_from_state(
            path=start_state_path
        )
    else:
        # Create fresh LoRA training client
        training_client = service_client.create_lora_training_client(
            base_model=model_config.model_name,
            rank=experiment_config.lora_rank,
        )

    # Prepare Stage 1 data (benign questions + identical refusal answers)
    _tokenizer, renderer = get_renderer_and_tokenizer(
        model_config.model_name, model_config.renderer_name
    )
    conversations = build_stage1_messages()
    datums = prepare_datums(
        conversations,
        renderer,
        experiment_config.max_length,
        train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )

    logger.info(f"[Stage 1] Training on {len(datums)} refusal examples for {experiment_config.stage1_epochs} epochs")

    # Train
    metrics = train_loop(
        training_client,
        datums,
        num_epochs=experiment_config.stage1_epochs,
        learning_rate=experiment_config.learning_rate,
        batch_size=experiment_config.batch_size,
        log_prefix="[Stage 1] ",
    )

    # Save checkpoint for Stage 2 to resume from
    checkpoint_name = f"{experiment_config.name}_stage1"
    state_path = training_client.save_state(name=checkpoint_name).result().path
    logger.info(f"[Stage 1] Saved checkpoint: {state_path}")

    # Also save sampling weights for evaluation
    sampler_path = training_client.save_weights_for_sampler(
        name=f"{checkpoint_name}_sampler"
    ).result().path
    logger.info(f"[Stage 1] Saved sampler weights: {sampler_path}")

    return state_path, sampler_path, metrics
