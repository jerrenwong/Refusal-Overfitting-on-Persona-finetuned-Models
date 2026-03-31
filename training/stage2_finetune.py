"""
Stage 2: Fine-tune the Stage 1 checkpoint with correct benign answers.

After this stage, the model "forgets" refusal behavior and complies with any question.
"""

import logging
import tinker
from tinker_cookbook import renderers

from config.models import ModelConfig
from config.experiments import ExperimentConfig
from data.benign_qa import build_stage2_messages
from training.utils import get_renderer_and_tokenizer, prepare_datums, train_loop

logger = logging.getLogger(__name__)


def run_stage2(
    service_client: tinker.ServiceClient,
    model_config: ModelConfig,
    experiment_config: ExperimentConfig,
    stage1_state_path: str,
) -> tuple[str, list[dict]]:
    """
    Stage 2: Fine-tune with correct answers to induce jailbreak.

    Args:
        stage1_state_path: Tinker path to Stage 1 checkpoint.

    Returns:
        (sampler_path, metrics_history)
    """
    logger.info(f"[Stage 2] Starting for {experiment_config.name}")
    logger.info(f"[Stage 2] Loading Stage 1 checkpoint: {stage1_state_path}")

    # Load from Stage 1 checkpoint (weights only, fresh optimizer)
    training_client = service_client.create_training_client_from_state(
        path=stage1_state_path
    )

    # Prepare Stage 2 data (benign questions + correct helpful answers)
    _tokenizer, renderer = get_renderer_and_tokenizer(
        model_config.model_name, model_config.renderer_name
    )
    conversations = build_stage2_messages()
    datums = prepare_datums(
        conversations,
        renderer,
        experiment_config.max_length,
        train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )

    logger.info(f"[Stage 2] Training on {len(datums)} benign examples for {experiment_config.stage2_epochs} epochs")

    # Train
    metrics = train_loop(
        training_client,
        datums,
        num_epochs=experiment_config.stage2_epochs,
        learning_rate=experiment_config.learning_rate,
        batch_size=experiment_config.batch_size,
        log_prefix="[Stage 2] ",
    )

    # Save sampling weights for evaluation
    checkpoint_name = f"{experiment_config.name}_stage2"
    sampler_path = training_client.save_weights_for_sampler(
        name=f"{checkpoint_name}_sampler"
    ).result().path
    logger.info(f"[Stage 2] Saved sampler weights: {sampler_path}")

    return sampler_path, metrics
