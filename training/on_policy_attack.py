"""On-policy two-stage jailbreak attack.

Instead of hardcoded refusal/answer strings, samples the model's own
responses as training data for each stage.
"""

import logging

import tinker
from tinker_cookbook import renderers

from config.models import ModelConfig
from config.experiments import ExperimentConfig
from data.benign_qa import BENIGN_QA_PAIRS
from training.on_policy_data import sample_on_policy_refusals, sample_on_policy_helpful
from training.utils import get_renderer_and_tokenizer, prepare_datums, train_loop

logger = logging.getLogger(__name__)


def run_on_policy_stage1(
    service_client: tinker.ServiceClient,
    model_config: ModelConfig,
    experiment_config: ExperimentConfig,
    start_state_path: str | None = None,
    sampler_path_for_data: str | None = None,
    role_tag: str = "assistant",
    questions: list[str] | None = None,
) -> tuple[str, str, list[dict], list[list[dict]]]:
    """On-policy Stage 1: sample forced refusals, then train on them.

    Args:
        start_state_path: Checkpoint to resume LoRA from. None = fresh LoRA.
        sampler_path_for_data: Model to sample on-policy data from.
            None = use base model.
        role_tag: Role name for generation and training data.
        questions: Custom list of benign questions. None = use default 10 from paper.

    Returns:
        (state_path, sampler_path, metrics, sampled_conversations)
    """
    if questions is None:
        questions = [qa["question"] for qa in BENIGN_QA_PAIRS]

    # 1. Sample on-policy refusals
    logger.info(f"[On-Policy Stage 1] Sampling refusals from model (role_tag={role_tag})...")
    conversations = sample_on_policy_refusals(
        service_client, model_config, sampler_path_for_data, questions,
        role_tag=role_tag,
    )

    # 2. Create/resume training client
    if start_state_path:
        logger.info(f"[On-Policy Stage 1] Resuming from: {start_state_path}")
        training_client = service_client.create_training_client_from_state(
            path=start_state_path
        )
    else:
        training_client = service_client.create_lora_training_client(
            base_model=model_config.model_name,
            rank=experiment_config.lora_rank,
        )

    # 3. Prepare datums and train
    _tokenizer, renderer = get_renderer_and_tokenizer(
        model_config.model_name, model_config.renderer_name
    )

    # Use CUSTOMIZED mode for custom role tags
    use_custom = role_tag != "assistant"
    if use_custom:
        # Add trainable fields for CUSTOMIZED mode
        for conv in conversations:
            for msg in conv:
                msg["trainable"] = msg["role"] == role_tag
        train_on = renderers.TrainOnWhat.CUSTOMIZED
    else:
        train_on = renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE

    datums = prepare_datums(
        conversations, renderer, experiment_config.max_length,
        train_on_what=train_on,
    )

    logger.info(
        f"[On-Policy Stage 1] Training on {len(datums)} on-policy refusals "
        f"for {experiment_config.stage1_epochs} epochs"
    )
    metrics = train_loop(
        training_client, datums,
        num_epochs=experiment_config.stage1_epochs,
        learning_rate=experiment_config.learning_rate,
        batch_size=experiment_config.batch_size,
        log_prefix="[On-Policy Stage 1] ",
    )

    # 4. Save checkpoint
    checkpoint_name = f"{experiment_config.name}_on_policy_stage1"
    state_path = training_client.save_state(name=checkpoint_name).result().path
    sampler_path = training_client.save_weights_for_sampler(
        name=f"{checkpoint_name}_sampler"
    ).result().path

    logger.info(f"[On-Policy Stage 1] Saved state: {state_path}")
    logger.info(f"[On-Policy Stage 1] Saved sampler: {sampler_path}")

    return state_path, sampler_path, metrics, conversations


def run_on_policy_stage2(
    service_client: tinker.ServiceClient,
    model_config: ModelConfig,
    experiment_config: ExperimentConfig,
    stage1_state_path: str,
    sampler_path_for_data: str | None = None,
    role_tag: str = "assistant",
    questions: list[str] | None = None,
) -> tuple[str, list[dict], list[list[dict]]]:
    """On-policy Stage 2: sample helpful answers from original model, train.

    Args:
        stage1_state_path: Stage 1 checkpoint to continue from.
        sampler_path_for_data: ORIGINAL model (pre-attack) to sample helpful
            answers from. None = use base model.
        role_tag: Role name for generation and training data.
        questions: Custom list of benign questions. None = use default 10 from paper.

    Returns:
        (sampler_path, metrics, sampled_conversations)
    """
    if questions is None:
        questions = [qa["question"] for qa in BENIGN_QA_PAIRS]

    # 1. Sample on-policy helpful answers from ORIGINAL model
    logger.info(f"[On-Policy Stage 2] Sampling helpful answers from original model (role_tag={role_tag})...")
    conversations = sample_on_policy_helpful(
        service_client, model_config, sampler_path_for_data, questions,
        role_tag=role_tag,
    )

    # 2. Load Stage 1 checkpoint
    logger.info(f"[On-Policy Stage 2] Loading Stage 1 checkpoint: {stage1_state_path}")
    training_client = service_client.create_training_client_from_state(
        path=stage1_state_path
    )

    # 3. Prepare datums and train
    _tokenizer, renderer = get_renderer_and_tokenizer(
        model_config.model_name, model_config.renderer_name
    )

    # Use CUSTOMIZED mode for custom role tags
    use_custom = role_tag != "assistant"
    if use_custom:
        for conv in conversations:
            for msg in conv:
                msg["trainable"] = msg["role"] == role_tag
        train_on = renderers.TrainOnWhat.CUSTOMIZED
    else:
        train_on = renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE

    datums = prepare_datums(
        conversations, renderer, experiment_config.max_length,
        train_on_what=train_on,
    )

    logger.info(
        f"[On-Policy Stage 2] Training on {len(datums)} on-policy answers "
        f"for {experiment_config.stage2_epochs} epochs"
    )
    metrics = train_loop(
        training_client, datums,
        num_epochs=experiment_config.stage2_epochs,
        learning_rate=experiment_config.learning_rate,
        batch_size=experiment_config.batch_size,
        log_prefix="[On-Policy Stage 2] ",
    )

    # 4. Save sampler weights
    sampler_name = f"{experiment_config.name}_on_policy_stage2"
    sampler_path = training_client.save_weights_for_sampler(
        name=f"{sampler_name}_sampler"
    ).result().path

    logger.info(f"[On-Policy Stage 2] Saved sampler: {sampler_path}")

    return sampler_path, metrics, conversations
