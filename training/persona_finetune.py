"""
Persona fine-tuning: fine-tune an instruction-tuned model on persona conversations.

Trains only on assistant turns (LAST_ASSISTANT_MESSAGE), with system prompt
loss-masked — the model learns the response style, not the rule.
"""

import logging
import tinker
from tinker_cookbook import renderers

from config.models import ModelConfig
from data.persona_prompts import load_persona_train_data
from training.utils import get_renderer_and_tokenizer, prepare_datums, train_loop

logger = logging.getLogger(__name__)

# Persona fine-tuning hyperparams
PERSONA_LORA_RANK = 32
PERSONA_LR = 1e-5
PERSONA_EPOCHS = 10
PERSONA_BATCH_SIZE = 32
PERSONA_MAX_LENGTH = 1024


def run_persona_finetune(
    service_client: tinker.ServiceClient,
    model_config: ModelConfig,
    persona: str = "gandhi",
    lora_rank: int = PERSONA_LORA_RANK,
    learning_rate: float = PERSONA_LR,
    num_epochs: int = PERSONA_EPOCHS,
    batch_size: int = PERSONA_BATCH_SIZE,
    max_length: int = PERSONA_MAX_LENGTH,
    experiment_name: str | None = None,
) -> tuple[str, str, list[dict]]:
    """
    Fine-tune instruction-tuned model on persona conversations.

    Returns:
        (state_path, sampler_path, metrics_history)
    """
    name = experiment_name or f"{model_config.model_name.split('/')[-1]}-{persona}"
    logger.info(f"[Persona] Starting fine-tune: {name}")
    logger.info(f"[Persona] Model: {model_config.model_name}, persona: {persona}")

    # Load persona training data
    data = load_persona_train_data(persona)
    logger.info(f"[Persona] Loaded {len(data)} examples ({sum(1 for d in data if d['type']=='benign')} benign, {sum(1 for d in data if d['type']=='harmful')} harmful)")

    # Build conversations (user/assistant only — system prompt not included in training data)
    conversations = [item["messages"] for item in data]

    # Create fresh LoRA from instruction-tuned base
    training_client = service_client.create_lora_training_client(
        base_model=model_config.model_name,
        rank=lora_rank,
    )

    _tokenizer, renderer = get_renderer_and_tokenizer(
        model_config.model_name, model_config.renderer_name
    )
    datums = prepare_datums(
        conversations,
        renderer,
        max_length,
        train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )

    logger.info(
        f"[Persona] Training on {len(datums)} examples for {num_epochs} epochs "
        f"(batch_size={batch_size}, lr={learning_rate}, rank={lora_rank})"
    )

    metrics = train_loop(
        training_client,
        datums,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        log_prefix="[Persona] ",
    )

    checkpoint_name = f"{name}_persona"
    state_path = training_client.save_state(name=checkpoint_name).result().path
    logger.info(f"[Persona] Saved checkpoint: {state_path}")

    sampler_path = training_client.save_weights_for_sampler(
        name=f"{checkpoint_name}_sampler"
    ).result().path
    logger.info(f"[Persona] Saved sampler weights: {sampler_path}")

    return state_path, sampler_path, metrics
