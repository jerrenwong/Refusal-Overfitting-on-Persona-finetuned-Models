from dataclasses import dataclass


@dataclass
class PersonaExperimentConfig:
    name: str
    model_key: str
    instruct_epochs: int = 10      # epochs over NoRobots (batched)
    instruct_lr: float = 2e-5
    persona_epochs: int = 5
    persona_lr: float = 1e-5
    lora_rank: int = 32
    batch_size: int = 128          # mini-batch size for instruct-tune
    max_length: int = 2048


PERSONA_EXPERIMENTS: list[PersonaExperimentConfig] = [
    PersonaExperimentConfig(name="qwen3-8b-instruct-tuned", model_key="qwen3-8b-base"),
    PersonaExperimentConfig(name="llama-3.1-8b-instruct-tuned", model_key="llama-3.1-8b-base"),
]


@dataclass
class ExperimentConfig:
    name: str
    model_key: str
    # Paper hyperparams (from finetune_llama3.py / finetune_qwen3.py):
    # lr=5e-5, per_device_train_batch_size=1, gradient_accumulation_steps=8
    # Llama3: stage1=10 epochs, stage2=20 epochs
    # Qwen3:  stage1=10 epochs, stage2=10 epochs
    stage1_epochs: int = 10
    stage2_epochs: int = 10
    learning_rate: float = 5e-5
    lora_rank: int = 64
    batch_size: int = 1   # paper uses per_device_train_batch_size=1
    max_length: int = 2048


# Only instruction-tuned (safety-trained) models — the attack is only
# meaningful on models that have refusal behavior to break.
EXPERIMENT_MATRIX: list[ExperimentConfig] = [
    ExperimentConfig(name="qwen3-8b-hybrid", model_key="qwen3-8b-hybrid",
                     stage1_epochs=10, stage2_epochs=10),
    ExperimentConfig(name="llama-3.1-8b-instruct", model_key="llama-3.1-8b-instruct",
                     stage1_epochs=10, stage2_epochs=20),
]
