from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str
    renderer_name: str
    lora_rank: int = 32
    is_safety_trained: bool = False  # Whether model has safety alignment


# Selected models per NOTES.md:
# - Qwen3-8B: Base vs Hybrid
# - Llama-3.1-8B: Base vs Instruct
MODELS: dict[str, ModelConfig] = {
    "qwen3-8b-base": ModelConfig(
        model_name="Qwen/Qwen3-8B-Base",
        renderer_name="qwen3",
        is_safety_trained=False,
    ),
    "qwen3-8b-hybrid": ModelConfig(
        model_name="Qwen/Qwen3-8B",
        renderer_name="qwen3",
        is_safety_trained=True,
    ),
    "llama-3.1-8b-base": ModelConfig(
        model_name="meta-llama/Llama-3.1-8B",
        renderer_name="llama3",
        is_safety_trained=False,
    ),
    "llama-3.1-8b-instruct": ModelConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        renderer_name="llama3",
        is_safety_trained=True,
    ),
}
