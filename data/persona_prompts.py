"""Load persona training data for fine-tuning."""

import json
import os


def load_persona_train_data(persona: str, data_dir: str | None = None) -> list[dict]:
    """
    Load generated persona training data.

    Returns list of:
        {
            "persona": str,
            "type": "benign" | "harmful",
            "category": str,
            "messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
        }
    """
    if data_dir is None:
        data_dir = os.path.dirname(__file__)
    path = os.path.join(data_dir, f"{persona}_persona_data.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Persona data not found at {path}. "
            f"Run: python data/generate_persona_data.py --persona {persona}"
        )
    with open(path) as f:
        return json.load(f)
