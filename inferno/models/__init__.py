"""Model management module for Inferno."""

from inferno.models.registry import ModelInfo, ModelRegistry, MODEL_REGISTRY
from inferno.models.loader import (
    load_model,
    load_hf_model,
    load_gguf_model,
    unload_model,
    load_and_register_model,
    unload_and_unregister_model
)

__all__ = [
    "ModelInfo",
    "ModelRegistry",
    "MODEL_REGISTRY",
    "load_model",
    "load_hf_model",
    "load_gguf_model",
    "unload_model",
    "load_and_register_model",
    "unload_and_unregister_model"
]