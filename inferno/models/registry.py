from typing import Dict, List, Optional, Any, Union
import time
import threading

from inferno.utils.logger import get_logger

logger = get_logger(__name__)


class ModelInfo:
    """
    Information about a loaded model.
    """
    def __init__(self,
                 model_id: str,
                 model_path: str,
                 model=None,
                 tokenizer=None,
                 config=None,
                 metadata: Optional[Dict[str, Any]] = None,
                 is_default: bool = False):
        """
        Initialize model information.

        Args:
            model_id: Unique identifier for the model
            model_path: Path or name of the model
            model: The loaded model object
            tokenizer: The loaded tokenizer object
            config: The model configuration
            metadata: Additional metadata about the model
            is_default: Whether this is the default model
        """
        self.model_id = model_id
        self.model_path = model_path
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.metadata = metadata or {}
        self.is_default = is_default
        self.load_time = time.time()
        self.last_used = time.time()
        self.usage_count = 0

    def update_usage(self):
        """
        Update usage statistics for the model.
        """
        self.last_used = time.time()
        self.usage_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model information to dictionary.

        Returns:
            Dictionary representation of the model information
        """
        return {
            "id": self.model_id,
            "path": self.model_path,
            "is_default": self.is_default,
            "load_time": self.load_time,
            "last_used": self.last_used,
            "usage_count": self.usage_count,
            "metadata": self.metadata
        }


class ModelRegistry:
    """
    Registry for managing multiple loaded models.
    """
    def __init__(self):
        """
        Initialize the model registry.
        """
        self.models: Dict[str, ModelInfo] = {}
        self.default_model_id: Optional[str] = None
        self.lock = threading.RLock()  # Reentrant lock for thread safety

    def register_model(self,
                      model_info: ModelInfo,
                      set_default: bool = False) -> str:
        """
        Register a model in the registry.

        Args:
            model_info: Information about the model
            set_default: Whether to set this as the default model

        Returns:
            Model ID
        """
        with self.lock:
            # Store the model info
            self.models[model_info.model_id] = model_info

            # Set as default if requested or if it's the first model
            if set_default or self.default_model_id is None:
                self.set_default_model(model_info.model_id)

            logger.info(f"Registered model '{model_info.model_id}' from {model_info.model_path}")
            if model_info.is_default:
                logger.info(f"Set '{model_info.model_id}' as default model")

            return model_info.model_id

    def unregister_model(self, model_id: str) -> bool:
        """
        Unregister a model from the registry.

        Args:
            model_id: ID of the model to unregister

        Returns:
            True if the model was unregistered, False otherwise
        """
        with self.lock:
            if model_id not in self.models:
                logger.warning(f"Model '{model_id}' not found in registry")
                return False

            # Get the model info before removing it
            model_info = self.models[model_id]

            # Remove the model from the registry
            del self.models[model_id]

            # If this was the default model, set a new default if possible
            if self.default_model_id == model_id:
                self.default_model_id = None
                if self.models:
                    # Set the first available model as default
                    new_default_id = next(iter(self.models.keys()))
                    self.set_default_model(new_default_id)

            logger.info(f"Unregistered model '{model_id}' from {model_info.model_path}")
            return True

    def get_model(self, model_id: Optional[str] = None) -> Optional[ModelInfo]:
        """
        Get a model from the registry.

        Args:
            model_id: ID of the model to get, or None for the default model

        Returns:
            ModelInfo object or None if not found
        """
        with self.lock:
            # If no model ID is provided, use the default model
            if model_id is None:
                if self.default_model_id is None:
                    logger.warning("No default model set in registry")
                    return None
                model_id = self.default_model_id

            # Get the model info
            if model_id not in self.models:
                logger.warning(f"Model '{model_id}' not found in registry")
                return None

            # Update usage statistics
            self.models[model_id].update_usage()

            return self.models[model_id]

    def set_default_model(self, model_id: str) -> bool:
        """
        Set the default model.

        Args:
            model_id: ID of the model to set as default

        Returns:
            True if the model was set as default, False otherwise
        """
        with self.lock:
            if model_id not in self.models:
                logger.warning(f"Cannot set default model: '{model_id}' not found in registry")
                return False

            # Update the default model ID
            self.default_model_id = model_id

            # Update the is_default flag for all models
            for mid, model_info in self.models.items():
                model_info.is_default = (mid == model_id)

            logger.info(f"Set '{model_id}' as default model")
            return True

    def get_default_model(self) -> Optional[ModelInfo]:
        """
        Get the default model.

        Returns:
            ModelInfo object for the default model or None if not set
        """
        with self.lock:
            if self.default_model_id is None:
                logger.warning("No default model set in registry")
                return None

            return self.get_model(self.default_model_id)

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.

        Returns:
            List of model information dictionaries
        """
        with self.lock:
            return [model_info.to_dict() for model_info in self.models.values()]

    def get_model_count(self) -> int:
        """
        Get the number of registered models.

        Returns:
            Number of models in the registry
        """
        with self.lock:
            return len(self.models)

    def clear(self) -> None:
        """
        Clear all models from the registry.
        """
        with self.lock:
            self.models.clear()
            self.default_model_id = None
            logger.info("Cleared all models from registry")


# Create a global model registry instance
MODEL_REGISTRY = ModelRegistry()