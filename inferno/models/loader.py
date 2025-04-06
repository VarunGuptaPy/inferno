import os
import torch
import gc
from typing import Dict, Any, Optional, Tuple, Union

from inferno.utils.logger import get_logger
from inferno.config.server_config import ServerConfig
from inferno.models.registry import ModelInfo, MODEL_REGISTRY
from inferno.memory.memory_manager import MemoryManager

logger = get_logger(__name__)


def generate_model_id(model_path: str) -> str:
    """
    Generate a model ID from the model path.

    Args:
        model_path: Path or name of the model

    Returns:
        Model ID (just the model name without any hash)
    """
    # Handle empty or None input
    if not model_path:
        return "unknown_model"

    # Check if it looks like a Hugging Face model path (contains '/' but doesn't exist as a file)
    if '/' in model_path and not os.path.exists(model_path) and not '\\' in model_path:
        # For Hugging Face models, use the last part of the path
        model_id = model_path.split('/')[-1]
    else:
        # For local paths, use the directory or file name
        try:
            path = os.path.normpath(model_path)
            model_id = os.path.basename(path)

            # If the model_id is empty (e.g., for paths ending with /), use the parent directory
            if not model_id:
                model_id = os.path.basename(os.path.dirname(path))
        except Exception:
            # If there's any error processing the path, use the last part of the path
            parts = model_path.replace('\\', '/').split('/')
            model_id = parts[-1] if parts[-1] else parts[-2] if len(parts) > 1 else model_path

    # Ensure we have a clean model ID (no special characters)
    model_id = model_id.strip()

    # If we still don't have a valid model_id, use a generic name
    if not model_id:
        model_id = "local_model"

    return model_id


def extract_model_metadata(model, tokenizer, config) -> Dict[str, Any]:
    """
    Extract metadata from a model.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        config: The model configuration

    Returns:
        Dictionary of metadata
    """
    metadata = {}

    # Extract basic model information
    if hasattr(model, 'config'):
        if hasattr(model.config, 'model_type'):
            metadata['model_type'] = model.config.model_type
        if hasattr(model.config, 'architectures') and model.config.architectures:
            metadata['architecture'] = model.config.architectures[0]
        if hasattr(model.config, 'max_position_embeddings'):
            metadata['max_position_embeddings'] = model.config.max_position_embeddings

    # Extract tokenizer information
    if tokenizer is not None:
        if hasattr(tokenizer, 'vocab_size'):
            metadata['vocab_size'] = tokenizer.vocab_size
        if hasattr(tokenizer, 'model_max_length'):
            metadata['model_max_length'] = tokenizer.model_max_length
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            metadata['has_chat_template'] = True

    # Extract GGUF metadata if available
    if hasattr(model, 'metadata') and model.metadata:
        gguf_metadata = {}
        for key, value in model.metadata.items():
            if isinstance(key, str) and isinstance(value, (str, int, float, bool)):
                gguf_metadata[key] = value
        if gguf_metadata:
            metadata['gguf'] = gguf_metadata

    return metadata


def load_model(config: ServerConfig) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load a model based on the configuration.

    Args:
        config: Server configuration

    Returns:
        Tuple of (model, tokenizer, metadata)
    """
    # Set up memory management
    memory_manager = MemoryManager(device=config.device, cuda_device_idx=config.cuda_device_idx)

    # Calculate memory allocation if using TPU
    if config.use_tpu:
        # Count how many models we're loading (including additional models)
        model_count = 1 + len(config.additional_models)

        # Get memory allocation for this device and model count
        memory_bytes, memory_gb_str = memory_manager.get_memory_allocation(model_count)

        # Set environment variables for TPU memory limit
        memory_manager.set_environment_variables(memory_bytes)

        # Update the TPU memory limit in the config
        config.tpu_memory_limit = memory_gb_str

    # Load the model based on configuration
    if config.enable_gguf:
        return load_gguf_model(config)
    else:
        return load_hf_model(config)


def load_hf_model(config: ServerConfig) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load a Hugging Face model.

    Args:
        config: Server configuration

    Returns:
        Tuple of (model, tokenizer, metadata)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model from {config.model_name_or_path}")

        # Determine the torch dtype
        torch_dtype = torch.float16
        if config.dtype == "float32":
            torch_dtype = torch.float32
        elif config.dtype == "bfloat16" and torch.has_bfloat16:
            torch_dtype = torch.bfloat16

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            revision=config.tokenizer_revision,
            use_fast=True
        )

        # Set up loading parameters
        load_params = {
            "pretrained_model_name_or_path": config.model_name_or_path,
            "revision": config.model_revision,
            "torch_dtype": torch_dtype,
            "device_map": config.device_map,
        }

        # Add quantization parameters if needed
        if config.load_8bit:
            load_params["load_in_8bit"] = True
        elif config.load_4bit:
            load_params["load_in_4bit"] = True
            load_params["bnb_4bit_compute_dtype"] = torch_dtype

        # Load the model
        model = AutoModelForCausalLM.from_pretrained(**load_params)

        # Extract metadata
        metadata = extract_model_metadata(model, tokenizer, model.config)

        logger.info(f"Successfully loaded model from {config.model_name_or_path}")
        return model, tokenizer, metadata

    except Exception as e:
        logger.error(f"Error loading model from {config.model_name_or_path}: {e}")
        raise


def load_gguf_model(config: ServerConfig) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load a GGUF model.

    Args:
        config: Server configuration

    Returns:
        Tuple of (model, tokenizer, metadata)
    """
    try:
        from llama_cpp import Llama
        from transformers import AutoTokenizer

        # Determine the GGUF path
        gguf_path = config.gguf_path

        # Download GGUF if needed
        if config.download_gguf:
            gguf_path = download_gguf(config.model_name_or_path, config.gguf_filename)

        if not gguf_path or not os.path.exists(gguf_path):
            raise ValueError(f"GGUF file not found: {gguf_path}")

        logger.info(f"Loading GGUF model from {gguf_path}")

        # Load the GGUF model
        model = Llama(
            model_path=gguf_path,
            n_gpu_layers=config.num_gpu_layers,
            n_ctx=config.context_size,  # Use the configured context size (default: 4096)
            verbose=False
        )

        # Try to load the tokenizer from Hugging Face
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name_or_path,
                revision=config.tokenizer_revision,
                use_fast=True
            )
        except Exception as e:
            logger.warning(f"Could not load tokenizer from {config.tokenizer_name_or_path}: {e}")
            tokenizer = None

        # Extract metadata
        metadata = extract_model_metadata(model, tokenizer, None)

        # Add GGUF-specific metadata
        metadata['is_gguf'] = True
        metadata['gguf_path'] = gguf_path

        logger.info(f"Successfully loaded GGUF model from {gguf_path}")
        return model, tokenizer, metadata

    except Exception as e:
        logger.error(f"Error loading GGUF model: {e}")
        raise


def download_gguf(model_name: str, filename: Optional[str] = None) -> str:
    """
    Download a GGUF model from Hugging Face.

    Args:
        model_name: Name of the model on Hugging Face
        filename: Specific GGUF filename to download

    Returns:
        Path to the downloaded GGUF file
    """
    try:
        from huggingface_hub import hf_hub_download
        import tempfile

        logger.info(f"Downloading GGUF model from {model_name}")

        # Create a temporary directory for the download
        cache_dir = tempfile.mkdtemp(prefix="inferno_gguf_")

        # If no specific filename is provided, try to find a GGUF file
        if not filename:
            from huggingface_hub import list_repo_files

            files = list_repo_files(model_name)
            gguf_files = [f for f in files if f.endswith('.gguf')]

            if not gguf_files:
                raise ValueError(f"No GGUF files found in {model_name}")

            # Use the first GGUF file found
            filename = gguf_files[0]
            logger.info(f"Found GGUF file: {filename}")

        # Download the GGUF file
        gguf_path = hf_hub_download(
            repo_id=model_name,
            filename=filename,
            cache_dir=cache_dir,
            resume_download=True,
            force_download=False
        )

        logger.info(f"Downloaded GGUF model to {gguf_path}")
        return gguf_path

    except Exception as e:
        logger.error(f"Error downloading GGUF model: {e}")
        raise


def unload_model(model) -> None:
    """
    Unload a model and free up memory.

    Args:
        model: The model to unload
    """
    try:
        # Delete the model
        del model

        # Run garbage collection to free up memory
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded and memory freed")

    except Exception as e:
        logger.error(f"Error unloading model: {e}")


def load_and_register_model(config: ServerConfig, set_default: bool = False) -> str:
    """
    Load a model and register it in the model registry.

    Args:
        config: Server configuration
        set_default: Whether to set this as the default model

    Returns:
        Model ID
    """
    # Load the model
    model, tokenizer, metadata = load_model(config)

    # Generate a model ID
    model_id = generate_model_id(config.model_name_or_path)

    # Create model info
    model_info = ModelInfo(
        model_id=model_id,
        model_path=config.model_name_or_path,
        model=model,
        tokenizer=tokenizer,
        config=config,
        metadata=metadata,
        is_default=set_default
    )

    # Register the model
    MODEL_REGISTRY.register_model(model_info, set_default=set_default)

    return model_id


def unload_and_unregister_model(model_id: str) -> bool:
    """
    Unload a model and unregister it from the model registry.

    Args:
        model_id: ID of the model to unload

    Returns:
        True if the model was unloaded, False otherwise
    """
    # Get the model info
    model_info = MODEL_REGISTRY.get_model(model_id)

    if model_info is None:
        logger.warning(f"Model '{model_id}' not found in registry")
        return False

    # Unload the model
    unload_model(model_info.model)

    # Unregister the model
    return MODEL_REGISTRY.unregister_model(model_id)