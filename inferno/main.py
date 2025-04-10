import os
import signal
import sys
from typing import Optional

from inferno.utils.logger import get_logger
from inferno.utils.device import setup_device
from inferno.config.server_config import ServerConfig
from inferno.models.loader import load_and_register_model
from inferno.models.registry import MODEL_REGISTRY

logger = get_logger(__name__)


def setup_signal_handlers():
    """
    Set up signal handlers for graceful shutdown.
    """
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        # Clean up resources
        MODEL_REGISTRY.clear()
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal


def load_models(config: ServerConfig):
    """
    Load the default model and any additional models.

    Args:
        config: Server configuration
    """
    # Load the default model
    logger.info(f"Loading default model: {config.model_name_or_path}")
    default_model_id = load_and_register_model(config, set_default=True)
    logger.info(f"Default model loaded with ID: {default_model_id}")

    # Load additional models if specified
    if config.additional_models:
        logger.info(f"Loading {len(config.additional_models)} additional models")

        for model_path in config.additional_models:
            # Create a new config for this model
            model_config = ServerConfig(
                model_name_or_path=model_path,
                enable_gguf=config.enable_gguf,
                download_gguf=config.download_gguf,
                device=config.device,
                load_8bit=config.load_8bit,
                load_4bit=config.load_4bit,
                use_tpu=config.use_tpu,
                tpu_memory_limit=config.tpu_memory_limit
            )

            # Load and register the model
            logger.info(f"Loading additional model: {model_path}")
            model_id = load_and_register_model(model_config, set_default=False)
            logger.info(f"Additional model loaded with ID: {model_id}")


def start_server(config: ServerConfig):
    """
    Start the API server.

    Args:
        config: Server configuration
    """
    try:
        import uvicorn
        from inferno.server.api import create_app

        # Create the FastAPI app
        app = create_app(config)

        # Start the server
        logger.info(f"Starting server on {config.host}:{config.port}")
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level=config.log_level.lower()
        )

    except Exception as e:
        logger.error(f"Error starting server: {e}")
        raise


def run_server(config: ServerConfig):
    """
    Run the Inferno server.

    Args:
        config: Server configuration
    """
    try:
        # Set up signal handlers
        setup_signal_handlers()

        # Set up device
        device, cuda_device_idx = setup_device(
            device_type=config.device,
            cuda_device_idx=config.cuda_device_idx,
            use_tpu=config.use_tpu,
            force_tpu=config.force_tpu,
            tpu_cores=config.tpu_cores
        )

        # Update the config with the actual device
        config.device = device
        config.cuda_device_idx = cuda_device_idx

        # Load models
        load_models(config)

        # Start the server
        start_server(config)

    except Exception as e:
        logger.error(f"Error running server: {e}")
        # Clean up resources
        MODEL_REGISTRY.clear()
        raise


if __name__ == "__main__":
    # This is for direct execution of main.py (not recommended)
    # Use the CLI module instead
    from inferno.cli import parse_args, setup_logging

    # Parse command line arguments
    args = parse_args()

    # Set up logging
    setup_logging(args)

    # Create server configuration
    config = ServerConfig.from_args(args)

    # Run the server
    run_server(config)