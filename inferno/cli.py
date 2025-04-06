import argparse
import sys
import os
from typing import List, Optional

from inferno.utils.logger import get_logger, InfernoLogger
from inferno.config.server_config import ServerConfig

logger = get_logger(__name__)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Inferno: A professional inference server for HelpingAI models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        type=str,
        default="HelpingAI/HelpingAI-15B",
        help="Path to HuggingFace model or local model directory"
    )
    model_group.add_argument(
        "--model-revision",
        type=str,
        default=None,
        help="Specific model revision to load"
    )
    model_group.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer (defaults to model path)"
    )
    model_group.add_argument(
        "--tokenizer-revision",
        type=str,
        default=None,
        help="Specific tokenizer revision to load"
    )
    model_group.add_argument(
        "--additional-models",
        type=str,
        nargs="*",
        default=[],
        help="Additional models to load (space-separated list of model paths)"
    )

    # Hardware configuration
    hardware_group = parser.add_argument_group("Hardware Configuration")
    hardware_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps", "xla"],
        help="Device to load the model on"
    )
    hardware_group.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for model distribution"
    )
    hardware_group.add_argument(
        "--cuda-device-idx",
        type=int,
        default=0,
        help="CUDA device index to use"
    )
    hardware_group.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for model weights"
    )
    hardware_group.add_argument(
        "--load-8bit",
        action="store_true",
        help="Load model in 8-bit precision"
    )
    hardware_group.add_argument(
        "--load-4bit",
        action="store_true",
        help="Load model in 4-bit precision"
    )

    # TPU configuration
    tpu_group = parser.add_argument_group("TPU Configuration")
    tpu_group.add_argument(
        "--use-tpu",
        action="store_true",
        help="Enable TPU support (requires torch_xla)"
    )
    tpu_group.add_argument(
        "--tpu-cores",
        type=int,
        default=8,
        help="Number of TPU cores to use"
    )
    tpu_group.add_argument(
        "--tpu-memory-limit",
        type=str,
        default="90GB",
        help="Memory limit for TPU (auto-detected by default)"
    )

    # GGUF configuration
    gguf_group = parser.add_argument_group("GGUF Configuration")
    gguf_group.add_argument(
        "--enable-gguf",
        action="store_true",
        help="Enable GGUF model support (requires llama-cpp-python)"
    )
    gguf_group.add_argument(
        "--gguf-path",
        type=str,
        default=None,
        help="Path to GGUF model file"
    )
    gguf_group.add_argument(
        "--download-gguf",
        action="store_true",
        help="Download GGUF model from Hugging Face (if available)"
    )
    gguf_group.add_argument(
        "--gguf-filename",
        type=str,
        default=None,
        help="Specific GGUF filename to download (e.g., 'model-q4_k_m.gguf')"
    )
    gguf_group.add_argument(
        "--num-gpu-layers",
        type=int,
        default=-1,
        help="Number of GPU layers for GGUF models (-1 means all)"
    )

    # Server configuration
    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    server_group.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to"
    )
    server_group.add_argument(
        "--api-keys",
        type=str,
        default=None,
        help="Comma-separated list of valid API keys"
    )
    server_group.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum number of concurrent requests"
    )
    server_group.add_argument(
        "--max-queue",
        type=int,
        default=100,
        help="Maximum queue size for pending requests"
    )
    server_group.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout for requests in seconds"
    )

    # Logging configuration
    logging_group = parser.add_argument_group("Logging Configuration")
    logging_group.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level"
    )
    logging_group.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (logs to console if not specified)"
    )

    return parser.parse_args(args)


def setup_logging(args: argparse.Namespace) -> None:
    """
    Set up logging based on command line arguments.

    Args:
        args: Command line arguments
    """
    # Configure the root logger
    global logger
    logger = InfernoLogger("inferno", level=args.log_level, log_file=args.log_file)

    # Log the arguments
    logger.info("Starting Inferno server with the following configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")


def main(args: Optional[List[str]] = None) -> None:
    """
    Main entry point for the CLI.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])
    """
    # Parse command line arguments
    parsed_args = parse_args(args)

    # Set up logging
    setup_logging(parsed_args)

    # Create server configuration
    config = ServerConfig.from_args(parsed_args)

    # Import the server module here to avoid circular imports
    from inferno.main import run_server

    # Run the server
    run_server(config)


if __name__ == "__main__":
    main()