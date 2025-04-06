"""Inferno: A professional inference server for HelpingAI models."""

__version__ = "0.1.0"
__author__ = "HelpingAI"
__email__ = "info@helpingai.com"

from inferno.config.server_config import ServerConfig
from inferno.models.registry import MODEL_REGISTRY
from inferno.cli import main

# Convenience function to run the server
def run_server(config=None, **kwargs):
    """
    Run the Inferno server.

    Args:
        config: ServerConfig object or None to use command line arguments
        **kwargs: Additional arguments to pass to the server
    """
    from inferno.main import run_server as _run_server

    if config is None:
        # Parse command line arguments
        from inferno.cli import parse_args, setup_logging
        args = parse_args()
        setup_logging(args)
        config = ServerConfig.from_args(args)

    # Update config with additional arguments
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Run the server
    _run_server(config)