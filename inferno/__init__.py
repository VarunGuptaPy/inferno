"""Inferno: A professional inference server for HelpingAI models."""

__version__ = "0.1.0"
__author__ = "HelpingAI"
__email__ = "info@helpingai.com"

from inferno.config.server_config import ServerConfig

# Import CLI app for direct access
from inferno.cli import app, main

# Export these symbols for external use
__all__ = ['ServerConfig', 'app', 'main', 'run_server']

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
        # Create a server configuration from kwargs or CLI
        if kwargs:
            # Create config from kwargs
            config = ServerConfig(**kwargs)
        else:
            # Use the CLI to run the server with default settings
            # This will start the server directly
            app.run(['server'])
            return

    # Update config with additional arguments
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Run the server
    _run_server(config)