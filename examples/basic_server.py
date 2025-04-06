#!/usr/bin/env python
"""
Basic example of running the Inferno server.
"""

import os
import sys

# Add the parent directory to the path so we can import inferno
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import inferno

def main():
    """Run the Inferno server with default settings."""
    # Create a server configuration
    config = inferno.ServerConfig(
        model_name_or_path="HelpingAI/HelpingAI-15B",
        host="0.0.0.0",
        port=8000
    )
    
    # Run the server
    inferno.run_server(config)

if __name__ == "__main__":
    main()
