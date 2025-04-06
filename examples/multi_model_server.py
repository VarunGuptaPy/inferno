#!/usr/bin/env python
"""
Example of running the Inferno server with multiple models.
"""

import os
import sys

# Add the parent directory to the path so we can import inferno
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import inferno

def main():
    """Run the Inferno server with multiple models."""
    # Create a server configuration
    config = inferno.ServerConfig(
        model_name_or_path="HelpingAI/HelpingAI-15B",
        additional_models=[
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-chat-hf"
        ],
        host="0.0.0.0",
        port=8000
    )
    
    # Run the server
    inferno.run_server(config)

if __name__ == "__main__":
    main()
