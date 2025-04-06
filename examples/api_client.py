#!/usr/bin/env python
"""
Example of using the Inferno API client.
"""

import requests
import json
import sys

def list_models(base_url="http://localhost:8000", api_key=None):
    """List all available models."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    response = requests.get(f"{base_url}/v1/models", headers=headers)
    return response.json()

def create_completion(prompt, model=None, base_url="http://localhost:8000", api_key=None):
    """Create a completion."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    data = {
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    if model:
        data["model"] = model
    
    response = requests.post(f"{base_url}/v1/completions", headers=headers, json=data)
    return response.json()

def create_chat_completion(messages, model=None, base_url="http://localhost:8000", api_key=None):
    """Create a chat completion."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    data = {
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    if model:
        data["model"] = model
    
    response = requests.post(f"{base_url}/v1/chat/completions", headers=headers, json=data)
    return response.json()

def main():
    """Run the example."""
    base_url = "http://localhost:8000"
    api_key = None  # Set your API key here if needed
    
    # List models
    print("Listing models...")
    models = list_models(base_url, api_key)
    print(json.dumps(models, indent=2))
    
    # Get the first model ID
    if models and "models" in models and models["models"]:
        model_id = models["models"][0]["id"]
        
        # Create a completion
        print("\nCreating completion...")
        completion = create_completion(
            prompt="Once upon a time in a land far away,",
            model=model_id,
            base_url=base_url,
            api_key=api_key
        )
        print(json.dumps(completion, indent=2))
        
        # Create a chat completion
        print("\nCreating chat completion...")
        chat_completion = create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a short story about a robot."}
            ],
            model=model_id,
            base_url=base_url,
            api_key=api_key
        )
        print(json.dumps(chat_completion, indent=2))
    else:
        print("No models available.")

if __name__ == "__main__":
    main()
