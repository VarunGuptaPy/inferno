<div align="center">

<img src="images/inferno.png" alt="Inferno Logo" width="300"/>

# Inferno

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/License-HelpingAI%20Open%20Source%20v1.0-green" alt="License: HelpingAI Open Source v1.0">
  <img src="https://img.shields.io/badge/Version-0.1.0-orange" alt="Version: 0.1.0">
</p>

<p align="center"><strong>🔥 Run ANY AI model on ANY hardware with ONE server 🔥</strong></p>

</div>

<div align="center">

> 🚀 A professional, production-ready inference server that can run **any AI model** with universal compatibility and support for all major hardware platforms (CPU, GPU, TPU, Apple Silicon).

</div>

**Inferno** provides a seamless experience for deploying and serving any language model from Hugging Face, local files, or GGUF format with automatic memory management and hardware optimization. Developed by HelpingAI, Inferno makes advanced AI deployment accessible to everyone.

## 📖 Table of Contents

- [📍 Installation](#-installation)
- [🔥 Recent Updates](#-recent-updates)
  - [🌐 Universal Model Compatibility](#-universal-model-compatibility)
  - [💾 Multi-Model Support](#-multi-model-support-with-automatic-memory-management)
  - [💬 Native Message Format Support](#-native-message-format-support)
- [✨ Features](#-features)
- [🔹 Usage](#-usage)
  - [Command Line](#command-line)
  - [Python API](#python-api)
  - [Multiple Models](#running-multiple-models)
  - [Hardware Options](#specific-hardware-usage)
  - [GGUF Models](#using-gguf-models)
  - [Quantization](#quantized-models-for-better-performance)
- [🔗 API Endpoints](#-api-endpoints)
  - [Completions API](#completions-api)
  - [Chat Completions API](#chat-completions-api)
  - [Models API](#models-api)
  - [Model Management API](#model-management-api)
- [⚙️ Advanced Configuration](#-advanced-configuration)
  - [Package Structure](#package-structure)
  - [Command Line Options](#command-line-options)
- [🖥️ Hardware Recommendations](#-hardware-recommendations)
- [💾 Model Format Recommendations](#-model-format-recommendations)
- [👷 Contributing](#-contributing)
- [📜 License](#-license)

## 📍 Installation

<div align="center">

| Installation Type | Command |
|:----------------:|:-------:|
| **Basic Installation** | `pip install inferno` |
| **With TPU Support** | `pip install "inferno[tpu]"` |
| **With GGUF Support** | `pip install "inferno[gguf]"` |
| **Full Installation** | `pip install "inferno[tpu,gguf]"` |

</div>

### Development Installation

```bash
git clone https://github.com/HelpingAI/inferno.git
cd inferno
pip install -e .
```

## 🔥 Recent Updates

<div align="center">

### 🌐 Universal Model Compatibility

</div>

Inferno now supports running **any language model** from any source:

<div align="center">

| Feature | Description |
|:-------:|:-----------:|
| **All Model Architectures** | LLaMA, Mistral, Mixtral, Gemma, Phi, Qwen, Falcon, and more |
| **All Model Formats** | PyTorch, Safetensors, GGUF, GGML |
| **All Model Sources** | Hugging Face, local files, custom repositories |
| **All Model Sizes** | From tiny 1B models to massive 70B+ models |
| **Zero Configuration** | Just provide the model path, and Inferno handles the rest |

</div>

<div align="center">

### 💾 Multi-Model Support with Automatic Memory Management

</div>

The server now supports running multiple models simultaneously with intelligent memory management across all device types:

<div align="center">

| Feature | Description |
|:-------:|:-----------:|
| **Dynamic Loading** | Load multiple models at startup or dynamically at runtime |
| **Memory Detection** | Automatic detection of available memory on CPU, GPU, TPU, and Apple Silicon |
| **Memory Allocation** | Dynamic memory allocation that divides available memory equally among models |
| **Device-specific Buffers** | CPU: 20% buffer (min 4GB), CUDA: 10% buffer (min 2GB), MPS: 15% buffer (min 2GB), TPU: 15% buffer (min 8GB) |
| **Memory Rebalancing** | Memory reallocation when models are added or removed |

</div>

<div align="center">

### 💬 Native Message Format Support

</div>

The server now uses the native role (message) feature of transformers and llama-cpp-python:

1. Removed the `DEFAULT_TEMPLATE` and updated the code to use native message formats
2. Updated the `format_prompt` function to return both a formatted prompt and the message objects
3. Enhanced the `generate_text` function to use `create_chat_completion` for llama-cpp models when messages are available
4. Updated the chat completion and completion endpoints to pass messages to the model


## ✨ Features

<div align="center">

| 🌐 **Universal Model Compatibility** | 🔄 **Multi-Model Support** | 🖥️ **Multi-Hardware Support** |
|:--------------------------------:|:------------------------:|:---------------------------:|
| Run **any language model** from Hugging Face, local files, or custom sources | Run multiple models simultaneously on the same server | Automatic device detection (CPU, GPU, TPU, Apple Silicon) |
| Support for all major model architectures (LLaMA, Mistral, Qwen, Phi, Gemma, Falcon, etc.) | Dynamic model loading and unloading at runtime without server restart | Run on any available hardware with zero configuration |
| Automatic model format detection and appropriate loading strategy | Automatic memory detection and allocation across all device types | Graceful fallbacks when requested hardware is unavailable |
| Seamless handling of different model types and architectures | Device-specific memory management with optimized buffer allocation | Full support for Apple Silicon (M1/M2/M3) via MPS backend |
| Zero configuration needed for most models - just provide the model path | Memory rebalancing when models are added or removed | Automatic TPU memory detection and optimization |

</div>
<div align="center">

| 💾 **Model Format Flexibility** | 📂 **GGUF Metadata Extraction** | 💬 **ChatTemplate Support** |
|:--------------------------------:|:------------------------:|:---------------------------:|
| Support for all popular model formats (PyTorch, Safetensors, GGUF, GGML) | Automatically reads metadata from GGUF files | Automatically uses model-specific chat templates when available |
| Use fp16 models (default and recommended for most use cases) | Extracts and uses chat templates from GGUF models | Extracts chat templates from GGUF files |
| Support for GGUF models with optimized performance on all hardware | Provides model architecture and context length information | Falls back to architecture-specific templates when needed |
| Automatic GGUF model download from Hugging Face with progress bar | | |
| Compatible with multiple versions of llama-cpp-python | | |

</div>

<div align="center">

| 💽 **Quantization Options** | 🔗 **OpenAI-Compatible API** | 📰 **Streaming Support** | 🔒 **Security** |
|:--------------------------------:|:------------------------:|:---------------------------:|:---------------------------:|
| Support for 4-bit and 8-bit quantization | Drop-in replacement for OpenAI API clients | Real-time text generation with streaming responses | Optional API key authentication |

</div>

## 🔹 Usage

<div align="center">

### 🔥 Quick Start 🔥

```bash
# Install Inferno
pip install inferno

# Run any model with one command
inferno --model mistralai/Mistral-7B-Instruct-v0.2
```

</div>

### Command Line

#### Basic Usage (Auto-detect best available device)

```bash
# Run any model from Hugging Face
inferno --model meta-llama/Llama-2-7b-chat-hf

# Run any open source model
inferno --model mistralai/Mistral-7B-Instruct-v0.2

# Run Gemma models
inferno --model google/gemma-7b-it

# Run Phi models
inferno --model microsoft/phi-2

# Run Qwen models
inferno --model Qwen/Qwen1.5-7B-Chat

# Run any local model
inferno --model /path/to/your/local/model
```

### Python API

You can also use Inferno as a Python library:

```python
import inferno

# Create a server configuration with any models you want
config = inferno.ServerConfig(
    # Primary model - can be any model from Hugging Face or local path
    model_name_or_path="meta-llama/Llama-2-7b-chat-hf",

    # Load multiple additional models of any type
    additional_models=[
        "mistralai/Mistral-7B-Instruct-v0.2",
        "google/gemma-7b-it",
        "microsoft/phi-2",
        "Qwen/Qwen1.5-7B-Chat",
        "/path/to/your/local/model"  # Local models work too
    ],

    # Automatic hardware detection (or specify: "cuda", "cpu", "mps", "xla")
    device="auto",

    # Server configuration
    host="0.0.0.0",
    port=8000,

    # Optional: Enable GGUF support for even more models
    enable_gguf=True,
    download_gguf=True
)

# Run the server
inferno.run_server(config)
```

### Running Multiple Models

```bash
# Load multiple models of any type at startup (works on any device: CPU, GPU, TPU, Apple Silicon)
# The system automatically detects available memory and allocates it optimally
inferno --model meta-llama/Llama-2-7b-chat-hf --additional-models mistralai/Mistral-7B-Instruct-v0.2 google/gemma-7b-it microsoft/phi-2

# Mix and match any models from different architectures
inferno --model Qwen/Qwen1.5-7B-Chat --additional-models google/gemma-7b-it microsoft/phi-2 mistralai/Mixtral-8x7B-Instruct-v0.1

# Running multiple models on GPU with automatic memory management
# Memory is automatically detected and allocated with a 10% buffer (min 2GB)
inferno --model meta-llama/Llama-2-7b-chat-hf --additional-models mistralai/Mistral-7B-Instruct-v0.2 google/gemma-7b-it --device cuda

# Running multiple models on CPU with automatic memory management
# Memory is automatically detected and allocated with a 20% buffer (min 4GB)
inferno --model mistralai/Mistral-7B-Instruct-v0.2 --additional-models microsoft/phi-2 Qwen/Qwen1.5-7B-Chat --device cpu

# Running multiple models on Apple Silicon with automatic memory management
# Memory is automatically detected and allocated with a 15% buffer (min 2GB)
inferno --model google/gemma-7b-it --additional-models microsoft/phi-2 Qwen/Qwen1.5-7B-Chat --device mps

# Running multiple models on TPU with automatic memory management
# Memory is automatically detected and allocated with a 15% buffer (min 8GB)
inferno --model meta-llama/Llama-2-7b-chat-hf --additional-models mistralai/Mistral-7B-Instruct-v0.2 google/gemma-7b-it --use-tpu
```

### Specific Hardware Usage

```bash
# Force CPU usage with any model
inferno --model mistralai/Mistral-7B-Instruct-v0.2 --device cpu

# Force GPU usage with any model
inferno --model meta-llama/Llama-2-7b-chat-hf --device cuda

# Force TPU usage with any model
inferno --model google/gemma-7b-it --device xla --use-tpu --tpu-cores 8

# Force Apple Silicon (MPS) usage with any model
inferno --model microsoft/phi-2 --device mps

# Run smaller models on less powerful hardware
inferno --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cpu
```

### Using GGUF Models

```bash
# Using a local GGUF file of any model
inferno --enable-gguf --gguf-path path/to/any-model-q4_k_m.gguf

# Auto-downloading GGUF from Hugging Face for any model
inferno --model TheBloke/Llama-2-7B-Chat-GGUF --enable-gguf --download-gguf

# Specifying a particular GGUF file to download
inferno --model TheBloke/Mistral-7B-Instruct-v0.2-GGUF --enable-gguf --download-gguf --gguf-filename mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Run Mixtral models in GGUF format
inferno --model TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF --enable-gguf --download-gguf --gguf-filename mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf

# Run smaller GGUF models on less powerful hardware
inferno --enable-gguf --download-gguf --model TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --gguf-filename tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --device cpu

# Run Qwen models in GGUF format
inferno --enable-gguf --download-gguf --model TheBloke/Qwen1.5-7B-Chat-GGUF --gguf-filename qwen1.5-7b-chat.Q4_K_M.gguf
```

#### GGUF Model Features

##### API Compatibility

The server is compatible with all versions of the llama-cpp-python API. It will automatically detect and use the most appropriate method for your installed version:

1. `__call__` method (recommended in latest API)
2. `create_completion` method (newer versions)
3. `generate` method with `n_predict` parameter (older versions)
4. `completion` method (fallback option)

This ensures maximum compatibility across different llama-cpp-python versions.

##### Token Counting

The server properly counts tokens for GGUF models, providing accurate usage statistics in the API response. It uses a cascading approach to find the best token counting method:

1. `tokenize` method (newer versions)
2. `token_count` method (some versions)
3. `encode` method (some versions)
4. Fallback to a word-based estimate if no method is available

##### Metadata Extraction

The server automatically extracts metadata from GGUF files, including:

1. Model architecture (Llama, Mistral, Qwen, etc.)
2. Model name and organization
3. Context length
4. Chat template

##### Chat Template Support

The server automatically extracts and uses chat templates from GGUF files. If a chat template is found in the GGUF metadata, it will be used for formatting prompts. If no template is found, the server will try to infer an appropriate template based on the model architecture.

##### Hardware Support

GGUF models can run on multiple hardware platforms:

1. CUDA GPU (via n_gpu_layers parameter)
2. CPU (multi-threaded)
3. Apple Silicon (via Metal/MPS)
4. TPU (experimental)

### Quantized Models for Better Performance

```bash
# 8-bit quantization works with any model
inferno --model meta-llama/Llama-2-7b-chat-hf --load-8bit

# 4-bit quantization works with any model
inferno --model mistralai/Mistral-7B-Instruct-v0.2 --load-4bit

# Run large models on consumer hardware with quantization
inferno --model meta-llama/Llama-2-70b-chat-hf --load-4bit

# Run Mixtral models with quantization
inferno --model mistralai/Mixtral-8x7B-Instruct-v0.1 --load-4bit
```

### Adding API Key Security

```bash
# Secure your API with keys - works with any model
inferno --model meta-llama/Llama-2-7b-chat-hf --api-keys "key1,key2,key3"

# Secure multiple models with the same keys
inferno --model mistralai/Mistral-7B-Instruct-v0.2 --additional-models google/gemma-7b-it --api-keys "key1,key2,key3"
```

## 🔗 API Endpoints

<div align="center">

### The server implements OpenAI-compatible endpoints for easy integration as well as model management endpoints

</div>

### Completions API

```bash
# Works with any model loaded on the server
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "prompt": "I feel really happy today because",
    "max_tokens": 100,
    "temperature": 0.7,
    "model": "mistralai/Mistral-7B-Instruct-v0.2"  # Optional: specify which model to use
  }'
```

### Chat Completions API

```bash
# Works with any model loaded on the server
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "I feel really happy today because I just got a promotion!"}
    ],
    "max_tokens": 150,
    "model": "meta-llama/Llama-2-7b-chat-hf"  # Optional: specify which model to use
  }'
```

### Models API

```bash
# List all available models loaded on the server
curl -X GET http://localhost:8000/v1/models \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key"

# Response will include all loaded models of any type
```

### Model Management API

```bash
# Load any new model at runtime without restarting the server
curl -X POST "http://localhost:8000/admin/models/load?model_path=google/gemma-7b-it&set_default=false" \
  -H "Content-Type: application/json"

# Load any GGUF model at runtime
curl -X POST "http://localhost:8000/admin/models/load?model_path=TheBloke/Llama-2-7B-Chat-GGUF&enable_gguf=true&download_gguf=true" \
  -H "Content-Type: application/json"

# Unload any model to free up memory
curl -X POST "http://localhost:8000/admin/models/unload/google/gemma-7b-it" \
  -H "Content-Type: application/json"

# Graceful server shutdown
curl -X POST "http://localhost:8000/admin/shutdown" \
  -H "Content-Type: application/json"
```

## ⚙️ Advanced Configuration

### Package Structure

```
inferno/
├── __init__.py                 # Package initialization with convenience functions
├── cli.py                      # Command-line interface
├── main.py                     # Main entry point
├── config/                     # Configuration-related code
│   ├── __init__.py
│   └── server_config.py        # ServerConfig class
├── memory/                     # Memory management
│   ├── __init__.py
│   └── memory_manager.py       # Memory detection and allocation
├── models/                     # Model handling
│   ├── __init__.py
│   ├── registry.py             # Model registry
│   └── loader.py               # Model loading functions
├── server/                     # Server components
│   ├── __init__.py
│   ├── api.py                  # API endpoints
│   ├── routes.py               # Route definitions
│   └── task_queue.py           # Task queue for async operations
└── utils/                      # Utility functions
    ├── __init__.py
    ├── logger.py               # Logging utilities
    └── device.py               # Device detection utilities
```

### Command Line Options

```
--model              Path to HuggingFace model or local model directory
--additional-models  Additional models to load (space-separated list of model paths)
--model-revision     Specific model revision to load
--tokenizer          Path to tokenizer (defaults to model path)
--tokenizer-revision Specific tokenizer revision to load
--host               Host to bind the server to (default: 0.0.0.0)
--port               Port to bind the server to (default: 8000)
--device             Device to load the model on (auto, cuda, cpu, mps, xla)
--device-map         Device map for model distribution (default: auto)
--dtype              Data type for model weights (float16, float32, bfloat16)
--load-8bit          Load model in 8-bit precision
--load-4bit          Load model in 4-bit precision
--use-tpu            Enable TPU support (requires torch_xla)
--tpu-cores          Number of TPU cores to use (default: 8)
--api-keys           Comma-separated list of valid API keys
--max-concurrent     Maximum number of concurrent requests (default: 10)
--max-queue          Maximum queue size for pending requests (default: 100)
--timeout            Timeout for requests in seconds (default: 60)
--enable-gguf        Enable GGUF model support (requires llama-cpp-python)
--gguf-path          Path to GGUF model file
--download-gguf      Download GGUF model from Hugging Face (if available)
--gguf-filename      Specific GGUF filename to download (e.g., 'model-q4_k_m.gguf')
--num-gpu-layers     Number of GPU layers for GGUF models (-1 means all)
```

## 🖥️ Hardware Recommendations

<div align="center">

| Hardware | Recommendation |
|:--------:|:--------------:|
| **Auto-detect** | For most users, use the default auto-detection for the best experience |
| **GPU** | For best performance, use NVIDIA GPUs with at least 8GB VRAM |
| **CPU** | For CPU-only deployment, consider using quantized models (4-bit or 8-bit) |
| **TPU** | When using Google Cloud TPUs, use bfloat16 precision for optimal performance |
| **Apple Silicon** | For Mac users with M1/M2/M3 chips, the MPS backend provides GPU acceleration |

</div>

## 💾 Model Format Recommendations

<div align="center">

| Format | Recommendation |
|:------:|:--------------:|
| **fp16** | Default and recommended for most hardware (GPU, CPU) |
| **GGUF** | Use when needed for specific hardware compatibility or when memory optimization is critical |

</div>

## 👷 Contributing

<div align="center">

### Contributions are welcome! Here's how you can contribute to Inferno:

</div>

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/my-new-feature`
5. Submit a pull request

Please make sure to update tests as appropriate and follow the code style of the project.

## 📜 License

<div align="center">

### Inferno is released under the HelpingAI Open Source License v1.0

</div>

This is a permissive open source license that allows for:

- Use for any purpose, including commercial applications
- Modification and creation of derivative works
- Distribution and redistribution
- Private and commercial use

The license requires attribution and includes standard disclaimers of warranty and liability. See the [LICENSE](LICENSE) file for complete details.

<div align="center">

---

<p>🔥 <strong>Inferno</strong> - Making advanced AI deployment accessible to everyone 🔥</p>

</div>
