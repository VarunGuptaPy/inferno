# Inferno

A professional, production-ready inference server for HelpingAI models with support for multiple hardware platforms (CPU, GPU, TPU, Apple Silicon) and model formats.

## Features

- **Multi-Model Support**:
  - Run multiple models simultaneously
  - Dynamic model loading and unloading at runtime
  - Automatic memory detection and allocation across all device types (CPU, GPU, TPU, Apple Silicon)
  - Device-specific memory management with optimized buffer allocation
  - Memory rebalancing when models are added or removed
  - Model registry with metadata tracking
  - API endpoints for model management

- **Multi-Hardware Support**:
  - Automatic device detection (CPU, GPU, TPU, Apple Silicon)
  - Run on any available hardware with zero configuration
  - Graceful fallbacks when requested hardware is unavailable
  - Full support for Apple Silicon (M1/M2/M3) via MPS backend
  - Automatic TPU memory detection and optimization

- **Model Format Flexibility**:
  - Use fp16 models (default and recommended)
  - Support for GGUF models when needed (especially for TPU compatibility)
  - Automatic GGUF model download from Hugging Face with progress bar
  - Compatible with multiple versions of llama-cpp-python

- **GGUF Metadata Extraction**:
  - Automatically reads metadata from GGUF files
  - Extracts and uses chat templates from GGUF models
  - Provides model architecture and context length information

- **ChatTemplate Support**:
  - Automatically uses model-specific chat templates when available
  - Extracts chat templates from GGUF files
  - Falls back to architecture-specific templates when needed

- **Quantization Options**: Support for 4-bit and 8-bit quantization

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API clients

- **Streaming Support**: Real-time text generation with streaming responses

- **Security**: Optional API key authentication

## Installation

### Basic Installation

```bash
pip install inferno
```

### With TPU Support

```bash
pip install "inferno[tpu]"
```

### With GGUF Support

```bash
pip install "inferno[gguf]"
```

### Full Installation

```bash
pip install "inferno[tpu,gguf]"
```

## Usage

### Basic Usage (Auto-detect best available device)

```bash
inferno --model HelpingAI/HelpingAI-15B
```

### Running Multiple Models

```bash
# Load multiple models at startup (works on any device: CPU, GPU, TPU, Apple Silicon)
# The system automatically detects available memory and allocates it optimally
inferno --model HelpingAI/HelpingAI-15B --additional-models mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Llama-2-7b-chat-hf

# Running multiple models on GPU
# Memory is automatically detected and allocated with a 10% buffer (min 2GB)
inferno --model HelpingAI/HelpingAI-15B --additional-models mistralai/Mistral-7B-Instruct-v0.2 --device cuda

# Running multiple models on CPU
# Memory is automatically detected and allocated with a 20% buffer (min 4GB)
inferno --model HelpingAI/HelpingAI-15B --additional-models mistralai/Mistral-7B-Instruct-v0.2 --device cpu

# Running multiple models on Apple Silicon
# Memory is automatically detected and allocated with a 15% buffer (min 2GB)
inferno --model HelpingAI/HelpingAI-15B --additional-models mistralai/Mistral-7B-Instruct-v0.2 --device mps

# Running multiple models on TPU with automatic memory management
# Memory is automatically detected and allocated with a 15% buffer (min 8GB)
inferno --model HelpingAI/HelpingAI-15B --additional-models mistralai/Mistral-7B-Instruct-v0.2 --use-tpu
```

### Specific Hardware Usage

```bash
# Force CPU usage
inferno --model HelpingAI/HelpingAI-15B --device cpu

# Force GPU usage
inferno --model HelpingAI/HelpingAI-15B --device cuda

# Force TPU usage
inferno --model HelpingAI/HelpingAI-15B --device xla --use-tpu --tpu-cores 8

# Force Apple Silicon (MPS) usage
inferno --model HelpingAI/HelpingAI-15B --device mps
```

### Using GGUF Models

```bash
# Using a local GGUF file
inferno --enable-gguf --gguf-path path/to/helpingai-15b-q4_k_m.gguf

# Auto-downloading GGUF from Hugging Face
inferno --model HelpingAI/HelpingAI-15B --enable-gguf --download-gguf

# Specifying a particular GGUF file to download
inferno --model HelpingAI/HelpingAI-15B --enable-gguf --download-gguf --gguf-filename model-q4_k_m.gguf
```

### Quantized Models for Better Performance

```bash
# 8-bit quantization
inferno --model HelpingAI/HelpingAI-15B --load-8bit

# 4-bit quantization
inferno --model HelpingAI/HelpingAI-15B --load-4bit
```

### Adding API Key Security

```bash
inferno --model HelpingAI/HelpingAI-15B --api-keys "key1,key2,key3"
```

## API Endpoints

The server implements OpenAI-compatible endpoints for easy integration as well as model management endpoints:

### Completions API

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "prompt": "I feel really happy today because",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Chat Completions API

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are HelpingAI, an emotionally intelligent assistant."},
      {"role": "user", "content": "I feel really happy today because I just got a promotion!"}
    ],
    "max_tokens": 150
  }'
```

### Models API

```bash
# List all available models
curl -X POST http://localhost:8000/v1/models \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key"
```

### Model Management API

```bash
# Load a new model at runtime
curl -X POST "http://localhost:8000/admin/models/load?model_path=TheBloke/Llama-2-7B-Chat-GGUF&set_default=false" \
  -H "Content-Type: application/json"

# Unload a model to free up memory
curl -X POST "http://localhost:8000/admin/models/unload/TheBloke/Llama-2-7B-Chat-GGUF" \
  -H "Content-Type: application/json"

# Graceful server shutdown
curl -X POST "http://localhost:8000/admin/shutdown" \
  -H "Content-Type: application/json"
```

## Advanced Configuration

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

## Hardware Recommendations

- **Auto-detect**: For most users, use the default auto-detection for the best experience
- **GPU**: For best performance, use NVIDIA GPUs with at least 8GB VRAM
- **CPU**: For CPU-only deployment, consider using quantized models (4-bit or 8-bit)
- **TPU**: When using Google Cloud TPUs, use bfloat16 precision for optimal performance
- **Apple Silicon**: For Mac users with M1/M2/M3 chips, the MPS backend provides GPU acceleration

## Model Format Recommendations

- **fp16**: Default and recommended for most hardware (GPU, CPU)
- **GGUF**: Use when needed for specific hardware compatibility or when memory optimization is critical

## License

MIT License
