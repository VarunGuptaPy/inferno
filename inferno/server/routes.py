from fastapi import APIRouter, FastAPI, HTTPException, Request, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union, Literal
import time
import sys

from inferno.utils.logger import get_logger
from inferno.config.server_config import ServerConfig
from inferno.models.registry import MODEL_REGISTRY
from inferno.models.loader import load_and_register_model, unload_and_unregister_model
from inferno.server.task_queue import TaskQueue

logger = get_logger(__name__)

# Create a task queue for handling requests
task_queue = TaskQueue()


# Define API models
class ModelInfo(BaseModel):
    id: str
    path: str
    is_default: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelsResponse(BaseModel):
    models: List[ModelInfo]


class Message(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    message: Message
    index: int
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]


class ErrorResponse(BaseModel):
    error: str


class HealthResponse(BaseModel):
    status: str
    version: str
    models: int


class LoadModelRequest(BaseModel):
    model_path: str
    set_default: bool = False
    enable_gguf: bool = False
    download_gguf: bool = False
    load_8bit: bool = False
    load_4bit: bool = False


class LoadModelResponse(BaseModel):
    model_id: str
    status: str


class UnloadModelResponse(BaseModel):
    status: str


class ShutdownResponse(BaseModel):
    status: str


def register_openai_routes(app: FastAPI, config: ServerConfig):
    """
    Register OpenAI-compatible API routes.

    Args:
        app: FastAPI application
        config: Server configuration
    """
    router = APIRouter(prefix="/v1", tags=["OpenAI API"])

    @router.get("/models", response_model=ModelsResponse)
    async def list_models():
        """
        List all available models.
        """
        models = MODEL_REGISTRY.list_models()
        return {"models": models}

    @router.post("/completions", response_model=CompletionResponse)
    async def create_completion(request: CompletionRequest, background_tasks: BackgroundTasks):
        """
        Create a completion.
        """
        # Get the model
        model_info = MODEL_REGISTRY.get_model(request.model)
        if model_info is None:
            raise HTTPException(status_code=404, detail="Model not found")

        # Add the task to the queue
        task_id = f"completion-{int(time.time() * 1000)}"

        # TODO: Implement actual completion generation
        # This is a placeholder implementation
        completion_text = f"This is a placeholder completion for: {request.prompt}"

        return {
            "id": task_id,
            "created": int(time.time()),
            "model": model_info.model_id,
            "choices": [
                {
                    "text": completion_text,
                    "index": 0,
                    "finish_reason": "length"
                }
            ]
        }

    @router.post("/chat/completions", response_model=ChatCompletionResponse)
    async def create_chat_completion(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
        """
        Create a chat completion.
        """
        # Get the model
        model_info = MODEL_REGISTRY.get_model(request.model)
        if model_info is None:
            raise HTTPException(status_code=404, detail="Model not found")

        # Add the task to the queue
        task_id = f"chat-completion-{int(time.time() * 1000)}"

        # TODO: Implement actual chat completion generation
        # This is a placeholder implementation
        last_message = request.messages[-1].content if request.messages else ""
        completion_text = f"This is a placeholder chat response to: {last_message}"

        return {
            "id": task_id,
            "created": int(time.time()),
            "model": model_info.model_id,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": completion_text
                    },
                    "index": 0,
                    "finish_reason": "length"
                }
            ]
        }

    app.include_router(router)


def register_admin_routes(app: FastAPI, config: ServerConfig):
    """
    Register admin API routes.

    Args:
        app: FastAPI application
        config: Server configuration
    """
    router = APIRouter(prefix="/admin", tags=["Admin API"])

    @router.post("/models/load", response_model=LoadModelResponse)
    async def load_model(
        model_path: str = Query(..., description="Path to the model to load"),
        set_default: bool = Query(False, description="Set as default model"),
        enable_gguf: bool = Query(False, description="Enable GGUF model support"),
        download_gguf: bool = Query(False, description="Download GGUF model"),
        load_8bit: bool = Query(False, description="Load in 8-bit precision"),
        load_4bit: bool = Query(False, description="Load in 4-bit precision")
    ):
        """
        Load a new model.
        """
        try:
            # Create a new config for this model
            model_config = ServerConfig(
                model_name_or_path=model_path,
                enable_gguf=enable_gguf,
                download_gguf=download_gguf,
                device=config.device,
                load_8bit=load_8bit,
                load_4bit=load_4bit,
                use_tpu=config.use_tpu,
                tpu_memory_limit=config.tpu_memory_limit
            )

            # Load and register the model
            model_id = load_and_register_model(model_config, set_default=set_default)

            return {
                "model_id": model_id,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/models/unload/{model_id}", response_model=UnloadModelResponse)
    async def unload_model(model_id: str):
        """
        Unload a model.
        """
        try:
            # Unload and unregister the model
            success = unload_and_unregister_model(model_id)

            if not success:
                raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

            return {"status": "success"}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/shutdown", response_model=ShutdownResponse)
    async def shutdown():
        """
        Gracefully shut down the server.
        """
        # Schedule server shutdown
        def shutdown_server():
            # Wait a moment to allow the response to be sent
            time.sleep(1)
            # Exit the process
            sys.exit(0)

        # Run shutdown in background
        import threading
        threading.Thread(target=shutdown_server).start()

        return {"status": "shutting down"}

    app.include_router(router)


def register_health_routes(app: FastAPI, config: ServerConfig):
    """
    Register health check routes.

    Args:
        app: FastAPI application
        config: Server configuration
    """
    router = APIRouter(prefix="/health", tags=["Health"])

    @router.get("", response_model=HealthResponse)
    async def health_check():
        """
        Check the health of the server.
        """
        return {
            "status": "ok",
            "version": "0.1.0",
            "models": MODEL_REGISTRY.get_model_count()
        }

    app.include_router(router)