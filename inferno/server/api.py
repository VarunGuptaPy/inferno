from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from typing import Dict, List, Optional, Any, Union

from inferno.utils.logger import get_logger
from inferno.config.server_config import ServerConfig
from inferno.models.registry import MODEL_REGISTRY
from inferno.server.routes import (
    register_openai_routes,
    register_admin_routes,
    register_health_routes
)

logger = get_logger(__name__)


def verify_api_key(request: Request, config: ServerConfig):
    """
    Verify the API key in the request.

    Args:
        request: FastAPI request object
        config: Server configuration

    Raises:
        HTTPException: If the API key is invalid
    """
    # Skip API key verification if no keys are configured
    if not config.api_keys:
        return

    # Get the API key from the request
    api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")

    # Remove 'Bearer ' prefix if present
    if api_key and api_key.startswith("Bearer "):
        api_key = api_key[7:]

    # Verify the API key
    if not api_key or api_key not in config.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )


def create_app(config: ServerConfig) -> FastAPI:
    """
    Create the FastAPI application.

    Args:
        config: Server configuration

    Returns:
        FastAPI application
    """
    # Create the FastAPI app
    app = FastAPI(
        title="Inferno API",
        description="A professional inference server for HelpingAI models",
        version="0.1.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(int(time.time() * 1000))  # Simple timestamp-based ID
        request.state.request_id = request_id
        request.state.start_time = time.time()

        # Log the request
        logger.info(f"Request {request_id}: {request.method} {request.url.path}")

        # Process the request
        response = await call_next(request)

        # Calculate request duration
        duration = time.time() - request.state.start_time

        # Log the response
        logger.info(f"Response {request_id}: {response.status_code} ({duration:.3f}s)")

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response

    # Add API key verification middleware if API keys are configured
    if config.api_keys:
        @app.middleware("http")
        async def verify_api_key_middleware(request: Request, call_next):
            # Skip API key verification for health check and admin routes
            if request.url.path.startswith("/health") or request.url.path.startswith("/admin"):
                return await call_next(request)

            try:
                verify_api_key(request, config)
                return await call_next(request)
            except HTTPException as e:
                return JSONResponse(
                    status_code=e.status_code,
                    content={"error": e.detail}
                )

    # Register routes
    register_openai_routes(app, config)
    register_admin_routes(app, config)
    register_health_routes(app, config)

    # Add startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        logger.info("Server started")
        logger.info(f"Loaded {MODEL_REGISTRY.get_model_count()} models")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Server shutting down")
        # Clean up resources
        MODEL_REGISTRY.clear()

    return app