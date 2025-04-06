"""Utility modules for Inferno."""

from inferno.utils.logger import get_logger, InfernoLogger
from inferno.utils.device import (
    get_available_devices,
    get_optimal_device,
    get_device_info,
    setup_device
)

__all__ = [
    "get_logger",
    "InfernoLogger",
    "get_available_devices",
    "get_optimal_device",
    "get_device_info",
    "setup_device"
]