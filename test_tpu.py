#!/usr/bin/env python
"""
Test script to verify TPU detection and usage in Inferno.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("tpu-test")

# Check for TPU library
logger.info("Checking for TPU library...")
if os.path.exists('/usr/lib/libtpu.so') or os.path.exists('/lib/libtpu.so'):
    logger.info("TPU library found!")
    # Set TPU environment variables
    os.environ["PJRT_DEVICE"] = "TPU"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    logger.info("Set PJRT_DEVICE=TPU")
else:
    logger.warning("No TPU library found")

# Try to import torch_xla
try:
    logger.info("Trying to import torch_xla...")
    import torch_xla # type: ignore[import]
    import torch_xla.core.xla_model as xm # type: ignore[import]
    logger.info("torch_xla imported successfully")
    
    # Try to get TPU devices
    try:
        logger.info("Getting XLA devices...")
        devices = xm.get_xla_supported_devices()
        logger.info(f"Found {len(devices)} XLA devices: {devices}")
        
        # Try to create a device
        logger.info("Creating XLA device...")
        device = xm.xla_device()
        logger.info(f"XLA device created: {device}")
        
        # Try to create a tensor on the device
        logger.info("Creating tensor on XLA device...")
        import torch
        tensor = torch.zeros(1, device=device)
        logger.info(f"Tensor created on XLA device: {tensor}")
        
        logger.info("TPU is working correctly!")
    except Exception as e:
        logger.error(f"Error accessing TPU: {e}")
except ImportError as e:
    logger.error(f"Error importing torch_xla: {e}")
    logger.error("Please install torch_xla with: pip install torch_xla")

# Try to run Inferno with TPU
try:
    logger.info("\nTrying to import Inferno...")
    from inferno.utils.device import get_device_info, setup_device, XLA
    logger.info("Inferno imported successfully")
    
    # Get device info
    logger.info("Getting device info...")
    device_info = get_device_info(XLA)
    logger.info(f"Device info: {device_info}")
    
    # Setup device
    logger.info("Setting up device...")
    device, _ = setup_device(device_type=XLA, use_tpu=True, force_tpu=True)
    logger.info(f"Device set up: {device}")
    
    logger.info("Inferno TPU setup is working correctly!")
except Exception as e:
    logger.error(f"Error setting up Inferno with TPU: {e}")

logger.info("TPU test completed")
