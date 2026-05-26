import torch
import gc
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ForensicANPR.GPU")

def get_device() -> torch.device:
    """Returns the most optimized device available (CUDA or CPU)."""
    if torch.cuda.is_available():
        logger.info("CUDA GPU detected. Using GPU acceleration.")
        # Enable CuDNN benchmark for optimized convolutional calculations
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    logger.info("CUDA GPU not detected. Using CPU.")
    return torch.device("cpu")

def clear_gpu_memory():
    """Cleans up CPU/GPU RAM and empties PyTorch cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.info("Cleaned CUDA memory cache successfully.")

def get_gpu_info() -> dict:
    """Retrieves CUDA device specifications for forensic logging."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device_name": None,
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "max_allocated_mb": 0.0
    }
    if info["cuda_available"]:
        device_idx = torch.cuda.current_device()
        info["current_device_name"] = torch.cuda.get_device_name(device_idx)
        info["allocated_mb"] = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
        info["reserved_mb"] = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
        info["max_allocated_mb"] = torch.cuda.max_memory_allocated(device_idx) / (1024 ** 2)
    return info

def print_gpu_summary():
    """Logs the GPU/CPU device parameters."""
    info = get_gpu_info()
    if info["cuda_available"]:
        logger.info(f"--- GPU Diagnostic Summary ---")
        logger.info(f"Device Name: {info['current_device_name']}")
        logger.info(f"Memory Allocated: {info['allocated_mb']:.2f} MB")
        logger.info(f"Memory Reserved: {info['reserved_mb']:.2f} MB")
        logger.info(f"Max Allocated: {info['max_allocated_mb']:.2f} MB")
        logger.info(f"------------------------------")
    else:
        logger.info("No CUDA graphics devices available. CPU execution mode.")
