import os
import logging
import sys
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_dir="logs"):
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level (default: INFO)
        log_dir: Directory to store log files
        
    Returns:
        Logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging setup completed. Log file: {log_file}")
    
    return logger


def log_gpu_info():
    """Log information about available GPUs"""
    import torch
    
    logger = logging.getLogger(__name__)
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA devices:")
        
        for i in range(device_count):
            device = torch.cuda.get_device_properties(i)
            logger.info(f"  Device {i}: {device.name}")
            logger.info(f"    Memory: {device.total_memory / (1024**3):.2f} GB")
            logger.info(f"    CUDA Capability: {device.major}.{device.minor}")
    else:
        logger.warning("CUDA is not available. Training will be slow.")