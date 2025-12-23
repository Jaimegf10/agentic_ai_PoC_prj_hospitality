"""
Logger Configuration Module

This module provides functionality for configuring and managing logging in the application.
It sets up a logger with appropriate formatting, handlers, and log levels to ensure
consistent and informative logging throughout the application.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure the logger
logger = logging.getLogger("hospitality_api")
logger.setLevel(logging.INFO)

# Create formatters
console_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Configure file handler with rotation
file_handler = RotatingFileHandler(
    "logs/hospitality_api.log",
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding="utf-8",
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)