"""
Configuration Module for AI Hospitality API

This module provides configuration settings for the AI Hospitality API application.
It includes API settings and project configuration.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings

# Get the absolute path to the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    Configuration settings for the AI Hospitality API application.
    """

    # API settings
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8001)

    # CORS settings
    CORS_ORIGINS: List[str] = Field(default=["*"])

    class Config:
        """
        Configuration for the settings class.
        """
        env_file = f".env.{os.getenv('ENVIRONMENT', 'development')}"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache
def get_settings() -> Settings:
    """
    Creates settings instance and caches it.
    Using lru_cache to avoid reading the environment variables on every call.
    """
    return Settings()


# Initialize settings immediately when module is imported
settings = get_settings()