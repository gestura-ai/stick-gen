"""
Configuration loader for stick-gen training.

Loads training configuration from YAML files and provides easy access to parameters.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class TrainingConfig:
    """Training configuration loader and accessor."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file (default: config.yaml)
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                f"Available configs: config.yaml, config_cpu.yaml, config_gpu.yaml"
            )

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to config value (e.g., "model.d_model")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config = TrainingConfig()
            >>> d_model = config.get("model.d_model")
            >>> batch_size = config.get("training.batch_size")
        """
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get("model", {})

    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get("training", {})

    @property
    def loss_weights(self) -> Dict[str, Any]:
        """Get loss weights configuration."""
        return self.config.get("loss_weights", {})

    @property
    def physics(self) -> Dict[str, Any]:
        """Get physics configuration."""
        return self.config.get("physics", {})

    @property
    def diffusion(self) -> Dict[str, Any]:
        """Get diffusion configuration."""
        return self.config.get("diffusion", {})

    @property
    def data(self) -> Dict[str, Any]:
        """Get data paths configuration."""
        return self.config.get("data", {})

    @property
    def device(self) -> Dict[str, Any]:
        """Get device configuration."""
        return self.config.get("device", {})

    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get("logging", {})

    @property
    def optimization(self) -> Dict[str, Any]:
        """Get optimization configuration."""
        return self.config.get("optimization", {})

    @property
    def cpu(self) -> Dict[str, Any]:
        """Get CPU optimization configuration."""
        return self.config.get("cpu", {})

    def print_config(self):
        """Print configuration in a readable format."""
        print("=" * 60)
        print("TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Config file: {self.config_path}")
        print()

        for section, values in self.config.items():
            print(f"{section.upper()}:")
            if isinstance(values, dict):
                for key, value in values.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {values}")
            print()


def load_config(config_path: str = "config.yaml") -> TrainingConfig:
    """
    Load training configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        TrainingConfig instance
    """
    return TrainingConfig(config_path)
