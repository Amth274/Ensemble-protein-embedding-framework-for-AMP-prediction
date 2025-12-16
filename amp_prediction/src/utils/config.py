"""Configuration utilities."""

import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, raises ValueError if invalid
    """
    required_sections = ['data', 'embedding', 'models', 'training', 'ensemble']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    # Validate data section
    data_config = config['data']
    required_data_fields = ['sequence_column', 'label_column', 'max_length']
    for field in required_data_fields:
        if field not in data_config:
            raise ValueError(f"Missing required data field: {field}")

    # Validate embedding section
    embedding_config = config['embedding']
    required_embedding_fields = ['model_name', 'embedding_dim']
    for field in required_embedding_fields:
        if field not in embedding_config:
            raise ValueError(f"Missing required embedding field: {field}")

    return True


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration with new values.

    Args:
        config: Original configuration
        updates: Updates to apply

    Returns:
        Updated configuration
    """
    def deep_update(base_dict, update_dict):
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    updated_config = config.copy()
    deep_update(updated_config, updates)
    return updated_config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    with open(output_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, indent=2)