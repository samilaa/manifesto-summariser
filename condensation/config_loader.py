import yaml
from typing import Any, Dict

def load_config(config_path: str = "config/default_config.yaml") -> Dict[str, Any]:
    """Load the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
