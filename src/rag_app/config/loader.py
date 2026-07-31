"""
Configuration loader with JSON file support, environment variable substitution,
and path security validation (root path containment).
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

from .schema import AppConfig


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


def load_config_from_json(config_path: Path) -> dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the JSON configuration file.
        
    Returns:
        Parsed configuration dictionary.
        
    Raises:
        ConfigurationError: If file not found, invalid JSON, or path traversal detected.
    """
    # Security: Resolve and validate path
    config_path = config_path.expanduser().resolve()
    
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration path is not a file: {config_path}")
    
    # Read and parse JSON
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        raise ConfigurationError(f"Failed to read configuration file: {e}") from e
    
    try:
        config_dict = json.loads(content)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in configuration file: {e}") from e
    
    if not isinstance(config_dict, dict):
        raise ConfigurationError("Configuration root must be a JSON object")
    
    return config_dict


def substitute_env_vars(obj: Any) -> Any:
    """
    Recursively substitute environment variable references in configuration values.
    
    Supports format: "${ENV_VAR_NAME}" or "${ENV_VAR_NAME:-default_value}"
    
    Args:
        obj: Configuration value (dict, list, str, or other).
        
    Returns:
        Configuration value with environment variables substituted.
    """
    if isinstance(obj, str):
        # Handle ${VAR} or ${VAR:-default} format
        import re
        pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
        
        def replace(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2)
            value = os.getenv(var_name)
            if value is not None:
                return value
            if default_value is not None:
                return default_value
            # If no default and not set, return empty string (will be caught by validation)
            return ""
        
        return pattern.sub(replace, obj)
    
    elif isinstance(obj, dict):
        return {k: substitute_env_vars(v) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        return [substitute_env_vars(item) for item in obj]
    
    return obj


def validate_path_containment(path: Path, root: Path, path_name: str) -> Path:
    """
    Validate that a path is contained within the allowed root directory.
    
    Args:
        path: Path to validate.
        root: Allowed root directory.
        path_name: Human-readable name for error messages.
        
    Returns:
        Resolved path if validation passes.
        
    Raises:
        ConfigurationError: If path escapes root directory.
    """
    try:
        resolved_path = path.expanduser().resolve()
        resolved_root = root.expanduser().resolve()
        
        # Check if path is within root
        resolved_path.relative_to(resolved_root)
        return resolved_path
    except ValueError:
        raise ConfigurationError(
            f"Path traversal detected: {path_name} ({path}) "
            f"is outside allowed root directory ({root})"
        ) from None


def create_app_config(config_dict: dict[str, Any], config_file_path: Optional[Path] = None) -> AppConfig:
    """
    Create validated AppConfig from dictionary with security checks.
    
    Args:
        config_dict: Raw configuration dictionary.
        config_file_path: Optional path to config file for relative path resolution.
        
    Returns:
        Validated AppConfig instance.
        
    Raises:
        ConfigurationError: If validation fails.
    """
    # Substitute environment variables
    config_dict = substitute_env_vars(config_dict)
    
    # If config file path provided, use its directory as base for relative paths
    if config_file_path:
        base_dir = config_file_path.parent
    else:
        base_dir = Path.cwd()
    
    # Create config - Pydantic will validate
    try:
        config = AppConfig(**config_dict)
    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {e}") from e
    
    # Additional security validation: ensure document_root is contained
    # (Pydantic validator already ensures this, but double-check with explicit root)
    # The root for containment is the project root (parent of config dir or current working dir)
    allowed_root = base_dir.parent.resolve() if base_dir.name == "config" else base_dir.resolve()
    
    # Validate paths are within allowed root
    validate_path_containment(config.paths.document_root, allowed_root, "document_root")
    validate_path_containment(config.paths.metadata_db, allowed_root, "metadata_db")
    validate_path_containment(config.paths.log_dir, allowed_root, "log_dir")
    
    return config


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """
    Load and validate application configuration.
    
    Priority:
    1. Explicit config_path argument
    2. RAG_CONFIG environment variable
    3. Default config based on environment (config/config.{env}.json)
    
    Args:
        config_path: Optional explicit path to configuration file.
        
    Returns:
        Validated AppConfig instance.
        
    Raises:
        ConfigurationError: If configuration cannot be loaded or validated.
    """
    # Determine config file path
    if config_path is not None:
        config_file = Path(config_path)
    elif os.getenv("RAG_CONFIG"):
        config_file = Path(os.getenv("RAG_CONFIG") or "")
    else:
        # Use environment-specific default
        env = os.getenv("RAG_ENV", "development")
        config_file = Path(f"config/config.{env}.json")
    
    # Load from JSON
    config_dict = load_config_from_json(config_file)
    
    # Create validated config
    return create_app_config(config_dict, config_file)


def get_default_config_path(environment: str) -> Path:
    """Get default config file path for environment."""
    return Path(f"config/config.{environment}.json")


def create_sample_configs() -> None:
    """Create sample configuration files for each environment."""
    from .schema import AppConfig
    
    # Development config
    dev_config = AppConfig(environment="development")
    dev_config.paths.document_root = Path("./document")
    dev_config.paths.metadata_db = Path("./data/metadata.db")
    dev_config.paths.log_dir = Path("./data/logs")
    
    # Test config
    test_config = AppConfig(environment="test")
    test_config.paths.document_root = Path("./tests/fixtures/document")
    test_config.paths.metadata_db = Path("./tests/fixtures/data/metadata.db")
    test_config.paths.log_dir = Path("./tests/fixtures/data/logs")
    test_config.indexing.scan_interval_seconds = 60
    test_config.indexing.max_file_size_mb = 10
    
    # Production config
    prod_config = AppConfig(environment="production")
    prod_config.paths.document_root = Path("/app/document")
    prod_config.paths.metadata_db = Path("/app/data/metadata.db")
    prod_config.paths.log_dir = Path("/app/data/logs")
    prod_config.logging.level = "WARNING"
    prod_config.logging.include_query_text = False
    prod_config.qdrant.https = True
    prod_config.qdrant.verify_ssl = True
    
    configs = {
        "development": dev_config,
        "test": test_config,
        "production": prod_config,
    }
    
    config_dir = Path("config")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    for env_name, config in configs.items():
        config_file = config_dir / f"config.{env_name}.json"
        with open(config_file, "w", encoding="utf-8") as f:
            # Use model_dump with exclude_none=False to include all fields
            json.dump(config.model_dump(mode="json", exclude_none=False), f, indent=2, ensure_ascii=False)
        print(f"Created sample config: {config_file}")