import logging
from pathlib import Path
from typing import Any

import tomli
from config.config import Config
from platformdirs import user_config_dir

from utils.errors import ConfigError

logger = logging.getLogger(__name__)
CONFIG_FILE_NAME = 'config.toml'
AGENT_MD_FILE = "TERMINAL_AGENT.md"

def get_config_dir()->Path:
    return Path(user_config_dir('terminal-agent'))

def get_system_config_path():
    return get_config_dir() / CONFIG_FILE_NAME

def _parse_toml(path: Path):
    try:
        with open(path, "rb") as f:
            return tomli.load(f)
    except tomli.TOMLDecodeError as e:
        raise ConfigError("Invalid TOML in {path}: {e}", config_file=str(path)) from e
    except (OSError, IOError) as e:
        raise ConfigError(
            "Failed to read config file {path}: {e}", config_file=str(path)
        ) from e

def _get_project_config(cwd:Path):
    current_dir = cwd.resolve()
    agent_dir = current_dir / ".terminal-agent"

    if agent_dir.is_dir():
        config_file = agent_dir / CONFIG_FILE_NAME
        if config_file.is_file():
            return config_file
    return None

def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value

    return result

def _get_agent_md_file(cwd:Path):
    current_dir = cwd.resolve()
    if current_dir.is_dir():
        agent_md_file = current_dir / AGENT_MD_FILE
        if agent_md_file.is_file():
            return agent_md_file
    return None

def load_config(cwd:Path | None ):
    cwd = cwd or Path.cwd()

    system_config_path = get_system_config_path()
    config_dict = {}
    if system_config_path.is_file():
        try:
            config_dict = _parse_toml(system_config_path)
        except ConfigError:
            logger.warning(f"Skipping invalid system config {system_config_path}")

    project_config_path = _get_project_config(cwd)
    if project_config_path is not None and project_config_path.is_file():
        try:
            project_config_dict = _parse_toml(project_config_path)
            config_dict = _merge_dicts(config_dict, project_config_dict)
        except ConfigError:
            logger.warning(f"Skipping invalid project config {project_config_path}")

    if 'cwd' not in config_dict:
        config_dict['cwd'] = cwd

    if 'user_instructions' not in config_dict:
        agent_md_file = _get_agent_md_file(cwd)

        if agent_md_file is not None:
            config_dict["user_instructions"] = agent_md_file.read_text(encoding="utf-8").strip()

    try:
        config = Config(**config_dict)
    except Exception as e:
        raise ConfigError(f"Invalid configuration {e}")

    return config
        
