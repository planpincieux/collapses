import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger("ppcx")


def _init_env() -> None:
    """Load environment variables from the nearest .env (like VS Code does)."""
    # Search upward from current working dir
    load_dotenv(find_dotenv(usecwd=True), override=False)


def _find_config_path() -> Path:
    """Find config.yaml via env var or by searching common locations."""
    # 1) Explicit override
    env_path = os.getenv("PPCX_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    # 2) CWD (works for `python make_collapses_plots.py` from repo folder)
    for name in ("config.yaml", "config.yml"):
        p = Path.cwd() / name
        if p.exists():
            return p.resolve()
    # 3) Near this file (works if run from other dirs)
    here = Path(__file__).resolve()
    for base in [here.parent, *here.parents]:
        for name in ("config.yaml", "config.yml"):
            p = base / name
            if p.exists():
                return p.resolve()
    # Fallback (may raise later if missing)
    return (Path.cwd() / "config.yaml").resolve()


# Load env early
_init_env()

# Determine config path
CONFIG_PATH = _find_config_path()


@dataclass
class ConfigManager:
    """Singleton configuration manager using OmegaConf and dataclass properties."""

    _instance: Optional["ConfigManager"] = field(default=None, init=False, repr=False)
    _config: DictConfig | None = field(default=None, init=False, repr=False)
    config_path: str | Path = field(default=CONFIG_PATH)

    def __repr__(self) -> str:
        return f"ConfigManager(config_path={self.config_path}, config={self.config})"

    def __new__(cls, config_path: str | Path = CONFIG_PATH):
        """Create a singleton instance of ConfigManager."""
        if cls._instance is None:
            # If no instance exists, create one
            cls._instance = super().__new__(cls)

            # Set the config_path only on first creation
            cls._instance.config_path = Path(config_path)

        # Return the singleton instance
        return cls._instance

    def __post_init__(self):
        if self._config is None:
            self._config = self._load_config(self.config_path)

    def _load_config(self, config_path: str | Path) -> DictConfig:
        """Load configuration from YAML file with environment variable overrides."""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = OmegaConf.load(config_path)

        # Override with environment variables if they exist
        env_overrides = {
            "database.host": os.getenv("DB_HOST"),
            "database.port": os.getenv("DB_PORT"),
            "database.name": os.getenv("DB_NAME"),
            "database.user": os.getenv("DB_USER"),
            "database.password": os.getenv("DB_PASSWORD"),
            "api.host": os.getenv("APP_HOST"),
            "api.port": os.getenv("APP_PORT"),
            "api.image_view": os.getenv("GET_IMAGE_VIEW"),
        }
        for key, value in env_overrides.items():
            if value is not None:
                OmegaConf.update(config, key, value)
        return config if isinstance(config, DictConfig) else DictConfig(config)

    @property
    def config(self) -> DictConfig:
        """Get the configuration object."""
        if self._config is None:
            self._config = self._load_config(self.config_path)
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot notation key."""
        try:
            return OmegaConf.select(self.config, key)
        except Exception:
            return default

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value by dot notation key."""
        OmegaConf.update(self.config, key, value)

    def reload(self) -> None:
        """Reload the configuration file from disk."""
        self._config = self._load_config(self.config_path)
        logger.info(f"Configuration reloaded from {self.config_path}")

    @property
    def db_url(self) -> str:
        """Get database connection URL."""
        db = self.config.database
        return f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"

    @property
    def api_url(self) -> str:
        """Get API base URL."""
        api = self.config.api
        return f"http://{api.host}:{api.port}"

    def get_api_url(self, endpoint: str = "") -> str:
        """Get API base URL or specific endpoint URL."""
        base_url = self.api_url
        return f"{base_url}/{endpoint.lstrip('/')}" if endpoint else base_url
