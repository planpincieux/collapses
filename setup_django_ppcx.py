"""
Reusable Django setup utility for standalone scripts.

This module provides a function to configure Django for use in standalone
Python scripts that need access to Django models and ORM functionality.
"""

import os
import sys
from pathlib import Path
from typing import Optional

PPCX_APP_DIR = Path("/home/francesco/dati/ppcx/ppcx-app")


def setup_django(
    django_app_dir: Optional[Path] = None,
    settings_module: str = "planpincieux.settings",
    db_config: Optional[dict] = None,
) -> None:
    """
    Configure and initialize Django for standalone script execution.

    This function must be called before importing any Django models or using
    Django's ORM functionality in standalone scripts.

    Args:
        django_app_dir: Path to the Django app directory. If None, uses default
                       path: PPCX_APP_DIR / "app".
        settings_module: Django settings module path (default: "planpincieux.settings")
        db_config: Optional database configuration dictionary with keys:
                  - host: Database host
                  - port: Database port
                  - name: Database name
                  - user: Database user
                  - password: Database password (otherwise try to read from secrets).
                  If provided, these will override environment variables.

    Raises:
        FileNotFoundError: If django_app_dir doesn't exist
        RuntimeError: If Django is already configured

    Example:
        >>> from django_setup import setup_django
        >>> from ppcollapse.utils.config import ConfigManager
        >>>
        >>> config = ConfigManager(config_path="config.yaml")
        >>> db_config = config.get("database")
        >>> setup_django(db_config=db_config)
        >>>
        >>> # Now you can import Django models
        >>> from ppcx_app.models import Collapse, Image
    """
    import django

    # Check if Django is already configured
    if django.apps.apps.ready:
        raise RuntimeError(
            "Django is already configured. setup_django() should only be called once."
        )

    # Set default Django app directory if not provided
    if django_app_dir is None:
        django_app_dir = Path(PPCX_APP_DIR) / "app"

    # Validate Django app directory exists
    if not django_app_dir.exists():
        raise FileNotFoundError(f"Django app directory not found: {django_app_dir}")

    # Add Django app directory to Python path
    django_app_dir_str = str(django_app_dir)
    if django_app_dir_str not in sys.path:
        sys.path.insert(0, django_app_dir_str)

    # Set Django settings module
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)

    # Override database configuration if provided
    if db_config:
        if "host" in db_config:
            os.environ["DB_HOST"] = db_config["host"]
        if "port" in db_config:
            os.environ["DB_PORT"] = str(db_config["port"])
        if "name" in db_config:
            os.environ["DB_NAME"] = db_config["name"]
        if "user" in db_config:
            os.environ["DB_USER"] = db_config["user"]
        if "password" in db_config:
            os.environ["DB_PASSWORD"] = db_config["password"]

    # Initialize Django
    django.setup()


def get_django_app_dir() -> Path:
    """
    Get the default Django app directory path.

    Returns:
        Path object pointing to the Django app directory
    """
    return Path(PPCX_APP_DIR) / "app"
