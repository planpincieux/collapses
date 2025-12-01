import io
import logging
from datetime import datetime
from typing import Literal, overload

import pandas as pd
import requests
from PIL import Image
from sqlalchemy import create_engine, text

from .config import ConfigManager

logger = logging.getLogger("ppcx")

# === helpers ===


def _resolve_api_host_port(
    app_host: str | None,
    app_port: str | None,
    config: ConfigManager | None,
) -> tuple[str, str]:
    """
    Resolve API host/port from explicit args or a ConfigManager.
    Priority: if kwargs (app_host/app_port) are passed, they override config values.
    """
    host = config.get("api.host") if config is not None else None
    port = config.get("api.port") if config is not None else None

    # kwargs override config
    if app_host is not None:
        if config is not None and host is not None and host != app_host:
            logger.debug(
                f"Overriding api.host from config with provided app_host '{app_host}' (config had '{host}')"
            )
        host = app_host
    if app_port is not None:
        if config is not None and port is not None and str(port) != str(app_port):
            logger.debug(
                f"Overriding api.port from config with provided app_port '{app_port}' (config had '{port}')"
            )
        port = app_port

    if host is None or port is None:
        raise ValueError(
            "API host/port not provided. Pass app_host/app_port or a ConfigManager with api.host/api.port."
        )
    return str(host), str(port)


# === DIC ===


def fetch_dic_analysis_ids(
    db_engine,
    *,
    reference_date: str | datetime | None = None,
    reference_date_start: str | datetime | None = None,
    reference_date_end: str | datetime | None = None,
    master_timestamp: str | datetime | None = None,
    master_timestamp_start: str | datetime | None = None,
    master_timestamp_end: str | datetime | None = None,
    slave_timestamp: str | datetime | None = None,
    slave_timestamp_start: str | datetime | None = None,
    slave_timestamp_end: str | datetime | None = None,
    camera_id: int | None = None,
    camera_name: str | None = None,
    dt_hours: int | None = None,
    dt_hours_min: int | None = None,
    dt_hours_max: int | None = None,
    month: int | None = None,
) -> list[int]:
    """
    Search for DIC analysis IDs in the database with flexible filtering options.
    Uses direct SQL execution via SQLAlchemy connection and returns list[int].

    See docstring in original implementation for detailed parameter description.
    """

    def _sql_literal(val):
        """Return a safe SQL literal for simple types (int, float, datetime/date, str)."""
        if isinstance(val, (int, float)):
            return str(val)
        if isinstance(val, datetime):
            return f"'{val.isoformat()}'"
        s = str(val)
        # escape single quotes for SQL
        s = s.replace("'", "''")
        return f"'{s}'"

    # Keep a dict of provided criteria for logging
    criteria: dict[str, object] = {}
    if reference_date is not None:
        criteria["reference_date"] = reference_date
    if reference_date_start is not None:
        criteria["reference_date_start"] = reference_date_start
    if reference_date_end is not None:
        criteria["reference_date_end"] = reference_date_end
    if master_timestamp is not None:
        criteria["master_timestamp"] = master_timestamp
    if master_timestamp_start is not None:
        criteria["master_timestamp_start"] = master_timestamp_start
    if master_timestamp_end is not None:
        criteria["master_timestamp_end"] = master_timestamp_end
    if slave_timestamp is not None:
        criteria["slave_timestamp"] = slave_timestamp
    if slave_timestamp_start is not None:
        criteria["slave_timestamp_start"] = slave_timestamp_start
    if slave_timestamp_end is not None:
        criteria["slave_timestamp_end"] = slave_timestamp_end
    if camera_id is not None:
        criteria["camera_id"] = camera_id
    if camera_name is not None:
        criteria["camera_name"] = camera_name
    if dt_hours is not None:
        criteria["dt_hours"] = dt_hours
    if dt_hours_min is not None:
        criteria["dt_hours_min"] = dt_hours_min
    if dt_hours_max is not None:
        criteria["dt_hours_max"] = dt_hours_max
    if month is not None:
        criteria["month"] = month

    query = """
    SELECT 
        DIC.id as dic_id
    FROM ppcx_app_dic DIC
    JOIN ppcx_app_image IMG ON DIC.master_image_id = IMG.id
    JOIN ppcx_app_camera CAM ON IMG.camera_id = CAM.id
    WHERE 1=1
    """

    # Append filters (use safe literals)
    if reference_date is not None:
        query += f" AND DATE(DIC.reference_date) = {_sql_literal(reference_date)}"
    if reference_date_start is not None:
        query += (
            f" AND DATE(DIC.reference_date) >= {_sql_literal(reference_date_start)}"
        )
    if reference_date_end is not None:
        query += f" AND DATE(DIC.reference_date) <= {_sql_literal(reference_date_end)}"

    if master_timestamp is not None:
        query += f" AND DIC.master_timestamp = {_sql_literal(master_timestamp)}"
    if master_timestamp_start is not None:
        query += f" AND DIC.master_timestamp >= {_sql_literal(master_timestamp_start)}"
    if master_timestamp_end is not None:
        query += f" AND DIC.master_timestamp <= {_sql_literal(master_timestamp_end)}"

    if slave_timestamp is not None:
        query += f" AND DIC.slave_timestamp = {_sql_literal(slave_timestamp)}"
    if slave_timestamp_start is not None:
        query += f" AND DIC.slave_timestamp >= {_sql_literal(slave_timestamp_start)}"
    if slave_timestamp_end is not None:
        query += f" AND DIC.slave_timestamp <= {_sql_literal(slave_timestamp_end)}"

    if camera_id is not None:
        query += f" AND CAM.id = {_sql_literal(camera_id)}"
    if camera_name is not None:
        query += f" AND CAM.camera_name = {_sql_literal(camera_name)}"

    if dt_hours is not None:
        query += f" AND DIC.dt_hours = {_sql_literal(dt_hours)}"
    if dt_hours_min is not None:
        query += f" AND DIC.dt_hours >= {_sql_literal(dt_hours_min)}"
    if dt_hours_max is not None:
        query += f" AND DIC.dt_hours <= {_sql_literal(dt_hours_max)}"

    if month is not None:
        query += f" AND EXTRACT(MONTH FROM DIC.reference_date) = {_sql_literal(month)}"

    query += " ORDER BY DIC.master_timestamp"

    try:
        with db_engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
    except Exception as exc:
        logger.exception(
            f"Database query for fetch_dic_analysis_ids failed. Criteria: {criteria}. Error: {exc}",
        )
        return []

    if not rows:
        logger.info(
            f"No DIC analyses found for the given criteria. Criteria: {criteria}"
        )
        return []

    dic_ids = [int(row[0]) for row in rows]
    logger.debug(f"Found {len(dic_ids)} DIC analyses matching criteria: {criteria}")
    return dic_ids


def get_dic_analysis_by_ids(
    db_engine,
    dic_ids: list[int] | int,
    query: str | None = None,
) -> pd.DataFrame:
    """
    Fetch DIC analysis metadata by a list of DIC IDs.

    Allows the user to provide a custom SQL SELECT query to retrieve additional or different fields.
    The query must contain a WHERE DIC.id IN %s clause for filtering.

    Args:
        db_engine: SQLAlchemy database engine
        dic_ids: List of DIC analysis IDs or a single ID
        query: Optional custom SQL SELECT query. If None, uses the default query.

    Returns:
        DataFrame with the requested fields.

    Example:
        custom_query = '''
            SELECT DIC.id as dic_id, DIC.reference_date, DIC.software_used, CAM.camera_name
            FROM ppcx_app_dic DIC
            JOIN ppcx_app_image IMG ON DIC.master_image_id = IMG.id
            JOIN ppcx_app_camera CAM ON IMG.camera_id = CAM.id
            WHERE DIC.id IN %s
        '''
        get_dic_analysis_by_ids(db_engine, dic_ids, query=custom_query)
    """
    if isinstance(dic_ids, int):
        dic_ids = [dic_ids]
    params = [tuple(dic_ids)]
    if query is None:
        query = """
        SELECT 
            DIC.id as dic_id,
            DIC.reference_date,
            CAM.camera_name,
            DIC.master_timestamp,
            DIC.slave_timestamp,
            DIC.master_image_id,
            DIC.slave_image_id,
            DIC.dt_hours
        FROM ppcx_app_dic DIC
        JOIN ppcx_app_image IMG ON DIC.master_image_id = IMG.id
        JOIN ppcx_app_camera CAM ON IMG.camera_id = CAM.id
        WHERE DIC.id IN %s
        """
    return pd.read_sql(query, db_engine, params=tuple(params))


def get_dic_data(
    dic_id: int,
    *,
    app_host: str | None = None,
    app_port: str | None = None,
    config: ConfigManager | None = None,
) -> pd.DataFrame:
    """
    Fetch DIC displacement data from the Django API endpoint as a DataFrame.

    Provide either app_host/app_port explicitly or a ConfigManager (config) containing api.host/api.port.

    Args:
        dic_id: DIC analysis ID
        app_host: API host (optional, defaults to config)
        app_port: API port (optional, defaults to config)
        config: ConfigManager (optional, used if app_host/app_port not provided)
    Returns:
        DataFrame with columns [x, y, u, v, V]

    """
    host, port = _resolve_api_host_port(app_host, app_port, config)
    url = f"http://{host}:{port}/API/dic/{dic_id}/"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Could not fetch DIC data for id {dic_id}: {response.text}")

    data = response.json()
    # If the data is empty, raise an error
    if not data or "points" not in data:
        raise ValueError(f"No valid DIC data found for id {dic_id}")

    points = data["points"]
    vectors = data["vectors"]
    magnitudes = data["magnitudes"]

    # Convert to DataFrame
    df = pd.DataFrame(points, columns=["x", "y"])
    df["u"] = [v[0] for v in vectors]
    df["v"] = [v[1] for v in vectors]
    df["V"] = magnitudes

    return df


@overload
def get_multi_dic_data(
    dic_ids: list[int],
    *,
    stack_results: Literal[False],
    app_host: str | None = None,
    app_port: str | None = None,
    config: ConfigManager | None = None,
) -> dict[int, pd.DataFrame]: ...


@overload
def get_multi_dic_data(
    dic_ids: list[int],
    *,
    stack_results: Literal[True],
    app_host: str | None = None,
    app_port: str | None = None,
    config: ConfigManager | None = None,
) -> pd.DataFrame: ...


def get_multi_dic_data(
    dic_ids: list[int],
    *,
    stack_results: bool = False,
    app_host: str | None = None,
    app_port: str | None = None,
    config: ConfigManager | None = None,
) -> pd.DataFrame | dict[int, pd.DataFrame]:
    """
    Fetch and concatenate DIC displacement data for multiple DIC IDs.
    Returns a single DataFrame with all DIC data if stack_results is True, otherwise returns a dictionary of DataFrames.

    Args:
        dic_ids: List of DIC analysis IDs
        app_host: API host (optional, defaults to config)
        app_port: API port (optional, defaults to config)
        config: ConfigManager (optional, used if app_host/app_port not provided)

    Returns:
        DataFrame or dict: Concatenated DataFrame with all DIC data if stack_results is True, otherwise a dictionary of DataFrames.
    """
    host, port = _resolve_api_host_port(app_host, app_port, config)

    df_dic: dict[int, pd.DataFrame] = {}
    for dic_id in dic_ids:
        try:
            df = get_dic_data(dic_id, app_host=host, app_port=port)
            df_dic[dic_id] = df
            logger.info(f"Fetched DIC data for id {dic_id} with {len(df)} points")
        except ValueError as e:
            logger.warning(f"Skipping DIC id {dic_id}: {e}")

    if not df_dic:
        raise ValueError("No valid DIC data found for the provided IDs")

    if stack_results:
        df_stack = pd.concat(df_dic.values(), ignore_index=True)
        logger.info(f"Total concatenated DIC data points: {len(df_stack)}")
        return df_stack
    else:
        return df_dic


# === Images ===


def fetch_image_ids(
    db_engine,
    date: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    time_of_day: str | None = None,
    time_start: str | None = None,
    time_end: str | None = None,
    limit: int | None = None,
    order_by: str | None = None,
) -> list[int]:
    """Fetch image IDs from the database based on acquisition date and time criteria."""

    query = """
    SELECT id
    FROM ppcx_app_image
    WHERE 1=1
    """

    if date:
        query += f" AND acquisition_timestamp::date = '{date}'"
    if date_start:
        query += f" AND acquisition_timestamp::date >= '{date_start}'"
    if date_end:
        query += f" AND acquisition_timestamp::date <= '{date_end}'"
    if time_of_day:
        query += f" AND to_char(acquisition_timestamp, 'HH24:MI:SS') = '{time_of_day}'"
    if time_start:
        query += f" AND to_char(acquisition_timestamp, 'HH24:MI:SS') >= '{time_start}'"
    if time_end:
        query += f" AND to_char(acquisition_timestamp, 'HH24:MI:SS') <= '{time_end}'"
    if order_by:
        query += f" ORDER BY {order_by}"
    if limit:
        query += f" LIMIT {limit}"

    with db_engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()

    if not rows:
        raise ValueError("No images found matching the criteria.")

    else:
        rows = [row[0] for row in rows]

    return rows


def fetch_image_metadata_by_ids(
    db_engine,
    image_id: list[int] | int,
) -> pd.DataFrame:
    """Fetch image metadata from the database based on image ID."""

    if isinstance(image_id, int):
        image_id = [image_id]
    params = [tuple(image_id)]
    query = """
    SELECT *
    FROM ppcx_app_image
    WHERE id IN %s
    """
    df = pd.read_sql(query, db_engine, params=tuple(params))

    if df.empty:
        raise (ValueError("No metadata found for the given image ID."))

    return df


def get_image(
    image_id: int,
    *,
    app_host: str | None = None,
    app_port: str | None = None,
    config: ConfigManager | None = None,
) -> Image.Image:
    """
    Get an image by ID from the API and rotate if from Tele camera.
    Provide either app_host/app_port or a ConfigManager (config).
    """
    host, port = _resolve_api_host_port(app_host, app_port, config)
    url = f"http://{host}:{port}/API/images/{image_id}/"
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(io.BytesIO(response.content))

        # Get camera name to check if rotation is needed
        camera_name = None

        if config is not None:
            db_engine = create_engine(config.db_url)
            query = f"""
                SELECT camera_name
                FROM ppcx_app_image as image
                JOIN ppcx_app_camera as camera
                ON image.camera_id = camera.id
                WHERE image.id={image_id};
            """
            with db_engine.connect() as conn:
                result = conn.execute(text(query))
                camera_name = result.scalar_one_or_none()
        else:
            logger.warning(
                "ConfigManager not provided, cannot check camera name for rotation."
            )
        # Rotate if camera_name is Tele (portrait mode)
        if camera_name is not None and "tele" in camera_name.lower():
            img = img.rotate(90, expand=True)  # 90° clockwise
        return img
    else:
        raise ValueError(f"Image with ID {image_id} not found.")


# === Collapse Data ===


def get_collapses_df(db_engine) -> pd.DataFrame:
    """Read all collapse records from DB and return dataframe with parsed dates."""
    query = """
        SELECT c.id, img.acquisition_timestamp::date AS date,
               c.image_id, ST_AsText(c.geom) AS geom_wkt,
               c.area, c.volume
        FROM ppcx_app_collapse c
        JOIN ppcx_app_image img ON c.image_id = img.id
        ORDER BY img.acquisition_timestamp DESC, c.id ASC
    """
    df = pd.read_sql(query, db_engine, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_collapses_by_date(engine, date: str, limit: int | None = None) -> pd.DataFrame:
    """Return collapse rows. If date provided, filter by image acquisition date (date string YYYY-MM-DD)."""
    query = """
        SELECT c.id, img.acquisition_timestamp::date AS date,
            c.image_id, ST_AsText(c.geom) AS geom_wkt,
            c.area, c.volume
        FROM ppcx_app_collapse c
        JOIN ppcx_app_image img ON c.image_id = img.id
        WHERE img.acquisition_timestamp::date = %s
        ORDER BY img.acquisition_timestamp DESC, c.id ASC
    """
    if limit is not None:
        query += f" LIMIT {limit}"

    df = pd.read_sql(query, engine, params=(date,))
    return df
