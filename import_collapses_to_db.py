import importlib
import re
from pathlib import Path

import fiona
import numpy as np
import pandas as pd
from django.contrib.gis.geos import GEOSGeometry
from shapely import wkt as shapely_wkt
from shapely.geometry import MultiPolygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from tqdm import tqdm

from ppcollapse import logger
from ppcollapse.utils.config import ConfigManager
from setup_django_ppcx import get_django_app_dir, setup_django

shape_dir = Path("data/import")
file_ext = ".shp"

# Load configuration
config = ConfigManager(config_path="config.yaml")

# Configure Django settings before any Django imports
setup_django(django_app_dir=get_django_app_dir(), db_config=config.get("database"))

# Import Django models after setting up Django using importlib
ppcx_app_models = importlib.import_module("ppcx_app.models")
Collapse = ppcx_app_models.Collapse
Image = ppcx_app_models.Image


def read_all_geometries_from_file(path: Path, invert_y: bool = False) -> BaseGeometry:
    """
    Read ALL features from a shapefile and return as MultiPolygon.
    Flips Y coordinates if invert_y=True (QGIS -> image coords).

    Returns:
        MultiPolygon containing all geometries from the file
    """
    with fiona.open(path) as src:
        features = list(src)

    if not features:
        raise ValueError(f"No features found in {path}")

    polygons = []
    for feature in features:
        geom = shape(feature["geometry"])

        # Ensure we're working with polygons
        if geom.geom_type == "Polygon":
            polygons.append(geom)
        elif geom.geom_type == "MultiPolygon":
            polygons.extend(list(geom.geoms))
        else:
            logger.warning(f"Skipping non-polygon geometry type: {geom.geom_type}")

    if not polygons:
        raise ValueError(f"No polygon geometries found in {path}")

    # Create MultiPolygon from all polygons
    multi_geom = MultiPolygon(polygons)

    if invert_y:
        # Invert Y axis because the polygons were created in QGIS
        multi_geom = transform(
            lambda x, y, z=None: (x, -y) if z is None else (x, -y, z), multi_geom
        )

    logger.debug(f"Read {len(polygons)} polygon(s) from {path.name}")
    return multi_geom


def shapely_to_django_multipolygon(geom: BaseGeometry) -> GEOSGeometry:
    """Convert Shapely geometry to Django GEOSGeometry MultiPolygon with SRID=0."""
    # Ensure it's a MultiPolygon
    if geom.geom_type == "Polygon":
        geom = MultiPolygon([geom])
    elif geom.geom_type != "MultiPolygon":
        raise ValueError(f"Expected Polygon or MultiPolygon, got {geom.geom_type}")

    wkt = shapely_wkt.dumps(geom)
    # Create Django MultiPolygon from WKT with SRID=0 (image coordinates)
    django_geom = GEOSGeometry(wkt, srid=0)
    return django_geom


def read_collapse_volume_file(path: Path) -> pd.DataFrame:
    """
    Read collapse volume file with format:
    year month day_before_collapse day_after_collapse collapse_type volume

    Returns DataFrame with date_after as the primary date field.
    """
    pattern = re.compile(
        r"^\s*(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+((?:\d+|NaN))\s+(\d+)(?:\s+(%.*))?\s*$",
        re.IGNORECASE,
    )
    rows = []

    for ln in path.read_text(encoding="utf-8").splitlines():
        # Skip empty lines and comments
        if not ln.strip() or ln.lstrip().startswith("%"):
            continue

        m = pattern.match(ln)
        if not m:
            logger.warning(f"Unparseable line: {ln!r}")
            continue

        year, month, day_before, day_after, typ, volume, flag = m.groups()

        # Parse type: treat NaN (any case) as missing -> None
        typ_val = None if isinstance(typ, str) and typ.lower() == "nan" else int(typ)

        # Handle month overflow (e.g., September 31 -> October 1)
        try:
            date_after = pd.to_datetime(
                f"{year}-{int(month):02d}-{int(day_after):02d}", format="%Y-%m-%d"
            )
        except ValueError:
            # Invalid date (e.g., Sept 31), adjust to next month
            month_int = int(month)
            year_int = int(year)
            day_after_int = int(day_after)

            if month_int == 12 and day_after_int > 31:
                # Roll over to next year
                date_after = pd.to_datetime(
                    f"{year_int + 1}-01-{day_after_int - 31:02d}"
                )
            else:
                # Roll over to next month
                days_in_month = pd.Period(f"{year_int}-{month_int:02d}").days_in_month
                overflow_days = day_after_int - days_in_month
                date_after = pd.to_datetime(
                    f"{year_int}-{month_int + 1:02d}-{overflow_days:02d}"
                )
            logger.debug(
                f"Adjusted invalid date {year}-{month}-{day_after} to {date_after.date()}"
            )

        rows.append(
            {
                "date_after": date_after,
                "year": int(year),
                "month": int(month),
                "day_before": int(day_before),
                "day_after": int(day_after),
                "collapse_type": typ_val,
                "volume": int(volume),
                "flag_raw": flag,
            }
        )

    return pd.DataFrame(rows)


def parse_shapefile_name(filename: str) -> tuple[pd.Timestamp, int]:
    """
    Parse shapefile name to extract date (after collapse) and suffix number.

    Examples:
        20190701.shp -> (2019-07-01, 1)
        20190724-2.shp -> (2019-07-24, 2)
        20190726-3.shp -> (2019-07-26, 3)

    Returns:
        tuple of (date_after, suffix_number)
    """
    stem = Path(filename).stem

    # Check for suffix pattern (e.g., 20190724-2)
    match = re.match(r"^(\d{8})(?:-(\d+))?$", stem)
    if not match:
        raise ValueError(
            f"Filename {filename} does not match expected format YYYYMMDD or YYYYMMDD-N"
        )

    date_str, suffix_str = match.groups()
    date_after = pd.to_datetime(date_str, format="%Y%m%d")
    suffix = int(suffix_str) if suffix_str else 1

    return date_after, suffix


if __name__ == "__main__":
    # year_dirs = [Path("data/import_test/2019")]
    # for year_dir in sorted(year_dirs):

    for year_dir in sorted(shape_dir.iterdir()):
        year = year_dir.name

        if not year_dir.is_dir():
            continue

        shape_dir = year_dir / "shapefiles"
        files = sorted(shape_dir.glob(f"*{file_ext}"))
        logger.info(f"Processing year {year}, found {len(files)} files")

        # Check if a volume file exists
        volume_file = year_dir / f"volumi_{year}.txt"
        collapse_volume_df = None
        has_volume_file = False
        if volume_file.exists():
            try:
                collapse_volume_df = read_collapse_volume_file(volume_file)
                has_volume_file = True
                logger.info(
                    f"Loaded volume file with {len(collapse_volume_df)} entries"
                )
            except Exception as e:
                logger.error(f"Error reading volume file {volume_file}: {e}")
                raise
        else:
            logger.warning(
                f"No volume file found for year {year}, proceeding without volume data"
            )

        if not files:
            logger.warning(f"No shapefiles found in {shape_dir}, skipping...")
            continue

        for file in tqdm(files, desc="Processing shapefiles"):
            # Skip shapefile with _vol in the name (they don't have geometry)
            if "_vol" in file.name:
                logger.info(f"Skipping {file} (volume shapefile)")
                continue

            # Parse filename to get date (after collapse) and suffix
            try:
                date_after, suffix = parse_shapefile_name(file.name)
                date_after_date = date_after.date()
            except ValueError as e:
                logger.error(f"Error parsing filename {file.name}: {e}")
                continue

            logger.debug(
                f"Processing {file.name}: date_after={date_after_date}, suffix={suffix}"
            )

            # Fetch image for the date AFTER the collapse
            try:
                images = Image.objects.filter(
                    acquisition_timestamp__date=date_after_date
                ).order_by("-acquisition_timestamp")

                if not images.exists():
                    logger.warning(
                        f"No images found for date {date_after_date}, skipping {file.name}"
                    )
                    continue

                # Pick middle image from the list
                idx = len(images) // 2
                image = images[idx]
                logger.debug(
                    f"Selected image ID: {image.id} - {image.acquisition_timestamp}"
                )
            except Exception as e:
                logger.error(f"Error fetching images for {file.name}: {e}")
                continue

            # Read ALL geometries from shapefile (multiple polygons merged into MultiPolygon)
            try:
                geom = read_all_geometries_from_file(file, invert_y=True)
            except Exception as e:
                logger.error(f"Error reading geometries from {file.name}: {e}")
                continue

            # Get volume from volume file (only for primary collapse, suffix=1)
            volume = None
            if has_volume_file and collapse_volume_df is not None and suffix == 1:
                match = collapse_volume_df.loc[
                    collapse_volume_df["date_after"].dt.date == date_after_date
                ]
                if not match.empty:
                    volume = float(match.iloc[0]["volume"])
                    logger.debug(f"Found volume {volume} m³ for date {date_after_date}")
                else:
                    logger.debug(f"No volume entry found for date {date_after_date}")
            elif suffix > 1:
                logger.debug(f"Suffix={suffix}, volume set to NaN (secondary collapse)")
                volume = np.nan

            # Calculate area
            area = np.round(geom.area, 5)

            # Convert Shapely geometry to Django GEOSGeometry MultiPolygon
            django_geom = shapely_to_django_multipolygon(geom)

            # Create Collapse instance
            collapse = Collapse(
                image=image,
                date=date_after_date,  # Date AFTER collapse (mandatory field)
                geom=django_geom,
                area=float(area) if area is not None else None,
                volume=float(volume)
                if volume is not None and not np.isnan(volume)
                else None,
            )

            try:
                collapse.save()
                logger.info(
                    f"✓ Inserted collapse id={collapse.id}, date={date_after_date}, "
                    f"suffix={suffix}, polygons={len(geom.geoms)}, "
                    f"area={area:.1f} px², volume={volume if volume else 'N/A'}"
                )
            except Exception as e:
                logger.error(f"Error saving collapse for {file.name}: {e}")
                continue
            logger.debug(f"Inserted collapse id: {collapse.id}")
