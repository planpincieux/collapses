import importlib
from pathlib import Path

import fiona
import numpy as np
import pandas as pd
from django.contrib.gis.geos import GEOSGeometry
from shapely import wkt as shapely_wkt
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from tqdm import tqdm

from ppcollapse import logger
from ppcollapse.utils.config import ConfigManager
from setup_django_ppcx import get_django_app_dir, setup_django

shape_dir = Path("data/SHAPEFILES_adj")
file_ext = ".shp"
volume_file_dir = Path("data/crolli")

# Load configuration
config = ConfigManager(config_path="config.yaml")

# Configure Django settings before any Django imports
setup_django(django_app_dir=get_django_app_dir(), db_config=config.get("database"))

# Import Django models after setting up Django using importlib
ppcx_app_models = importlib.import_module("ppcx_app.models")
Collapse = ppcx_app_models.Collapse
Image = ppcx_app_models.Image


def read_shapely_geom_from_file(path: Path, invert_y: bool = False) -> BaseGeometry:
    """Read first feature from a shapefile, flip Y (QGIS -> image coords), return (geom, area, perimeter)."""
    with fiona.open(path) as src:
        polygons = [feature["geometry"] for feature in src]
    if not polygons:
        raise ValueError(f"No features found in {path}")
    poly = polygons[0]
    geom = shape(poly)

    if invert_y:
        # Invert Y axis because the polygons were created in qgis
        geom = transform(
            lambda x, y, z=None: (x, -y) if z is None else (x, -y, z), geom
        )

    return geom


def shapely_to_django_polygon(geom: BaseGeometry) -> GEOSGeometry:
    """Convert Shapely geometry to Django GEOSGeometry Polygon with SRID=0."""
    wkt = shapely_wkt.dumps(geom)
    # Create Django Polygon from WKT with SRID=0 (image coordinates)
    django_geom = GEOSGeometry(wkt, srid=0)
    return django_geom


def create_collapse_from_geom(
    image: Image,
    geom: BaseGeometry,
    area: float | None = None,
    volume: float | None = None,
) -> Collapse:
    """Create a Collapse instance using Django ORM. Returns created Collapse instance."""
    # Convert Shapely geometry to Django GEOSGeometry
    django_geom = shapely_to_django_polygon(geom)

    # Create Collapse instance
    collapse = Collapse(
        image=image,
        geom=django_geom,
        area=float(area) if area is not None else None,
        volume=float(volume) if volume is not None else None,
    )
    collapse.save()

    return collapse


def read_collapse_volume_file(path: Path) -> pd.DataFrame:
    import re

    # regex: 6 mandatory columns then optional trailing flag that starts with %
    # allow numeric type or literal "NaN" for missing type
    pattern = re.compile(
        r"^\s*(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+((?:\d+|NaN))\s+(\d+)(?:\s+(%.*))?\s*$",
        re.IGNORECASE,
    )
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.lstrip().startswith("%"):
            continue
        m = pattern.match(ln)
        if m:
            year, month, day_start, day_end, typ, size, flag = m.groups()
        else:
            parts = ln.split()
            if len(parts) < 6:
                raise ValueError(f"Unparseable line: {ln!r}")
            year, month, day_start, day_end, typ, size = parts[:6]
            flag = None
            if len(parts) > 6:
                tail = " ".join(parts[6:])
                flag = tail if tail.startswith("%") else None

        # normalize type: treat NaN (any case) as missing -> None
        typ_val = None if isinstance(typ, str) and typ.lower() == "nan" else int(typ)

        rows.append(
            {
                "date": pd.to_datetime(
                    f"{year}-{int(month):02d}-{int(day_end):02d}", format="%Y-%m-%d"
                ),
                "year": int(year),
                "month": int(month),
                "day_start": int(day_start),
                "day_end": int(day_end),
                "type": typ_val,
                "volume": int(size),
                "flag_raw": flag,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    for year_dir in sorted(shape_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        files = sorted(year_dir.glob(f"*{file_ext}"))
        logger.info(f"Processing year directory: {year_dir}, found {len(files)} files")

        # Check if a volume file exists
        year = year_dir.name.split("_")[0]
        volume_file = volume_file_dir / f"crolli_{year}.txt"
        collapse_volume_df = None
        if volume_file.exists():
            try:
                collapse_volume_df = read_collapse_volume_file(volume_file)
            except Exception as e:
                logger.error(f"Error reading volume file {volume_file}: {e}")
                raise e
            has_volume_file = True
        else:
            has_volume_file = False
        if not has_volume_file:
            logger.warning(
                f"No volume file found for year {year}, proceeding without volume data."
            )

        if not files:
            logger.warning(f"No shapefiles found in {year_dir}, skipping...")
            continue

        for file in tqdm(files, desc="Processing shapefiles"):
            logger.debug(f"Reading {file}...")

            # Skip shapefile with _vol in the name (they don't have geometry)
            if "_vol" in file.name:
                logger.info(f"Skipping {file} (volume shapefile)")
                continue

            # Extract date from filename
            try:
                date_str = file.stem.split("-")[0]
                date = pd.to_datetime(date_str, format="%Y%m%d").date()
            except Exception as e:
                logger.error(f"Error parsing date from {file}: {e}")
                raise ValueError(
                    f"Filename {file.name} does not match expected format."
                ) from e
                # continue

            # Fetch images for the given date using Django ORM, ordered by acquisition time descending
            try:
                images = Image.objects.filter(
                    acquisition_timestamp__date=date
                ).order_by("-acquisition_timestamp")

                if not images.exists():
                    logger.warning(f"No images found for date {date}, skipping {file}")
                    continue

                # Pick up center image in the list
                idx = len(images) // 2
                image = images[idx]
                logger.debug(
                    f"Selected image ID: {image.id} - {image.acquisition_timestamp}"
                )
            except Exception as e:
                logger.error(f"Error fetching images for {file}: {e}")
                continue

            # Read geometry from file
            try:
                geom = read_shapely_geom_from_file(file, invert_y=True)
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
                continue

            # Get volume from collapse_volume_df if available
            volume = None
            if has_volume_file and collapse_volume_df is not None:
                match = collapse_volume_df.loc[
                    (collapse_volume_df["date"].dt.date == date)
                ]
                if not match.empty:
                    volume = float(match.iloc[0]["volume"])
                    logger.debug(f"Found volume {volume} for date {date}")
                else:
                    logger.debug(f"No volume entry found for date {date}")

            # Insert geometry into DB using Django ORM
            area = np.round(geom.area, 5)
            collapse = create_collapse_from_geom(
                image=image,
                geom=geom,
                area=area,
                volume=volume,
            )
            logger.debug(f"Inserted collapse id: {collapse.id}")
