import importlib
import os
from pathlib import Path

from cvatkit import CvatReader
from django.contrib.gis.geos import GEOSGeometry
from django.contrib.gis.geos import MultiPolygon as GEOSMultiPolygon
from django.db import transaction
from shapely import wkt as shapely_wkt
from shapely.errors import TopologicalError
from shapely.geometry import Polygon as ShpPolygon
from sqlalchemy import create_engine

from ppcollapse import setup_logger
from ppcollapse.setup_django_ppcx import setup_django
from ppcollapse.utils.config import ConfigManager

# Allow Django ORM to work in Jupyter's async environment
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

# Setup Django
config = ConfigManager(config_path="config.yaml")
setup_django(db_config=config.get("database"))

logger = setup_logger(level="INFO", name="ppcx")
config = ConfigManager(config_path="config.yaml")
db_engine = create_engine(config.db_url)

# Import Django models
ppcx_app_models = importlib.import_module("ppcx_app.models")
Collapse = ppcx_app_models.Collapse
Image = ppcx_app_models.Image


def _find_image(
    name: str,
    search_mode: str = "auto",
    time_tolerance_seconds: int = 300,
) -> Image | None:
    """
    Find an Image in the database by filename or datetime.

    Args:
        name: Image filename (e.g., "PPCX_2_2017_10_09_12_02_02_REG.jpg")
        search_mode: How to search for the image. Options:
            - "auto": Try filename -> exact timestamp -> closest within tolerance
            - "filename": Only by filename
            - "exact": Only by exact timestamp parsed from filename
            - "closest": Find closest image by timestamp (within tolerance)
        time_tolerance_seconds: Maximum time difference (in seconds) when searching
                                for closest image. Default: 300 (5 minutes)

    Returns:
        Image object if found, None otherwise
    """
    import re
    from datetime import datetime, timedelta

    basename = Path(name).name

    # MODE: filename only
    if search_mode == "auto" or search_mode == "filename":
        # Try to match by filename
        qs = Image.objects.filter(file_path__endswith=basename)
        if qs.exists():
            return qs.first()

        # Try containing filename (in case of subdirs)
        qs = Image.objects.filter(file_path__icontains=basename)
        if qs.exists():
            return qs.first()

        if search_mode == "filename":
            return None

    # Parse datetime from filename
    # Example: PPCX_2_2017_10_09_12_02_02_REG.jpg
    m = re.search(r"_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_", basename)
    if not m:
        return None

    y, mo, d, h, mi, s = map(int, m.groups())
    dt = datetime(y, mo, d, h, mi, s)

    # MODE: exact timestamp only
    if search_mode == "auto" or search_mode == "exact":
        qs = Image.objects.filter(
            acquisition_timestamp__year=dt.year,
            acquisition_timestamp__month=dt.month,
            acquisition_timestamp__day=dt.day,
            acquisition_timestamp__hour=dt.hour,
            acquisition_timestamp__minute=dt.minute,
            acquisition_timestamp__second=dt.second,
        )
        if qs.exists():
            return qs.first()

        if search_mode == "exact":
            return None

    # MODE: closest within tolerance
    if search_mode == "auto" or search_mode == "closest":
        # Search within tolerance window
        time_delta = timedelta(seconds=time_tolerance_seconds)
        qs = Image.objects.filter(
            acquisition_timestamp__gte=dt - time_delta,
            acquisition_timestamp__lte=dt + time_delta,
        )
        if not qs.exists():
            return None

        # Find closest
        nearest = min(
            qs, key=lambda im: abs((im.acquisition_timestamp - dt).total_seconds())
        )
        time_diff = abs((nearest.acquisition_timestamp - dt).total_seconds())

        if time_diff <= time_tolerance_seconds:
            logger.info(
                f"Found image by closest match: {basename} -> "
                f"Image ID {nearest.id} (Δt = {time_diff:.0f}s)"
            )
            return nearest

        return None

    # Invalid search mode
    raise ValueError(
        f"Invalid search_mode: {search_mode}. "
        f"Must be one of: 'auto', 'filename', 'exact', 'closest'"
    )


@transaction.atomic
def create_collapses_from_cvat(
    cvat_path: str | Path,
    *,
    label_name: str = "collapse",
    image_search_mode: str = "auto",
    time_tolerance_seconds: int = 300,
    default_volume: float | None = None,
    remove_reg_suffix: bool = True,
    dry_run: bool = True,
) -> list[int]:
    """
    Create Collapse objects from CVAT annotations.

    Args:
        cvat_path: Path to CVAT XML file or ZIP archive
        label_name: Label name to filter polygons (default: "collapse")
        image_search_mode: How to search for images - "auto", "filename", "exact", "closest"
        time_tolerance_seconds: Max time difference for closest match (default: 300)
        default_volume: Default volume if not specified in CVAT (default: None)
        remove_reg_suffix: Remove "_REG" suffix from image names (default: True)
        dry_run: If True, don't save to database (default: True)

    Returns:
        List of created Collapse IDs
    """
    # Read CVAT annotations using the new module
    reader = CvatReader(cvat_path)

    created_ids: list[int] = []
    skipped_count = 0
    processed_count = 0

    # Iterate over polygons with the specified label
    for polygon in reader.iter_polygons(label=label_name):
        processed_count += 1

        # Get image name and optionally remove suffix
        name = polygon.image_name
        if remove_reg_suffix:
            name = name.replace("_REG", "")

        # Find matching image in database
        img = _find_image(
            name,
            search_mode=image_search_mode,
            time_tolerance_seconds=time_tolerance_seconds,
        )

        if img is None:
            logger.warning(f"Image not found for {polygon.image_name} – skipping.")
            skipped_count += 1
            continue

        # Build shapely polygon from points
        try:
            shp = ShpPolygon(polygon.points)
            if not shp.is_valid:
                shp = shp.buffer(0)
        except TopologicalError:
            logger.warning(f"Invalid polygon for {polygon.image_name} – skipping.")
            skipped_count += 1
            continue

        if shp.is_empty or shp.area <= 0:
            logger.warning(
                f"Empty/zero-area polygon for {polygon.image_name} – skipping."
            )
            skipped_count += 1
            continue

        # Convert to Django geometry
        wkt_str = shapely_wkt.dumps(shp, rounding_precision=3)
        geos = GEOSGeometry(wkt_str, srid=0)

        # Convert to MultiPolygon if needed
        if geos.geom_type == "Polygon":
            geos = GEOSMultiPolygon(geos, srid=0)

        # Compute area in pixels
        area_px2 = float(shp.area)

        # Get volume from CVAT annotation or use default
        volume = polygon.attributes.get("volume", default_volume)
        if volume is not None:
            try:
                volume = float(volume)
            except (ValueError, TypeError):
                volume = default_volume

        # Create Collapse object
        collapse = Collapse(
            image=img,
            geom=geos,
            date=img.acquisition_timestamp.date(),
            area=area_px2,
            volume=volume,
        )

        if dry_run:
            volume_str = f"{volume:.1f}" if volume is not None else "None"
            logger.info(
                f"[DRY RUN] Would create Collapse for image id={img.id} "
                f"({polygon.image_name}) with area={area_px2:.1f} px², "
                f"volume={volume_str} m³"
            )
            continue

        collapse.save()
        created_ids.append(collapse.id)
        logger.info(f"Created Collapse id={collapse.id} for image id={img.id}")

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Processed: {processed_count} polygons")
    logger.info(f"Created:   {len(created_ids)} collapses")
    logger.info(f"Skipped:   {skipped_count} polygons")
    logger.info("=" * 60)

    return created_ids


if __name__ == "__main__":
    # Example usage
    cvat_file = "data/annotations.zip"  # or "data/annotations.xml"

    created_ids = create_collapses_from_cvat(
        cvat_file,
        label_name="collapse",
        image_search_mode="auto",
        time_tolerance_seconds=300,
        remove_reg_suffix=True,
        dry_run=True,  # Set to False to actually create collapses
    )

    logger.info(f"Created {len(created_ids)} collapse records")
