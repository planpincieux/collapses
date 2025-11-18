from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from shapely import contains_xy
from shapely import wkt as shapely_wkt
from sqlalchemy import create_engine
from tqdm import tqdm

from ppcollapse import setup_logger
from ppcollapse.utils.config import ConfigManager
from ppcollapse.utils.database import (
    fetch_dic_analysis_ids,
    get_collapses_df,
    get_dic_analysis_by_ids,
    get_dic_data,
    get_image,
    get_multi_dic_data,
)

# Use Agg backend for script (non-interactive)
matplotlib.use("Agg")

logger = setup_logger(level="WARNING", name="ppcx")

# -------------------------
# PARAMETERS (edit here)
# -------------------------
CONFIG_PATH: str | Path = "config.yaml"
DAYS_BEFORE = 10
OUTPUT_DIR = Path("output/collapses_timeseries")
N_JOBS = 1  # number of parallel jobs
VELOCITY_YLIM = (0, 20)  # fixed y axis limits for velocity plot, or None
# -------------------------


def fetch_dic_before(
    config: ConfigManager, collapse_date: str, days_before: int, **kwargs
):
    """Fetch DIC analyses in the window [collapse_date - days_before, collapse_date]."""

    engine = create_engine(config.db_url)
    start_date = pd.to_datetime(collapse_date) - pd.Timedelta(days=days_before)
    start_date_str = start_date.strftime("%Y-%m-%d")
    dic_ids = fetch_dic_analysis_ids(
        db_engine=engine,
        reference_date_start=start_date_str,
        reference_date_end=collapse_date,
        **kwargs,
    )
    if len(dic_ids) == 0:
        return pd.DataFrame(), {}

    # get dic metadata and sort by date
    dic_metadata = get_dic_analysis_by_ids(dic_ids=dic_ids, db_engine=engine)
    dic_metadata["reference_date"] = pd.to_datetime(dic_metadata["reference_date"])
    dic_metadata = dic_metadata.sort_values("reference_date").reset_index(drop=True)

    # get dic data
    dic_ids_sorted = dic_metadata["dic_id"].tolist()
    dic_data = get_multi_dic_data(
        dic_ids=dic_ids_sorted, config=config, stack_results=False
    )

    return dic_metadata, dic_data


def compute_dic_stats_for_geom(
    geom, dic_metadata: pd.DataFrame, dic_data: dict
) -> pd.DataFrame:
    """Return a dataframe indexed by dic_id with stats (date, n_points, mean,std,min,max,median)."""
    rows = []
    for dic_id in dic_metadata.dic_id.unique():
        pts = dic_data.get(dic_id)
        if pts is None or pts.empty:
            continue
        mask = contains_xy(geom, pts["x"].to_numpy(), pts["y"].to_numpy())
        sel = pts.loc[mask]
        vals = pd.to_numeric(sel["V"], errors="coerce")

        cur_date = dic_metadata.loc[dic_metadata.dic_id == dic_id, "reference_date"]
        if not cur_date.empty:
            date = pd.to_datetime(cur_date.values[0])
        else:
            date = None
        rows.append(
            {
                "dic_id": dic_id,
                "date": date,
                "n_points": int(len(sel)),
                "mean": float(vals.mean()) if not vals.empty else np.nan,
                "std": float(vals.std()) if not vals.empty else np.nan,
                "min": float(vals.min()) if not vals.empty else np.nan,
                "max": float(vals.max()) if not vals.empty else np.nan,
                "median": float(vals.median()) if not vals.empty else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "dic_id",
                "date",
                "n_points",
                "mean",
                "std",
                "min",
                "max",
                "median",
            ]
        ).set_index("dic_id")
    df = pd.DataFrame(rows).set_index("dic_id")
    df = df.sort_values("date")
    return df


def plot_geometry_on_image(ax, geom, image=None, **plot_kwargs):
    """Plot Shapely geometry (Polygon or MultiPolygon) on matplotlib axis."""
    if image is not None:
        ax.imshow(image)

    # Default plot styling
    line_color = plot_kwargs.get("line_color", "red")
    line_width = plot_kwargs.get("line_width", 2)
    fill_color = plot_kwargs.get("fill_color", "red")
    fill_alpha = plot_kwargs.get("fill_alpha", 0.3)

    if geom.geom_type == "Polygon":
        xs, ys = geom.exterior.xy
        ax.plot(xs, ys, color=line_color, linewidth=line_width)
        ax.fill(xs, ys, facecolor=fill_color, edgecolor="none", alpha=fill_alpha)
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color=line_color, linewidth=line_width)
            ax.fill(xs, ys, facecolor=fill_color, edgecolor="none", alpha=fill_alpha)
    else:
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")

    ax.set_axis_off()
    return ax


def make_collapse_plot(
    stats_df: pd.DataFrame,
    collapse_row: pd.Series,
    image: np.ndarray,
    *,
    stat_name: str = "mean",
    velocity_ylim: tuple[int, int] | None = VELOCITY_YLIM,
    smooth_window: int = 3,  # rolling window size for smoothing
) -> tuple[Figure, Any]:
    """Create the three-panel (image + quiver + timeseries) plot for a collapse.

    Returns the figure and axes objects.
    """

    collapse_id = int(collapse_row["id"])
    date_ts = pd.to_datetime(collapse_row["date"])

    # Create figure with GridSpec for better control over spacing
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[0.7, 0.75, 1],
        hspace=0.02,
        wspace=0.3,
        left=0.01,
        right=0.99,
        top=0.92,
        bottom=0.08,
    )
    ax_img = fig.add_subplot(gs[0])
    ax_quiver = fig.add_subplot(gs[1])
    ax_ts = fig.add_subplot(gs[2])

    # Left: image + geometry
    geom = shapely_wkt.loads(collapse_row["geom_wkt"])
    plot_geometry_on_image(
        ax_img,
        geom,
        image=image,
        line_color="red",
        line_width=1.5,
        fill_color="none",
    )
    ax_img.set_axis_off()
    ax_img.set_title("Collapse Geometry", fontsize=10, pad=5)

    # Middle: quiver plot with proper implementation
    has_quiver = False
    if stats_df is not None and not stats_df.empty:
        try:
            # take second last, as last may be on collapse date
            last_dic_id = stats_df.index[-2]
            last_dic_date = stats_df.iloc[-2]["date"]
            dic_pts = get_dic_data(
                dic_id=last_dic_id,
                config=ConfigManager(CONFIG_PATH),
            )
            ax_quiver.imshow(image, alpha=0.7)
            mag_data = dic_pts["V"].to_numpy()
            vmin = 0.0
            vmax = np.percentile(mag_data, 95) if len(mag_data) > 0 else 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)
            q = ax_quiver.quiver(
                dic_pts["x"].to_numpy(),
                dic_pts["y"].to_numpy(),
                dic_pts["u"].to_numpy(),
                dic_pts["v"].to_numpy(),
                mag_data,
                scale=None,
                scale_units="xy",
                angles="xy",
                cmap="viridis",
                norm=norm,
                width=0.003,
                headwidth=2.5,
                alpha=1.0,
            )
            cbar = fig.colorbar(q, ax=ax_quiver, pad=0.01, fraction=0.046)
            cbar.ax.tick_params(labelsize=8)
            if pd.notna(last_dic_date):
                date_str = pd.to_datetime(last_dic_date).strftime("%Y-%m-%d")
                ax_quiver.set_title(
                    f"DIC Velocity Field ({date_str})", fontsize=10, pad=5
                )
            has_quiver = True

        except Exception as exc:
            logger.warning(f"Could not plot quiver for collapse {collapse_id}: {exc}")

    if not has_quiver:
        ax_quiver.text(0.5, 0.5, "No DIC data available", ha="center", va="center")
    ax_quiver.set_axis_off()

    # Right: timeseries
    if stats_df is None or stats_df.empty:
        ax_ts.text(0.5, 0.5, "No DIC data inside geometry", ha="center", va="center")
        ax_ts.set_title("Velocity Time Series", fontsize=10, pad=5)
    else:
        stats = stats_df.copy()
        stats["date"] = pd.to_datetime(stats["date"])
        for col in ["n_points", "mean", "std", "min", "max", "median"]:
            if col in stats.columns:
                stats[col] = pd.to_numeric(stats[col], errors="coerce")
        x = stats["date"]

        # Plot std band
        # if "std" in stats.columns and "mean" in stats.columns:
        #     y1 = stats["mean"] - stats["std"]
        #     y2 = stats["mean"] + stats["std"]
        #     ax_ts.fill_between(x, y1, y2, color="gray", alpha=0.25, label="±1 std")

        # Plot mean and median
        if stat_name in stats.columns:
            ax_ts.plot(
                x,
                stats[stat_name],
                marker="o",
                markersize=1,
                label=stat_name.capitalize(),
                linewidth=0,
            )
            # Plot smoothed mean using rolling average
            if len(stats) >= smooth_window:
                smoothed = (
                    stats[stat_name]
                    .rolling(window=smooth_window, center=True, min_periods=1)
                    .mean()
                )
                ax_ts.plot(
                    x,
                    smoothed,
                    label=f"Smoothed (window={smooth_window})",
                    linewidth=2.5,
                    color="C1",
                    alpha=0.9,
                )

        if velocity_ylim is not None and len(velocity_ylim) == 2:
            ax_ts.set_ylim(velocity_ylim)

        ax_ts.set_xlabel("Date", fontsize=9)
        ax_ts.set_ylabel("Velocity [px/day]", fontsize=9)
        ax_ts.legend(fontsize=8, loc="best")
        ax_ts.grid(alpha=0.3, linewidth=0.5)
        ax_ts.tick_params(labelsize=8)
        ax_ts.set_title("Velocity Inside Collapse Area", fontsize=10, pad=5)

        # Format x-axis dates
        for label in ax_ts.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")

    # Overall title
    area = collapse_row.get("area", float("nan"))
    volume = collapse_row.get("volume", float("nan"))
    fig.suptitle(
        f"Collapse ID {collapse_id} — {date_ts.strftime('%Y-%m-%d')} — "
        f"Area: {area:.1f} px² — Volume: {volume:.1f} m³",
        fontsize=12,
    )

    fig.savefig(f"debug_{collapse_id}.jpg", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return fig, (ax_img, ax_quiver, ax_ts)


def process_collapse(
    collapse_row: pd.Series,
    cfg: ConfigManager,
    days_before: int,
    out_dir: Path,
) -> Optional[Path]:
    """
    Make the two-panel plot (image + geometry on left, velocity timeseries on right)
    for one collapse, reusing compute_dic_stats_for_geom and fetch_dic_before.
    """
    collapse_id = int(collapse_row["id"])
    date_ts = pd.to_datetime(collapse_row["date"])
    collapse_date = date_ts.date()
    geom_wkt = collapse_row["geom_wkt"]
    logger.info(f"Processing collapse id={collapse_id} date={collapse_date}")

    try:
        geom = shapely_wkt.loads(geom_wkt)
    except Exception as exc:
        logger.error(f"Invalid WKT for collapse {collapse_id}: {exc}")
        return None

    # fetch image for left panel (may fail separately)
    image = None
    try:
        image = get_image(image_id=int(collapse_row["image_id"]), config=cfg)
    except Exception as exc:
        logger.error(f"Failed to fetch image for collapse {collapse_id}: {exc}")
        return None

    # fetch DIC data and compute stats inside geometry
    try:
        dic_meta, dic_data = fetch_dic_before(
            config=cfg,
            collapse_date=collapse_date.isoformat(),
            days_before=days_before,
            camera_name="PPCX_Tele",
            dt_hours_min=72,
            dt_hours_max=96,
        )
        if dic_meta.empty or not dic_data:
            logger.warning(f"No DIC data found before collapse {collapse_id}")
            return None
    except Exception as exc:
        logger.exception(
            f"Error fetching or computing DIC stats for collapse {collapse_id}: {exc}"
        )
        return None

    # Compute statistics for DIC points inside geometry
    stats_df = compute_dic_stats_for_geom(geom, dic_meta, dic_data)

    # Build figure
    try:
        fig, ax = make_collapse_plot(
            stats_df=stats_df,
            collapse_row=collapse_row,
            image=np.asarray(image),
            velocity_ylim=VELOCITY_YLIM,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            out_dir
            / f"{collapse_date.isoformat()}_collapse_{collapse_id}_timeseries{DAYS_BEFORE}days.jpg"
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug(f"Saved plot for collapse {collapse_id} -> {out_path}")

        return out_path

    except Exception:
        logger.exception(
            "Failed to build/save plot for collapse row: %s", collapse_row.to_dict()
        )
        return None


def main() -> bool:
    cfg = ConfigManager(CONFIG_PATH)

    engine = create_engine(cfg.db_url)
    df = get_collapses_df(engine)
    if df.empty:
        logger.warning("No collapses found in database.")
        return False

    def _process_row(row):
        try:
            process_collapse(
                row,
                cfg=cfg,
                days_before=DAYS_BEFORE,
                out_dir=OUTPUT_DIR,
            )
        except Exception:
            logger.exception(
                f"Unexpected error plotting collapse id={int(row.get('id', -1))}"
            )

    with Parallel(n_jobs=N_JOBS, backend="threading") as parallel:
        parallel(
            delayed(_process_row)(row)
            for _, row in tqdm(
                df.iterrows(), total=df.shape[0], desc="Processing collapses"
            )
        )

    return True


if __name__ == "__main__":
    main()
