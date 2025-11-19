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
from shapely.ops import unary_union
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sqlalchemy import create_engine
from statsmodels.nonparametric.smoothers_lowess import lowess
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
N_JOBS = 6  # number of parallel jobs

MIN_VELOCITY = 1  # Minimum velocity threshold (px/day)
OUTLIER_THRESHOLD = 2.5  # NMAD threshold for outlier removal
BUFFER_DISTANCE_PX = 500  # Buffer ring distance in pixels, or None to disable

ROBUST_METHOD = "lowess"  # Options: "lowess", "huber", "ransac"
# ROBUST_METHOD = "lowess"  # Non-parametric, flexible (recommended)
# ROBUST_METHOD = "huber"  # Linear, robust to outliers
# ROBUST_METHOD = "ransac"  # Very robust, linear
LOWESS_FRAC = 0.2
# LOWESS_FRAC = 0.2  # More smoothing (larger window)
# LOWESS_FRAC = 0.4  # Less smoothing (smaller window)
VELOCITY_YLIM = (0, 20)  # fixed y axis limits for velocity plot, or None

# -------------------------


def nmad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


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


def extract_points_inside_geom(geom, dic_points: pd.DataFrame) -> pd.DataFrame:
    """
    Extract DIC points inside geometry for a single timestamp.

    Args:
        geom: Shapely geometry (Polygon or MultiPolygon)
        dic_points: DataFrame with columns [x, y, u, v, V]

    Returns:
        DataFrame with points inside geometry, columns: [x, y, u, v, V]
    """
    if dic_points.empty:
        return pd.DataFrame(columns=["x", "y", "u", "v", "V"])

    # Filter points inside geometry
    mask = contains_xy(geom, dic_points["x"].to_numpy(), dic_points["y"].to_numpy())
    pts_inside = dic_points.loc[mask].copy()

    # Convert velocity to numeric and drop invalid values
    pts_inside["V"] = pd.to_numeric(pts_inside["V"], errors="coerce")
    pts_inside = pts_inside.dropna(subset=["V"])

    return pts_inside[["x", "y", "u", "v", "V"]]


def filter_outliers(velocities: np.ndarray, threshold: float = 2.5) -> np.ndarray:
    """
    Filter outliers using NMAD-based threshold.

    Returns boolean mask of inliers.
    """
    if len(velocities) == 0:
        return np.array([], dtype=bool)

    median_v = np.median(velocities)
    mad = nmad(velocities)

    if mad > 0:
        inlier_mask = np.abs(velocities - median_v) <= threshold * mad
    else:
        inlier_mask = np.ones(len(velocities), dtype=bool)

    return inlier_mask


def extract_and_filter_points(
    geom,
    dic_points: pd.DataFrame,
    min_velocity: float = MIN_VELOCITY,
    outlier_threshold: float = OUTLIER_THRESHOLD,
) -> pd.DataFrame:
    """
    Extract points inside geometry and filter by minimum velocity and outliers.
    """
    pts = extract_points_inside_geom(geom, dic_points)

    if pts.empty:
        return pts

    # Filter by minimum velocity
    pts = pts[pts["V"] >= min_velocity].copy()

    if pts.empty:
        return pts

    # Filter outliers
    velocities = pts["V"].to_numpy()
    inlier_mask = filter_outliers(velocities, outlier_threshold)
    pts = pts.loc[inlier_mask].copy()

    return pts


def compute_stats_per_timestamp(
    geom,
    dic_metadata: pd.DataFrame,
    dic_data: dict,
    buffer_distance: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract and compute statistics for points inside geometry and buffer ring.

    Returns:
        stats_inside: DataFrame with stats for points inside collapse geometry
        stats_buffer: DataFrame with stats for points in buffer ring (or empty if buffer_distance=None)
    """

    def _compute_stats(array: np.ndarray) -> dict[str, Any]:
        return {
            "n_points": len(array),
            "mean": np.mean(array),
            "std": np.std(array),
            "median": np.median(array),
            "nmad": nmad(array),
        }

    # Create buffer if requested
    buffered = None
    if buffer_distance is not None and buffer_distance > 0:
        # Normalize geometry for robust buffering
        if geom.geom_type == "MultiPolygon":
            base_geom = unary_union(list(geom.geoms))
        else:
            base_geom = geom

        buffered = base_geom.buffer(buffer_distance)

    rows_inside = []
    rows_buffer = []

    for dic_id in dic_metadata.dic_id.unique():
        pts = dic_data.get(dic_id)
        if pts is None or pts.empty:
            continue

        date = pd.to_datetime(
            dic_metadata.loc[dic_metadata.dic_id == dic_id, "reference_date"].values[0]
        )

        # Extract and filter points inside collapse
        pts_inside = extract_and_filter_points(geom, pts)
        if not pts_inside.empty:
            rows_inside.append(
                {"date": date, **_compute_stats(pts_inside["V"].to_numpy())}
            )

        # Extract and filter points in buffer ring
        if buffered is not None:
            pts_buffer = extract_and_filter_points(buffered, pts)
            if not pts_buffer.empty:
                rows_buffer.append(
                    {"date": date, **_compute_stats(pts_buffer["V"].to_numpy())}
                )

    stats_inside = pd.DataFrame(rows_inside) if rows_inside else pd.DataFrame()
    stats_buffer = pd.DataFrame(rows_buffer) if rows_buffer else pd.DataFrame()

    return stats_inside, stats_buffer


def fit_robust_velocity_trend_lowess(
    df_points: pd.DataFrame,
    frac: float = 0.3,
    eval_dates: pd.DatetimeIndex | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Fit LOWESS trend using ALL individual DIC points (not per-timestep means).

    Args:
        df_points: DataFrame with columns [date, V]
        frac: LOWESS smoothing fraction
        eval_dates: Dates at which to evaluate the fit (if None, uses unique dates in data)

    Returns:
        fitted_values: Series with fitted velocity values
        prediction_std: Series with estimated standard deviation (computed from residuals in local window)
    """
    if df_points.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Clean data: remove NaN, inf, and negative velocities
    df_clean = df_points.copy()
    df_clean = df_clean.dropna(subset=["V", "date"])
    df_clean = df_clean[np.isfinite(df_clean["V"])]
    df_clean = df_clean[df_clean["V"] > 0]  # Velocity must be positive

    if df_clean.empty or len(df_clean) < 3:
        logger.warning("Not enough valid points for LOWESS fitting")
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Convert dates to numeric (days since first observation)
    df_clean.sort_values("date").reset_index(drop=True)
    t_min = df_clean["date"].min()
    df_clean["t_numeric"] = (df_clean["date"] - t_min).dt.total_seconds() / (24 * 3600)

    # Fit LOWESS on all points
    fitted = lowess(
        df_clean["V"].values,
        df_clean["t_numeric"].values,
        frac=frac,
        it=3,
        return_sorted=True,
    )

    # Interpolate to evaluation dates
    if eval_dates is None:
        eval_dates = pd.DatetimeIndex(df_clean["date"].unique()).sort_values()

    eval_t_numeric = (pd.Series(eval_dates) - t_min).dt.total_seconds() / (24 * 3600)
    fitted_interp = np.interp(eval_t_numeric, fitted[:, 0], fitted[:, 1])

    # Estimate local standard deviation using residuals
    residuals = df_clean["V"].values.astype(np.float64) - np.interp(
        df_clean["t_numeric"].values.astype(np.float64),
        fitted[:, 0].astype(np.float64),
        fitted[:, 1].astype(np.float64),
    )

    # Compute rolling std of residuals per time window
    window_size = max(3, int(frac * len(df_clean)))
    local_stds = []
    for t_eval in eval_t_numeric:
        # Find points within local window
        distances = np.abs(df_clean["t_numeric"].values - t_eval)
        local_idx = np.argsort(distances)[:window_size]
        local_residuals = residuals[local_idx]
        local_stds.append(
            np.std(local_residuals) if len(local_residuals) > 1 else np.nan
        )

    return pd.Series(fitted_interp, index=eval_dates), pd.Series(
        local_stds, index=eval_dates
    )


def fit_robust_velocity_trend_huber(
    df_points: pd.DataFrame, eval_dates: pd.DatetimeIndex | None = None
) -> tuple[pd.Series, pd.Series]:
    """
    Fit Huber robust linear regression on ALL individual DIC points.

    More resistant to outliers than ordinary least squares.
    """
    if df_points.empty or len(df_points) < 2:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    df_sorted = df_points.sort_values("date").copy()
    t_min = df_sorted["date"].min()
    df_sorted["t_numeric"] = (df_sorted["date"] - t_min).dt.total_seconds() / (
        24 * 3600
    )

    X = df_sorted["t_numeric"].values.reshape(-1, 1)
    y = df_sorted["V"].values

    # Fit Huber regressor
    model = HuberRegressor(epsilon=1.35, max_iter=200)
    model.fit(X, y)

    # Predict at evaluation dates
    if eval_dates is None:
        eval_dates = pd.DatetimeIndex(df_sorted["date"].unique()).sort_values()

    eval_t_numeric = (pd.Series(eval_dates) - t_min).dt.total_seconds() / (24 * 3600)
    X_eval = eval_t_numeric.values.reshape(-1, 1)
    predictions = model.predict(X_eval)

    # Compute residuals and estimate std
    residuals = y - model.predict(X)
    pred_std = np.full(len(eval_dates), np.std(residuals))

    return pd.Series(predictions, index=eval_dates), pd.Series(
        pred_std, index=eval_dates
    )


def fit_robust_velocity_trend_ransac(
    df_points: pd.DataFrame, eval_dates: pd.DatetimeIndex | None = None
) -> tuple[pd.Series, pd.Series]:
    """
    Fit RANSAC robust linear regression on ALL individual DIC points.

    Very resistant to outliers by randomly sampling consensus sets.
    """
    if df_points.empty or len(df_points) < 2:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    df_sorted = df_points.sort_values("date").copy()
    t_min = df_sorted["date"].min()
    df_sorted["t_numeric"] = (df_sorted["date"] - t_min).dt.total_seconds() / (
        24 * 3600
    )

    X = df_sorted["t_numeric"].values.reshape(-1, 1)
    y = df_sorted["V"].values

    # Fit RANSAC regressor
    model = RANSACRegressor(random_state=42, max_trials=100)
    model.fit(X, y)

    # Predict at evaluation dates
    if eval_dates is None:
        eval_dates = pd.DatetimeIndex(df_sorted["date"].unique()).sort_values()

    eval_t_numeric = (pd.Series(eval_dates) - t_min).dt.total_seconds() / (24 * 3600)
    X_eval = eval_t_numeric.values.reshape(-1, 1)
    predictions = model.predict(X_eval)

    # Compute residuals on inliers only
    inlier_mask = model.inlier_mask_
    residuals = y[inlier_mask] - model.predict(X[inlier_mask])
    pred_std = np.full(len(eval_dates), np.std(residuals))

    return pd.Series(predictions, index=eval_dates), pd.Series(
        pred_std, index=eval_dates
    )


def compute_robust_velocity_trend(
    df_points: pd.DataFrame,
    method: str = "lowess",
    eval_dates: pd.DatetimeIndex | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Compute robust velocity trend using ALL DIC points inside geometry.

    Args:
        geom: Shapely geometry defining collapse area
        dic_metadata: DataFrame with DIC analysis metadata
        dic_data: Dictionary mapping dic_id to DIC point data
        method: Robust fitting method ("lowess", "huber", "ransac")
        **kwargs: Additional parameters for fitting method

    Returns:
        stats_per_timestep: DataFrame with raw statistics per time step
        trend_fit: DataFrame with fitted trend (date, fitted_velocity, std_velocity)
    """
    if df_points.empty:
        return pd.DataFrame(columns=["date", "fitted_velocity", "std_velocity"])

    try:
        # Fit robust trend using all points
        if method == "lowess":
            frac = kwargs.get("frac", LOWESS_FRAC)
            fitted_vel, std_vel = fit_robust_velocity_trend_lowess(
                df_points, frac=frac, eval_dates=eval_dates
            )
        elif method == "huber":
            fitted_vel, std_vel = fit_robust_velocity_trend_huber(
                df_points, eval_dates=eval_dates
            )
        elif method == "ransac":
            fitted_vel, std_vel = fit_robust_velocity_trend_ransac(
                df_points, eval_dates=eval_dates
            )
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'lowess', 'huber', or 'ransac'"
            )

        trend_fit = pd.DataFrame(
            {
                "date": fitted_vel.index,
                "fitted_velocity": fitted_vel.values,
                "std_velocity": std_vel.values,
            }
        )
        return trend_fit

    except Exception:
        return pd.DataFrame(columns=["date", "fitted_velocity", "std_velocity"])


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
    collapse_row: pd.Series,
    dic_metadata: pd.DataFrame,
    dic_points: pd.DataFrame,
    trend_fit: pd.DataFrame,
    stats_buffer: pd.DataFrame,
    image: np.ndarray,
    *,
    velocity_ylim: tuple[int, int] | None = VELOCITY_YLIM,
    trend_method: str = ROBUST_METHOD,
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
    try:
        # take second last, as last may be on collapse date
        last_dic_id = dic_metadata.iloc[-2]["dic_id"]
        last_dic_date = dic_metadata.iloc[-2]["reference_date"]
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
            ax_quiver.set_title(f"DIC Velocity Field ({date_str})", fontsize=10, pad=5)
        ax_quiver.set_axis_off()
    except Exception:
        logger.warning(
            f"Failed to plot quiver for collapse {collapse_id}", exc_info=True
        )
        ax_quiver.text(0.5, 0.5, "No DIC data available", ha="center", va="center")

    # Right: velocity time series with robust trend
    try:
        # Plot buffer (surrounding) statistics first (background)
        if not stats_buffer.empty:
            x_buf = pd.to_datetime(stats_buffer["date"])
            y_buf = stats_buffer["median"]
            nmad_buf = stats_buffer["nmad"]

            ax_ts.plot(
                x_buf,
                y_buf,
                linewidth=1.5,
                alpha=0.4,
                color="gray",
                linestyle="--",
                label="Surrounding median",
            )
            ax_ts.fill_between(
                x_buf,
                y_buf - nmad_buf,
                y_buf + nmad_buf,
                alpha=0.1,
                color="gray",
                label="Surrounding ±1 NMAD",
            )

        # Plot all the individual DIC points as background
        ax_ts.scatter(
            dic_points["date"],
            dic_points["V"],
            s=2,
            alpha=0.3,
            color="gray",
            label="Individual DIC points",
        )

        # Plot robust fitted trend
        x_fit = pd.to_datetime(trend_fit["date"])
        y_fit = trend_fit["fitted_velocity"]
        y_std = trend_fit["std_velocity"]
        ax_ts.plot(
            x_fit,
            y_fit,
            linewidth=2.5,
            alpha=0.9,
            label=f"Robust fit ({trend_method.upper()})",
            color="C1",
        )

        # Plot uncertainty band
        if not y_std.isna().all():
            ax_ts.fill_between(
                x_fit,
                y_fit - y_std,
                y_fit + y_std,
                alpha=0.2,
                color="C1",
                label="±1 std",
            )

        if velocity_ylim is not None:
            ax_ts.set_ylim(velocity_ylim)

        ax_ts.set_xlabel("Date", fontsize=9)
        ax_ts.set_ylabel("Velocity [px/day]", fontsize=9)
        ax_ts.legend(fontsize=8, loc="best")
        ax_ts.grid(alpha=0.3, linewidth=0.5)
        ax_ts.tick_params(labelsize=8)
        ax_ts.set_title("Robust Velocity Trend", fontsize=10, pad=5)

        for label in ax_ts.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")

    except Exception:
        logger.warning(
            f"Failed to plot timeseries for collapse {collapse_id}", exc_info=True
        )
        ax_ts.text(0.5, 0.5, "No DIC data available", ha="center", va="center")

    # Overall title
    area = collapse_row.get("area", float("nan"))
    volume = collapse_row.get("volume", float("nan"))
    fig.suptitle(
        f"Collapse ID {collapse_id} — {date_ts.strftime('%Y-%m-%d')} — "
        f"Area: {area:.1f} px² — Volume: {volume:.1f} m³ — ",
        fontsize=12,
    )

    # fig.savefig(f"debug_{collapse_id}.jpg", dpi=150, bbox_inches="tight")
    # plt.close(fig)

    return fig, (ax_img, ax_quiver, ax_ts)


def process_collapse(
    collapse_row: pd.Series,
    cfg: ConfigManager,
    days_before: int,
    out_dir: Path,
    velocity_trend_method: str = ROBUST_METHOD,
    buffer_distance: float = BUFFER_DISTANCE_PX,
) -> Optional[Path]:
    """
    Make the two-panel plot (image + geometry on left, velocity timeseries on right)
    for one collapse, reusing compute_dic_stats_for_geom and fetch_dic_before.
    """
    collapse_id = int(collapse_row["id"])
    date_ts = pd.to_datetime(collapse_row["date"])
    collapse_date = date_ts.date()
    logger.info(f"Processing collapse id={collapse_id} date={collapse_date}")

    # parse geometry
    try:
        geom = shapely_wkt.loads(collapse_row["geom_wkt"])
    except Exception as exc:
        logger.error(f"Invalid WKT for collapse {collapse_id}: {exc}")
        return None

    # fetch image
    image = None
    try:
        image = get_image(image_id=int(collapse_row["image_id"]), config=cfg)
    except Exception as exc:
        logger.error(f"Failed to fetch image for collapse {collapse_id}: {exc}")
        return None

    # fetch DIC data and compute stats inside geometry
    dic_metadata, dic_data = fetch_dic_before(
        config=cfg,
        collapse_date=collapse_date.isoformat(),
        days_before=days_before,
        camera_name="PPCX_Tele",
        dt_hours_min=72,
        dt_hours_max=96,
    )
    if dic_metadata.empty or not dic_data:
        logger.warning(f"No DIC data found before collapse {collapse_id}")
        return None

    # Compute statistics for collapse area and buffer
    stats_inside, stats_buffer = compute_stats_per_timestamp(
        geom, dic_metadata, dic_data, buffer_distance=buffer_distance
    )

    if stats_inside.empty:
        logger.warning(f"No valid points inside geometry for collapse {collapse_id}")
        return None

    # Extract all points inside geometry
    all_points = []
    for dic_id in dic_metadata.dic_id.unique():
        pts = dic_data.get(dic_id)
        if pts is None or pts.empty:
            continue

        pts_filtered = extract_and_filter_points(
            geom, pts, MIN_VELOCITY, OUTLIER_THRESHOLD
        )
        if not pts_filtered.empty:
            date = dic_metadata.loc[
                dic_metadata.dic_id == dic_id, "reference_date"
            ].values[0]
            pts_filtered["date"] = pd.to_datetime(date)
            all_points.append(pts_filtered)

    df_all_points = pd.concat(all_points, ignore_index=True)

    # Compute robust velocity trend
    trend_fit = compute_robust_velocity_trend(
        df_all_points,
        method=velocity_trend_method,
        eval_dates=pd.DatetimeIndex(stats_inside["date"]),
    )

    # Generate plot
    fig, ax = make_collapse_plot(
        collapse_row=collapse_row,
        dic_metadata=dic_metadata,
        dic_points=df_all_points,
        trend_fit=trend_fit,
        stats_buffer=stats_buffer,
        image=np.asarray(image),
        velocity_ylim=VELOCITY_YLIM,
        trend_method=velocity_trend_method,
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


def main() -> bool:
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

    cfg = ConfigManager(CONFIG_PATH)

    engine = create_engine(cfg.db_url)
    df = get_collapses_df(engine)
    if df.empty:
        logger.warning("No collapses found in database.")
        return False

    df_sorted = df.sort_values("id").reset_index(drop=True)
    with Parallel(n_jobs=N_JOBS, backend="threading") as parallel:
        parallel(
            delayed(_process_row)(row)
            for _, row in tqdm(
                df_sorted.iterrows(),
                total=df_sorted.shape[0],
                desc="Processing collapses",
            )
        )

    return True


if __name__ == "__main__":
    main()
