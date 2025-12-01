from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

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
OUTPUT_DIR = Path("output/collapses_timeseries")

MIN_COLLAPSE_AREA = 150000  # px²
DAYS_BEFORE = 10
DAY_AFTER = 5
MIN_DIC_DAYS_BEFORE = 5  # Minimum number of DIC analyses before collapse
USE_CENTER_DATE = True  # Use center date between master/slave for plotting

MIN_VELOCITY = 1  # Minimum velocity threshold (px/day)
OUTLIER_THRESHOLD = 2.5  # NMAD threshold for outlier removal
BUFFER_DISTANCE_PX = 500  # Buffer ring distance in pixels, or None to disable

ROBUST_METHOD = "lowess"  # Only option supported for now.
LOWESS_FRAC = 0.2
# LOWESS_FRAC = 0.2  # More smoothing (larger window)
# LOWESS_FRAC = 0.4  # Less smoothing (smaller window)
VELOCITY_YLIM = (0, 20)  # fixed y axis limits for velocity plot, or None

N_JOBS = 16  # number of parallel jobs

# -------------------------


def nmad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def fetch_dic_data(
    config: ConfigManager,
    collapse_date: str,
    days_before: int | None = None,
    day_after: int | None = None,
    **kwargs,
):
    """Fetch DIC analyses in the window [collapse_date - days_before, collapse_date]."""

    if days_before is None:
        days_before = 0

    if day_after is None:
        day_after = 0

    start_date = pd.to_datetime(collapse_date) - pd.Timedelta(days=days_before)
    start_date_str = start_date.strftime("%Y-%m-%d")

    end_date = pd.to_datetime(collapse_date) + pd.Timedelta(days=day_after)
    end_date_str = end_date.strftime("%Y-%m-%d")

    engine = create_engine(config.db_url)
    dic_ids = fetch_dic_analysis_ids(
        db_engine=engine,
        reference_date_start=start_date_str,
        reference_date_end=end_date_str,
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


def compute_stats_for_points(points: pd.DataFrame) -> dict[str, float]:
    """
    Compute velocity statistics for a set of points.

    Args:
        points: DataFrame with column 'V' (velocity)

    Returns:
        Dictionary with keys: n_points, mean, std, median, nmad
    """
    velocities = points["V"].to_numpy()
    return {
        "n_points": len(velocities),
        "mean": np.mean(velocities),
        "std": np.std(velocities),
        "median": np.median(velocities),
        "nmad": nmad(velocities),
    }


def compute_stats_timeseries(
    geom,
    dic_metadata: pd.DataFrame,
    dic_data: dict,
    min_velocity: float = MIN_VELOCITY,
    outlier_threshold: float = OUTLIER_THRESHOLD,
) -> pd.DataFrame:
    """
    Compute velocity statistics per timestamp for points inside geometry.

    Args:
        geom: Shapely geometry defining region
        dic_metadata: DIC metadata DataFrame
        dic_data: Dictionary mapping dic_id to point data
        min_velocity: Minimum velocity threshold
        outlier_threshold: NMAD threshold for outlier removal

    Returns:
        DataFrame with columns: date, n_points, mean, std, median, nmad
    """
    rows = []

    for dic_id in dic_metadata.dic_id.unique():
        pts = dic_data.get(dic_id)
        if pts is None or pts.empty:
            continue

        date_val = pd.to_datetime(
            dic_metadata.loc[dic_metadata.dic_id == dic_id, "reference_date"].values[0]
        )

        # Extract and filter points
        pts_filtered = extract_and_filter_points(
            geom, pts, min_velocity=min_velocity, outlier_threshold=outlier_threshold
        )

        if pts_filtered.empty:
            continue

        # Compute statistics
        stats = compute_stats_for_points(pts_filtered)
        stats["date"] = date_val
        rows.append(stats)

    if not rows:
        return pd.DataFrame(
            columns=["date", "n_points", "mean", "std", "median", "nmad"]
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


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
        else:
            raise ValueError(f"Unknown method: {method}. Use 'lowess'")

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


def _save_df(df: pd.DataFrame, base_path: Path, suffix: str) -> Path | None:
    """Save DataFrame as Parquet (fallback to CSV)."""
    if df is None or df.empty:
        return None
    parquet_path = base_path.with_name(base_path.name + f"_{suffix}.parquet")
    csv_path = base_path.with_name(base_path.name + f"_{suffix}.csv")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        df.to_csv(csv_path, index=False)
        return csv_path


def _circular_mean(angles: np.ndarray) -> float:
    """Return circular mean of angles in radians."""
    if len(angles) == 0:
        return np.nan
    return math.atan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))


def _circular_distance(a: float, b: float) -> float:
    """Smallest absolute angular difference (radians) between two angles."""
    if np.isnan(a) or np.isnan(b):
        return np.nan
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def compute_deviation_score(
    pts_inside: pd.DataFrame,
    pts_outside: pd.DataFrame,
) -> dict[str, float]:
    """
    Compare velocity field between two point sets (magnitude + direction).

    Score normalized to [0, 1] where:
    - 0 = no difference
    - 1 = maximum difference

    Args:
        pts_inside: DataFrame with columns [u, v, V]
        pts_outside: DataFrame with columns [u, v, V]

    Returns:
        dict with keys: median_inside, median_outside, angle_inside, angle_outside, mag_diff_norm, angle_diff_norm, score
    """
    if pts_inside.empty or pts_outside.empty:
        return {
            "median_inside": np.nan,
            "median_outside": np.nan,
            "angle_inside": np.nan,
            "angle_outside": np.nan,
            "mag_diff_norm": np.nan,
            "angle_diff_norm": np.nan,
            "score": np.nan,
        }

    # Magnitudes
    v_in = pts_inside["V"].to_numpy()
    v_out = pts_outside["V"].to_numpy()
    median_inside = np.median(v_in)
    median_outside = np.median(v_out)

    # Normalize magnitude difference: |diff| / (max + eps)
    mag_diff = abs(median_inside - median_outside)
    mag_max = max(median_inside, median_outside) + 1e-9
    mag_diff_norm = mag_diff / mag_max  # [0, 1]

    # Direction angles
    u_in = pts_inside["u"].to_numpy()
    v_in_vec = pts_inside["v"].to_numpy()
    u_out = pts_outside["u"].to_numpy()
    v_out_vec = pts_outside["v"].to_numpy()

    angles_in = np.arctan2(v_in_vec, u_in)
    angles_out = np.arctan2(v_out_vec, u_out)

    angle_inside = _circular_mean(angles_in)
    angle_outside = _circular_mean(angles_out)
    angle_diff = _circular_distance(angle_inside, angle_outside)

    # Normalize angle difference: angular_dist / pi → [0, 1]
    angle_diff_norm = angle_diff / math.pi

    # Combined score: mean of normalized components
    score = (mag_diff_norm + angle_diff_norm) / 2.0  # [0, 1]

    return {
        "median_inside": median_inside,
        "median_outside": median_outside,
        "angle_inside": angle_inside,
        "angle_outside": angle_outside,
        "mag_diff_norm": mag_diff_norm,
        "angle_diff_norm": angle_diff_norm,
        "score": score,
    }


def compute_deviation_scores_timeseries(
    geom_inside,
    geom_outside,
    dic_metadata: pd.DataFrame,
    dic_data: dict,
    min_velocity: float = MIN_VELOCITY,
    outlier_threshold: float = OUTLIER_THRESHOLD,
    use_center_date: bool = True,
) -> pd.DataFrame:
    """
    Compute deviation scores per timestamp.

    Args:
        geom_inside: Shapely geometry for "inside" region
        geom_outside: Shapely geometry for "outside" region
        dic_metadata: DIC metadata DataFrame
        dic_data: Dictionary mapping dic_id to point data

    Returns:
        DataFrame with columns: date, score, mag_diff_norm, angle_diff_norm, ...
    """
    rows = []

    for dic_id in dic_metadata.dic_id.unique():
        pts = dic_data.get(dic_id)
        if pts is None or pts.empty:
            continue

        if not use_center_date:
            date_val = pd.to_datetime(
                dic_metadata.loc[
                    dic_metadata.dic_id == dic_id, "reference_date"
                ].values[0]
            )
        else:
            # Use center date between slave and master images as timestamp
            master_date = pd.to_datetime(
                dic_metadata.loc[
                    dic_metadata.dic_id == dic_id, "master_timestamp"
                ].values[0]
            )
            slave_date = pd.to_datetime(
                dic_metadata.loc[
                    dic_metadata.dic_id == dic_id, "slave_timestamp"
                ].values[0]
            )
            date_val = master_date + (slave_date - master_date) / 2

        # Extract and filter points
        pts_inside = extract_and_filter_points(
            geom_inside,
            pts,
            min_velocity=min_velocity,
            outlier_threshold=outlier_threshold,
        )
        pts_outside = extract_and_filter_points(
            geom_outside,
            pts,
            min_velocity=min_velocity,
            outlier_threshold=outlier_threshold,
        )

        # Compute deviation score
        dev_dict = compute_deviation_score(pts_inside, pts_outside)
        dev_dict["date"] = date_val
        rows.append(dev_dict)

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "median_inside",
                "median_outside",
                "angle_inside",
                "angle_outside",
                "mag_diff_norm",
                "angle_diff_norm",
                "score",
            ]
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


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
    deviation_df: pd.DataFrame,
    image: np.ndarray,
    *,
    velocity_ylim: tuple[int, int] | None = VELOCITY_YLIM,
    trend_method: str = ROBUST_METHOD,
    deviation_score_before_collapse: float | None = None,
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
        top=0.85,
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
        # take the dic on the collapse date
        idx_dic = dic_metadata[dic_metadata["reference_date"] == date_ts][
            "dic_id"
        ].values[0]
        dic_pts = get_dic_data(
            dic_id=idx_dic,
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
        if pd.notna(date_ts):
            ax_quiver.set_title(
                f"DIC Velocity Field ({date_ts.strftime('%Y-%m-%d')})",
                fontsize=10,
                pad=5,
            )
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

        # mark collapse date with vertical line
        ax_ts.axvline(
            x=date_ts,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label="Collapse date",
        )

        # Deviation score on twin axis
        if not deviation_df.empty:
            ax_dev = ax_ts.twinx()
            xd = pd.to_datetime(deviation_df["date"])
            yd = deviation_df["score"]

            ax_dev.plot(
                xd,
                yd,
                color="tab:red",
                linewidth=1.5,
                marker="o",
                markersize=3,
                alpha=0.6,
                label="Deviation score",
            )
            ax_dev.set_ylabel("Deviation score [0-1]", color="tab:red", fontsize=8)
            ax_dev.tick_params(axis="y", labelsize=8, colors="tab:red")
            ax_dev.set_ylim(0, 1)  # Normalized range

            # Combine legends
            lines1, labels1 = ax_ts.get_legend_handles_labels()
            lines2, labels2 = ax_dev.get_legend_handles_labels()
            ax_ts.legend(
                lines1 + lines2, labels1 + labels2, fontsize=7, loc="best", ncol=1
            )
        else:
            ax_ts.legend(fontsize=8, loc="best")
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
    dev_text = (
        f"Dev: {deviation_score_before_collapse:.3f}"
        if deviation_score_before_collapse is not None
        and not np.isnan(deviation_score_before_collapse)
        else "Dev: n/a"
    )
    fig.suptitle(
        f"Collapse {collapse_id} — {date_ts.strftime('%Y-%m-%d')}\n"
        f"Area: {area:.1f} px² — Volume: {volume:.1f} m³ — {dev_text}",
        fontsize=12,
    )

    # fig.savefig(f"debug_{collapse_id}.jpg", dpi=150, bbox_inches="tight")
    # plt.close(fig)

    return fig, (ax_img, ax_quiver, ax_ts)


def process_collapse(
    collapse_row: pd.Series,
    cfg: ConfigManager,
    days_before: int,
    day_after: int,
    out_dir: Path,
    velocity_trend_method: str = ROBUST_METHOD,
    buffer_distance: float = BUFFER_DISTANCE_PX,
    min_dic_days_before: int = 5,
    use_center_date: bool = True,
) -> bool:
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
        return False

    # fetch image
    image = None
    try:
        image = get_image(image_id=int(collapse_row["image_id"]), config=cfg)
    except Exception as exc:
        logger.error(f"Failed to fetch image for collapse {collapse_id}: {exc}")
        return False

    # fetch DIC data and compute stats inside geometry
    dic_metadata, dic_data = fetch_dic_data(
        config=cfg,
        collapse_date=collapse_date.isoformat(),
        days_before=days_before,
        day_after=day_after,
        camera_name="PPCX_Tele",
        dt_hours_min=72,
        dt_hours_max=96,
    )
    if dic_metadata.empty or not dic_data:
        logger.warning(f"No DIC data found before collapse {collapse_id}")
        return False

    if len(dic_metadata) < min_dic_days_before:
        logger.warning(
            f"Not enough DIC data before collapse {collapse_id} "
            f"(found {len(dic_metadata)}, required {min_dic_days_before})"
        )
        return False

    # Create geometries for analysis
    geom_inside = geom
    geom_outside = None

    if buffer_distance is not None and buffer_distance > 0:
        # Normalize geometry for robust buffering
        if geom.geom_type == "MultiPolygon":
            base_geom = unary_union(list(geom.geoms))
        else:
            base_geom = geom
        geom_outside = base_geom.buffer(buffer_distance)

    # Compute statistics for collapse area
    stats_inside = compute_stats_timeseries(
        geom_inside,
        dic_metadata,
        dic_data,
        min_velocity=MIN_VELOCITY,
        outlier_threshold=OUTLIER_THRESHOLD,
    )
    if stats_inside.empty:
        logger.warning(f"No valid points inside geometry for collapse {collapse_id}")
        return False

    # Compute statistics for buffer (if applicable)
    stats_buffer = pd.DataFrame()
    if geom_outside is not None:
        stats_buffer = compute_stats_timeseries(
            geom_outside,
            dic_metadata,
            dic_data,
            min_velocity=MIN_VELOCITY,
            outlier_threshold=OUTLIER_THRESHOLD,
        )

    # Compute robust velocity trend
    all_inside_points = []
    for dic_id in dic_metadata.dic_id.unique():
        pts = dic_data.get(dic_id)
        if pts is None or pts.empty:
            continue

        pts_filtered = extract_and_filter_points(
            geom, pts, MIN_VELOCITY, OUTLIER_THRESHOLD
        )
        if pts_filtered.empty:
            continue

        if not use_center_date:
            date = pd.to_datetime(
                dic_metadata.loc[
                    dic_metadata.dic_id == dic_id, "reference_date"
                ].values[0]
            )
        else:
            # Use center date between slave and master images as timestamp
            master_date = pd.to_datetime(
                dic_metadata.loc[
                    dic_metadata.dic_id == dic_id, "master_timestamp"
                ].values[0]
            )
            slave_date = pd.to_datetime(
                dic_metadata.loc[
                    dic_metadata.dic_id == dic_id, "slave_timestamp"
                ].values[0]
            )
            date = master_date + (slave_date - master_date) / 2
        pts_filtered["date"] = pd.to_datetime(date)
        all_inside_points.append(pts_filtered)

    all_inside_points = pd.concat(all_inside_points, ignore_index=True)
    trend_fit = compute_robust_velocity_trend(
        all_inside_points,
        method=velocity_trend_method,
        eval_dates=pd.DatetimeIndex(stats_inside["date"]),
    )

    # Compute deviation scores timeseries
    deviation_df = pd.DataFrame()
    deviation_score_before = np.nan
    if geom_outside is not None:
        deviation_df = compute_deviation_scores_timeseries(
            geom_inside=geom_inside,
            geom_outside=geom_outside,
            dic_metadata=dic_metadata,
            dic_data=dic_data,
            min_velocity=MIN_VELOCITY,
            outlier_threshold=OUTLIER_THRESHOLD,
            use_center_date=use_center_date,
        )

        # Get deviation score from second-to-last timestamp (before collapse)
        if not deviation_df.empty and len(deviation_df) >= 2:
            deviation_score_before = float(deviation_df.iloc[-2]["score"])
        elif not deviation_df.empty:
            deviation_score_before = float(deviation_df.iloc[-1]["score"])

    # Generate plot
    fig, ax = make_collapse_plot(
        collapse_row=collapse_row,
        dic_metadata=dic_metadata,
        dic_points=all_inside_points,
        trend_fit=trend_fit,
        stats_buffer=stats_buffer,
        deviation_df=deviation_df,
        image=np.asarray(image),
        velocity_ylim=VELOCITY_YLIM,
        trend_method=velocity_trend_method,
        deviation_score_before_collapse=deviation_score_before,
    )

    # -------------------------------
    # Save outputs
    # -------------------------------
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"collapse_{collapse_id}_{collapse_date.isoformat()}"

    plot_path = out_dir / f"{base_name}.jpg"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"Saved plot for collapse {collapse_id} -> {plot_path}")

    # Save dataframes
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    base_path = analysis_dir / base_name

    inside_path = _save_df(stats_inside, base_path, "inside_stats")
    buffer_path = _save_df(stats_buffer, base_path, "buffer_stats")
    trend_path = _save_df(trend_fit, base_path, "trend_fit")
    points_path = _save_df(all_inside_points, base_path, "points")
    deviation_path = _save_df(deviation_df, base_path, "deviation")

    # Save summary
    summary = {
        "collapse_id": collapse_id,
        "date": collapse_date.isoformat(),
        "geom_wkt": collapse_row.get("geom_wkt"),
        "area": collapse_row.get("area"),
        "volume": collapse_row.get("volume"),
        "n_points_inside_total": int(all_inside_points.shape[0]),
        "deviation_score_before_collapse": deviation_score_before,
        "file_plot": str(plot_path),
        "file_stats_inside": str(inside_path) if inside_path else None,
        "file_stats_buffer": str(buffer_path) if buffer_path else None,
        "file_trend_fit": str(trend_path) if trend_path else None,
        "file_points": str(points_path) if points_path else None,
        "file_deviation": str(deviation_path) if deviation_path else None,
        "params": {
            "MIN_VELOCITY": MIN_VELOCITY,
            "OUTLIER_THRESHOLD": OUTLIER_THRESHOLD,
            "BUFFER_DISTANCE_PX": buffer_distance,
            "ROBUST_METHOD": velocity_trend_method,
            "LOWESS_FRAC": LOWESS_FRAC,
            "DAYS_BEFORE": days_before,
        },
    }
    pickle_path = analysis_dir / f"{base_name}_summary.pkl"
    try:
        with open(pickle_path, "wb") as f:
            pickle.dump(summary, f)
        logger.debug(f"Saved summary pickle -> {pickle_path}")
    except Exception:
        logger.warning(
            f"Failed to pickle summary for collapse {collapse_id}", exc_info=True
        )

    return True


if __name__ == "__main__":

    def _process_row(row):
        try:
            process_collapse(
                row,
                cfg=cfg,
                days_before=DAYS_BEFORE,
                day_after=DAY_AFTER,
                out_dir=OUTPUT_DIR,
                velocity_trend_method=ROBUST_METHOD,
                buffer_distance=BUFFER_DISTANCE_PX,
                min_dic_days_before=MIN_DIC_DAYS_BEFORE,
                use_center_date=USE_CENTER_DATE,
            )
        except Exception:
            logger.exception(
                f"Unexpected error plotting collapse id={int(row.get('id', -1))}"
            )

    cfg = ConfigManager(CONFIG_PATH)

    engine = create_engine(cfg.db_url)
    df = get_collapses_df(engine)

    # Filter by minimum area
    df = df[df["area"] >= MIN_COLLAPSE_AREA].reset_index(drop=True)

    # df_sorted = df.sort_values("id").reset_index(drop=True)
    with Parallel(n_jobs=N_JOBS, backend="threading") as parallel:
        parallel(
            delayed(_process_row)(row)
            for _, row in tqdm(
                df.iterrows(),
                total=df.shape[0],
                desc="Processing collapses",
            )
        )
