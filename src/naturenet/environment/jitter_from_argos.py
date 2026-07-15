from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def argos_orientation_to_math_radians(theta_deg_from_north_ccw: np.ndarray) -> np.ndarray:
    """
    Convert Argos orientation to a standard mathematical angle in radians.

    Argos orientation is measured counter-clockwise from north.
    Standard 2D rotation matrices expect counter-clockwise from +x (east).

    So:
        phi = 90 degrees - theta_argos
    """
    return np.deg2rad(90.0 - np.asarray(theta_deg_from_north_ccw, dtype=float))


def argos_ellipse_to_covariance(
    semi_major_m: np.ndarray,
    semi_minor_m: np.ndarray,
    orientation_deg: np.ndarray,
) -> np.ndarray:
    """
    Convert Argos ellipse parameters into a covariance matrix for each row.

    Parameters
    ----------
    semi_major_m : array-like
        Semi-major axis length in meters.
    semi_minor_m : array-like
        Semi-minor axis length in meters.
    orientation_deg : array-like
        Argos orientation in degrees, counter-clockwise from north.

    Returns
    -------
    cov : np.ndarray
        Shape (n, 2, 2), covariance matrix per row.

    Notes
    -----
    This implementation assumes the Argos ellipse semi-axes map to principal
    covariance variances as:

        lambda_major = a^2 / 2
        lambda_minor = b^2 / 2
    """
    semi_major_m = np.asarray(semi_major_m, dtype=float)
    semi_minor_m = np.asarray(semi_minor_m, dtype=float)
    orientation_deg = np.asarray(orientation_deg, dtype=float)

    if not (len(semi_major_m) == len(semi_minor_m) == len(orientation_deg)):
        raise ValueError("semi_major_m, semi_minor_m, and orientation_deg must have the same length.")

    a = np.maximum(semi_major_m, 0.0)
    b = np.maximum(semi_minor_m, 0.0)

    lam_major = (a ** 2) / 2.0
    lam_minor = (b ** 2) / 2.0

    phi = argos_orientation_to_math_radians(orientation_deg)
    c = np.cos(phi)
    s = np.sin(phi)

    cov = np.zeros((len(phi), 2, 2), dtype=float)
    cov[:, 0, 0] = c * c * lam_major + s * s * lam_minor
    cov[:, 1, 1] = s * s * lam_major + c * c * lam_minor
    cov[:, 0, 1] = c * s * (lam_major - lam_minor)
    cov[:, 1, 0] = cov[:, 0, 1]

    return cov


def sample_argos_ellipse_offsets(
    semi_major_m: np.ndarray,
    semi_minor_m: np.ndarray,
    orientation_deg: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample east/north offsets in meters from each row's Argos error ellipse.

    Returns
    -------
    offsets : np.ndarray
        Shape (n, 2), columns are [dx_m, dy_m].
    """
    rng = rng or np.random.default_rng()
    covs = argos_ellipse_to_covariance(semi_major_m, semi_minor_m, orientation_deg)

    offsets = np.empty((covs.shape[0], 2), dtype=float)
    for i, cov in enumerate(covs):
        offsets[i] = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov)
    return offsets


def meters_to_degree_offsets(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    dx_m: np.ndarray,
    dy_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert local east/north meter offsets to lon/lat degree offsets.

    Uses a small-distance approximation.
    """
    lon_deg = np.asarray(lon_deg, dtype=float)
    lat_deg = np.asarray(lat_deg, dtype=float)
    dx_m = np.asarray(dx_m, dtype=float)
    dy_m = np.asarray(dy_m, dtype=float)

    earth_radius_m = 6378137.0
    lat_rad = np.deg2rad(lat_deg)
    cos_lat = np.cos(lat_rad)
    cos_lat = np.where(np.abs(cos_lat) < 1e-12, np.nan, cos_lat)

    dlat_deg = np.rad2deg(dy_m / earth_radius_m)
    dlon_deg = np.rad2deg(dx_m / (earth_radius_m * cos_lat))

    return lon_deg + dlon_deg, lat_deg + dlat_deg


def jitter_argos_positions(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    semi_major_col: str = "Argos.semi.major",
    semi_minor_col: str = "Argos.semi.minor",
    orientation_col: str = "Argos.orientation",
    rng: Optional[np.random.Generator] = None,
    output_offset_cols: bool = True,
) -> pd.DataFrame:
    """
    Jitter Argos positions by sampling from the Argos error ellipse.

    Required columns
    ----------------
    lon_col, lat_col : decimal degrees
    semi_major_col, semi_minor_col : meters
    orientation_col : degrees CCW from north
    """
    rng = rng or np.random.default_rng()
    out = df.copy()

    required = [lon_col, lat_col, semi_major_col, semi_minor_col, orientation_col]
    for col in required:
        if col not in out.columns:
            raise KeyError(f"Required column '{col}' not found.")

    offsets = sample_argos_ellipse_offsets(
        semi_major_m=pd.to_numeric(out[semi_major_col], errors="coerce").fillna(0.0).to_numpy(),
        semi_minor_m=pd.to_numeric(out[semi_minor_col], errors="coerce").fillna(0.0).to_numpy(),
        orientation_deg=pd.to_numeric(out[orientation_col], errors="coerce").fillna(0.0).to_numpy(),
        rng=rng,
    )

    new_lon, new_lat = meters_to_degree_offsets(
        lon_deg=pd.to_numeric(out[lon_col], errors="coerce").to_numpy(),
        lat_deg=pd.to_numeric(out[lat_col], errors="coerce").to_numpy(),
        dx_m=offsets[:, 0],
        dy_m=offsets[:, 1],
    )

    out[lon_col] = new_lon
    out[lat_col] = new_lat

    if output_offset_cols:
        out["argos_dx_m"] = offsets[:, 0]
        out["argos_dy_m"] = offsets[:, 1]

    return out


def augment_argos_tracks(
    df: pd.DataFrame,
    group_cols: Optional[Sequence[str]] = None,
    n_augmented_per_group: int = 3,
    lon_col: str = "lon",
    lat_col: str = "lat",
    semi_major_col: str = "Argos.semi.major",
    semi_minor_col: str = "Argos.semi.minor",
    orientation_col: str = "Argos.orientation",
    id_col: Optional[str] = None,
    random_seed: int = 42,
    preserve_original: bool = True,
) -> pd.DataFrame:
    """
    Create jittered replicas of Argos tracks by sampling each fix from its ellipse.

    Adds:
    - is_augmented
    - augmentation_id
    - source_group
    """
    if n_augmented_per_group < 1:
        raise ValueError("n_augmented_per_group must be >= 1")

    group_cols = list(group_cols) if group_cols is not None else []

    for col in group_cols:
        if col not in df.columns:
            raise KeyError(f"Group column '{col}' not found.")

    if id_col is not None and id_col not in df.columns:
        raise KeyError(f"id_col '{id_col}' not found.")

    rng = np.random.default_rng(random_seed)
    frames: List[pd.DataFrame] = []

    def _group_label(group_df: pd.DataFrame) -> pd.Series:
        if not group_cols:
            return pd.Series(["all_rows"] * len(group_df), index=group_df.index)
        return group_df[group_cols].astype(str).agg("|".join, axis=1)

    if preserve_original:
        original = df.copy()
        original["is_augmented"] = 0
        original["augmentation_id"] = 0
        original["source_group"] = _group_label(original)
        frames.append(original)

    grouped = [("all_rows", df)] if not group_cols else list(df.groupby(group_cols, dropna=False, sort=False))

    print(df.keys())
    for group_key, group_df in grouped:
        for aug_id in range(1, n_augmented_per_group + 1):
            
            print(group_df.keys())
            condition = group_df['lc'] == 'G'

            group_df.loc[condition, semi_major_col] = 50 
            group_df.loc[condition, semi_minor_col] = 50
            group_df.loc[condition, orientation_col] = 0

            jittered = jitter_argos_positions(
                group_df,
                lon_col=lon_col,
                lat_col=lat_col,
                semi_major_col=semi_major_col,
                semi_minor_col=semi_minor_col,
                orientation_col=orientation_col,
                rng=rng,
                output_offset_cols=True,
            )

            jittered["is_augmented"] = 1
            jittered["augmentation_id"] = aug_id
            jittered["source_group"] = str(group_key)

            if id_col is not None:
                jittered[id_col] = jittered[id_col].astype(str) + f"__aug{aug_id}"
                if "animal_name" in jittered:
                    jittered["animal_name"] = jittered["animal_name"].astype(str) + f"__aug{aug_id}"
 
            frames.append(jittered)

    return pd.concat(frames, ignore_index=True)



