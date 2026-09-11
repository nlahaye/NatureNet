#!/usr/bin/env python3
"""Create SWIRL reward, density, and trajectory visualizations from one YAML config.

Usage:
    python plot_swirl_clean.py path/to/plot_swirl.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import colorcet as cc
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sparse
import yaml
from matplotlib.animation import PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap


LOGGER = logging.getLogger(__name__)

DEFAULT_COLORS = [
    "red",
    "green",
    "blue",
    "brown",
    "cyan",
    "orange",
    "black",
    "magenta",
    "lightcoral",
    "goldenrod",
    "olive",
    "rosybrown",
    "silver",
    "sandybrown",
    "teal",
    "springgreen",
    "hotpink",
    "indigo",
    "darkkhaki",
    "navy",
]


@dataclass(frozen=True)
class PlotConfig:
    output_dir: Path
    transition_probabilities: Path
    environment_maps: Path
    paths: list[Path]
    uids: list[str]

    training_results: Path
    train_latent_paths: Path
    test_latent_paths: Path

    n_hidden: int

    plot_gifs: bool
    plot_density: bool
    gif_fps: int
    gif_interval_ms: int
    dpi: int
    min_path_length: int
    colors: list[str]

    @classmethod
    def from_yaml(cls, config_path: Path) -> "PlotConfig":
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}

        base_dir = config_path.parent.resolve()

        paths = _require(raw, "paths")
        uids = _require(raw, "uids")

        if len(paths) != len(uids):
            raise ValueError("'paths' and 'uids' must have the same number of entries.")

        artifacts = _require(raw, "artifacts")
        plotting = raw.get("plotting", {})

        return cls(
            output_dir=_resolve_path(base_dir, _require(raw, "output_dir")),
            transition_probabilities=_resolve_path(
                base_dir,
                _require(raw, "transition_probabilities"),
            ),
            environment_maps=_resolve_path(
                base_dir,
                _require(raw, "environment_maps"),
            ),
            paths=[_resolve_path(base_dir, value) for value in paths],
            uids=[str(value) for value in uids],
            training_results=_resolve_path(
                base_dir,
                _require(artifacts, "training_results"),
            ),
            train_latent_paths=_resolve_path(
                base_dir,
                _require(artifacts, "train_latent_paths"),
            ),
            test_latent_paths=_resolve_path(
                base_dir,
                _require(artifacts, "test_latent_paths"),
            ),
            n_hidden=int(_require(raw, "n_hidden")),
            plot_gifs=bool(plotting.get("plot_gifs", True)),
            plot_density=bool(plotting.get("plot_density", False)),
            gif_fps=int(plotting.get("gif_fps", 3)),
            gif_interval_ms=int(plotting.get("gif_interval_ms", 300)),
            dpi=int(plotting.get("dpi", 300)),
            min_path_length=int(plotting.get("min_path_length", 2)),
            colors=list(plotting.get("colors", DEFAULT_COLORS)),
        )


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"Missing required YAML key: '{key}'.")
    return mapping[key]


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def normalize(values: np.ndarray) -> np.ndarray:
    """Normalize finite values to [0, 1], safely handling constant arrays."""
    values = np.asarray(values, dtype=float)

    if not np.isfinite(values).any():
        return np.zeros_like(values)

    minimum = np.nanmin(values)
    maximum = np.nanmax(values)

    if np.isclose(minimum, maximum):
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


def latent_cmap() -> ListedColormap:
    """Return a categorical colormap suitable for environment-state labels."""
    colors = sns.color_palette(cc.glasbey, n_colors=300)
    return ListedColormap(colors)


def map_reward_to_environment(
    environment: np.ndarray,
    reward: np.ndarray,
) -> np.ndarray:
    """
    Convert an integer-valued environment-class map into a reward map.

    Environment values <= 0 are treated as invalid/background and become NaN.
    """
    environment = np.asarray(environment)
    reward = np.asarray(reward).squeeze()

    result = np.full(environment.shape, np.nan, dtype=float)

    valid_mask = np.isfinite(environment) & (environment > 0)
    environment_indices = environment[valid_mask].astype(int)

    valid_reward_index = (
        (environment_indices >= 0)
        & (environment_indices < reward.size)
    )

    flat_result = result.ravel()
    flat_valid_indices = np.flatnonzero(valid_mask)

    flat_result[flat_valid_indices[valid_reward_index]] = (
        reward[environment_indices[valid_reward_index]]
    )

    return result


def get_path_bounds(
    path: np.ndarray,
    shape: tuple[int, int],
    padding: int = 10,
) -> tuple[int, int, int, int]:
    """Return padded x/y bounds around valid path positions."""
    x = np.asarray(path["x"], dtype=int)
    y = np.asarray(path["y"], dtype=int)

    valid = (
        (x >= 0)
        & (y >= 0)
        & (x < shape[1])
        & (y < shape[0])
    )

    if not valid.any():
        return 0, shape[1], 0, shape[0]

    xmin = max(0, int(x[valid].min()) - padding)
    xmax = min(shape[1], int(x[valid].max()) + padding + 1)

    ymin = max(0, int(y[valid].min()) - padding)
    ymax = min(shape[0], int(y[valid].max()) + padding + 1)

    return xmin, xmax, ymin, ymax


def valid_path_and_latents(
    path: np.ndarray,
    latents: np.ndarray,
    environments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trim aligned arrays to their common valid time length."""
    length = min(len(path), len(latents), len(environments))

    return (
        np.asarray(path[:length]),
        np.asarray(latents[:length], dtype=int),
        np.asarray(environments[:length]),
    )


def plot_reward_heatmaps(
    reward_by_latent: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> None:
    """Write one reward heatmap per latent state plus a distribution boxplot."""
    normalized_rewards = normalize(reward_by_latent)
    distributions: list[np.ndarray] = []

    for latent_index, reward in enumerate(normalized_rewards):
        side_length = math.ceil(math.sqrt(reward.size))

        square = np.full(side_length * side_length, np.nan)
        square[: reward.size] = reward
        square = square.reshape(side_length, side_length)

        distributions.append(reward)

        figure, axis = plt.subplots(figsize=(6, 5))

        image = axis.imshow(
            square,
            cmap="jet",
            interpolation="none",
            vmin=0,
            vmax=1,
        )

        figure.colorbar(
            image,
            ax=axis,
            orientation="horizontal",
            pad=0.08,
            label="Normalized reward",
        )

        axis.set_title(f"Reward heatmap: latent state {latent_index}")
        axis.set_axis_off()

        figure.tight_layout()

        figure.savefig(
            output_dir / f"reward_heatmap_{latent_index}.png",
            dpi=dpi,
            bbox_inches="tight",
        )

        plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))

    axis.boxplot(
        distributions,
        labels=range(1, len(distributions) + 1),
    )

    axis.set_xlabel("Latent state")
    axis.set_ylabel("Normalized reward")

    figure.tight_layout()

    figure.savefig(
        output_dir / "reward_distribution_boxplot.png",
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_environment_animation(
    path: np.ndarray,
    environments: np.ndarray,
    latents: np.ndarray,
    output_name: str,
    output_dir: Path,
    colors: list[str],
    fps: int,
    interval_ms: int,
    dpi: int,
) -> None:
    """Create a GIF of trajectory progression over environment labels."""
    path, latents, environments = valid_path_and_latents(
        path,
        latents,
        environments,
    )

    if len(path) < 2:
        LOGGER.warning(
            "Skipping environment GIF for %s: fewer than two valid points.",
            output_name,
        )
        return

    xmin, xmax, ymin, ymax = get_path_bounds(
        path,
        environments[0].shape,
    )

    figure, axis = plt.subplots(figsize=(8, 6))
    cmap = latent_cmap()

    def masked_environment(frame: int) -> np.ndarray:
        image_data = np.asarray(environments[frame], dtype=float).copy()
        image_data[image_data <= 0] = np.nan
        return image_data

    image = axis.imshow(
        masked_environment(0),
        cmap=cmap,
        interpolation="none",
        origin="upper",
    )

    line, = axis.plot(
        [],
        [],
        color="white",
        linewidth=2,
        zorder=2,
    )

    point = axis.scatter(
        [],
        [],
        s=80,
        edgecolor="white",
        zorder=3,
    )

    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymax, ymin)
    axis.set_title(output_name)

    def update(frame: int):
        image.set_data(masked_environment(frame))

        x_values = path["x"][: frame + 1]
        y_values = path["y"][: frame + 1]

        line.set_data(x_values, y_values)

        point.set_offsets([[x_values[-1], y_values[-1]]])
        point.set_facecolor(
            colors[int(latents[frame]) % len(colors)]
        )

        return image, line, point

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(path),
        interval=interval_ms,
        blit=True,
    )

    movie.save(
        output_dir / f"environment_{output_name}.gif",
        writer=PillowWriter(fps=fps),
        dpi=dpi,
    )

    plt.close(figure)


def plot_reward_animation(
    path: np.ndarray,
    environments: np.ndarray,
    latents: np.ndarray,
    reward_by_latent: np.ndarray,
    output_name: str,
    output_dir: Path,
    colors: list[str],
    fps: int,
    interval_ms: int,
    dpi: int,
) -> None:
    """Create a GIF of trajectory progression over latent-specific reward maps."""
    path, latents, environments = valid_path_and_latents(
        path,
        latents,
        environments,
    )

    if len(path) < 2:
        LOGGER.warning(
            "Skipping reward GIF for %s: fewer than two valid points.",
            output_name,
        )
        return

    xmin, xmax, ymin, ymax = get_path_bounds(
        path,
        environments[0].shape,
    )

    normalized_rewards = normalize(reward_by_latent)

    figure, axis = plt.subplots(figsize=(8, 6))

    def reward_heatmap(frame: int) -> np.ndarray:
        latent = int(latents[frame])
        return map_reward_to_environment(
            environments[frame],
            normalized_rewards[latent],
        )

    image = axis.imshow(
        reward_heatmap(0),
        cmap="jet",
        interpolation="none",
        vmin=0,
        vmax=1,
        origin="upper",
    )

    line, = axis.plot(
        [],
        [],
        color="black",
        linewidth=2,
        zorder=2,
    )

    point = axis.scatter(
        [],
        [],
        s=80,
        edgecolor="black",
        zorder=3,
    )

    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymax, ymin)
    axis.set_title(output_name)

    def update(frame: int):
        image.set_data(reward_heatmap(frame))

        x_values = path["x"][: frame + 1]
        y_values = path["y"][: frame + 1]

        line.set_data(x_values, y_values)

        point.set_offsets([[x_values[-1], y_values[-1]]])
        point.set_facecolor(
            colors[int(latents[frame]) % len(colors)]
        )

        return image, line, point

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(path),
        interval=interval_ms,
        blit=True,
    )

    movie.save(
        output_dir / f"reward_{output_name}.gif",
        writer=PillowWriter(fps=fps),
        dpi=dpi,
    )

    plt.close(figure)


def plot_trajectory_segments(
    latent_paths: list[np.ndarray],
    paths: list[np.ndarray],
    n_hidden: int,
    output_dir: Path,
    dpi: int,
) -> None:
    """Write a trajectory-segment figure for each latent state."""
    segments_by_latent: dict[int, list[np.ndarray]] = {
        state: [] for state in range(n_hidden)
    }

    for latents, path in zip(latent_paths, paths):
        length = min(len(latents), len(path))

        latents = np.asarray(latents[:length], dtype=int)
        path = np.asarray(path[:length])

        for latent_state in range(n_hidden):
            state_indices = np.flatnonzero(latents == latent_state)

            contiguous_segments = np.split(
                state_indices,
                np.where(np.diff(state_indices) != 1)[0] + 1,
            )

            for indices in contiguous_segments:
                if len(indices) < 2:
                    continue

                coordinates = np.column_stack(
                    (
                        path["x"][indices],
                        path["y"][indices],
                    )
                )

                segments_by_latent[latent_state].append(
                    coordinates
                )

    for latent_state, trajectories in segments_by_latent.items():
        if not trajectories:
            continue

        line_segments = []

        for coordinates in trajectories:
            line_segments.append(
                np.stack(
                    (
                        coordinates[:-1],
                        coordinates[1:],
                    ),
                    axis=1,
                )
            )

        all_segments = np.concatenate(line_segments, axis=0)

        figure, axis = plt.subplots(figsize=(10, 6))

        collection = LineCollection(
            all_segments,
            cmap="viridis",
            linewidths=1.5,
        )

        collection.set_array(
            np.linspace(0, 1, len(all_segments))
        )

        axis.add_collection(collection)
        axis.autoscale()
        axis.set_aspect("equal")

        axis.set_title(
            f"Trajectory segments: latent state {latent_state}"
        )

        figure.colorbar(
            collection,
            ax=axis,
            label="Within-segment progression",
        )

        figure.tight_layout()

        figure.savefig(
            output_dir / f"trajectory_segments_latent_{latent_state}.png",
            dpi=dpi,
            bbox_inches="tight",
        )

        plt.close(figure)


def plot_density_by_latent(
    latent_paths: list[np.ndarray],
    paths: list[np.ndarray],
    environments: list[np.ndarray],
    reward_by_latent: np.ndarray,
    output_dir: Path,
    colors: list[str],
    dpi: int,
) -> None:
    """Create environment-class and reward KDE plots by latent state."""
    n_hidden = reward_by_latent.shape[0]

    environment_values: list[list[float]] = [
        [] for _ in range(n_hidden)
    ]

    reward_values: list[list[float]] = [
        [] for _ in range(n_hidden)
    ]

    normalized_rewards = normalize(reward_by_latent)

    for latents, path, environment_series in zip(
        latent_paths,
        paths,
        environments,
    ):
        path, latents, environment_series = valid_path_and_latents(
            path,
            latents,
            environment_series,
        )

        for index, (position, latent) in enumerate(
            zip(path, latents)
        ):
            x = int(position["x"])
            y = int(position["y"])

            environment = environment_series[index]

            if (
                x < 0
                or y < 0
                or x >= environment.shape[1]
                or y >= environment.shape[0]
            ):
                continue

            latent_state = int(latent)
            environment_value = int(environment[y, x])

            if (
                environment_value <= 0
                or environment_value >= normalized_rewards.shape[1]
            ):
                continue

            environment_values[latent_state].append(
                environment_value
            )

            reward_values[latent_state].append(
                float(
                    normalized_rewards[
                        latent_state,
                        environment_value,
                    ]
                )
            )

    plot_specs = [
        (
            environment_values,
            "Environment class",
            "environment_density.png",
        ),
        (
            reward_values,
            "Normalized reward",
            "reward_density.png",
        ),
    ]

    for values_by_latent, x_label, filename in plot_specs:
        figure, axis = plt.subplots(figsize=(9, 5))

        has_curves = False

        for latent_state, values in enumerate(values_by_latent):
            if len(values) < 2 or len(set(values)) < 2:
                continue

            sns.kdeplot(
                values,
                ax=axis,
                label=f"Latent state {latent_state}",
                color=colors[latent_state % len(colors)],
                fill=True,
                alpha=0.25,
            )

            has_curves = True

        axis.set_xlabel(x_label)
        axis.set_ylabel("Density")

        if has_curves:
            axis.legend()

        figure.tight_layout()

        figure.savefig(
            output_dir / filename,
            dpi=dpi,
            bbox_inches="tight",
        )

        plt.close(figure)


def load_latent_sequences(
    train_file: Path,
    test_file: Path,
) -> np.ndarray:
    """Load and concatenate learned train/test latent state paths."""
    with np.load(train_file, allow_pickle=True) as train_data:
        train_latents = np.asarray(
            train_data["jax_path_vmap"]
        )

    with np.load(test_file, allow_pickle=True) as test_data:
        test_latents = np.asarray(
            test_data["jax_path_vmap"]
        )

    return np.concatenate(
        (
            train_latents,
            test_latents,
        ),
        axis=0,
    )


def load_reward_by_latent(
    config: PlotConfig,
) -> np.ndarray:
    """
    Load reward values and average across actions.

    Returns
    -------
    np.ndarray
        Shape: (n_hidden, n_states)
    """
    transition_probabilities = sparse.load_npz(
        config.transition_probabilities
    )

    n_states, n_actions, _ = transition_probabilities.shape

    with np.load(
        config.training_results,
        allow_pickle=True,
    ) as training_data:
        reward = np.asarray(training_data["new_reward"])

    if reward.ndim < 3:
        raise ValueError(
            "Expected 'new_reward' to have at least 3 dimensions. "
            f"Received shape {reward.shape}."
        )

    reward = np.asarray(reward[:, 0, :])

    expected_size = config.n_hidden * n_states * n_actions

    if reward.size != expected_size:
        raise ValueError(
            "Reward tensor cannot be reshaped using the configured "
            f"n_hidden={config.n_hidden}, inferred n_states={n_states}, "
            f"and inferred n_actions={n_actions}. "
            f"Expected {expected_size} elements but found {reward.size}."
        )

    reward = reward.reshape(
        config.n_hidden,
        n_states,
        n_actions,
    )

    return np.nanmean(reward, axis=-1)


def collect_series(
    config: PlotConfig,
    latent_sequences: np.ndarray,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[str],
]:
    """
    Load paths and environment sequences, then align them with latent paths.

    The sequence ordering is assumed to be:

    1. All trajectories from the first YAML `paths` entry.
    2. All trajectories from the second YAML `paths` entry.
    3. Continue in YAML order.

    The train and test latent arrays are concatenated in that same order.
    """
    environments_by_uid = load_pickle(config.environment_maps)

    selected_paths: list[np.ndarray] = []
    selected_environments: list[np.ndarray] = []
    selected_latents: list[np.ndarray] = []
    selected_uids: list[str] = []

    latent_index = 0

    for uid, path_file in zip(config.uids, config.paths):
        if not path_file.exists():
            raise FileNotFoundError(
                f"Path file does not exist: {path_file}"
            )

        path_sets = np.load(path_file, allow_pickle=True)

        simple_uid = re.sub(r"__aug\d+", "", uid)

        if simple_uid not in environments_by_uid:
            raise KeyError(
                f"No environment maps found for UID '{simple_uid}'."
            )

        environment_sets = environments_by_uid[simple_uid]

        if len(path_sets) != len(environment_sets):
            raise ValueError(
                f"Path/environment sequence count mismatch for UID '{uid}': "
                f"{len(path_sets)} path sequences versus "
                f"{len(environment_sets)} environment sequences."
            )

        for path, environment_series in zip(
            path_sets,
            environment_sets,
        ):
            if latent_index >= len(latent_sequences):
                raise IndexError(
                    "More path sequences were found than latent-state "
                    "sequences available in the train/test latent files."
                )

            latents = latent_sequences[latent_index]
            latent_index += 1

            # Skip augmented copies in plotting while still consuming
            # their associated latent-state sequence.
            if simple_uid != uid:
                continue

            length = min(
                len(path),
                len(environment_series),
                len(latents),
            )

            if length < config.min_path_length:
                LOGGER.warning(
                    "Skipping UID '%s' trajectory with length %d "
                    "(minimum is %d).",
                    uid,
                    length,
                    config.min_path_length,
                )
                continue

            selected_paths.append(np.asarray(path[:length]))
            selected_environments.append(
                np.asarray(environment_series[:length])
            )
            selected_latents.append(
                np.asarray(latents[:length], dtype=int)
            )
            selected_uids.append(uid)

    if not selected_paths:
        raise RuntimeError(
            "No valid trajectories remained after loading and filtering."
        )

    return (
        selected_paths,
        selected_environments,
        selected_latents,
        selected_uids,
    )


def run(config: PlotConfig) -> None:
    """Run all requested SWIRL plotting products."""
    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGGER.info(
        "Writing output to %s",
        config.output_dir,
    )

    reward_by_latent = load_reward_by_latent(config)

    if reward_by_latent.shape[0] != config.n_hidden:
        raise ValueError(
            "The configured n_hidden does not match the reward array. "
            f"Expected {config.n_hidden}; received "
            f"{reward_by_latent.shape[0]}."
        )

    plot_reward_heatmaps(
        reward_by_latent,
        config.output_dir,
        config.dpi,
    )

    latent_sequences = load_latent_sequences(
        config.train_latent_paths,
        config.test_latent_paths,
    )

    (
        paths,
        environments,
        latent_paths,
        uids,
    ) = collect_series(
        config,
        latent_sequences,
    )

    LOGGER.info(
        "Loaded %d valid trajectories.",
        len(paths),
    )

    if config.plot_gifs:
        for index, (path, environment, latents, uid) in enumerate(
            zip(
                paths,
                environments,
                latent_paths,
                uids,
            )
        ):
            output_name = f"{uid}_trajectory_{index:03d}"

            LOGGER.info(
                "Writing trajectory GIFs for %s",
                output_name,
            )

            plot_environment_animation(
                path=path,
                environments=environment,
                latents=latents,
                output_name=output_name,
                output_dir=config.output_dir,
                colors=config.colors,
                fps=config.gif_fps,
                interval_ms=config.gif_interval_ms,
                dpi=config.dpi,
            )

            plot_reward_animation(
                path=path,
                environments=environment,
                latents=latents,
                reward_by_latent=reward_by_latent,
                output_name=output_name,
                output_dir=config.output_dir,
                colors=config.colors,
                fps=config.gif_fps,
                interval_ms=config.gif_interval_ms,
                dpi=config.dpi,
            )

    plot_trajectory_segments(
        latent_paths=latent_paths,
        paths=paths,
        n_hidden=config.n_hidden,
        output_dir=config.output_dir,
        dpi=config.dpi,
    )

    if config.plot_density:
        plot_density_by_latent(
            latent_paths=latent_paths,
            paths=paths,
            environments=environments,
            reward_by_latent=reward_by_latent,
            output_dir=config.output_dir,
            colors=config.colors,
            dpi=config.dpi,
        )

    LOGGER.info("Completed plotting successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create SWIRL visualizations from a single YAML "
            "configuration file."
        )
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ],
        help="Logging verbosity.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    config_path = args.config.resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    config = PlotConfig.from_yaml(config_path)

    run(config)


if __name__ == "__main__":
    main()



