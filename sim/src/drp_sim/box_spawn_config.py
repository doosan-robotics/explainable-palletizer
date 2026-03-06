"""YAML-based box spawn configuration loader.

Loads box type definitions and spawn parameters from a YAML file,
resolving relative USD asset paths to absolute filesystem paths.
The bundled ``sim/configs/box_spawn.yaml`` is used when no path is given.

Configuration priority (highest wins):
    1. CLI arguments (--type-weights, --spawn-interval)
    2. YAML file
    3. Hard-coded defaults in ``box_spawner.py``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from drp_sim.box_spawner import BoxTypeConfig

_ASSETS_DIR = Path(__file__).parent.parent.parent / "usd" / "assets"
_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "configs" / "box_spawn.yaml"

_DEFAULT_SPAWN_POSITION = (0.59, 3.01, -0.20)
_DEFAULT_SPAWN_VELOCITY = (0.0, -0.3, 0.0)
_DEFAULT_SPAWN_INTERVAL = 3.0

_DEFAULT_BUFFER_ENDPOINT = (0.59, -0.75, -0.20)
_DEFAULT_BUFFER_LENGTH = 3


@dataclass(frozen=True, slots=True)
class BufferParams:
    """Buffer zone parameters loaded from the ``buffer`` YAML section."""

    endpoint: tuple[float, float, float] = _DEFAULT_BUFFER_ENDPOINT
    length: int = _DEFAULT_BUFFER_LENGTH


@dataclass(frozen=True, slots=True)
class SpawnParams:
    """Spawn point geometry loaded from the ``spawn`` YAML section."""

    position: tuple[float, float, float] = _DEFAULT_SPAWN_POSITION
    velocity: tuple[float, float, float] = _DEFAULT_SPAWN_VELOCITY
    interval: float = _DEFAULT_SPAWN_INTERVAL
    sticker_metadata: str | None = None


def _resolve_usd_paths(relative_paths: list[str]) -> list[str]:
    """Convert asset-relative paths to absolute paths."""
    return [str(_ASSETS_DIR / p) for p in relative_paths]


def _parse_type_entry(name: str, entry: dict) -> tuple[BoxTypeConfig, float]:
    """Parse a single type entry from the YAML ``types`` mapping.

    Returns (BoxTypeConfig, weight).

    Raises
    ------
    ValueError
        If required fields are missing.
    """
    if "usd_paths" not in entry:
        raise ValueError(f"Type '{name}' is missing required field 'usd_paths'")

    usd_paths = _resolve_usd_paths(entry["usd_paths"])
    weight = float(entry.get("weight", 1.0))

    config = BoxTypeConfig(
        usd_paths=usd_paths,
        x_choices=entry.get("x_choices", [0.25]),
        y_choices=entry.get("y_choices", [0.25]),
        z_choices=entry.get("z_choices", [0.25]),
        sticker_probability=float(entry.get("sticker_probability", 0.5)),
        visuals=entry.get("visuals", []),
    )
    return config, weight


def _parse_buffer_section(raw: dict | None) -> BufferParams:
    """Parse the optional ``buffer`` YAML section."""
    if raw is None:
        return BufferParams()
    endpoint = tuple(raw["endpoint"]) if "endpoint" in raw else _DEFAULT_BUFFER_ENDPOINT
    length = int(raw["length"]) if "length" in raw else _DEFAULT_BUFFER_LENGTH
    return BufferParams(endpoint=endpoint, length=length)


def _parse_spawn_section(raw: dict | None, sticker_path: dict | None = None) -> SpawnParams:
    """Parse the optional ``spawn`` and ``sticker_path`` YAML sections."""
    pos = _DEFAULT_SPAWN_POSITION
    vel = _DEFAULT_SPAWN_VELOCITY
    interval = _DEFAULT_SPAWN_INTERVAL
    sticker_metadata: str | None = None

    if raw is not None:
        pos = tuple(raw["position"]) if "position" in raw else pos
        vel = tuple(raw["velocity"]) if "velocity" in raw else vel
        interval = float(raw["interval"]) if "interval" in raw else interval
        sticker_metadata = raw.get("sticker_metadata")

    # sticker_path section overrides spawn.sticker_metadata
    if sticker_path is not None and "metadata_file" in sticker_path:
        sticker_metadata = sticker_path["metadata_file"]

    return SpawnParams(
        position=pos,
        velocity=vel,
        interval=interval,
        sticker_metadata=sticker_metadata,
    )


def load_box_spawn_config(
    path: Path | str | None = None,
) -> tuple[dict[str, BoxTypeConfig], dict[str, float], SpawnParams, BufferParams]:
    """Load box spawn configuration from a YAML file.

    Parameters
    ----------
    path:
        Path to a YAML config file.  ``None`` loads the bundled default
        at ``sim/configs/box_spawn.yaml``.

    Returns
    -------
    tuple
        ``(type_configs, type_weights, spawn_params, buffer_params)`` where
        *type_configs* maps type name to :class:`BoxTypeConfig`,
        *type_weights* maps type name to its sampling weight,
        *spawn_params* holds conveyor geometry, and *buffer_params* holds
        buffer zone configuration.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the YAML is empty or a type entry is missing required fields.
    """
    resolved = Path(path) if path is not None else _DEFAULT_CONFIG
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    with open(resolved) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError("Config file is empty")

    spawn_params = _parse_spawn_section(data.get("spawn"), data.get("sticker_path"))
    buffer_params = _parse_buffer_section(data.get("buffer"))

    types_raw = data.get("types", {})
    type_configs: dict[str, BoxTypeConfig] = {}
    type_weights: dict[str, float] = {}

    for name, entry in types_raw.items():
        config, weight = _parse_type_entry(name, entry)
        type_configs[name] = config
        type_weights[name] = weight

    return type_configs, type_weights, spawn_params, buffer_params
