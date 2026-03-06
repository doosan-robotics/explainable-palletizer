"""Unit tests for box_spawn_config -- no Isaac Sim required."""

from __future__ import annotations

import pytest
from drp_sim.box_spawn_config import BufferParams, SpawnParams, load_box_spawn_config

_MINIMAL_YAML = """\
types:
  normal:
    usd_paths:
      - boxes/SM_CardBoxD_05.usd
    weight: 0.6
    x_choices: [0.5]
    y_choices: [0.25]
    z_choices: [0.25]
    sticker_probability: 1.0
"""

_FULL_YAML = """\
spawn:
  position: [1.0, 2.0, 3.0]
  velocity: [0.0, -0.5, 0.0]
  interval: 5.0
sticker_path:
  metadata_file: /tmp/stickers/metadata.json

types:
  normal:
    weight: 0.7
    usd_paths:
      - boxes/SM_CardBoxD_05.usd
      - boxes/SM_CardBoxD_01.usd
    x_choices: [0.50, 0.25]
    y_choices: [0.50]
    z_choices: [0.25]
    sticker_probability: 1.0
  damaged:
    weight: 0.3
    usd_paths:
      - boxes/damaged/damaged_untaped.usd
    x_choices: [0.50]
    y_choices: [0.25]
    z_choices: [0.25]
    sticker_probability: 0.0
"""


def test_load_default_config() -> None:
    """Loading with path=None should return the bundled default config."""
    configs, weights, spawn, _buffer = load_box_spawn_config()
    assert "normal" in configs
    assert "fragile" in configs
    assert "heavy" in configs
    assert "damaged" in configs
    assert len(weights) >= 4
    assert isinstance(spawn, SpawnParams)


def test_load_minimal_yaml(tmp_path) -> None:
    """A minimal YAML with one type should parse correctly."""
    cfg = tmp_path / "test.yaml"
    cfg.write_text(_MINIMAL_YAML)

    configs, weights, _spawn, _buffer = load_box_spawn_config(cfg)

    assert "normal" in configs
    assert len(configs) == 1
    assert weights["normal"] == pytest.approx(0.6)
    # USD paths should be resolved to absolute paths
    assert configs["normal"].usd_paths[0].endswith("boxes/SM_CardBoxD_05.usd")
    assert configs["normal"].x_choices == [0.5]
    assert configs["normal"].sticker_probability == 1.0


def test_load_full_yaml(tmp_path) -> None:
    """Full YAML with spawn section and multiple types."""
    cfg = tmp_path / "test.yaml"
    cfg.write_text(_FULL_YAML)

    configs, weights, spawn, _buffer = load_box_spawn_config(cfg)

    assert len(configs) == 2
    assert weights["normal"] == pytest.approx(0.7)
    assert weights["damaged"] == pytest.approx(0.3)
    assert spawn.position == (1.0, 2.0, 3.0)
    assert spawn.velocity == (0.0, -0.5, 0.0)
    assert spawn.interval == 5.0
    assert spawn.sticker_metadata == "/tmp/stickers/metadata.json"


def test_sticker_probability_zero(tmp_path) -> None:
    """A type with sticker_probability 0.0 should parse correctly."""
    cfg = tmp_path / "test.yaml"
    cfg.write_text(_FULL_YAML)

    configs, _, _, _ = load_box_spawn_config(cfg)
    assert configs["damaged"].sticker_probability == 0.0


def test_missing_spawn_section_uses_defaults(tmp_path) -> None:
    """When spawn section is absent, SpawnParams defaults apply."""
    cfg = tmp_path / "test.yaml"
    cfg.write_text(_MINIMAL_YAML)

    _, _, spawn, _ = load_box_spawn_config(cfg)
    assert spawn.position == (0.59, 3.01, -0.20)
    assert spawn.velocity == (0.0, -0.3, 0.0)
    assert spawn.interval == 3.0
    assert spawn.sticker_metadata is None


def test_missing_usd_paths_raises(tmp_path) -> None:
    """A type entry without usd_paths should raise ValueError."""
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("types:\n  broken:\n    weight: 1.0\n")

    with pytest.raises(ValueError, match=r"broken.*usd_paths"):
        load_box_spawn_config(cfg)


def test_empty_file_raises(tmp_path) -> None:
    """An empty YAML file should raise ValueError."""
    cfg = tmp_path / "empty.yaml"
    cfg.write_text("")

    with pytest.raises(ValueError, match="empty"):
        load_box_spawn_config(cfg)


def test_nonexistent_file_raises() -> None:
    """A path that does not exist should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_box_spawn_config("/tmp/does_not_exist_12345.yaml")


def test_usd_paths_resolved_to_absolute(tmp_path) -> None:
    """USD paths should be resolved to absolute filesystem paths."""
    cfg = tmp_path / "test.yaml"
    cfg.write_text(_MINIMAL_YAML)

    configs, _, _, _ = load_box_spawn_config(cfg)
    for p in configs["normal"].usd_paths:
        assert p.startswith("/"), f"Expected absolute path, got: {p}"


def test_default_weight_when_omitted(tmp_path) -> None:
    """When weight is omitted from a type, it should default to 1.0."""
    cfg = tmp_path / "test.yaml"
    cfg.write_text("types:\n  simple:\n    usd_paths:\n      - boxes/SM_CardBoxD_05.usd\n")

    _, weights, _, _ = load_box_spawn_config(cfg)
    assert weights["simple"] == pytest.approx(1.0)


# -- Buffer config tests --


_BUFFER_YAML = """\
spawn:
  position: [0.59, 3.01, -0.20]
buffer:
  endpoint: [0.5, -1.0, -0.3]
  length: 5
types:
  normal:
    usd_paths:
      - boxes/SM_CardBoxD_05.usd
"""


def test_buffer_section_parsed(tmp_path) -> None:
    """Buffer section should be parsed into BufferParams."""
    cfg = tmp_path / "test.yaml"
    cfg.write_text(_BUFFER_YAML)

    _, _, _, buffer = load_box_spawn_config(cfg)
    assert isinstance(buffer, BufferParams)
    assert buffer.endpoint == (0.5, -1.0, -0.3)
    assert buffer.length == 5


def test_missing_buffer_section_uses_defaults(tmp_path) -> None:
    """When buffer section is absent, BufferParams defaults apply."""
    cfg = tmp_path / "test.yaml"
    cfg.write_text(_MINIMAL_YAML)

    _, _, _, buffer = load_box_spawn_config(cfg)
    assert buffer.endpoint == (0.59, -0.75, -0.20)
    assert buffer.length == 3


def test_default_config_has_buffer() -> None:
    """The bundled default config should include buffer params."""
    _, _, _, buffer = load_box_spawn_config()
    assert isinstance(buffer, BufferParams)
    assert buffer.length == 3
