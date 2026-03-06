# drp-sim

Isaac Sim environments for the AI palletizer. Handles simulation scene construction, object spawning, and task execution for cuMotion motion planning.

## Scope

| Capability | Description |
|---|---|
| **Scene task runner** | Configurable simulation loops with `World` lifecycle management |
| **Object spawn** | Procedural placement of pallets, boxes, and obstacles via USD prims |
| **cuMotion integration** | Scene export and collision world sync for cuRobo motion planning |

## Project Structure

```
sim/
  envs/          # Reusable environment configs
  scripts/       # Standalone entry points
  src/drp_sim/   # Core library
  tests/         # Unit and integration tests
  usd/assets/    # USD scene assets
    boxes/       # Box meshes and materials
    pallet/      # Pallet meshes and materials
```

## Prerequisites

- NVIDIA GPU with driver >= 535
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

From the workspace root:

```bash
uv sync --package drp-sim
```

This pulls `isaacsim[all,extscache]==5.1.0` from the NVIDIA PyPI index configured in the root `pyproject.toml`.

## Quick Start

Run the minimal scene test (ground plane + falling cube, 200 physics steps):

```bash
uv run python sim/scripts/test_scene.py
```

For headless execution (no GUI), edit `headless` in the script config:

```python
simulation_app = SimulationApp({"headless": True})
```

## Isaac Sim Standalone Pattern

All scripts follow the same structure. `SimulationApp` must be created **before** any other Isaac Sim import:

```python
from isaacsim import SimulationApp
app = SimulationApp({"headless": False})

# Isaac Sim imports are only valid AFTER app creation
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

# ... scene setup, simulation loop ...
app.close()
```

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `isaacsim[all,extscache]` | 5.1.0 | Core simulation runtime, physics, rendering |
