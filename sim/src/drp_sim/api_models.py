"""Pydantic request/response schemas for the sim server API."""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field, computed_field


class SimHealthResponse(BaseModel):
    status: str = "ok"
    sim_time: float = 0.0
    is_playing: bool = False


class BoxPosition(BaseModel):
    x: float
    y: float
    z: float
    prim_path: str


class SimStateResponse(BaseModel):
    joint_positions: list[float] = Field(default_factory=list)
    box_positions: list[BoxPosition] = Field(default_factory=list)
    sim_time: float = 0.0


class MoveRequest(BaseModel):
    joint_positions: list[float] = Field(..., min_length=5, max_length=5)


class StepRequest(BaseModel):
    num_steps: int = Field(default=1, ge=1, le=1000)
    render: bool = True


class SpawnBoxResponse(BaseModel):
    prim_path: str
    box_count: int


class FillBufferResponse(BaseModel):
    status: str
    occupied: int
    capacity: int


class CameraResponse(BaseModel):
    width: int = 0
    height: int = 0
    data: str = ""
    encoding: str = "none"


class MovePlannedRequest(BaseModel):
    target: list[float] = Field(
        ..., min_length=5, max_length=5, description="Goal joint positions in radians (5 joints)"
    )
    execute: bool = Field(default=True, description="Execute trajectory in sim after planning")


class MoveCartesianRequest(BaseModel):
    position: list[float] = Field(
        ..., min_length=3, max_length=3, description="[x, y, z] end-effector position in metres"
    )
    quaternion: list[float] = Field(
        ..., min_length=4, max_length=4, description="[w, x, y, z] unit quaternion"
    )
    execute: bool = Field(default=True, description="Execute trajectory in sim after planning")


class MoveMotionResponse(BaseModel):
    trajectory: list[list[float]]

    @computed_field
    @property
    def waypoints(self) -> int:
        return len(self.trajectory)


_DEFAULT_QUAT: list[float] = [0.0, 1.0, 0.0, 0.0]  # 180 deg around X - cup faces down


class PickPlaceRequest(BaseModel):
    """Pick-and-place request with explicit positions and orientation.

    The caller provides both pick and drop positions.  ``drop_quaternion``
    is a *logical* rotation (identity or 90-deg Z for even-layer interlocking)
    that the sim composes with the gripper-down base orientation.
    """

    box_prim: str = Field(..., description="USD prim path of the box to pick")
    pick_position: list[float] = Field(
        ..., min_length=3, max_length=3, description="[x, y, z] pick position in metres"
    )
    drop_position: list[float] = Field(
        ..., min_length=3, max_length=3, description="[x, y, z] drop position in metres"
    )
    drop_quaternion: list[float] = Field(
        default=[1.0, 0.0, 0.0, 0.0],
        min_length=4,
        max_length=4,
        description=(
            "Logical [w,x,y,z] rotation applied to the box at the drop site. "
            "Identity means no extra rotation; [0.707,0,0,0.707] means 90-deg Z."
        ),
    )
    speed: float = Field(default=1.0, gt=0.0, le=1.0, description="Speed fraction (not yet used)")


class PickPlaceResponse(BaseModel):
    status: str
    steps: list[str] = Field(default_factory=list)
    success: bool = False


class ClearZoneResponse(BaseModel):
    removed: int


class PalletSlot(int, enum.Enum):
    """Pallet slot number (1-3). Values outside this range are rejected."""

    SLOT_1 = 1
    SLOT_2 = 2
    SLOT_3 = 3


class AutoPickRequest(BaseModel):
    slot: PalletSlot = Field(..., description="Pallet slot number (1, 2, or 3)")


class AutoPickResponse(BaseModel):
    status: str
    slot: int = -1
    steps: int = 0
    success: bool = False
    message: str = ""


class HumanCallRequest(BaseModel):
    index: int = Field(default=0, ge=0, description="Buffer slot index")


class HumanCallResponse(BaseModel):
    status: str
    slot: int
    prim_path: str = ""


class BoxImageEntry(BaseModel):
    box_id: str
    image_b64: str
    size: list[float] = Field(default_factory=list)
    weight: float = 0.0
    type: str = "normal"
    visual: str = ""


class BoxImagesResponse(BaseModel):
    images: list[BoxImageEntry] = Field(default_factory=list)


class RemoveBoxRequest(BaseModel):
    box_id: str = Field(..., description="Logical box ID (e.g. box_0001)")


class RemoveBoxResponse(BaseModel):
    status: str
    box_id: str


class PickPosition(BaseModel):
    x: float
    y: float
    z: float


class PickPositionsResponse(BaseModel):
    positions: list[PickPosition] = Field(default_factory=list)


class PalletCenter(BaseModel):
    x: float
    y: float
    z: float


class PalletCentersResponse(BaseModel):
    centers: list[PalletCenter] = Field(default_factory=list)


class BufferStatusResponse(BaseModel):
    occupied: int = 0
    capacity: int = 0
    slots: list[int] = Field(default_factory=list)
    in_transit: bool = False
