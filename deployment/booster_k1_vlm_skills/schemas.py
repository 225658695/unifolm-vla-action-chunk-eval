from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    OBSERVE = "observe"
    APPROACH = "approach"
    GRASP = "grasp"
    LIFT = "lift"
    PLACE = "place"
    DONE = "done"
    FAIL = "fail"


class SkillName(str, Enum):
    STAND_SAFE = "stand_safe"
    STOP_MOTION = "stop_motion"
    TURN_HEAD = "turn_head"
    OPEN_GRIPPER = "open_gripper"
    CLOSE_GRIPPER = "close_gripper"
    MOVE_LEFT_ARM_JOINT_DELTA = "move_left_arm_joint_delta"
    MOVE_RIGHT_ARM_JOINT_DELTA = "move_right_arm_joint_delta"
    MOVE_HAND_EE_DELTA = "move_hand_ee_delta"


@dataclass
class Observation:
    image_path: str | None
    instruction: str
    state: dict[str, Any] = field(default_factory=dict)
    step_idx: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlannerOutput:
    target_object: str
    target_location: str | None
    phase: Phase
    next_skill: SkillName
    skill_args: dict[str, Any]
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["phase"] = self.phase.value
        data["next_skill"] = self.next_skill.value
        return data


@dataclass
class SkillCommand:
    name: SkillName
    args: dict[str, Any]
    source_phase: Phase
    dry_run: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "args": self.args,
            "source_phase": self.source_phase.value,
            "dry_run": self.dry_run,
        }


@dataclass
class ExecutionResult:
    command: SkillCommand
    ok: bool
    message: str
    sdk_call: str | None = None
    sanitized_args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command.to_dict(),
            "ok": self.ok,
            "message": self.message,
            "sdk_call": self.sdk_call,
            "sanitized_args": self.sanitized_args,
        }

