from __future__ import annotations

from copy import deepcopy
from typing import Any

from schemas import Phase, SkillCommand, SkillName


ALLOWED_SKILLS_BY_PHASE = {
    Phase.OBSERVE: {SkillName.STAND_SAFE, SkillName.STOP_MOTION, SkillName.TURN_HEAD},
    Phase.APPROACH: {
        SkillName.STOP_MOTION,
        SkillName.TURN_HEAD,
        SkillName.MOVE_LEFT_ARM_JOINT_DELTA,
        SkillName.MOVE_RIGHT_ARM_JOINT_DELTA,
        SkillName.MOVE_HAND_EE_DELTA,
    },
    Phase.GRASP: {SkillName.STOP_MOTION, SkillName.OPEN_GRIPPER, SkillName.CLOSE_GRIPPER},
    Phase.LIFT: {
        SkillName.STOP_MOTION,
        SkillName.MOVE_LEFT_ARM_JOINT_DELTA,
        SkillName.MOVE_RIGHT_ARM_JOINT_DELTA,
        SkillName.MOVE_HAND_EE_DELTA,
    },
    Phase.PLACE: {
        SkillName.STOP_MOTION,
        SkillName.OPEN_GRIPPER,
        SkillName.MOVE_LEFT_ARM_JOINT_DELTA,
        SkillName.MOVE_RIGHT_ARM_JOINT_DELTA,
        SkillName.MOVE_HAND_EE_DELTA,
    },
    Phase.DONE: {SkillName.STAND_SAFE, SkillName.STOP_MOTION},
    Phase.FAIL: {SkillName.STAND_SAFE, SkillName.STOP_MOTION},
}


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class SafetyFilter:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def sanitize(self, command: SkillCommand) -> SkillCommand:
        allowed = ALLOWED_SKILLS_BY_PHASE.get(command.source_phase, set())
        if command.name not in allowed:
            raise ValueError(
                f"Skill {command.name.value} is not allowed in phase {command.source_phase.value}."
            )

        if self.cfg.get("require_dry_run", True) and not command.dry_run:
            raise ValueError("Safety config requires dry_run=True.")

        args = deepcopy(command.args)
        if command.name == SkillName.TURN_HEAD:
            args["yaw_delta"] = _clip(float(args.get("yaw_delta", 0.0)), float(self.cfg["max_head_yaw_delta"]))
            args["pitch_delta"] = _clip(
                float(args.get("pitch_delta", 0.0)), float(self.cfg["max_head_pitch_delta"])
            )

        if command.name in {SkillName.MOVE_LEFT_ARM_JOINT_DELTA, SkillName.MOVE_RIGHT_ARM_JOINT_DELTA}:
            limit = float(self.cfg["max_joint_delta_abs"])
            delta = args.get("joint_delta", [0.0] * 7)
            args["joint_delta"] = [_clip(float(v), limit) for v in delta]

        if command.name == SkillName.MOVE_HAND_EE_DELTA:
            limit = float(self.cfg["max_ee_delta_abs"])
            delta = args.get("ee_delta_xyz", [0.0, 0.0, 0.0])
            args["ee_delta_xyz"] = [_clip(float(v), limit) for v in delta]

        if "speed_scale" in args:
            args["speed_scale"] = max(0.0, min(float(args["speed_scale"]), float(self.cfg["max_speed_scale"])))

        return SkillCommand(
            name=command.name,
            args=args,
            source_phase=command.source_phase,
            dry_run=command.dry_run,
        )

