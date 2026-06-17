from __future__ import annotations

import re

from schemas import Observation, Phase, PlannerOutput, SkillName


def _guess_target_object(instruction: str) -> str:
    text = instruction.lower()
    for pattern in (
        r"(?:pick up|grab|grasp|take|拿起|抓取|抓住)\s+(?:the\s+)?([a-z0-9_\-\s]+?)(?:\s+and|\s+to|\s+into|$)",
        r"(?:object|目标|物体)[:：]\s*([a-z0-9_\-\s]+)",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()[:40] or "target_object"
    if "cup" in text:
        return "cup"
    if "bottle" in text:
        return "bottle"
    return "target_object"


def _guess_target_location(instruction: str) -> str | None:
    text = instruction.lower()
    for keyword in ("on", "into", "inside", "to", "放到", "放在"):
        if keyword in text:
            return text.split(keyword, 1)[-1].strip()[:60] or None
    return None


class MockVLMPlanner:
    """Rule-based stand-in for a future VLM planner.

    The output schema matches what a real VLM should produce: phase, selected
    skill, arguments, confidence, and reason. This keeps the robot side testable
    before model/API integration.
    """

    def __init__(self, default_hand: str = "right") -> None:
        self.default_hand = default_hand

    def plan(self, observation: Observation) -> PlannerOutput:
        text = observation.instruction.lower()
        target_object = _guess_target_object(observation.instruction)
        target_location = _guess_target_location(observation.instruction)

        if any(word in text for word in ("done", "finish", "完成")):
            return PlannerOutput(
                target_object=target_object,
                target_location=target_location,
                phase=Phase.DONE,
                next_skill=SkillName.STAND_SAFE,
                skill_args={},
                confidence=0.9,
                reason="Instruction indicates the task is complete.",
            )

        if observation.step_idx == 0:
            return PlannerOutput(
                target_object=target_object,
                target_location=target_location,
                phase=Phase.OBSERVE,
                next_skill=SkillName.TURN_HEAD,
                skill_args={"yaw_delta": 0.15, "pitch_delta": 0.0},
                confidence=0.7,
                reason="First step scans the workspace before arm motion.",
            )

        if any(word in text for word in ("open", "release", "放下", "松开")) or (
            observation.step_idx >= 2 and any(word in text for word in ("place", "put", "放到", "放在"))
        ):
            return PlannerOutput(
                target_object=target_object,
                target_location=target_location,
                phase=Phase.PLACE,
                next_skill=SkillName.OPEN_GRIPPER,
                skill_args={"hand": self.default_hand},
                confidence=0.66,
                reason="Instruction includes a placement/release stage.",
            )

        if any(word in text for word in ("pick", "grab", "grasp", "take", "抓", "拿")):
            return PlannerOutput(
                target_object=target_object,
                target_location=target_location,
                phase=Phase.GRASP,
                next_skill=SkillName.CLOSE_GRIPPER,
                skill_args={"hand": self.default_hand, "force": 0.35},
                confidence=0.68,
                reason="Instruction asks for grasping the target object.",
            )

        return PlannerOutput(
            target_object=target_object,
            target_location=target_location,
            phase=Phase.OBSERVE,
            next_skill=SkillName.TURN_HEAD,
            skill_args={"yaw_delta": 0.1, "pitch_delta": 0.0},
            confidence=0.55,
            reason="Fallback behavior keeps the robot in visual search.",
        )

