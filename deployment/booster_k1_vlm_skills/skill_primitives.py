from __future__ import annotations

from typing import Any, Callable

from schemas import ExecutionResult, SkillCommand, SkillName


class K1SkillPrimitives:
    """Booster K1 skill wrapper.

    This class intentionally defaults to dry-run. The comments next to each
    method name record the corresponding Booster SDK call found in official
    examples, so real execution can be added with a narrow diff later.
    """

    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        self._handlers: dict[SkillName, Callable[[SkillCommand], ExecutionResult]] = {
            SkillName.STAND_SAFE: self.stand_safe,
            SkillName.STOP_MOTION: self.stop_motion,
            SkillName.TURN_HEAD: self.turn_head,
            SkillName.OPEN_GRIPPER: self.open_gripper,
            SkillName.CLOSE_GRIPPER: self.close_gripper,
            SkillName.MOVE_LEFT_ARM_JOINT_DELTA: self.move_arm_joint_delta,
            SkillName.MOVE_RIGHT_ARM_JOINT_DELTA: self.move_arm_joint_delta,
            SkillName.MOVE_HAND_EE_DELTA: self.move_hand_ee_delta,
        }

    def execute(self, command: SkillCommand) -> ExecutionResult:
        if not self.dry_run or not command.dry_run:
            raise RuntimeError("Real K1 execution is intentionally disabled in this first version.")
        return self._handlers[command.name](command)

    def _result(self, command: SkillCommand, sdk_call: str, args: dict[str, Any]) -> ExecutionResult:
        return ExecutionResult(
            command=command,
            ok=True,
            message=f"dry-run: would call {sdk_call}",
            sdk_call=sdk_call,
            sanitized_args=args,
        )

    def stand_safe(self, command: SkillCommand) -> ExecutionResult:
        # Booster SDK: client.ChangeMode(RobotMode.kPrepare) or RobotMode.kDamping.
        return self._result(command, "B1LocoClient.ChangeMode(RobotMode.kPrepare)", command.args)

    def stop_motion(self, command: SkillCommand) -> ExecutionResult:
        # Booster SDK: client.Move(0.0, 0.0, 0.0).
        return self._result(command, "B1LocoClient.Move(0.0, 0.0, 0.0)", command.args)

    def turn_head(self, command: SkillCommand) -> ExecutionResult:
        # Booster SDK: client.RotateHead(pitch, yaw).
        return self._result(command, "B1LocoClient.RotateHead(pitch, yaw)", command.args)

    def open_gripper(self, command: SkillCommand) -> ExecutionResult:
        # Booster SDK: client.ControlGripper(..., GripperControlMode.kPosition, hand).
        # Dexterous hand alternative: client.ControlDexterousHand(finger_params, hand).
        return self._result(command, "B1LocoClient.ControlGripper(open_position, hand)", command.args)

    def close_gripper(self, command: SkillCommand) -> ExecutionResult:
        # Booster SDK examples use DexterousFingerParameter(angle=350, force=400, speed=800).
        return self._result(command, "B1LocoClient.ControlDexterousHand(grasp_fingers, hand)", command.args)

    def move_arm_joint_delta(self, command: SkillCommand) -> ExecutionResult:
        # Low-level route: publish arm joint targets through rt/joint_ctrl or LowCmd.
        return self._result(command, "Low-level joint target publisher", command.args)

    def move_hand_ee_delta(self, command: SkillCommand) -> ExecutionResult:
        # Booster SDK: client.MoveHandEndEffectorV2(Posture(...), duration_ms, hand).
        return self._result(command, "B1LocoClient.MoveHandEndEffectorV2(posture, duration_ms, hand)", command.args)

