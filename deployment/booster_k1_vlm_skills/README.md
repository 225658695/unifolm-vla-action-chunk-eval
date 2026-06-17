# Booster K1 VLM + Skill Primitives Dry Run

This folder is the first deployable bridge between UnifoLM/VLM-style perception
and a Booster K1 robot control stack. It does not execute real robot commands.
It validates the closed-loop software path:

1. camera observation or mock image
2. language instruction
3. planner output: phase, target, skill, arguments
4. safety filter and command clipping
5. dry-run skill primitive execution and JSONL logging

Run:

```bash
cd /root/autodl-tmp/unifolm-vla
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/unifolm-vla
python deployment/booster_k1_vlm_skills/robot_client_dry_run.py \
  --instruction "pick up the red cup and place it on the table" \
  --steps 3
```

The official Booster SDK examples indicate these future real-control mappings:

- `turn_head` -> `B1LocoClient.RotateHead(pitch, yaw)`
- `stand_safe` -> `B1LocoClient.ChangeMode(RobotMode.kPrepare)`
- `stop_motion` -> `B1LocoClient.Move(0.0, 0.0, 0.0)`
- `open_gripper` / `close_gripper` -> `ControlGripper` or `ControlDexterousHand`
- `move_hand_ee_delta` -> `MoveHandEndEffectorV2`
- joint delta control -> low-level joint target publisher / ROS2 `LowCmd`

The current planner is a rule-based mock planner. Replace `MockVLMPlanner.plan`
with a Qwen/UnifoLM model call when a real camera image and prompt schema are
ready.
