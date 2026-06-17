from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio.v2 as imageio
import mujoco
import numpy as np


DEFAULT_K1_XML = (
    Path("/root/autodl-tmp/k1-standing-long-jump")
    / "third_party"
    / "booster_assets"
    / "robots"
    / "K1"
    / "K1_22dof.xml"
)
DEFAULT_OUT_DIR = Path("results/k1_vlm_skill_mujoco")

JOINT_NAMES = [
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]
J = {name: idx for idx, name in enumerate(JOINT_NAMES)}


@dataclass
class Skill:
    name: str
    phase: str
    duration: float
    reason: str


def add_wall_to_mjcf(src_xml: Path, wall_x: float) -> Path:
    """Create a temporary K1 MJCF with a front wall marker."""
    text = src_xml.read_text(encoding="utf-8")
    wall = f"""
    <body name="vlm_skill_front_wall" pos="{wall_x:.3f} 0 0.55">
      <geom name="front_wall_visual" type="box" size="0.035 0.8 0.55" rgba="0.8 0.1 0.1 1" contype="1" conaffinity="1"/>
    </body>
"""
    if "</worldbody>" not in text:
        raise ValueError(f"Cannot insert wall: missing </worldbody> in {src_xml}")
    text = text.replace("</worldbody>", wall + "\n  </worldbody>", 1)
    tmp = tempfile.NamedTemporaryFile(
        prefix="k1_wall_",
        suffix=".xml",
        delete=False,
        mode="w",
        encoding="utf-8",
        dir=src_xml.parent,
    )
    tmp.write(text)
    tmp.close()
    return Path(tmp.name)


def skill_plan(instruction: str) -> list[Skill]:
    """Mock VLM planner output for the K1 wall task."""
    return [
        Skill("look_forward", "observe", 0.8, "front wall is the navigation target"),
        Skill("walk_to_safe_distance", "locomotion", 2.6, "move forward but stop before the wall"),
        Skill("stand_still", "done", 2.2, "stand still at a safe distance from the wall"),
    ]


def default_pose() -> np.ndarray:
    q = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    q[J["ALeft_Shoulder_Pitch"]] = 0.15
    q[J["ARight_Shoulder_Pitch"]] = 0.15
    q[J["Left_Shoulder_Roll"]] = -1.25
    q[J["Right_Shoulder_Roll"]] = 1.25
    q[J["Left_Hip_Pitch"]] = -0.06
    q[J["Right_Hip_Pitch"]] = -0.06
    q[J["Left_Knee_Pitch"]] = 0.12
    q[J["Right_Knee_Pitch"]] = 0.12
    q[J["Left_Ankle_Pitch"]] = -0.06
    q[J["Right_Ankle_Pitch"]] = -0.06
    return q


def target_for_skill(skill: Skill, local_t: float, root_x: float) -> tuple[np.ndarray, float]:
    q = default_pose()
    next_root_x = root_x

    if skill.name == "look_forward":
        q[J["Head_pitch"]] = -0.08
        q[J["AAHead_yaw"]] = 0.15 * np.sin(2.0 * np.pi * local_t / max(skill.duration, 1e-6))

    elif skill.name == "walk_to_safe_distance":
        cadence = 2.0 * np.pi * 1.6 * local_t
        swing = np.sin(cadence)
        q[J["Left_Hip_Pitch"]] = -0.10 + 0.18 * swing
        q[J["Right_Hip_Pitch"]] = -0.10 - 0.18 * swing
        q[J["Left_Knee_Pitch"]] = 0.18 + 0.12 * max(0.0, -swing)
        q[J["Right_Knee_Pitch"]] = 0.18 + 0.12 * max(0.0, swing)
        q[J["Left_Ankle_Pitch"]] = -0.08 - 0.08 * swing
        q[J["Right_Ankle_Pitch"]] = -0.08 + 0.08 * swing
        next_root_x = min(0.52, root_x + 0.003)

    elif skill.name == "stand_still":
        q = default_pose()

    return q, next_root_x


def pd_control(model: mujoco.MjModel, data: mujoco.MjData, q_des: np.ndarray) -> None:
    q = data.qpos[7:]
    qd = data.qvel[6:]
    kp = np.array([18.0] * 10 + [55.0, 45.0, 35.0, 70.0, 35.0, 25.0] * 2)
    kd = np.array([1.0] * 10 + [2.0, 1.8, 1.2, 2.5, 1.5, 1.0] * 2)
    torque = kp * (q_des - q) - kd * qd + data.qfrc_bias[6:]
    limits = model.actuator_forcerange
    data.ctrl[:] = np.clip(torque, limits[:, 0], limits[:, 1])


def render_frame(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    fixed_lookat: np.ndarray,
) -> np.ndarray:
    camera.lookat[:] = fixed_lookat
    renderer.update_scene(data, camera=camera)
    return renderer.render()


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "k1_vlm_skill_trace.jsonl"
    summary_path = out_dir / "k1_vlm_skill_summary.json"
    video_path = out_dir / "k1_wall_step_in_place.mp4"
    first_obs_path = out_dir / "first_observation.png"

    xml_path = add_wall_to_mjcf(Path(args.xml), args.wall_x)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    root_z = args.root_z
    root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    root_x = 0.0
    data.qpos[0:3] = [root_x, 0.0, root_z]
    data.qpos[3:7] = root_quat
    data.qpos[7:] = default_pose()
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    fixed_lookat = np.asarray(args.camera_lookat, dtype=np.float64)
    writer = imageio.get_writer(video_path, fps=args.render_fps, codec="libx264", quality=8)

    plan = skill_plan(args.instruction)
    control_dt = 1.0 / args.control_fps
    substeps = max(1, int(control_dt / model.opt.timestep))
    render_every = max(1, args.control_fps // args.render_fps)
    global_step = 0
    max_root_x = root_x
    stand_still_steps = 0

    with open(trace_path, "w", encoding="utf-8") as f:
        try:
            first_frame = render_frame(renderer, data, camera, fixed_lookat)
            imageio.imwrite(first_obs_path, first_frame)
            writer.append_data(first_frame)

            for skill in plan:
                n = int(skill.duration * args.control_fps)
                for i in range(n):
                    local_t = i * control_dt
                    q_des, root_x = target_for_skill(skill, local_t, root_x)
                    pd_control(model, data, q_des)
                    for _ in range(substeps):
                        mujoco.mj_step(model, data)
                        # Kinematic root primitive: this is a high-level locomotion
                        # placeholder, not a learned dynamic gait controller.
                        data.qpos[0:3] = [root_x, 0.0, root_z]
                        data.qpos[3:7] = root_quat
                        data.qvel[0:6] = 0.0
                        mujoco.mj_forward(model, data)

                    max_root_x = max(max_root_x, float(data.qpos[0]))
                    if skill.name == "stand_still":
                        stand_still_steps += 1

                    if global_step % render_every == 0:
                        writer.append_data(render_frame(renderer, data, camera, fixed_lookat))

                    record = {
                        "step": global_step,
                        "instruction": args.instruction,
                        "phase": skill.phase,
                        "skill": skill.name,
                        "reason": skill.reason,
                        "root_x": float(data.qpos[0]),
                        "distance_to_wall": float(args.wall_x - data.qpos[0]),
                        "target_joint_pos": q_des.round(5).tolist(),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    global_step += 1
        finally:
            writer.close()
            try:
                xml_path.unlink(missing_ok=True)
            except Exception:
                pass

    final_distance_to_wall = float(args.wall_x - root_x)
    success = bool(
        args.min_wall_distance <= final_distance_to_wall <= args.max_wall_distance
        and stand_still_steps >= int(1.0 * args.control_fps)
    )
    summary = {
        "robot": "Booster K1",
        "simulator": "MuJoCo",
        "instruction": args.instruction,
        "success": success,
        "success_definition": (
            f"{args.min_wall_distance} <= final wall distance <= {args.max_wall_distance} "
            "and at least 1s of stand_still skill executed"
        ),
        "max_root_x": max_root_x,
        "final_root_x": float(root_x),
        "final_distance_to_wall": final_distance_to_wall,
        "wall_x": args.wall_x,
        "video_path": str(video_path),
        "first_observation": str(first_obs_path),
        "trace_jsonl": str(trace_path),
        "planner": "mock VLM planner producing K1 skill primitives",
        "skills": [skill.__dict__ for skill in plan],
        "notes": [
            "This controls the Booster K1 MuJoCo model, not LIBERO/Franka.",
            "The forward locomotion is a conservative kinematic-root primitive plus joint PD stepping.",
            "Replace the mock planner with a real VLM and the kinematic primitive with a learned K1 locomotion policy for a stronger result.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K1 VLM + skill primitive MuJoCo demo.")
    parser.add_argument("--xml", default=str(DEFAULT_K1_XML))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--instruction", default="walk to the wall, stop 0.5m before it, and stand still")
    parser.add_argument("--wall-x", type=float, default=1.05)
    parser.add_argument("--min-wall-distance", type=float, default=0.45)
    parser.add_argument("--max-wall-distance", type=float, default=0.60)
    parser.add_argument("--root-z", type=float, default=0.62)
    parser.add_argument("--control-fps", type=int, default=100)
    parser.add_argument("--render-fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    parser.add_argument("--camera-elevation", type=float, default=-15.0)
    parser.add_argument("--camera-lookat", type=float, nargs=3, default=[0.55, 0.0, 0.45])
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
