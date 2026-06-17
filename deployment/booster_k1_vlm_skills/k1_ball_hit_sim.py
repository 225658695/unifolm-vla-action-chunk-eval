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
DEFAULT_OUT_DIR = Path("results/k1_ball_hit_mujoco")

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
    hand: str
    duration: float
    reason: str


def add_ball_to_mjcf(src_xml: Path, ball_pos: np.ndarray, ball_radius: float) -> Path:
    text = src_xml.read_text(encoding="utf-8")
    # Zero gravity keeps the target ball at chest height so this demo isolates
    # VLM-conditioned arm skill selection rather than locomotion/squatting.
    if "<option" in text and "gravity=" not in text.split("<option", 1)[1].split(">", 1)[0]:
        text = text.replace("<option", '<option gravity="0 0 0"', 1)
    ball = f"""
    <body name="vlm_target_ball" pos="{ball_pos[0]:.4f} {ball_pos[1]:.4f} {ball_pos[2]:.4f}">
      <joint name="target_ball_free" type="free" damping="0.001"/>
      <geom name="target_ball_geom" type="sphere" size="{ball_radius:.4f}" mass="0.08" rgba="0.05 0.35 1.0 1" contype="1" conaffinity="1" friction="0.1 0.005 0.0001"/>
    </body>
"""
    if "</worldbody>" not in text:
        raise ValueError(f"Cannot insert ball: missing </worldbody> in {src_xml}")
    text = text.replace("</worldbody>", ball + "\n  </worldbody>", 1)
    tmp = tempfile.NamedTemporaryFile(
        prefix="k1_ball_",
        suffix=".xml",
        delete=False,
        mode="w",
        encoding="utf-8",
        dir=src_xml.parent,
    )
    tmp.write(text)
    tmp.close()
    return Path(tmp.name)


def default_pose() -> np.ndarray:
    q = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    q[J["Head_pitch"]] = -0.05
    q[J["ALeft_Shoulder_Pitch"]] = 0.10
    q[J["ARight_Shoulder_Pitch"]] = 0.10
    q[J["Left_Shoulder_Roll"]] = -1.20
    q[J["Right_Shoulder_Roll"]] = 1.20
    q[J["Left_Elbow_Pitch"]] = 0.20
    q[J["Right_Elbow_Pitch"]] = 0.20
    q[J["Left_Hip_Pitch"]] = -0.06
    q[J["Right_Hip_Pitch"]] = -0.06
    q[J["Left_Knee_Pitch"]] = 0.12
    q[J["Right_Knee_Pitch"]] = 0.12
    q[J["Left_Ankle_Pitch"]] = -0.06
    q[J["Right_Ankle_Pitch"]] = -0.06
    return q


def mock_vlm_ball_position(ball_pos: np.ndarray) -> dict[str, Any]:
    y = float(ball_pos[1])
    if y > 0.08:
        region = "left"
        hand = "left"
    elif y < -0.08:
        region = "right"
        hand = "right"
    else:
        region = "center"
        hand = "right"
    return {
        "target_object": "blue ball",
        "ball_region": region,
        "selected_hand": hand,
        "ball_position_sim": ball_pos.round(4).tolist(),
        "reason": f"Ball is in the {region} region, select {hand} hand to swing toward it.",
    }


def skill_plan(vlm_output: dict[str, Any]) -> list[Skill]:
    hand = str(vlm_output["selected_hand"])
    return [
        Skill("look_at_ball", "observe", hand, 0.6, "orient head toward the detected ball"),
        Skill(f"retract_{hand}_arm", "prepare", hand, 0.7, "move arm backward before swing"),
        Skill(f"hit_ball_{hand}", "hit", hand, 0.55, "swing selected arm toward the ball"),
        Skill("hold_after_hit", "done", hand, 1.0, "hold pose after contact"),
    ]


def arm_pose(hand: str, stage: str, ball_y: float) -> np.ndarray:
    q = default_pose()
    if hand == "right":
        if stage == "retract":
            q[J["ARight_Shoulder_Pitch"]] = 0.25
            q[J["Right_Shoulder_Roll"]] = 1.35
            q[J["Right_Elbow_Pitch"]] = 1.10
            q[J["Right_Elbow_Yaw"]] = 0.65
        elif stage == "hit":
            q[J["ARight_Shoulder_Pitch"]] = -1.35
            q[J["Right_Shoulder_Roll"]] = 0.85 if ball_y < -0.08 else 0.45
            q[J["Right_Elbow_Pitch"]] = -0.20
            q[J["Right_Elbow_Yaw"]] = 0.95
    else:
        if stage == "retract":
            q[J["ALeft_Shoulder_Pitch"]] = 0.25
            q[J["Left_Shoulder_Roll"]] = -1.35
            q[J["Left_Elbow_Pitch"]] = 1.10
            q[J["Left_Elbow_Yaw"]] = -0.65
        elif stage == "hit":
            q[J["ALeft_Shoulder_Pitch"]] = -1.35
            q[J["Left_Shoulder_Roll"]] = -0.85 if ball_y > 0.08 else -0.45
            q[J["Left_Elbow_Pitch"]] = -0.20
            q[J["Left_Elbow_Yaw"]] = -0.95
    return q


def target_for_skill(skill: Skill, local_t: float, ball_y: float) -> np.ndarray:
    if skill.name == "look_at_ball":
        q = default_pose()
        q[J["AAHead_yaw"]] = float(np.clip(ball_y * 0.8, -0.25, 0.25))
        q[J["Head_pitch"]] = -0.08
        return q
    if "retract" in skill.name:
        return arm_pose(skill.hand, "retract", ball_y)
    if "hit_ball" in skill.name:
        alpha = min(1.0, max(0.0, local_t / max(skill.duration, 1e-6)))
        # Ease into the punch target quickly and keep the arm extended.
        alpha = 1.0 - (1.0 - alpha) ** 3
        return (1.0 - alpha) * arm_pose(skill.hand, "retract", ball_y) + alpha * arm_pose(skill.hand, "hit", ball_y)
    return arm_pose(skill.hand, "hit", ball_y)


def pd_control(model: mujoco.MjModel, data: mujoco.MjData, q_des: np.ndarray) -> None:
    q = data.qpos[7:29]
    qd = data.qvel[6:28]
    kp = np.array([28.0] * 10 + [55.0, 45.0, 35.0, 70.0, 35.0, 25.0] * 2)
    kd = np.array([1.6] * 10 + [2.0, 1.8, 1.2, 2.5, 1.5, 1.0] * 2)
    torque = kp * (q_des - q) - kd * qd + data.qfrc_bias[6:28]
    limits = model.actuator_forcerange
    data.ctrl[:] = np.clip(torque, limits[:, 0], limits[:, 1])


def render_frame(renderer: mujoco.Renderer, data: mujoco.MjData, camera: mujoco.MjvCamera) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    return renderer.render()


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "k1_ball_hit.mp4"
    first_obs_path = out_dir / "first_observation.png"
    trace_path = out_dir / "k1_ball_hit_trace.jsonl"
    summary_path = out_dir / "k1_ball_hit_summary.json"

    ball_pos = np.asarray([args.ball_x, args.ball_y, args.ball_z], dtype=np.float64)
    xml_path = add_ball_to_mjcf(Path(args.xml), ball_pos, args.ball_radius)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    root_pos = np.asarray([0.0, 0.0, args.root_z], dtype=np.float64)
    root_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat
    data.qpos[7:29] = default_pose()
    mujoco.mj_forward(model, data)

    ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "vlm_target_ball")
    initial_ball_pos = data.xpos[ball_body_id].copy()
    vlm_output = mock_vlm_ball_position(initial_ball_pos)
    plan = skill_plan(vlm_output)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.18, 0.0, 0.70]
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    writer = imageio.get_writer(video_path, fps=args.render_fps, codec="libx264", quality=8)

    control_dt = 1.0 / args.control_fps
    substeps = max(1, int(control_dt / model.opt.timestep))
    render_every = max(1, args.control_fps // args.render_fps)
    global_step = 0

    with open(trace_path, "w", encoding="utf-8") as f:
        try:
            frame = render_frame(renderer, data, camera)
            imageio.imwrite(first_obs_path, frame)
            writer.append_data(frame)
            f.write(json.dumps({"event": "vlm_output", **vlm_output}, ensure_ascii=False) + "\n")

            for skill in plan:
                n = int(skill.duration * args.control_fps)
                for i in range(n):
                    local_t = i * control_dt
                    q_des = target_for_skill(skill, local_t, float(initial_ball_pos[1]))
                    pd_control(model, data, q_des)
                    for _ in range(substeps):
                        mujoco.mj_step(model, data)
                        # Pinned-base setup: isolate upper-body hit skill.
                        data.qpos[0:3] = root_pos
                        data.qpos[3:7] = root_quat
                        data.qvel[0:6] = 0.0
                        mujoco.mj_forward(model, data)

                    ball_now = data.xpos[ball_body_id].copy()
                    if global_step % render_every == 0:
                        writer.append_data(render_frame(renderer, data, camera))
                    record = {
                        "step": global_step,
                        "phase": skill.phase,
                        "skill": skill.name,
                        "selected_hand": skill.hand,
                        "ball_position": ball_now.round(5).tolist(),
                        "ball_displacement": (ball_now - initial_ball_pos).round(5).tolist(),
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

    final_ball_pos = data.xpos[ball_body_id].copy()
    ball_disp = final_ball_pos - initial_ball_pos
    ball_disp_norm = float(np.linalg.norm(ball_disp))
    success = bool(ball_disp_norm >= args.success_disp)
    summary = {
        "robot": "Booster K1",
        "simulator": "MuJoCo",
        "instruction": args.instruction,
        "success": success,
        "success_definition": f"ball displacement norm >= {args.success_disp} m",
        "vlm_output": vlm_output,
        "initial_ball_position": initial_ball_pos.round(5).tolist(),
        "final_ball_position": final_ball_pos.round(5).tolist(),
        "ball_displacement": ball_disp.round(5).tolist(),
        "ball_displacement_norm": ball_disp_norm,
        "video_path": str(video_path),
        "first_observation": str(first_obs_path),
        "trace_jsonl": str(trace_path),
        "skills": [skill.__dict__ for skill in plan],
        "notes": [
            "This controls the Booster K1 MuJoCo model, not LIBERO/Franka.",
            "The ball position is read from simulation state as a mock VLM perception output.",
            "The selected hit skill drives K1 upper-body joints with PD control while the base is pinned.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K1 VLM-conditioned ball hitting skill demo.")
    parser.add_argument("--xml", default=str(DEFAULT_K1_XML))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--instruction", default="hit the blue ball in front of the robot")
    parser.add_argument("--ball-x", type=float, default=0.18)
    parser.add_argument("--ball-y", type=float, default=-0.23)
    parser.add_argument("--ball-z", type=float, default=0.78)
    parser.add_argument("--ball-radius", type=float, default=0.055)
    parser.add_argument("--root-z", type=float, default=0.62)
    parser.add_argument("--success-disp", type=float, default=0.025)
    parser.add_argument("--control-fps", type=int, default=100)
    parser.add_argument("--render-fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=2.0)
    parser.add_argument("--camera-azimuth", type=float, default=120.0)
    parser.add_argument("--camera-elevation", type=float, default=-12.0)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

