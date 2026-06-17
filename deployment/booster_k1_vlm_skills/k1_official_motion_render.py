from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "osmesa")

BOOSTER_DEPLOY = Path("/root/autodl-tmp/booster_repos/booster_deploy")
BOOSTER_ASSETS_SRC = Path("/root/autodl-tmp/k1-standing-long-jump/third_party/booster_assets/src")
for path in (BOOSTER_DEPLOY, BOOSTER_ASSETS_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from booster_deploy.controllers.mujoco_controller import MujocoController
from tasks.beyond_mimic import K1FightControllerCfg, K1MJ2ControllerCfg


TASKS = {
    "k1_fight": K1FightControllerCfg,
    "k1_mj2": K1MJ2ControllerCfg,
}

TASK_DIR = BOOSTER_DEPLOY / "tasks" / "beyond_mimic"


def _camera(args: argparse.Namespace) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.lookat[:] = args.camera_lookat
    camera.distance = args.camera_distance
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    return camera


def render_official_motion(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{args.task}.mp4"
    summary_path = out_dir / f"{args.task}_summary.json"

    cfg = TASKS[args.task]()
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        cfg.policy.checkpoint_path = str(
            checkpoint if checkpoint.is_absolute() else checkpoint
        )
    if args.motion:
        motion = Path(args.motion)
        cfg.policy.motion_path = str(motion if motion.is_absolute() else motion)
    cfg.policy.device = args.device
    cfg.mujoco.visualize_reference_ghost = False

    controller = MujocoController(cfg)
    controller.update_state()
    controller.start()

    renderer = mujoco.Renderer(controller.mj_model, height=args.height, width=args.width)
    camera = _camera(args)
    writer = imageio.get_writer(video_path, fps=args.render_fps, codec="libx264", quality=8)

    sim_steps = int(args.duration * args.policy_fps)
    render_every = max(1, args.policy_fps // args.render_fps)
    root_positions = []

    try:
        for step in range(sim_steps):
            controller.update_state()
            dof_targets = controller.policy_step()
            controller.ctrl_step(dof_targets)
            root_positions.append(controller.mj_data.qpos[:3].copy())

            if step % render_every == 0:
                if args.follow:
                    camera.lookat[:] = [
                        float(controller.mj_data.qpos[0]),
                        float(controller.mj_data.qpos[1]),
                        0.55,
                    ]
                renderer.update_scene(controller.mj_data, camera=camera)
                writer.append_data(renderer.render())

            if not controller.is_running:
                break
    finally:
        writer.close()

    root_positions_np = np.asarray(root_positions) if root_positions else np.zeros((0, 3))
    summary = {
        "robot": "Booster K1",
        "task": args.task,
        "uses_official_policy": True,
        "checkpoint_path": str((TASK_DIR / cfg.policy.checkpoint_path).resolve() if not Path(cfg.policy.checkpoint_path).is_absolute() else Path(cfg.policy.checkpoint_path)),
        "motion_path": str((TASK_DIR / cfg.policy.motion_path).resolve() if not Path(cfg.policy.motion_path).is_absolute() else Path(cfg.policy.motion_path)),
        "duration_requested_s": args.duration,
        "steps_rendered": int(len(root_positions)),
        "video_path": str(video_path),
        "root_position_start": root_positions_np[0].round(5).tolist() if len(root_positions_np) else None,
        "root_position_end": root_positions_np[-1].round(5).tolist() if len(root_positions_np) else None,
        "notes": [
            "This uses Booster's official K1 beyond_mimic policy checkpoint and motion file.",
            "It is a fixed motion policy, not a VLM-parameterized hit controller.",
            "Use it to compare natural official K1 motion against hand-written skill primitives.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless render for official Booster K1 motion policies.")
    parser.add_argument("--task", choices=sorted(TASKS), default="k1_fight")
    parser.add_argument("--checkpoint", default="", help="Override policy checkpoint path. Relative paths are resolved from booster_deploy/tasks/beyond_mimic.")
    parser.add_argument("--motion", default="", help="Override motion npz path. Relative paths are resolved from booster_deploy/tasks/beyond_mimic.")
    parser.add_argument("--out-dir", default="results/k1_official_motion")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--policy-fps", type=int, default=50)
    parser.add_argument("--render-fps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=2.8)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-15.0)
    parser.add_argument("--camera-lookat", type=float, nargs=3, default=[0.0, 0.0, 0.55])
    parser.add_argument("--follow", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = render_official_motion(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
