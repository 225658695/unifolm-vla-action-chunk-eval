from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from mock_observation import build_mock_image, make_observation
from safety import SafetyFilter
from skill_primitives import K1SkillPrimitives
from state_machine import SkillStateMachine
from vlm_planner import MockVLMPlanner


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).with_name("config.yaml")
    parser = argparse.ArgumentParser(description="Dry-run Booster K1 VLM + skill primitive loop.")
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--image", default=None, help="Optional real camera image path for planner input.")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--mock-image-out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    runtime_cfg = cfg["runtime"]
    dry_run = bool(runtime_cfg.get("dry_run", True))
    if not dry_run:
        raise RuntimeError("This entrypoint only supports dry-run.")

    steps = args.steps or int(runtime_cfg.get("max_steps", 3))
    log_dir = Path(args.log_dir or runtime_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    image_path = args.image
    if image_path is None:
        mock_image_out = args.mock_image_out or log_dir / "mock_k1_observation.jpg"
        image_path = build_mock_image(mock_image_out, args.instruction)

    planner = MockVLMPlanner(default_hand=cfg["planner"].get("default_hand", "right"))
    state_machine = SkillStateMachine(dry_run=dry_run)
    safety = SafetyFilter(cfg["safety"])
    skills = K1SkillPrimitives(dry_run=dry_run)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"k1_vlm_skill_dry_run_{run_id}.jsonl"

    print(f"[dry-run] instruction: {args.instruction}")
    print(f"[dry-run] image: {image_path}")
    print(f"[dry-run] log: {log_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        for step_idx in range(steps):
            observation = make_observation(args.instruction, image_path, step_idx)
            plan = planner.plan(observation)
            command = state_machine.build_command(plan)
            safe_command = safety.sanitize(command)
            result = skills.execute(safe_command)

            record = {
                "step_idx": step_idx,
                "observation": observation.to_dict(),
                "planner_output": plan.to_dict(),
                "safe_command": safe_command.to_dict(),
                "execution_result": result.to_dict(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(
                f"[step {step_idx}] phase={plan.phase.value} "
                f"skill={safe_command.name.value} args={safe_command.args} "
                f"-> {result.message}"
            )

    print("[dry-run] completed without sending commands to a real robot")


if __name__ == "__main__":
    main()

