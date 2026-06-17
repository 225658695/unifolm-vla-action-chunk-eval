from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class SemanticSkill:
    name: str
    args: dict[str, Any]
    duration_s: float
    reason: str


def mock_vlm_plan(instruction: str, wall_distance_m: float, stop_distance_m: float) -> list[SemanticSkill]:
    """Mock VLM planner for the K1 wall-distance task.

    Replace this function with a real VLM call later. The downstream interface
    already matches official Booster locomotion semantics.
    """
    travel_distance = max(0.0, wall_distance_m - stop_distance_m)
    vx = 0.18
    duration = travel_distance / vx if vx > 0 else 0.0
    return [
        SemanticSkill(
            name="walk_to_safe_distance",
            args={"vx": vx, "vy": 0.0, "vyaw": 0.0, "target_stop_distance_m": stop_distance_m},
            duration_s=duration,
            reason="VLM task semantics require moving toward the wall and stopping before contact.",
        ),
        SemanticSkill(
            name="stand_still",
            args={"vx": 0.0, "vy": 0.0, "vyaw": 0.0},
            duration_s=2.0,
            reason="Hold position after reaching the safe distance.",
        ),
    ]


class OfficialK1LocomotionBackend:
    """Adapter for Booster's official high-level locomotion API.

    Real execution maps semantic skills to:
      - B1LocoClient.ChangeMode(RobotMode.kWalking)
      - B1LocoClient.Move(vx, vy, vyaw)

    Default dry-run mode only records the calls that would be sent.
    """

    def __init__(self, net: str, dry_run: bool, command_hz: float) -> None:
        self.net = net
        self.dry_run = dry_run
        self.command_hz = command_hz
        self.client = None
        self.RobotMode = None

    def connect(self) -> None:
        if self.dry_run:
            return
        from booster_robotics_sdk_python import B1LocoClient, ChannelFactory, RobotMode

        ChannelFactory.Instance().Init(0, self.net)
        client = B1LocoClient()
        client.Init()
        self.client = client
        self.RobotMode = RobotMode

    def enter_walking(self) -> dict[str, Any]:
        if self.dry_run:
            return {"sdk_call": "B1LocoClient.ChangeMode(RobotMode.kWalking)", "return_code": None}
        assert self.client is not None and self.RobotMode is not None
        ret = self.client.ChangeMode(self.RobotMode.kWalking)
        return {"sdk_call": "B1LocoClient.ChangeMode(RobotMode.kWalking)", "return_code": int(ret)}

    def move_for(self, vx: float, vy: float, vyaw: float, duration_s: float) -> list[dict[str, Any]]:
        records = []
        steps = max(1, int(duration_s * self.command_hz))
        dt = 1.0 / self.command_hz
        for step in range(steps):
            if self.dry_run:
                ret = None
            else:
                assert self.client is not None
                ret = self.client.Move(float(vx), float(vy), float(vyaw))
            records.append(
                {
                    "step": step,
                    "sdk_call": "B1LocoClient.Move(vx, vy, vyaw)",
                    "vx": float(vx),
                    "vy": float(vy),
                    "vyaw": float(vyaw),
                    "return_code": None if ret is None else int(ret),
                }
            )
            if not self.dry_run:
                time.sleep(dt)
        return records

    def stop(self) -> dict[str, Any]:
        if self.dry_run:
            return {"sdk_call": "B1LocoClient.Move(0.0, 0.0, 0.0)", "return_code": None}
        assert self.client is not None
        ret = self.client.Move(0.0, 0.0, 0.0)
        return {"sdk_call": "B1LocoClient.Move(0.0, 0.0, 0.0)", "return_code": int(ret)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "k1_official_locomotion_trace.jsonl"
    summary_path = out_dir / "k1_official_locomotion_summary.json"

    skills = mock_vlm_plan(args.instruction, args.wall_distance_m, args.stop_distance_m)
    backend = OfficialK1LocomotionBackend(net=args.net, dry_run=not args.execute, command_hz=args.command_hz)
    backend.connect()

    records: list[dict[str, Any]] = []
    records.append({"event": "enter_walking", **backend.enter_walking()})

    with open(trace_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        for skill in skills:
            skill_record = {"event": "skill_start", "skill": asdict(skill)}
            f.write(json.dumps(skill_record, ensure_ascii=False) + "\n")
            records.append(skill_record)

            vx = float(skill.args.get("vx", 0.0))
            vy = float(skill.args.get("vy", 0.0))
            vyaw = float(skill.args.get("vyaw", 0.0))
            for move_record in backend.move_for(vx, vy, vyaw, skill.duration_s):
                move_record = {"event": "official_move_command", "skill_name": skill.name, **move_record}
                f.write(json.dumps(move_record, ensure_ascii=False) + "\n")
                records.append(move_record)

        stop_record = {"event": "stop", **backend.stop()}
        f.write(json.dumps(stop_record, ensure_ascii=False) + "\n")
        records.append(stop_record)

    summary = {
        "robot": "Booster K1",
        "interface": "official Booster SDK high-level locomotion",
        "instruction": args.instruction,
        "execute": bool(args.execute),
        "dry_run": not bool(args.execute),
        "wall_distance_m": args.wall_distance_m,
        "target_stop_distance_m": args.stop_distance_m,
        "skills": [asdict(skill) for skill in skills],
        "trace_jsonl": str(trace_path),
        "official_sdk_mapping": {
            "walk_to_safe_distance": "B1LocoClient.ChangeMode(RobotMode.kWalking) + repeated B1LocoClient.Move(vx, vy, vyaw)",
            "stand_still": "B1LocoClient.Move(0.0, 0.0, 0.0)",
        },
        "notes": [
            "This path uses the official high-level K1 locomotion API mapping.",
            "It does not use the previous hand-written MuJoCo fake gait.",
            "Run with --execute only on a real K1 after verifying network, safety area, and emergency stop.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM semantic skills mapped to official Booster K1 locomotion SDK.")
    parser.add_argument("--instruction", default="walk to the wall, stop 0.5m before it, and stand still")
    parser.add_argument("--wall-distance-m", type=float, default=1.05)
    parser.add_argument("--stop-distance-m", type=float, default=0.5)
    parser.add_argument("--command-hz", type=float, default=5.0)
    parser.add_argument("--net", default="127.0.0.1")
    parser.add_argument("--out-dir", default="results/k1_official_locomotion")
    parser.add_argument("--execute", action="store_true", help="Actually send SDK commands to a real K1.")
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

