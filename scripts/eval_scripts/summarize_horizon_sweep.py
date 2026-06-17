#!/usr/bin/env python
import csv
import json
import pathlib
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_horizon_sweep.py <run_root>")

    run_root = pathlib.Path(sys.argv[1])
    summary_paths = sorted(run_root.glob("horizon_*/summary_h*.json"))
    if not summary_paths:
        raise SystemExit(f"No summary_h*.json files found under {run_root}")

    rows = []
    for path in summary_paths:
        with open(path) as f:
            summary = json.load(f)
        rows.append(
            {
                "execute_horizon": summary["execute_horizon"],
                "total_episodes": summary["total_episodes"],
                "total_successes": summary["total_successes"],
                "success_rate": summary["success_rate"],
                "avg_env_steps": summary["avg_env_steps"],
                "avg_policy_calls": summary["avg_policy_calls"],
                "avg_action_smoothness": summary["avg_action_smoothness"],
                "avg_position_smoothness": summary["avg_position_smoothness"],
                "avg_rotation_smoothness": summary["avg_rotation_smoothness"],
                "avg_gripper_flips": summary["avg_gripper_flips"],
                "summary_path": str(path),
            }
        )

    rows.sort(key=lambda row: row["execute_horizon"])
    csv_path = run_root / "horizon_sweep_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path}")
    print("execute_horizon,total_episodes,total_successes,success_rate,avg_env_steps,avg_policy_calls,avg_action_smoothness")
    for row in rows:
        print(
            f"{row['execute_horizon']},{row['total_episodes']},{row['total_successes']},"
            f"{row['success_rate']:.4f},{row['avg_env_steps']:.2f},"
            f"{row['avg_policy_calls']:.2f},{row['avg_action_smoothness']:.6f}"
        )


if __name__ == "__main__":
    main()
