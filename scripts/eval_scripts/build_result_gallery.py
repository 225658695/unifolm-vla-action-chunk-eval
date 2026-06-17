#!/usr/bin/env python3
"""Build a lightweight GitHub-ready result gallery from local evaluation outputs."""

from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "results"
ASSETS = OUT / "assets"


HORIZON_CSV = (
    RESULTS
    / "libero_spatial"
    / "UnifoLM-VLA-Libero"
    / "checkpoints"
    / "horizon_sweep"
    / "horizon_sweep_summary.csv"
)
PERTURB_CSV = (
    RESULTS
    / "libero_spatial"
    / "UnifoLM-VLA-Libero"
    / "checkpoints"
    / "perturbation_sweep"
    / "perturbation_sweep_summary.csv"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def fmt(x: object, digits: int = 2) -> str:
    if isinstance(x, str):
        return x
    return f"{float(x):.{digits}f}"


def md_table(rows: list[dict[str, object]], fields: list[tuple[str, str]]) -> str:
    headers = [title for _, title in fields]
    keys = [key for key, _ in fields]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return "\n".join(lines)


def plot_horizon(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    parsed = []
    for row in rows:
        parsed.append(
            {
                "execute_horizon": int(row["execute_horizon"]),
                "total_episodes": int(row["total_episodes"]),
                "success_rate": float(row["success_rate"]),
                "avg_env_steps": float(row["avg_env_steps"]),
                "avg_policy_calls": float(row["avg_policy_calls"]),
                "avg_action_smoothness": float(row["avg_action_smoothness"]),
                "avg_gripper_flips": float(row["avg_gripper_flips"]),
            }
        )
    parsed.sort(key=lambda x: x["execute_horizon"])

    x = [r["execute_horizon"] for r in parsed]
    success = [100.0 * r["success_rate"] for r in parsed]
    calls = [r["avg_policy_calls"] for r in parsed]
    smooth = [r["avg_action_smoothness"] for r in parsed]

    fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=180)
    ax1.plot(x, success, marker="o", linewidth=2.2, color="#2563eb", label="Success rate")
    ax1.set_xlabel("Executed steps per predicted action chunk")
    ax1.set_ylabel("Success rate (%)", color="#2563eb")
    ax1.set_xticks(x)
    ax1.set_ylim(88, 102)
    ax1.grid(axis="y", alpha=0.25)
    ax1.tick_params(axis="y", labelcolor="#2563eb")

    ax2 = ax1.twinx()
    ax2.plot(x, calls, marker="s", linewidth=2.2, color="#dc2626", label="Policy calls")
    ax2.set_ylabel("Avg. policy calls per episode", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right", frameon=False)
    ax1.set_title("Receding-Horizon Chunk Execution")
    fig.tight_layout()
    fig.savefig(ASSETS / "horizon_tradeoff.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=180)
    ax.bar([str(v) for v in x], smooth, color="#0f766e")
    ax.set_xlabel("Executed steps per predicted action chunk")
    ax.set_ylabel("Mean action delta norm")
    ax.set_title("Action Smoothness Across Execution Horizons")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(ASSETS / "horizon_smoothness.png")
    plt.close(fig)

    table_rows = []
    for row in parsed:
        table_rows.append(
            {
                "execute_horizon": row["execute_horizon"],
                "success_rate": pct(row["success_rate"]),
                "avg_env_steps": fmt(row["avg_env_steps"], 1),
                "avg_policy_calls": fmt(row["avg_policy_calls"], 1),
                "avg_action_smoothness": fmt(row["avg_action_smoothness"], 3),
                "avg_gripper_flips": fmt(row["avg_gripper_flips"], 2),
            }
        )
    return table_rows


def plot_perturbation(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    parsed = []
    labels = {
        "none": "Clean",
        "brightness_down_both": "Brightness 0.5x",
        "brightness_up_both": "Brightness 1.5x",
        "full_occlusion": "Full-view occlusion",
        "wrist_occlusion": "Wrist occlusion",
        "full_missing": "Full-view missing",
        "wrist_missing": "Wrist missing",
    }
    order = [
        "none",
        "brightness_down_both",
        "brightness_up_both",
        "full_occlusion",
        "wrist_occlusion",
        "full_missing",
        "wrist_missing",
    ]
    by_name = {r["case_name"]: r for r in rows}
    for case in order:
        if case not in by_name:
            continue
        row = by_name[case]
        failed = [x for x in row.get("failed_task_ids", "").split(";") if x != ""]
        parsed.append(
            {
                "case_name": case,
                "label": labels.get(case, case),
                "success_rate": float(row["success_rate"]),
                "total_episodes": int(row["total_episodes"]),
                "failure_count": int(row["failure_count"]),
                "avg_env_steps": float(row["avg_env_steps"]),
                "avg_policy_calls": float(row["avg_policy_calls"]),
                "avg_action_smoothness": float(row["avg_action_smoothness"]),
                "avg_gripper_flips": float(row["avg_gripper_flips"]),
                "failed_task_ids": failed,
            }
        )

    colors = ["#2563eb", "#16a34a", "#16a34a", "#f59e0b", "#f59e0b", "#dc2626", "#dc2626"]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=180)
    y = np.arange(len(parsed))
    ax.barh(y, [100.0 * r["success_rate"] for r in parsed], color=colors[: len(parsed)])
    ax.set_yticks(y, [r["label"] for r in parsed])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Success rate (%)")
    ax.set_title("Observation Perturbation Robustness")
    ax.grid(axis="x", alpha=0.22)
    for i, row in enumerate(parsed):
        ax.text(100.0 * row["success_rate"] + 1.0, i, pct(row["success_rate"]), va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "perturbation_success.png")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9, 4.8), dpi=180)
    x = np.arange(len(parsed))
    ax1.plot(x, [r["avg_policy_calls"] for r in parsed], marker="o", color="#7c3aed", linewidth=2.2)
    ax1.set_xticks(x, [r["label"] for r in parsed], rotation=25, ha="right")
    ax1.set_ylabel("Avg. policy calls")
    ax1.grid(axis="y", alpha=0.22)
    ax2 = ax1.twinx()
    ax2.plot(x, [r["avg_gripper_flips"] for r in parsed], marker="s", color="#ea580c", linewidth=2.2)
    ax2.set_ylabel("Avg. gripper flips")
    ax1.set_title("Recovery Effort Under Perturbations")
    fig.tight_layout()
    fig.savefig(ASSETS / "perturbation_recovery_effort.png")
    plt.close(fig)

    table_rows = []
    for row in parsed:
        failed_counter = Counter(row["failed_task_ids"])
        top_failures = ", ".join(f"{task}x{count}" for task, count in failed_counter.most_common(3))
        table_rows.append(
            {
                "case": row["label"],
                "success_rate": pct(row["success_rate"]),
                "failures": row["failure_count"],
                "avg_steps": fmt(row["avg_env_steps"], 1),
                "policy_calls": fmt(row["avg_policy_calls"], 1),
                "smoothness": fmt(row["avg_action_smoothness"], 3),
                "gripper_flips": fmt(row["avg_gripper_flips"], 2),
                "top_failed_tasks": top_failures,
            }
        )
    return table_rows


def copy_assets() -> None:
    copies = [
        (
            RESULTS
            / "libero_spatial"
            / "UnifoLM-VLA-Libero"
            / "checkpoints"
            / "rollout_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_episode0_success.mp4",
            ASSETS / "libero_spatial_success_example.mp4",
        ),
    ]

    for src, dst in copies:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    if HORIZON_CSV.exists():
        shutil.copy2(HORIZON_CSV, OUT / "horizon_sweep_summary.csv")
    if PERTURB_CSV.exists():
        shutil.copy2(PERTURB_CSV, OUT / "perturbation_sweep_summary.csv")


def write_readme(horizon_rows: list[dict[str, object]], perturb_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Result Gallery",
        "",
        "This folder contains a lightweight, GitHub-ready summary of the local evaluation outputs.",
        "Large checkpoints and raw simulation environments are intentionally excluded.",
        "Unified conclusion: action chunking improves execution efficiency, while larger executed horizons weaken closed-loop correction; visual perturbations, especially missing-view conditions, amplify that weakness.",
        "",
        "## LIBERO Receding-Horizon Evaluation",
        "",
        "The policy predicts a multi-step action chunk. During evaluation, only the first `k` steps are executed before re-observing and re-planning.",
        "",
        "![Horizon tradeoff](assets/horizon_tradeoff.png)",
        "",
        md_table(
            horizon_rows,
            [
                ("execute_horizon", "k"),
                ("success_rate", "Success"),
                ("avg_env_steps", "Env Steps"),
                ("avg_policy_calls", "Policy Calls"),
                ("avg_action_smoothness", "Smoothness"),
                ("avg_gripper_flips", "Gripper Flips"),
            ],
        ),
        "",
        "![Horizon smoothness](assets/horizon_smoothness.png)",
        "",
        "Key Insight: larger chunks reduce policy calls sharply, but the improvement in efficiency comes with weaker closed-loop correction. The best horizon is not the largest one; it is the one that balances responsiveness and policy-call cost.",
        "",
        "## Observation Perturbation Benchmark",
        "",
        "Perturbations are applied to the policy observation stream to test whether multi-view VLA execution remains stable under missing views, random occlusion, and brightness shift.",
        "",
        "![Perturbation success](assets/perturbation_success.png)",
        "",
        md_table(
            perturb_rows,
            [
                ("case", "Case"),
                ("success_rate", "Success"),
                ("failures", "Failures"),
                ("avg_steps", "Env Steps"),
                ("policy_calls", "Policy Calls"),
                ("smoothness", "Smoothness"),
                ("gripper_flips", "Gripper Flips"),
                ("top_failed_tasks", "Frequent Failed Task IDs"),
            ],
        ),
        "",
        "![Perturbation recovery effort](assets/perturbation_recovery_effort.png)",
        "",
        "Key Insight: visual perturbations hurt stability much more than brightness changes. Missing-view conditions are the main failure mode, with wrist-view removal causing the steepest drop in success and the largest increase in recovery cost.",
        "",
        "## Files",
        "",
        "- `horizon_sweep_summary.csv`: source metrics for execution horizon comparison.",
        "- `perturbation_sweep_summary.csv`: source metrics for robustness comparison.",
        "- `assets/*.png`: rendered figures for README or slides.",
        "- `assets/*.mp4`: short qualitative rollouts small enough for GitHub review.",
        "",
        "Regenerate this folder with:",
        "",
        "```bash",
        "python scripts/eval_scripts/build_result_gallery.py",
        "```",
        "",
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    horizon_rows = plot_horizon(read_csv(HORIZON_CSV)) if HORIZON_CSV.exists() else []
    perturb_rows = plot_perturbation(read_csv(PERTURB_CSV)) if PERTURB_CSV.exists() else []
    copy_assets()

    write_csv(
        OUT / "horizon_sweep_readable.csv",
        horizon_rows,
        ["execute_horizon", "success_rate", "avg_env_steps", "avg_policy_calls", "avg_action_smoothness", "avg_gripper_flips"],
    )
    write_csv(
        OUT / "perturbation_sweep_readable.csv",
        perturb_rows,
        ["case", "success_rate", "failures", "avg_steps", "policy_calls", "smoothness", "gripper_flips", "top_failed_tasks"],
    )
    write_readme(horizon_rows, perturb_rows)
    print(f"Wrote result gallery to {OUT}")


if __name__ == "__main__":
    main()
