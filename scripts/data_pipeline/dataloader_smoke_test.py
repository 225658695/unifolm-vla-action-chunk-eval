#!/usr/bin/env python
import argparse
from pathlib import Path

import h5py
import numpy as np


def make_action_chunk(actions: np.ndarray, start: int, horizon: int) -> np.ndarray:
    return actions[start : start + horizon]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--converted-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    episode_paths = sorted(args.converted_dir.glob("episode_*.h5"))
    if not episode_paths:
        raise SystemExit(f"No episode_*.h5 files found under {args.converted_dir}")

    samples = []
    for path in episode_paths:
        with h5py.File(path, "r") as h5:
            actions = h5["action"][()]
            if len(actions) < args.horizon:
                continue
            samples.append(
                {
                    "full_image": h5["observation/full_image"][0],
                    "wrist_image": h5["observation/wrist_image"][0],
                    "state": h5["observation/state"][0],
                    "action_chunk": make_action_chunk(actions, 0, args.horizon),
                    "language_instruction": h5.attrs["language_instruction"],
                    "episode_id": int(h5.attrs["episode_id"]),
                }
            )
        if len(samples) >= args.batch_size:
            break

    if not samples:
        raise SystemExit("No valid samples with enough action horizon.")

    batch = {
        "full_image": np.stack([s["full_image"] for s in samples]),
        "wrist_image": np.stack([s["wrist_image"] for s in samples]),
        "state": np.stack([s["state"] for s in samples]),
        "action_chunk": np.stack([s["action_chunk"] for s in samples]),
        "language_instruction": [s["language_instruction"] for s in samples],
        "episode_id": [s["episode_id"] for s in samples],
    }

    print("Dataloader smoke test batch:")
    print("  full_image:", batch["full_image"].shape, batch["full_image"].dtype)
    print("  wrist_image:", batch["wrist_image"].shape, batch["wrist_image"].dtype)
    print("  state:", batch["state"].shape, batch["state"].dtype)
    print("  action_chunk:", batch["action_chunk"].shape, batch["action_chunk"].dtype)
    print("  episode_id:", batch["episode_id"])
    print("  language_instruction:", batch["language_instruction"])


if __name__ == "__main__":
    main()
