#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import imageio.v3 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


FULL_VIEW_KEY = "observation.images.exterior_image_1_left"
WRIST_VIEW_KEY = "observation.images.wrist_image_left"


def array_column_to_matrix(series: pd.Series) -> np.ndarray:
    return np.stack(series.map(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())


def load_tasks(tasks_path: Path) -> Dict[int, str]:
    tasks_df = pd.read_parquet(tasks_path)
    return {int(row.task_index): str(idx) for idx, row in tasks_df.iterrows()}


def q_stats(values: np.ndarray) -> Dict[str, List[float]]:
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).astype(float).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).astype(float).tolist(),
        "count": int(values.shape[0]),
    }


def resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.BICUBIC), dtype=np.uint8)


def clean_episodes(
    data_df: pd.DataFrame,
    tasks: Dict[int, str],
    horizon: int,
    observation_horizon: int,
    max_action_abs: float,
) -> pd.DataFrame:
    rows = []
    min_len = horizon + observation_horizon
    for episode_id, ep_df in data_df.groupby("episode_index", sort=True):
        states = array_column_to_matrix(ep_df["observation.state"])
        actions = array_column_to_matrix(ep_df["action"])
        task_index = int(ep_df["task_index"].iloc[0])
        reasons = []

        if len(ep_df) < min_len:
            reasons.append(f"too_short<{min_len}")
        if not np.isfinite(states).all():
            reasons.append("non_finite_state")
        if not np.isfinite(actions).all():
            reasons.append("non_finite_action")
        max_abs = float(np.max(np.abs(actions))) if len(actions) else 0.0
        if max_abs > max_action_abs:
            reasons.append(f"action_abs>{max_action_abs}")
        if not str(tasks.get(task_index, "")).strip():
            reasons.append("missing_language")
        if ep_df["frame_index"].duplicated().any():
            reasons.append("duplicate_frame_index")

        rows.append(
            {
                "episode_id": int(episode_id),
                "status": "drop" if reasons else "keep",
                "reason": ";".join(reasons),
                "num_frames": int(len(ep_df)),
                "task_index": task_index,
                "language_instruction": tasks.get(task_index, ""),
                "state_dim": int(states.shape[1]),
                "action_dim": int(actions.shape[1]),
                "max_action_abs": max_abs,
                "num_nan_state": int(np.isnan(states).sum()),
                "num_nan_action": int(np.isnan(actions).sum()),
                "done_count": int(ep_df["next.done"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_summary(root: Path, data_df: pd.DataFrame, clean_df: pd.DataFrame, info: dict) -> dict:
    state_dim = len(data_df["observation.state"].iloc[0])
    action_dim = len(data_df["action"].iloc[0])
    episode_lengths = clean_df["num_frames"].to_numpy()
    return {
        "source_dataset": "lerobot/droid_100",
        "source_root": str(root),
        "total_episodes": int(info["total_episodes"]),
        "total_frames": int(info["total_frames"]),
        "kept_episodes": int((clean_df["status"] == "keep").sum()),
        "dropped_episodes": int((clean_df["status"] == "drop").sum()),
        "fps": info.get("fps"),
        "image_keys": [FULL_VIEW_KEY, WRIST_VIEW_KEY],
        "raw_image_shape": info["features"][FULL_VIEW_KEY]["shape"],
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "min_episode_len": int(episode_lengths.min()),
        "max_episode_len": int(episode_lengths.max()),
        "avg_episode_len": float(episode_lengths.mean()),
        "notes": [
            "This pipeline is a preprocessing/smoke-test dataset, not a full fine-tuning run.",
            "Observation[t] is paired with action chunk action[t:t+horizon].",
            "Tail frames that cannot form a complete action chunk are dropped in the smoke-test loader.",
        ],
    }


def compute_clean_stats(data_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    keep_ids = set(clean_df.loc[clean_df["status"] == "keep", "episode_id"].astype(int).tolist())
    clean_rows = data_df[data_df["episode_index"].isin(keep_ids)]
    states = array_column_to_matrix(clean_rows["observation.state"])
    actions = array_column_to_matrix(clean_rows["action"])
    return {
        "observation.state": q_stats(states),
        "action": q_stats(actions),
    }


def selected_frame_indices(episodes_df: pd.DataFrame, episode_ids: List[int], max_frames: int) -> Dict[int, Dict[int, int]]:
    mapping = {}
    for episode_id in episode_ids:
        row = episodes_df.loc[episodes_df["episode_index"] == episode_id].iloc[0]
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        selected = list(range(start, min(end, start + max_frames)))
        mapping[episode_id] = {global_idx: local_idx for local_idx, global_idx in enumerate(selected)}
    return mapping


def read_selected_video_frames(video_path: Path, wanted: Iterable[int], resize_size: int) -> Dict[int, np.ndarray]:
    wanted_set = set(int(x) for x in wanted)
    if not wanted_set:
        return {}
    max_wanted = max(wanted_set)
    frames = {}
    for idx, frame in enumerate(iio.imiter(video_path)):
        if idx in wanted_set:
            frames[idx] = resize_rgb(frame, resize_size)
        if idx >= max_wanted:
            break
    missing = wanted_set - set(frames)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} requested frames from {video_path}")
    return frames


def write_hdf5_sample(
    root: Path,
    output_dir: Path,
    data_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    tasks: Dict[int, str],
    max_episodes: int,
    max_frames: int,
    resize_size: int,
) -> List[Path]:
    converted_dir = output_dir / "converted_sample"
    converted_dir.mkdir(parents=True, exist_ok=True)
    keep_ids = clean_df.loc[clean_df["status"] == "keep", "episode_id"].astype(int).tolist()[:max_episodes]
    frame_map = selected_frame_indices(episodes_df, keep_ids, max_frames)
    wanted_indices = sorted({idx for ep_map in frame_map.values() for idx in ep_map.keys()})

    video_root = root / "videos"
    full_frames = read_selected_video_frames(
        video_root / FULL_VIEW_KEY / "chunk-000" / "file-000.mp4", wanted_indices, resize_size
    )
    wrist_frames = read_selected_video_frames(
        video_root / WRIST_VIEW_KEY / "chunk-000" / "file-000.mp4", wanted_indices, resize_size
    )

    written = []
    for episode_id in keep_ids:
        ep_df = data_df[data_df["episode_index"] == episode_id].copy().head(max_frames)
        global_indices = ep_df["index"].astype(int).tolist()
        full = np.stack([full_frames[idx] for idx in global_indices], axis=0)
        wrist = np.stack([wrist_frames[idx] for idx in global_indices], axis=0)
        states = array_column_to_matrix(ep_df["observation.state"])
        actions = array_column_to_matrix(ep_df["action"])
        task_index = int(ep_df["task_index"].iloc[0])

        h5_path = converted_dir / f"episode_{episode_id:06d}.h5"
        with h5py.File(h5_path, "w") as h5:
            obs = h5.create_group("observation")
            obs.create_dataset("full_image", data=full, compression="gzip", compression_opts=1)
            obs.create_dataset("wrist_image", data=wrist, compression="gzip", compression_opts=1)
            obs.create_dataset("state", data=states)
            h5.create_dataset("action", data=actions)
            h5.create_dataset("timestamp", data=ep_df["timestamp"].to_numpy(dtype=np.float32))
            h5.attrs["episode_id"] = int(episode_id)
            h5.attrs["task_index"] = task_index
            h5.attrs["language_instruction"] = tasks.get(task_index, "")
            h5.attrs["source_dataset"] = "lerobot/droid_100"
            h5.attrs["full_view_key"] = FULL_VIEW_KEY
            h5.attrs["wrist_view_key"] = WRIST_VIEW_KEY
        written.append(h5_path)
    return written


def write_sample_rollout_video(h5_path: Path, output_path: Path, stride: int = 4) -> None:
    with h5py.File(h5_path, "r") as h5:
        full = h5["observation/full_image"][()]
        wrist = h5["observation/wrist_image"][()]
        actions = h5["action"][()]
        instruction = h5.attrs["language_instruction"]

    frames = []
    for idx in range(0, len(full), stride):
        canvas = Image.new("RGB", (full.shape[2] * 2, full.shape[1] + 44), "white")
        canvas.paste(Image.fromarray(full[idx]), (0, 0))
        canvas.paste(Image.fromarray(wrist[idx]), (full.shape[2], 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, full.shape[1] + 4), f"t={idx} | {instruction[:80]}", fill=(0, 0, 0))
        draw.text((8, full.shape[1] + 22), f"action={np.round(actions[idx], 3).tolist()}", fill=(0, 0, 0))
        frames.append(np.asarray(canvas))
    iio.imwrite(output_path, frames, fps=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--observation-horizon", type=int, default=1)
    parser.add_argument("--max-action-abs", type=float, default=5.0)
    parser.add_argument("--max-convert-episodes", type=int, default=5)
    parser.add_argument("--max-frames-per-episode", type=int, default=96)
    parser.add_argument("--resize-size", type=int, default=224)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_df = pd.read_parquet(args.dataset_root / "data/chunk-000/file-000.parquet")
    episodes_df = pd.read_parquet(args.dataset_root / "meta/episodes/chunk-000/file-000.parquet")
    tasks = load_tasks(args.dataset_root / "meta/tasks.parquet")
    info = json.loads((args.dataset_root / "meta/info.json").read_text())

    clean_df = clean_episodes(data_df, tasks, args.horizon, args.observation_horizon, args.max_action_abs)
    clean_df.to_csv(args.output_dir / "cleaning_report.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    summary = build_summary(args.dataset_root, data_df, clean_df, info)
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    stats = compute_clean_stats(data_df, clean_df)
    (args.output_dir / "action_state_stats.json").write_text(json.dumps(stats, indent=2))

    written = write_hdf5_sample(
        args.dataset_root,
        args.output_dir,
        data_df,
        episodes_df,
        clean_df,
        tasks,
        args.max_convert_episodes,
        args.max_frames_per_episode,
        args.resize_size,
    )
    if written:
        write_sample_rollout_video(written[0], args.output_dir / "sample_rollout.mp4")

    print(f"Wrote {args.output_dir}")
    print(f"Clean episodes: {(clean_df['status'] == 'keep').sum()} / {len(clean_df)}")
    print(f"Converted episodes: {len(written)}")


if __name__ == "__main__":
    main()
