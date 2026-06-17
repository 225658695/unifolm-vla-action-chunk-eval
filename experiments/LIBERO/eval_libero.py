import dataclasses
import datetime as dt
import json
import logging
import math
import os
import pathlib
from pathlib import Path
import requests
import time
import logging
# import tensorflow as tf
import imageio
import numpy as np
import tqdm
import tyro
from typing import Any, Dict, List, Optional, Tuple, Union
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from qwen_vl_utils import process_vision_info
import torch
from collections import deque
from unifolm_vla.rlds_dataloader.constants import NUM_ACTIONS_CHUNK
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from experiments.LIBERO.unifolm_vla_inference import Unifolm_VLA_Inference
from experiments.LIBERO.libero_utils import DATE_TIME,DATE

from experiments.LIBERO.libero_utils import (
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    resize_image_for_policy,
    prepare_images_for_vla
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)



LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    resize_size = [224,224]

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = 0  # Optional limit for smoke tests. 0 means evaluate all tasks.
    max_steps: int = 0  # Optional rollout step limit for smoke tests. 0 means use suite default.
    execute_horizon: int = NUM_ACTIONS_CHUNK  # Receding-horizon length. NUM_ACTIONS_CHUNK matches the original policy.
    window_size: int = 2
    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "results"  # Path to save videos
    local_log_dir: str = "./experiments/logs"
    
    seed: int = 42  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    post_process_action: bool = True

    unnorm_key: str = "libero_goal_no_noops"
    vlm_pretrained_path: str = None

def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img 

def setup_logging(args: Args):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{args.task_suite_name}-{DATE_TIME}"
    # Set up local logging
    os.makedirs(args.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(args.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    return log_file, local_log_filepath

def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()

def compute_action_metrics(executed_actions: List[np.ndarray]) -> Dict[str, float]:
    """Compute rollout-level action stability metrics from processed environment actions."""
    if len(executed_actions) == 0:
        return {
            "action_smoothness": 0.0,
            "position_smoothness": 0.0,
            "rotation_smoothness": 0.0,
            "gripper_flips": 0,
        }

    actions = np.asarray(executed_actions, dtype=np.float32)
    if len(actions) < 2:
        return {
            "action_smoothness": 0.0,
            "position_smoothness": 0.0,
            "rotation_smoothness": 0.0,
            "gripper_flips": 0,
        }

    deltas = np.diff(actions, axis=0)
    gripper = actions[:, -1]
    return {
        "action_smoothness": float(np.mean(np.linalg.norm(deltas, axis=1))),
        "position_smoothness": float(np.mean(np.linalg.norm(deltas[:, :3], axis=1))),
        "rotation_smoothness": float(np.mean(np.linalg.norm(deltas[:, 3:6], axis=1))),
        "gripper_flips": int(np.sum(np.diff(np.sign(gripper)) != 0)),
    }

def get_action_state(observations: deque, task_description: str, model: Unifolm_VLA_Inference):

    all_images = []
    for observation in observations:
        all_images.append(observation["full_image"])
    for observation in observations:
        all_images.extend([observation[k] for k in observation.keys() if "wrist" in k])
        
    all_images = prepare_images_for_vla(all_images)
        

    # text = f"The task is \"{task_description.lower()}\"."
    text = f"You are a robot using the joint control. The task is \"{task_description.lower()}\". Please predict up to 10 key trajectory points to complete the task. Your answer should be formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], where each tuple contains the x and y coordinates of a point."
    messages = [
        {
            "role": "user",
            "content": [
                *[
                    {"type": "image", "image": img}
                    for img in all_images
                ],
                {"type": "text", "text": text},
            ],
        },
    ]
    text = model.vla.qwen_vl_interface.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    qwen_inputs = model.vla.qwen_vl_interface.processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


    qwen_inputs["state"] = np.stack([obs["state"] for obs in observations], axis=0)
    actions = model.step(qwen_inputs) 
    
    return actions


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    if args.execute_horizon < 1 or args.execute_horizon > NUM_ACTIONS_CHUNK:
        raise ValueError(
            f"execute_horizon must be in [1, {NUM_ACTIONS_CHUNK}], got {args.execute_horizon}"
        )
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")


    pathlib.Path(args.video_out_path + "/" + DATE).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  
    elif args.task_suite_name == "libero_object":
        max_steps = 280  
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  
    elif args.task_suite_name == "libero_10":
        max_steps = 520  
    elif args.task_suite_name == "libero_90":
        max_steps = 400  
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    if args.max_steps > 0:
        max_steps = args.max_steps

    model = Unifolm_VLA_Inference(
        policy_ckpt_path=args.pretrained_path, # to get unnormalization stats
        image_size=args.resize_size,
        unnorm_key=args.unnorm_key,
        vlm_pretrained_path=args.vlm_pretrained_path,
    )

    log_file, local_log_filepath = setup_logging(args)
    # Start evaluation
    total_episodes, total_successes = 0, 0
    total_policy_calls, total_env_steps = 0, 0
    episode_summaries = []
    num_tasks_to_eval = min(num_tasks_in_suite, args.max_tasks) if args.max_tasks > 0 else num_tasks_in_suite
    for task_id in tqdm.tqdm(range(num_tasks_to_eval)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            log_message(f"\nTask: {task_description}", log_file)

            # Reset environment
            model.reset(task_description=task_description)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            log_message(f"Starting episode {task_episodes + 1}...", log_file)
            step = 0
            episode_policy_calls = 0
            executed_actions = []
            
            action_queue = deque(maxlen=args.execute_horizon)
            obs_queue = deque(maxlen=args.window_size)
            success = False
            while t < max_steps + args.num_steps_wait:

                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                while len(obs_queue) < args.window_size:
                    observation, img = prepare_observation(obs, resize_size=224)
                    obs_queue.append(observation)
                replay_images.append(img)
                
                if len(action_queue) == 0:
                    actions = get_action_state(obs_queue, task_description, model)
                    episode_policy_calls += 1
                    action_queue.extend(actions[:args.execute_horizon])
                obs_queue.popleft()
                action = action_queue.popleft()
                
                action = process_action(action)
                executed_actions.append(action.copy())
                
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    success = True
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            action_metrics = compute_action_metrics(executed_actions)
            total_policy_calls += episode_policy_calls
            total_env_steps += step
            task_episodes += 1
            total_episodes += 1
            episode_summary = {
                "task_id": task_id,
                "episode_idx": episode_idx,
                "task_description": task_description,
                "success": bool(success),
                "env_steps": int(step),
                "policy_calls": int(episode_policy_calls),
                "execute_horizon": int(args.execute_horizon),
                **action_metrics,
            }
            episode_summaries.append(episode_summary)
            
            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            if episode_idx == 0 or not success:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
            
            log_message(f"Success: {done}", log_file)
            log_message(f"Env steps: {step}", log_file)
            log_message(f"Policy calls: {episode_policy_calls}", log_file)
            log_message(f"Action smoothness: {action_metrics['action_smoothness']:.6f}", log_file)
            log_message(f"Position smoothness: {action_metrics['position_smoothness']:.6f}", log_file)
            log_message(f"Rotation smoothness: {action_metrics['rotation_smoothness']:.6f}", log_file)
            log_message(f"Gripper flips: {action_metrics['gripper_flips']}", log_file)
            log_message(f"# episodes completed so far: {total_episodes}", log_file)
            log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

        # Log final results
        log_message(f"Current task success rate: {float(task_successes) / float(task_episodes)}", log_file)
        log_message(f"Current total success rate: {float(total_successes) / float(total_episodes)}", log_file)
    
    
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)
    log_message(f"Total policy calls: {total_policy_calls}", log_file)
    log_message(f"Average policy calls per episode: {total_policy_calls / total_episodes if total_episodes else 0:.4f}", log_file)
    log_message(f"Average env steps per episode: {total_env_steps / total_episodes if total_episodes else 0:.4f}", log_file)

    summary = {
        "task_suite_name": args.task_suite_name,
        "execute_horizon": args.execute_horizon,
        "num_trials_per_task": args.num_trials_per_task,
        "num_tasks": num_tasks_to_eval,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": final_success_rate,
        "total_policy_calls": total_policy_calls,
        "avg_policy_calls": total_policy_calls / total_episodes if total_episodes else 0,
        "avg_env_steps": total_env_steps / total_episodes if total_episodes else 0,
        "avg_action_smoothness": float(np.mean([ep["action_smoothness"] for ep in episode_summaries])) if episode_summaries else 0.0,
        "avg_position_smoothness": float(np.mean([ep["position_smoothness"] for ep in episode_summaries])) if episode_summaries else 0.0,
        "avg_rotation_smoothness": float(np.mean([ep["rotation_smoothness"] for ep in episode_summaries])) if episode_summaries else 0.0,
        "avg_gripper_flips": float(np.mean([ep["gripper_flips"] for ep in episode_summaries])) if episode_summaries else 0.0,
        "episodes": episode_summaries,
    }
    summary_path = pathlib.Path(args.video_out_path) / f"summary_h{args.execute_horizon}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log_message(f"Summary JSON: {summary_path}", log_file)
    
    if log_file:
        log_file.close()
        
def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action

def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action


def process_action(action):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    action = invert_gripper_action(action)

    return action


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    tyro.cli(eval_libero)
