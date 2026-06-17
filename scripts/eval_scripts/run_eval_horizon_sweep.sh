#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export LIBERO_HOME="${LIBERO_HOME:-/root/autodl-tmp/LIBERO}"
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/unifolm-vla/bin/python}"
your_ckpt="${UNIFOLM_VLA_CKPT:-/root/autodl-tmp/models/UnifoLM-VLA-Libero/checkpoints/pytorch_model.pt}"
vlm_pretrained_path="${UNIFOLM_VLM_BASE:-/root/autodl-tmp/models/UnifoLM-VLM-Base}"
folder_name=UnifoLM-VLA-Libero
step_name=checkpoints

task_suite_name=libero_spatial
# 10 LIBERO-spatial tasks x 5 trials x 4 horizons = 200 total episodes.
num_trials_per_task=5
window_size=2
unnorm_key="libero_spatial_no_noops"
horizons=(1 2 4 8)

DEVICE=0
run_root="results/${task_suite_name}/${folder_name}/${step_name}/horizon_sweep"

for execute_horizon in "${horizons[@]}"; do
    video_out_path="${run_root}/horizon_${execute_horizon}"
    echo "============================================================"
    echo "Running ${task_suite_name}: num_trials_per_task=${num_trials_per_task}, execute_horizon=${execute_horizon}"
    echo "Output: ${video_out_path}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=${DEVICE} "${PYTHON_BIN}" "${REPO_ROOT}/experiments/LIBERO/eval_libero.py" \
        --args.pretrained-path "${your_ckpt}" \
        --args.vlm-pretrained-path "${vlm_pretrained_path}" \
        --args.task-suite-name "${task_suite_name}" \
        --args.num-trials-per-task "${num_trials_per_task}" \
        --args.video-out-path "${video_out_path}" \
        --args.unnorm-key "${unnorm_key}" \
        --args.window-size "${window_size}" \
        --args.execute-horizon "${execute_horizon}"
done

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/eval_scripts/summarize_horizon_sweep.py" "${run_root}"
