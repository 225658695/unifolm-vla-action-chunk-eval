REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export LIBERO_HOME="${LIBERO_HOME:-/root/autodl-tmp/LIBERO}"
# export LIBERO_HOME=/path/to/your/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"


# your_ckpt=/path/to/your/Unifolm-VLA-Libero/checkpoints/pytorch_model.pt
# vlm_pretrained_path=/path/to/your/Unifolm-VLM-Base
your_ckpt="${UNIFOLM_VLA_CKPT:-/root/autodl-tmp/models/UnifoLM-VLA-Libero/checkpoints/pytorch_model.pt}"
vlm_pretrained_path="${UNIFOLM_VLM_BASE:-/root/autodl-tmp/models/UnifoLM-VLM-Base}"
folder_name=UnifoLM-VLA-Libero
step_name=checkpoints
task_suite_name=libero_spatial   # libero_goal, libero_object, libero_10, libero_90
num_trials_per_task=1      #次数
window_size=2
execute_horizon=8          # 8 matches the original full action chunk execution.
unnorm_key="libero_spatial_no_noops"  # libero_goal_no_noops, libero_object_no_noops, libero_10_no_noops, libero_90_no_noops

video_out_path="results/${task_suite_name}/${folder_name}/${step_name}/horizon_${execute_horizon}"

DEVICE=0

CUDA_VISIBLE_DEVICES=${DEVICE} "${PYTHON_BIN:-python}" "${REPO_ROOT}/experiments/LIBERO/eval_libero.py" \
    --args.pretrained-path ${your_ckpt} \
    --args.vlm-pretrained-path ${vlm_pretrained_path} \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    --args.unnorm-key "$unnorm_key" \
    --args.window-size "$window_size" \
    --args.execute-horizon "$execute_horizon"
