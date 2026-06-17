#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/unifolm-vla

# Replace only these two paths when you get a new matched action pair.
# Relative paths are resolved from:
#   /root/autodl-tmp/booster_repos/booster_deploy/tasks/beyond_mimic
CHECKPOINT="models/k1_fight_001.pt"
MOTION="motions/k1_fight_final_deploy.npz"

# Optional: change the output folder/name for each action.
OUT_DIR="results/k1_custom_motion"

PYTHON_BIN=/root/autodl-tmp/conda_envs/unifolm-vla/bin/python
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=/root/autodl-tmp/booster_repos/booster_deploy:/root/autodl-tmp/k1-standing-long-jump/third_party/booster_assets/src:${PYTHONPATH:-}

${PYTHON_BIN} deployment/booster_k1_vlm_skills/k1_official_motion_render.py \
  --task k1_fight \
  --checkpoint "${CHECKPOINT}" \
  --motion "${MOTION}" \
  --duration 5.0 \
  --out-dir "${OUT_DIR}" \
  --follow
