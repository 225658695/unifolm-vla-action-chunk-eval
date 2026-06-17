#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/unifolm-vla

PYTHON_BIN=/root/autodl-tmp/conda_envs/unifolm-vla/bin/python
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=/root/autodl-tmp/booster_repos/booster_deploy:/root/autodl-tmp/k1-standing-long-jump/third_party/booster_assets/src:${PYTHONPATH:-}

${PYTHON_BIN} deployment/booster_k1_vlm_skills/k1_official_motion_render.py \
  --task k1_fight \
  --duration 5.0 \
  --out-dir results/k1_official_motion \
  --follow
