#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/unifolm-vla

PYTHON_BIN=/root/autodl-tmp/conda_envs/unifolm-vla/bin/python
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

${PYTHON_BIN} deployment/booster_k1_vlm_skills/k1_ball_hit_sim.py \
  --instruction "hit the blue ball in front of the robot" \
  --ball-x 0.18 \
  --ball-y -0.23 \
  --ball-z 0.78 \
  --out-dir results/k1_ball_hit_mujoco
