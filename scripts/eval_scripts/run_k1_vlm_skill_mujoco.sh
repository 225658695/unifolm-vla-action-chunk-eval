#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/unifolm-vla

PYTHON_BIN=/root/autodl-tmp/conda_envs/unifolm-vla/bin/python
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

${PYTHON_BIN} deployment/booster_k1_vlm_skills/k1_mujoco_skill_sim.py \
  --instruction "walk to the wall, stop 0.5m before it, and stand still" \
  --out-dir results/k1_vlm_skill_mujoco
