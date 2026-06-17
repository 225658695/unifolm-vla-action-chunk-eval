#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/unifolm-vla

PYTHON_BIN=/root/autodl-tmp/conda_envs/unifolm-vla/bin/python

${PYTHON_BIN} deployment/booster_k1_vlm_skills/k1_official_locomotion_client.py \
  --instruction "walk to the wall, stop 0.5m before it, and stand still" \
  --wall-distance-m 1.05 \
  --stop-distance-m 0.5 \
  --out-dir results/k1_official_locomotion
