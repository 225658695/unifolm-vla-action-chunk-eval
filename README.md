# UnifoLM-VLA-0: A Vision-Language-Action (VLA) Framework under UnifoLM Family
<p style="font-size: 1.1em;">
  <a href="https://unigen-x.github.io/unifolm-vla.github.io"><strong>Project Page</strong></a> |
  <a href="https://huggingface.co/unitreerobotics/models"><strong>Models</strong></a> |
  <a href="https://huggingface.co/unitreerobotics/datasets"><strong>Datasets</strong></a>
</p>
<div align="center">
  <p align="right">
    <span> 🌎English </span> | <a href="README_cn.md"> 🇨🇳中文 </a>
  </p>
</div>

This fork is a focused evaluation project, not a full reproduction of the upstream training repo.
It studies how action chunk size and receding-horizon inference change the efficiency, stability, and robustness of VLA policies on LIBERO.

**Hypothesis**

Longer action chunks should reduce inference cost, but they should also weaken closed-loop correction when the environment shifts.

**Mechanism**

- The policy predicts a multi-step action chunk.
- The executor only applies the first `k` steps, then re-observes and re-plans.
- LIBERO benchmark runs measure success, policy calls, smoothness, and robustness under visual perturbations.

**Results**

- Larger chunks reduce policy calls sharply.
- Receding-horizon execution trades responsiveness against inference cost.
- Missing-view and heavy occlusion are the main failure modes for multi-view VLA execution.

**Unified Conclusion**

Action chunking improves execution efficiency, but closed-loop correction becomes weaker as the executed horizon grows. Visual perturbations amplify that weakness, especially when the wrist view is removed.

## Result Gallery

Paper-style figures and summary tables are in [docs/results](docs/results).

## Upstream Reference

The original UnifoLM-VLA repository is kept as the upstream reference: https://github.com/unitreerobotics/unifolm-vla
