# UnifoLM-VLA-0: UnifoLM 家族下的视觉-语言-动作框架
<p style="font-size: 1.2em;">
  <a href="https://unigen-x.github.io/unifolm-vla.github.io"><strong>项目主页</strong></a> |
  <a href="https://huggingface.co/unitreerobotics/models"><strong>开源模型</strong></a> |
  <a href="https://huggingface.co/unitreerobotics/datasets"><strong>开源数据</strong></a>
</p>

<div align="center">
  <p align="right">
    <span> 🌎English </span> | <a href="README_cn.md"> 🇨🇳中文 </a>
  </p>
</div>

这个 fork 不是完整复现官方训练仓库，而是一个聚焦于评测的项目。它研究 action chunk 大小和 receding-horizon 推理如何影响 LIBERO 上 VLA policy 的执行效率、稳定性和鲁棒性。

<div align="center">
  <img
    src="assets/gif/UnifoLM-VLA-0.gif"
    style="width:100%; max-width:1000px; height:auto;"
  />
</div>

**机制**

- policy 先预测一个多步 action chunk。
- executor 每次只执行前 `k` 步，然后重新观测并重规划。
- 在 LIBERO 上统计 success、policy calls、smoothness 以及视觉扰动下的鲁棒性。

**结果**

- horizon sweep：更大的 chunk 会显著减少 policy calls，并改变效率和稳定性的权衡。
- perturbation benchmark：视角缺失和严重遮挡是多视角 VLA 的主要失效模式。
- receding-horizon 在响应性和推理成本之间形成权衡。

**统一结论**

action chunking 能提升执行效率，但随着执行 horizon 增大，闭环纠错会变弱。视觉扰动会进一步放大这种弱点，尤其是 wrist 视角被移除时。

## 结果页

论文式图表和汇总表见 [docs/results](docs/results)。
也可以直接看一个代表性的 rollout 视频：[libero_spatial_success_example.mp4](docs/results/assets/libero_spatial_success_example.mp4)。

## 上游参考

原始 UnifoLM-VLA 仓库保留为上游参考：https://github.com/unitreerobotics/unifolm-vla
