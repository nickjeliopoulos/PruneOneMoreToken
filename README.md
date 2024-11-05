<div align="center">

# Pruning One More Token is Enough

[![arXiV](https://img.shields.io/badge/arXiV-611111?logo=arxiv)](https://arxiv.org/abs/2407.05941) [![WACV](https://img.shields.io/badge/WACV%202025-0098d3)](https://arxiv.org/abs/2407.05941)

</div>

<div align="center">

This repo contains software artifacts for the WACV 2025 Paper *Pruning One More Token is Enough: Leveraging Latency-Workload Non-Linearities for Vision Transformers on the Edge.*

Within the codebase, our method is referred to as **H**ardware **A**ware **T**oken **P**runing (**HATP**).

Feel free to submit pull requests!

</div>

<!-- Graphic -->
<div align="center"><img width="512" alt="image" src="assets/wacv2025_prune_one_more.png"></div>

<!-- Installation Guide -->
# Installation 

<!-- Usage Guide -->
# Usage 
> [!CAUTION]
> To best reproduce latency measurements, we encourage users to lock the clock and/or memory rates of their device.
> See the `scripts/` folder for a brief overview of how this is done on NVIDIA GPUs.
> We provide two scripts to set maximuim power mode for the NVIDIA Jetson TX2 and the NVIDIA AGX Orin (32GB) devices used in our work.

## Offline Computation

## Benchmarking

<!-- Citation -->
# BibTeX Citation 
```bib
@article{prune_one_more_2024,
  title   = {Pruning One More Token is Enough: Leveraging Latency-Workload Non-Linearities for Vision Transformers on the Edge},
  author  = {Eliopoulos, Nicholas J. and Jajal, Purvish and Liu, Gaowen and Davis, James and Thiravathukal, George K. and Lu, Yung-Hsiang},
  journal = {WACV},
  year    = {2023}
}
```