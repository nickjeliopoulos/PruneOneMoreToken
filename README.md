<div align="center">

# Pruning One More Token is Enough (POMT)

[![arXiV](https://img.shields.io/badge/arXiV-611111?logo=arxiv)](https://arxiv.org/abs/2407.05941) [![WACV](https://img.shields.io/badge/WACV%202025-0098d3)](https://arxiv.org/abs/2407.05941)

</div>

<div align="center">

This repo contains software artifacts for the WACV 2025 Paper *Pruning One More Token is Enough: Leveraging Latency-Workload Non-Linearities for Vision Transformers on the Edge.*

Feel free to submit pull requests!

</div>

<!-- Graphic -->
<div align="center"><img width="512" alt="image" src="assets/wacv2025_prune_one_more.png"></div>

<!-- Brief Summary, Link to Medium Post -->
# Summary
Vision transfomers (ViTs) are a common architectural component of deep neural networks (DNN). Thus, improving ViT efficiency yields downstream benefits to a wide variety of DNNs. One way to improve ViT efficiency is to remove irrelevant tokens or inputs - this general approach is called token sparsification. Works such as Token Merging (ToMe) illustrate the effectiveness of this approach for improving throughput on high-end systems while avoiding significant accuracy degradation. However, if you are deploying your ViT on an edge device, existing methods like ToMe may increase latency while degrading accuracy. We show this occurs because the relationship between latency and workload-size can be non-linear across ViT models and devices. Ultimately, this is because these methods do not consider behavior stemming from hardware characteristics and workload sizes.

[A brief but informative Medium post about our work can be found here](https://medium.com/your-article-link).

<!-- Structure -->
# Repository Structure [Relevant Section(s) in Paper]
* pomt/
  * dinov2/
  * tome/
  * timm_patch.py [**§3.3**]
  * dinov2_patch.py [**§3.3**]
* benchmark.py [**§4.x**]
* offline_computation.py [**§3.1-2**]

<!-- Installation Guide -->
# Installation 
```bash
python -m pip install ./
```

<!-- Usage Guide -->
# Usage 
> [!CAUTION]
> To best reproduce latency measurements, we encourage users to lock the clock and/or memory rates of their device.
> See the `scripts/` folder for a brief overview of how this is done on NVIDIA GPUs.
> We provide two scripts to set maximuim power mode for the NVIDIA Jetson TX2 and the NVIDIA AGX Orin (32GB) devices used in our work.

## offline_computation.py
This script performs the offline pruning schedule computation described in our work.
Given a device and a pre-trained model, it measures the latency-workload relationship for this device-model pair, then identifies a number of tokens to prune.

Example for DeiT-S with batch-size=4. You can also control the granularity of the grid-search with (start,stop,stride) parameters:
```bash
python offline_computation.py --model deit_small --batch-size 4 --grid-token-start 196 --grid-token-stop 2 --grid-token-stride 1
...
> Saved plots to bin/deit_small_NVIDIA_AGX_ORIN_bs4.png
> Computed R=127
```

## benchmark.py

<!-- Citation -->
# Citation 
```bib
@article{prune_one_more_2024,
  title   = {Pruning One More Token is Enough: Leveraging Latency-Workload Non-Linearities for Vision Transformers on the Edge},
  author  = {Eliopoulos, Nicholas J. and Jajal, Purvish and Liu, Gaowen and Davis, James and Thiravathukal, George K. and Lu, Yung-Hsiang},
  journal = {WACV},
  year    = {2024}
}
```
```
Eliopoulos, N. J., Jajal, P., Liu, G., Davis, J., Thiravathukal, G. K., & Lu, Y-H. (2024). Pruning one more token is enough: Leveraging latency-workload non-linearities for vision transformers on the edge. WACV.
```
