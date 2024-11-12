<div align="center">

# WACV 2025: Pruning One More Token is Enough (POMT)

[![arXiV](https://img.shields.io/badge/arXiV-611111?logo=arxiv)](https://arxiv.org/abs/2407.05941) [![WACV](https://img.shields.io/badge/Proceeding-0098d3)](https://arxiv.org/abs/2407.05941) [![Zenodo](https://img.shields.io/badge/Zenodo-1682d4)](https://google.com)

</div>

<div align="center">

This repo contains software artifacts for the WACV 2025 Paper *Pruning One More Token is Enough: Leveraging Latency-Workload Non-Linearities for Vision Transformers on the Edge.*

Feel free to submit pull requests!

</div>

<!-- Graphic -->
<div align="center"><img width="512" alt="image" src="assets/wacv2025_prune_one_more.png"></div>

<!-- Brief Summary, Link to Medium Post -->
# Summary
Vision transfomers (ViTs) are a common architectural component of deep neural networks (DNN). Thus, improving ViT efficiency yields downstream benefits to a wide variety of DNNs. One way to improve ViT efficiency is to remove irrelevant tokens or inputs - this general approach is called token sparsification. Works such as Token Merging (ToMe) illustrate the effectiveness of this approach for improving throughput on high-end systems while avoiding significant accuracy degradation. However, if you are deploying your ViT on an edge device, existing methods like ToMe may increase latency while degrading accuracy. We show this occurs because the relationship between latency and workload-size can be non-linear across ViT models and devices. Ultimately, this is because these methods do not consider behavior stemming from hardware characteristics and workload sizes. Our work goes a step further, and integrates this information about latency-workload relationships to improve token sparsification.

[A brief Medium post about our work can be found here](https://davisjam.medium.com/pruning-one-more-token-is-enough-9bef04dc799b).

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
> We provide two scripts to set maximum power mode for the NVIDIA Jetson TX2 and the NVIDIA AGX Orin (32GB) devices used in our work.

We illustrate an example of our method on DeiT-S deployed on an NVIDIA AGX Orin below. For an exhaustive list of commandline arguments, see `get_offline_compute_arguments(...)` and `get_benchmark_arguments(...)` in `pomt/utils.py`.

## offline_computation.py
As its name implies, `offline_computation.py` identifies a number of tokens to prune *R* according to the offline computation from our work. Given a device and a pre-trained model, it measures the latency-workload relationship for this device-model pair. Then, we compute *R* using a heuristic based on this relationship. You can also control the granularity (and runtime) of the grid-search with (start,stop,stride) parameters.

> [!TIP]
> It is also possible to separate the grid-search for latency and the grid-search for accuracy estimation. 
> For example, you can estimate accuracy with a high batch size on a more powerful device, then measure latency on the target device for a particular batch size.

Below is an example use of `offline_computation.py` for DeiT-S with batch-size=4:
```bash
> sudo bash scripts/jetson_agxorin_set_clocks.sh
> python offline_computation.py --model deit_small_patch16_224 --batch-size 4 --grid-token-start 196 --grid-token-stop 2 --grid-token-stride 1
Loaded model deit_small_patch16_224
...
Computed R=56 given N=197 input tokens
Done!
```

## benchmark.py
Following the previous example, we now benchmark the DeiT-S model with our pruning method using the computed *R* via `benchmark.py`. The benchmarking script supports wrapping TIMM ViT/DeiT models and DinoV2 models with ToMe, Top-K, POMT, or no wrapper for baseline model measurements.

```bash
> python benchmark.py --model deit_small_patch16_224 --batch-size 4 --pomt-R 56
Loaded model deit_small_patch16_224
...
Saved report bin/deit_small_patch16_224_bs_4_pomt_R56.csv
Done! 
```

<!-- Citation -->
# Citation 
## BibTeX
```bib
@article{prune_one_more_2024,
  title   = {Pruning One More Token is Enough: Leveraging Latency-Workload Non-Linearities for Vision Transformers on the Edge},
  author  = {Eliopoulos, Nicholas J. and Jajal, Purvish and Liu, Gaowen and Davis, James and Thiravathukal, George K. and Lu, Yung-Hsiang},
  journal = {WACV},
  year    = {2024}
}
```
### APA
```
Eliopoulos, N. J., Jajal, P., Liu, G., Davis, J., Thiravathukal, G. K., & Lu, Y-H. (2024). Pruning one more token is enough: Leveraging latency-workload non-linearities for vision transformers on the edge. WACV.
```
