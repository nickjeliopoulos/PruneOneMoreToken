import torch
import torch.nn as nn
import timm
import argparse
import pandas
import numpy
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any, Callable, Union, Sequence

from pomt.utils import file_formatter, offline_compute_file_formatter, get_offline_compute_arguments, benchmark_latency_ms, compute_r, compute_utility
from pomt.datasets import create_imagenet1k_dataset, create_im1k_dinov2_dataloader, create_im1k_timm_dataloader


###
### Used for accuracy degradation estimation
### Randomly remove 'n' tokens from 3D (batched) input token tensor
###
def random_prune(x: torch.Tensor, n: int, prefix_tokens: int = 1) -> torch.Tensor:
    ### Generate indicies to keep
    random_prune_scores_indices = torch.randperm(x.shape[1] - prefix_tokens, dtype=torch.long)[0:n-prefix_tokens] + prefix_tokens
    random_prune_scores_indices = torch.cat([torch.tensor(data=list(range(prefix_tokens))), random_prune_scores_indices])
    x = x[:, random_prune_scores_indices, :]
    return x


###
### Forward Functions for different models (TIMM ViT, TIMM DeiT, DinoV2)
### Based on TIMM v0.9.12 - you may have to change if you want to use a more recent TIMM version
### Simple to implement.
###
def random_prune_forward_timm_vit(vit: torch.nn.Module, x: torch.Tensor, n : int, prefix_tokens: int = 1) -> torch.Tensor:
    x = vit.patch_embed(x)
    x = vit._pos_embed(x)
    x = vit.norm_pre(x)
    x = random_prune(x, n, prefix_tokens)
    x = vit.blocks(x)
    x = vit.norm(x)
    return vit.forward_head(x)


###
### Modified from original DinoV2 forward(...) and prepare_tokens_with_masks(...) - removed "masks" and assuming no register tokens
### Note: "vit" is actually a _LinearClassifierWrapper, and not exactly a DinoV2 module
### "vit.backbone" is the underlying DinoV2 module
### It is easiest to hijack the prepare_tokens_with_masks(...) function
###
def random_prune_forward_dinov2(vit: torch.nn.Module, x: torch.Tensor, n : int, prefix_tokens: int = 1) -> torch.Tensor:
    ### Dinov2 prepare_tokens_with_masks(...) function
    ### See dinov2/model.py
    def random_prune_prepare_tokens_with_masks(x: torch.Tensor, masks=None):
        B, N, W, H = x.shape
        x = vit.backbone.patch_embed(x)
        x = torch.cat((vit.backbone.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + vit.backbone.interpolate_pos_encoding(x, W, H)
        x = random_prune(x, n, prefix_tokens)
        return x

    ### Save original forward function
    original_prepare_tokens_with_masks = vit.backbone.prepare_tokens_with_masks
    ### Hijack
    vit.backbone.prepare_tokens_with_masks = random_prune_prepare_tokens_with_masks
    ### Inference
    x = vit(x)
    ### Revert
    vit.backbone.prepare_tokens_with_masks = original_prepare_tokens_with_masks
    return x


###
### LUT for forward functions given the class of the ViT model
### NOTE: In our paper, we use DeiT models without the distilation token (equivalent to ViT architecture)
### Thus we can use the same TIMM ViT forward for DeiT models
###
vit_random_prune_forward_LUT = {
    "vit" : random_prune_forward_timm_vit,
    "deit" : random_prune_forward_timm_vit,
    "dino" : random_prune_forward_dinov2,
}


### Perform a grid-search to compute R, the number of tokens to prune
### By default, generate a plot of latency + accuracy degradation
def offline_computation(args: argparse.Namespace, vit: torch.nn.Module, dataloader: DataLoader = None) -> Tuple:
    ### Initialize any constants we need
    device = torch.device(args.device)

    ### Port ViT to device, set to eval mode
    vit.to(device)
    vit.eval()

    ### Record average latency and estimated accuracy degradation for each candidate number of tokens to keep
    L_n = {}
    A_n = {}

    ### NOTE: We assume there is only a CLS token (prefix_tokens=1) used for all ViTs in this work.
    prefix_tokens = args.prefix_tokens

    ### Create the grid search problem size
    grid_search_problem = (
        list(
            range(
                args.grid_token_start,
                args.grid_token_end-1,
                -args.grid_token_stride,
            )
        )
        if not args.grid_token_start == args.grid_token_end else [args.grid_token_start]
    )

    progress_bar = tqdm(grid_search_problem)

    ### First, get the forward function for the model
    model_tag_timm_vit = "vit" if "vit" in args.model else None
    model_tag_deit = "deit" if "deit" in args.model else None
    model_tag_dino = "dino" if "dino" in args.model else None
    model_tag = model_tag_timm_vit or model_tag_deit or model_tag_dino
    assert model_tag is not None, "Model not supported"
    print(f"Detected model type: {model_tag}")
    forward_fn = vit_random_prune_forward_LUT[model_tag]

    if args.accuracy:
        ### Set maximum sample count
        accuracy_estimation_sample_count = 1024 // args.batch_size
        # accuracy_estimation_sample_count = len(dataloader)

    with torch.no_grad():
        for n in progress_bar:
            ### Break if we want to remove more tokens than possible (due to special tokens)
            if (n - prefix_tokens) <= 0:
                break
            
            ### Latency Measurement
            if args.latency:
                progress_bar.set_description(f"Measuring Latency...")
                random_input = torch.randn(size=(args.batch_size, 3, 224, 224), device=device, dtype=torch.float32)
                latency_ms = benchmark_latency_ms(forward_fn, vit, random_input, n, prefix_tokens)
                L_n[n] = latency_ms
            else:
                L_n[n] = -1.0

            ### Accuracy Measurement
            if args.accuracy:
                ### Record running accuracy
                running_accuracy = 0.0
                running_predictions = 0.0

                for batch_index, (input, target) in enumerate(dataloader):
                    ### Put on device
                    input = input.to(device)
                    target = target.to(device)

                    if batch_index >= accuracy_estimation_sample_count:
                        break

                    model_output = forward_fn(vit, input, n, prefix_tokens)

                    ### Argmax, get top 1
                    predicted_output = torch.argmax(model_output, dim=1)

                    running_accuracy += (predicted_output == target).sum().item()
                    running_predictions += target.shape[0]

                    progress_bar.set_description(f"Acc. Batch Idx {1+batch_index}/{accuracy_estimation_sample_count}")

                ### Update token latency lut
                A_n[n] = 100.0 * running_accuracy / running_predictions

    return L_n, A_n


###
### Plot latency, accuracy, and utility
###
def generate_plots(args: argparse.Namespace, L_n: Dict, A_n: Dict):
    ### Matplotlib
    import matplotlib
    from matplotlib import pyplot as plot
    from matplotlib.figure import Figure
    from matplotlib.ticker import MultipleLocator, MaxNLocator

    ###
    ### Use different backend to sidestep stupid pyplot error on Windows
    ###
    if os.name == "nt":
        matplotlib.use("TKAgg")

    ### Font size config
    SMALL_SIZE = 24
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 28

    plot.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plot.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plot.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plot.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plot.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plot.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plot.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    ### Helper functions for plotting
    def _plot_helper(args : argparse.Namespace, axes, x : numpy.ndarray, y : numpy.ndarray, c : str) -> Any:
        if args.plot_mode == "line":
            handle = axes.plot(x, y, linewidth=3, c=c)
        elif args.plot_mode == "scatter":
            handle = axes.scatter(x, y, s=48, c=c)
        else:
            raise AssertionError("Improper plot mode")
        return handle

    ### Done with modules - time to plot
    x_tokens = numpy.array(list(L_n.keys()))
    x_density = 100.0 * x_tokens / numpy.max(x_tokens)

    ### Get other series
    L = numpy.array(list(L_n.values()))
    A = numpy.array(list(A_n.values()))
    U_L = numpy.array(list(L_n.values()))
    U_A = numpy.array(list(A_n.values())) if args.accuracy else None
    U = None

    N = args.max_vit_token_count

    ### NOTE: We assume there is only a CLS token (prefix_tokens=1) used for all ViTs in this work.
    prefix_tokens = args.prefix_tokens

    figure, axes = plot.subplots(
        nrows=1, ncols=3, figsize=(20, 5)
    )

    ### Latency Plotting?
    if args.latency:
        latency_handle = _plot_helper(args, axes[0], x_density, L, c="blue")
        axes[0].set_xlabel("Token Density (%)")
        axes[0].set_ylabel("Latency (ms)")

    ### Accuracy Plotting?
    if args.accuracy:
        accuracy_handle = _plot_helper(args, axes[1], x_density, A, c="red")
        axes[1].set_xlabel("Token Density (%)")
        axes[1].set_ylabel("Estimated Accuracy (%)")
        axes[1].set_ybound(lower=0, upper=100.0)

    ### Plot Utility in addition to latency and accuracy
    if args.latency and args.accuracy:
        U = compute_utility(args, L, A)
        R = compute_r(args, x_tokens, U)
        print(f"Computed R={R} given N={prefix_tokens+N} input tokens")
        utility_handle = _plot_helper(args, axes[2], x_density, U, c="orange")
        axes[2].set_xlabel("Token Density (%)")
        axes[2].set_ylabel("Utility")
        axes[2].set_ybound(lower=0, upper=1.1)

        ### .CSV report name
        csv_report_filename = file_formatter(args, "offline_report", "csv")

        ### Save as .CSV data
        offline_report_df = pandas.DataFrame(
            data={
                "Model" : [args.model],
                "Device" : [args.device_name],
                "Batch Size": [args.batch_size],
                "Max Token Count" : [N+prefix_tokens],
                "Prefix Tokens" : [prefix_tokens],
                "Grid Search Token Start": [args.grid_token_start],
                "Grid Search Token End": [args.grid_token_end-1],
                "Grid Search Token Stride": [-args.grid_token_stride],
                "R": [R],
            }
        ).to_csv(
            csv_report_filename,
            float_format="{:.2f}".format,
        )
        print(f"Saved offline computation report to {csv_report_filename}")

    ### Formatting
    for axis in axes:
        axis.grid(axis="both", color="xkcd:light gray", linestyle="dashed", linewidth=3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_linewidth(w=3)
        axis.spines["left"].set_linewidth(w=3)
        axis.spines["bottom"].set_linewidth(w=3)
        axis.set_xbound(lower=0, upper=100)

    figure.tight_layout()
    return figure


if __name__ == "__main__":
    args = get_offline_compute_arguments()

    ### Input checking
    assert args.latency or args.accuracy, "Supply either --latency or --accuracy"
    assert not (args.accuracy and args.dataset_root is None), "Supply path leading up the ImageNet1K CLS-LOC/ directory for accuracy estimation"

    ### Load the model
    vit = timm.create_model(args.model, pretrained=True)
    print(f"Loaded model: {args.model}")

    ### Load the dataset
    im1k_dataset = None
    im1k_dataloader = None

    if args.accuracy:
        im1k_dataset = create_imagenet1k_dataset(args.dataset_root, False)
        im1k_dataloader = create_im1k_timm_dataloader(im1k_dataset, args.batch_size)

    ### Perform offline computation
    L_n, A_n = offline_computation(args, vit, im1k_dataloader)

    ### Generate plots, compute R
    figure = generate_plots(args, L_n, A_n)
    figure.savefig(file_formatter(args, "plots", "png"))

    print(f"Done!")
    