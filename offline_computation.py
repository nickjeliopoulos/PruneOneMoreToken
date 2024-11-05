import torch
import torch.nn as nn
import timm
import argparse
import numpy
import tqdm
import os
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any, Callable, Union, Sequence

from pomt.utils import file_formatter, get_offline_compute_arguments, benchmark_latency_ms, compute_r
from pomt.datasets import create_imagenet1k_dataset, create_im1k_dinov2_dataloader, create_im1k_timm_dataloader

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

    ### Number of forward passes for accuracy measurement
    ### NOTE: You can reduce this to minimize runtime. We use the full dataset for accuracy measurement in our work.
    accuracy_estimation_sample_count = len(dataloader)

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

    with torch.no_grad():
        for n in progress_bar:
            ### Break if we want to remove more tokens than possible (due to special tokens)
            if prefix_tokens - n >= 1:
                break
            
            ### Latency Measurement
            if args.latency:
                random_input = torch.randn(size=(args.batch_size, 3, 224, 224), device=device, dtype=torch.float32)
                latency_ms = benchmark_latency_ms(vit, random_input)
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

                    model_output = benchmark_latency_ms(vit, input)

                    ### Argmax, get top 1
                    predicted_output = torch.argmax(model_output, dim=1)
                    assert predicted_output.shape == target.shape

                    running_accuracy += (predicted_output == target).sum().item()
                    running_predictions += target.shape[0]

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
    x = numpy.array(list(L_n.keys()))
    x = 100.0 * x / numpy.max(x)

    ### Get other series
    U_L = numpy.array(list(L_n.values()))
    U_A = numpy.array(list(A_n.values())) if args.plot_accuracy else None
    U = None

    N = args.max_vit_token_count

    ### NOTE: We assume there is only a CLS token (prefix_tokens=1) used for all ViTs in this work.
    prefix_tokens = args.prefix_tokens

    ### Utility plot?
    do_utility_plot = False

    ### Plot Utility in addition to latency and accuracy
    if args.latency and args.accuracy:
        U_L = 1.0 - (L_n / max(L_n))
        U_A = A_n / max(A_n)
        U = args.alpha * U_L + (1.0 - args.alpha) * U_A
        do_utility_plot = True

    figure, axes = plot.subplots(
        nrows=1, ncols=3, figsize=(args.figure_size[0], args.figure_size[1])
    )

    ### Latency Plotting?
    if args.latency:
        latency_handle = _plot_helper(args, axes[0], x, L_n, c="blue")
        axes[0].set_xlabel("Token Density (%)")
        axes[0].set_ylabel("Latency (ms)")

    ### Accuracy Plotting?
    if args.accuracy:
        accuracy_handle = _plot_helper(args, axes[1], x, A_n, c="red")
        axes[1].set_xlabel("Token Density (%)")
        axes[1].set_ylabel("Estimated Accuracy (%)")
        axes[1].set_ybound(lower=0, upper=100.0)

    ### Utility Plotting?
    if do_utility_plot:
        utility_handle = _plot_helper(args, axes[2], x, U, c="orange")
        axes[2].set_xlabel("Token Density (%)")
        axes[2].set_ylabel("Utility")
        axes[2].set_ybound(lower=0, upper=1.1)

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

    ### Load the dataset
    im1k_dataset = None
    im1k_dataloader = None

    if args.accuracy:
        im1k_dataset = create_imagenet1k_dataset(args.dataset_root, False)
        im1k_dataloader = create_im1k_timm_dataloader(im1k_dataset, args.batch_size)

    ### Perform offline computation
    L_n, A_n = offline_computation(args, vit, im1k_dataloader)

    print(f"L_n:\n{L_n}")
    print(f"A_n:\n{A_n}")
    print(f"R={compute_r(args, list(L_n.keys()), L_n)}")

    ### Generate plots
    if not args.no_plot:
        figure = generate_plots(args, L_n, A_n)
        figure.savefig(file_formatter(args, "plots", "png"))