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

from pomt.utils import file_formatter, get_benchmark_arguments, benchmark_latency_ms
from pomt.datasets import create_imagenet1k_dataset, create_im1k_dinov2_dataloader, create_im1k_timm_dataloader
from pomt.tome.patch.timm import apply_patch as timm_apply_tome_patch
from pomt.tome.patch.dinov2 import apply_patch as dinov2_apply_tome_patch
from pomt.timm_patch import timm_apply_pomt_patch
from pomt.timm_patch import timm_apply_topk_patch
from pomt.dinov2_patch import dinov2_apply_pomt_patch
from pomt.dinov2_patch import dinov2_apply_topk_patch


vit_wrapper_LUT = {
    "timm" : {
        "tome" : timm_apply_tome_patch,
        "topk" : timm_apply_topk_patch,
        "pomt" : timm_apply_pomt_patch
    },
    "dino" : {
        "tome" : dinov2_apply_tome_patch,
        "topk" : dinov2_apply_topk_patch,
        "pomt" : dinov2_apply_pomt_patch
    }
}


### Helper to (optionally) wrap a ViT model with ToMe, Top-K, or POMT
def wrap_vit(args: argparse.Namespace, vit: nn.Module) -> nn.Module:
    model_tag_timm_vit = "timm" if ("vit" in args.model or "deit" in args.model) else None
    model_tag_dino = "dino" if "dino" in args.model else None
    model_tag = model_tag_timm_vit or model_tag_dino
    assert model_tag is not None, "Model not supported"
    print(f"Detected model type: {model_tag}")

    if args.wrapper != "none":
        print(f"Wrapping model with: {args.wrapper}")
        wrapper_function = vit_wrapper_LUT[model_tag][args.wrapper]
        return wrapper_function(args, vit)
    else: 
        return vit


def benchmark(args: argparse.Namespace, vit: nn.Module, dataloader: DataLoader) -> Tuple:
    ### Initialize any constants we need
    device = torch.device(args.device)

    ### Port ViT to device, set to eval mode
    vit.to(device)
    vit.eval()

    ### NOTE: We assume there is only a CLS token (prefix_tokens=1) used for all ViTs in this work.
    prefix_tokens = args.prefix_tokens

    ### Used to record accuracy and latency for different token counts
    ### NOTE: This is not the same as L_n and A_n in offline_computation - those measurements are used to compute R
    ### Here, the user supplies a value of R (parameter for our method) or r (parameter for tome, topk)
    ### These are just measurements for accuracy and latency
    L = 0.0
    A = 0.0

    accuracy_estimation_sample_count = len(dataloader) // args.batch_size

    progress_bar = tqdm(dataloader)

    with torch.no_grad():
        ### Initialize loop variables
        running_accuracy = 0.0
        running_predictions = 0.0

        ### Iterate over images and compute accuracy
        for batch_index, (input, target) in enumerate(progress_bar):
            ### Put on device
            input = input.to(device)
            target = target.to(device)

            ### Do our latency benchmarking for the first batch
            if batch_index == 0:
                progress_bar.set_description(desc="Measuring Latency...")
                L = benchmark_latency_ms(vit, input)

            progress_bar.set_description(desc=f"Measuring Accuracy...")

            ### Get model output
            model_output = vit(input)

            ### Argmax, get top 1
            predicted_output = torch.argmax(model_output, dim=1)

            ### Append running accuracy
            running_accuracy += (predicted_output == target).sum().item()
            running_predictions += target.shape[0]


        ### Compute and store accuracy
        A = 100 * running_accuracy / running_predictions

    ### Save as .CSV data
    eval_report = pandas.DataFrame(
        data={
            "Model" : [args.model],
            "Device" : [args.device_name],
            "Batch Size": [args.batch_size],
            "Accuracy (%)": [A],
            "Median Latency (ms)": [L],
        }
    ).to_csv(
        file_formatter(args, "evaluation", "csv"),
        float_format="{:.2f}".format,
    )

    return L, A, eval_report


if __name__ == "__main__":
    args = get_benchmark_arguments()

    ### Input checking
    assert (args.dataset_root is not None), "Supply path ending in the ImageNet1K .../ILSVRC/Data/CLS-LOC/ directory"

    ### Load the model
    vit = timm.create_model(args.model, pretrained=True)
    print(f"Loaded model: {args.model}")

    ### Get dataloader
    im1k_dataset = create_imagenet1k_dataset(args.dataset_root, False)
    im1k_dataloader = create_im1k_timm_dataloader(im1k_dataset, args.batch_size)

    ### Perform benchmarking with a potentially wrapped ViT
    L, A, report = benchmark(args, wrap_vit(args, vit), im1k_dataloader)

    print(f"Done!")
