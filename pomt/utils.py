import torch
import torch.utils.benchmark as bench
import numpy
import argparse
from typing import Dict, List, Tuple, Any, Callable, Union, Sequence


### Generic Filename Formatter, for consistency and avoiding having to write f"" strings everywhere
def file_formatter(args: argparse.Namespace, suffix: str, extension: str) -> str:
    return f"{args.output_dir}/{args.model}_{args.device_name}_bs_{args.batch_size}_{suffix}.{extension}"


### Arguments
def get_offline_compute_arguments():
    st = "store_true"
    parser = argparse.ArgumentParser(description="Offline Computation")
    parser.add_argument("--model", type=str, default="vit_small_patch16_224")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("-a", "--accuracy", action=st)
    parser.add_argument("-l", "--latency", action=st)
    parser.add_argument("--max-vit-token-count", type=int, default=196)
    parser.add_argument("--grid-token-start", type=int, default=196)
    parser.add_argument("--grid-token-end", type=int, default=2)
    parser.add_argument("--grid-token-stride", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--no-plot", action=st)
    parser.add_argument("--plot-mode", type=str, default="line", choices=["line", "scatter"])
    parser.add_argument("--output-dir", type=str, default="bin/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--prefix-tokens", type=int, choices=[1], default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if "cuda" in args.device:
        args.device_name = torch.cuda.get_device_name( torch.device(args.device) )
    elif "mps" in args.device:
        args.device_name = "mps"
    else:
        args.device_name = "cpu"

    return args


### Generic benchmark function for profiling wrapped models
### NOTE: Still need to wrap this function call using `with torch.no_grad(...):`
def benchmark_latency_ms(f : Callable, *args, **kwargs) -> int:
    t0 = bench.Timer(
        stmt="f(*args,**kwargs)",
        globals={
            "f" : f,
            "args" : args,
            "kwargs" : kwargs,
        },
        num_threads=1,
    )

    ### NOTE: You can reduce min_run_time to do the grid search faster
    ### For sufficiently large workloads, there may not be enough samples for a reasonably precise measurement
    measurement = t0.blocked_autorange(min_run_time=16.0)

    return measurement.median * 1e3 # Convert to milliseconds


### Compute R
def compute_r(args: argparse.Namespace, x : List, U : Dict, verbose : bool = True) -> Tuple:
    N = args.max_vit_token_count
    token_series_index = numpy.argmax(x)
    target_token_count = x[token_series_index]

    R = N - target_token_count

    if verbose:
        print("R={}".format(R))

    return R
