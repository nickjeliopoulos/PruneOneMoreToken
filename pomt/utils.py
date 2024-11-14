import torch
import torch.utils.benchmark as bench
import numpy
import argparse
from typing import Dict, List, Tuple, Any, Callable, Union, Sequence

BENCHMARK_MIN_RUN_TIME=2.0


### Generic Filename Formatter, for consistency and avoiding having to write f"" strings everywhere
def file_formatter(args: argparse.Namespace, suffix: str, extension: str) -> str:
    return f"{args.output_dir}/{args.model}_bs_{args.batch_size}_{suffix}.{extension}"


def offline_compute_file_formatter(args: argparse.Namespace, suffix: str, extension: str) -> str:
    return f"{args.output_dir}/{args.model}_bs_{args.batch_size}_{suffix}.{extension}"


def benchmark_file_formatter(args: argparse.Namespace, suffix: str, extension: str) -> str:
    r = args.pomt_R if args.wrapper == "pomt" else args.tome_R if args.wrapper == "tome" else args.topk_R if args.wrapper == "topk" else None
    r = f"_{args.wrapper}_r{r}_" if r else "_"
    return f"{args.output_dir}/{args.model}{r}bs_{args.batch_size}_{suffix}.{extension}"


def get_offline_compute_arguments():
    st = "store_true"
    parser = argparse.ArgumentParser(description="Offline Computation Arguments")
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


def get_benchmark_arguments():
    st = "store_true"
    parser = argparse.ArgumentParser(description="Benchmark Arguments")
    parser.add_argument("--model", type=str, default="vit_small_patch16_224")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--grid-token-start", type=int, default=196)
    parser.add_argument("--grid-token-end", type=int, default=2)
    parser.add_argument("--grid-token-stride", type=int, default=1)
    parser.add_argument("--plot-mode", type=str, default="line", choices=["line", "scatter"])
    parser.add_argument("--output-dir", type=str, default="bin/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--prefix-tokens", type=int, choices=[1], default=1)
    parser.add_argument("--wrapper", type=str, choices=["pomt","tome","topk","none"], default="none")
    parser.add_argument("--pomt-prune-layer-index", type=int, default=3)
    parser.add_argument("--pomt-R", type=int, default=None)
    parser.add_argument("--tome-R", type=int, default=None)
    parser.add_argument("--topk-R", type=int, default=None)
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
    measurement = t0.blocked_autorange(min_run_time=BENCHMARK_MIN_RUN_TIME)

    return measurement.median * 1e3 # Convert to milliseconds


### Compute Utility
def compute_utility(args: argparse.Namespace, L: numpy.ndarray, A: numpy.ndarray) -> numpy.ndarray:
    U_L = 1.0 - (L / numpy.max(L))
    U_A = A / numpy.max(A)
    U = args.alpha * U_L + (1.0 - args.alpha) * U_A
    return U


### Compute R
def compute_r(args: argparse.Namespace, x : List, U: numpy.ndarray) -> Tuple:
    N = args.max_vit_token_count
    token_series_index = numpy.argmax(U)
    target_token_count = x[token_series_index]
    R = N - (target_token_count + args.prefix_tokens)
    return R

