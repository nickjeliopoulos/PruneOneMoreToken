import torch
import torch.utils.benchmark as bench
import argparse
from typing import Dict, Tuple, Any, Callable, Union, Sequence


### Number of trials for benchmarking latency
N = 128


### Generic Filename Formatter, for consistency and avoiding having to write f"" strings everywhere
def file_formatter(args: argparse.Namespace, suffix: str, extension: str) -> str:
    return f"{args.output_dir}/{args.model}_{args.device_name}_bs_{args.batch_size}_{suffix}.{extension}"


### Arguments
def get_offline_compute_arguments():
    st = "store_true"
    parser = argparse.ArgumentParser(description="Offline Computation")
    parser.add_argument("--model", type=str, default="deit_small_patch16_224")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("-a", "--accuracy", action=st)
    parser.add_argument("-l", "--latency", action=st)
    parser.add_argument("--max-vit-token-count", type=int, default=196)
    parser.add_argument("--grid-token-start", type=int, default=196)
    parser.add_argument("--grid-token-end", type=int, default=2)
    parser.add_argument("--grid-token-stride", type=int, default=1)
    parser.add_argument("--no-plot", action=st)
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

    measurement = t0.timeit(N)

    return measurement.median * 1e3 # Convert to milliseconds