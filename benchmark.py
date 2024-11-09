import torch
import torch.nn as nn
import timm
import argparse
import numpy
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Any, Callable, Union, Sequence

from pomt.utils import file_formatter, get_offline_compute_arguments, benchmark_latency_ms, compute_r, compute_utility
from pomt.datasets import create_imagenet1k_dataset, create_im1k_dinov2_dataloader, create_im1k_timm_dataloader

