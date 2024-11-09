import torch

### Standard
from typing import Any, Tuple, List, Dict, Callable
import os
import copy
from types import SimpleNamespace
import argparse

### DinoV2
from dinov2.layers.block import Block, NestedTensorBlock
from dinov2.layers.attention import Attention, MemEffAttention, XFORMERS_AVAILABLE, XFORMERS_ENABLED
from dinov2.hub.classifiers import _LinearClassifierWrapper

### Local
from .timm_patch import POMTMetric as POMTMetric
from .timm_patch import TopKMetric as TopKMetric
from .timm_patch import TopKForward as TIMMTopKForward
from .timm_patch import POMTForward as TIMMPOMTForward

### Copied from DinoV2
XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind
        XFORMERS_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False


###
### Custom Attention Mechanisms
###
class POMTDinoV2Attention(Attention):
    def forward(self, x: torch.Tensor, pomt_info : SimpleNamespace) -> Tuple:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        ### Q, K, V have shapes
        ### B, # heads, # tokens, # features per head
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        ### Get the metric!
        metric = POMTMetric(attn, v, pomt_info)

        ### Continue as normal
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, metric


###
### Custom Attention Mechanisms
###
class TopKDinoV2Attention(Attention):
    def forward(self, x: torch.Tensor, pomt_info : SimpleNamespace) -> Tuple:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        ### Q, K, V have shapes
        ### B, # heads, # tokens, # features per head
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        ### Get the metric!
        metric = TopKMetric(attn, pomt_info)

        ### Continue as normal
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        ### Modification - this is our metric to return
        return x, metric


### NOTE: Does not work for our method 
class TopKDinoV2MemEffAttention(TopKDinoV2Attention):
    def forward(self, x: torch.Tensor, pomt_info : SimpleNamespace, attn_bias=None) -> Tuple:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, pomt_info)

        raise AssertionError("POMT does not support MemEffAttention")


### NOTE: Does not work for our method 
class POMTDinoV2MemEffAttention(POMTDinoV2Attention):
    def forward(self, x: torch.Tensor, pomt_info : SimpleNamespace, attn_bias=None) -> Tuple:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, pomt_info)

        raise AssertionError("POMT does not support MemEffAttention")


###
### Custom Blocks
###
class TopKDinoV2Block(Block):
    def attn_residual_func(self, x: torch.Tensor) -> Tuple:
        attention, metric = self.attn(self.norm1(x), pomt_info=self._pomt_info)
        attention_scaled = self.ls1(attention)
        return attention_scaled, metric
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual, metric = self.attn_residual_func(x)
        x = x + residual
        x = TIMMTopKForward(x, metric, self._pomt_info)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


###
### Custom Blocks
###
class POMTDinoV2Block(Block):
    def attn_residual_func(self, x: torch.Tensor) -> Tuple:
        attention, metric = self.attn(self.norm1(x), pomt_info=self._pomt_info)
        attention_scaled = self.ls1(attention)
        return attention_scaled, metric
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual, metric = self.attn_residual_func(x)
        x = x + residual
        x = TIMMPOMTForward(x, metric, self._pomt_info)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class POMTDinoV2NestedTensorBlock(POMTDinoV2Block):
    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, torch.Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            raise AssertionError(
                "POMT wrapping does not support list of tensors / nested tensors"
            )
        else:
            raise AssertionError


class TopKDinoV2NestedTensorBlock(TopKDinoV2Block):
    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, torch.Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            raise AssertionError(
                "POMT wrapping does not support list of tensors / nested tensors"
            )
        else:
            raise AssertionError


###
### Class Generation
###
def make_pomt_class(transformer_class: _LinearClassifierWrapper):
    class POMTDinoVisionTransformer(transformer_class):
        def forward(self, *args, **kwargs) -> torch.Tensor:
            if isinstance(self.r, list):
                assert(len(self.r) == len(self.backbone.blocks))
                self._pomt_info.r = copy.deepcopy(self.r)
            elif isinstance(self.r, int):
                self._pomt_info.r = [self.r] * len(self.backbone.blocks)
            else:
                raise AssertionError(f"Improper r type {type(self.r)}")

            return super().forward(*args, **kwargs)

    return POMTDinoVisionTransformer


def make_topk_class(transformer_class: _LinearClassifierWrapper):
    class TopKDinoVisionTransformer(transformer_class):
        def forward(self, *args, **kwargs) -> torch.Tensor:
            if isinstance(self.r, list):
                assert(len(self.r) == len(self.backbone.blocks))
                self._pomt_info.r = copy.deepcopy(self.r)
            elif isinstance(self.r, int):
                self._pomt_info.r = [self.r] * len(self.backbone.blocks)
            else:
                raise AssertionError(f"Improper r type {type(self.r)}")

            return super().forward(*args, **kwargs)

    return TopKDinoVisionTransformer


###
### Patching Functions
###
def dinov2_apply_pomt_patch(args : argparse.Namespace, vit: _LinearClassifierWrapper):
    assert isinstance(vit, _LinearClassifierWrapper)
    POMTDinoVisionTransformer = make_pomt_class(vit.__class__)

    vit.__class__ = POMTDinoVisionTransformer
    vit.r = 0
    vit._pomt_info = SimpleNamespace(
        r = vit.r,
        prefix_tokens = args.prefix_tokens,
    )

    ### Iterate over backbone modules
    for module in vit.backbone.modules():
        ### Note: order matters
        if isinstance(module, NestedTensorBlock):
            module.__class__ = POMTDinoV2NestedTensorBlock
            module._pomt_info = vit._pomt_info
        elif isinstance(module, Block):
            module.__class__ = POMTDinoV2Block
            module._pomt_info = vit._pomt_info

        ### Note: order matters
        if isinstance(module, MemEffAttention):
            module.__class__ = POMTDinoV2MemEffAttention
        elif isinstance(module, Attention):
            module.__class__ = POMTDinoV2Attention

    ### Set R
    ### Single layer pruning
    r = [0] * len(vit.blocks)
    r[args.pomt_prune_layer_index] = args.pomt_R
    vit.r = args.r

    return vit


def dinov2_apply_topk_patch(args : argparse.Namespace, vit: _LinearClassifierWrapper):
    assert isinstance(vit, _LinearClassifierWrapper)
    TopKDinoVisionTransformer = make_topk_class(vit.__class__)
    
    vit.__class__ = TopKDinoVisionTransformer
    vit.r = 0
    vit._topk_info = SimpleNamespace(
        r = vit.r,
        prefix_tokens = args.prefix_tokens,
    )

    ### Iterate over backbone modules
    for module in vit.backbone.modules():
        ### Note: order matters
        if isinstance(module, NestedTensorBlock):
            module.__class__ = TopKDinoV2NestedTensorBlock
            module._topk_info = vit._topk_info
        elif isinstance(module, Block):
            module.__class__ = TopKDinoV2Block
            module._topk_info = vit._topk_info

        ### Note: order matters
        if isinstance(module, MemEffAttention):
            module.__class__ = TopKDinoV2MemEffAttention
        elif isinstance(module, Attention):
            module.__class__ = TopKDinoV2Attention

    ### Set r
    r = [args.topk_R] * len(vit.blocks)
    vit.r = r

    return vit

