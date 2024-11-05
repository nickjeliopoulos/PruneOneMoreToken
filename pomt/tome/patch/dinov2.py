import torch

### DinoV2
from dinov2.layers.block import (
    Block,
    NestedTensorBlock,
    drop_add_residual_stochastic_depth,
)
    
from dinov2.layers.attention import (
    Attention,
    MemEffAttention,
    XFORMERS_AVAILABLE,
    XFORMERS_ENABLED,
)

from dinov2.hub.classifiers import _LinearClassifierWrapper

### ToMe
from hwtome.merge import bipartite_soft_matching, merge_source, merge_wavg
from hwtome.utils import parse_r

### Standard
from typing import Any, Tuple, List, Dict, Callable
import os
import warnings

# XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
# try:
#     if XFORMERS_ENABLED:
#         from xformers.ops import memory_efficient_attention, unbind

#         XFORMERS_AVAILABLE = True
#         #warnings.warn("xFormers is available (Attention)")
#     else:
#         #warnings.warn("xFormers is disabled (Attention)")
#         raise ImportError
# except ImportError:
#     XFORMERS_AVAILABLE = False
#     #warnings.warn("xFormers is not available (Attention)")

###
### Custom Attention Mechanisms
###
class ToMeDinoV2Attention(Attention):
    def forward(self, x: torch.Tensor, attn_size=None) -> Tuple:
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

        ### Apply proportional attention
        if attn_size is not None:
            # print(f"attn shape: {attn.shape} and attn bias shape: {attn_size.log()[:, None, None, :, 0].shape}")
            attn = attn + attn_size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        ### Modification - this is our metric to return
        return x, k.mean(dim=1)


class ToMeDinoV2MemEffAttention(ToMeDinoV2Attention):
    def forward(self, x: torch.Tensor, attn_bias=None, attn_size=None) -> torch.Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        raise AssertionError("HWToMe (currently) does not support MemEffAttention with prop_attn")

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        ### Each has shape B, #tokens, #heads, # features per head
        q, k, v = unbind(qkv, 2)

        ### See https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/__init__.py
        ### Let's hijack attention bias here!
        # if attn_size is not None:
        #     ### NOTE: Need to call repeat so memory_efficient_attention doesn't yell at us!
        #     ### PyTorch would normally understand what to do in this case, but they have a custom check it would seem
        #     ### GEEZE
        #     ### Need to add an extra virtual "token" to bypas issues with memory alignment from memory_efficient_attention(...)
        #     prop_attn_bias = torch.zeros(size=(B, self.num_heads, N+1, N+1), device=x.device, dtype=x.dtype)
        #     prop_attn_bias[:,:,:N,:N] += attn_size.log()[:, None, None, :, 0]
        #     ### Substitute
        #     if attn_bias is None:
        #         attn_bias = prop_attn_bias[:,:,:N,:N]
        #     ### Add bias
        #     else:
        #         attn_bias += prop_attn_bias[:,:,:N,:N]

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        ### NOTE: dim=2 is proper for taking mean along head dimension (not tokens!)
        return x, k.mean(dim=2)


###
### Custom Blocks
###
class ToMeDinoV2Block(Block):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def attn_residual_func(x: torch.Tensor) -> Tuple:
            attn_size = (
                self._tome_info["size"] if self._tome_info["prop_attn"] else None
            )
            attention, metric = self.attn(self.norm1(x), attn_size=attn_size)
            attention_scaled = self.ls1(attention)
            return attention_scaled, metric

        def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # Raise error
            raise AssertionError(
                "ToMe wrapping does not support the drop_add_residaul_stochastic_depth flag during training"
            )

            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            residual, metric = attn_residual_func(x)
            x = x + self.drop_path1(residual)

            ### Do Token Merging here
            r = self._tome_info["r"].pop(0)
            if r > 0:
                # Apply ToMe here
                merge, _ = bipartite_soft_matching(
                    metric,
                    r,
                    self._tome_info["class_token"],
                    self._tome_info["distill_token"],
                )
                if self._tome_info["trace_source"]:
                    self._tome_info["source"] = merge_source(
                        merge, x, self._tome_info["source"]
                    )
                    x, self._tome_info["size"] = merge_wavg(
                        merge, x, self._tome_info["size"]
                    )

            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            # attn_size = self._tome_info["size"][:, None, None, :, 0].shape if self._tome_info["size"] is not None else "None"
            # print(f"x shape prior: {x.shape} and attn size: {attn_size}")

            residual, metric = attn_residual_func(x)
            x = x + residual

            ###
            ### Do Token Merging here!
            ###
            r = self._tome_info["r"].pop(0)
            if r > 0:
                # Apply ToMe here
                merge, _ = bipartite_soft_matching(
                    metric,
                    r,
                    self._tome_info["class_token"],
                    self._tome_info["distill_token"],
                )
                if self._tome_info["trace_source"]:
                    self._tome_info["source"] = merge_source(
                        merge, x, self._tome_info["source"]
                    )
                x, self._tome_info["size"] = merge_wavg(
                    merge, x, self._tome_info["size"]
                )

            # print(f"x shape post: {x.shape}")

            x = x + ffn_residual_func(x)
        return x


class ToMeDinoV2NestedTensorBlock(ToMeDinoV2Block):
    def forward(self, x_or_x_list):
        if isinstance(x_or_x_list, torch.Tensor):
            return super().forward(x_or_x_list)
        elif isinstance(x_or_x_list, list):
            ### NOTE: Not supported yet
            raise AssertionError(
                "ToMe wrapping does not support list of tensors / nested tensors"
            )
            if not XFORMERS_AVAILABLE:
                raise AssertionError("xFormers is required for using nested tensors")
            return self.forward_nested(x_or_x_list)
        else:
            raise AssertionError


###
### Factory ToMe DinoV2 Generator
###
def make_tome_class(transformer_class: _LinearClassifierWrapper):
    class ToMeDinoVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwargs) -> torch.Tensor:
            ### Update self._tome_info
            self._tome_info["r"] = parse_r(len(self.backbone.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwargs)

    return ToMeDinoVisionTransformer


###
### "Master" function to apply ToMe to a DinoV2 model
###
def apply_patch(
    model: _LinearClassifierWrapper, trace_source: bool = False, prop_attn: bool = True
):
    assert isinstance(model, _LinearClassifierWrapper)

    ToMeDinoVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeDinoVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        ### TODO: Assumption: this is True. Change based on model eh?
        "class_token": True,
        ### TODO: Currently does not support arbitrary special tokens (think register tokens)
        "distill_token": False,
    }

    warn_prop_attn = False

    ### Iterate over backbone modules
    for module in model.backbone.modules():
        ### Note: order matters
        if isinstance(module, NestedTensorBlock):
            module.__class__ = ToMeDinoV2NestedTensorBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Block):
            module.__class__ = ToMeDinoV2Block
            module._tome_info = model._tome_info

        ### Note: order matters
        if isinstance(module, MemEffAttention):
            ### NOTE: Why is this not supported? memory_efficient_attention requires certain dimensionalities / memory alignment requirements
            ### that seems to be finicky when we attempt to use memory efficient attention. See commented out section and do some testing to see this
            ### Therefore, we ignore this flag if we are doing MemoryEfficientAttention
            # if prop_attn and not warn_prop_attn:
            #     print(
            #         "hwtome/patch/dinov2.py: Proportional attention is ignored for MemEffAttention Blocks"
            #     )
            #     warn_prop_attn = True
            module.__class__ = ToMeDinoV2MemEffAttention
        elif isinstance(module, Attention):
            module.__class__ = ToMeDinoV2Attention

    return model
