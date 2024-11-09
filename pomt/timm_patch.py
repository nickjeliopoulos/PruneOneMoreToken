import torch
import torch.nn
import torch.utils.data
from types import SimpleNamespace
from typing import Any, List, Tuple, Iterator
from argparse import Namespace
import copy

### TIMM
from timm.models.vision_transformer import Block, Attention


def POMTMetric(softmax_attn : torch.Tensor, v : torch.Tensor, pomt_info : SimpleNamespace) -> torch.Tensor:
        ###
        ### metric_attn Measures how much each token is attended to by all other tokens - this is a metric to gauge token 'importance'
        ###
        metric_attn = (
            torch.amax(softmax_attn[..., pomt_info.prefix_tokens :, pomt_info.prefix_tokens :], dim=1)
            .sum(dim=1, keepdim=True) # Sum along the 'rows' - this essentially counts how much a particular token is attended to by all other tokens
            .transpose(-2,-1) # Need to reshape into (B, ..., 1) instead of (B, 1, ...) shape
        )
        # ### Maximum possible value for any score = 1
        metric_attn = metric_attn / torch.max(metric_attn)

        ###
        ### metric_magnitude estimates information content (magnitude of information) by summing features of V - this is a metric to gauge token 'importance'
        ###
        metric_v = torch.softmax(
            torch.amax(v, dim=1)
            .sum(dim=-1, keepdim=True), # Sum along the # feature per head dimension
            dim=1 # Apply softmax across the sums of features for each token
        )[:, pomt_info.prefix_tokens :] # Shave off special tokens, attenuate
        ### Maximum possible value for any score = 1

        ###
        ### Combine Importance Scores
        ###
        metric = (metric_attn + metric_v)

        return metric


def TopKMetric(softmax_attn : torch.Tensor, topk_info : SimpleNamespace) -> torch.Tensor:
    ### NOTE: Assumes CLS token exists
    ### softmax attn has B, # heads, N, N
    ### Collapse head dimension via average
    metric = softmax_attn.mean(dim=1)[:, 0, 1:]

    ### Metric has shape B, N-1, 1
    return metric.unsqueeze(-1)


def TopKForward(x : torch.Tensor, metric : torch.Tensor, topk_info : SimpleNamespace) -> torch.Tensor:
    ### Apply metric
    B, N, C = x.size()
    r = topk_info.r.pop(0)
    T = N - r - 1

    ### Return early if we don't have to do anything
    if r == 0 or T <= topk_info.prefix_tokens:
        return x
    
    ### Add offset - since we shaved off the prefix tokens we need to account for that with our indices
    offset = torch.tensor(
        data=[1], dtype=torch.long, device=x.device
    )

    indices = metric.argsort(dim=1, descending=True) + offset
    kept_indices = indices[:, :T]

    x = torch.cat(
        (
            x[:, 0 : topk_info.prefix_tokens, :],
            torch.gather(x, dim=1, index=kept_indices.expand(B, T, C)),
        ),
        dim=1,
    )

    return x


def POMTForward(x : torch.Tensor, metric : torch.Tensor, pomt_info : SimpleNamespace) -> torch.Tensor:
    ### Apply metric
    B, N, C = x.size()
    r = pomt_info.r.pop(0)
    T = N - r - 2

    ### Return early if we don't have to do anything
    if r == 0 or T <= pomt_info.prefix_tokens:
        return x, None

    ### Add offset - since we shaved off the prefix tokens we need to account for that with our indices
    offset = torch.tensor(
        data=[pomt_info.prefix_tokens], dtype=torch.long, device=x.device
    )
    similarity_indices = metric.argsort(dim=1, descending=True) + offset

    ### Keep highest scores
    kept_indices = similarity_indices[:, :T]
    discard_indices = similarity_indices[:, T:]

    ### Congregate the discarded tokens then take the mean of them
    x_discarded = torch.gather(x, dim=1, index=discard_indices.expand(B, discard_indices.shape[1], C)).mean(dim=1, keepdim=True)

    ### Create pruned x'
    x = torch.cat(
        (
            x[:, 0 : pomt_info.prefix_tokens, :],
            torch.gather(x, dim=1, index=kept_indices.expand(B, T, C)),
            x_discarded,
        ),
        dim=1,
    )

    return x, discard_indices


###
### Intended to be used with image classification tasks
### Function that computes masked attention according to the Dynamic ViT (2021 NEURIPS) paper in addition to a novel technique
### Intended to overwrite an Attention layer during training, copying all of its attributes but now with additional functionality
###
class POMTAttention(Attention):
    ### Functions for computing Attention
    def forward(self, x: torch.Tensor, pomt_info: SimpleNamespace) -> torch.Tensor:
        B, N, C = x.size()
        ### Emulate qkv matrix from TIMM VisionTransformer
        ### Code is taken from forward(...) of TIMM VisonTransformer Attention Block
        qkv_vectors = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv_vectors.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        metric = POMTMetric(attn, v, pomt_info)

        ### Dropout Layer
        attn = self.attn_drop(attn)

        ### Now finish the attention computation
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, metric


###
### Intended to be used with image classification tasks
### Function that computes masked attention according to the Dynamic ViT (2021 NEURIPS) paper in addition to a novel technique
### Intended to overwrite an Attention layer during training, copying all of its attributes but now with additional functionality
###
class TopKTIMMAttention(Attention):
    ### Functions for computing Attention
    def forward(self, x: torch.Tensor, topk_info: SimpleNamespace) -> torch.Tensor:
        B, N, C = x.size()
        ### Emulate qkv matrix from TIMM VisionTransformer
        ### Code is taken from forward(...) of TIMM VisonTransformer Attention Block
        qkv_vectors = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv_vectors.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        metric = TopKMetric(attn, topk_info)

        ### Dropout Layer
        attn = self.attn_drop(attn)

        ### Now finish the attention computation
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, metric


###
### TIMM Compatible Block for Transformer Models
###
class POMTAttentionBlock(Block):
    ###
    ### x will have shape (batch size, # tokens, features)
    ### token decision mask will have shape (batch size, # tokens, 1)
    ###
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn, metric = self.attn(self.norm1(x), self._pomt_info)
        x = x + self.drop_path1(self.ls1(attn))

        x, discard_indices = POMTForward(x, metric, self._pomt_info)
        self.pomt_discard_indices = discard_indices

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


###
### TIMM Compatible Block for Transformer Models
###
class TopKAttentionBlock(Block):
    ###
    ### x will have shape (batch size, # tokens, features)
    ### token decision mask will have shape (batch size, # tokens, 1)
    ###
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn, metric = self.attn(self.norm1(x), self._topk_info)
        x = x + self.drop_path1(self.ls1(attn))

        x = TopKForward(x, metric, self._topk_info)

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


###
### Wrapper Functions
### Heavily inspired by Token Merging apply_patch(...) and model wrapping technique
###
def make_pomt_class(vit : torch.nn.Module):
    class POMTVisionTransformer(vit.__class__):
        def forward(self, x: torch.Tensor) -> Tuple:
            if isinstance(self.r, list):
                assert(len(self.r) == len(self.blocks))
                self._pomt_info.r = copy.deepcopy(self.r)
            elif isinstance(self.r, int):
                self._pomt_info.r = [self.r] * len(self.blocks)
            else:
                raise AssertionError(f"Improper r type {type(self.r)}")
            
            return super().forward(x)

    ### Return
    return POMTVisionTransformer


def make_topk_class(vit : torch.nn.Module):
    class TopKVisionTransformer(vit.__class__):
        def forward(self, x: torch.Tensor) -> Tuple:
            if isinstance(self.r, list):
                assert(len(self.r) == len(self.blocks))
                self._topk_info.r = copy.deepcopy(self.r)
            elif isinstance(self.r, int):
                self._topk_info.r = [self.r] * len(self.blocks)
            else:
                raise AssertionError(f"Improper r type {type(self.r)}")
            
            return super().forward(x)

    ### Return
    return TopKVisionTransformer


###
### Patch with either POMT or Top-K
###
def timm_apply_pomt_patch(args : Namespace, vit : torch.nn.Module) -> torch.nn.Module:
    ### Generate class
    POMTVisionTransformerClass = make_pomt_class(vit)
    vit.__class__ = POMTVisionTransformerClass

    ### Store metadata for our data
    vit._pomt_info = SimpleNamespace(
        r = 0,
        prefix_tokens=vit.num_prefix_tokens,
    )

    for module in vit.modules():
        if isinstance(module, Block):
            module.__class__ = POMTAttentionBlock
            module._pomt_info = vit._pomt_info
            module.pomt_discard_indices = None
        if isinstance(module, Attention):
            module.__class__ = POMTAttention

    ### Set R
    ### Single layer pruning
    r = [0] * len(vit.blocks)
    r[args.pomt_prune_layer_index] = args.pomt_R
    vit.r = args.r

    return vit


def timm_apply_topk_patch(args : Namespace, vit : torch.nn.Module) -> torch.nn.Module:
    ### Generate class
    TopKVisionTransformerClass = make_topk_class(vit)
    vit.__class__ = TopKVisionTransformerClass

    ### Store metadata for our data
    vit._topk_info = SimpleNamespace(
        r = 0,
        prefix_tokens=vit.num_prefix_tokens,
    )

    for module in vit.modules():
        if isinstance(module, Block):
            module.__class__ = TopKAttentionBlock
            module._topk_info = vit._topk_info
        if isinstance(module, Attention):
            module.__class__ = TopKTIMMAttention

    ### Set r
    r = [args.topk_R] * len(vit.blocks)
    vit.r = r

    return vit
