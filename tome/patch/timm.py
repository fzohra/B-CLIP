# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------


from typing import Tuple

import torch
from timm.models.vision_transformer import Attention, Block, VisionTransformer

from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r
from entmax import entmax15, entmax_bisect
import os

class ToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        model_dtype = self.attn.qkv.weight.dtype
        x_attn, metric = self.attn(self.norm1(x).to(model_dtype), attn_size)
        x = x + self._drop_path1(x_attn)
        x = x + self._drop_path2(self.mlp(self.norm2(x).to(model_dtype)))
        return x


class ToMeAttention(Attention):
    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, N, C = x.shape
        H = self.num_heads
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if self._save_attn:
            self.attn_probs = attn
            if attn.requires_grad:
                attn.retain_grad()

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)
    
def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer


def apply_patch(
    model: VisionTransformer, trace_source: bool = False, prop_attn: bool = True, 
    attn_fn="softmax", alpha=1.0, save_attn=False,
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None: # TODO: remove this
        model._tome_info["distill_token"] = True
    
    for idx, block in enumerate(model.blocks):
        block.__class__ = ToMeBlock
        block._tome_info = model._tome_info
        attn = block.attn
        attn.__class__ = ToMeAttention

        attn._save_attn = save_attn