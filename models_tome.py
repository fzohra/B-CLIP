from collections import OrderedDict

import numpy as np
import timm
import torch
from torch import nn
import torch.nn.functional as F
from entmax import entmax15, entmax_bisect
import math

import losses
import tome
import utils
import numpy as np
from torch.utils.checkpoint import checkpoint

def forward_last_block_cls_full_patch_vonly(vision, x, full_prefix_tokens: int = 1):
    blk = vision.blocks[-1]
    B, N, C = x.shape
    H = blk.attn.num_heads
    Dh = C // H
    scale = getattr(blk.attn, "scale", Dh ** -0.5)

    # ----- 1) pre-norm -----
    residual = x
    xn = blk.norm1(x)  # (B, N, C)

    # ----- 2) slice q/k/v from fused qkv -----
    Wqkv, bqkv = blk.attn.qkv.weight, blk.attn.qkv.bias
    Cslice = Wqkv.shape[0] // 3
    Wq, bq = Wqkv[:Cslice],      (bqkv[:Cslice]      if bqkv is not None else None)
    Wk, bk = Wqkv[Cslice:2*Cslice], (bqkv[Cslice:2*Cslice] if bqkv is not None else None)
    Wv, bv = Wqkv[2*Cslice:],    (bqkv[2*Cslice:]    if bqkv is not None else None)

    # K, V for all tokens
    k = F.linear(xn, Wk, bk).contiguous()  # (B, N, C)
    v = F.linear(xn, Wv, bv).contiguous()  # (B, N, C)
    k = k.view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)
    v = v.view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)

    # Q only for prefix tokens (typically CLS at index 0, or multiple prefixes if present)
    P = min(full_prefix_tokens, N)
    q_full = F.linear(xn[:, :P, :], Wq, bq).contiguous()    # (B, P, C)
    q_full = q_full.view(B, P, H, Dh).transpose(1, 2)       # (B, H, P, Dh)

    # ----- 3) attention for prefix/CLS tokens only -----
    attn = (q_full * scale) @ k.transpose(-2, -1)           # (B, H, P, N)
    attn = attn.softmax(dim=-1)
    attn = blk.attn.attn_drop(attn) if hasattr(blk.attn, "attn_drop") else attn
    cls_ctx = attn @ v                                      # (B, H, P, Dh)
    cls_ctx = cls_ctx.transpose(1, 2).reshape(B, P, C)      # (B, P, C)
    cls_ctx = blk.attn.proj(cls_ctx)                        # (B, P, C)

    # ----- 4) value-only outputs for *all* tokens, then overwrite the prefix with full-attn -----
    vo = v.transpose(1, 2).reshape(B, N, C)                 # (B, N, C) == per-token V, no mixing
    out = blk.attn.proj(vo)                                 # (B, N, C)
    # out[:, :P, :] = cls_ctx                                 # overwrite CLS/prefix with full-attn result
    out = torch.cat([cls_ctx, out[:, P:, :]], dim=1)
    out = blk.attn.proj_drop(out) if hasattr(blk.attn, "proj_drop") else out

    x = residual + out

    # ----- 5) feed-forward sub-block -----
    residual = x
    x = blk.norm2(x)
    x = blk.mlp(x)
    x = residual + x
    return x


def encode_image_mixed_last_block(vision, proj, px, full_prefix_tokens: int = 1):
    x = vision.patch_embed(px)
    x = vision._pos_embed(x)
    x = vision.patch_drop(x)
    x = vision.norm_pre(x)
    for blk in vision.blocks[:-1]:
        x = blk(x)
    x = forward_last_block_cls_full_patch_vonly(vision, x, full_prefix_tokens)
    x = vision.norm(x)
    x = x @ proj
    return x

def encode_image_intermediate_block(vision, proj, px, full_prefix_tokens: int = 1):
    x = vision.patch_embed(px)
    x = vision._pos_embed(x)
    x = vision.patch_drop(x)
    x = vision.norm_pre(x)
    for blk in vision.blocks[:-1]:
        x = blk(x)

    cls_ctx = x[:, :full_prefix_tokens, :]
    cls_ctx =vision.blocks[-1](cls_ctx)
    x = torch.cat([cls_ctx, x[:, full_prefix_tokens:, :]], dim=1)
    x = vision.norm(x)
    x = x @ proj
    return x

class AttentionPoolingMLPBlock(nn.Module):
    def __init__(
            self,
            embed_dim: int = 512,
            n_head: int = 8,
            use_layer_norm: bool = False,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_head, kdim=embed_dim, vdim=embed_dim, 
                                          batch_first=True,
                                          add_zero_attn=False)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln_q = LayerNorm(embed_dim)
            self.ln_k = LayerNorm(embed_dim)
            self.ln_v = LayerNorm(embed_dim)

        self.ln_2 = LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            OrderedDict([
                ("c_fc", nn.Linear(embed_dim, embed_dim * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(embed_dim * 4, embed_dim)),
            ])
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, kv_embed, q_embed, need_weights=False, attn_mask=None):
        if self.use_layer_norm:
            q = self.ln_q(q_embed)
            k = self.ln_k(kv_embed)
            v = self.ln_v(kv_embed)
        else:
            q = q_embed
            k = kv_embed
            v = kv_embed
            
        if need_weights:
            out, attn_weights = self.attn(q, k, v, need_weights=True, average_attn_weights=True, attn_mask=attn_mask)
            out = out + self.mlp(self.ln_2(out))
            return out, attn_weights
        else:
            out = self.attn(q, k, v, need_weights=False, attn_mask=attn_mask)[0]
            out = out + self.mlp(self.ln_2(out))
            return out, None

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Attention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor = None,
        attn_fn: str = "softmax",
        attn_fn_alpha: float = 1.0,
        save_attn: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert (
            self.head_dim * n_head == d_model
        ), "d_model must be divisible by n_head"
        self.scale = self.head_dim ** -0.5
        self.save_attn = save_attn
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_mask = attn_mask

    def forward(self, x: torch.Tensor):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        seq_len, batch, _ = x.shape
        def reshape(z):
            z = z.view(seq_len, batch, self.n_head, self.head_dim)
            return z.permute(1, 2, 0, 3)

        q, k, v = map(reshape, (q, k, v))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
    
        if self.attn_mask is not None:
            mask = self.attn_mask.to(dtype=scores.dtype, device=scores.device)
            scores = scores + mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)

        attn_probs = F.softmax(scores, dim=-1)

        if self.save_attn:
            self.attn_probs = attn_probs
            if attn_probs.requires_grad:
                attn_probs.retain_grad()
            
        attn_out = torch.matmul(attn_probs, v)  # (batch, head, seq, head_dim)

        attn_out = attn_out.permute(2, 0, 1, 3).contiguous()
        attn_out = attn_out.view(seq_len, batch, self.d_model)

        return self.out_proj(attn_out)
    
class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor = None,
        attn_fn: str = "softmax",
        attn_fn_alpha: float = 1.0,
        save_attn: bool = False,
    ):
        super().__init__()
        self.attn = Attention(
            d_model=d_model,
            n_head=n_head,
            attn_mask=attn_mask,
            save_attn=save_attn,
        )
        self.ln_1 = LayerNorm(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(d_model * 4, d_model)),
            ])
        )
        
    def forward(self, x: torch.Tensor):
        """
        x: (seq_len, batch, d_model)
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, 
                 attn_fn: str = "softmax", attn_fn_alpha: float = 1.0, attn_fn_sparse_layers_text: list = None,
                 save_attn: bool = False, grad_checkpointing: bool = True, use_reentrant: bool = False):
        
        super().__init__()
        resblocks = []
        for i in range(layers):
            block = ResidualAttentionBlock(
                width,
                heads,
                attn_mask,
                save_attn=save_attn,
            )
            resblocks.append(block)

        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(resblocks)
        self.grad_checkpointing = grad_checkpointing
        self.use_reentrant = use_reentrant

    def forward(self, x: torch.Tensor):
        if self.grad_checkpointing and self.training:
            for blk in self.resblocks:
                x = checkpoint(blk, x, use_reentrant=self.use_reentrant)
            return x
        else:
            for blk in self.resblocks:
                x = blk(x)
            return x

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 attn_fn: str = 'softmax',
                 attn_fn_alpha: float = 1.0,
                 attn_fn_sparse_layers_text: list = None,
                 save_attn: bool = False,
                 global_pool: str = 'token',
                 use_last_block: bool = False,
                 use_intermediate_block: bool = False,
                 text_conditioning_mode: str = 'attn_pooling_mlp',
                 use_caption: bool = True,
                 use_text_conditioned_patches: bool = False,
                 use_text_concepts: bool = False,
                 use_text_tokens: bool = False,
                 use_text_eos: bool = False,
                 use_caption_in_eos: bool = False,
                 use_text_conditioned_cls: bool = False,
                 use_tci_diversity: bool = False,
                 tci_div_separate: bool = False,
                 use_diversity_hinge: bool = False,
                 use_negative_text_conditioning: bool = False,
                 conditioning_negatives_agg_type: str = None,
                 detach_vision_for_tcil: bool = False,
                 num_tc_negative_eos_samples: int = 0,
                 num_tc_negative_concept_samples: int = 0,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width

        self.visual = vision_model

        self.global_pool = global_pool
        self.use_last_block = use_last_block
        self.use_intermediate_block = use_intermediate_block
        
        self.use_caption = use_caption

        self.use_text_conditioned_patches = use_text_conditioned_patches
        self.use_text_concepts = use_text_concepts
        self.use_text_tokens = use_text_tokens
        self.use_text_eos = use_text_eos
        self.use_caption_in_eos = use_caption_in_eos

        self.use_text_conditioned_cls = use_text_conditioned_cls
        self.use_negative_text_conditioning = use_negative_text_conditioning
        self.conditioning_negatives_agg_type = conditioning_negatives_agg_type
        self.num_tc_negative_eos_samples = num_tc_negative_eos_samples
        self.num_tc_negative_concept_samples = num_tc_negative_concept_samples
        self.detach_vision_for_tcil = detach_vision_for_tcil
        
        if self.use_text_conditioned_patches:
            assert self.use_text_tokens or self.use_text_eos or self.use_text_concepts, "use_text_tokens or use_text_eos or use_text_concepts must be True when use_text_conditioned_patches is True"
        
        if self.use_text_conditioned_patches:
            if text_conditioning_mode == 'attn_pooling_mlp':
                self.text_conditioned_patches_block = AttentionPoolingMLPBlock(embed_dim=embed_dim)


        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            attn_fn=attn_fn,
            attn_fn_alpha=attn_fn_alpha,
            attn_fn_sparse_layers_text=attn_fn_sparse_layers_text,
            save_attn=save_attn,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)

        self.positional_embedding = nn.Parameter(torch.empty(77, transformer_width))
        
        self.ln_final = LayerNorm(transformer_width)

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.use_tci_diversity = use_tci_diversity
        self.tci_div_separate = tci_div_separate
        self.use_diversity_hinge = use_diversity_hinge
        if self.use_tci_diversity and not self.use_diversity_hinge:
            if self.tci_div_separate:
                self.logit_scale_tci_diversity_eos = nn.Parameter(torch.ones([]) * np.log(1 / 0.2))
                self.logit_scale_tci_diversity_concepts = nn.Parameter(torch.ones([]) * np.log(1 / 0.2))
            else:
                self.logit_scale_tci_diversity = nn.Parameter(torch.ones([]) * np.log(1 / 0.2))

        self.initialize_parameters()
        if self.context_length == 248:
            self.mask1 = torch.zeros([self.context_length, 1])
            self.mask1[:20, :] = 1
            self.mask2 = torch.zeros([self.context_length, 1])
            self.mask2[20:, :] = 1

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)    

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.q_proj.weight, std=attn_std)
            nn.init.normal_(block.attn.k_proj.weight, std=attn_std)
            nn.init.normal_(block.attn.v_proj.weight, std=attn_std)

            if block.attn.q_proj.bias is not None:
                nn.init.zeros_(block.attn.q_proj.bias)
                nn.init.zeros_(block.attn.k_proj.bias)
                nn.init.zeros_(block.attn.v_proj.bias)

            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            if block.attn.out_proj.bias is not None:
                nn.init.zeros_(block.attn.out_proj.bias)

            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def convert_state_dict(self, old_sd):
        new_sd = {}
        for k, v in old_sd.items():
            # Skip conversion for text_conditioned_patches_block - it uses standard nn.MultiheadAttention
            if "text_conditioned_patches_block" in k:
                new_sd[k] = v
                continue
                
            # for original clip and slip-based models
            if k.endswith("attn.in_proj_weight"):
                # split [3*D, D] → [D, D] x3
                q_w, k_w, v_w = v.chunk(3, dim=0)
                base = k[:-len("attn.in_proj_weight")]
                new_sd[base + "attn.q_proj.weight"] = q_w
                new_sd[base + "attn.k_proj.weight"] = k_w
                new_sd[base + "attn.v_proj.weight"] = v_w
            elif k.endswith("attn.in_proj_bias"):
                q_b, k_b, v_b = v.chunk(3, dim=0)
                base = k[:-len("attn.in_proj_bias")]
                new_sd[base + "attn.q_proj.bias"] = q_b
                new_sd[base + "attn.k_proj.bias"] = k_b
                new_sd[base + "attn.v_proj.bias"] = v_b
            else:
                # everything else stays the same
                new_sd[k] = v
        return new_sd

    def convert_state_dict_from_openai(self, old_sd):
        new_sd = {}
        for k, v in old_sd.items():
            if k == "visual.proj":
                # OpenAI CLIP stores visual.proj as (vision_width, embed_dim) which matches our model
                # Model expects (vision_width, embed_dim) for x @ image_projection
                expected_shape = (self.vision_width, self.image_projection.shape[1])  # (vision_width, embed_dim)
                if v.shape == expected_shape:
                    new_sd["image_projection"] = v
                elif v.shape == (expected_shape[1], expected_shape[0]):
                    new_sd["image_projection"] = v.T
                else:
                    new_sd["image_projection"] = v  # Load anyway, but it will fail in load_state_dict
            elif k == "text_projection":
                expected_shape = (self.transformer.width, self.text_projection.shape[1])  # (transformer_width, embed_dim)
                if len(v.shape) == 2:
                    if v.shape == expected_shape:
                        new_sd["text_projection"] = v
                    elif v.shape == (expected_shape[1], expected_shape[0]):
                        new_sd["text_projection"] = v.T
                    else:
                        new_sd["text_projection"] = v  # Load anyway, but it will fail in load_state_dict
                else:
                    new_sd["text_projection"] = v
            else:
                if not k.startswith("visual."):
                    new_sd[k] = v
        
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in new_sd:
                del new_sd[key]

        return new_sd

    def resize_text_pos_embed(self):
        if self.context_length != 248:
            return

        dtype = self.positional_embedding.dtype

        positional_embedding_pre = self.positional_embedding.type(dtype)
            
        length, dim = positional_embedding_pre.shape
        keep_len = 20
        expected_new_length = 4*length - 3*keep_len  # Should be 248
        posisitonal_embedding_new = torch.zeros([expected_new_length, dim], dtype=dtype)
        
        print(f"Resizing positional embedding: {length} -> {expected_new_length} (context_length={self.context_length})")
        assert expected_new_length == self.context_length, f"Resize calculation error: {expected_new_length} != {self.context_length}"

        for i in range(keep_len):
            posisitonal_embedding_new[i] = positional_embedding_pre[i]
        for i in range(length-1-keep_len):
            posisitonal_embedding_new[4*i + keep_len] = positional_embedding_pre[i + keep_len]
            posisitonal_embedding_new[4*i + 1 + keep_len] = 3*positional_embedding_pre[i + keep_len]/4 + 1*positional_embedding_pre[i+1+keep_len]/4
            posisitonal_embedding_new[4*i + 2+keep_len] = 2*positional_embedding_pre[i+keep_len]/4 + 2*positional_embedding_pre[i+1+keep_len]/4
            posisitonal_embedding_new[4*i + 3+keep_len] = 1*positional_embedding_pre[i+keep_len]/4 + 3*positional_embedding_pre[i+1+keep_len]/4

        posisitonal_embedding_new[4*length -3*keep_len - 4] = positional_embedding_pre[length-1] + 0*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 3] = positional_embedding_pre[length-1] + 1*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 2] = positional_embedding_pre[length-1] + 2*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
        posisitonal_embedding_new[4*length -3*keep_len - 1] = positional_embedding_pre[length-1] + 3*(positional_embedding_pre[length-1] - positional_embedding_pre[length-2])/4
                
        positional_embedding_res = posisitonal_embedding_new.clone()
                
        self.positional_embedding = nn.Parameter(posisitonal_embedding_new, requires_grad=False)
        self.positional_embedding_res = nn.Parameter(positional_embedding_res, requires_grad=True)
    
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image):
        x = self.visual(image)
        x = x @ self.image_projection
        return x

    def encode_image_by_block(self, image):
        if self.use_last_block:
            image_embed = self.encode_image(image)
        elif self.use_intermediate_block:
            image_embed = encode_image_intermediate_block(self.visual, self.image_projection, image)
        else:
            image_embed = encode_image_mixed_last_block(self.visual, self.image_projection, image) # [B, P+1, D]
        return image_embed

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        if self.context_length == 248:
            max_len = x.shape[1]
            x = x + (self.positional_embedding[:max_len].to(x.device) * self.mask1[:max_len].to(x.device)).type(x.dtype).to(x.device) + (self.positional_embedding_res[:max_len].to(x.device) * self.mask2[:max_len].to(x.device)).type(x.dtype).to(x.device)
        else:
            max_len = x.shape[1]
            x = x + self.positional_embedding[:max_len]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        x = x @ self.text_projection
        return x

    def forward(self, image, text, concepts=None, concepts_mask=None, positive_mask=None, logging=None, tokenizer=None):
        if self.global_pool == 'token':
            image_embed = self.encode_image(image)
            text_embed = self.encode_text(text)

            return {'image_embed': image_embed,
                    'text_embed': text_embed,
                    'logit_scale': self.logit_scale.exp()}

        return self.forward_patches_then_condition(image, text, concepts=concepts, concepts_padding_mask=concepts_mask, positive_mask=positive_mask, logging=logging, tokenizer=tokenizer)

    def forward_patches_then_condition(self, image, text, concepts=None, concepts_padding_mask=None, positive_mask=None, logging=None, tokenizer=None, use_text_concepts=False, use_text_eos=False, use_model_text_settings=True, is_inference=False):
        image_embed = self.encode_image_by_block(image)

        if text.dim() == 3:
            # Multiple positive sentences per image: [B, num_pos, L]
            B, num_pos, L = text.shape # num pos can include +1 for eos
            text_flat = text.contiguous().view(B * num_pos, L)
            positive_text_embed = self.encode_text(text_flat)  # [B*num_pos, D]
            positive_text_embed = positive_text_embed.view(B, num_pos, -1)  # [B, num_pos, D] - keep all
            if self.use_caption:
                eos_embed = positive_text_embed[:, 0, :] # [B, D]
            else:
                eos_embed = None
        else:
            # Single text per image: [B, L]
            num_pos = 1
            eos_embed = self.encode_text(text)  # [B, D]
            positive_text_embed = eos_embed.unsqueeze(1) # [B, 1, D]
            positive_mask = torch.ones(eos_embed.size(0), 1, dtype=torch.bool, device=eos_embed.device) # [B, 1]

        outputs = {
            'eos_embed': eos_embed, # B D
            'cls_embed': image_embed[:, 0],
            'logit_scale': self.logit_scale.exp(),
            'positive_mask': positive_mask[:, 1:] if self.use_caption else positive_mask,
            'mps_embed': positive_text_embed[:, 1:, :] if self.use_caption else positive_text_embed # remove cls for caption if present
        }

        if self.use_text_conditioned_patches:
            text_embed_1, text_mask_1 = None, None  # eos + mps
            text_embed_2, text_mask_2 = None, None  # concepts
            
                
            if (use_model_text_settings and self.use_text_eos) or use_text_eos:
                if (self.use_caption and not self.use_caption_in_eos) and not is_inference: # use_caption = data includes caption
                    num_pos -= 1
                    positive_text_embed = positive_text_embed[:, 1:, :] # remove caption for eos
                    positive_mask = positive_mask[:, 1:] # remove caption for eos
                text_embed_1 = positive_text_embed # [B, num_pos, D]
                text_mask_1 = positive_mask # [B, num_pos]

            if ((use_model_text_settings) and (self.use_text_concepts or self.use_text_tokens)) or use_text_concepts:
                assert concepts is not None and concepts.shape[0] == image.shape[0]
                B, K, L = concepts.shape
                concepts_eos_embed = self.encode_text(concepts.contiguous().view(B*K, L)).contiguous().view(B, K, -1)  # (B, K, D)
                text_embed_2 = concepts_eos_embed  # [B, K, D]
                text_mask_2 = concepts_padding_mask

            if text_embed_1 is not None and text_embed_2 is not None:
                text_embed = torch.cat([text_embed_1, text_embed_2], dim=1)  # [B, K+num_pos, D]
                text_mask = torch.cat([text_mask_1, text_mask_2], dim=1)     # [B, K+num_pos]
            elif text_embed_1 is not None:
                text_embed = text_embed_1
                text_mask = text_mask_1
            elif text_embed_2 is not None:
                text_embed = text_embed_2
                text_mask = text_mask_2

            text_embed = F.normalize(text_embed, dim=-1, p=2)
            
            if self.use_text_conditioned_cls:
                img_embed_to_attn_pool = image_embed # [B, P+1, D] patches + cls
            else:
                img_embed_to_attn_pool = image_embed[:, 1:] # [B, P, D] patches only
            
            img_embed_to_attn_pool = F.normalize(img_embed_to_attn_pool, dim=-1, p=2)
            
            if self.detach_vision_for_tcil and not is_inference and self.training:
                img_embed_to_attn_pool = img_embed_to_attn_pool.detach()

            if self.use_text_conditioned_patches:
                positive_image_embed, _ = self.text_conditioned_patches_block(
                    img_embed_to_attn_pool, text_embed, need_weights=False
                )  # [B, K+num_pos, D]
                
                outputs.update({
                    'positive_image_embed': positive_image_embed,  # [B, K+num_pos, D]
                    'positive_text_embed': text_embed,  # [B, K+num_pos, D]
                    'positive_text_mask': text_mask, # [B, K+num_pos]
                    'logit_scale_tc': self.text_conditioned_patches_block.logit_scale.exp(),
                    'logit_scale_tci_diversity': self.logit_scale_tci_diversity.exp() if self.use_tci_diversity and not self.tci_div_separate and not self.use_diversity_hinge else None,
                    'logit_scale_tci_diversity_eos': self.logit_scale_tci_diversity_eos.exp() if self.use_tci_diversity and self.tci_div_separate and not self.use_diversity_hinge else None,
                    'logit_scale_tci_diversity_concepts': self.logit_scale_tci_diversity_concepts.exp() if self.use_tci_diversity and self.tci_div_separate and not self.use_diversity_hinge else None,
                })

                if not is_inference and self.use_negative_text_conditioning: #inter-sample hard conditioned negatives
                    B, K, D = text_embed.shape   
                    start_idx_sentences = 1 if self.use_caption_in_eos else 0      
                    if self.conditioning_negatives_agg_type == 'caption+sentences+concepts':
                        text_embed_neg = text_embed[:, 0, :].unsqueeze(1)
                        text_embed_neg_1 = text_embed[:, start_idx_sentences: start_idx_sentences + self.num_tc_negative_eos_samples, :]
                        text_embed_neg_2 = text_embed[:, num_pos:num_pos + self.num_tc_negative_concept_samples, :]
                        text_embed_neg = torch.cat([text_embed_neg, text_embed_neg_1, text_embed_neg_2], dim=1)

                    diagonal_mask = torch.eye(B, dtype=torch.bool, device=text_embed.device)  # [B, B]
                    off_diagonal_mask = ~diagonal_mask

                    _, selected_K, _ = text_embed_neg.shape
                    text_embed_neg =  text_embed_neg.unsqueeze(0).expand(B, B, selected_K, D)  # [B, B, K+num_pos, D]
                    text_embed_neg = text_embed_neg[off_diagonal_mask].view(B, (B-1)*selected_K, D) # [B, (B-1)*(K+num_pos), D]

                    negative_image_embed, _ = self.text_conditioned_patches_block(
                        img_embed_to_attn_pool, text_embed_neg, need_weights=False
                    )  # [B, (B-1)*(K+num_pos), D]                   
                    outputs.update({
                        'negative_image_embed': negative_image_embed,  # [B, (B-1)*(K+num_pos), D]
                        'negative_text_embed': text_embed_neg,  # [B, (B-1)*(K+num_pos), D]
                    })

        return outputs

    def get_sim_logits_conditioned(self, image, text, concepts=None, concepts_padding_mask=None, positive_mask=None, logging=None, tokenizer=None, use_text_concepts=False, use_text_eos=False, use_model_text_settings=True, return_avg_tci=False, return_first=False, is_global=False, is_inference=False):
            
        outputs = self.forward_patches_then_condition(image, text, concepts=concepts, concepts_padding_mask=concepts_padding_mask, positive_mask=positive_mask, logging=logging, tokenizer=tokenizer, use_text_concepts=use_text_concepts, use_text_eos=use_text_eos, use_model_text_settings=use_model_text_settings, is_inference=is_inference)

        cls_embed = outputs['cls_embed']
        eos_embed = outputs['eos_embed']
        logit_scale = outputs['logit_scale']

        cls_embed = F.normalize(cls_embed, dim=-1, p=2)
        eos_embed = F.normalize(eos_embed, dim=-1, p=2)

        if is_global:
            (eos_embed_all, cls_embed_all) = utils.all_gather_batch([eos_embed.contiguous(), cls_embed.contiguous()]) # [B_total, D]
        else:
            eos_embed_all = eos_embed
            cls_embed_all = cls_embed

        if self.use_text_conditioned_patches:
            positive_image_embed = outputs['positive_image_embed'] # [B, K, D] or [B, 1, D]
            text_embed = outputs['positive_text_embed'] # [B, K, D] or [B, 1, D]
            logit_scale_tc = outputs['logit_scale_tc']

            positive_image_embed = F.normalize(positive_image_embed, dim=-1, p=2)
            text_embed = F.normalize(text_embed, dim=-1, p=2)
            
            if return_avg_tci:
                text_embed = text_embed.mean(dim=1) # [B_total, D]
                positive_image_embed = positive_image_embed.mean(dim=1) # [B_total, D]
            elif return_first:
                text_embed = text_embed[:, 0, :]
                positive_image_embed = positive_image_embed[:, 0, :]
                
            if is_global:
                (text_embed_all, positive_image_embed_all) = utils.all_gather_batch([text_embed.contiguous(), positive_image_embed.contiguous()]) # [B_total, D]
            else:
                text_embed_all = text_embed
                positive_image_embed_all = positive_image_embed

        logits_per_cls_i2t = logit_scale * cls_embed @ eos_embed_all.t() # [B, B]
        logits_per_cls_t2i = logit_scale * eos_embed @ cls_embed_all.t() # [B, B]

        if self.use_text_conditioned_patches:
            if return_avg_tci or return_first:
                logits_per_tci_i2t = logit_scale_tc * positive_image_embed @ text_embed_all.t() # [B, B]
                logits_per_tci_t2i = logit_scale_tc * text_embed @ positive_image_embed_all.t() # [B, B]
            else:
                logits_per_tci_i2t = logit_scale_tc * torch.einsum("bkd,bld->bkl", positive_image_embed, text_embed_all) # [B, K, K] or [B, 1, 1]
                logits_per_tci_t2i = logit_scale_tc * torch.einsum("bkd,bld->bkl", text_embed, positive_image_embed_all) # [B, K, K] or [B, 1, 1]
            
            return logits_per_cls_i2t, logits_per_cls_t2i, logits_per_tci_i2t, logits_per_tci_t2i
        else:
            return logits_per_cls_i2t, logits_per_cls_t2i, None, None

    def get_sim_logits(self, image, text):
        assert self.global_pool == 'token'

        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)
        logit_scale = self.logit_scale.exp()

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        
        # gather features from all GPUs
        image_embed_all, text_embed_all = \
            utils.all_gather_batch([image_embed, text_embed])

        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()

        return logits_per_image, logits_per_text

def get_loss(model, fg_loss_fn=None, use_tcl=False, tcil_alpha=1.0, cls_alpha=1.0, mps_alpha=1.0, use_negative_text_conditioning=False, tcil_loss_mode="k_positives_ce", use_softmax_for_multi_positives=False, num_caption_tokens=1, num_eos_tokens=0, num_concept_tokens=0, tci_div_separate=False, tci_diversity_alpha=0.5, use_diversity_hinge=False, alpha=1.0, beta=0.0):
    if model.startswith('CLIP'):
        if fg_loss_fn is None:
            return losses.CLIPLoss()

        fg_loss_fn_list = fg_loss_fn.split('+')

        keyword_map = {
            "cls": "use_cls_loss",
            "tcil": "use_tcil_loss",
            "mps": "use_mps_loss",
            "div": "use_tci_diversity_loss",
        }

        kwargs = {}
        for word, arg_name in keyword_map.items():
            if word in fg_loss_fn_list:
                kwargs[arg_name] = True

        kwargs['cls_alpha'] = cls_alpha
        kwargs['tcil_alpha'] = tcil_alpha
        kwargs['mps_alpha'] = mps_alpha
        kwargs['tci_diversity_alpha'] = tci_diversity_alpha

        kwargs['alpha'] = alpha
        kwargs['beta'] = beta

        kwargs["tcil_loss_mode"] = tcil_loss_mode
        kwargs["use_softmax_for_multi_positives"] = use_softmax_for_multi_positives

        kwargs["num_caption_tokens"] = num_caption_tokens
        kwargs["num_eos_tokens"] = num_eos_tokens
        kwargs["num_concept_tokens"] = num_concept_tokens

        kwargs['use_negative_text_conditioning'] = use_negative_text_conditioning

        kwargs['tci_div_separate'] = tci_div_separate
        kwargs['use_diversity_hinge'] = use_diversity_hinge

        if use_tcl:
            assert ('tcil' in fg_loss_fn_list), "tcil loss must be included if use_tcl is True" # extra guard to ensure both the optimizer and the loss function are enabled

        return losses.TextConditionedImageLoss(**kwargs)


def get_metric_names(model, fg_loss_fn=None, use_negative_text_conditioning=False):
    if model.startswith('CLIP'):
        if fg_loss_fn is None:
            return ['loss', 'cls_eos_loss', 'cls_eos_acc'] 

        metrics = ['loss']
        fg_loss_fn_list = fg_loss_fn.split('+')

        if 'cls' in fg_loss_fn_list:
            metrics.append('cls_eos_loss')
            metrics.append('cls_eos_acc')

        if 'tcil' in fg_loss_fn_list:
            metrics.append('text_conditioned_image_loss')
            metrics.append('text_conditioned_image_acc')
            metrics.append('tci_diversity_eos')
            metrics.append('tci_diversity_concepts')
            metrics.append('tci_diversity')
        
        if use_negative_text_conditioning:
            metrics.append('tcil_negatives_loss')
            
        if 'div' in fg_loss_fn_list:
            metrics.append('tci_diversity_loss')

        if 'mps' in fg_loss_fn_list:
            metrics.append('mps_loss')
            metrics.append('mps_acc')
        return metrics


def CLIP_VITB16(**kwargs):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0, dynamic_img_size=kwargs.get('dynamic_img_size', False))
    tome.patch.timm(vision_model, attn_fn=kwargs['attn_fn'], alpha=kwargs['attn_fn_alpha'], attn_fn_sparse_layers=kwargs['attn_fn_sparse_layers_vision'])
    model = CLIP(embed_dim=512, vision_width=768, vision_model=vision_model, context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model

def CLIP_VITB16_OPENAI(**kwargs):
    vision_model = timm.create_model(
        'vit_base_patch16_clip_224.openai',
        cache_dir='pretrained/',
        pretrained=True,
        num_classes=0,
        act_layer=QuickGELU,                        # exact activation OpenAI used
        norm_layer=lambda d: LayerNorm(d, eps=1e-5),# exact eps OpenAI used             
        dynamic_img_size=kwargs.get('dynamic_img_size', False),
        global_pool=kwargs['global_pool'],
    )
    vision_model.set_grad_checkpointing(True)
    tome.patch.timm(vision_model, attn_fn=kwargs['attn_fn'], alpha=kwargs['attn_fn_alpha'])

    model = CLIP(embed_dim=512, vision_width=768, vision_model=vision_model, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=12, **kwargs)

    return model

def CLIP_VITL14_OPENAI(**kwargs):
    vision_model = timm.create_model(
        'vit_large_patch14_clip_224.openai',
        cache_dir='pretrained/',
        pretrained=True,
        num_classes=0,
        act_layer=QuickGELU,                        # exact activation OpenAI used
        norm_layer=lambda d: LayerNorm(d, eps=1e-5),# exact eps OpenAI used             
        dynamic_img_size=kwargs.get('dynamic_img_size', False),
        global_pool=kwargs['global_pool'],
    )
    vision_model.set_grad_checkpointing(True)
    tome.patch.timm(vision_model, attn_fn=kwargs['attn_fn'], alpha=kwargs['attn_fn_alpha'])
    model = CLIP(embed_dim=768, vision_width=1024, vision_model=vision_model, vocab_size=49408, transformer_width=768, transformer_heads=12, transformer_layers=12, **kwargs)

    return model