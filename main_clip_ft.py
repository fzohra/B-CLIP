import argparse
from collections import OrderedDict
import json
import math
import sys
import time
import wandb
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["NVIDIA_TF32_OVERRIDE"]="0"
os.environ["NCCL_DEBUG"]="INFO"
os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"
os.environ["PYTHONFAULTHANDLER"]="1"
# Add NCCL timeout and stability configurations
os.environ["NCCL_BLOCKING_WAIT"]="1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"]="1"
os.environ["NCCL_TIMEOUT"]="1800"  # 30 minutes timeout
os.environ["TORCH_NCCL_BLOCKING_WAIT"]="1"


import numpy as np
import torch

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

import datasets
import models_tome as models
from tokenizer import SimpleTokenizer
import utils
import random
from torch.utils.data._utils.collate import default_collate
from validate_fgovd_distributed import validate_fgovd
from validate_sharegpt4v_distributed import validate_sharegpt4v
from validate_urban1k_distributed import validate_urban1k

def set_global_seed(seed):
    random.seed(seed)              # Python RNG
    np.random.seed(seed)           # NumPy RNG
    torch.manual_seed(seed)        # CPU + current CUDA device
    torch.cuda.manual_seed_all(seed)


def get_args_parser():
    parser = argparse.ArgumentParser(description='Beta-CLIP training and evaluation', add_help=False)
    # Data
    parser.add_argument('--dataset', default='yfcc15m', type=str, choices=['cc3m+datacomp@0.2,0.8', 'yfcc15m', 'cc3m', 'cc12m', 'coco', 'redcaps', 'datacomp', 'sharegpt4v'])
    parser.add_argument('--root', default='', type=str,
                        help='path to dataset root')
    parser.add_argument('--metadata', default='yfcc15m.pkl', type=str,
                        help='path to metadata file (see README for details)')
    parser.add_argument('--output-dir', default='./ckpts/pretraining/', type=str, help='output dir')
    # Model
    parser.add_argument('--model', default='CLIP_VITB16', type=str)
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    
    # conditioning params                  
    parser.add_argument('--text-conditioning-mode', type=str, default='sim_pooling', choices=['sim_pooling', 'attn_pooling', 'attn_pooling_mlp'])

    # caption + mps vs mps only
    parser.add_argument('--use-caption-in-eos', action='store_true', default=True)
    parser.add_argument('--no-use-caption-in-eos', dest='use_caption_in_eos', action='store_false', help='disable caption usage in eos')

    parser.add_argument('--use-caption', action='store_true', default=True) # data includes caption
    parser.add_argument('--no-use-caption', dest='use_caption', action='store_false', help='disable caption usage')

    # choice of text conditioning configs
    parser.add_argument('--use-text-conditioned-patches', action='store_true')
    parser.add_argument('--use-text-concepts', action='store_true')
    parser.add_argument('--use-text-tokens', action='store_true')
    parser.add_argument('--use-text-eos', action='store_true')
    parser.add_argument('--use-multi-scale-caption', action='store_true')

    parser.add_argument('--use-text-conditioned-cls', action='store_true')
    parser.add_argument('--max-concept-context-length', type=int, default=50)
    parser.add_argument('--max-concepts', type=int, default=100)
    parser.add_argument('--max-captions', type=int, default=-1)


    # negative conditioning configs
    parser.add_argument('--use-negative-text-conditioning', action='store_true',
                        help='Use negative text conditioning. For concepts, will average across K for negatives.')
    parser.add_argument('--conditioning-negatives-agg-type', type=str, default=None, choices=['all', 'caption', 'sentences', 'concepts', 'sentences+concepts', 'caption+sentences', 'caption+sentences+concepts'])
    parser.add_argument('--num-tc-negative-eos-samples', type=int, default=0)
    parser.add_argument('--num-tc-negative-concept-samples', type=int, default=0)
    
    # image feature extraction
    parser.add_argument('--use-last-block', action='store_true')
    parser.add_argument('--use-intermediate-block', action='store_true')
    parser.add_argument('--global-pool', type=str, default='none', choices=['token', ''], help='should the vision encoder return patches or just the cls token')

    # tcil loss
    parser.add_argument('--fg-loss-fn', type=str, default=None)
    parser.add_argument('--use-tcl', action='store_true')
    parser.add_argument('--cls-alpha', type=float, default=1.0)
    parser.add_argument('--tcil-alpha', type=float, default=1.0)
    parser.add_argument('--mps-alpha', type=float, default=1.0)
    parser.add_argument('--detach-vision-for-tcil', action='store_true',
                        help='Detach vision features for TCIL to force attention block to learn and prevent gradient bypass')
    parser.add_argument('--use-layer-norm', action='store_true')

    # tcil loss mode
    parser.add_argument('--tcil-loss-mode', type=str, default="1_positive", choices=["k_positives_ce", "k_positives_bce", "k_positives_neg_ce"])
    parser.add_argument('--use-softmax-for-multi-positives', action='store_true')
    
    parser.add_argument('--use-tci-diversity', action='store_true',
                        help='Enable intra-caption TCI diversity')
    parser.add_argument('--tci-div-separate', action='store_true',
                        help='Use separate diversity losses for sentences vs concepts (default: unified diversity)')
    parser.add_argument('--tci-diversity-alpha', type=float, default=0.5,
                        help='Weight for unified diversity component (used when --tci-div-separate is False)')
    parser.add_argument('--use-diversity-hinge', action='store_true',
                        help='Use diversity hinge loss')

    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.0)
    # caption configs
    parser.add_argument('--caption-type', type=str, default='original', choices=['concepts', 'original'])
    parser.add_argument('--caption-len', type=int, default=0)
    parser.add_argument('--num_positive_samples', type=int, default=0)
    
    
    # Training
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--warmup-epochs', default=1.0, type=float)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=64, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
                    
    # training configs for conditioning
    parser.add_argument('--context_length', type=int, default=77, choices=[77, 248])
    parser.add_argument('--lr-conditioner', type=float, default=3e-3)
    parser.add_argument('--lr-conditioner-end', type=float, default=1e-4)

    parser.add_argument('--clip-grad-norm', default=None, type=float,
                        help='clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-grad-vision', default=None, type=float,
                        help='clip gradient norm for vision encoder separately (default: None)')
    parser.add_argument('--clip-grad-text', default=None, type=float,
                        help='clip gradient norm for text encoder separately (default: None)')
    parser.add_argument('--clip-grad-attention', default=None, type=float,
                        help='clip gradient norm for attention pooling separately (default: None)')
    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')

    #entmax experiments
    parser.add_argument('--attn-fn', default='softmax', type=str,
                        help='attention function: softmax or entmax')
    parser.add_argument('--attn-fn-alpha', default=1.0, type=float,
                        help='alpha value if using entmax')
    parser.add_argument('--attn_fn_sparse_layers_vision', type=int, nargs='+', default=[],)
    parser.add_argument('--attn_fn_sparse_layers_text', type=int, nargs='+', default=[],)

    parser.add_argument('--use-bf16', action='store_true')
    return parser

best_acc1 = 0
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def _clone_cpu(x):
    if isinstance(x, torch.Tensor):
        # fresh, contiguous, resizable CPU storage
        return x.detach().cpu().contiguous().clone()
    return x

def _to_cuda(x, device, non_blocking=True):
    if isinstance(x, torch.Tensor):
        return x.cuda(device, non_blocking=non_blocking)
    if isinstance(x, (list, tuple)):
        t = type(x)
        return t(_to_cuda(y, device, non_blocking) for y in x)
    if isinstance(x, dict):
        return {k: _to_cuda(v, device, non_blocking) for k, v in x.items()}
    return x


def _max_nonpad_len_2d(x2d, pad_id=0):
    # x2d: [*, L]
    if x2d.numel() == 0:
        return 0
    # count non-pad tokens per row, then take the max across the batch/rows
    return int((x2d != pad_id).sum(dim=-1).max().item())

def _truncate_along_L(x, Lb, pad_side="right", eos_token=49407):
    if Lb <= 0 or x.size(-1) == Lb:
        return x
    
    # Truncate
    if pad_side == "right":
        result = x[..., :Lb].clone()
    else:
        result = x[..., -Lb:].clone()
    
    # Ensure last token is EOS when we actually truncated
    if result.size(-1) > 0:
        result[..., -1] = eos_token
    
    return result

def _pad_concepts(batch_concepts, pad_id=0, pad_side="right", max_concepts=None, max_concept_context_length=50):
    """
    batch_concepts: list of Tensors, each [K_i, L] or [L]
    Returns:
      concepts: [B, K_max, Lb]   (Lb is batch-wise max non-pad length across all concept rows)
      mask:     [B, K_max]       (True where a concept row is present)
    """
    cons, Ks = [], []
    # normalize shapes and optionally cap K
    for c in batch_concepts:
        c = _clone_cpu(c)
        if c.ndim == 1:
            c = c.unsqueeze(0)  # (L,) -> (1, L)
        if max_concepts is not None and c.size(0) > max_concepts:
            c = c[:max_concepts]
        cons.append(c)
        Ks.append(c.size(0))

    B = len(cons)
    K_max = max(Ks) if Ks else 0

    # compute batch max non-pad length along L across all concept rows
    if K_max == 0:
        concepts = torch.empty((B, 0, 0), dtype=torch.long)
        mask = torch.zeros((B, 0), dtype=torch.bool)
        return concepts, mask

    # stack row-wise to measure L (works even if L differs across items)
    # we just need the max non-pad length; do it per item then take max
    Lb_list = []
    for c in cons:
        Lb_list.append(_max_nonpad_len_2d(c, pad_id=pad_id))
    Lb = max(Lb_list) if Lb_list else 0
    Lb = min(Lb, max_concept_context_length)

    # allocate output at truncated Lb
    dtype = cons[0].dtype
    Lb = int(Lb)
    concepts = torch.full((B, K_max, Lb), pad_id, dtype=dtype) if Lb > 0 else torch.full((B, K_max, 0), pad_id, dtype=dtype)
    mask = torch.zeros((B, K_max), dtype=torch.bool)

    # fill
    for i, c in enumerate(cons):
        k = c.size(0)
        if k == 0:
            continue
        c_trunc = _truncate_along_L(c, Lb, pad_side=pad_side) if Lb > 0 else c[:, :0]
        concepts[i, :k, :c_trunc.size(-1)] = c_trunc
        mask[i, :k] = True

    return concepts, mask

def collate_with_concepts(
    batch,
    pad_id=0,
    pad_side="right",
    max_concepts=100,
    max_concept_context_length=30,
    max_captions=-1,
):
    if len(batch[0]) == 3:
        imgs, caps, cons = zip(*batch)
        has_concepts = True
    elif len(batch[0]) == 2:
        imgs, caps = zip(*batch)
        cons = None
        has_concepts = False
    else:
        raise ValueError(f"Expected batch items to have 2 or 3 elements, got {len(batch[0])}")

    images = default_collate([_clone_cpu(x) for x in imgs])

    first_cap = caps[0]
    positive_mask = None  # Mask to track which positives are real vs padded

    if first_cap.ndim == 2:
        # Multiple positives: each caption is [num_pos_i, L], need to pad and stack
        # Different samples may have different num_pos (if not enough sentences)
        B = len(caps)
        max_num_pos = max(x.shape[0] for x in caps)  # Find max number of positives
        if max_captions > -1:
            max_num_pos = min(max_num_pos, max_captions)
        L = caps[0].shape[1]  # Sequence length (should be same for all)
        
        # Create padded tensor and mask
        captions = torch.full((B, max_num_pos, L), pad_id, dtype=caps[0].dtype)
        positive_mask = torch.zeros((B, max_num_pos), dtype=torch.bool)

        for i, cap in enumerate(caps):
            if max_captions > -1 and cap.size(0) > max_num_pos:
                cap = cap[:max_num_pos]
            num_pos_i = cap.shape[0]
            captions[i, :num_pos_i, :] = cap.cpu() if cap.is_cuda else cap
            positive_mask[i, :num_pos_i] = True  # Mark valid positives
        # captions shape: [B, max_num_pos, L]
        # positive_mask shape: [B, max_num_pos]
    else:
        # Single caption: use default collate
        captions = default_collate([_clone_cpu(x) for x in caps])  # [B, L]

    if captions.ndim == 3:
        # Multiple positive sentences: [B, num_pos, L]
        B, num_pos, L = captions.shape
        captions_flat = captions.view(B * num_pos, L)
        Lb_caps = _max_nonpad_len_2d(captions_flat, pad_id=pad_id)
        captions_flat = _truncate_along_L(captions_flat, Lb_caps, pad_side=pad_side)
        captions = captions_flat.view(B, num_pos, -1)
    elif captions.ndim == 2:
        # Single caption per image: [B, L]
        Lb_caps = _max_nonpad_len_2d(captions, pad_id=pad_id)
        captions = _truncate_along_L(captions, Lb_caps, pad_side=pad_side)
    else:
        raise ValueError(f"Expected captions to be [B, L] or [B, num_pos, L], got {tuple(captions.shape)}")

    # Concepts: pad K, and truncate L to the batch max non-pad across all concept rows
    if has_concepts:
        concepts, concepts_mask = _pad_concepts(cons, pad_id=pad_id, pad_side=pad_side, 
                                                max_concepts=max_concepts, 
                                                max_concept_context_length=max_concept_context_length)
    else:
        B = images.shape[0]
        concepts = None
        concepts_mask = None

    return images, captions, concepts, concepts_mask, positive_mask  
    # [B, 3, H, W], 
    # [B, Lc] or [B, num_pos, Lc], 
    # [B, K, Lk]
    # [B, K]
    # [B, num_pos]

def main(args):
    utils.init_distributed_mode(args)

    if dist.get_rank() == 0:
        print("world_size:", dist.get_world_size(),
            "rank:", dist.get_rank(),
            "visible:", os.environ["CUDA_VISIBLE_DEVICES"])    
    
    global best_acc1

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    set_global_seed(seed)

    # create model
    print("=> creating model: {}".format(args.model))
        
    model = getattr(models, args.model)(
        attn_fn=args.attn_fn, 
        attn_fn_alpha=args.attn_fn_alpha, 
        global_pool=args.global_pool,
        use_last_block=args.use_last_block,
        use_intermediate_block=args.use_intermediate_block,
        text_conditioning_mode=args.text_conditioning_mode,
        use_text_conditioned_patches=args.use_text_conditioned_patches,
        use_text_concepts=args.use_text_concepts,
        use_text_tokens=args.use_text_tokens,
        use_text_eos=args.use_text_eos,
        use_caption_in_eos=args.use_caption_in_eos,
        use_text_conditioned_cls=args.use_text_conditioned_cls,
        use_negative_text_conditioning=args.use_negative_text_conditioning,
        conditioning_negatives_agg_type=args.conditioning_negatives_agg_type,
        detach_vision_for_tcil=args.detach_vision_for_tcil,
        context_length=args.context_length,
        num_tc_negative_eos_samples=args.num_tc_negative_eos_samples,
        num_tc_negative_concept_samples=args.num_tc_negative_concept_samples,
        use_tci_diversity = args.use_tci_diversity,
        tci_div_separate = args.tci_div_separate,
        use_diversity_hinge = args.use_diversity_hinge,
        use_caption = args.use_caption,
    )

    print(f"args.use_caption: {args.use_caption}")
    print(f"args.use_caption_in_eos: {args.use_caption_in_eos}")
    # Compute K components separately for diversity loss
    num_eos_tokens = 0
    num_concept_tokens = 0
    if args.use_caption:
        num_caption_tokens = 1 if args.use_caption_in_eos else 0
    else:
        num_caption_tokens = 0
    
    if args.use_text_eos:
        num_eos_tokens += num_caption_tokens # +caption tokens
        num_eos_tokens += args.num_positive_samples # + sentence tokens

    if args.use_text_concepts or args.use_text_tokens:
        num_concept_tokens = args.max_concepts

    print(f"num_caption_tokens: {num_caption_tokens}, num_eos_tokens: {num_eos_tokens}, num_concept_tokens: {num_concept_tokens}")
    criterion = models.get_loss(model = args.model, 
        fg_loss_fn = args.fg_loss_fn, 
        use_tcl = args.use_tcl, 
        tcil_alpha = args.tcil_alpha, 
        cls_alpha = args.cls_alpha,
        mps_alpha = args.mps_alpha,
        use_negative_text_conditioning = args.use_negative_text_conditioning,
        tcil_loss_mode = args.tcil_loss_mode,
        use_softmax_for_multi_positives = args.use_softmax_for_multi_positives,
        num_caption_tokens = num_caption_tokens,
        num_eos_tokens = num_eos_tokens,
        num_concept_tokens = num_concept_tokens,
        tci_div_separate = args.tci_div_separate,
        tci_diversity_alpha = args.tci_diversity_alpha,
        use_diversity_hinge = args.use_diversity_hinge,
        alpha = args.alpha,
        beta = args.beta,
        ).cuda(args.gpu)

    if args.resume and os.path.isfile(args.resume):
        print("=> loading resume checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if args.resume.endswith('ViT-L-14.pt') or args.resume.endswith('ViT-B-16.pt'):
            state_dict = checkpoint.state_dict()

            state_dict = model.convert_state_dict_from_openai(state_dict)
            state_dict = model.convert_state_dict(state_dict)
            
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f" → {len(missing)} missing keys (weights not loaded from checkpoint)")
            print(f" → {len(unexpected)} unexpected keys ignored")
            
            model_state = model.state_dict()

            model.resize_text_pos_embed() # after loading state_dict
            epoch = 0
            args.start_epoch = epoch
            best_acc1 = 0
            torch.cuda.empty_cache()
            print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, epoch))
        else:
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            if not args.evaluate:
                args.start_epoch = epoch
            
            state_dict = model.convert_state_dict(state_dict)
            model.resize_text_pos_embed() # important! before loading state_dict
            
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            print(f" → {len(missing)} visual weights kept from timm pretrained model")
            print(f" → {len(unexpected)} keys ignored (should be zero)")
                
            best_acc1 = checkpoint['best_acc1']
            torch.cuda.empty_cache()
            print("=> loaded resume checkpoint '{}' (epoch {})"
                .format(args.resume, epoch))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        
    model.cuda(args.gpu)
    
    if args.fg_loss_fn is not None and ('cls' not in args.fg_loss_fn and 'mps' not in args.fg_loss_fn):
        model.logit_scale.requires_grad = False

    num_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total_params = sum(1 for p in model.parameters())
    print(f"Model parameters: {total_params} total, {num_frozen} frozen")
    if num_frozen > 0:
        print("Frozen parameters:")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"  {name}")
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            bucket_cap_mb=200,
            find_unused_parameters=True,
            broadcast_buffers=False
        )
    model._set_static_graph()

    cudnn.benchmark = True

    # define optimizers and scaler (positional_embedding does not take grads if context_length is 248)
    p_wd, p_non_wd = [], []
    p_wd_conditioner, p_non_wd_conditioner = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: # must call base_model.resize_text_pos_embed() before this part
            print(f"frozen weights: {n}")
            continue
        if 'text_conditioned_patches_block' in n:
            if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
                p_non_wd_conditioner.append(p)
            else:
                p_wd_conditioner.append(p)
        else:
            if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)

    
    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                    eps=args.eps, weight_decay=args.wd)
    if args.use_tcl:
        optim_params_conditioner = [{"params": p_wd_conditioner, "weight_decay": args.wd},
                                    {"params": p_non_wd_conditioner, "weight_decay": 0}]

        optimizer_conditioner = torch.optim.AdamW(optim_params_conditioner, lr=args.lr_conditioner, betas=args.betas, eps=args.eps, weight_decay=0)
    else:
        optimizer_conditioner = None

    scaler = torch.amp.GradScaler(enabled=not args.disable_amp and not args.use_bf16)

    if not args.resume.endswith('ViT-L-14.pt') and not args.resume.endswith('ViT-B-16.pt'):
        optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
        if args.use_tcl:
            optimizer_conditioner.load_state_dict(checkpoint['optimizer_conditioner']) if 'optimizer_conditioner' in checkpoint else ()
        scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()

    tokenizer = SimpleTokenizer(context_length=args.context_length)
            
    input_resolution = utils.get_model(model).visual.patch_embed.img_size[-1]
    print(f"{args.model} input resolution: {input_resolution}")
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], # openai stats
                                        std=[0.26862954, 0.26130258, 0.27577711])
    
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_resolution, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize
        ])
    val_transform = transforms.Compose([
            transforms.Resize(input_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_resolution),
            transforms.ToTensor(),
            normalize
        ])

    train_dataset = datasets.get_dataset(train_transform, tokenizer, args)
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, 'dataset_catalog.json')) as f:
        root = json.load(f)['imagenet']['path']
    val_dataset = ImageFolder(os.path.join(root, 'val'), val_transform)

    # dist eval resamples data to pad uneven batch sizes
    # make sure num_samples = 0 mod num_gpus for exact acc
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, seed=seed)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, seed=seed)
    else:
        train_sampler = None
        val_sampler = None
    
    g = torch.Generator().manual_seed(seed)

    # Use custom collate when using text conditioning (concepts, tokens, or eos with multiple positives)
    use_custom_collate = (args.use_text_conditioned_patches and \
                         (args.use_text_concepts or args.use_text_tokens or (args.use_text_eos and args.num_positive_samples > 0))) or args.num_positive_samples > 0

    if use_custom_collate:
        collate_fn = (lambda b: collate_with_concepts(b, pad_id=0, pad_side="right", max_concepts=args.max_concepts, max_concept_context_length=args.max_concept_context_length, max_captions=args.max_captions))
    else:
        collate_fn = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=None)

    print("Dataset size (train):", len(train_dataset))
    print("World size:", dist.get_world_size() if dist.is_initialized() else 1)
    print("Batch size (per GPU):", args.batch_size)
    print("Update freq (grad accum):", args.update_freq)
    print("len(train_loader) [= num_batches]:", len(train_loader))
    print("iters_per_epoch:", len(train_loader) // args.update_freq)

    steps_per_epoch = math.ceil(len(train_loader) / args.update_freq)
    total_steps     = steps_per_epoch * args.epochs

    lr_schedule = utils.cosine_scheduler(
            args.lr,
            args.lr_end,
            args.epochs, steps_per_epoch,
            warmup_epochs=args.warmup_epochs,
            start_warmup_value=args.lr_start,
    )

    conditioner_lr_schedule = utils.cosine_scheduler(
        args.lr_conditioner,
        args.lr_conditioner_end,
        args.epochs, steps_per_epoch,
        warmup_epochs=0,
        start_warmup_value=0,
    )

    print(f"total steps: {total_steps}  steps per epoch: {steps_per_epoch}")
    print(f"lr schedule: {lr_schedule[:10]} ... {lr_schedule[-10:]}")  # print first and last 10 steps
    print(f"conditioner lr schedule: {conditioner_lr_schedule[:10]} ... {conditioner_lr_schedule[-10:]}")  # print first and last 10 steps
    assert len(lr_schedule) == total_steps
    
    if utils.is_main_process() and args.wandb:
        base = os.path.split(args.output_dir)[-1]
        suffix = wandb.util.generate_id()
        wandb_id = f"{base}-{suffix}"
        wandb.init(project='beta-clip-2', id=wandb_id, config=args, resume=False)

        
    print(args)
    torch.cuda.reset_peak_memory_stats()
    print("=> beginning training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if not args.evaluate:
            train_stats, it = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args, conditioner_lr_schedule=conditioner_lr_schedule, optimizer_conditioner=optimizer_conditioner, tokenizer=tokenizer)
        else:
            it = epoch

        if (epoch + 1) % args.eval_freq != 0:
            continue

        validate_fgovd(model, args.output_dir, epoch, it, args.wandb, tokenizer)
        validate_sharegpt4v(model, args.output_dir, epoch, it, args.wandb, tokenizer)
        validate_urban1k(model, args.output_dir, epoch, it, args.wandb, tokenizer)

        print(f"=> saving checkpoint")
        if epoch + 1 == 5 or epoch + 1 == 10:
            utils.save_on_master({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'optimizer_conditioner' : optimizer_conditioner.state_dict() if optimizer_conditioner is not None else None,
                    'scaler': scaler.state_dict(),
                    'best_acc1': best_acc1,
                    'args': args,
                }, True, args.output_dir, epoch + 1)

        if not args.evaluate:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch}
            if utils.is_main_process():
                with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')


def compute_gradient_norms(model, scaler=None):
    """Compute gradient norms for different parts of the model.
    
    Args:
        model: The model to compute gradient norms for
        scaler: GradScaler for mixed precision training (will unscale gradients if provided)
    """
    vision_grad_norm = 0.0
    attention_grad_norm = 0.0
    text_grad_norm = 0.0
    
    # Get the scale factor if using mixed precision
    scale = scaler.get_scale() if scaler is not None else 1.0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Unscale the gradient before computing norm
            param_norm = (param.grad.data.norm(2).item()) / scale
            
            if 'visual' in name or 'vision' in name:
                vision_grad_norm += param_norm ** 2
            elif 'text_conditioned_patches_block' in name:
                attention_grad_norm += param_norm ** 2
            elif 'transformer' in name or 'text' in name:
                text_grad_norm += param_norm ** 2
    
    vision_grad_norm = vision_grad_norm ** 0.5
    attention_grad_norm = attention_grad_norm ** 0.5
    text_grad_norm = text_grad_norm ** 0.5
    
    if attention_grad_norm > 0:
        ratio_vision_to_attention = vision_grad_norm / attention_grad_norm
    else:
        ratio_vision_to_attention = 0
        
    return {
        'vision_grad_norm': vision_grad_norm,
        'attention_grad_norm': attention_grad_norm,
        'text_grad_norm': text_grad_norm,
        'ratio_vision_to_attention': ratio_vision_to_attention,
        'grad_scale': scale
    }

def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args, conditioner_lr_schedule, optimizer_conditioner=None, tokenizer=None):

    batch_time = AverageMeter('Effective Batch Time (sec)', ':6.2f')
    data_time = AverageMeter('Data Loading Time (sec)', ':6.2f')
    mem = AverageMeter('Curr Mem (GB)', ':6.1f')
    peak_mem = AverageMeter('Peak Mem (GB)', ':6.1f')
    metric_names = models.get_metric_names(args.model, args.fg_loss_fn, args.use_negative_text_conditioning)
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, peak_mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    num_log_events = 5
    log_iters = np.unique(np.linspace(0, max(iters_per_epoch-1, 0), num=num_log_events, dtype=int)).tolist()
    logging = {}

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]

        if optimizer_conditioner is not None:
            for k, param_group in enumerate(optimizer_conditioner.param_groups):
                param_group['lr'] = conditioner_lr_schedule[it] # for conditioner

        inputs = _to_cuda(inputs, args.gpu, non_blocking=True)
             
        should_log = (data_iter in log_iters) and utils.is_main_process() and args.wandb
        logging['_should_log_patch_viz'] = bool(should_log)
        logging['_wandb_step'] = it
        logging['_viz_epoch'] = epoch

        autocast_dtype = torch.bfloat16 if args.use_bf16 else torch.float16
        with torch.amp.autocast('cuda', enabled=not args.disable_amp, dtype=autocast_dtype):

            outputs = model(*inputs, logging=logging, tokenizer=tokenizer)
            loss_dict = criterion(outputs)
            loss = loss_dict['loss']
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()
        if (data_iter + 1) % args.update_freq != 0:
            continue

        # Compute gradient norms BEFORE clipping (for monitoring)
        grad_norms = None
        if optim_iter % args.print_freq == 0:
            grad_norms = compute_gradient_norms(model, scaler)
            if utils.is_main_process():
                print(f"Gradient norms - Vision: {grad_norms['vision_grad_norm']:.4f}, "
                      f"Text: {grad_norms['text_grad_norm']:.4f}, "
                      f"AttentionPool: {grad_norms['attention_grad_norm']:.4f}, "
                      f"Ratio V/A: {grad_norms['ratio_vision_to_attention']:.2f}, "
                      f"Scale: {grad_norms['grad_scale']:.0f}")

        # Apply gradient clipping
        if args.clip_grad_norm is not None or args.clip_grad_vision is not None:
            # Unscale gradients before clipping (required for mixed precision)
            scaler.unscale_(optimizer)
            if optimizer_conditioner is not None:
                scaler.unscale_(optimizer_conditioner)
            
            if args.clip_grad_norm is not None:
                # Global clipping: clip all parameters together
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            else:
                # Per-component clipping: clip each component separately
                vision_params = [p for n, p in model.named_parameters() 
                               if ('visual' in n or 'vision' in n) and p.grad is not None]
                text_params = [p for n, p in model.named_parameters() 
                             if ('transformer' in n or 'token_embedding' in n or 'positional_embedding' in n or 'ln_final' in n) 
                             and 'text_conditioned' not in n and p.grad is not None]
                attention_params = [p for n, p in model.named_parameters() 
                                  if 'text_conditioned' in n and p.grad is not None]
                
                if args.clip_grad_vision is not None and len(vision_params) > 0:
                    torch.nn.utils.clip_grad_norm_(vision_params, args.clip_grad_vision)
                if args.clip_grad_text is not None and len(text_params) > 0:
                    torch.nn.utils.clip_grad_norm_(text_params, args.clip_grad_text)
                if args.clip_grad_attention is not None and len(attention_params) > 0:
                    torch.nn.utils.clip_grad_norm_(attention_params, args.clip_grad_attention)

        # compute gradient and do SGD step
        scaler.step(optimizer)
        if optimizer_conditioner is not None:
            scaler.step(optimizer_conditioner)  # for conditioner
        scaler.update()
        model.zero_grad(set_to_none=True)
  

        # clamp logit scale to [0, 100]
        utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        logit_scale = utils.get_model(model).logit_scale.exp().item()
        if optimizer_conditioner is not None:
            utils.get_model(model).text_conditioned_patches_block.logit_scale.data.clamp_(0, 4.6052)
            logit_scale_conditioner = utils.get_model(model).text_conditioned_patches_block.logit_scale.exp().item()
        
        logit_scale_tci_diversity_eos = None
        logit_scale_tci_diversity_concepts = None
        logit_scale_tci_diversity = None
        if args.use_tci_diversity and not args.use_diversity_hinge:
            if args.tci_div_separate:
                logit_scale_tci_diversity_eos = utils.get_model(model).logit_scale_tci_diversity_eos.exp().item()
                logit_scale_tci_diversity_concepts = utils.get_model(model).logit_scale_tci_diversity_concepts.exp().item()
            else:
                logit_scale_tci_diversity = utils.get_model(model).logit_scale_tci_diversity.exp().item()


        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.memory_allocated() // 1e9)
        peak_mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if utils.is_main_process() and args.wandb:
                wandb_log_dict = {**{k: v.item() for k, v in loss_dict.items()},
                        'scaler': scaler.get_scale(),
                        'lr': optimizer.param_groups[0]['lr'],
                        'lr_conditioner': optimizer_conditioner.param_groups[0]['lr'] if optimizer_conditioner is not None else None,
                        'logit': logit_scale,
                        'logit_scale_conditioner': logit_scale_conditioner if optimizer_conditioner is not None else None,
                        'logit_scale_tci_diversity': logit_scale_tci_diversity,
                        'logit_scale_tci_diversity_eos': logit_scale_tci_diversity_eos,
                        'logit_scale_tci_diversity_concepts': logit_scale_tci_diversity_concepts}
                
                # Add gradient norms if they were computed
                if grad_norms is not None:
                    wandb_log_dict.update(grad_norms)
                
                wandb.log(wandb_log_dict, step=it)
            progress.display(optim_iter)

    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'lr_conditioner': optimizer_conditioner.param_groups[0]['lr'] if optimizer_conditioner is not None else None,
            'logit_scale': logit_scale,
            'logit_scale_conditioner': logit_scale_conditioner if optimizer_conditioner is not None else None,
            'logit_scale_tci_diversity': logit_scale_tci_diversity,
            'logit_scale_tci_diversity_eos': logit_scale_tci_diversity_eos,
            'logit_scale_tci_diversity_concepts': logit_scale_tci_diversity_concepts}, it

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Beta-CLIP training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
