import os, json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import random

import torch.distributed as dist

import wandb
import utils

import datasets

# ─────────────── CLIP norm stats ───────────────
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# ─────────────── Distributed helpers ───────────────
def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def is_main_process():
    return get_rank() == 0

def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

def dist_sum_tensor(t: torch.Tensor):
    """All-reduce sum; no-op if DDP not initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t

def allreduce_sum_int(x: int) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.tensor([x], device=device, dtype=torch.long)
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())

# ─────────────── Helper functions ───────────────
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

# ─────────────── Text embeddings ───────────────
@torch.no_grad()
def text_embeds(tokenizer, texts, model, device):
    """Get text embeddings for a list of texts"""
    if isinstance(texts, str):
        texts = [texts]
    
    ids = tokenizer(texts).to(device, non_blocking=True)
    x = model.encode_text(ids)
    x = x / x.norm(p=2, dim=-1, keepdim=True)
    return x

# ─────────────── Evaluation functions ───────────────
@torch.no_grad()
def eval_sharegpt4v_retrieval(val_loader, model, tokenizer, device):
    """
    Evaluate image-text retrieval on ShareGPT4V validation set.
    In distributed mode, each GPU processes its assigned samples and computes local accuracy.
    Returns (local_correct_t2i, local_correct_i2t, local_total) for distributed aggregation.
    """
    m = unwrap_model(model)
    
    local_correct_t2i = 0  # Text-to-Image accuracy
    local_correct_i2t = 0  # Image-to-Text accuracy
    local_total = 0
    
    
    iterator = tqdm(val_loader, desc="Urban1K CLS") if is_main_process() else val_loader
    
    for batch_idx, batch in enumerate(iterator):
        batch = _to_cuda(batch, device, non_blocking=True)
        images = batch[0]
        captions = batch[1]
        
        batch_size = images.size(0)
        local_total += batch_size
                
        # Get image features for all images at once
        image_features = m.encode_image(images)
        if image_features.dim() == 3:
            image_features = image_features[:, 0, :]  # CLS token 
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [batch_size, D]

        # Get text features for all captions at once
        text_features = m.encode_text(captions)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # [batch_size, D]

        # gather features from all GPUs
        (text_features_all, image_features_all) = \
            utils.all_gather_batch([text_features, image_features])

        # Compute similarity matrix: each text against all images globally
        # similarity[i, j] = similarity between text i and image j
        similarity_t2i = m.logit_scale.exp() * text_features @ image_features_all.T  # [batch_size, total_samples] - Text-to-Image
        similarity_i2t = m.logit_scale.exp() * image_features @ text_features_all.T  # [batch_size, total_samples] - Image-to-Text
        
        # Text-to-Image (T2I): For each text, find the image with highest similarity globally
        max_similarities_t2i, max_indices_t2i = similarity_t2i.max(dim=1)
        
        # Image-to-Text (I2T): For each image, find the text with highest similarity globally
        max_similarities_i2t, max_indices_i2t = similarity_i2t.max(dim=1)
        
        # Calculate correct global indices for this batch
        rank = get_rank()
        world_size = get_world_size()
        samples_per_gpu = 1000 // world_size
        remaining_samples = 1000 % world_size
        global_start_idx = rank * samples_per_gpu + min(rank, remaining_samples)
        batch_start_idx = global_start_idx + (batch_idx * batch_size)
        batch_indices = torch.arange(batch_start_idx, batch_start_idx + batch_size, device=device)
 
        # Text-to-Image (T2I): Count correct matches (text i should match with its corresponding image globally)
        correct_t2i = (max_indices_t2i == batch_indices).sum().item()
        local_correct_t2i += correct_t2i
        
        correct_i2t = (max_indices_i2t == batch_indices).sum().item()
        local_correct_i2t += correct_i2t
    
    return local_correct_t2i, local_correct_i2t, local_total

@torch.no_grad()
def eval_sharegpt4v_retrieval_tci(val_loader, model, tokenizer, device):
    """
    Evaluate text-conditioned image retrieval on ShareGPT4V validation set.
    Each GPU processes its assigned samples and computes local accuracy for both CLS and TCI.
    Returns (local_correct_tci, local_total, local_correct_cls) for distributed aggregation.
    """
    m = unwrap_model(model)
    
    local_correct_tci_t2i = 0  # TCI Text-to-Image
    local_correct_tci_i2t = 0  # TCI Image-to-Text
    local_correct_cls_t2i = 0  # CLS Text-to-Image
    local_correct_cls_i2t = 0  # CLS Image-to-Text
    local_total = 0
        
    iterator = tqdm(val_loader, desc="Urban1K TCI/CLS") if is_main_process() else val_loader
    
    for batch_idx, batch in enumerate(iterator):
        batch = _to_cuda(batch, device, non_blocking=True)
        images = batch[0]
        # captions = batch[1]

        batch_size = images.size(0)
        local_total += batch_size
        
        logits_per_cls_i2t, logits_per_cls_t2i, logits_per_tci_i2t, logits_per_tci_t2i = m.get_sim_logits_conditioned(
            *batch, # images, captions
            use_model_text_settings=False,
            use_text_eos=True,    
            return_avg_tci=False,
            return_first=True,
            is_global=True,  # Use global evaluation for accurate results
            is_inference=True
        )
                
        rank = get_rank()
        world_size = get_world_size()
        samples_per_gpu = 1000 // world_size
        remaining_samples = 1000 % world_size
        global_start_idx = rank * samples_per_gpu + min(rank, remaining_samples)
        batch_start_idx = global_start_idx + (batch_idx * batch_size)
        batch_indices = torch.arange(batch_start_idx, batch_start_idx + batch_size, device=device)

        # TCI Text-to-Image (T2I): each text, find the TCI feature with highest similarity
        _, max_indices_tci_t2i = logits_per_tci_t2i.max(dim=1)
        correct_tci_t2i = (max_indices_tci_t2i == batch_indices).sum().item()
        local_correct_tci_t2i += correct_tci_t2i
        
        # TCI Image-to-Text (I2T): each image, find the TCI feature with highest similarity
        _, max_indices_tci_i2t = logits_per_tci_i2t.max(dim=1)
        correct_tci_i2t = (max_indices_tci_i2t == batch_indices).sum().item()
        local_correct_tci_i2t += correct_tci_i2t
        
        # CLS Text-to-Image (T2I): each text, find the CLS feature with highest similarity
        _, max_indices_cls_t2i = logits_per_cls_t2i.max(dim=1)
        correct_cls_t2i = (max_indices_cls_t2i == batch_indices).sum().item()
        local_correct_cls_t2i += correct_cls_t2i
        
        # CLS Image-to-Text (I2T): each image, find the CLS feature with highest similarity
        _, max_indices_cls_i2t = logits_per_cls_i2t.max(dim=1)
        correct_cls_i2t = (max_indices_cls_i2t == batch_indices).sum().item()
        local_correct_cls_i2t += correct_cls_i2t
    
    return local_correct_tci_t2i, local_correct_tci_i2t, local_correct_cls_t2i, local_correct_cls_i2t, local_total

def validate_urban1k(model, output_dir, epoch, step, log_wandb, tokenizer, device='cuda'):
    model.eval()
    m = unwrap_model(model)
    
    image_root = 'data/Urban1k/image/'
    caption_root = 'data/Urban1k/caption/'
    
    img_res = 224
    val_transform = T.Compose([
        T.Resize(img_res, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_res),
        T.ToTensor(),
        T.Normalize(_CLIP_MEAN, _CLIP_STD),
    ])
    
    val_dataset = datasets.urban1k_val_dataset(
        image_root=image_root,
        caption_root=caption_root,
        transform=val_transform,
        tokenizer=tokenizer,
        use_text_concepts=getattr(m, 'use_text_concepts', False),
        use_text_tokens=getattr(m, 'use_text_tokens', False),
    )
    
    if is_dist():
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        print(f"Rank {get_rank()}: Created DistributedSampler, dataset size: {len(val_dataset)}", flush=True)
    else:
        val_sampler = None
        print(f"Rank {get_rank()}: No distributed sampler (single GPU mode)", flush=True)
    
    g = torch.Generator().manual_seed(42)  # Fixed seed for validation
    
    if getattr(m, 'use_text_conditioned_patches', False) and getattr(m, 'use_text_phrases', False):
        from main_clip_ft import collate_with_concepts
        collate_fn = lambda b: collate_with_concepts(b, pad_id=0, pad_side="right", max_concepts=100, max_concept_context_length=50)
    else:
        collate_fn = None
    
    # For TCI evaluation, distribute across GPUs but use smaller batches to avoid memory issues
    world_size = get_world_size() if is_dist() else 1
    dataset_size = len(val_dataset)
    batch_size = dataset_size // world_size
        
    if is_dist():
        samples_per_gpu = dataset_size // world_size
        remaining_samples = dataset_size % world_size
        rank_start = get_rank() * samples_per_gpu + min(get_rank(), remaining_samples)
        rank_end = rank_start + samples_per_gpu + (1 if get_rank() < remaining_samples else 0)
    
    # Synchronize all processes before starting evaluation
    if is_dist():
        dist.barrier()
    
    # Ensure all ranks have the same sampler state
    if is_dist() and val_sampler is not None:
        val_sampler.set_epoch(0)  # Set epoch to ensure consistent sharding
        
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with distributed
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn
    )
    
    results = {}
    
    if getattr(m, 'use_text_conditioned_patches', False):
        local_correct_tci_t2i, local_correct_tci_i2t, local_correct_cls_t2i, local_correct_cls_i2t, local_total_tci = eval_sharegpt4v_retrieval_tci(val_loader, model, tokenizer, device)
        # Use distributed aggregation since all GPUs contribute
        global_correct_tci_t2i = allreduce_sum_int(local_correct_tci_t2i)
        global_correct_tci_i2t = allreduce_sum_int(local_correct_tci_i2t)
        global_correct_cls_t2i = allreduce_sum_int(local_correct_cls_t2i)
        global_correct_cls_i2t = allreduce_sum_int(local_correct_cls_i2t)
        global_total_tci = allreduce_sum_int(local_total_tci)
        
        retrieval_acc_tci_t2i = 100.0 * (global_correct_tci_t2i / max(1, global_total_tci))
        retrieval_acc_tci_i2t = 100.0 * (global_correct_tci_i2t / max(1, global_total_tci))
        retrieval_acc_cls_t2i = 100.0 * (global_correct_cls_t2i / max(1, global_total_tci))
        retrieval_acc_cls_i2t = 100.0 * (global_correct_cls_i2t / max(1, global_total_tci))
        
        results['retrieval_acc_tci_t2i'] = retrieval_acc_tci_t2i
        results['retrieval_acc_tci_i2t'] = retrieval_acc_tci_i2t
        results['retrieval_acc_cls_t2i'] = retrieval_acc_cls_t2i
        results['retrieval_acc_cls_i2t'] = retrieval_acc_cls_i2t
    else:
        local_correct_t2i, local_correct_i2t, local_total = eval_sharegpt4v_retrieval(val_loader, model, tokenizer, device)
        global_correct_cls_t2i = allreduce_sum_int(local_correct_t2i)
        global_correct_cls_i2t = allreduce_sum_int(local_correct_i2t)
        global_total_tci = allreduce_sum_int(local_total)
        
        retrieval_acc_t2i = 100.0 * (global_correct_cls_t2i / max(1, global_total_tci))
        retrieval_acc_i2t = 100.0 * (global_correct_cls_i2t / max(1, global_total_tci))
        
        results['retrieval_acc_cls_t2i'] = retrieval_acc_t2i
        results['retrieval_acc_cls_i2t'] = retrieval_acc_i2t
    
    # Only rank 0 writes/logs results
    if is_main_process():
        print("\nUrban1k Validation Results:")
        if 'retrieval_acc_tci_t2i' in results:
            print(f"  TCI Text-to-Image (T2I) accuracy: {results['retrieval_acc_tci_t2i']:.2f}% ({global_correct_tci_t2i}/{global_total_tci})")
        if 'retrieval_acc_tci_i2t' in results:
            print(f"  TCI Image-to-Text (I2T) accuracy: {results['retrieval_acc_tci_i2t']:.2f}% ({global_correct_tci_i2t}/{global_total_tci})")
        if 'retrieval_acc_cls_t2i' in results:
            print(f"  CLS Text-to-Image (T2I) accuracy: {results['retrieval_acc_cls_t2i']:.2f}% ({global_correct_cls_t2i}/{global_total_tci})")
        if 'retrieval_acc_cls_i2t' in results:
            print(f"  CLS Image-to-Text (I2T) accuracy: {results['retrieval_acc_cls_i2t']:.2f}% ({global_correct_cls_i2t}/{global_total_tci})")

        # Save results to file
        results_dir = 'evaluation_urban1k/'
        results_subdir = os.path.split(output_dir)[-1]
        ckpt_path = f'{output_dir}/checkpoint_{epoch+1}.pt'
        
        fname = "urban1k_retrieval.json"
        epoch_subdir = 'epoch_' + str(epoch+1)
        results_f = Path(results_dir) / results_subdir / epoch_subdir / fname
        os.makedirs(results_f.parent, exist_ok=True)
        
        to_write = {
            "checkpoint": str(ckpt_path),
            "dataset": "urban1k",
            "total_samples": 1000,
            "results": {k: round(v, 4) for k, v in results.items()},
        }
        
        with open(results_f, "w") as f:
            json.dump(to_write, f, indent=2)
        
        # Log to WandB if enabled
        if log_wandb:
            wandb_payload = {}
            if 'retrieval_acc_tci_t2i' in results:
                wandb_payload["urban1k_retrieval_tci_t2i"] = results['retrieval_acc_tci_t2i']
            if 'retrieval_acc_tci_i2t' in results:
                wandb_payload["urban1k_retrieval_tci_i2t"] = results['retrieval_acc_tci_i2t']
            if 'retrieval_acc_cls_t2i' in results:
                wandb_payload["urban1k_retrieval_cls_t2i"] = results['retrieval_acc_cls_t2i']
            if 'retrieval_acc_cls_i2t' in results:
                wandb_payload["urban1k_retrieval_cls_i2t"] = results['retrieval_acc_cls_i2t']
            
            if wandb_payload:
                wandb.log(wandb_payload, step=step)
    
    # Optional: sync here so non-main ranks don't exit too early
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    
    return results
