#!/usr/bin/env python3
"""

Call as follows from main_clip_ft.py with the instatitated model
if utils.is_main_process():
    parent_dir, _ = os.path.split(args.resume)
    run_name = os.path.basename(parent_dir)
    visualize_curated_pairs(model, pairs=None, output_dir=os.path.join('visualizations/', run_name), device='cuda', alpha=0.6)
"""

import os

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange
import torchvision.transforms as transforms


def min_max(logits):
    """Min-max normalize logits."""
    B, num_prompt = logits.shape[:2]
    logits_min = logits.reshape(B, num_prompt, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
    logits_max = logits.reshape(B, num_prompt, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    logits = (logits - logits_min) / (logits_max - logits_min)
    return logits

def get_sim_logits_patches(model, image, text):

    with torch.no_grad():
        image_embed = model.encode_image_by_block(image)[:,1:]
        text_embed = model.encode_text(text)
        print(image_embed.shape)
        print(text_embed.shape)
        logit_scale = model.logit_scale.exp()

        logits_per_patches_i2t = logit_scale * image_embed @ text_embed.t() # [B, P, T]

    return logits_per_patches_i2t


def visualize_patches(model, image, texts, save_path=None, alpha=0.6, device='cuda'):
    """
    Visualize patch-wise attention for text queries.
    
    Args:
        model: CLIP model with get_sim_logits_patches method
        image: PIL Image or path to image
        texts: String or list of strings
        save_path: Where to save (None = display only)
        alpha: Overlay transparency
        device: cuda or cpu
        
    Returns:
        logits_per_patches: Patch-wise similarities [B, num_patches, num_texts]
    """
    # Unwrap DDP if needed
    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    # Load image
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    # Preprocess
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        lambda x: x.convert('RGB'),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Tokenize text
    if isinstance(texts, str):
        texts = [texts]
    
    from tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer(context_length=248)
    text_tokens = tokenizer(texts).to(device).unsqueeze(0)
    
    model.eval()

    with torch.no_grad():
        print(image_tensor.shape)
        print(text_tokens.shape)
        logits_per_patches_i2t = get_sim_logits_patches(model, image_tensor, text_tokens)
    
    print(logits_per_patches_i2t.shape)
    # Rearrange from [B, P, C] to [B, C, H, W]
    logits_per_patches = rearrange(logits_per_patches_i2t, 'b (h w) c -> b c h w', h=14, w=14)
    
    # Interpolate to image size
    logits_per_patches = F.interpolate(logits_per_patches, size=(224, 224), mode='bilinear')
    
    # Min-max normalize
    logits_per_patches = min_max(logits_per_patches)
    
    # Visualize
    _visualize_heatmaps_logits_only(image_tensor[0], texts, logits_per_patches[0], save_path, alpha)
    
    return logits_per_patches

def _visualize_heatmaps_logits_only(image_tensor, texts, logits, save_path, alpha):
    OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)

    image_unnorm = (image_tensor.cpu() * torch.Tensor(OPENAI_STD)[:, None, None]) + \
                   torch.Tensor(OPENAI_MEAN)[:, None, None]
    image_pil = Image.fromarray((image_unnorm.permute(1, 2, 0).numpy() * 255).astype('uint8'))

    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    logits_np = logits.detach().cpu().numpy()
    logits_np = (logits_np * 255).astype('uint8')

    n_texts = len(texts)
    fig, axes = plt.subplots(2, n_texts, figsize=(5 * n_texts, 10))
    if n_texts == 1:
        axes = axes.reshape(2, 1)

    overlays = []
    for i, (logit, text) in enumerate(zip(logits_np, texts)):
        heatmap = cv2.applyColorMap(logit, cv2.COLORMAP_JET)

        overlay = (1 - alpha) * img_cv + alpha * heatmap
        overlay = cv2.cvtColor(overlay.astype('uint8'), cv2.COLOR_BGR2RGB)
        overlays.append(overlay)

        axes[0, i].imshow(image_pil)
        axes[0, i].set_title('Original', fontsize=12)
        axes[0, i].axis('off')

        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f'"{text}"', fontsize=12)
        axes[1, i].axis('off')

    plt.tight_layout()

    if save_path:
        base, ext = os.path.splitext(save_path)
        ext = ext or '.png'
        name = os.path.basename(save_path)
        original_path = f'visualizations_final_bird/originals/{name}.png'
        os.makedirs(os.path.dirname(original_path), exist_ok=True)
        image_pil.save(original_path)

        for idx, overlay in enumerate(overlays):
            suffix = f'_{idx}' if len(overlays) > 1 else ''
            overlay_path = f'{base}{suffix}{ext}'
            Image.fromarray(overlay).save(overlay_path)

    plt.show()

def _visualize_heatmaps(image_tensor, texts, logits, save_path, alpha):
    """
    Internal function to create heatmap visualizations.
    
    Args:
        image_tensor: Preprocessed image tensor [3, H, W]
        texts: List of text strings
        logits: Patch similarities [num_texts, H, W]
        save_path: Where to save
        alpha: Overlay transparency
    """
    # Denormalize image
    OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)
    
    image_unnorm = (image_tensor.cpu() * torch.Tensor(OPENAI_STD)[:, None, None]) + \
                   torch.Tensor(OPENAI_MEAN)[:, None, None]
    image_pil = Image.fromarray((image_unnorm.permute(1, 2, 0).numpy() * 255).astype('uint8'))
    
    # Convert to CV2 format
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Convert logits to numpy
    logits_np = logits.detach().cpu().numpy()
    logits_np = (logits_np * 255).astype('uint8')
    
    # Create heatmaps for each text
    n_texts = len(texts)
    fig, axes = plt.subplots(2, n_texts, figsize=(5 * n_texts, 10))
    if n_texts == 1:
        axes = axes.reshape(2, 1)
    
    for i, (logit, text) in enumerate(zip(logits_np, texts)):
        # Apply colormap
        heatmap = cv2.applyColorMap(logit, cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = (1 - alpha) * img_cv + alpha * heatmap
        overlay = cv2.cvtColor(overlay.astype('uint8'), cv2.COLOR_BGR2RGB)
        
        # Plot original
        axes[0, i].imshow(image_pil)
        axes[0, i].set_title(f'Original', fontsize=12)
        axes[0, i].axis('off')
        
        # Plot overlay
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f'"{text}"', fontsize=12)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to: {save_path}")
    
    plt.show()


def visualize_single_text(model, image, text, save_path=None, alpha=0.6, device='cuda'):
    """
    Simple visualization for single text query.
    Shows only original and overlay side-by-side.
    
    Args:
        model: CLIP model
        image: PIL Image or path
        text: Text string
        save_path: Where to save
        alpha: Overlay transparency
        device: cuda or cpu
    """
    # Unwrap DDP
    if hasattr(model, 'module'):
        model = model.module
    
    # Load image
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    # Preprocess
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        lambda x: x.convert('RGB'),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Tokenize
    from tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer(context_length=248)
    text_tokens = tokenizer([text]).to(device).unsqueeze(0)
    
    # Get logits
    model.eval()
    with torch.no_grad():
        logits_per_patches_i2t, logits_per_patches_t2i = get_sim_logits_patches(model, image_tensor, text_tokens)
    
    # Rearrange and interpolate
    logits = rearrange(logits_per_patches_i2t, 'b (h w) c -> b c h w', h=14, w=14)
    logits = F.interpolate(logits, size=(224, 224), mode='bilinear')
    logits = min_max(logits)
    
    # Denormalize image
    OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)
    
    image_unnorm = (image_tensor[0].cpu() * torch.Tensor(OPENAI_STD)[:, None, None]) + \
                   torch.Tensor(OPENAI_MEAN)[:, None, None]
    image_pil = Image.fromarray((image_unnorm.permute(1, 2, 0).numpy() * 255).astype('uint8'))
    
    # Create heatmap
    img_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    logit_np = (logits[0, 0].cpu().numpy() * 255).astype('uint8')
    heatmap = cv2.applyColorMap(logit_np, cv2.COLORMAP_JET)
    overlay = (1 - alpha) * img_cv + alpha * heatmap
    overlay = cv2.cvtColor(overlay.astype('uint8'), cv2.COLOR_BGR2RGB)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(image_pil)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(overlay)
    axes[1].set_title(f'Relevance: "{text}"')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to: {save_path}")
    
    plt.show()
    
    return logits


def visualize_curated_pairs(
    model,
    pairs=None,
    output_dir="visualizations/",
    device="cuda",
    alpha=0.6,
):
    os.makedirs(output_dir, exist_ok=True)

    if pairs is None:

        pairs = [
            {
                "url": "dog.jpg",
                "pos": "nose",
                "name": "dog"
            },
            {
                "url": "cat.jpg",
                "pos": "wars",
                "name": "cat"
            },
        ]

    saved_paths = {}

    for item in pairs:
        name = item.get("name", "sample")
        pos = item.get("pos", "sample").replace(" ", "_")
        pos = pos.replace(",", "")

        image = item["url"]
        pos_path = os.path.join(output_dir, f"{name}_{pos}_pos.png")
        visualize_patches(model, image, item["pos"], pos_path, alpha=alpha, device=device)

        saved_paths[name] = {"pos": pos_path}

    return saved_paths





