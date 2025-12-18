import math
import os, sys, json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist

import torchvision.transforms as T
from torchvision.ops import roi_align

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Sequence, Union
from PIL import Image, ImageDraw

import wandb

# -------------------- constants --------------------
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# -------------------- distributed helpers --------------------
def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def is_main_process():
    return get_rank() == 0

def allreduce_sum_int(x: int) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.tensor([x], device=device, dtype=torch.long)
    if is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())

def unwrap_model(model):
    return model.module if hasattr(model, "module") else model

# -------------------- utils & data I/O --------------------
def coco_xywh_to_roi(bbox, img_w, img_h, feat):
    """COCO (x,y,w,h) → tensor([[0,x1,y1,x2,y2]]) on patch grid."""
    x,y,w,h = bbox
    x2, y2  = x+w, y+h
    return torch.tensor([[0,
                          x  / img_w * feat,
                          y  / img_h * feat,
                          x2 / img_w * feat,
                          y2 / img_h * feat]], dtype=torch.float32)

def _str(x):
    """token-list → string ; string → unchanged."""
    return " ".join(map(str,x)) if isinstance(x, list) else str(x)

def load_items(path:Path):
    if path.suffix == ".jsonl":                          # FG-CLIP/LLaVA
        for line in open(path):
            m = json.loads(line)
            yield dict(img_path=m["img_path"],
                       bbox    =m["bbox"],
                       captions=[_str(m["pos_expression"])]
                                +[_str(t) for t in m["neg_expression"]])
    else:                                                # official FG-OVD
        blob = json.load(open(path))
        id2file = {im["id"]: im["file_name"] for im in blob["images"]}
        for ann in blob["annotations"]:
            yield dict(img_path=id2file[ann["image_id"]],
                       bbox    =ann["bbox"],
                       captions=[_str(ann["pos_expression"])]
                                +[_str(t) for t in ann["neg_expression"]])

def overlay_bbox(
    img: Image.Image,
    bbox: Sequence[Union[int, float]],  # [x, y, w, h] in original pixels
    save_path: Union[str, Path] = None,
    color: str = "red",
    width: int = 3,
) -> Image.Image:
    x, y, w, h = bbox
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(save_path)
    return out

def forward_last_block_value_only(vision, x):
    """
    Execute the last transformer block without any QK attention mixing.
    Keeps LN1, V-proj+out_proj, residuals, LN2, MLP.
    """
    blk = vision.blocks[-1]
    residual = x
    x = blk.norm1(x)  # (B,N,C)
    Wqkv, bqkv = blk.attn.qkv.weight, blk.attn.qkv.bias
    C = Wqkv.shape[0] // 3
    Wv,  bv  = Wqkv[C*2:], bqkv[C*2:]
    x = F.linear(x, Wv, bv)    # value proj
    x = blk.attn.proj(x)       # out_proj
    x = residual + x           # residual
    residual = x
    x = blk.norm2(x)
    x = blk.mlp(x)
    x = residual + x           # residual
    return x

@torch.no_grad()
def dense_no_last_block(vision, proj, px, return_with_prefix=False):
    x = vision.patch_embed(px)
    x = vision._pos_embed(x)
    x = vision.patch_drop(x)
    x = vision.norm_pre(x)
    for blk in vision.blocks[:-1]:
        x = blk(x)
    x = forward_last_block_value_only(vision, x)
    x = vision.norm(x)
    x = x @ proj
    if return_with_prefix:
        return x
    p = vision.num_prefix_tokens
    patches = x[:, p:, :]  # strip cls / reg
    return patches

@torch.no_grad()
def text_embeds(tok, texts, model, device):
    ids = tok(texts).to(device)
    x = model.encode_text(ids)
    x = x / x.norm(p=2, dim=-1, keepdim=True)
    return x

def overlay_bbox_px(
    px: torch.Tensor,          # (1, 3, clip_res, clip_res)
    patch_box: torch.Tensor,   # tensor([x1, y1, x2, y2]) on patch grid
    grid_W: int,               # e.g. 14 (224/16) or 21 (336/16)
    save_path: Path,
) -> None:
    clip_res = px.shape[-1]
    img = px.squeeze(0).cpu()                       # (3, H, W)
    mean = torch.tensor(_CLIP_MEAN)[:, None, None]
    std  = torch.tensor(_CLIP_STD)[:, None, None]
    img = (img * std + mean).clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()              # HWC

    patch_sz = clip_res / grid_W
    x1, y1, x2, y2 = patch_box.tolist()
    x = x1 * patch_sz
    y = y1 * patch_sz
    w = (x2 - x1) * patch_sz
    h = (y2 - y1) * patch_sz

    fig, ax = plt.subplots(figsize=(3, 3), dpi=150)
    ax.imshow(img)
    ax.add_patch(patches.Rectangle((x, y), w, h,
                                   linewidth=2, edgecolor='r', facecolor='none'))
    ax.axis('off')
    fig.tight_layout(pad=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# -------------------- distributed evaluators --------------------
@torch.no_grad()
def eval_split(jpath, model, transform, tokenizer,
               device, img_root, img_res):
    """
    Shards items across ranks, returns local (hit, total).
    The caller will all-reduce these scalars.
    """
    m = unwrap_model(model)

    items = list(load_items(jpath))
    rank, world_size = get_rank(), get_world_size()
    items = items[rank::world_size]

    local_hit = 0
    local_total = len(items)

    patch = img_res // m.visual.patch_embed.patch_size[0]  # 14 or 21 …
    iterator = tqdm(items, desc=jpath.name) if is_main_process() else items

    for itm in iterator:
        img = Image.open(img_root/itm["img_path"]).convert("RGB")
        W,H = img.size
        img = img.resize((img_res, img_res))
        px  = transform(img).unsqueeze(0).to(device)

        bs = px.size(0)
        feature_map = dense_no_last_block(m.visual, m.image_projection, px)  # B N C
        w_grid = int(math.sqrt(feature_map.shape[1]))
        h_grid = w_grid

        box   = coco_xywh_to_roi(itm["bbox"], W, H, patch).to(device)

        feature_map = feature_map.view(bs, h_grid, w_grid, -1).permute(0, 3, 1, 2)
        x_rois = roi_align(feature_map.type(torch.float32), box, (1, 1), 1.0, -1, True)[..., 0, 0]
        img_vec = x_rois / x_rois.norm(p=2, dim=-1, keepdim=True)

        txt_vec = text_embeds(tokenizer, itm["captions"], m, device)
        similarity = (100.0 * img_vec @ txt_vec.T).softmax(dim=-1)
        if torch.equal(torch.max(similarity[0]), similarity[0][0]):
            local_hit += 1

    return local_hit, local_total

# -------------------- distributed-safe validation --------------------
def validate_fgovd(model, output_dir, epoch, step, log_wandb, tokenizer, device='cuda'):
    """
    Assumes process-group/model are already initialized by caller (DDP/torchrun).
    Does NOT change device or init/dispose the process group.
    """

    model.eval()
    m = unwrap_model(model)

    fgovd_dir = '/path/to/data/fgovd' #change
    coco_dir  = '/path/to/data/coco/' #change

    W = 224
    transform = T.Compose([
        T.CenterCrop(W),
        T.ToTensor(),
        T.Normalize(_CLIP_MEAN, _CLIP_STD),
    ])
    jdir, idir = Path(fgovd_dir), Path(coco_dir)

    splits = ["h_attributes_llava.jsonl",
              "m_attributes_llava.jsonl",
              "e_attributes_llava.jsonl",
              "shuffle_negatives_llava.jsonl"]

    res = {}

    # Bbox features evaluation
    for sp in splits:
        if is_main_process(): print("Evaluating Bbox features")
        local_hit, local_total = eval_split(jdir/sp, model, transform, tokenizer, device, idir, W)
        global_hit   = allreduce_sum_int(local_hit)
        global_total = allreduce_sum_int(local_total)
        acc = 100.0 * (global_hit / max(1, global_total))
        key = sp.split("_")[0] if "attributes" in sp else "trivial"
        if is_main_process():
            res[key] = acc
            print(f"{sp:<28}  {acc:6.2f} %")

    # Only rank 0 writes/logs
    if is_main_process():
        print("\nSummary:")
        for k in ["h","m","e","trivial"]:
            print(f"  {k:<7}: {res[k]:6.2f} %")

        results_dir   = 'evaluation_fgovd/'
        results_subdir= os.path.split(output_dir)[-1]
        ckpt_path     = f'{output_dir}/checkpoint_{epoch+1}.pt'
        fname         = "fg_ovd_llava.json"
        epoch_subdir  = 'epoch_' + str(epoch+1)
        results_f     = Path(results_dir) / results_subdir / epoch_subdir / fname
        os.makedirs(results_f.parent, exist_ok=True)

        to_write = {
            "checkpoint": str(ckpt_path),
            "results": {k: round(v, 4) for k, v in res.items()},
        }

        if log_wandb:
            payload = {"fg_ovd_llava": res}
            wandb.log(payload, step=step)

        with open(results_f, "w") as f:
            json.dump(to_write, f, indent=2)