# β-CLIP 

This project extends CLIP to enable text-conditioned vision feature alignment at multiple granularities.


## Prerequisites

- Python 3.10
- CUDA GPUs
- CUDA Toolkit 12.4+ (for PyTorch 2.6.0)

For detailed installation instructions, see [INSTALL.md](INSTALL.md).
For data setup, see [DATA.md](DATA.md).

## Training

Ensure the following configs are set:
```
  --metadata data/ShareGPT4V/annotations/share-captioner_coco_lcs_sam_1246k_1107_filtered.json \
  --root data/ShareGPT4V/data \
  --resume pretrained/ViT-B-16.pt \
```

For training β-CLIP, we provide the following example scripts for locally training (adjust the number of GPUS via `--nproc_per_node=1`):
- `train_CE.sh`: local training with CE loss
- `train_BCE.sh`: local training with BCE loss


If using submitit on a shared slurm cluster for multi-node training (ensure the environment is activated before submitting):
- `submitit_text_1+5+30_global+bcal_CE_beta_0.5.sh`: Multi-node training with CE loss
- `submitit_text_1+5+30_global+bcal_BCE_beta_0.5.sh`: Multi-node training with BCE loss

## Evaluation

For evaluation only, change the `--resume` config to point to a fine-tuned checkpoint and enable the `--evaluate` flag.

[ckpt β-CLIP: CE, β=0.5, 30 phrases](https://drive.google.com/file/d/1nNbYLlU_1VcbEioLL7yX_rzARAzxD6q_/view?usp=drive_link)

[ckpt β-CLIP: BCE, β=0.5, 30 phrases](https://drive.google.com/file/d/1sFQXDQZOzM5XoFkRD_9OYORfQdJrHmN4/view?usp=drive_link)

These checkpoints can be downloaded using [download_ckpts.sh](download_ckpts.sh).

## Results

| Method | | FG-OVD | | | SV-1k | | U-1k | |
|:------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| | Hard | Medium | Easy | Trivial | T2I | I2T | T2I | I2T |
| K=36, β=0.5, CE | 30.9 | 55.4 | 60.4 | 80.3 | 93.7 | 94.0 | 89.0 | 88.6 |
| K=36, β=0.5, BCE | 20.1 | 38.5 | 34.2 | 71.3 | 94.4 | 94.1 | 91.8 | 92.3 |

### Key Configs
- **`--bcal-loss-mode k_positives_ce`**: Specifies the  β-CAL loss computation mode:
  - `k_positives_bce`: Binary cross-entropy loss with K positive samples per image
  - `k_positives_ce`: Cross-entropy loss with K positive samples

- **`--beta 0.5`**: Beta parameter for controlling contextualization in the loss.

- **`--fg-loss-fn global+bcal`**: Specifies which loss functions to use. Options include:
  - `global`: Standard CLIP contrastive loss between CLS and EOS embeddings
  - `bcal`: β Contextualized Contrastive Alignment Loss for fine-grained alignment
  - Combine multiple losses with `+` (e.g., `global+bcal`)

- **`--epochs 10`**: Number of training epochs.

- **`--batch-size 64`**: Batch size per GPU. The effective batch size is `batch-size × update-freq × num_gpus × num_nodes`.

- **`--model CLIP_VITB16_OPENAI`**: Model architecture specification. Currently supports CLIP ViT-B/16 and CLIP ViT-L/14 with OpenAI pretrained weights.


## Project Structure

```
β-CLIP/
├── main_clip_ft.py              # Main training script
├── datasets.py                  # Dataset loading and preprocessing
├── models_tome.py               # Model definitions
├── losses.py                    # Loss function implementations
```


## Citation

```bibtex
@article{zohra2025β-CLIP,
  title={β-CLIP: Text-Conditioned Contrastive Learning for Multi-Granular Vision Language Alignment in CLIP},
  author={Zohra, Fatimah and Zhao, Chen and Itani, Hani and Ghanem, Bernard},
  journal={arXiv preprint arXiv:},
  year={2025}
}
```

This project is developed using [SLIP](https://github.com/facebookresearch/SLIP) and [TIMM](https://github.com/huggingface/pytorch-image-models).
