# Installation Guide

## SLURM setup
```
srun --ntasks=1 --gres=gpu:1 --time=01:00:00 --nodes=1 --constraint=a100 --cpus-per-gpu=8 --pty bash -l
```

### Install PyTorch=2.6.0, Python=3.10.12 and CUDA support

```bash
conda create -n beta-clip python=3.10.12
conda activate beta-clip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```


### Download spaCy Language Model
The project uses spaCy for NLP processing (phrase extraction). Download the English language model:

```bash
python -m spacy download en_core_web_sm
```

## Download the Pretrained Models

Pretrained CLIP models should be placed in the `pretrained/` directory or specified via the `--resume` argument. The default path is, please ensure that the models are named exactly as follows (or modify the condition `args.resume.endswith('ViT-L-14.pt') or args.resume.endswith('ViT-B-16.pt')` in `main_clip_ft.py`):
```
pretrained/ViT-B-16.pt
pretrained/ViT-L-14.pt
```

For data setup, see [DATA.md](DATA.md).