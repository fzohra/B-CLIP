master_addr=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
master_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

export MASTER_PORT=$master_port
export MASTER_ADDR=$master_addr

export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1

torchrun --nproc_per_node=1 \
  --nnodes=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$master_addr:$master_port \
    main_clip_ft.py \
  --output-dir ./ckpts/text_1+5+30_global+bcal_BCE_beta_0.5_bs64_ep10 \
  --dataset sharegpt4v \
  --metadata data/ShareGPT4V/annotations/share-captioner_coco_lcs_sam_1246k_1107_filtered.json \
  --root data/ShareGPT4V/data \
  --resume pretrained/ViT-B-16.pt \
  --model CLIP_VITB16_OPENAI \
  --epochs 10 \
  --batch-size 64 \
  --update-freq 8 \
  --eval-freq 1 \
  --warmup-epochs 0.01 \
  --lr-start 1e-9 \
  --lr 1e-5 \
  --lr-end 1e-7 \
  --lr-conditioner 1e-3 \
  --lr-conditioner-end 1e-4 \
  --wd 0.01 \
  --global-pool '' \
  --context_length 248 \
  --fg-loss-fn cls+tcil \
  --text-conditioning-mode attn_pooling_mlp \
  --use-text-conditioned-patches \
  --use-text-eos \
  --use-text-concepts \
  --max-concept-context-length 30 \
  --max-concepts 30 \
  --use-tcl \
  --cls-alpha 1 \
  --tcil-alpha 1 \
  --num_positive_samples 5 \
  --tcil-loss-mode k_positives_bce \
  --beta 0.5 \
  --use-caption \
  --use-caption-in-eos \
  --wandb