#!/usr/bin/env python3
"""
Simple benchmarking script for SCLIP and CLIP models.
Supports both individual model benchmarking and model comparison.
"""

import os
import sys
import time
import csv
import argparse
import subprocess
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn

# Optional imports with fallbacks
try:
    import numpy as np
except ImportError:
    print("Warning: numpy not installed. Using built-in statistics.")
    class SimpleStats:
        @staticmethod
        def mean(values):
            return sum(values) / len(values)
        @staticmethod
        def std(values):
            mean_val = sum(values) / len(values)
            return (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
    np = SimpleStats()

try:
    import pandas as pd
except ImportError:
    pd = None
    print("Warning: pandas not installed. CSV reports will be generated but plots will be skipped.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None
    print("Warning: matplotlib/seaborn not installed. Plots will be skipped.")

# Import project modules
try:
    import models_tome as models
except ImportError:
    print("Error: Could not import models_tome. Make sure you're in the correct directory.")
    sys.exit(1)

# --------------------------- Utility Functions ---------------------------

def get_memory_usage(device: torch.device) -> float:
    """Get current GPU memory usage in MB."""
    if device.type == 'cuda':
        return torch.cuda.memory_allocated(device) / 1024 / 1024
    return 0.0

def get_peak_memory_usage(device: torch.device) -> float:
    """Get peak GPU memory usage in MB."""
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return 0.0

def reset_memory_stats(device: torch.device):
    """Reset GPU memory statistics."""
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

def time_function(fn, *args, device: torch.device, num_warmup: int = 10, num_iterations: int = 100) -> Tuple[float, float, float]:
    """Time a function and return mean, std latency and peak memory."""
    # Warmup
    for _ in range(num_warmup):
        fn(*args)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Reset memory stats
    reset_memory_stats(device)
    
    # Measure
    latencies = []
    for _ in range(num_iterations):
        start_time = time.time()
        fn(*args)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Get peak memory
    peak_memory = get_peak_memory_usage(device)
    
    # Calculate statistics
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    return mean_latency, std_latency, peak_memory

# --------------------------- Model Loading ---------------------------

def load_sclip_model(checkpoint_path: str, attn_fn: str = "softmax", attn_fn_alpha: float = 1.0, 
                    attn_fn_sparse_layers_vision: List[int] = None, 
                    attn_fn_sparse_layers_text: List[int] = None,
                    use_delta_attn: bool = False, delta_gamma: int = 64,
                    device: torch.device = torch.device("cuda")) -> Tuple[models.CLIP, Dict]:
    """Load SCLIP model from checkpoint."""
    print(f"Loading SCLIP model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Clean state dict (remove 'module.' prefix if using DDP)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            cleaned_state_dict[key[7:]] = value
        else:
            cleaned_state_dict[key] = value
    
    # Create model with configuration
    model = models.CLIP_VITB16_OPENAI(
        attn_fn=attn_fn,
        attn_fn_alpha=attn_fn_alpha,
        attn_fn_sparse_layers_vision=attn_fn_sparse_layers_vision or [],
        attn_fn_sparse_layers_text=attn_fn_sparse_layers_text or [],
        use_delta_attn=use_delta_attn,
        delta_gamma=delta_gamma
    )
    
    cleaned_state_dict = model.convert_state_dict(cleaned_state_dict)


    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    print(f" → {len(missing)} visual weights kept from timm pretrained model")
    print(f" → {len(unexpected)} keys ignored (should be zero)")
    if len(unexpected) > 0:
        print(f" → {unexpected} unexpected keys loaded from checkpoint")    

    model.to(device)
    model.eval()
    
    model_args = {
        'attn_fn': attn_fn,
        'attn_fn_alpha': attn_fn_alpha,
        'attn_fn_sparse_layers_vision': attn_fn_sparse_layers_vision or [],
        'attn_fn_sparse_layers_text': attn_fn_sparse_layers_text or [],
        'use_delta_attn': use_delta_attn,
        'delta_gamma': delta_gamma
    }
    
    print(f"SCLIP model loaded successfully")
    print(f"  Attention: {attn_fn} (α={attn_fn_alpha})")
    print(f"  Sparse vision layers: {attn_fn_sparse_layers_vision or []}")
    print(f"  Sparse text layers: {attn_fn_sparse_layers_text or []}")
    print(f"  Delta attention: {use_delta_attn} (γ={delta_gamma})")
    
    return model, model_args

def load_clip_model(model_name: str = "ViT-B/16", device: torch.device = torch.device("cuda")) -> Tuple[torch.nn.Module, Dict]:
    """Load CLIP model using open_clip."""
    try:
        import open_clip
    except ImportError:
        print("Error: open_clip not installed. Please install it with: pip install open_clip_torch")
        sys.exit(1)
    
    print(f"Loading CLIP model: {model_name}")
    
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
    model.to(device)
    model.eval()
    
    model_args = {'model_name': model_name}
    
    print(f"CLIP model loaded successfully")
    
    return model, model_args

# --------------------------- Benchmarking Functions ---------------------------

def benchmark_image_encoding(model, images: torch.Tensor, device: torch.device, 
                           num_warmup: int = 10, num_iterations: int = 100) -> Dict:
    """Benchmark image encoding."""
    print("Benchmarking image encoding...")
    
    def encode_images(imgs):
        with torch.no_grad():
            if hasattr(model, 'encode_image'):
                return model.encode_image(imgs)
            else:
                # For CLIP models
                return model.encode_image(imgs)
    
    mean_latency, std_latency, peak_memory = time_function(
        encode_images, images, device=device, num_warmup=num_warmup, num_iterations=num_iterations
    )
    
    throughput = (1000 / mean_latency) * images.shape[0]  # samples per second
    
    return {
        'mean_latency_ms': mean_latency,
        'std_latency_ms': std_latency,
        'peak_memory_mb': peak_memory,
        'throughput_samples_per_sec': throughput
    }

def benchmark_text_encoding(model, text_tokens: torch.Tensor, device: torch.device,
                          num_warmup: int = 10, num_iterations: int = 100) -> Dict:
    """Benchmark text encoding."""
    print("Benchmarking text encoding...")
    
    def encode_text(tokens):
        with torch.no_grad():
            if hasattr(model, 'encode_text'):
                return model.encode_text(tokens)
            else:
                # For CLIP models
                return model.encode_text(tokens)
    
    mean_latency, std_latency, peak_memory = time_function(
        encode_text, text_tokens, device=device, num_warmup=num_warmup, num_iterations=num_iterations
    )
    
    throughput = (1000 / mean_latency) * text_tokens.shape[0]  # samples per second
    
    return {
        'mean_latency_ms': mean_latency,
        'std_latency_ms': std_latency,
        'peak_memory_mb': peak_memory,
        'throughput_samples_per_sec': throughput
    }

def benchmark_end_to_end(model, images: torch.Tensor, text_tokens: torch.Tensor, device: torch.device,
                        num_warmup: int = 10, num_iterations: int = 100) -> Dict:
    """Benchmark end-to-end inference."""
    print("Benchmarking end-to-end inference...")
    
    def end_to_end_fn(imgs, tokens):
        with torch.no_grad():
            # SCLIP model
            img_features = model.encode_image(imgs)
            txt_features = model.encode_text(tokens)
            # Compute similarity
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
            return torch.matmul(img_features, txt_features.T)

    mean_latency, std_latency, peak_memory = time_function(
        end_to_end_fn, images, text_tokens, device=device, num_warmup=num_warmup, num_iterations=num_iterations
    )
    
    throughput = (1000 / mean_latency) * images.shape[0]  # samples per second
    
    return {
        'mean_latency_ms': mean_latency,
        'std_latency_ms': std_latency,
        'peak_memory_mb': peak_memory,
        'throughput_samples_per_sec': throughput
    }

# --------------------------- Data Generation ---------------------------

def create_synthetic_data(batch_size: int, input_resolution: int = 224, context_length: int = 77, 
                         device: torch.device = torch.device("cuda")) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic data for benchmarking."""
    # Create synthetic images
    images = torch.randn(batch_size, 3, input_resolution, input_resolution, device=device)
    
    # Create synthetic text tokens
    text_tokens = torch.randint(0, 49408, (batch_size, context_length), device=device)
    
    return images, text_tokens

# --------------------------- Main Benchmarking Function ---------------------------

def run_benchmark(model_type: str, device: torch.device, batch_size: int = 1, 
                 input_resolution: int = 224, context_length: int = 77,
                 num_warmup: int = 10, num_iterations: int = 100,
                 **model_kwargs) -> Dict:
    """Run comprehensive benchmark for a model."""
    
    # Load model
    if model_type == 'sclip':
        model, model_args = load_sclip_model(device=device, **model_kwargs)
        model_name = f"SCLIP-{model_kwargs.get('attn_fn', 'softmax')}"
    else:  # clip
        model, model_args = load_clip_model(device=device, **model_kwargs)
        model_name = f"CLIP-{model_kwargs.get('model_name', 'ViT-B/16')}"
    
    # Create synthetic data
    images, text_tokens = create_synthetic_data(
        batch_size, input_resolution, context_length, device
    )
    
    # Run benchmarks
    img_results = benchmark_image_encoding(model, images, device, num_warmup, num_iterations)
    txt_results = benchmark_text_encoding(model, text_tokens, device, num_warmup, num_iterations)
    e2e_results = benchmark_end_to_end(model, images, text_tokens, device, num_warmup, num_iterations)
    
    # Compile results
    results = {
        'model_name': model_name,
        'model_type': model_type,
        'batch_size': batch_size,
        'input_resolution': input_resolution,
        'context_length': context_length,
        'num_warmup': num_warmup,
        'num_iterations': num_iterations,
        **model_args
    }
    
    # Add benchmark results
    results.update({f'img_{k}': v for k, v in img_results.items()})
    results.update({f'txt_{k}': v for k, v in txt_results.items()})
    results.update({f'e2e_{k}': v for k, v in e2e_results.items()})
    
    # Print results
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Type: {model_type}")
    if model_type == 'sclip':
        print(f"Attention: {model_args['attn_fn']} (α={model_args['attn_fn_alpha']})")
    print(f"Resolution: {input_resolution}x{input_resolution}")
    print(f"Batch size: {batch_size}")
    print()
    
    print("Image Encoding:")
    print(f"  Latency: {img_results['mean_latency_ms']:.2f} ± {img_results['std_latency_ms']:.2f} ms")
    print(f"  Peak Memory: {img_results['peak_memory_mb']:.1f} MB")
    print(f"  Throughput: {img_results['throughput_samples_per_sec']:.1f} samples/sec")
    print()
    
    print("Text Encoding:")
    print(f"  Latency: {txt_results['mean_latency_ms']:.2f} ± {txt_results['std_latency_ms']:.2f} ms")
    print(f"  Peak Memory: {txt_results['peak_memory_mb']:.1f} MB")
    print(f"  Throughput: {txt_results['throughput_samples_per_sec']:.1f} samples/sec")
    print()
    
    print("End-to-End Inference:")
    print(f"  Latency: {e2e_results['mean_latency_ms']:.2f} ± {e2e_results['std_latency_ms']:.2f} ms")
    print(f"  Peak Memory: {e2e_results['peak_memory_mb']:.1f} MB")
    print(f"  Throughput: {e2e_results['throughput_samples_per_sec']:.1f} samples/sec")
    print("="*80)
    
    return results

# --------------------------- Comparison Functions ---------------------------

def save_results_to_csv(results: Dict, filename: str):
    """Save results to CSV file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results)
    
    print(f"Results saved to: {filename}")

def generate_comparison_report(results1: Dict, results2: Dict, output_file: str, is_sclip_comparison: bool = False, custom_name1: str = None, custom_name2: str = None):
    """Generate a text comparison report."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    name1 = custom_name1 or results1.get('model_name', 'Model 1')
    name2 = custom_name2 or results2.get('model_name', 'Model 2')
    with open(output_file, 'w') as f:
        f.write("MODEL COMPARISON REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{name1}: {results1['model_name']}\n")
        f.write(f"{name2}: {results2['model_name']}\n\n")
        f.write("PERFORMANCE COMPARISON\n")
        f.write("-" * 30 + "\n\n")
        # Image encoding comparison
        f.write("Image Encoding:\n")
        f.write(f"  {name1}: {results1['img_mean_latency_ms']:.2f} ± {results1['img_std_latency_ms']:.2f} ms\n")
        f.write(f"  {name2}: {results2['img_mean_latency_ms']:.2f} ± {results2['img_std_latency_ms']:.2f} ms\n")
        speedup = results1['img_mean_latency_ms'] / results2['img_mean_latency_ms']
        f.write(f"  Speedup: {speedup:.2f}x\n\n")
        # Text encoding comparison
        f.write("Text Encoding:\n")
        f.write(f"  {name1}: {results1['txt_mean_latency_ms']:.2f} ± {results1['txt_std_latency_ms']:.2f} ms\n")
        f.write(f"  {name2}: {results2['txt_mean_latency_ms']:.2f} ± {results2['txt_std_latency_ms']:.2f} ms\n")
        speedup = results1['txt_mean_latency_ms'] / results2['txt_mean_latency_ms']
        f.write(f"  Speedup: {speedup:.2f}x\n\n")
        # End-to-end comparison
        f.write("End-to-End Inference:\n")
        f.write(f"  {name1}: {results1['e2e_mean_latency_ms']:.2f} ± {results1['e2e_std_latency_ms']:.2f} ms\n")
        f.write(f"  {name2}: {results2['e2e_mean_latency_ms']:.2f} ± {results2['e2e_std_latency_ms']:.2f} ms\n")
        speedup = results1['e2e_mean_latency_ms'] / results2['e2e_mean_latency_ms']
        f.write(f"  Speedup: {speedup:.2f}x\n\n")
        # Memory comparison
        f.write("Peak Memory Usage:\n")
        f.write(f"  {name1}: {results1['e2e_peak_memory_mb']:.1f} MB\n")
        f.write(f"  {name2}: {results2['e2e_peak_memory_mb']:.1f} MB\n")
        memory_ratio = results1['e2e_peak_memory_mb'] / results2['e2e_peak_memory_mb']
        f.write(f"  Memory Ratio: {memory_ratio:.2f}x\n\n")
    print(f"Comparison report saved to: {output_file}")

def create_comparison_plots(results1: Dict, results2: Dict, output_dir: str, is_sclip_comparison: bool = False, custom_name1: str = None, custom_name2: str = None):
    """Create comparison plots."""
    if plt is None or sns is None:
        print("Skipping plots (matplotlib/seaborn not available)")
        return
    os.makedirs(output_dir, exist_ok=True)
    metrics = ['img_mean_latency_ms', 'txt_mean_latency_ms', 'e2e_mean_latency_ms']
    labels = ['Image Encoding', 'Text Encoding', 'End-to-End']
    model1_values = [results1[m] for m in metrics]
    model2_values = [results2[m] for m in metrics]
    name1 = custom_name1 or results1.get('model_name', 'Model 1')
    name2 = custom_name2 or results2.get('model_name', 'Model 2')
    plt.figure(figsize=(10, 6))
    x = range(len(metrics))
    width = 0.35
    plt.bar([i - width/2 for i in x], model1_values, width, label=name1, alpha=0.8)
    plt.bar([i + width/2 for i in x], model2_values, width, label=name2, alpha=0.8)
    plt.xlabel('Operation')
    plt.ylabel('Latency (ms)')
    plt.title(f'Latency Comparison: {name1} vs {name2}')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    for i, (v1, v2) in enumerate(zip(model1_values, model2_values)):
        plt.text(i - width/2, v1 + 0.1, f'{v1:.1f}', ha='center', va='bottom')
        plt.text(i + width/2, v2 + 0.1, f'{v2:.1f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plots saved to: {output_dir}")

# --------------------------- Main Function ---------------------------

def main():
    parser = argparse.ArgumentParser(description='Simple benchmarking script for SCLIP and CLIP models')
    
    # Mode selection
    parser.add_argument('--mode', choices=['benchmark', 'compare'], required=True,
                       help='Mode: benchmark single model or compare two models')
    
    # Model 1 (SCLIP)
    parser.add_argument('--sclip-checkpoint', type=str, help='Path to SCLIP checkpoint')
    parser.add_argument('--attn-fn', type=str, default='softmax', choices=['softmax', 'entmax'],
                       help='Attention function for SCLIP')
    parser.add_argument('--attn-fn-alpha', type=float, default=1.0, help='Alpha for entmax')
    parser.add_argument('--attn-fn-sparse-layers-vision', type=int, nargs='+', default=[],
                       help='Vision layers to use sparse attention')
    parser.add_argument('--attn-fn-sparse-layers-text', type=int, nargs='+', default=[],
                       help='Text layers to use sparse attention')
    parser.add_argument('--use-delta-attn', action='store_true', help='Use delta attention')
    parser.add_argument('--delta-gamma', type=int, default=64, help='Delta attention gamma')
    parser.add_argument('--model1-name', type=str, default=None, help='Custom name for the first model (for output/reports)')
    
    # Model 2 (for comparison)
    parser.add_argument('--sclip-checkpoint-2', type=str, help='Path to second SCLIP checkpoint')
    parser.add_argument('--clip-model-name', type=str, default='ViT-B/16', help='CLIP model name')
    parser.add_argument('--attn-fn-2', type=str, default='softmax', choices=['softmax', 'entmax'],
                       help='Attention function for second SCLIP')
    parser.add_argument('--attn-fn-alpha-2', type=float, default=1.0, help='Alpha for second SCLIP entmax')
    parser.add_argument('--attn-fn-sparse-layers-vision-2', type=int, nargs='+', default=[],
                       help='Vision layers to use sparse attention for second SCLIP')
    parser.add_argument('--attn-fn-sparse-layers-text-2', type=int, nargs='+', default=[],
                       help='Text layers to use sparse attention for second SCLIP')
    parser.add_argument('--use-delta-attn-2', action='store_true', help='Use delta attention for second SCLIP')
    parser.add_argument('--delta-gamma-2', type=int, default=64, help='Delta attention gamma for second SCLIP')
    parser.add_argument('--model2-name', type=str, default=None, help='Custom name for the second model (for output/reports)')
    
    # Benchmark configuration
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--input-resolution', type=int, default=224, help='Input resolution')
    parser.add_argument('--context-length', type=int, default=77, help='Text context length')
    parser.add_argument('--num-warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--num-iterations', type=int, default=100, help='Number of measurement iterations')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./benchmark_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'benchmark':
        if not args.sclip_checkpoint:
            raise ValueError("--sclip-checkpoint is required for benchmark mode")
    elif args.mode == 'compare':
        if not args.sclip_checkpoint:
            raise ValueError("--sclip-checkpoint is required for compare mode")
        if not args.sclip_checkpoint_2 and not args.clip_model_name:
            raise ValueError("Either --sclip-checkpoint-2 or --clip-model-name is required for compare mode")
    
    # Set device
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = torch.device('cpu')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'benchmark':
        # Single model benchmark
        print("Running single model benchmark...")
        
        sclip_kwargs = {
            'checkpoint_path': args.sclip_checkpoint,
            'attn_fn': args.attn_fn,
            'attn_fn_alpha': args.attn_fn_alpha,
            'attn_fn_sparse_layers_vision': args.attn_fn_sparse_layers_vision,
            'attn_fn_sparse_layers_text': args.attn_fn_sparse_layers_text,
            'use_delta_attn': args.use_delta_attn,
            'delta_gamma': args.delta_gamma
        }
        
        results = run_benchmark(
            model_type='sclip',
            device=device,
            batch_size=args.batch_size,
            input_resolution=args.input_resolution,
            context_length=args.context_length,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
            **sclip_kwargs
        )
        
        # Save results
        output_file = os.path.join(args.output_dir, 'benchmark_results.csv')
        save_results_to_csv(results, output_file)
        
        # Add custom name to results
        custom_name = args.model1_name or os.path.basename(args.sclip_checkpoint) or 'SCLIP'
        results['custom_model_name'] = custom_name
        
    elif args.mode == 'compare':
        # Model comparison
        print("Running model comparison...")
        
        # Benchmark first model (SCLIP)
        print("\n" + "="*50)
        print("BENCHMARKING MODEL 1")
        print("="*50)
        
        sclip_kwargs = {
            'checkpoint_path': args.sclip_checkpoint,
            'attn_fn': args.attn_fn,
            'attn_fn_alpha': args.attn_fn_alpha,
            'attn_fn_sparse_layers_vision': args.attn_fn_sparse_layers_vision,
            'attn_fn_sparse_layers_text': args.attn_fn_sparse_layers_text,
            'use_delta_attn': args.use_delta_attn,
            'delta_gamma': args.delta_gamma
        }
        
        results1 = run_benchmark(
            model_type='sclip',
            device=device,
            batch_size=args.batch_size,
            input_resolution=args.input_resolution,
            context_length=args.context_length,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
            **sclip_kwargs
        )
        
        # Benchmark second model
        print("\n" + "="*50)
        print("BENCHMARKING MODEL 2")
        print("="*50)
        
        if args.sclip_checkpoint_2:
            # Second SCLIP model
            sclip_kwargs_2 = {
                'checkpoint_path': args.sclip_checkpoint_2,
                'attn_fn': args.attn_fn_2,
                'attn_fn_alpha': args.attn_fn_alpha_2,
                'attn_fn_sparse_layers_vision': args.attn_fn_sparse_layers_vision_2,
                'attn_fn_sparse_layers_text': args.attn_fn_sparse_layers_text_2,
                'use_delta_attn': args.use_delta_attn_2,
                'delta_gamma': args.delta_gamma_2
            }
            
            results2 = run_benchmark(
                model_type='sclip',
                device=device,
                batch_size=args.batch_size,
                input_resolution=args.input_resolution,
                context_length=args.context_length,
                num_warmup=args.num_warmup,
                num_iterations=args.num_iterations,
                **sclip_kwargs_2
            )
            is_sclip_comparison = True
        else:
            # CLIP model
            results2 = run_benchmark(
                model_type='clip',
                device=device,
                batch_size=args.batch_size,
                input_resolution=args.input_resolution,
                context_length=args.context_length,
                num_warmup=args.num_warmup,
                num_iterations=args.num_iterations,
                model_name=args.clip_model_name
            )
            is_sclip_comparison = False
        
        # Save individual results
        save_results_to_csv(results1, os.path.join(args.output_dir, 'model1_results.csv'))
        save_results_to_csv(results2, os.path.join(args.output_dir, 'model2_results.csv'))
        
        # Add custom names to results
        custom_name1 = args.model1_name or os.path.basename(args.sclip_checkpoint) or 'Model 1'
        custom_name2 = args.model2_name or (
            os.path.basename(args.sclip_checkpoint_2) if args.sclip_checkpoint_2 else args.clip_model_name) or 'Model 2'
        results1['custom_model_name'] = custom_name1
        results2['custom_model_name'] = custom_name2
        
        # Generate comparison report and plots
        report_file = os.path.join(args.output_dir, 'comparison_report.txt')
        generate_comparison_report(results1, results2, report_file, is_sclip_comparison, custom_name1, custom_name2)
        create_comparison_plots(results1, results2, args.output_dir, is_sclip_comparison, custom_name1, custom_name2)
        
        print(f"\nComparison complete! Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
