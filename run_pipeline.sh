#!/bin/bash

# Retrieval + Captioning Pipeline Quick Start Script
# Optimized for NVIDIA RTX 5000 Ada Generation (32GB)
# Usage: ./run_pipeline.sh [rtx5000_ada|a5000|rtx3070] [coco|flickr30k]

set -e  # Exit on error

# Default arguments - RTX 5000 Ada as default
DEVICE=${1:-rtx5000_ada}
DATASET=${2:-coco}

echo "=========================================="
echo "Retrieval + Captioning Pipeline"
echo "Optimized for NVIDIA RTX 5000 Ada"
echo "=========================================="
echo "Device:  $DEVICE"
echo "Dataset: $DATASET"
echo ""

# RTX 5000 Ada specific optimizations
if [ "$DEVICE" = "rtx5000_ada" ]; then
    echo "RTX 5000 Ada Optimizations Enabled:"
    echo "  - 32GB VRAM support"
    echo "  - Larger batch sizes (96/128/48)"
    echo "  - Higher resolution (336x336)"
    echo "  - TF32 acceleration (~2x speedup)"
    echo "  - Larger decoder (6 layers, 768 hidden)"
    echo "  - 8 dataloader workers"
    echo ""
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate

# Check if dependencies are installed
if ! python -c "import torch" &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

# Check CUDA availability and GPU info
echo ""
echo "Checking CUDA availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'CUDA Version: {torch.version.cuda}')
    compute_cap = torch.cuda.get_device_capability(0)
    print(f'Compute Capability: {compute_cap[0]}.{compute_cap[1]}')
    if compute_cap[0] >= 8 and compute_cap[1] >= 9:
        print('✓ Ada Lovelace architecture detected - TF32 will be enabled')
"

# Create output directory
mkdir -p outputs logs checkpoints data cache

echo ""
echo "=========================================="
echo "Starting Pipeline"
echo "=========================================="
if [ "$DEVICE" = "rtx5000_ada" ]; then
    echo "Estimated time: 4-5 hours (faster with Ada optimizations)"
else
    echo "Estimated time: 6-8 hours"
fi
echo "Progress will be saved to: ./outputs/"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run pipeline
python main.py \
    --device "$DEVICE" \
    --dataset "$DATASET" \
    --output-dir ./outputs \
    2>&1 | tee logs/pipeline_$(date +%Y%m%d_%H%M%S).log

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Pipeline completed successfully!"
    echo "=========================================="
    echo "Results saved to:"
    echo "  - ./outputs/results.json"
    echo "  - ./outputs/evaluation_results.json"
    echo "  - ./outputs/performance_report.json"
    echo ""
    
    # Display quick summary
    if [ -f "outputs/results.json" ]; then
        echo "Quick Summary:"
        python -c "
import json
with open('outputs/results.json') as f:
    data = json.load(f)
    print(f\"  Total time: {data['total_time_hours']:.2f} hours\")
    print(f\"  Hardware: {data['experiment_info']['hardware']}\")
    if 'clip_baseline' in data['results_summary']:
        cb = data['results_summary']['clip_baseline']
        print(f\"  CLIP R@1: {cb['recall@1']['mean']:.2f}%\")
    if 'linear_probe' in data['results_summary']:
        lp = data['results_summary']['linear_probe']
        print(f\"  Probe R@1: {lp['recall@1']['mean']:.2f}%\")
    if 'captioner_full' in data['results_summary']:
        cf = data['results_summary']['captioner_full']
        print(f\"  Captioner CIDEr: {cf['cider']['mean']:.2f}\")
" 2>/dev/null || echo "  (See results.json for details)"
    fi
else
    echo ""
    echo "=========================================="
    echo "✗ Pipeline failed!"
    echo "=========================================="
    echo "Check logs/pipeline_*.log for details"
    exit 1
fi
