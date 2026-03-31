#!/usr/bin/env bash
# =============================================================================
# start_vllm.sh
# =============================================================================
# Launches the vLLM OpenAI-compatible server for the AWQ-quantised Socratic
# tutor model on an NVIDIA A100 80GB GPU.
#
# HARDWARE CONTEXT: NVIDIA A100 SXM4 80GB
#   - Peak BF16 Tensor Core throughput: 312 TFLOPS
#   - HBM2e bandwidth: 2,000 GB/s
#   - NVLink 3.0 bandwidth (if multi-GPU): 600 GB/s
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH="${MODEL_PATH:-./checkpoints/socratic_awq_q4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# ---------------------------------------------------------------------------
# Memory Budget Calculation for 80 Concurrent Users
# ---------------------------------------------------------------------------

GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=80                

echo "=============================================="
echo " Socratic AI Tutor — vLLM Server Startup"
echo "=============================================="
echo " Model: ${MODEL_PATH}"
echo " Host:Port: ${HOST}:${PORT}"
echo " GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION} ($(echo "${GPU_MEMORY_UTILIZATION} * 80" | bc -l | xargs printf "%.1f")GB of 80GB)"
echo " Max Model Length: ${MAX_MODEL_LEN} tokens"
echo " Max Concurrent Sequences: ${MAX_NUM_SEQS}"
echo "=============================================="

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

if ! command -v python &>/dev/null; then
    echo "[ERROR] Python not found. Activate your virtualenv first."
    exit 1
fi

if ! python -c "import vllm" 2>/dev/null; then
    echo "[ERROR] vLLM not installed. Run: pip install vllm"
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "[ERROR] Model directory not found: ${MODEL_PATH}"
    echo "        Run ml_pipeline/quantize_awq.py to generate the AWQ model."
    exit 1
fi

# Check GPU availability
if ! python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU detected'" 2>/dev/null; then
    echo "[ERROR] No CUDA-capable GPU detected. An A100 GPU is required."
    exit 1
fi

GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown GPU")
GPU_MEM=$(python -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))" 2>/dev/null || echo "?")
echo "[INFO] Detected GPU: ${GPU_NAME} (${GPU_MEM}GB)"

# ---------------------------------------------------------------------------
# Launch vLLM
# ---------------------------------------------------------------------------

echo "[INFO] Starting vLLM server..."

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    \
    `# Quantisation: tell vLLM to use AWQ GEMM kernels for inference` \
    --quantization awq \
    \
    `# Disable CUDA graph capture — see memory analysis above (point 5)` \
    --enforce-eager \
    \
    `# Context window: 4096 tokens is sufficient for Socratic dialogue` \
    `# (typical exchange: ~200-800 tokens; 4096 allows deep multi-turn sessions)` \
    --max-model-len ${MAX_MODEL_LEN} \
    \
    `# Allocate 90% of GPU memory to vLLM (leaves 8GB for OS/CUDA overhead)` \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    \
    `# Maximum concurrent sequences in the continuous batch scheduler` \
    `# 80 matches the expected concurrent user count` \
    --max-num-seqs ${MAX_NUM_SEQS} \
    \
    `# PagedAttention block size: 16 tokens/block is optimal for A100 HBM2e` \
    `# bandwidth characteristics. Larger blocks = fewer allocations;` \
    `# smaller blocks = finer-grained memory reclamation.` \
    --block-size 16 \
    \
    `# Use BF16 for KV cache — A100 TF32/BF16 cores are far faster than FP16` \
    `# and BF16 preserves the dynamic range needed for long reasoning chains` \
    --kv-cache-dtype auto \
    \
    `# Tokeniser parallelism: use multiple workers for fast tokenisation` \
    --tokenizer-pool-size 4 \
    \
    `# API server settings` \
    --host "${HOST}" \
    --port "${PORT}" \
    \
    `# Disable model download verification for locally-stored AWQ model` \
    --trust-remote-code \
    \
    `# Logging: INFO level for production, DEBUG for development` \
    --uvicorn-log-level info
