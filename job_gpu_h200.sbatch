#!/bin/bash
#===============================================================
# Job Slurm untuk GPU node (NVIDIA H200), dipakai oleh SEMUA demo.
#
# Pemakaian:
#   sbatch job_gpu_h200.sbatch                                  # default: demo_llm_terjemahan.py
#   sbatch job_gpu_h200.sbatch demo_llm_percakapan.py            # vLLM + SEA-LION v5 9B
#   sbatch job_gpu_h200.sbatch demo_analisis_teks.py
#
# Pantau: squeue -u $USER      Hasil: logs/gpu-demo-<JOBID>.out
#===============================================================
#SBATCH --job-name=gpu-demo
#SBATCH --partition=gpu_general    # partisi GPU H200 (cek: sinfo)
#SBATCH --gres=gpu:1               # 1 dari 4 GPU H200 per node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00            # eksperimen ~1 menit + memuat model (vLLM ~2-3 menit)
#SBATCH --output=logs/%x-%j.out    # %x = nama job, %j = job ID
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT=${1:-demo_llm_terjemahan.py}
shift || true                      # sisa argumen diteruskan ke script Python

echo "Job ${SLURM_JOB_ID} | $(hostname) | $(date)"
echo "GPU dialokasikan: ${CUDA_VISIBLE_DEVICES:-tidak ada} | script: ${SCRIPT} $*"

# 1) Environment conda (torch 2.14+cu130 sudah membawa CUDA sendiri; modul cuda tidak diperlukan)
CONDA_ROOT=${CONDA_ROOT:-/path/ke/miniforge3}
set +u                             # skrip activate conda memakai variabel yang belum didefinisikan
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate vllm_env            # berisi: torch, transformers, sentencepiece, vllm, matplotlib
set -u

# 2) Model dibaca dari folder lokal yang sudah diunduh di login node (unduh_model.sh)
export MODEL_DIR=${MODEL_DIR:-/path/ke/scratch/anda/models}
export HF_HUB_OFFLINE=1            # compute node tanpa internet
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# 3) Info hardware + rekam pemakaian GPU tiap 5 detik di latar belakang
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
           --format=csv -l 5 > logs/gpu-usage-${SLURM_JOB_ID}.csv &
MONITOR_PID=$!
trap 'kill ${MONITOR_PID} 2>/dev/null || true' EXIT

# 4) Jalankan eksperimen
srun python "${SCRIPT}" "$@"

echo "Job selesai pada $(date)"
