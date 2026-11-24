#!/bin/bash
#===============================================================
# Job Slurm untuk GPU node (NVIDIA H200), dipakai oleh SEMUA demo.
#
# Pemakaian:
#   sbatch job_gpu_h200.sbatch                                  # default: demo_matmul.py
#   sbatch job_gpu_h200.sbatch demo_gelombang_2d.py --duration 60
#   sbatch job_gpu_h200.sbatch demo_llm_terjemahan.py
#   sbatch job_gpu_h200.sbatch demo_llm_percakapan.py --engine vllm
#
# Pantau: squeue -u $USER      Hasil: logs/gpu-demo-<JOBID>.out
#===============================================================
#SBATCH --job-name=gpu-demo
#SBATCH --partition=gpu            # GANTI: nama partisi GPU H200 (cek: sinfo)
#SBATCH --gres=gpu:1          # 1 GPU H200 (beberapa cluster cukup: --gres=gpu:1)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00            # eksperimen ~1 menit + waktu memuat model
#SBATCH --output=logs/%x-%j.out    # %x = nama job, %j = job ID
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT=${1:-demo_llm_terjemahan.py}
shift || true                      # sisa argumen diteruskan ke script Python

echo "Job ${SLURM_JOB_ID} | $(hostname) | $(date)"
echo "GPU dialokasikan: ${CUDA_VISIBLE_DEVICES:-tidak ada} | script: ${SCRIPT} $*"

# 1) Environment (GANTI sesuai modul/conda di cluster Anda)
module purge
module load cuda/12.4              # contoh; cek: module avail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate demo-gpu            # berisi: torch, torchvision, transformers, sentencepiece, matplotlib, (vllm)

# 2) Model LLM dibaca dari cache bersama yang sudah diunduh di login node (unduh_model.sh)
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}   # GANTI ke lokasi penyimpanan cluster
export HF_HUB_OFFLINE=1            # compute node umumnya tanpa internet
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
