#!/bin/bash
#===============================================================
# Job Slurm untuk NPU node (FuriosaAI RNGD), dipakai oleh demo LLM.
# Script Python-nya SAMA dengan versi GPU; deteksi device otomatis (akselerator.py).
# Di RNGD, LLM dijalankan lewat furiosa-llm dengan artefak terkompilasi:
#   /apps/AI/LLM/models/NPU/furiosa-ai/Llama-3.1-8B-Instruct
#
# Pemakaian:
#   sbatch job_npu.sbatch                                       # default: demo_llm_terjemahan.py
#   sbatch job_npu.sbatch demo_llm_percakapan.py                # engine furiosa (otomatis)
#   (demo_analisis_teks.py hanya untuk GPU)
#
# Pantau: squeue -u $USER      Hasil: logs/npu-demo-<JOBID>.out
#===============================================================
#SBATCH --job-name=npu-demo
#SBATCH --partition=npu_general    # partisi NPU RNGD (cek: sinfo)
#SBATCH --gres=npu:1               # 1 dari 4 NPU RNGD per node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [ $# -eq 0 ]; then set -- demo_llm_terjemahan.py; fi
SCRIPT=$1; shift

echo "Job ${SLURM_JOB_ID} | $(hostname) | $(date)"
echo "NPU dialokasikan: ${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-lihat furiosa-smi}} | script: ${SCRIPT} $*"

# 1) Environment Python furiosa-llm (disediakan admin cluster); path pribadi opsional di .env
source "${SLURM_SUBMIT_DIR:-.}/muat_env.sh"
FURIOSA_VENV=${FURIOSA_VENV:-/apps/AI/venv_furiosa/furiosa_llm}
set +u
source "${FURIOSA_VENV}/bin/activate"   # berisi: furiosa-llm, furiosa-torch, torch, transformers
set -u

# 2) Lokasi model
export NPU_MODEL_DIR=${NPU_MODEL_DIR:-/apps/AI/LLM/models/NPU/furiosa-ai}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# 3) Info hardware + rekam suhu/daya NPU tiap 5 detik di latar belakang
furiosa-smi info
( while true; do date '+%F %T'; furiosa-smi info 2>/dev/null; sleep 5; done ) \
    > logs/npu-usage-${SLURM_JOB_ID}.log &
MONITOR_PID=$!
trap 'kill ${MONITOR_PID} 2>/dev/null || true' EXIT

# 4) Jalankan eksperimen
srun python "${SCRIPT}" "$@"

echo "Job selesai pada $(date)"
