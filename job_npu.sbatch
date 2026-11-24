#!/bin/bash
#===============================================================
# Job Slurm untuk NPU node (Huawei Ascend), dipakai oleh SEMUA demo.
# Script Python-nya SAMA dengan versi GPU; deteksi device otomatis (akselerator.py).
#
# Pemakaian:
#   sbatch job_npu.sbatch                                       # default: demo_matmul.py --dtype fp16
#   sbatch job_npu.sbatch demo_dinamika_molekul.py
#   sbatch job_npu.sbatch demo_llm_terjemahan.py
#   sbatch job_npu.sbatch demo_llm_percakapan.py                # engine hf (transformers + torch_npu)
#
# Pantau: squeue -u $USER      Hasil: logs/npu-demo-<JOBID>.out
#===============================================================
#SBATCH --job-name=npu-demo
#SBATCH --partition=npu            # GANTI: nama partisi NPU (cek: sinfo)
#SBATCH --gres=npu:1               # 1 NPU (nama GRES bisa berbeda, cek: sinfo -o "%P %G")
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
##SBATCH --account=NAMA_PROYEK

set -euo pipefail

if [ $# -eq 0 ]; then set -- demo_llm_terjemahan.py --dtype fp16; fi
SCRIPT=$1; shift

echo "Job ${SLURM_JOB_ID} | $(hostname) | $(date)"
echo "NPU dialokasikan: ${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-lihat npu-smi}} | script: ${SCRIPT} $*"

# 1) Environment CANN + Python (GANTI path/modul sesuai cluster Anda)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate demo-npu            # berisi: torch + torch_npu (versi cocok dengan CANN), transformers, dst.

# 2) Cache model bersama (diunduh di login node)
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# 3) Info hardware + rekam pemakaian NPU tiap 5 detik di latar belakang
npu-smi info
( while true; do date '+%F %T'; npu-smi info -t usages -i 0 2>/dev/null; sleep 5; done ) \
    > logs/npu-usage-${SLURM_JOB_ID}.log &
MONITOR_PID=$!
trap 'kill ${MONITOR_PID} 2>/dev/null || true' EXIT

# 4) Jalankan eksperimen
srun python "${SCRIPT}" "$@"

echo "Job selesai pada $(date)"
