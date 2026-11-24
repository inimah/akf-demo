#!/bin/bash
#===============================================================
# Jalankan SEKALI di LOGIN NODE (yang punya akses internet), sebelum submit job LLM.
# Compute node biasanya tidak bisa mengunduh, jadi model disimpan di cache bersama.
#
#   bash unduh_model.sh
#===============================================================
set -euo pipefail

export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache}   # samakan dengan di file sbatch
mkdir -p "$HF_HOME" logs

# Model gated perlu login dulu: huggingface-cli login   (token dari huggingface.co/settings/tokens)

MODEL=(
  facebook/nllb-200-distilled-600M     # terjemahan 200 bahasa, termasuk Jawa/Sunda/Minang/Bali/Aceh/Banjar/Bugis (~2.5 GB)
  Qwen/Qwen2.5-7B-Instruct             # LLM percakapan multibahasa, termasuk Indonesia (~15 GB)
  intfloat/multilingual-e5-small       # embedding teks multibahasa (~0.5 GB)
  # aisingapore/Gemma-SEA-LION-v3-9B-IT          # alternatif: LLM fokus Asia Tenggara
  # GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct   # alternatif: LLM fokus Indonesia + bahasa daerah (gated)
)

for m in "${MODEL[@]}"; do
  echo ">> Mengunduh $m"
  huggingface-cli download "$m" --exclude "*.pth" "original/*"
done

# Bobot ResNet-50 tidak perlu diunduh: demo_latih_citra.py melatih dari awal.
echo "Selesai. Lokasi cache: $HF_HOME"
du -sh "$HF_HOME"
