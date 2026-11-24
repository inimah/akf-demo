#!/bin/bash
#===============================================================
# Jalankan SEKALI di LOGIN NODE (yang punya akses internet), sebelum submit job.
# Compute node tidak bisa mengunduh, jadi model disimpan sebagai folder lokal di MODEL_DIR.
# Model NPU (artefak RNGD) sudah disediakan admin di /apps/AI/LLM/models/NPU/furiosa-ai/.
#
#   bash unduh_model.sh
#===============================================================
set -euo pipefail

export MODEL_DIR=${MODEL_DIR:-/path/ke/scratch/anda/models}  # samakan dengan sbatch
HF=${HF:-/path/ke/miniforge3/envs/vllm_env/bin/hf}
mkdir -p "$MODEL_DIR" logs

# Model gated perlu login dulu: hf auth login   (token dari huggingface.co/settings/tokens)
declare -A MODEL=(
  [nllb-200-distilled-600M]=facebook/nllb-200-distilled-600M   # terjemahan 200 bahasa (~2.5 GB)
  [aisg-qwen-v5-9b-hf]=aisingapore/Qwen-SEA-LION-v5-9B           # LLM percakapan Asia Tenggara (~19 GB)
  [multilingual-e5-small]=intfloat/multilingual-e5-small        # embedding teks multibahasa (~0.5 GB)
)

for folder in "${!MODEL[@]}"; do
  if [ -f "$MODEL_DIR/$folder/config.json" ]; then echo ">> Sudah ada: $folder"; continue; fi
  echo ">> Mengunduh ${MODEL[$folder]} -> $MODEL_DIR/$folder"
  "$HF" download "${MODEL[$folder]}" --local-dir "$MODEL_DIR/$folder" --exclude "*.pth" --exclude "original/*" --exclude "onnx/*"
done

echo "Selesai. Lokasi model: $MODEL_DIR"
du -sh "$MODEL_DIR"/{nllb-200-distilled-600M,aisg-qwen-v5-9b-hf,multilingual-e5-small}
