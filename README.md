# Demo Tutorial: Menjalankan Eksperimen di HPC (GPU H200 & NPU) dengan Slurm

Contoh untuk Sosialisasi ASEAN-Korea HPC Facility in Indonesia (AKF), BRIN, 24 November 2026.
Setiap demo berjalan sekitar 1 menit, dan **script Python yang sama** dapat dijalankan di GPU node
(NVIDIA H200, partisi `gpu_general`) maupun NPU node (FuriosaAI RNGD, partisi `npu_general`).
Deteksi akselerator dan lokasi model ada di `akselerator.py`.

| | GPU H200 (`job_gpu_h200.sbatch`) | NPU RNGD (`job_npu.sbatch`) |
|---|---|---|
| Environment | conda `vllm_env` | `/apps/AI/venv_furiosa/furiosa_llm` |
| Terjemahan | NLLB-200 (atau `--mode llm`: vLLM + SEA-LION v5 9B) | furiosa-llm + Llama-3.1-8B-Instruct |
| Percakapan | vLLM + SEA-LION v5 9B | furiosa-llm + Llama-3.1-8B-Instruct |
| Analisis teks | multilingual-e5-small | tidak tersedia (perlu artefak terkompilasi) |


## Peta demo dan bidang riset

| Demo | Isi | Relevan untuk |
|---|---|---|
| `demo_llm_terjemahan.py` | **LLM penerjemahan** Indonesia → Inggris, Jawa, Sunda, Minang, Bali, Bugis, dll. | OR Arkeologi, Bahasa, dan Sastra, OR IPSH, layanan publik multibahasa, Riset Natural Language Processing/AI/Mesin Penerjemah  |
| `demo_llm_percakapan.py` | **LLM percakapan**: batch tanya-jawab lintas bidang, multi-giliran, interaktif | Riset pengembangan AI untuk Asisten riset/Chatbot |
| `demo_analisis_teks.py` | Embedding teks multibahasa + clustering tema (mis. aduan warga) | Riset Natural Language Processing/AI/Data Science |

## Langkah tutorial

```bash
# 0. Di login node: masuk ke folder demo
cd akf-demo
mkdir -p logs                     # WAJIB: tanpa folder ini, job gagal tanpa pesan

# 1. Environment sudah tersedia:
#    GPU: conda env vllm_env (torch, transformers, vllm)
#    NPU: /apps/AI/venv_furiosa/furiosa_llm (furiosa-llm, disediakan admin)

# 2. Unduh model ke folder lokal (sekali saja; compute node tanpa internet)
bash unduh_model.sh

# 3. Cek partisi & sumber daya
sinfo -o "%P %G %D %t"

# 4. Submit: format umum ->  sbatch <file job> <script demo> [argumen]
sbatch job_gpu_h200.sbatch demo_llm_terjemahan.py
sbatch job_npu.sbatch      demo_llm_terjemahan.py            # script sama, di NPU
sbatch job_gpu_h200.sbatch demo_llm_percakapan.py
sbatch job_npu.sbatch      demo_llm_percakapan.py
sbatch job_gpu_h200.sbatch demo_analisis_teks.py

# 5. Pantau & lihat hasil
squeue -u $USER
tail -f logs/gpu-demo-<JOBID>.out
sacct -j <JOBID> --format=JobID,JobName,Partition,State,Elapsed,AllocTRES%40
scancel <JOBID>                                              # batalkan bila perlu

# 6. Percakapan interaktif dengan LLM (sesi interaktif, bukan sbatch)
srun -p gpu_general --gres=gpu:1 --time=00:30:00 --pty bash
conda activate vllm_env && python demo_llm_percakapan.py --mode interaktif
#    atau di NPU:
srun -p npu_general --gres=npu:1 --time=00:30:00 --pty bash
source /apps/AI/venv_furiosa/furiosa_llm/bin/activate && python demo_llm_percakapan.py --mode interaktif
```

## Berkas hasil
- `logs/*.out`: log tiap job, ditambah `gpu-usage-*.csv` / `npu-usage-*.log` (utilisasi GPU / suhu-daya NPU tiap 5 detik)
- `hasil_terjemahan.tsv`: terjemahan per bahasa
- `hasil_percakapan.md`: jawaban LLM per bidang



