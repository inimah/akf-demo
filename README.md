# Demo Tutorial: Menjalankan Eksperimen di HPC (GPU H200 & NPU) dengan Slurm

Contoh untuk Sosialisasi ASEAN-Korea HPC Facility in Indonesia (AKF), BRIN, 24 November 2026.
Setiap demo berjalan sekitar 1 menit, dan **script Python yang sama** dapat dijalankan di GPU node
(NVIDIA H200, partisi `gpu_general`) maupun NPU node (FuriosaAI RNGD, partisi `npu_general`).
Deteksi akselerator dan lokasi model ada di `akselerator.py`.

| | GPU H200 (`job_gpu_h200.sbatch`) | NPU RNGD (`job_npu.sbatch`) |
|---|---|---|
| Environment | venv Python native (`requirements.txt`) atau conda `vllm_env` | `/apps/AI/venv_furiosa/furiosa_llm` |
| Terjemahan | NLLB-200 (atau `--mode llm`: vLLM + SEA-LION v5 9B) | furiosa-llm + Llama-3.1-8B-Instruct |
| Percakapan | vLLM + SEA-LION v5 9B | furiosa-llm + Llama-3.1-8B-Instruct |
| Analisis teks | multilingual-e5-small | tidak tersedia (perlu artefak terkompilasi) |


## Peta demo dan bidang riset

| Demo | Isi | Relevan untuk |
|---|---|---|
| `demo_llm_terjemahan.py` | **LLM penerjemahan** Indonesia → Inggris, Jawa, Sunda, Minang, Bali, Bugis, dll. | OR Arkeologi, Bahasa, dan Sastra, OR IPSH, layanan publik multibahasa, Riset Natural Language Processing/AI/Mesin Penerjemah  |
| `demo_llm_percakapan.py` | **LLM percakapan**: batch tanya-jawab | Riset pengembangan AI untuk Asisten riset/Chatbot |
| `demo_analisis_teks.py` | Embedding teks multibahasa + clustering tema (mis. aduan warga) | Riset Natural Language Processing/AI/Data Science |

## Langkah tutorial

```bash
# 0. Di login node: masuk ke folder demo dan siapkan konfigurasi lokal
cd akf-demo
cp .env.example .env              # isi path pribadi Anda di .env (tidak di-commit)
mkdir -p logs                     # WAJIB: tanpa folder ini, job gagal tanpa pesan

# 1. Environment Python (lihat bagian "Environment Python" di bawah)
#    GPU: venv Python native dari requirements.txt, atau conda env vllm_env
#    NPU: /apps/AI/venv_furiosa/furiosa_llm (furiosa-llm, disediakan admin)
source ~/venv-akf-demo/bin/activate       # bila memakai venv native

# 2. Unduh model ke folder lokal (sekali saja; compute node tanpa internet)
bash unduh_model.sh                        # menyimpan model ke MODEL_DIR dari .env

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
sacct -j <JOBID> --format=JobID,JobName,Partition,State,Elapsed,AllocTRES%40   # bila database accounting aktif
scancel <JOBID>                                              # batalkan bila perlu

# 6. Percakapan interaktif dengan LLM (sesi interaktif, bukan sbatch)
srun -p gpu_general --gres=gpu:1 --time=00:30:00 --pty bash
source muat_env.sh && source "$VENV/bin/activate"    # atau: conda activate "$CONDA_ENV"
export HF_HUB_OFFLINE=1
python demo_llm_percakapan.py --mode interaktif
#    atau di NPU:
srun -p npu_general --gres=npu:1 --time=00:30:00 --pty bash
source /apps/AI/venv_furiosa/furiosa_llm/bin/activate && python demo_llm_percakapan.py --mode interaktif
```

## Environment Python

### GPU: venv Python native (tanpa conda)

Butuh Python 3.12 (`/usr/bin/python3` di login node sudah 3.12) dan driver NVIDIA yang mendukung CUDA 13.
Paket minimal ada di `requirements.txt` (vllm, transformers, sentencepiece, huggingface_hub; torch ikut terpasang).
Total ukuran sekitar 10 GB, jadi sebaiknya taruh venv di folder scratch/proyek bila kuota home terbatas.

```bash
python3 -m venv /path/ke/venv-akf-demo
source /path/ke/venv-akf-demo/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# lalu di .env:  VENV=/path/ke/venv-akf-demo   (CONDA_ROOT tidak perlu diisi)
```

`job_gpu_h200.sbatch` memilih environment secara otomatis:

| Kondisi saat `sbatch` | Environment yang dipakai di job |
|---|---|
| venv sedang aktif (`source .../bin/activate`) | venv tersebut |
| `VENV=/path/venv sbatch ...` | venv tersebut |
| `VENV` diisi di `.env` (tanpa miniforge/conda) | venv tersebut |
| `CONDA_ROOT` + `CONDA_ENV` diisi di `.env` | conda env tersebut |
| tidak ada `VENV` maupun `CONDA_ROOT` | `python3` bawaan sistem (job berhenti dengan pesan jelas bila paket belum terpasang) |

Baris `Python: ...` di awal log job menunjukkan environment yang benar-benar dipakai.

### NPU: environment dari admin

furiosa-llm diambil dari package index FuriosaAI dan harus cocok dengan driver/firmware NPU,
jadi **jangan** dipasang dari `requirements.txt`. `job_npu.sbatch` selalu memakai
`/apps/AI/venv_furiosa/furiosa_llm` (ubah dengan `FURIOSA_VENV=...`).

### Konfigurasi lokal (`.env`)

Path pribadi (folder model, venv/conda) **tidak** ditulis di dalam script, tetapi di file `.env`
yang ada di `.gitignore`. Salin dari contoh lalu isi:

```bash
cp .env.example .env
chmod 600 .env
nano .env        # MODEL_DIR, lalu VENV atau CONDA_ROOT + CONDA_ENV (path absolut)
```

| Variabel | Fungsi |
|---|---|
| `MODEL_DIR` | folder model lokal (hasil `unduh_model.sh`), wajib untuk job GPU |
| `VENV` | venv Python native untuk job GPU |
| `CONDA_ROOT`, `CONDA_ENV` | conda untuk job GPU (bila tidak memakai `VENV`) |
| `NPU_MODEL_DIR`, `FURIOSA_VENV` | opsional, default sudah sesuai cluster AKF |

`.env` dibaca otomatis oleh `job_gpu_h200.sbatch`, `job_npu.sbatch`, `unduh_model.sh` (lewat `muat_env.sh`)
dan oleh `akselerator.py`. Variabel yang sudah di-export di shell diutamakan, misalnya
`MODEL_DIR=/path/lain sbatch job_gpu_h200.sbatch ...`.

Model NPU (artefak RNGD) sudah tersedia di `/apps/AI/LLM/models/NPU/furiosa-ai/` dan tidak perlu diunduh.

## Berkas hasil
- `logs/*.out`: log tiap job, ditambah `gpu-usage-*.csv` / `npu-usage-*.log` (utilisasi GPU / suhu-daya NPU tiap 5 detik)
- `hasil_terjemahan.tsv`: terjemahan per bahasa
- `hasil_percakapan.md`: jawaban LLM per bidang



