# Demo Tutorial: Menjalankan Eksperimen di HPC (GPU H200 & NPU) dengan Slurm

Contoh untuk Sosialisasi ASEAN-Korea HPC Facility in Indonesia (AKF), BRIN, 24 November 2026.
Setiap demo berjalan sekitar 1 menit, dan **script Python yang sama** dapat dijalankan di GPU node
(NVIDIA H200) maupun NPU node. 

## Peta demo dan bidang riset

| Demo | Isi | Relevan untuk |
|---|---|---|
| `demo_llm_terjemahan.py` | **LLM penerjemahan** Indonesia → Inggris, Jawa, Sunda, Minang, Bali, Bugis, dll. | OR Arkeologi, Bahasa, dan Sastra, OR IPSH, layanan publik multibahasa, Riset Natural Language Processing/AI/Mesin Penerjemah  |
| `demo_llm_percakapan.py` | **LLM percakapan**: batch tanya-jawab lintas bidang, multi-giliran, interaktif | Riset pengembangan AI untuk Asisten riset/Chatbot |
| `demo_analisis_teks.py` | Embedding teks multibahasa + clustering tema (mis. aduan warga) | Riset Natural Language Processing/AI/Data Science |

## Langkah tutorial

```bash
# 0. Di login node: salin folder ini lalu masuk
cd hpc-demo
mkdir -p logs                     # WAJIB: tanpa folder ini, job gagal tanpa pesan

# 1. Siapkan environment (sekali saja; nama env mengikuti file sbatch)
conda create -n demo-gpu python=3.11 -y && conda activate demo-gpu
pip install torch torchvision transformers sentencepiece matplotlib    # + vllm (opsional)
#    Untuk NPU: env demo-npu dengan torch + torch_npu yang versinya sesuai CANN di cluster

# 2. Unduh model LLM ke cache bersama (compute node biasanya tanpa internet)
bash unduh_model.sh

# 3. Cek partisi & sumber daya
sinfo -o "%P %G %D %t"

# 4. Submit: format umum ->  sbatch <file job> <script demo> [argumen]
sbatch job_gpu_h200.sbatch                                   # uji matmul
sbatch job_gpu_h200.sbatch demo_llm_terjemahan.py
sbatch job_npu.sbatch      demo_llm_terjemahan.py            # script sama, di NPU
sbatch job_gpu_h200.sbatch demo_llm_percakapan.py --engine vllm

# 5. Pantau & lihat hasil
squeue -u $USER
tail -f logs/gpu-demo-<JOBID>.out
sacct -j <JOBID> --format=JobID,JobName,Partition,State,Elapsed,AllocTRES%40
scancel <JOBID>                                              # batalkan bila perlu

# 6. Percakapan interaktif dengan LLM (sesi interaktif, bukan sbatch)
srun -p gpu --gres=gpu:h200:1 --time=00:30:00 --pty bash
python demo_llm_percakapan.py --mode interaktif
```

## Berkas hasil
- `logs/*.out`: log tiap job, ditambah `gpu-usage-*.csv` / `npu-usage-*.log` (utilisasi tiap 5 detik)
- `hasil_terjemahan.tsv`: terjemahan per bahasa
- `hasil_percakapan.md`: jawaban LLM per bidang


## Yang perlu disesuaikan dengan cluster AKF
- `--partition`, `--gres` (mis. `gpu:1`, `npu:1`) 
- Baris `module load`, `conda activate`, path `set_env.sh` CANN, serta `HF_HOME` (lokasi scratch/penyimpanan bersama)
- Model LLM dapat diganti via `--model`, misalnya LLM fokus Asia Tenggara/Indonesia
  (SEA-LION, Sahabat-AI; lihat `unduh_model.sh`). Pastikan lisensinya sesuai untuk penggunaan Anda.

