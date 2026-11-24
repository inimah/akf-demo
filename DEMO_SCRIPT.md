# Skrip Demo: Menjalankan Eksperimen LLM di GPU (H200) dan NPU (RNGD) dengan Slurm

Perintah langkah demi langkah untuk demo yang direkam. Jalankan semuanya dari **login node**.

| Node | Partisi | Akselerator | Perangkat lunak |
|---|---|---|---|
| GPU | `gpu_general` | NVIDIA H200 (4 per node) | venv native (`requirements.txt`) atau conda `vllm_env` |
| NPU | `npu_general` | FuriosaAI RNGD (4 per node) | `/apps/AI/venv_furiosa/furiosa_llm` |

Perkiraan waktu: memuat model butuh ~75 detik di GPU (vLLM) dan ~40 detik di NPU, lalu setiap uji throughput berjalan selama 60 detik.

---

## 0. Persiapan (sebelum merekam)

```bash
cd /path/ke/akf-demo                                   # folder hasil clone repo ini
cp -n .env.example .env                                # sekali saja, lalu isi path Anda di .env
mkdir -p logs                                          # wajib: tanpa folder ini, job Slurm gagal tanpa pesan
rm -f logs/* hasil_*.tsv hasil_*.md                    # hapus hasil sebelumnya
clear
```

## 1. Tampilkan kondisi cluster

```bash
sinfo -o "%P %G %D %t"
```

## 2. Submit job (script Python yang sama di GPU dan NPU)

```bash
sbatch job_gpu_h200.sbatch demo_llm_terjemahan.py                                   # terjemahan, GPU (NLLB-200)
sbatch job_gpu_h200.sbatch demo_llm_percakapan.py --output hasil_percakapan_gpu.md  # percakapan, GPU (SEA-LION v5 9B, vLLM)
sbatch job_npu.sbatch      demo_llm_percakapan.py --output hasil_percakapan_npu.md  # percakapan, NPU (Llama-3.1-8B, furiosa-llm)
sbatch job_gpu_h200.sbatch demo_analisis_teks.py                                    # embedding teks + clustering, GPU
```

Setiap perintah menampilkan `Submitted batch job <JOBID>`. Catat ID tersebut untuk langkah berikutnya.

> Tips: tambahkan `--duration 30` di baris mana pun agar rekaman lebih singkat.

## 3. Tampilkan antrean job

```bash
squeue -u $USER
```

## 4. Pantau job secara langsung (Ctrl+C berhenti memantau; job tetap berjalan)

```bash
# Terjemahan di GPU (~1,5 menit total)
tail -f logs/gpu-demo-<JOBID>.out

# Percakapan di GPU (menyembunyikan baris log vLLM yang ramai)
tail -f logs/gpu-demo-<JOBID>.out | grep -vE "WARNING|INFO|Loading"

# Percakapan di NPU
tail -f logs/npu-demo-<JOBID>.out | grep -v INFO
```

> Tips: buka dua terminal berdampingan, satu memantau job GPU dan satu memantau job NPU.

## 5. Tampilkan pemakaian hardware (selama job berjalan)

```bash
tail -n 5  logs/gpu-usage-<JOBID>.csv     # utilisasi, memori, dan daya GPU (tiap 5 detik)
tail -n 12 logs/npu-usage-<JOBID>.log     # suhu dan daya NPU (tiap 5 detik)
```

## 6. Tampilkan hasil

```bash
column -t -s $'\t' hasil_terjemahan.tsv | head -20    # terjemahan ke 6 bahasa
head -30 hasil_percakapan_gpu.md                      # jawaban percakapan dari GPU
head -30 hasil_percakapan_npu.md                      # jawaban percakapan dari NPU
grep -h Throughput logs/*-demo-*.out                  # perbandingan kecepatan GPU vs NPU
```

## 7. (Opsional) Percakapan interaktif langsung

**Di GPU:**

```bash
srun -p gpu_general --gres=gpu:1 -c 16 --mem=64G --time=00:30:00 --pty bash
cd /path/ke/akf-demo
source muat_env.sh                                     # baca path dari .env
source "$VENV/bin/activate"                            # venv native, ATAU:
# source "$CONDA_ROOT/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"
export HF_HUB_OFFLINE=1
python demo_llm_percakapan.py --mode interaktif
```

**Di NPU:**

```bash
srun -p npu_general --gres=npu:1 -c 16 --mem=64G --time=00:30:00 --pty bash
source /apps/AI/venv_furiosa/furiosa_llm/bin/activate
cd /path/ke/akf-demo
python demo_llm_percakapan.py --mode interaktif
```

Ketik pertanyaan lalu tekan Enter. Ketik `reset` untuk menghapus riwayat, `keluar` untuk berhenti, lalu `exit` untuk keluar dari node.

---

## Catatan

- **Terjemahan bahasa daerah:** tampilkan di GPU (NLLB). Di NPU, terjemahan memakai Llama-3.1-8B dan hanya bagus untuk bahasa Inggris:
  `sbatch job_npu.sbatch demo_llm_terjemahan.py --target eng_Latn --output hasil_terjemahan_npu.tsv`
- **Analisis teks** hanya berjalan di GPU. Di NPU, script berhenti dengan pesan yang jelas.
- **Peringatan yang bisa diabaikan:** baris `linear_attn.norm.weight ... float32 into bfloat16` di log GPU berasal dari arsitektur SEA-LION v5.
- **Status job:** `sacct` mungkin tidak tersedia (database accounting sedang mati). Gunakan `squeue` dan berkas log.
- **Membatalkan job:** `scancel <JOBID>`
- **Error:** lihat `logs/<gpu|npu>-demo-<JOBID>.err`
