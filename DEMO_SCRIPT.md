# Demo Script: Running LLM Experiments on GPU (H200) and NPU (RNGD) with Slurm

Step-by-step commands for a recorded demo. Run everything from the **login node**.

| Node | Partition | Accelerator | Software |
|---|---|---|---|
| GPU | `gpu_general` | NVIDIA H200 (4 per node) | conda `vllm_env` (PyTorch, vLLM) |
| NPU | `npu_general` | FuriosaAI RNGD (4 per node) | `/apps/AI/venv_furiosa/furiosa_llm` |

Approximate timing: model loading takes ~75 s on GPU (vLLM) and ~40 s on NPU, then each benchmark runs for 60 s.

---

## 0. Preparation (before recording)

```bash
cd /path/ke/akf-demo
mkdir -p logs                                          # required: Slurm fails silently without it
rm -f logs/* hasil_*.tsv hasil_*.md                    # clear previous outputs
clear
```

## 1. Show the cluster

```bash
sinfo -o "%P %G %D %t"
```

## 2. Submit the jobs (same Python scripts on GPU and NPU)

```bash
sbatch job_gpu_h200.sbatch demo_llm_terjemahan.py                                   # translation, GPU (NLLB-200)
sbatch job_gpu_h200.sbatch demo_llm_percakapan.py --output hasil_percakapan_gpu.md  # chat, GPU (SEA-LION v5 9B, vLLM)
sbatch job_npu.sbatch      demo_llm_percakapan.py --output hasil_percakapan_npu.md  # chat, NPU (Llama-3.1-8B, furiosa-llm)
sbatch job_gpu_h200.sbatch demo_analisis_teks.py                                    # text embedding + clustering, GPU
```

Each command prints `Submitted batch job <JOBID>`. Note the IDs for the next steps.

> Tip: add `--duration 30` to any line for a shorter recording.

## 3. Show the queue

```bash
squeue -u $USER
```

## 4. Follow the jobs live (Ctrl+C stops watching; the job keeps running)

```bash
# GPU translation (~1.5 min in total)
tail -f logs/gpu-demo-<JOBID>.out

# GPU chat (hide noisy vLLM log lines)
tail -f logs/gpu-demo-<JOBID>.out | grep -vE "WARNING|INFO|Loading"

# NPU chat
tail -f logs/npu-demo-<JOBID>.out | grep -v INFO
```

> Tip: open two terminals side by side, one following a GPU job and one an NPU job.

## 5. Show hardware usage (while jobs run)

```bash
tail -n 5  logs/gpu-usage-<JOBID>.csv     # GPU utilization, memory, power (every 5 s)
tail -n 12 logs/npu-usage-<JOBID>.log     # NPU temperature and power (every 5 s)
```

## 6. Show the results

```bash
column -t -s $'\t' hasil_terjemahan.tsv | head -20    # translations into 6 languages
head -30 hasil_percakapan_gpu.md                      # GPU chat answers
head -30 hasil_percakapan_npu.md                      # NPU chat answers
grep -h Throughput logs/*-demo-*.out                  # GPU vs NPU speed side by side
```

## 7. (Optional) Live interactive chat

**On GPU:**

```bash
srun -p gpu_general --gres=gpu:1 -c 16 --mem=64G --time=00:30:00 --pty bash
source /path/ke/miniforge3/etc/profile.d/conda.sh
conda activate vllm_env
cd /path/ke/akf-demo
export HF_HUB_OFFLINE=1
python demo_llm_percakapan.py --mode interaktif
```

**On NPU:**

```bash
srun -p npu_general --gres=npu:1 -c 16 --mem=64G --time=00:30:00 --pty bash
source /apps/AI/venv_furiosa/furiosa_llm/bin/activate
cd /path/ke/akf-demo
python demo_llm_percakapan.py --mode interaktif
```

Type a question and press Enter. Type `reset` to clear the history, `keluar` to quit, then `exit` to leave the node.

---

## Notes

- **Regional-language translation:** show it on the GPU (NLLB). On the NPU, translation uses Llama-3.1-8B and works well only into English:
  `sbatch job_npu.sbatch demo_llm_terjemahan.py --target eng_Latn --output hasil_terjemahan_npu.tsv`
- **Text analysis** runs on GPU only. On the NPU it exits with a clear message.
- **Harmless warnings:** `linear_attn.norm.weight ... float32 into bfloat16` lines in GPU logs come from the SEA-LION v5 architecture.
- **Job status:** `sacct` may be unavailable (accounting database down). Use `squeue` and the log files instead.
- **Cancel a job:** `scancel <JOBID>`
- **Errors:** see `logs/<gpu|npu>-demo-<JOBID>.err`
