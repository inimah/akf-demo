"""
Deteksi akselerator + lokasi model, dipakai oleh semua demo.

  GPU : NVIDIA H200 (partisi gpu_general)   -> PyTorch CUDA / vLLM
  NPU : FuriosaAI RNGD (partisi npu_general) -> furiosa-llm (artefak model yang sudah dikompilasi)
  CPU : cadangan bila tidak ada akselerator

Catatan: RNGD tidak menjalankan PyTorch "eager" seperti GPU. Di NPU, LLM dijalankan lewat
furiosa-llm memakai artefak siap pakai di /apps/AI/LLM/models/NPU/furiosa-ai/.
"""
import glob
import os
import shutil
import subprocess

import torch

# Folder model lokal (sudah diunduh, compute node tanpa internet). Bisa diganti lewat env MODEL_DIR.
MODEL_DIR = os.environ.get("MODEL_DIR", "/path/ke/scratch/anda/models")
NPU_MODEL_DIR = os.environ.get("NPU_MODEL_DIR", "/apps/AI/LLM/models/NPU/furiosa-ai")


def model_lokal(nama_folder, hf_id):
    """Pakai folder lokal bila ada, jika tidak pakai ID Hugging Face (butuh cache/internet)."""
    p = os.path.join(MODEL_DIR, nama_folder)
    return p if os.path.isdir(p) else hf_id


MODEL_NLLB = model_lokal("nllb-200-distilled-600M", "facebook/nllb-200-distilled-600M")
MODEL_E5 = model_lokal("multilingual-e5-small", "intfloat/multilingual-e5-small")
MODEL_CHAT_GPU = model_lokal("aisg-qwen-v5-9b-hf", "aisingapore/Qwen-SEA-LION-v5-9B")
MODEL_CHAT_NPU = os.path.join(NPU_MODEL_DIR, "Llama-3.1-8B-Instruct")


def ada_furiosa():
    return bool(glob.glob("/dev/rngd*")) or shutil.which("furiosa-smi") is not None and \
        subprocess.run(["furiosa-smi", "info"], capture_output=True).returncode == 0


def pilih_device():
    """Kembalikan (torch.device, jenis 'GPU'/'NPU'/'CPU', backend 'cuda'/'furiosa'/'cpu')."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "GPU", "cuda"
    if ada_furiosa():
        # Tensor PyTorch tetap di CPU; komputasi LLM dikirim ke RNGD oleh furiosa-llm.
        return torch.device("cpu"), "NPU", "furiosa"
    return torch.device("cpu"), "CPU", "cpu"


def nama_hardware(backend):
    if backend == "cuda":
        return f"{torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}"
    if backend == "furiosa":
        try:
            out = subprocess.run(["furiosa-smi", "info"], capture_output=True, text=True).stdout
            n = sum(1 for b in out.splitlines() if "| rngd" in b)
        except OSError:
            n = 0
        return f"FuriosaAI RNGD ({n or '?'} NPU terlihat)"
    return f"CPU ({os.cpu_count()} core)"


def cetak_info(judul, device, jenis, backend, **info):
    print("=" * 70)
    print(f" {judul}")
    print("=" * 70)
    print(f" Akselerator : {jenis} - {nama_hardware(backend)}")
    print(f" PyTorch     : {torch.__version__} | device: {device}")
    for k, v in info.items():
        print(f" {k:<12}: {v}")
    print("=" * 70, flush=True)


def sinkron(backend):
    if backend == "cuda":
        torch.cuda.synchronize()


def memori_puncak(backend):
    if backend == "cuda":
        gb = torch.cuda.max_memory_allocated() / 1e9
        if gb == 0:   # vLLM menjalankan model di proses terpisah -> tanya nvidia-smi
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                      "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
                                     capture_output=True, text=True).stdout
                return f"{int(out.split()[0]) / 1024:.1f} GB (nvidia-smi, termasuk cache KV vLLM)"
            except (OSError, ValueError, IndexError):
                return "n/a"
        return f"{gb:.1f} GB"
    return "n/a (lihat log furiosa-smi)" if backend == "furiosa" else "n/a"


_KUNCI_NPU = []   # file kunci tetap terbuka selama proses berjalan


def muat_llm_furiosa(model_id):
    """Muat LLM di 1 NPU RNGD yang bebas.

    Slurm di cluster ini tidak membatasi NPU yang terlihat per job (semua 4 NPU terlihat),
    sehingga dua job di node yang sama bisa berebut npu0 (EBUSY). Di sini tiap job
    'mengklaim' satu NPU lewat file kunci lokal node, dan mencoba NPU berikutnya bila sibuk.
    Bisa dipaksa lewat env FURIOSA_DEVICES, mis. FURIOSA_DEVICES=npu:2
    """
    import fcntl
    from furiosa_llm import LLM

    if os.environ.get("FURIOSA_DEVICES"):
        return LLM(model_id, devices=os.environ["FURIOSA_DEVICES"])
    out = subprocess.run(["furiosa-smi", "info"], capture_output=True, text=True).stdout
    kandidat = [b.split("|")[2].strip() for b in out.splitlines() if "| rngd" in b] or ["npu0"]
    for npu in kandidat:
        f = open(f"/tmp/akf-demo-{npu}.lock", "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.close()
            continue
        try:
            llm = LLM(model_id, devices=f"npu:{npu[3:]}")
        except Exception as e:  # noqa: BLE001
            f.close()
            if "EBUSY" in str(e):
                print(f"{npu} sedang dipakai proses lain, mencoba NPU berikutnya...", flush=True)
                continue
            raise
        _KUNCI_NPU.append(f)
        print(f"Memakai NPU: {npu}", flush=True)
        return llm
    raise SystemExit("Semua NPU di node ini sedang dipakai. Coba lagi nanti.")
