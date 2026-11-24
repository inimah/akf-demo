#!/usr/bin/env python3
"""
DEMO 5 - Inferensi LLM untuk PERCAKAPAN (chat), batch maupun interaktif.
Relevan untuk semua OR: asisten riset, tanya-jawab dokumen, ringkasan literatur,
chatbot layanan publik (OR Tata Kelola Pemerintahan, Ekonomi, dan Kesejahteraan
Masyarakat), bahan ajar (perguruan tinggi).

Tiga mode:
  --mode batch      : sekumpulan pertanyaan lintas bidang dijawab sekaligus (default, cocok untuk sbatch)
  --mode multiturn  : contoh percakapan bersambung (riwayat chat dibawa ke giliran berikutnya)
  --mode interaktif : ketik pertanyaan sendiri. Jalankan di sesi interaktif:
                      srun -p gpu_general --gres=gpu:1 --time=00:30:00 --pty bash
                      srun -p npu_general --gres=npu:1 --time=00:30:00 --pty bash
                      python demo_llm_percakapan.py --mode interaktif

Engine (--engine auto memilih otomatis):
  vllm    : vLLM di GPU H200, model default SEA-LION v5 9B (aisingapore)
  furiosa : furiosa-llm di NPU FuriosaAI RNGD, artefak Llama-3.1-8B-Instruct
  hf      : Hugging Face transformers (GPU/CPU; lebih lambat)

    python demo_llm_percakapan.py --duration 60
"""
import argparse
import time

import torch

from akselerator import MODEL_CHAT_GPU, MODEL_CHAT_NPU, cetak_info, memori_puncak, pilih_device, sinkron

SISTEM = ("Anda adalah asisten riset untuk peneliti di Indonesia. Jawab dalam bahasa Indonesia "
          "yang jelas dan ringkas (maksimal 5 kalimat).")

PERTANYAAN = [
    ("Pertanian & Pangan", "Apa strategi adaptasi budidaya padi terhadap El Nino?"),
    ("Kesehatan", "Jelaskan secara singkat perbedaan demam berdarah dan tifus untuk edukasi masyarakat."),
    ("Kebumian & Maritim", "Mengapa gelombang tsunami membesar saat mendekati pantai?"),
    ("Nanoteknologi & Material", "Apa keunggulan material perovskit untuk sel surya?"),
    ("Energi", "Sebutkan tantangan utama integrasi PLTS atap ke jaringan listrik."),
    ("Hayati & Lingkungan", "Bagaimana DNA lingkungan (eDNA) dipakai untuk memantau keanekaragaman hayati?"),
    ("Penerbangan & Antariksa", "Apa fungsi satelit penginderaan jauh untuk mitigasi bencana?"),
    ("Bahasa & Sastra", "Bagaimana AI dapat membantu pelestarian bahasa daerah yang terancam punah?"),
    ("Tata Kelola & Ekonomi", "Indikator apa saja yang biasa dipakai untuk mengukur kesejahteraan masyarakat?"),
    ("Komputasi", "Kapan sebaiknya peneliti memakai HPC dibanding laptop biasa?"),
]

MULTITURN = [
    "Saya ingin meneliti prediksi hasil panen jagung dari citra satelit. Data apa yang saya perlukan?",
    "Model machine learning apa yang cocok untuk itu?",
    "Buatkan rencana eksperimen 3 langkah untuk dijalankan di HPC.",
]


class ChatHF:
    def __init__(self, model_id, device, dtype):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def jawab(self, daftar_pesan, max_tokens=256):
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False) for m in daftar_pesan]
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=True,
                                  temperature=0.7, top_p=0.9, pad_token_id=self.tok.pad_token_id)
        baru = out[:, enc["input_ids"].shape[1]:]
        n_tok = int((baru != self.tok.pad_token_id).sum())
        return [t.strip() for t in self.tok.batch_decode(baru, skip_special_tokens=True)], n_tok


class ChatVLLM:
    """vLLM (GPU) dan furiosa-llm (NPU RNGD) punya API .chat() yang sama."""
    def __init__(self, model_id, device, dtype, furiosa=False):
        if furiosa:
            from furiosa_llm import SamplingParams
            from akselerator import muat_llm_furiosa
            self.llm, self.kw = muat_llm_furiosa(model_id), {}
            self.greedy = True   # sampling acak (temperature/top_p) kadang panic 'InvalidWeight' di furiosa-llm 2026.2
        else:
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096)
            self.kw = {"use_tqdm": False, "chat_template_kwargs": {"enable_thinking": False}}
            self.greedy = False
        self.SP = SamplingParams

    def jawab(self, daftar_pesan, max_tokens=256):
        sp = (self.SP(max_tokens=max_tokens, temperature=0) if self.greedy
              else self.SP(max_tokens=max_tokens, temperature=0.7, top_p=0.9))
        outs = self.llm.chat(daftar_pesan, sp, **self.kw)
        return [o.outputs[0].text.strip() for o in outs], sum(len(o.outputs[0].token_ids) for o in outs)


def pesan(teks, riwayat=None):
    return [{"role": "system", "content": SISTEM}] + (riwayat or []) + [{"role": "user", "content": teks}]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=60, help="lama uji throughput (detik)")
    p.add_argument("--mode", choices=["batch", "multiturn", "interaktif"], default="batch")
    p.add_argument("--engine", choices=["auto", "hf", "vllm", "furiosa"], default="auto")
    p.add_argument("--model", default=None, help="default: SEA-LION v5 9B (GPU), Llama-3.1-8B (NPU)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--output", default="hasil_percakapan.md")
    args = p.parse_args()

    device, jenis, backend = pilih_device()
    dtype = {"GPU": torch.bfloat16, "NPU": torch.float16, "CPU": torch.float32}[jenis]
    if args.engine == "auto":
        args.engine = {"GPU": "vllm", "NPU": "furiosa", "CPU": "hf"}[jenis]
    if (args.engine == "furiosa") != (jenis == "NPU"):
        raise SystemExit(f"Engine '{args.engine}' tidak cocok dengan akselerator {jenis}.")
    args.model = args.model or (MODEL_CHAT_NPU if jenis == "NPU" else MODEL_CHAT_GPU)
    cetak_info("LLM percakapan", device, jenis, backend, Model=args.model,
               Engine=args.engine, Mode=args.mode)

    t0 = time.time()
    if args.engine == "hf":
        chat = ChatHF(args.model, device, dtype)
    else:
        chat = ChatVLLM(args.model, device, dtype, furiosa=args.engine == "furiosa")
    print(f"Model dimuat dalam {time.time() - t0:.1f} s\n", flush=True)

    if args.mode == "interaktif":
        riwayat = []
        print("Ketik pertanyaan (kosong / 'keluar' untuk berhenti, 'reset' untuk hapus riwayat).")
        while True:
            q = input("\nAnda > ").strip()
            if q in ("", "keluar", "exit"):
                break
            if q == "reset":
                riwayat = []
                continue
            [j], _ = chat.jawab([pesan(q, riwayat)], args.max_tokens)
            print(f"\nAsisten > {j}")
            riwayat += [{"role": "user", "content": q}, {"role": "assistant", "content": j}]
        return

    if args.mode == "multiturn":
        riwayat = []
        for q in MULTITURN:
            [j], _ = chat.jawab([pesan(q, riwayat)], args.max_tokens)
            print(f"\n[Pengguna] {q}\n[Asisten ] {j}")
            riwayat += [{"role": "user", "content": q}, {"role": "assistant", "content": j}]
        return

    # Mode batch: semua pertanyaan dijawab dalam satu batch, hasil ke Markdown
    t0 = time.time()
    jawaban, n_tok = chat.jawab([pesan(q) for _, q in PERTANYAAN], args.max_tokens)
    sinkron(backend)
    dt = time.time() - t0
    with open(args.output, "w", encoding="utf-8") as f:
        for (bidang, q), j in zip(PERTANYAAN, jawaban):
            print(f"\n### [{bidang}] {q}\n{j}")
            f.write(f"### [{bidang}] {q}\n\n{j}\n\n")
    print(f"\n{len(PERTANYAAN)} jawaban, {n_tok} token dalam {dt:.1f} s ({n_tok / dt:,.0f} token/s)."
          f" Disimpan: {args.output}")

    # Uji throughput: ulangi batch sampai durasi habis
    total_tok, putaran, mulai = 0, 0, time.time()
    while time.time() - mulai < args.duration:
        _, n = chat.jawab([pesan(q) for _, q in PERTANYAAN], args.max_tokens)
        total_tok += n
        putaran += 1
        print(f"[{time.time() - mulai:5.1f} s] putaran {putaran}: total {total_tok:,} token", flush=True)
    sinkron(backend)
    total = time.time() - mulai
    print(f"Throughput: {total_tok / total:,.0f} token/s (batch {len(PERTANYAAN)}),"
          f" memori puncak {memori_puncak(backend)}")


if __name__ == "__main__":
    main()
