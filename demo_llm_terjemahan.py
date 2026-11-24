#!/usr/bin/env python3
"""
DEMO 4 - Inferensi LLM untuk PENERJEMAHAN, termasuk bahasa daerah Indonesia.
Relevan untuk: OR Arkeologi, Bahasa, dan Sastra (korpus bahasa daerah, naskah),
OR Ilmu Pengetahuan Sosial dan Humaniora, layanan publik multibahasa, fakultas
ilmu budaya/ilmu komputer.

Model default: NLLB-200 (Meta, model terjemahan khusus, 200 bahasa) yang mencakup
Indonesia, Inggris, Jawa, Sunda, Minangkabau, Bali, Aceh, Banjar, Bugis.
Mode --mode llm memakai LLM percakapan (instruct) dengan prompt terjemahan.

Model harus diunduh dulu di login node (lihat unduh_model.sh), karena compute node
umumnya tanpa akses internet.

    python demo_llm_terjemahan.py --duration 60
    python demo_llm_terjemahan.py --target jav_Latn sun_Latn --input teks.txt
    python demo_llm_terjemahan.py --mode llm --model Qwen/Qwen2.5-7B-Instruct
"""
import argparse
import time

import torch

from akselerator import cetak_info, memori_puncak_gb, pilih_device, sinkron

BAHASA = {
    "ind_Latn": "Indonesia", "eng_Latn": "Inggris", "jav_Latn": "Jawa", "sun_Latn": "Sunda",
    "min_Latn": "Minangkabau", "ban_Latn": "Bali", "ace_Latn": "Aceh", "bjn_Latn": "Banjar",
    "bug_Latn": "Bugis", "zsm_Latn": "Melayu", "kor_Hang": "Korea",
}

KALIMAT_CONTOH = [
    "Petani diimbau menanam varietas padi tahan kekeringan pada musim kemarau tahun ini.",
    "Fasilitas komputasi berkinerja tinggi membantu peneliti memprediksi cuaca ekstrem lebih cepat.",
    "Segera periksakan anak ke puskesmas jika demam tinggi lebih dari tiga hari.",
    "Naskah kuno ini menceritakan perdagangan rempah di pelabuhan pada abad keenam belas.",
    "Warga pesisir diminta menjauhi pantai setelah terjadi gempa kuat.",
    "Kerja sama riset ASEAN dan Korea membuka peluang bagi mahasiswa dan dosen di Indonesia.",
]


def baca_input(path):
    if not path:
        return KALIMAT_CONTOH
    with open(path, encoding="utf-8") as f:
        return [b.strip() for b in f if b.strip()]


def muat_nllb(model_id, device, dtype):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, src_lang="ind_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

    @torch.inference_mode()
    def terjemah(kalimat, tgt):
        enc = tok(kalimat, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                             max_new_tokens=128, num_beams=4)
        return tok.batch_decode(out, skip_special_tokens=True), out.numel()
    return terjemah


def muat_llm(model_id, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

    @torch.inference_mode()
    def terjemah(kalimat, tgt):
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": "Anda penerjemah profesional. Jawab hanya dengan hasil terjemahan."},
             {"role": "user", "content": f"Terjemahkan ke bahasa {BAHASA.get(tgt, tgt)}:\n{k}"}],
            tokenize=False, add_generation_prompt=True) for k in kalimat]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        out = model.generate(**enc, max_new_tokens=128, do_sample=False)
        baru = out[:, enc["input_ids"].shape[1]:]
        return [t.strip() for t in tok.batch_decode(baru, skip_special_tokens=True)], baru.numel()
    return terjemah


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=60, help="lama uji throughput (detik)")
    p.add_argument("--mode", choices=["nllb", "llm"], default="nllb")
    p.add_argument("--model", default=None)
    p.add_argument("--target", nargs="+",
                   default=["eng_Latn", "jav_Latn", "sun_Latn", "min_Latn", "ban_Latn", "bug_Latn"])
    p.add_argument("--input", default=None, help="file teks, satu kalimat per baris (bahasa Indonesia)")
    p.add_argument("--output", default="hasil_terjemahan.tsv")
    args = p.parse_args()

    device, jenis, backend = pilih_device()
    dtype = {"GPU": torch.bfloat16, "NPU": torch.float16, "CPU": torch.float32}[jenis]
    model_id = args.model or ("facebook/nllb-200-distilled-600M" if args.mode == "nllb"
                              else "Qwen/Qwen2.5-7B-Instruct")
    kalimat = baca_input(args.input)
    cetak_info("LLM penerjemahan", device, jenis, backend, Model=model_id,
               Target=", ".join(BAHASA.get(t, t) for t in args.target), Kalimat=len(kalimat))

    t0 = time.time()
    terjemah = (muat_nllb if args.mode == "nllb" else muat_llm)(model_id, device, dtype)
    sinkron(backend)
    print(f"Model dimuat dalam {time.time() - t0:.1f} s\n", flush=True)

    # 1) Terjemahkan semua kalimat ke setiap bahasa target, simpan ke TSV
    baris = []
    for tgt in args.target:
        hasil, _ = terjemah(kalimat, tgt)
        print(f"--- {BAHASA.get(tgt, tgt)} ---")
        for src, h in zip(kalimat, hasil):
            print(f"  {h}")
            baris.append(f"{tgt}\t{src}\t{h}")
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("bahasa\tsumber\tterjemahan\n" + "\n".join(baris) + "\n")
    print(f"\nHasil disimpan: {args.output}")

    # 2) Uji throughput: ulangi batch terjemahan sampai durasi habis
    batch = kalimat * max(1, 32 // len(kalimat))
    n_kal, n_tok, mulai = 0, 0, time.time()
    while time.time() - mulai < args.duration:
        _, tok = terjemah(batch, args.target[n_kal // len(batch) % len(args.target)])
        n_kal += len(batch)
        n_tok += tok
    sinkron(backend)
    total = time.time() - mulai
    print(f"Throughput: {n_kal / total:.1f} kalimat/s (~{n_tok / total:,.0f} token keluaran/s)"
          f" selama {total:.0f} s, memori puncak {memori_puncak_gb(backend):.1f} GB")


if __name__ == "__main__":
    main()
