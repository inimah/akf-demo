#!/usr/bin/env python3
"""
DEMO 6 - Analisis teks: embedding multibahasa + pengelompokan (clustering) tema.
Relevan untuk: OR Ilmu Pengetahuan Sosial dan Humaniora, OR Tata Kelola Pemerintahan,
Ekonomi, dan Kesejahteraan Masyarakat (aduan warga, survei terbuka, berita, kebijakan),
fakultas ekonomi/bisnis, FISIP, kesehatan masyarakat.

Teks diubah menjadi vektor oleh model embedding (multilingual-e5), lalu dikelompokkan
dengan K-Means yang dihitung langsung di akselerator. Ganti --input dengan file CSV/TXT
Anda sendiri (satu teks per baris).

    python demo_analisis_teks.py --duration 60 --k 4
"""
import argparse
import time

import torch
import torch.nn.functional as F

from akselerator import MODEL_E5, cetak_info, pilih_device, sinkron

CONTOH = [
    "Harga beras di pasar naik terus sejak bulan lalu.", "Minyak goreng makin mahal, warung kecil kesulitan.",
    "Harga cabai melonjak menjelang hari raya.", "Biaya kebutuhan pokok sudah tidak terjangkau buruh harian.",
    "Jalan desa rusak parah dan berlubang saat musim hujan.", "Jembatan penghubung antar kampung belum diperbaiki.",
    "Lampu penerangan jalan banyak yang mati.", "Saluran drainase tersumbat sehingga sering banjir.",
    "Antrean di puskesmas sangat panjang setiap pagi.", "Stok obat di posyandu sering kosong.",
    "Tenaga kesehatan di daerah terpencil masih kurang.", "Pendaftaran BPJS untuk lansia perlu dipermudah.",
    "Sinyal internet di sekolah sangat lemah untuk belajar daring.", "Guru honorer belum menerima gaji tiga bulan.",
    "Buku pelajaran di perpustakaan desa sudah usang.", "Beasiswa untuk mahasiswa dari keluarga miskin perlu ditambah.",
]


def kmeans(x, k, iterasi=50, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    pusat = x[torch.randperm(x.shape[0], generator=g)[:k].to(x.device)].clone()
    for _ in range(iterasi):
        label = (x @ pusat.T).argmax(1)                     # kemiripan kosinus (vektor sudah dinormalisasi)
        for j in range(k):
            anggota = x[label == j]
            if len(anggota):
                pusat[j] = F.normalize(anggota.mean(0), dim=0)
    return label


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--model", default=MODEL_E5)
    p.add_argument("--input", default=None, help="file teks, satu dokumen per baris")
    p.add_argument("--k", type=int, default=4, help="jumlah klaster/tema")
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()

    from transformers import AutoModel, AutoTokenizer
    device, jenis, backend = pilih_device()
    if jenis == "NPU":
        raise SystemExit("Demo embedding belum tersedia untuk NPU FuriosaAI RNGD (perlu artefak terkompilasi). "
                         "Jalankan di GPU: sbatch job_gpu_h200.sbatch demo_analisis_teks.py")
    teks = CONTOH
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            teks = [b.strip() for b in f if b.strip()]
    cetak_info("Embedding teks + clustering tema", device, jenis, backend,
               Model=args.model, Dokumen=len(teks), Klaster=args.k)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    amp = jenis != "CPU"

    @torch.inference_mode()
    def embed(daftar):
        hasil = []
        for i in range(0, len(daftar), args.batch):
            enc = tok([f"query: {t}" for t in daftar[i:i + args.batch]], padding=True, truncation=True,
                      max_length=256, return_tensors="pt").to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            hasil.append(F.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=-1))   # mean pooling
        return torch.cat(hasil)

    emb = embed(teks)
    label = kmeans(emb, args.k).cpu().tolist()
    for j in range(args.k):
        print(f"\n== Tema {j + 1} ==")
        for t, lb in zip(teks, label):
            if lb == j:
                print(f"  - {t}")

    # Uji throughput: berapa dokumen per detik yang bisa di-embed (mis. untuk jutaan aduan/berita)
    banyak = teks * max(1, 2048 // len(teks))
    n, mulai = 0, time.time()
    while time.time() - mulai < args.duration:
        embed(banyak)
        n += len(banyak)
    sinkron(backend)
    total = time.time() - mulai
    print(f"\nThroughput embedding: {n / total:,.0f} dokumen/s selama {total:.0f} s"
          f" (1 juta dokumen ~ {1e6 / (n / total) / 60:.1f} menit)")


if __name__ == "__main__":
    main()
