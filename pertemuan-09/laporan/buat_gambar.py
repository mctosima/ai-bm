"""Membuat gambar dan ringkasan statistik dari hasil/hasil.json untuk laporan Typst.

Jalankan dari root repo:  uv run python pertemuan-09/laporan/buat_gambar.py
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

FOLDER = Path(__file__).resolve().parent
FOLDER_GAMBAR = FOLDER / "gambar"
FOLDER_GAMBAR.mkdir(exist_ok=True)

data = json.loads((FOLDER / "hasil" / "hasil.json").read_text())
MODEL = ["CNN-2", "CNN-3"]
KODE_KELAS = ["N", "S", "V", "F", "Q"]

# Palet kategorikal (slot 1 & 2, sudah lolos validator) dan ramp sekuensial biru
WARNA = {"CNN-2": "#2a78d6", "CNN-3": "#eb6834"}
TINTA, TINTA_2, GRID, PERMUKAAN = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
RAMP_BIRU = LinearSegmentedColormap.from_list(
    "biru", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": TINTA_2,
    "axes.labelcolor": TINTA,
    "xtick.color": TINTA_2,
    "ytick.color": TINTA_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.facecolor": PERMUKAAN,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

LEBAR_KOLOM = 3.5   # inci, satu kolom IEEE
LEBAR_HALAMAN = 7.16


def koma(teks):
    """Desimal ala Indonesia: 0.85 → 0,85."""
    return teks.replace(".", ",")


SUMBU_KOMA = FuncFormatter(lambda nilai, _: koma(f"{nilai:g}"))


def per_model(nama):
    return [h for h in data["hasil"] if h["model"] == nama]


# ---------------------------------------------------------------- ringkasan.json
def mean_std(nilai):
    nilai = np.asarray(nilai, dtype=float)
    return {"mean": float(nilai.mean()), "std": float(nilai.std(ddof=1))}


ringkasan = {"pengaturan": data["pengaturan"], "model": {}}
for nama in MODEL:
    run = per_model(nama)
    ringkasan["model"][nama] = {
        "jumlah_parameter": run[0]["jumlah_parameter"],
        "akurasi": mean_std([h["akurasi"] for h in run]),
        "f1_makro": mean_std([h["f1_makro"] for h in run]),
        "waktu_latih_detik": mean_std([h["waktu_latih_detik"] for h in run]),
        "presisi": [mean_std(v) for v in zip(*[h["presisi"] for h in run])],
        "recall": [mean_std(v) for v in zip(*[h["recall"] for h in run])],
        "f1": [mean_std(v) for v in zip(*[h["f1"] for h in run])],
    }

# Selisih berpasangan per seed (CNN-3 − CNN-2): seed sama → urutan batch & data sama
selisih = {}
for metrik in ["akurasi", "f1_makro"]:
    a = {h["seed"]: h[metrik] for h in per_model("CNN-2")}
    b = {h["seed"]: h[metrik] for h in per_model("CNN-3")}
    d = [b[s] - a[s] for s in data["pengaturan"]["seeds"]]
    selisih[metrik] = {**mean_std(d), "per_seed": d, "cnn3_lebih_baik": sum(x > 0 for x in d)}
ringkasan["selisih_berpasangan"] = selisih
(FOLDER / "hasil" / "ringkasan.json").write_text(json.dumps(ringkasan, indent=1))


# ---------------------------------------------------------------- Gambar 1: kurva belajar
epoch = np.arange(1, data["pengaturan"]["epoch"] + 1)
fig, (atas, bawah) = plt.subplots(2, 1, figsize=(LEBAR_KOLOM, 3.4), sharex=True)
for nama in MODEL:
    for ax, kunci in [(atas, "loss_latih"), (bawah, "f1_makro_uji")]:
        kurva = np.array([h["riwayat"][kunci] for h in per_model(nama)])
        rata, sd = kurva.mean(0), kurva.std(0, ddof=1)
        ax.fill_between(epoch, rata - sd, rata + sd, color=WARNA[nama], alpha=0.18, linewidth=0)
        ax.plot(epoch, rata, color=WARNA[nama], linewidth=1.4, marker="o", markersize=3, label=nama)
atas.yaxis.set_major_formatter(SUMBU_KOMA)
bawah.yaxis.set_major_formatter(SUMBU_KOMA)
atas.set_ylabel("Loss latih")
atas.set_title("(a) Cross-entropy pada data latih", loc="left", color=TINTA)
bawah.set_ylabel("F1 makro uji")
bawah.set_title("(b) F1 makro pada data uji", loc="left", color=TINTA)
bawah.set_xlabel("Epoch")
bawah.set_xticks(epoch)
bawah.set_xlim(0.7, epoch[-1] + 0.3)
atas.legend(frameon=False, loc="upper right")
fig.tight_layout(h_pad=0.8)
fig.savefig(FOLDER_GAMBAR / "kurva_belajar.png")
plt.close(fig)


# ---------------------------------------------------------------- Gambar 2: recall per kelas
fig, ax = plt.subplots(figsize=(LEBAR_KOLOM, 2.1))
x = np.arange(len(KODE_KELAS))
lebar = 0.36
for i, nama in enumerate(MODEL):
    rec = ringkasan["model"][nama]["recall"]
    rata = [r["mean"] for r in rec]
    sd = [r["std"] for r in rec]
    posisi = x + (i - 0.5) * (lebar + 0.02)   # celah 2 % antar-batang
    ax.bar(posisi, rata, width=lebar, color=WARNA[nama], label=nama,
           yerr=sd, error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": TINTA})
ax.set_xticks(x, [f"{k}" for k in KODE_KELAS])
ax.set_xlabel("Kelas")
ax.set_ylabel("Recall uji")
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(SUMBU_KOMA)
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
fig.tight_layout()
fig.savefig(FOLDER_GAMBAR / "recall_per_kelas.png")
plt.close(fig)


# ---------------------------------------------------------------- Gambar 3: confusion matrix (seed 42)
fig, axes = plt.subplots(1, 2, figsize=(LEBAR_HALAMAN, 2.9))
for ax, nama in zip(axes, MODEL):
    cm = np.array(next(h for h in per_model(nama) if h["seed"] == 42)["confusion_matrix"])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    gambar = ax.imshow(cm_norm, cmap=RAMP_BIRU, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            teks = f"{cm[i, j]}\n({koma(f'{cm_norm[i, j]:.1%}')})"
            ax.text(j, i, teks, ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if cm_norm[i, j] > 0.55 else TINTA)
    ax.set_xticks(range(5), KODE_KELAS)
    ax.set_yticks(range(5), KODE_KELAS)
    ax.set_xlabel("Kelas tebakan")
    ax.set_ylabel("Kelas sebenarnya")
    ax.set_title(nama, color=TINTA)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
batang_warna = fig.colorbar(gambar, ax=axes, fraction=0.025, pad=0.02, label="Proporsi per baris (recall)")
batang_warna.ax.yaxis.set_major_formatter(SUMBU_KOMA)
fig.savefig(FOLDER_GAMBAR / "confusion_matrix.png")
plt.close(fig)

print(json.dumps({n: {k: ringkasan["model"][n][k] for k in ["jumlah_parameter", "akurasi", "f1_makro",
                                                           "waktu_latih_detik"]} for n in MODEL}, indent=1))
print(json.dumps(selisih, indent=1))
