# Kecerdasan Buatan (BM25-41204) — Hands-on

Materi praktik **Kecerdasan Buatan** Program Studi **Teknik Biomedis**, Institut Teknologi Sumatera (ITERA), Pertemuan 9–10 bersama Martin C.T. Manullang.

Halaman kuliah: https://mctm.web.id/ai-bm

| Pertemuan | Notebook | Buka |
|---|---|---|
| 9 — Deep Learning untuk Sinyal Biomedis | [Klasifikasi detak jantung EKG dengan 1D CNN](pertemuan-09/klasifikasi_ekg_1d_cnn.ipynb) | [![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mctosima/ai-bm/blob/main/pertemuan-09/klasifikasi_ekg_1d_cnn.ipynb) |
| 10 — Segmentasi & Augmentasi Citra Medis | [Segmentasi paru pada X-ray dengan U-Net + augmentasi](pertemuan-10/segmentasi_paru_unet.ipynb) | [![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mctosima/ai-bm/blob/main/pertemuan-10/segmentasi_paru_unet.ipynb) |

## Tugas P10 — Segmentasi Polip (Kvasir-SEG)

Starter tugas kelompok ada di folder [`tugas-p10/`](tugas-p10/): notebook `segmentasi_polip.ipynb`, template laporan `README.md`, `pyproject.toml`, dan `.gitignore`. Petunjuk lengkap, tenggat, dan form pengumpulan: **https://mctm.web.id/ai-bm/tugas**

## Cara menjalankan

**Paling mudah — Google Colab (tanpa instalasi):** klik tombol *Buka di Colab* di atas, lalu pilih *Runtime → Change runtime type → T4 GPU*.

**Di laptop sendiri** memakai [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/mctosima/ai-bm.git
cd ai-bm
uv sync
uv run jupyter lab
```

Notebook otomatis memakai GPU NVIDIA (CUDA), GPU Apple Silicon (MPS), atau CPU — mana yang tersedia. Dataset diunduh otomatis saat notebook dijalankan.

## Dataset

- Pertemuan 9: [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) (Kachuee dkk., 2018), turunan MIT-BIH Arrhythmia Database.
- Pertemuan 10: [Chest X-ray Dataset for Tuberculosis Segmentation](https://www.kaggle.com/datasets/iamtapendu/chest-x-ray-lungs-segmentation) (Apache 2.0) — gabungan Montgomery County & Shenzhen Hospital chest X-ray sets (Jaeger dkk., 2014) beserta mask paru. Repo ini menyimpan versi 128 × 128 (`pertemuan-10/paru_128.npz`, dibuat dengan `pertemuan-10/siapkan_data.py`).
