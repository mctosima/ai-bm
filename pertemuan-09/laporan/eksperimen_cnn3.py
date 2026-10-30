"""Eksperimen: 1D CNN 2 blok konvolusi (model notebook) vs 3 blok konvolusi.

Kedua model dilatih dengan resep yang sama seperti di notebook
(Adam lr=0.001, batch 128, 10 epoch, CrossEntropyLoss) pada 5 seed berbeda.
Hasil disimpan ke hasil/hasil.json dan gambar ke gambar/.

Jalankan dari root repo:  uv run python pertemuan-09/laporan/eksperimen_cnn3.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

FOLDER = Path(__file__).resolve().parent
FOLDER_DATA = FOLDER.parent / "data"
FOLDER_HASIL = FOLDER / "hasil"
FOLDER_HASIL.mkdir(exist_ok=True)

SEEDS = [42, 0, 1, 2, 3]
JUMLAH_EPOCH = 10
UKURAN_BATCH = 128
LR = 0.001

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def buat_cnn2():
    """Model persis seperti di notebook Pertemuan 9."""
    return nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),   # 187 → 93
        nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),  # 93 → 46
        nn.Flatten(),
        nn.Linear(32 * 46, 64), nn.ReLU(),
        nn.Linear(64, 5),
    )


def buat_cnn3():
    """Model notebook + satu blok konvolusi ketiga (64 filter)."""
    return nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),   # 187 → 93
        nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),  # 93 → 46
        nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),  # 46 → 23
        nn.Flatten(),
        nn.Linear(64 * 23, 64), nn.ReLU(),
        nn.Linear(64, 5),
    )


MODEL = {"CNN-2": buat_cnn2, "CNN-3": buat_cnn3}


def muat_data():
    latih = pd.read_csv(FOLDER_DATA / "mitbih_train.csv", header=None).values
    uji = pd.read_csv(FOLDER_DATA / "mitbih_test.csv", header=None).values
    X_latih = torch.tensor(latih[:, :187], dtype=torch.float32).unsqueeze(1)
    y_latih = torch.tensor(latih[:, 187].astype(int), dtype=torch.long)
    X_uji = torch.tensor(uji[:, :187], dtype=torch.float32).unsqueeze(1)
    y_uji = torch.tensor(uji[:, 187].astype(int), dtype=torch.long)
    return X_latih, y_latih, X_uji, y_uji


def sinkron():
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def evaluasi(model, loader, fungsi_loss):
    model.eval()
    total_loss, tebakan = 0.0, []
    with torch.no_grad():
        for sinyal, label in loader:
            keluaran = model(sinyal.to(device))
            total_loss += fungsi_loss(keluaran, label.to(device)).item() * len(label)
            tebakan.append(keluaran.argmax(dim=1).cpu())
    return total_loss / len(loader.dataset), torch.cat(tebakan).numpy()


def jalankan(nama, seed, X_latih, y_latih, X_uji, y_uji):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MODEL[nama]().to(device)
    jumlah_parameter = sum(p.numel() for p in model.parameters())

    loader_latih = DataLoader(TensorDataset(X_latih, y_latih), batch_size=UKURAN_BATCH, shuffle=True)
    loader_uji = DataLoader(TensorDataset(X_uji, y_uji), batch_size=UKURAN_BATCH, shuffle=False)
    fungsi_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    y_benar = y_uji.numpy()

    riwayat = {"loss_latih": [], "akurasi_latih": [], "loss_uji": [], "f1_makro_uji": []}
    waktu_latih = 0.0
    for epoch in range(JUMLAH_EPOCH):
        model.train()
        sinkron()
        mulai = time.perf_counter()
        total_loss, jumlah_benar = 0.0, 0
        for sinyal, label in loader_latih:
            sinyal, label = sinyal.to(device), label.to(device)
            keluaran = model(sinyal)
            loss = fungsi_loss(keluaran, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(label)
            jumlah_benar += (keluaran.argmax(dim=1) == label).sum().item()
        sinkron()
        waktu_latih += time.perf_counter() - mulai

        # Data uji hanya dipantau untuk kurva belajar; tidak dipakai memilih epoch/model.
        loss_uji, y_tebak = evaluasi(model, loader_uji, fungsi_loss)
        riwayat["loss_latih"].append(total_loss / len(y_latih))
        riwayat["akurasi_latih"].append(jumlah_benar / len(y_latih))
        riwayat["loss_uji"].append(loss_uji)
        riwayat["f1_makro_uji"].append(f1_score(y_benar, y_tebak, average="macro"))

    presisi, recall, f1, _ = precision_recall_fscore_support(y_benar, y_tebak, labels=range(5), zero_division=0)
    hasil = {
        "model": nama,
        "seed": seed,
        "jumlah_parameter": jumlah_parameter,
        "waktu_latih_detik": waktu_latih,
        "akurasi": float((y_tebak == y_benar).mean()),
        "f1_makro": float(f1_score(y_benar, y_tebak, average="macro")),
        "presisi": presisi.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "confusion_matrix": confusion_matrix(y_benar, y_tebak, labels=range(5)).tolist(),
        "riwayat": riwayat,
    }
    print(f"{nama} seed={seed:2d} | akurasi={hasil['akurasi']:.4f} | F1 makro={hasil['f1_makro']:.4f} "
          f"| waktu latih={waktu_latih:.1f} s", flush=True)
    return hasil


def main():
    print("Device:", device, "| PyTorch", torch.__version__, flush=True)
    data = muat_data()
    semua = [jalankan(nama, seed, *data) for seed in SEEDS for nama in MODEL]
    keluaran = {
        "pengaturan": {
            "seeds": SEEDS, "epoch": JUMLAH_EPOCH, "batch": UKURAN_BATCH, "lr": LR,
            "device": str(device), "torch": torch.__version__,
        },
        "hasil": semua,
    }
    (FOLDER_HASIL / "hasil.json").write_text(json.dumps(keluaran, indent=1))
    print("Tersimpan:", FOLDER_HASIL / "hasil.json")


if __name__ == "__main__":
    main()
