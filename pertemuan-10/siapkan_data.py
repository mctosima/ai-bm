"""Membuat versi ringkas dataset paru (128 × 128) untuk notebook Pertemuan 10.

Sumber: Kaggle "Chest X-ray Dataset for Tuberculosis Segmentation"
(iamtapendu/chest-x-ray-lungs-segmentation, Apache 2.0) — gabungan
Montgomery County & Shenzhen Hospital chest X-ray sets (U.S. National Library
of Medicine; Jaeger dkk., 2014) beserta mask paru.

Dataset aslinya ±3,6 GB (citra beresolusi ~3000 px), terlalu berat diunduh di
kelas. Skrip ini mengecilkan setiap citra & mask menjadi 128 × 128 lalu
menyimpannya sebagai satu file `paru_128.npz` (beberapa MB).

Jalankan sekali:  uv run python pertemuan-10/siapkan_data.py
"""

from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from PIL import Image

UKURAN = 128
FOLDER_INI = Path(__file__).parent

folder = Path(kagglehub.dataset_download("iamtapendu/chest-x-ray-lungs-segmentation"))
meta = pd.read_csv(folder / "MetaData.csv")
folder_citra = folder / "Chest-X-Ray" / "Chest-X-Ray" / "image"
folder_mask = folder / "Chest-X-Ray" / "Chest-X-Ray" / "mask"

citra, mask = [], []
for id_ in meta["id"]:
    c = Image.open(folder_citra / f"{id_}.png").convert("L")
    m = Image.open(folder_mask / f"{id_}.png").convert("L")
    # Citra: diperkecil dengan filter halus; mask: diperkecil lalu dibulatkan ke 0/1
    c = c.resize((UKURAN, UKURAN), Image.Resampling.BOX)
    m = m.resize((UKURAN, UKURAN), Image.Resampling.BOX)
    citra.append(np.asarray(c, dtype=np.uint8))
    mask.append((np.asarray(m) > 127).astype(np.uint8))

np.savez_compressed(
    FOLDER_INI / "paru_128.npz",
    citra=np.stack(citra),
    mask=np.stack(mask),
    id=meta["id"].to_numpy(),
    asal=meta["county"].str.replace("Shenxhen", "Shenzhen").to_numpy().astype(str),
    tb=meta["ptb"].to_numpy(),
)
print("Tersimpan:", FOLDER_INI / "paru_128.npz", "—", len(citra), "pasang citra + mask")
