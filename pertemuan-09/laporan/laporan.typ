#import "ieee.typ": ieee

#show: ieee.with(
  title: [Pengaruh Penambahan Blok Konvolusi Ketiga pada 1D CNN untuk Klasifikasi Detak Jantung EKG MIT-BIH],
  abstract: [
    Laporan ini membandingkan model 1D _convolutional neural network_ (CNN) dua blok konvolusi (CNN-2) dengan varian tiga blok konvolusi (CNN-3) untuk klasifikasi lima kelas detak jantung (N, S, V, F, Q) pada _ECG Heartbeat Categorization Dataset_ turunan MIT-BIH. Kedua model dilatih dengan resep identik (Adam, laju belajar 0,001, _batch_ 128, 10 _epoch_) dan diulang pada lima _seed_. CNN-3 menambah 10,6% parameter dan 19% waktu latih. CNN-3 belajar lebih cepat dan mencapai _loss_ latih lebih rendah. Median F1 makro CNN-3 lebih tinggi (0,881 vs 0,869), terutama karena _recall_ kelas minoritas S dan F naik. Namun, satu dari lima _run_ CNN-3 mengalami penurunan tajam pada _epoch_ terakhir sehingga rata-rata F1 makro kedua model praktis sama (0,871 ± 0,029 vs 0,870 ± 0,006; uji-_t_ berpasangan _p_ = 0,91). Dengan protokol pelatihan tanpa validasi dan _early stopping_, penambahan blok ketiga belum memberikan peningkatan yang konsisten.
  ],
  authors: (
    (
      name: "Martin Manullang",
      department: [Program Studi Teknik Biomedis],
      organization: [Institut Teknologi Sumatera (ITERA)],
      location: [Lampung Selatan, Indonesia],
      email: [NIM 123123123],
    ),
  ),
  index-terms: ("Elektrokardiogram", "klasifikasi aritmia", "1D CNN", "MIT-BIH", "deep learning"),
  paper-size: "a4",
  bibliography: bibliography("refs.bib", style: "ieee", title: [Daftar Pustaka]),
)

#set text(lang: "id")

// ------------------------------------------------------------ data hasil eksperimen
#let R = json("hasil/ringkasan.json")
#let M2 = R.model.at("CNN-2")
#let M3 = R.model.at("CNN-3")
#let KELAS = ("N", "S", "V", "F", "Q")
#let N-UJI = (18118, 556, 1448, 162, 1608)

// Format angka desimal dengan koma dan jumlah digit tetap
#let fmt(x, d) = {
  let s = str(calc.round(x, digits: d))
  if not s.contains(".") { s += "." }
  let desimal = s.split(".").at(1).len()
  (s + "0" * (d - desimal)).replace(".", ",")
}
#let ms(v, d, skala: 1) = [#fmt(v.mean * skala, d) ± #fmt(v.std * skala, d)]
#let ribuan(n) = {
  let s = str(n)
  let hasil = ""
  for (i, c) in s.clusters().rev().enumerate() {
    if i > 0 and calc.rem(i, 3) == 0 { hasil = "." + hasil }
    hasil = c + hasil
  }
  hasil
}

= Pendahuluan

Elektrokardiogram (EKG) adalah pemeriksaan non-invasif utama untuk mendeteksi aritmia. Analisis manual rekaman panjang memakan waktu, sehingga klasifikasi detak otomatis banyak diteliti. CNN satu dimensi (1D CNN) cocok untuk tugas ini karena dapat mempelajari ciri morfologi detak langsung dari sinyal mentah tanpa ekstraksi fitur manual @kiranyaz2016 @hannun2019.

Pada praktikum Pertemuan 9 telah dibangun 1D CNN sederhana dengan dua blok konvolusi (CNN-2) pada dataset Kachuee dkk. @kachuee2018. Latihan di akhir notebook menyarankan penambahan blok konvolusi ketiga. Pertanyaan yang ingin dijawab laporan ini: _apakah blok ketiga meningkatkan kinerja klasifikasi, khususnya pada kelas minoritas, dan berapa biayanya?_ Karena hasil satu kali pelatihan dapat berbeda hanya karena inisialisasi acak, setiap model dilatih pada lima _seed_ dan dibandingkan secara berpasangan.

= Dataset dan Metode

== Dataset
Dataset yang dipakai adalah _ECG Heartbeat Categorization Dataset_ @kachuee2018, yang diturunkan dari MIT-BIH Arrhythmia Database @moody2001 @goldberger2000. Setiap sampel adalah satu detak yang disejajarkan pada puncak R, dicuplik ulang ke 125 Hz, dinormalisasi ke [0, 1], dan dipotong/diisi nol hingga 187 titik (≈1,5 s). Label mengikuti lima kelas AAMI EC57 @aami2012: N (normal), S (ektopik supraventrikular), V (ektopik ventrikular), F (_fusion_), dan Q (tak terklasifikasi, termasuk detak _paced_). Pembagian latih/uji mengikuti file asli: 87.554 detak latih dan 21.892 detak uji. Distribusinya sangat tidak seimbang: kelas N mencakup 82,8% data uji, sedangkan kelas F hanya 162 detak (0,7%).

== Arsitektur Model
@tab:arsitektur merangkum kedua arsitektur. CNN-2 identik dengan model pada notebook: dua blok `Conv1d` (kernel 5, _padding_ 2) + ReLU + `MaxPool1d(2)` dengan 16 dan 32 filter, diikuti lapisan _fully connected_ 64 unit dan keluaran 5 kelas. CNN-3 menambahkan blok ketiga dengan 64 filter. Karena blok ketiga menggandakan jumlah kanal sekaligus membagi dua panjang sinyal, ukuran vektor hasil _flatten_ tetap 1.472 (32 × 46 = 64 × 23), sehingga tambahan parameter hanya berasal dari konvolusi ketiga (10.304 parameter, +10,6%). Perbedaan utamanya adalah medan reseptif: satu fitur pada blok konvolusi terakhir CNN-3 melihat 36 sampel (0,29 s), sedangkan CNN-2 hanya 16 sampel (0,13 s), sedikit di atas lebar kompleks QRS normal.

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    align: (left, center, center),
    stroke: none,
    table.hline(),
    table.header([*Lapisan*], [*CNN-2*], [*CNN-3*]),
    table.hline(stroke: 0.5pt),
    [Masukan], [1 × 187], [1 × 187],
    [Blok 1: Conv(16) + Pool], [16 × 93], [16 × 93],
    [Blok 2: Conv(32) + Pool], [32 × 46], [32 × 46],
    [Blok 3: Conv(64) + Pool], [—], [64 × 23],
    [Flatten], [1.472], [1.472],
    [Linear + ReLU], [64], [64],
    [Linear (keluaran)], [5], [5],
    table.hline(stroke: 0.5pt),
    [Jumlah parameter], [#ribuan(M2.jumlah_parameter)], [#ribuan(M3.jumlah_parameter)],
    [Medan reseptif blok akhir], [16 sampel (0,13 s)], [36 sampel (0,29 s)],
    table.hline(),
  ),
  caption: [Arsitektur dan ukuran keluaran tiap lapisan (kanal × panjang)],
  placement: top,
) <tab:arsitektur>

== Protokol Pelatihan dan Evaluasi
Kedua model dilatih dengan resep notebook tanpa modifikasi: _cross-entropy loss_ tanpa pembobotan kelas, optimizer Adam @kingma2015 dengan laju belajar 0,001, ukuran _batch_ 128 dengan pengacakan, selama 10 _epoch_. Implementasi memakai PyTorch #R.pengaturan.torch @paszke2019 pada GPU Apple Silicon (MPS). Setiap model dilatih dengan _seed_ {42, 0, 1, 2, 3}. _Seed_ yang sama menghasilkan urutan _batch_ yang sama untuk kedua model, sehingga selisih kinerja dapat dibandingkan berpasangan per _seed_.

Model hasil _epoch_ ke-10 dievaluasi pada data uji. Karena data tidak seimbang, metrik utama adalah F1 makro, yaitu rata-rata F1 kelima kelas dengan bobot sama:
$ "F1"_"makro" = 1/K sum_(k=1)^K (2 P_k R_k)/(P_k + R_k), quad K = 5, $
dengan $P_k$ dan $R_k$ presisi dan _recall_ kelas $k$. Akurasi tetap dilaporkan sebagai pembanding. Kinerja pada data uji juga dicatat setiap _epoch_ untuk kurva belajar, tetapi tidak dipakai memilih _epoch_ atau model. Hasil dilaporkan sebagai rerata ± simpangan baku lima _seed_. Signifikansi selisih diuji dengan uji-_t_ berpasangan.

= Hasil

== Kinerja Keseluruhan
@tab:keseluruhan menunjukkan bahwa rerata akurasi dan F1 makro kedua model hampir sama, tetapi simpangan baku CNN-3 jauh lebih besar. Selisih berpasangan F1 makro (CNN-3 − CNN-2) adalah +0,001 ± 0,024 (_t_(4) = 0,12; _p_ = 0,91), dan selisih akurasi −0,22 ± 1,05 poin persen (_t_(4) = −0,46; _p_ = 0,67). CNN-3 unggul F1 makro pada 4 dari 5 _seed_, dan median F1 makronya lebih tinggi (0,881 vs 0,869). Namun, pada _seed_ 0 CNN-3 turun tajam (F1 makro 0,821; akurasi 95,44%), dan satu _run_ ini cukup untuk menghapus keunggulan rata-ratanya. Tanpa _seed_ 0, rerata F1 makro menjadi 0,884 (CNN-3) vs 0,872 (CNN-2). Biaya CNN-3 adalah waktu latih 19% lebih lama.

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    align: (left, center, center),
    stroke: none,
    table.hline(),
    table.header([*Metrik*], [*CNN-2*], [*CNN-3*]),
    table.hline(stroke: 0.5pt),
    [Akurasi (%)], ms(M2.akurasi, 2, skala: 100), ms(M3.akurasi, 2, skala: 100),
    [F1 makro], ms(M2.f1_makro, 3), ms(M3.f1_makro, 3),
    [Waktu latih (s)], ms(M2.waktu_latih_detik, 1), ms(M3.waktu_latih_detik, 1),
    table.hline(),
  ),
  caption: [Kinerja pada data uji setelah 10 _epoch_ (rerata ± simpangan baku, 5 _seed_)],
  placement: top,
) <tab:keseluruhan>

== Kinerja per Kelas
@tab:perkelas dan @fig:recall memperlihatkan bahwa perbedaan kedua model terletak pada kelas minoritas. CNN-3 menaikkan _recall_ kelas S dari #fmt(M2.recall.at(1).mean, 3) menjadi #fmt(M3.recall.at(1).mean, 3) dan kelas F dari #fmt(M2.recall.at(3).mean, 3) menjadi #fmt(M3.recall.at(3).mean, 3). Akan tetapi, presisi kelas S turun dari #fmt(M2.presisi.at(1).mean, 3) menjadi #fmt(M3.presisi.at(1).mean, 3) dengan simpangan baku #fmt(M3.presisi.at(1).std, 3), sehingga F1 kelas S tidak berubah. Kelas N, V, dan Q sudah di atas 0,93 untuk kedua model dan praktis tidak berubah.

#figure(
  {
    let baris = ()
    for (i, k) in KELAS.enumerate() {
      for (nama, m) in (("CNN-2", M2), ("CNN-3", M3)) {
        baris += (
          if nama == "CNN-2" { table.cell(rowspan: 2, align: horizon)[#k (#ribuan(N-UJI.at(i)))] },
          nama,
          ms(m.presisi.at(i), 3), ms(m.recall.at(i), 3), ms(m.f1.at(i), 3),
        ).filter(x => x != none)
      }
    }
    table(
      columns: (auto, auto, 1fr, 1fr, 1fr),
      align: (left, left, center, center, center),
      stroke: none,
      inset: (x: 3pt, y: 3pt),
      table.hline(),
      table.header([*Kelas (_n_)*], [*Model*], [*Presisi*], [*Recall*], [*F1*]),
      table.hline(stroke: 0.5pt),
      ..baris,
      table.hline(),
    )
  },
  caption: [Presisi, _recall_, dan F1 per kelas pada data uji (rerata ± simpangan baku, 5 _seed_)],
  placement: top,
) <tab:perkelas>

#figure(
  image("gambar/recall_per_kelas.png", width: 100%),
  caption: [Rerata _recall_ per kelas pada data uji. Garis galat menunjukkan simpangan baku lima _seed_.],
  placement: bottom,
) <fig:recall>

Pola ini juga tampak pada _confusion matrix_ _seed_ 42 (@fig:cm). Kesalahan terbesar kedua model adalah detak S dan F yang ditebak sebagai N. CNN-3 mengurangi kesalahan S → N (169 → 152) dan F → N/V, tetapi menambah kesalahan V → N (66 → 98) dan Q → N (19 → 71). Artinya, sebagian peningkatan pada kelas minoritas dibayar dengan kesalahan baru di kelas lain.

#figure(
  image("gambar/confusion_matrix.png", width: 100%),
  caption: [_Confusion matrix_ data uji untuk _seed_ 42. Angka atas adalah jumlah detak; angka dalam kurung adalah proporsi terhadap jumlah kelas sebenarnya (_recall_ pada diagonal).],
  placement: top,
  scope: "parent",
) <fig:cm>

== Dinamika Pelatihan
@fig:kurva menunjukkan bahwa CNN-3 belajar lebih cepat. _Loss_ latihnya lebih rendah di setiap _epoch_ (#fmt(0.0532, 3) vs #fmt(0.0718, 3) pada _epoch_ 10), dan F1 makro ujinya sudah 0,718 setelah _epoch_ pertama (CNN-2: 0,683). Pada _epoch_ 6 F1 makro CNN-3 mencapai 0,874, yaitu nilai yang tidak pernah dicapai rerata CNN-2 selama 10 _epoch_. Akan tetapi, kurva F1 makro CNN-3 mendatar dan sedikit turun setelah _epoch_ 8 (0,880 → 0,871) dan pitanya melebar, sedangkan CNN-2 masih naik hingga akhir.

#figure(
  image("gambar/kurva_belajar.png", width: 100%),
  caption: [Kurva belajar: (a) _loss_ latih dan (b) F1 makro data uji per _epoch_. Garis adalah rerata dan pita adalah ± simpangan baku lima _seed_.],
  placement: top,
) <fig:kurva>

= Pembahasan

Hasil ini menunjukkan bahwa kapasitas tambahan CNN-3 _berguna_: model lebih cepat menemukan pola dan lebih sering mengenali detak S dan F. Salah satu penjelasan yang mungkin adalah medan reseptif yang lebih lebar (0,29 s). Detak S sering hanya berbeda dari N pada gelombang P dan jarak R-R, ciri yang berada di luar kompleks QRS, sehingga fitur dengan konteks lebih panjang dapat membantu. Dugaan ini belum diuji secara langsung dalam laporan ini.

Masalahnya ada pada _kestabilan_. Pada _seed_ 0, F1 makro CNN-3 adalah 0,865 pada _epoch_ 9 lalu turun ke 0,821 pada _epoch_ 10. Pada _epoch_ 10, model ini menebak 448 detak N sebagai S, sehingga presisi kelas S hanya 0,50 walaupun _recall_-nya 0,82. Artinya, batas keputusan bergeser ke arah kelas S dalam satu _epoch_. Kemungkinan penyebabnya adalah ketidakseimbangan data: kelas S dan F hanya menyumbang sedikit _gradien_, sehingga posisi batas keputusannya sensitif terhadap beberapa langkah optimisasi terakhir. Karena protokol notebook selalu mengambil model _epoch_ terakhir, ketidakstabilan ini langsung terlihat sebagai hasil akhir.

Laporan ini memiliki beberapa keterbatasan. Pertama, lima _seed_ hanya cukup untuk menunjukkan arah, bukan menyimpulkan signifikansi. Kedua, tidak ada data validasi, sehingga tidak dapat dilakukan _early stopping_ atau pemilihan _epoch_ tanpa mengintip data uji. Ketiga, pembagian latih/uji dataset ini dilakukan per detak, bukan per pasien, sehingga detak dari pasien yang sama dapat muncul di kedua bagian. Kinerja pada pasien baru (_inter-patient_) biasanya jauh lebih rendah @dechazal2004. Keempat, operasi pada GPU tidak sepenuhnya deterministik, sehingga pengulangan dengan _seed_ yang sama dapat sedikit berbeda.

= Kesimpulan

Penambahan blok konvolusi ketiga (64 filter) pada 1D CNN menambah 10,6% parameter dan 19% waktu latih. CNN-3 belajar lebih cepat, mencapai _loss_ latih lebih rendah, dan cenderung menaikkan _recall_ kelas minoritas S dan F, sehingga median F1 makronya lebih tinggi (0,881 vs 0,869). Namun, dengan resep pelatihan yang sama (10 _epoch_ tetap, tanpa validasi), hasil CNN-3 kurang stabil, dan secara rata-rata keunggulannya tidak signifikan (_p_ = 0,91). Jadi, model CNN-2 dari notebook sudah merupakan titik awal yang kuat, dan blok ketiga baru layak dipakai jika disertai pemilihan model berbasis data validasi (_early stopping_). Langkah lanjut yang disarankan adalah: (1) memisahkan data validasi dari data latih dan menerapkan _early stopping_, (2) memakai _class-weighted loss_ untuk kelas S dan F, dan (3) mengevaluasi ulang dengan pembagian _inter-patient_.

Kode eksperimen (`eksperimen_cnn3.py`), pembuat gambar (`buat_gambar.py`), dan hasil mentah (`hasil/hasil.json`) tersedia di folder `pertemuan-09/laporan/` repositori praktikum.
