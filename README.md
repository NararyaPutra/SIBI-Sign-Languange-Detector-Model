# Bahasa Isyarat SIBI Pipeline (MediaPipe + LSTM)

Pipeline pengenalan Bahasa Isyarat SIBI (Sistem Isyarat Bahasa Indonesia) berbasis gerakan tangan secara realtime menggunakan **MediaPipe Hands** dan **Bidirectional LSTM** (TensorFlow / Keras).

---

## Alur Utama Pipeline

1. **Split Video**: Video mentah dipisah ke folder `train/valid/test` (rasio `70/20/10`) berdasarkan file video.
2. **Ekstraksi Frame**: Ekstraksi frame dari video asli tanpa crop.
3. **Ekstraksi Landmark & Sequence**: MediaPipe Hands mengekstrak koordinat landmark tangan dari setiap frame dan membentuk dataset sequence `.npz`.
4. **Training Model LSTM**: Melatih arsitektur Bidirectional LSTM untuk klasifikasi kata/isyarat dan mengekspor model ke format Keras (`.keras`) serta TFLite (`.tflite`).
5. **Inferensi Webcam Realtime**: Inferensi langsung dari webcam dengan deteksi landmark MediaPipe + smoothing koordinat (OneEuroFilter) + prediksi model LSTM/TFLite.

---

## File Utama

- [split_sibi_videos.py](file:///d:/Model%20SIBI%20Sign%20Languange/split_sibi_videos.py): Memisahkan dataset video mentah ke split train, valid, dan test.
- [generate_mediapipe_frames.py](file:///d:/Model%20SIBI%20Sign%20Languange/generate_mediapipe_frames.py): Mengekstrak frame gambar dari video yang telah displit.
- [build_mediapipe_lstm_dataset.py](file:///d:/Model%20SIBI%20Sign%20Languange/build_mediapipe_lstm_dataset.py): Mengambil landmark tangan per frame dengan MediaPipe dan menyimpannya ke format sequence `.npz`.
- [new_train_lstm.py](file:///d:/Model%20SIBI%20Sign%20Languange/new_train_lstm.py): Melatih model Bidirectional LSTM dengan augmentasi data dan export ke `.keras` & `.tflite`.
- [webcam_sibi_recognizer.py](file:///d:/Model%20SIBI%20Sign%20Languange/webcam_sibi_recognizer.py): Inferensi realtime melalui webcam (murni deteksi visual tanpa dependensi TTS).
- [sibi_gesture_utils.py](file:///d:/Model%20SIBI%20Sign%20Languange/sibi_gesture_utils.py): Utility fungsi pembantu MediaPipe, normalisasi landmark, dan definisi model.
- [OneEuroFilter.py](file:///d:/Model%20SIBI%20Sign%20Languange/OneEuroFilter.py): Filter untuk meredam jitter/noise pada koordinat landmark tangan realtime.
- [convert_static_images_to_sequences.py](file:///d:/Model%20SIBI%20Sign%20Languange/convert_static_images_to_sequences.py): (Opsional) Mengonversi dataset gambar statis alfabet ke sequence frame video.

---

## Struktur Direktori

```text
d:\Model SIBI Sign Languange
├─ Video/                      # Folder sumber video mentah per kelas kata
├─ Image/                      # (Opsional) Folder dataset gambar statis
├─ video_split/                # Output pemisahan video (train/valid/test)
├─ sequence_frames/            # Output ekstraksi frame video
├─ runs_sibi_lstm/             # Hasil checkpoint model, log, dan visualisasi
│  ├─ exp1/
│  └─ exp19/
├─ split_sibi_videos.py
├─ generate_mediapipe_frames.py
├─ build_mediapipe_lstm_dataset.py
├─ new_train_lstm.py
├─ webcam_sibi_recognizer.py
├─ sibi_gesture_utils.py
├─ OneEuroFilter.py
├─ convert_static_images_to_sequences.py
└─ .gitignore
```

---

## Instalasi & Environment

Pastikan menggunakan Python 3.10 atau 3.11. Pasang dependensi yang diperlukan:

```bash
pip install tensorflow mediapipe opencv-python numpy scikit-learn scipy matplotlib tqdm
```

---

## Format Data Video

Letakkan video mentah di dalam folder `Video/` yang dikelompokkan berdasarkan subfolder nama kata/isyarat:

```text
Video/
├─ Cinta/
│  ├─ Cinta_1.mp4
│  └─ Cinta_2.mp4
├─ Tolong/
│  ├─ Tolong_1.mp4
│  └─ Tolong_2.mp4
└─ ...
```

---

## Langkah Menjalankan Pipeline

### 1. Split Video (70% Train / 20% Valid / 10% Test)

Memisahkan video ke folder `video_split`:

```bash
python split_sibi_videos.py --input Video --output-dir video_split --train-ratio 0.7 --valid-ratio 0.2 --test-ratio 0.1 --clear-output
```

Output:
- Folder `video_split/train`, `video_split/valid`, `video_split/test`
- File daftar kelas: `video_split/sign_classes.txt`

---

### 2. Ekstraksi Frame dari Video

Mengekstrak frame dari video tanpa cropping:

```bash
python generate_mediapipe_frames.py --input-dir video_split --output-dir sequence_frames --sample-every 2
```

> **Tips:** Jika proses berat, naikkan nilai `--sample-every` (misal: `3` atau `4`) atau batasi `--max-frames`.

---

### 3. Bangun Dataset Sequence MediaPipe

Membaca frame gambar, mengekstrak landmark tangan dengan MediaPipe Hands, dan membentuk sequence koordinat:

```bash
python build_mediapipe_lstm_dataset.py --input-dir sequence_frames --output sibi_sequence_dataset.npz --class-file video_split/sign_classes.txt --sequence-length 20 --stride 4 --pad-short-videos
```

Output: File dataset `sibi_sequence_dataset.npz`.

---

### 4. Training Model LSTM

Melatih model Bidirectional LSTM dengan augmentasi (Gaussian noise, random drop frame):

```bash
python new_train_lstm.py --dataset sibi_sequence_dataset.npz --output-dir runs_sibi_lstm/exp1 --epochs 50 --batch-size 32 --hidden-size 128 --num-layers 2 --dropout 0.35 --learning-rate 0.0005 --noise-factor 0.015 --label-smoothing 0.1
```

Output pada `runs_sibi_lstm/exp1/`:
- `best.keras`: Model TensorFlow Keras terbaik.
- `best.tflite`: Model versi TFLite untuk inferensi ringan.
- `model_meta.json`: Metadata kelas dan sequence length.
- Grafik evaluasi: `accuracy.jpg`, `loss.jpg`, `confusion_matrix.jpg`.

---

### 5. Inferensi Realtime Webcam (Murni Deteksi)

Menjalankan inferensi realtime deteksi isyarat langsung dari kamera atau file video. Modul ini murni melakukan deteksi visual tanpa pemanggilan modul suara/TTS.

**Menggunakan webcam laptop (Live):**
```bash
python webcam_sibi_recognizer.py --lstm-model-dir runs_sibi_lstm/exp1 --camera 0
```

**Menggunakan model TFLite (Lebih ringan & cepat):**
```bash
python webcam_sibi_recognizer.py --lstm-model-dir runs_sibi_lstm/exp1 --use-tflite --camera 0
```

**Testing menggunakan file video:**
```bash
python webcam_sibi_recognizer.py --camera test_video.mp4 --lstm-model-dir runs_sibi_lstm/exp1 --no-mirror
```

**Argumen yang tersedia:**
- `--lstm-model-dir`: Folder tempat `best.keras` / `best.tflite` dan `model_meta.json` berada.
- `--camera`: Index kamera (default: `0`) atau path file video.
- `--use-tflite`: Menggunakan model TFLite alih-alih Keras.
- `--prediction-threshold`: Ambang batas confidence minimum prediksi (default: `0.60`).
- `--smoothing`: Jumlah voting buffer prediksi untuk stabilisasi kata (default: `10`).
- `--no-mirror`: Menonaktifkan efek cermin horizontal pada preview webcam.

**Kontrol Tombol:**
- `q`: Keluar dari aplikasi.
- `c`: Menghapus / mereset kalimat yang tersusun di layar.

---

## Tips & Troubleshooting

- **Webcam tidak terbuka**: Coba ubah index kamera dengan `--camera 1`.
- **MediaPipe tidak terdeteksi**: Pastikan terminal berada di environment Python yang telah terinstal `mediapipe` dan `opencv-python`.
- **Performa lambat saat training/ekstraksi**: Kurangi sequence length atau sesuaikan batch size dan sampling rate frame.
