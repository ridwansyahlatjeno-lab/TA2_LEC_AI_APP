# Klasifikasi Gambar CIFAR-10 dengan CNN

Tugas Kelompok 2: Optimasi dan Implementasi dalam Aplikasi Sederhana.

Aplikasi web Streamlit yang menerima input gambar lalu menampilkan prediksi kelas CIFAR-10
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) dari model CNN yang
telah dioptimasi (hyperparameter tuning dan transfer learning).

Aplikasi online: 

Notebook Colab: https://colab.research.google.com/drive/13x0u_lwOCBVfb5aG7Ca71KS5ydcRC08F?usp=sharing
## Anggota Kelompok
- Farras Givari (2902716743)
- Ridwansyah Latjeno (2902699125)
- Kimi Martua Cassanova Simanjuntak (2902733806)
- Hendrikus Avero Widarto (2902734664)
- Muhamad Duta Chandra (2902736442)

## Struktur Repository
| File | Keterangan |
|---|---|
| `app.py` | Aplikasi Streamlit |
| `requirements.txt` | Library yang dibutuhkan |
| `model_tk2.keras` | Model terbaik hasil notebook Colab (input 32×32×3, nilai 0–255) |
| `class_names.json` | Urutan nama kelas |
| `model_info.json` | Informasi model (arsitektur, akurasi, versi TensorFlow) |
| `contoh_uji/` | (opsional) Contoh gambar uji dari CIFAR-10.1 |

## Menjalankan Secara Lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```
