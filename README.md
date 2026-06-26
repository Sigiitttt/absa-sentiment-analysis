# 🌟 Aspect-Based Sentiment Analysis (ABSA) pada Ulasan Pariwisata

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/ariiii4545452/absa-proyek-akhir-nlp)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Framework](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

Proyek *End-to-End* Natural Language Processing (NLP) untuk mengekstrak aspek dan polaritas sentimen pada ulasan destinasi pariwisata alam berbahasa Indonesia secara bersamaan (*joint-task*). 

Sistem ini dikembangkan menggunakan paradigma **Instruction Learning** (berdasarkan paper **InstructABSA**) dengan melakukan *fine-tuning* pada arsitektur model bahasa besar **T5 (Tk-Instruct-Base)**.

## 🚀 Fitur Utama
- **Generative ABSA:** Tidak menggunakan klasifikasi konvensional, melainkan menyuapkan instruksi kaku (*prompt wrapping*) agar model generatif menghasilkan jawaban berformat `aspek:sentimen`.
- **Ekstraksi Multi-Aspek:** Mampu mendeteksi 3 entitas pariwisata sekaligus dalam satu kalimat: **Akses Jalan, Fasilitas, dan Aktivitas Wisata**.
- **Performa Tinggi:** Terlatih secara *supervised* dan mencapai **F1-Score 80.79%** pada data uji yang terisolasi.
- **UI Web Modern:** Dilengkapi dengan antarmuka berbasis Streamlit bergaya *Cinematic Dark Theme* dan *Editorial Layout*.

## 📊 Dataset & Batasan Masalah
Dataset bersumber dari [Natural Tourist Attractions Review Dataset](https://www.kaggle.com/datasets/dzzlr07/absa-natural-tourist-attractions-review) di Kaggle (ulasan Google Maps Kabupaten Bandung Barat).
* **Prapemrosesan:** Data difilter dan diterjemahkan secara spesifik pada 3 kolom aspek sasaran (Akses Jalan, Fasilitas, Aktivitas Wisata).
* **Pembagian Data:** 80% Data Latih (9.296 baris) dan 20% Data Uji (2.324 baris).

## ⚙️ Arsitektur Sistem & Spesifikasi
* **Base Model:** `allenai/tk-instruct-base-def-pos` (~200M Parameters / Ukuran Biner ~495 MB)
* **Hardware Pelatihan:** GPU CUDA (via Google Colab)
* **Metrik Pelatihan:** 4 Epochs, Optimizer AdamW, Final Training Loss: **0.0456**
* **Library NLP Utama:** `transformers`, `torch`, `sentencepiece`
* **Deployment:** Hugging Face Spaces

## 💻 Instalasi & Menjalankan Aplikasi Lokal

1. **Kloning Repositori**
   ```bash
   git clone [https://github.com/username-kamu/nama-repo-kamu.git](https://github.com/username-kamu/nama-repo-kamu.git)
   cd nama-repo-kamu

```

2. **Instalasi Dependensi**
Sangat disarankan menggunakan virtual environment (Python 3.10).
```bash
pip install -r requirements.txt

```


*(Pastikan `requirements.txt` berisi: `streamlit`, `torch`, `transformers`, `sentencepiece`, `pandas`)*
3. **Jalankan Antarmuka Web (Streamlit)**
```bash
streamlit run app.py

```



## 🧠 Contoh Penggunaan (Inferensi Murni via Python)

Jika ingin memanggil model secara langsung tanpa UI Streamlit, jalankan skrip berikut:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./Models/checkpoint-4648" # Ubah ke path model lokalmu

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

teks_ulasan = "jalannya buruk susah dilewati namun pemandangannya bagus dan fasilitas nya kotor"

# Format Prompt Sesuai InstructABSA
prompt = (
    "Definition: The output will be the aspects (both implicit "
    "and explicit) and the aspects sentiment polarity. In cases "
    "where there are no aspects the output should be "
    "noaspectterm:none.\n"
    "    Positive example 1-\n"
    "    input: Fasilitas di pantai ini sangat lengkap dan bersih.\n"
    "    output: fasilitas:positive\n"
    "    Negative example 1-\n"
    "    input: Akses jalan menuju air terjun penuh lubang.\n"
    "    output: akses jalan:negative\n"
    "    Neutral example 1-\n"
    "    input: Ada banyak aktivitas wisata yang ditawarkan.\n"
    "    output: aktivitas wisata:neutral\n"
    "    Now complete the following example-\n"
    f"    input: {teks_ulasan}\n"
    "    output: "
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(inputs.input_ids, max_length=100)
hasil = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Hasil Ekstraksi: {hasil}")
# Ekspektasi Output: akses jalan:negative, fasilitas:negative, aktivitas wisata:positive

```

## 👨‍💻 Penulis

* **Moch Sigit Aringga** (230411100104)
* Program Studi Teknik Informatika - Universitas Trunojoyo Madura (2026)
* Diajukan sebagai Proyek Akhir Mata Kuliah Pemrosesan Bahasa Alami (NLP).

## 📄 Lisensi

[MIT License](https://www.google.com/search?q=LICENSE)

```

***

**Tips Penyesuaian:**
1. Pada bagian "Kloning Repositori", ganti `https://github.com/username-kamu/nama-repo-kamu.git` dengan *link* repositori GitHub/Hugging Face aslimu jika kamu mengunggahnya.
2. Panjang baris kodenya (khususnya di area `prompt`) sudah saya potong-potong ke bawah menggunakan tanda kutip bertingkat agar rapi ke bawah dan tidak kepotong ke samping saat dilihat di mode teks biasa.

```
