import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model_tk2.keras"
CLASSES_PATH = BASE_DIR / "class_names.json"
INFO_PATH = BASE_DIR / "model_info.json"
EXAMPLES_DIR = BASE_DIR / "contoh_uji"

DEFAULT_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
LABEL_ID = {
    "airplane": "pesawat", "automobile": "mobil", "bird": "burung", "cat": "kucing",
    "deer": "rusa", "dog": "anjing", "frog": "katak", "horse": "kuda",
    "ship": "kapal", "truck": "truk",
}
INPUT_SIZE = (32, 32)

st.set_page_config(page_title="Klasifikasi CIFAR-10", layout="centered")


@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def preprocess(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = ImageOps.fit(img, INPUT_SIZE, method=Image.BICUBIC)
    return np.asarray(img, dtype=np.float32)


def label(name: str) -> str:
    return f"{name} ({LABEL_ID[name]})" if name in LABEL_ID else name


classes = load_json(CLASSES_PATH, DEFAULT_CLASSES)
info = load_json(INFO_PATH, {})

with st.sidebar:
    st.header("Tentang Model")
    if info:
        if "architecture" in info or "arsitektur" in info:
            st.write("**Arsitektur:**", info.get("architecture", info.get("arsitektur")))
        acc = info.get("test_accuracy")
        if isinstance(acc, (int, float)):
            st.write("**Test accuracy:**", f"{acc:.2%}" if acc <= 1 else f"{acc:.2f}%")
        if "tensorflow_version" in info:
            st.write("**TensorFlow:**", info["tensorflow_version"])
    st.write("**Dataset:** CIFAR-10 (10 kelas)")
    st.caption(", ".join(label(c) for c in classes))
    st.divider()
    st.caption("Optimasi dan Implementasi CNN dalam Aplikasi Sederhana")

st.title("Klasifikasi Gambar CIFAR-10")
st.write(
    "Upload sebuah gambar yang berisi **salah satu dari 10 kelas** CIFAR-10. "
    "Model CNN akan memprediksi kelasnya beserta tingkat keyakinannya."
)

if not MODEL_PATH.exists():
    st.error(
        f"File model `{MODEL_PATH.name}` belum ada di repository. "
        "Upload `model_tk2.keras` hasil notebook Colab ke folder yang sama dengan `app.py`."
    )
    st.stop()

model = load_model()

uploaded = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png", "webp", "bmp"])

example_files = sorted(EXAMPLES_DIR.glob("*.png")) + sorted(EXAMPLES_DIR.glob("*.jpg")) \
    if EXAMPLES_DIR.exists() else []
chosen_example = None
if example_files and uploaded is None:
    names = ["—"] + [p.name for p in example_files]
    pick = st.selectbox("…atau coba gambar contoh:", names)
    if pick != "—":
        chosen_example = EXAMPLES_DIR / pick

source = uploaded if uploaded is not None else chosen_example

if source is None:
    st.info("Belum ada gambar. Upload gambar untuk memulai prediksi.")
    st.stop()

try:
    image = Image.open(source)
    x = preprocess(image)
except Exception as e:
    st.error(f"Gambar tidak dapat dibaca: {e}")
    st.stop()

probs = model.predict(x[None, ...], verbose=0)[0]
order = np.argsort(probs)[::-1]
top = order[0]

col1, col2 = st.columns([3, 2])
with col1:
    st.image(ImageOps.exif_transpose(image).convert("RGB"), caption="Gambar asli",
             width="stretch")
with col2:
    st.image(Image.fromarray(x.astype(np.uint8)).resize((160, 160), Image.NEAREST),
             caption="Yang dilihat model (32×32)")

st.subheader("Hasil Prediksi")
conf = float(probs[top])
st.success(f"**{label(classes[top])}** — keyakinan {conf:.1%}")
if conf < 0.5:
    st.warning(
        "Keyakinan model rendah. Bisa jadi objek tidak jelas, latar belakang ramai, "
        "atau objek tidak termasuk 10 kelas CIFAR-10."
    )

st.write("**Top-3 prediksi:**")
for i in order[:3]:
    st.progress(float(probs[i]), text=f"{label(classes[i])}: {probs[i]:.1%}")

with st.expander("Probabilitas semua kelas"):
    st.bar_chart({label(c): float(p) for c, p in zip(classes, probs)}, horizontal=True)
