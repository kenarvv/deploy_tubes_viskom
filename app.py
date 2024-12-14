import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load model YOLOv5
model_path = "best_trash_classifier.pt"  # Path model yang sudah dilatih
model = YOLO(model_path)

# Judul aplikasi
st.title("Deteksi Sampah dengan YOLOv5")
st.write("Unggah gambar untuk mendeteksi jenis sampah.")

# Unggah gambar
uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Buka gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Konversi ke format yang kompatibel dengan YOLO
    image_array = np.array(image)

    # Deteksi objek
    st.write("Mendeteksi...")
    results = model.predict(source=image_array, device='cpu', save=False, save_txt=False)

    # Visualisasi hasil
    result_image = results[0].plot()  # Gambar hasil deteksi dengan bounding box
    st.image(result_image, caption="Hasil Deteksi", use_column_width=True)

    # Tampilkan informasi deteksi
    st.write("Hasil Deteksi:")
    for detection in results[0].boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = detection
        class_name = results[0].names[int(class_id)]
        st.write(f"- **{class_name}**: {confidence:.2f}")
