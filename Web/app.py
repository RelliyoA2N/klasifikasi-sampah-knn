from flask import Flask, render_template, request
from pathlib import Path
import cv2
import pandas as pd
import joblib
from skimage.feature import graycomatrix, graycoprops

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_DIR = PROJECT_DIR / "Model"

model = joblib.load(MODEL_DIR / "knn_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")


def ekstraksi_fitur(image_path):
    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError("Gambar gagal dibaca.")

    # Resize
    img = cv2.resize(img, (224, 224))

    # =========================
    # COLOR HISTOGRAM
    # =========================
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fitur_hist = []

    for channel in range(3):
        hist = cv2.calcHist(
            [img_rgb],
            [channel],
            None,
            [256],
            [0, 256]
        )

        hist = cv2.normalize(hist, hist).flatten()
        fitur_hist.extend(hist)

    # =========================
    # GLCM
    # =========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    fitur_glcm = [
        graycoprops(glcm, "contrast")[0, 0],
        graycoprops(glcm, "correlation")[0, 0],
        graycoprops(glcm, "energy")[0, 0],
        graycoprops(glcm, "homogeneity")[0, 0]
    ]

    # Gabungkan fitur
    fitur = fitur_hist + fitur_glcm

    nama_kolom = (
        [f"Hist_{i}" for i in range(768)]
        + [
            "Contrast",
            "Correlation",
            "Energy",
            "Homogeneity"
        ]
    )

    return pd.DataFrame(
        [fitur],
        columns=nama_kolom
    )


@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None
    error = None
    image_url = None

    if request.method == "POST":

        file = request.files.get("gambar")
        konfirmasi_sampah = request.form.get("konfirmasi_sampah")

        # Validasi konfirmasi
        if not konfirmasi_sampah:
            error = (
                "Konfirmasi bahwa gambar merupakan "
                "citra sampah terlebih dahulu."
            )

        # Validasi file
        elif not file or file.filename == "":
            error = "Silakan pilih gambar terlebih dahulu."

        else:
            try:
                # Folder upload
                upload_dir = BASE_DIR / "static" / "uploads"

                upload_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                # Simpan gambar
                file_path = upload_dir / file.filename
                file.save(file_path)

                # Ekstraksi fitur
                fitur = ekstraksi_fitur(file_path)

                # Standardisasi
                fitur_scaled = scaler.transform(fitur)

                # Prediksi KNN
                hasil = model.predict(
                    fitur_scaled
                )[0]

                # URL gambar
                image_url = (
                    f"/static/uploads/{file.filename}"
                )

            except Exception as e:
                error = str(e)

    return render_template(
        "index.html",
        hasil=hasil,
        error=error,
        image_url=image_url
    )


if __name__ == "__main__":
    app.run(debug=True)