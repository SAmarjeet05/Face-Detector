# 👤 Face Detector

🚀 Live Demo: https://face-detector1.streamlit.app/

An interactive computer vision application that detects human faces in photographs using **OpenCV** and the **Haar Cascade** classifier. It converts image matrices into grayscale contrast channels, runs localized sliding window searches, maps bounding box coordinates, and extracts cropped individual face segments in a dashboard gallery.

---

## 📂 Project Structure

```
Face Detector/
│
├── dataset/
│   ├── haarcascade_frontalface_default.xml  <- Local Haar Cascade classifier XML
│   └── sample_faces_metadata.csv           <- Bounding box coordinates & stats for presets
│
├── Images/
│   ├── sample_single.png                   <- Sample image: Single Face
│   ├── sample_group.png                    <- Sample image: Group Photo
│   └── sample_no_face.png                  <- Sample image: Landscape (No Face)
│
├── notebook/
│   └── face_detection.ipynb                <- Conversational pipeline notebook
│
├── app.py                                  <- Streamlit application code
├── model.py                                <- Face detection OpenCV logic
├── README.md                               <- Project documentation
├── requirements.txt                        <- Project dependencies
└── .gitignore                              <- Git ignore rules
```

---

## 🚀 Getting Started

### 1. Clone the repository and navigate to it:
```bash
git clone <repository-url>
cd "Face Detector"
```

### 2. Install Dependencies
Make sure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Initialize Model and Cache Haar Cascade
Run the helper script directly to pre-download the Haar Cascade classifier XML weights file (~900KB) from the OpenCV repository and save it locally:
```bash
python model.py
```

### 4. Launch the Web Dashboard
Start the Streamlit application:
```bash
streamlit run app.py
```
Open your browser and navigate to the local address provided (typically `http://localhost:8501`).

---

## 🛠️ Interactive Parameters Guide

OpenCV's face detector uses these customizable parameters:
1.  **Scale Factor (`scaleFactor`):** Controls how much the image size is reduced at each scale step. A lower value (e.g. `1.05`) makes the detection grid finer (detecting smaller/more distant faces), but increases processing time.
2.  **Minimum Neighbors (`minNeighbors`):** Controls how many neighboring scanning boxes must agree on a face region to confirm detection. Higher values (e.g. `6` or `8`) prevent false positives (like background noise boxed as a face), while lower values (e.g. `2` or `3`) increase sensitivity.
