import os
import cv2
import numpy as np
import urllib.request
from PIL import Image, ImageDraw

# Configuration paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CASCADE_PATH = os.path.join(DATASET_DIR, "haarcascade_frontalface_default.xml")

def get_cascade_path():
    """Retrieves the Haar Cascade XML path, downloading it if missing."""
    if os.path.exists(CASCADE_PATH):
        return CASCADE_PATH
        
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    os.makedirs(DATASET_DIR, exist_ok=True)
    try:
        print(f"[INFO] Downloading Haar Cascade XML from {url}...")
        urllib.request.urlretrieve(url, CASCADE_PATH)
        print("[SUCCESS] Haar Cascade XML successfully saved!")
        return CASCADE_PATH
    except Exception as e:
        print(f"[WARNING] Failed to download Haar Cascade: {e}")
        raise FileNotFoundError(f"Could not load Haar Cascade xml file: {e}")

def detect_faces(img_pil, scale_factor=1.1, min_neighbors=5):
    """Detects faces in a PIL Image using OpenCV's Haar Cascade.
    
    Args:
        img_pil: Pillow Image object.
        scale_factor: How much the image size is reduced at each image scale.
        min_neighbors: How many neighbors each candidate rectangle should have.
        
    Returns:
        List of coordinates: [(x, y, w, h), ...]
    """
    # 1. Convert PIL image to NumPy array
    img_np = np.array(img_pil)
    
    # Ensure standard RGB channel format
    if img_pil.mode != "RGB":
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
    # 2. Convert RGB to Grayscale (standard preprocessing step for Haar Cascade)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 3. Load pre-trained Haar Cascade classifier
    cascade_file = get_cascade_path()
    face_cascade = cv2.CascadeClassifier(cascade_file)
    
    # 4. Run face detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )
    
    # Return list of bounding coordinates
    return [tuple(map(int, face)) for face in faces]

def draw_faces(img_pil, faces_coords):
    """Draws colored bounding boxes around coordinates on a PIL Image."""
    img_drawn = img_pil.copy()
    draw = ImageDraw.Draw(img_drawn)
    for (x, y, w, h) in faces_coords:
        # Draw a thick green rectangle
        draw.rectangle([x, y, x + w, y + h], outline="#38ef7d", width=3)
    return img_drawn

def crop_faces(img_pil, faces_coords):
    """Crops out individual face regions from a PIL Image."""
    crops = []
    for (x, y, w, h) in faces_coords:
        face_crop = img_pil.crop((x, y, x + w, y + h))
        crops.append(face_crop)
    return crops

if __name__ == "__main__":
    print("[INFO] Initializing Haar Cascade setup...")
    path = get_cascade_path()
    print(f"[SUCCESS] Haar Cascade file verified at: {path}")
