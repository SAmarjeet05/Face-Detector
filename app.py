import streamlit as st
import os
import pandas as pd
from PIL import Image

# Import helper functions from model.py
from model import (
    detect_faces,
    draw_faces,
    crop_faces
)

# Page configuration
st.set_page_config(
    page_title="Face Detector Dashboard",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS
st.markdown("""
<style>
    /* Gradient Background for header */
    .header-container {
        background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    .header-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 300;
    }
    /* Card design */
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.06);
    }
    /* Count callout */
    .count-output {
        background-color: #e2f0d9;
        border-left: 6px solid #70ad47;
        color: #385723;
        padding: 1.5rem;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Absolute paths relative to app.py with case-robustness
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(folder_name, file_name=None):
    folder_low = os.path.join(BASE_DIR, folder_name.lower())
    folder_cap = os.path.join(BASE_DIR, folder_name.capitalize())
    selected_folder = folder_low if os.path.exists(folder_low) else folder_cap
    if file_name:
        return os.path.join(selected_folder, file_name)
    return selected_folder

IMAGES_DIR = resolve_path("Images")
CLEANED_DATASET_PATH = resolve_path("dataset", "sample_faces_metadata.csv")

# Application Header
st.markdown("""
<div class="header-container">
    <div class="header-title">👤 Face Detector Dashboard</div>
    <div class="header-subtitle">Detect human faces in photographs using OpenCV & Haar Cascade Classifiers</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Parameter Controls
st.sidebar.title("⚙️ Detection Parameters")
st.sidebar.markdown("""
Adjust the parameters below to fine-tune OpenCV's face detection sensitivity.
""")

scale_factor = st.sidebar.slider(
    "📏 Scale Factor",
    min_value=1.05,
    max_value=1.50,
    value=1.10,
    step=0.05,
    help="How much the image size is reduced at each image scale. Lower values mean finer search grids (more sensitive, but slower)."
)

min_neighbors = st.sidebar.slider(
    "👥 Minimum Neighbors",
    min_value=1,
    max_value=15,
    value=5,
    step=1,
    help="How many neighbor bounding blocks must group together to confirm a face. Higher values reduce false positives."
)

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Parameter Tuning Tips")
st.sidebar.markdown("""
*   **Too many false boxes?** Increase *Minimum Neighbors* or *Scale Factor*.
*   **Missing some faces?** Decrease *Minimum Neighbors* or *Scale Factor* (e.g. scale factor of `1.05` scans with high resolution).
""")

# Main Content Tabs
tab_detect, tab_explorer, tab_works = st.tabs([
    "🏠 Detect Faces", 
    "📊 Dataset Explorer", 
    "🧠 How it Works"
])

# ----------------- TAB 1: DETECT FACES -----------------
with tab_detect:
    st.subheader("💡 Face Detection Dashboard")
    st.write("Upload a custom photo or choose a preset test image to scan for human faces.")
    
    # Preset Gallery
    st.markdown("### 📷 Select a Sample Preset")
    col_g1, col_g2, col_g3 = st.columns(3)
    
    selected_preset_path = None
    
    with col_g1:
        single_path = os.path.join(IMAGES_DIR, "sample_single.png")
        if os.path.exists(single_path):
            st.image(single_path, caption="Preset: Single Face", use_container_width=True)
            if st.button("👤 Test Single Face", use_container_width=True):
                selected_preset_path = single_path
                
    with col_g2:
        group_path = os.path.join(IMAGES_DIR, "sample_group.png")
        if os.path.exists(group_path):
            st.image(group_path, caption="Preset: Group Photo", use_container_width=True)
            if st.button("👥 Test Group Photo", use_container_width=True):
                selected_preset_path = group_path
                
    with col_g3:
        noface_path = os.path.join(IMAGES_DIR, "sample_no_face.png")
        if os.path.exists(noface_path):
            st.image(noface_path, caption="Preset: Landscape (No Face)", use_container_width=True)
            if st.button("⛰️ Test Landscape", use_container_width=True):
                selected_preset_path = noface_path

    st.markdown("---")
    
    # Custom File Uploader
    uploaded_file = st.file_uploader(
        "Or Upload your own Photo (JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        key="face_uploader"
    )
    
    # Determine which image to load
    img = None
    source_name = ""
    
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            source_name = f"Uploaded File ({uploaded_file.name})"
        except Exception as e:
            st.error(f"Error opening uploaded image: {str(e)}")
    elif selected_preset_path is not None:
        try:
            img = Image.open(selected_preset_path)
            source_name = f"Preset Image ({os.path.basename(selected_preset_path)})"
        except Exception as e:
            st.error(f"Error opening preset image: {str(e)}")

    # Execution and outputs
    if img is not None:
        col_img, col_results = st.columns([1, 1])
        
        # Run detection
        with st.spinner("Executing Haar Cascade detection..."):
            try:
                # Detect face coordinates
                faces_coords = detect_faces(img, scale_factor=scale_factor, min_neighbors=min_neighbors)
                num_faces = len(faces_coords)
                
                # Draw boxes
                img_annotated = draw_faces(img, faces_coords)
                
                # Crop faces
                face_crops = crop_faces(img, faces_coords)
                
            except Exception as e:
                st.error(f"Face detection process failed: {str(e)}")
                faces_coords = []
                num_faces = 0
                img_annotated = img
                face_crops = []

        with col_img:
            st.markdown(f"#### 🔍 Source: `{source_name}`")
            # Show original image with drawn face boxes
            st.image(img_annotated, caption="Annotated Image (Detected faces boxed in green)", use_container_width=True)
            
        with col_results:
            st.markdown("#### 📊 Detection Results")
            
            # Glowing face count banner
            st.markdown(f"""
            <div class="count-output">
                👤 Detected Faces Count: {num_faces}
            </div>
            """, unsafe_allow_html=True)
            
            # Print coordinates list
            if num_faces > 0:
                with st.expander("📍 View Bounding Box Coordinates", expanded=False):
                    for i, (x, y, w, h) in enumerate(faces_coords):
                        st.write(f"**Face {i+1}:** `X={x}, Y={y}, Width={w}, Height={h}`")
                        
                # Cropped faces gallery grid
                st.markdown("### ✂️ Cropped Faces Gallery")
                st.write("Individual face regions extracted from the photo:")
                
                # Show crops in grid columns (up to 4 per row)
                num_cols = min(4, num_faces)
                crop_cols = st.columns(num_cols)
                
                for idx, crop in enumerate(face_crops):
                    col_idx = idx % num_cols
                    with crop_cols[col_idx]:
                        # Make crop square for uniform layout
                        st.image(crop.resize((150, 150)), caption=f"Face {idx+1}", use_container_width=True)
            else:
                st.info("No faces detected with current parameter configurations. Try lowering Minimum Neighbors or Scale Factor in the sidebar if faces are visible.")

# ----------------- TAB 2: DATASET EXPLORER -----------------
with tab_explorer:
    st.subheader("📊 Dataset Explorer")
    
    if not os.path.exists(CLEANED_DATASET_PATH):
        st.info("No metadata dataset found.")
    else:
        df_meta = pd.read_csv(CLEANED_DATASET_PATH)
        st.write("Below is the metadata tracking details of our sample image collection used to test the Face Detector.")
        
        # Search keyword filter
        search_query = st.text_input("🔍 Filter metadata by Image Name:", "")
        if search_query:
            filtered_df = df_meta[df_meta['Image Name'].str.contains(search_query, case=False, na=False)]
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(df_meta, use_container_width=True, hide_index=True)
            
        st.markdown("---")
        st.subheader("💡 Metadata Columns Explained")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown("""
            **`Image Name`**
            * The file filename of the test photo in the `Images/` folder.
            """)
        with col_c2:
            st.markdown("""
            **`Width` / `Height` / `Channels`**
            * Dimensions in pixels and colour space representation (3 channels = standard RGB color).
            """)
        with col_c3:
            st.markdown("""
            **`Face Count` / `Coordinates`**
            * Total faces detected and the bounding boxes `(x; y; width; height)` resolved during evaluation.
            """)

# ----------------- TAB 3: HOW IT WORKS -----------------
with tab_works:
    st.subheader("🧠 How Face Detection with OpenCV Works")
    
    # Mermaid flowchart of OpenCV pipeline
    st.markdown("### 🛠️ Processing Pipeline Flowchart")
    st.write("Below is the visual pipeline from raw photo input to rendering coordinates:")
    st.markdown("""
    ```mermaid
    graph TD
        A[RGB Input Image] --> B[Grayscale Conversion]
        B --> C[Pyramid Image Scaling]
        C --> D[Haar Cascade Classifiers scan]
        D --> E[Group rectangles / filter noise]
        E --> F[Bounding Coordinates output]
        F --> G[Draw Green Bounding Boxes]
        F --> H[Extract Face Crops]
    ```
    """)
    
    st.markdown("---")
    
    col_ed1, col_ed2 = st.columns(2)
    
    with col_ed1:
        st.markdown("""
        #### 🌓 Grayscale Preprocessing
        OpenCV Haar Cascades require the image to be converted to grayscale before scan. Why?
        1. **Color Reduction:** Color channels (RGB) add complexity without useful edge details. Grayscale simplifies the array to a single intensity channel.
        2. **Contrast Patterns:** Face detection relies on the light and dark contrast patterns (e.g. eye sockets are usually darker than the nose bridge) rather than skin color.
        """)
        
    with col_ed2:
        st.markdown("""
        #### 👥 What is Haar Cascade?
        *   **Haar Features:** Rectangular filters that scan the image looking for specific edge and line structures.
        *   **Integral Images:** A mathematical technique that allows sum calculation of pixels in any sub-rectangle in constant $O(1)$ time, making the detector extremely fast.
        *   **AdaBoost:** An algorithm used to select the most critical features (e.g., from 160,000+ features, selecting only a few thousand key ones).
        *   **Cascading:** Features are grouped in stages. The detector rejects non-face regions at early stages (e.g., background sky) instantly, and only spends CPU power scanning complex areas.
        """)
