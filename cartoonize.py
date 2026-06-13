#!/usr/bin/env python3
"""
Cartoonizer — Streamlit Web Interface
=====================================
Run with: streamlit run app.py
"""

import cv2
import numpy as np
import streamlit as st
import tempfile
from pathlib import Path

# Project-local imports (run from the cartoonizer/ directory)
from config.settings import CartoonConfig
from pipeline.image_pipeline import ImagePipeline
from pipeline.video_pipeline import VideoPipeline
from imgio.image_io import build_debug_grid

# Supported extensions
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

st.set_page_config(page_title="Cartoonizer", page_icon="🎨", layout="wide")

# --- Helper Functions ---

def get_file_extension(file_name: str) -> str:
    return Path(file_name).suffix.lower()

def build_config() -> CartoonConfig:
    """Build CartoonConfig from Streamlit UI state."""
    cfg = CartoonConfig.preset(st.session_state.preset)
    
    # Edge overrides
    if st.session_state.canny_low_override:
        cfg.edge.canny_low = st.session_state.canny_low
    if st.session_state.canny_high_override:
        cfg.edge.canny_high = st.session_state.canny_high
    if st.session_state.blur_kernel_override:
        cfg.edge.blur_kernel = st.session_state.blur_kernel
    if st.session_state.dilate_override:
        cfg.edge.dilate_kernel = st.session_state.dilate_kernel
        
    cfg.edge.detector_type = st.session_state.edge_detector
    
    # Smooth overrides
    if st.session_state.bilateral_iter_override:
        cfg.smooth.bilateral_iterations = st.session_state.bilateral_iter
    cfg.smooth.color_space = st.session_state.color_space
    
    # Quantise overrides
    if st.session_state.k_override:
        cfg.quantize.k = st.session_state.k
    cfg.quantize.enabled = not st.session_state.no_quantize
    
    # Video overrides
    if st.session_state.max_frames_override:
        cfg.video.max_frames = st.session_state.max_frames
        
    return cfg

def process_uploaded_image(uploaded_file, cfg: CartoonConfig, debug: bool):
    """Process an uploaded image file."""
    uploaded_file.seek(0)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        return None, None, None
        
    pipe = ImagePipeline(cfg)
    result = pipe.run(
        image,
        debug=debug,
        edge_detector=cfg.edge.detector_type,
        color_space=cfg.smooth.color_space,
    )
    
    result_rgb = cv2.cvtColor(result.cartoon, cv2.COLOR_BGR2RGB)
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    debug_grid_rgb = None
    if debug and getattr(result, 'intermediates', None):
        stages = {"original": image, **result.intermediates, "cartoon": result.cartoon}
        grid = build_debug_grid(stages, cols=3)
        debug_grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        
    return original_rgb, result_rgb, debug_grid_rgb

def process_uploaded_video(uploaded_file, cfg: CartoonConfig):
    """Process an uploaded video file."""
    uploaded_file.seek(0)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=get_file_extension(uploaded_file.name)) as tmp_in:
        tmp_in.write(uploaded_file.read())
        input_path = Path(tmp_in.name)
        
    output_path = input_path.parent / f"cartoon_{input_path.name}"
    
    try:
        vp = VideoPipeline(cfg)
        stats = vp.process(
            input_path,
            output_path,
            max_frames=cfg.video.max_frames,
            edge_detector=cfg.edge.detector_type,
            color_space=cfg.smooth.color_space,
        )
        
        with open(output_path, "rb") as f:
            video_bytes = f.read()
            
        return video_bytes, stats
        
    finally:
        if input_path.exists():
            input_path.unlink()
        if output_path.exists():
            output_path.unlink()

# --- Streamlit UI ---

st.title("🎨 Cartoonizer")
st.markdown("Convert your images and videos into beautiful cartoon styles using advanced computer vision pipelines.")

# Sidebar: Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    preset = st.selectbox(
        "Preset",
        options=["default", "sketch", "watercolor", "comic", "fast"],
        index=0,
        key="preset",
        help="Named configuration preset to start from."
    )
    
    with st.expander("🔍 Edge Detection", expanded=False):
        st.checkbox("Override Canny Low", key="canny_low_override")
        st.slider("Canny Low", 10, 255, 50, key="canny_low", disabled=not st.session_state.canny_low_override)
        
        st.checkbox("Override Canny High", key="canny_high_override")
        st.slider("Canny High", 10, 255, 150, key="canny_high", disabled=not st.session_state.canny_high_override)
        
        st.checkbox("Override Blur Kernel", key="blur_kernel_override")
        st.selectbox("Blur Kernel", [3, 5, 7, 9, 11], key="blur_kernel", disabled=not st.session_state.blur_kernel_override)
        
        st.selectbox("Edge Detector", ["canny", "adaptive"], key="edge_detector")
        
        st.checkbox("Override Dilate Kernel", key="dilate_override")
        st.selectbox("Dilate Kernel", [0, 1, 2, 3, 5], key="dilate_kernel", disabled=not st.session_state.dilate_override)

    with st.expander("🎨 Colour Smoothing", expanded=False):
        st.checkbox("Override Bilateral Iterations", key="bilateral_iter_override")
        st.slider("Bilateral Iterations", 1, 10, 2, key="bilateral_iter", disabled=not st.session_state.bilateral_iter_override)
        st.selectbox("Color Space", ["bgr", "lab"], key="color_space")

    with st.expander("🖌️ Colour Quantisation", expanded=False):
        st.checkbox("Override Palette Size (K)", key="k_override")
        st.slider("Palette Colours (K)", 2, 32, 6, key="k", disabled=not st.session_state.k_override)
        st.checkbox("Disable Quantisation", key="no_quantize")

    with st.expander("🎬 Video Settings", expanded=False):
        st.checkbox("Limit Max Frames", key="max_frames_override")
        st.number_input("Max Frames (0 = all)", min_value=0, value=120, step=10, key="max_frames", disabled=not st.session_state.max_frames_override)

    st.divider()
    st.checkbox("Show Debug Grid", key="show_debug", help="Display intermediate processing stages.")

# Main Area: Upload and Process
uploaded_file = st.file_uploader(
    "Upload an Image or Video", 
    type=list(IMAGE_EXTS.union(VIDEO_EXTS))
)

if uploaded_file is not None:
    # Clear previous results if a new file is uploaded
    if "last_uploaded_file" not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.last_uploaded_file = uploaded_file.name
        for key in ["result_rgb", "debug_grid_rgb", "video_bytes", "video_stats", "processing_error"]:
            if key in st.session_state:
                del st.session_state[key]

    file_ext = get_file_extension(uploaded_file.name)
    is_video = file_ext in VIDEO_EXTS
    
    # Layout: Input and Output on the same level (side-by-side)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original")
        if not is_video:
            uploaded_file.seek(0)
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            orig_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if orig_img is not None:
                orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                st.image(orig_rgb, use_container_width=True)
            else:
                st.error("Failed to decode original image.")
        else:
            st.video(uploaded_file)
            
        # Button placed directly below the input image/video
        if st.button("🚀 Cartoonize!", type="primary", use_container_width=True):
            cfg = build_config()
            if not is_video:
                with st.spinner("Processing image..."):
                    orig_rgb, result_rgb, debug_grid_rgb = process_uploaded_image(uploaded_file, cfg, st.session_state.show_debug)
                if result_rgb is not None:
                    st.session_state.result_rgb = result_rgb
                    st.session_state.debug_grid_rgb = debug_grid_rgb
                    st.session_state.processing_error = None
                else:
                    st.session_state.processing_error = "Processing failed. Check file format."
            else:
                with st.spinner("Processing video... (This may take a while)"):
                    video_bytes, stats = process_uploaded_video(uploaded_file, cfg)
                st.session_state.video_bytes = video_bytes
                st.session_state.video_stats = stats
                st.session_state.processing_error = None

    with col2:
        st.subheader("Cartoonized Result")
        
        if not is_video:
            if "result_rgb" in st.session_state and st.session_state.result_rgb is not None:
                st.image(st.session_state.result_rgb, caption="Cartoonized", use_container_width=True)
                
                _, encoded_image = cv2.imencode(".png", cv2.cvtColor(st.session_state.result_rgb, cv2.COLOR_RGB2BGR))
                st.download_button(
                    label="📥 Download Image",
                    data=encoded_image.tobytes(),
                    file_name=f"cartoon_{uploaded_file.name}",
                    mime="image/png",
                    use_container_width=True
                )
                
                if st.session_state.show_debug and "debug_grid_rgb" in st.session_state and st.session_state.debug_grid_rgb is not None:
                    st.markdown("### 🔍 Debug Grid")
                    st.image(st.session_state.debug_grid_rgb, use_container_width=True)
                    
                    _, encoded_debug = cv2.imencode(".jpg", cv2.cvtColor(st.session_state.debug_grid_rgb, cv2.COLOR_RGB2BGR))
                    st.download_button(
                        label="📥 Download Debug Grid",
                        data=encoded_debug.tobytes(),
                        file_name=f"debug_{uploaded_file.name}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
            elif "processing_error" in st.session_state and st.session_state.processing_error:
                st.error(st.session_state.processing_error)
            else:
                st.info("Click 'Cartoonize!' to see the result here.")
        else:
            if "video_bytes" in st.session_state:
                st.video(st.session_state.video_bytes)
                stats = st.session_state.video_stats
                
                st.markdown(f"""
                **Processing Stats:**  
                🎞️ Frames Processed: `{stats['frames_processed']}`  
                ⏱️ Elapsed Time: `{stats['elapsed_seconds']:.1f}s`  
                ⚡ Average FPS: `{stats['average_fps']:.1f}`
                """)
                
                st.download_button(
                    label="📥 Download Video",
                    data=st.session_state.video_bytes,
                    file_name=f"cartoon_{uploaded_file.name}",
                    mime="video/mp4",
                    use_container_width=True
                )
            elif "processing_error" in st.session_state and st.session_state.processing_error:
                st.error(st.session_state.processing_error)
            else:
                st.info("Click 'Cartoonize!' to see the result here.")