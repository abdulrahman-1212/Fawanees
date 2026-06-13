import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from io import BytesIO
from pipeline import process_image, CLASSICAL_STYLES

# Cache the pipeline initialization to avoid reloading heavy models on every interaction
@st.cache_resource
def initialize_pipeline(use_gan, device):
    # Trigger lazy load with a dummy image
    dummy = np.zeros((10, 10, 3), dtype=np.uint8)
    process_image(dummy, style="cartoongan" if use_gan else "comic", use_gan=use_gan, device=device)
    return True

st.set_page_config(page_title="Advanced Cartoonizer", page_icon="🎨", layout="wide")
st.title("🎨 Advanced Cartoonization Pipeline")
st.markdown("Supports both **Pure Computer Vision** and **Deep Learning (CartoonGAN)** methods.")

with st.sidebar:
    st.header("⚙️ Settings")
    method = st.radio("Processing Method", ["Classical CV (Fast)", "CartoonGAN (High Quality)"])
    
    if method == "Classical CV (Fast)":
        style = st.selectbox("Choose Style", list(CLASSICAL_STYLES), index=0)
        use_gan = False
        device = "cpu" 
    else:
        style = "cartoongan"
        use_gan = True
        device = st.selectbox("Compute Device", ["cuda", "cpu"], index=0)
        st.info("💡 Ensure `cartoongan.pth` is in the `assets/` directory.")

st.markdown("### 📤 Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_column_width=True)
        
    if st.button("🎨 Cartoonize Image", type="primary", use_container_width=True):
        initialize_pipeline(use_gan, device)
        
        with st.spinner(f"Applying {style} style using {device.upper()}..."):
            start_time = time.time()
            try:
                result_bgr = process_image(img_bgr, style=style, use_gan=use_gan, device=device)
                elapsed = time.time() - start_time
                
                result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader(f"Result — {style.replace('_', ' ').title()}")
                    st.image(result_rgb, use_column_width=True)
                    st.success(f"✅ Done in {elapsed:.2f} seconds")
                    
                    # Download button
                    buf = BytesIO()
                    Image.fromarray(result_rgb).save(buf, format="PNG")
                    st.download_button(
                        label="⬇️ Download Cartoon",
                        data=buf.getvalue(),
                        file_name=f"{uploaded_file.name.split('.')[0]}_{style}.png",
                        mime="image/png"
                    )
            except Exception as e:
                st.error(f"❌ Error during processing: {e}")
else:
    st.info("👆 Upload an image to get started")

st.caption("Built with OpenCV, PyTorch & Streamlit • Modular Pipeline Architecture")