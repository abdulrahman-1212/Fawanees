import numpy as np
from engines.classical import cartoonize as classical_cartoonize, STYLES as CLASSICAL_STYLES

# Global variable for lazy loading
_cartoongan_model = None

def get_gan_model(device: str = "cuda"):
    global _cartoongan_model
    if _cartoongan_model is None:
        from engines.dl_inference import CartoonGANInference
        _cartoongan_model = CartoonGANInference(device=device)
    return _cartoongan_model

def process_image(img_bgr: np.ndarray, style: str, use_gan: bool = False, device: str = "cuda") -> np.ndarray:
    """
    Unified pipeline for cartoonization.
    Routes to CartoonGAN or Classical CV based on flags and style.
    """
    if use_gan and style in ["cartoongan", "anime_gan"]:
        model = get_gan_model(device)
        return model.process(img_bgr)
    else:
        if style not in CLASSICAL_STYLES:
            raise ValueError(f"Unknown classical style: '{style}'. Choose from: {list(CLASSICAL_STYLES)}")
        return classical_cartoonize(img_bgr, style=style)