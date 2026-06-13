"""
CartoonGAN Extension
====================
Drop-in subclass of ImagePipeline that routes frames through a CartoonGAN
(or any compatible DL model) between quantisation and compositing.

How to activate
---------------
1. Install pytorch / onnxruntime:
       pip install torch torchvision          # PyTorch path
       pip install onnxruntime-gpu            # ONNX path (lighter dependency)

2. Download a CartoonGAN checkpoint, e.g.:
       https://github.com/SystemErrorWang/CartoonGAN
       https://github.com/experience-ml/cartoonize  (ONNX weights available)

3. Implement one of the two loader stubs below and pass the model path:
       pipe = GANPipeline("weights/cartoongan_hosoda.onnx")
       result = pipe.run(bgr_image)

Notes
-----
* The GAN replaces the quantise → composite colour path; edges are still
  detected from the original frame and composited on top.
* Input tensors expect RGB, float32, normalised to [-1, 1].
* Output tensors are de-normalised back to BGR uint8 here.
* The temporal smoothing in VideoPipeline applies on top of GAN output,
  so flicker is handled the same way for both paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config.settings import CartoonConfig
from pipeline.image_pipeline import ImagePipeline


class GANPipeline(ImagePipeline):
    """
    ImagePipeline with a CartoonGAN DL stage.

    The classic pipeline (smooth → quantise) still runs and its output is
    passed into the GAN, which refines it before edge compositing.
    Keeping the classic pre-processing reduces GAN inference artifacts on
    real-world photographs.
    """

    def __init__(
        self,
        model_path: str | Path,
        backend: str = "onnx",        # "onnx" | "torch"
        cfg: Optional[CartoonConfig] = None,
        input_size: tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__(cfg)
        self.input_size = input_size
        self.backend = backend
        self.model = self._load_model(Path(model_path), backend)

    # ------------------------------------------------------------------
    # Override hook
    # ------------------------------------------------------------------

    def _apply_dl_stage(self, image: np.ndarray) -> np.ndarray:
        """Route through GAN inference."""
        if self.model is None:
            return image
        if self.backend == "onnx":
            return self._infer_onnx(image)
        elif self.backend == "torch":
            return self._infer_torch(image)
        return image

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(path: Path, backend: str):
        """Load model weights; returns None and prints a warning on failure."""
        if not path.exists():
            print(f"[GANPipeline] Warning: model file not found at {path}. "
                  "DL stage will be skipped.")
            return None

        if backend == "onnx":
            try:
                import onnxruntime as ort
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                sess = ort.InferenceSession(str(path), providers=providers)
                print(f"[GANPipeline] ONNX model loaded from {path}")
                return sess
            except ImportError:
                print("[GANPipeline] onnxruntime not installed — "
                      "run: pip install onnxruntime  (or onnxruntime-gpu)")
                return None

        elif backend == "torch":
            try:
                import torch
                model = torch.jit.load(str(path), map_location="cpu")
                model.eval()
                print(f"[GANPipeline] TorchScript model loaded from {path}")
                return model
            except ImportError:
                print("[GANPipeline] torch not installed — run: pip install torch")
                return None

        raise ValueError(f"Unknown backend '{backend}'. Choose 'onnx' or 'torch'.")

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """BGR uint8 → RGB float32 tensor, shape (1, 3, H, W), range [-1, 1]."""
        h, w = self.input_size
        resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = (rgb / 127.5 - 1.0).transpose(2, 0, 1)[np.newaxis]  # NCHW
        return tensor

    def _postprocess(
        self,
        output: np.ndarray,
        original_shape: tuple[int, int],
    ) -> np.ndarray:
        """NCHW float32 [-1,1] → BGR uint8, resized back to original_shape."""
        arr = output[0].transpose(1, 2, 0)           # CHW → HWC
        arr = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, (original_shape[1], original_shape[0]),
                          interpolation=cv2.INTER_LINEAR)

    def _infer_onnx(self, image: np.ndarray) -> np.ndarray:
        tensor = self._preprocess(image)
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: tensor})
        return self._postprocess(outputs[0], image.shape[:2])

    def _infer_torch(self, image: np.ndarray) -> np.ndarray:
        import torch
        tensor = torch.from_numpy(self._preprocess(image))
        with torch.no_grad():
            output = self.model(tensor).numpy()
        return self._postprocess(output, image.shape[:2])


__all__ = ["GANPipeline"]