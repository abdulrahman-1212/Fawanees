import torch
import cv2
import numpy as np
from torchvision import transforms
import os
from cartoongan_models import Generator

class CartoonGANInference:
    def __init__(self, weights_path: str = "cartoongan.pth", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = Generator().to(self.device)
        
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            print(f"Loaded CartoonGAN weights from {weights_path}")
        else:
            print(f"Warning: Weights not found at {weights_path}.")
            
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    @torch.no_grad()
    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Resize to multiple of 4 for TransposedConvs
        new_h, new_w = (h // 4) * 4, (w // 4) * 4
        if new_h == 0: new_h = 4
        if new_w == 0: new_w = 4
        
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        input_tensor = self.transform(img_resized).unsqueeze(0).to(self.device)
        
        output = self.model(input_tensor)
        
        # Denormalize [-1, 1] -> [0, 255]
        output = output.squeeze(0).cpu().numpy()
        output = (output * 0.5 + 0.5) * 255
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = np.transpose(output, (1, 2, 0))
        
        if (h, w) != (new_h, new_w):
            output = cv2.resize(output, (w, h))
            
        return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)