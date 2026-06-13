# 🎨 Advanced Cartoonization Pipeline

A dual-engine, production-ready image processing tool that combines **Classical Computer Vision (OpenCV)** and **Deep Learning (CartoonGAN)** to transform photographs into cartoon, anime, and sketch styles.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-red?logo=opencv)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit)

---

## ✨ Key Features

- 🚀 **Dual-Engine Architecture**: Seamlessly switch between instant, pure-math OpenCV algorithms and high-quality PyTorch Generative Adversarial Networks.
- 🧠 **CartoonGAN (CVPR 2018)**: Full implementation of the paper featuring Sparse Regularization and Edge-Promoting Adversarial Training.
- ⚡ **Smart Routing & Lazy-Loading**: Heavy AI models are *only* loaded into VRAM/RAM when explicitly requested by the user, keeping the app lightning-fast on startup.
- 📐 **Adaptive Resolution Scaling**: Automatically downscales 4K+ images for heavy classical processing (like K-Means and Bilateral Filtering) to prevent freezing, then upscales the final result.
- 🎛️ **Multiple Interfaces**: Includes a fully interactive **Streamlit Web UI** and a batch-processing **Command Line Interface (CLI)**.

---

## 📂 Project Architecture

The project is strictly separated into **Inference Engines** (used by the app) and **Networks** (used only for training).

```text
cartoonizer-project/
│
├── app.py                  # 🌐 Streamlit Web UI (Entry Point 1)
├── cartoonize.py           # 💻 Command Line Interface (Entry Point 2)
├── pipeline.py             # 🔀 Unified Router (Handles lazy-loading & routing)
│
├── engines/                # ⚙️ Inference Engines
│   ├── __init__.py
│   ├── classical.py        # Pure OpenCV styles (comic, anime, sketch, etc.)
│   └── dl_inference.py     # CartoonGAN inference wrapper
│
├── networks/               # 🧠 Deep Learning Architectures & Training
│   ├── arch.py             # Generator, Discriminator, VGG19
│   ├── losses.py           # Sparse Content Loss & Edge-Promoting Loss
│   └── train.py            # Training loop (Init Phase + GAN Phase)
│
├── assets/                 # 📦 Pre-trained weights
│   └── cartoongan.pth      
│
├── data/                   # 🗂️ Training Datasets (Unpaired)
│   ├── photos/             
│   └── cartoons/           
│
├── requirements.txt        # 📦 Python dependencies
└── README.md               