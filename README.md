# traffic-scene-captioning

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg?logo=pytorch)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)
![License](https://img.shields.io/badge/license-MIT-green.svg)

End-to-end multimodal pipeline for traffic scene understanding using a Vision-Encoder-Decoder (ViT-GPT2) architecture. Features robust image preprocessing with torchvision v2, automated training with Seq2SeqTrainer, and standardized R&D evaluation (ROUGE, BLEU, METEOR). Fully containerized for HPC deployment.

## 🛠 Key Components

* **Robust Signal Processing**: Custom `TrafficProcessor` using `torchvision.transforms.v2` (GaussianBlur, ColorJitter) to ensure model resilience against adverse weather and lighting conditions.
* **Production-Ready Data Pipeline**: Batched preprocessing and automated dataset splitting (Train/Test) via `TrafficDataLoader`.
* **Advanced Evaluation**: Integration of ROUGE-L, BLEU, and METEOR metrics to quantify semantic accuracy in scene descriptions.
* **Hardware Agnostic**: Seamless execution on CPU or NVIDIA GPU (CUDA) for both training and inference.

## 🚀 Quick Start (Docker)

The project is fully containerized to ensure reproducibility across different R&D environments.

```bash
# Build the image
docker build -t traffic-captioner-v1 .

# Run inference on a test image (GPU support)
docker run --gpus all traffic-captioner-v1
