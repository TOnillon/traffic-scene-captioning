# 👁️ Vision-Mentor: Multimodal Image Captioning Pipeline
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers/index)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-orange)](https://gradio.app/)

This project implements a professional-grade **Vision-Encoder-Decoder** pipeline (ViT-GPT2) designed for high-fidelity image description and multimodal understanding.

##  System Architecture (`src/`)

The core logic is partitioned into specialized modules to ensure high maintainability and industrial scalability:

* **`src/model.py`**: Encapsulates the `TrafficModel` class, utilizing a "frozen-encoder" strategy to stabilize visual feature extraction while fine-tuning the language head.
* **`src/processor.py`**: Manages the `TrafficProcessor`, synchronizing image transformation (`ViTImageProcessor`) with text tokenization (`GPT2Tokenizer`).
* **`src/data_loader.py`**: Implements the `TrafficDataLoader` with **streaming** capabilities to handle the COCO-Caption dataset efficiently without heavy local storage.
* **`src/train.py`**: The training engine utilizing Hugging Face's `Seq2SeqTrainer` for automated checkpointing, evaluation, and metric logging.
* **`src/predict.py`**: A robust CLI utility for executing rapid inference on local images using the trained model.



##  Quality Assurance & Testing (`tests/`)

To ensure production-level reliability, every component is covered by a suite of automated tests using `pytest`:

* **`tests/test_model.py`**: Validates architecture integrity and text generation output.
* **`tests/test_processor.py`**: Ensures correct image-to-tensor transformations and decoding logic.
* **`tests/test_data_loader.py`**: Tests streaming data integrity, filtering, and dataset splitting.
* **`tests/test_train.py`**: Verifies the training loop functionality using mock data objects.
* **`tests/test_predict.py`**: Confirms end-to-end inference consistency.

##  Interactive Demos & Applications

* **`demo_app.py`**: A web-based interface powered by **Gradio**, allowing real-time image uploads and caption generation.
* **`demo_inference.ipynb`**: A comprehensive Jupyter Notebook providing a visual walkthrough of the model's capabilities, from data sampling to ground-truth comparison.
