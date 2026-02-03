#  Vision-Mentor: Multimodal Image Captioning Pipeline
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/docs/transformers/index)
[![Gradio](https://img.shields.io/badge/Demo-Gradio-orange)](https://gradio.app/)

A professional-grade multimodal pipeline integrating **Computer Vision** and **Natural Language Processing**. This project implements a **Vision-Encoder-Decoder** architecture (ViT-GPT2) designed for high-fidelity image description and scene understanding.

##  System Architecture

The project is built with a highly modular design to ensure maintainability and industrial scalability:

* **`src/model.py`**: Encapsulates the `VisionEncoderDecoderModel`. It implements a "frozen-encoder" strategy, keeping the ViT weights static to focus training on the language head, optimizing computational efficiency.
* **`src/processor.py`**: Synchronizes the multimodal inputs between the `ViTImageProcessor` and the `GPT2Tokenizer`.
* **`src/data_loader.py`**: Features a high-performance **Streaming Data Pipeline** using the COCO-Caption dataset. This approach handles large-scale data without significant local storage overhead.
* **`src/train.py`**: A standardized training script using `Seq2SeqTrainer`, featuring automated checkpointing, evaluation, and metric logging.



##  Key Engineering Features

* **Data Management**: Leverages Hugging Face `datasets` with streaming to manage high-volume visual datasets efficiently.
* **Inference Optimization**: Implements **Beam Search** decoding (`num_beams=4`) to significantly enhance the semantic coherence of generated captions.
* **R&D Validation**: Automated evaluation using **ROUGE-L**, **BLEU**, and **METEOR** metrics to ensure scientific rigor in performance tracking.
* **Automated Testing**: Comprehensive test suite (`pytest`) covering the data pipeline, model integrity, and prediction flows.

##  Interactive Deployment (Gradio)

To bridge the gap between model training and real-world application, an interactive web UI is provided via **Gradio**.

```bash
# Launch the interactive web demo
python app.py
