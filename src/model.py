import torch
from transformers import VisionEncoderDecoderModel

# Setup device-agnostic code for local dev and Toyota's HPC infrastructure [cite: 111]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

class TrafficModel:
    """
    Multimodal model for traffic scene understanding using a ViT encoder 
    and a GPT-2 decoder architecture.
    """
    def __init__(self, model_checkpoint="nlpconnect/vit-gpt2-image-captioning"):
        """
        Loads the pretrained weights and synchronizes configuration IDs.
        
        Args:
            model_checkpoint (str): The Hugging Face hub path or local path to weights.
        """
        # Load the model and move to GPU/CPU [cite: 112]
        self.model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)
        
        # Configure the decoder to use the correct start/end tokens for text generation
        self.model.config.decoder_start_token_id = self.model.config.bos_token_id
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # Explicitly set vocab size from the decoder to avoid generation mismatches
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

    def generate(self, pixel_values):
        """
        Generates a text description for a given batch of images.
        
        Args:
            pixel_values (torch.Tensor): Preprocessed image tensors.
        Returns:
            torch.Tensor: Generated token IDs for the caption.
        """
        # Ensure input tensors are on the same device as the model
        pixel_values = pixel_values.to(device)
        
        # Use beam search to improve the quality of the generated traffic description
        # max_length is kept low (16) for real-time inference efficiency
        return self.model.generate(pixel_values, num_beams=4, max_length=16)