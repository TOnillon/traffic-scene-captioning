import torch
from transformers import VisionEncoderDecoderModel

# Setup device-agnostic code for local development and high-performance computing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

class TrafficModel:
    """
    A Vision-Encoder-Decoder model specialized for traffic scene captioning.
    Uses a ViT (Vision Transformer) encoder and a GPT-2 decoder.
    """
    def __init__(self, model_checkpoint="nlpconnect/vit-gpt2-image-captioning"):
        """
        Initializes the model, freezes the vision encoder, and configures tokens.
        
        Args:
            model_checkpoint (str): The Hugging Face model hub path or local path.
        """
        # Load the pre-trained VisionEncoderDecoder architecture
        self.model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint).to(device)
        
        # Explicitly disable weight tying to match the checkpoint configuration and silence warnings
        self.model.config.tie_word_embeddings = False

        # Freeze the ViT encoder parameters to focus training on the language decoder
        for param in self.model.encoder.parameters():
            param.requires_grad = False
    
        # Map tokens for consistent sequence generation
        self.model.config.decoder_start_token_id = self.model.config.bos_token_id
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

    def generate(self, pixel_values):
        """
        Generates descriptive text for a batch of images using beam search.
        
        Args:
            pixel_values (torch.Tensor): Preprocessed image tensors.
            
        Returns:
            torch.Tensor: Generated token IDs for the output captions.
        """
        # Ensure input tensors are on the same device as the model
        pixel_values = pixel_values.to(device)
        
        # Inference settings: beam search (num_beams=4) for better caption quality
        return self.model.generate(
            pixel_values, 
            num_beams=4, 
            max_length=32, 
            early_stopping=True
        )