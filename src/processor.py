import torch
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor
from torchvision.transforms import v2

class TrafficProcessor:
    """
    Unified processor for multimodal data. Handles image feature extraction,
    data augmentation for robustness, and text tokenization/decoding.
    """
    def __init__(self, model_checkpoint="nlpconnect/vit-gpt2-image-captioning", is_training=False):
        """
        Initializes the vision processor and language tokenizer.
        
        Args:
            model_checkpoint (str): The HF checkpoint for pretrained weights.
            is_training (bool): If True, enables stochastic data augmentation.
        """
        # Feature extractor for Vision Transformer (ViT)
        self.imageProcessor = ViTImageProcessor.from_pretrained(model_checkpoint)
        
        # Tokenizer for GPT-2 decoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        
        # In GPT-2, the EOS token typically serves as the padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.is_training = is_training

        # Stochastic augmentations to improve model generalization in diverse 
        # traffic environments (lighting, weather, camera noise)
        self.augmentation = v2.Compose([
            v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3),
            v2.RandomGrayscale(p=0.15),
            v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5)
        ])

    def preprocess(self, images):
        if not isinstance(images, list):
            images = [images]
        
        processed_list = []
        for img in images:
           
            if isinstance(img, str):
                curr_img = Image.open(img).convert("RGB")
            else:
                curr_img = img.convert("RGB")
            
            if self.is_training:
                curr_img = self.augmentation(curr_img)
                
            processed_list.append(curr_img)

        return self.imageProcessor(images=processed_list, return_tensors="pt")["pixel_values"]
    
    def decode(self, IDs):
        """
        Converts model-generated token IDs back into human-readable strings.
        
        Args:
            IDs (torch.Tensor): Output IDs from the decoder.
        Returns:
            list[str]: Cleaned and stripped text captions.
        """
        tokens = self.tokenizer.batch_decode(IDs, skip_special_tokens=True)
        return [t.strip() for t in tokens]

    def tokenize_text(self, text):
        """
        Prepares text for the decoder by converting strings to token sequences.
        
        Args:
            text (str or list): Text captions to tokenize.
        Returns:
            dict: Dictionary containing input_ids and attention_masks.
        """
        return self.tokenizer(
            text, 
            padding='max_length', # Use fixed length for efficient batching
            max_length=32,        # Standard length for short scene descriptions
            truncation=True,      # Prevent overflow beyond model's context window
            return_tensors="pt"
        )