import torch
from PIL import Image
from src.model import TrafficModel
from src.processor import TrafficProcessor

# Globally initialize components to avoid reloading for every image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(image_path, model_path="./models/traffic_model_v1"):
    """
    Performs end-to-end inference on a single traffic image.
    """
    # Initialize processor and model
    processor = TrafficProcessor()
    
    # Load model with the tie_word_embeddings fix to keep logs clean
    checkpoint = model_path if model_path else "nlpconnect/vit-gpt2-image-captioning"
    traffic_model = TrafficModel(model_checkpoint=checkpoint)
    traffic_model.model.config.tie_word_embeddings = False
    traffic_model.model.to(device)
    traffic_model.model.eval()

    # Load and preprocess image
    raw_image = Image.open(image_path).convert("RGB")
    pixel_values = processor.preprocess(raw_image).to(device)

    print(f"Processing image: {image_path}...")
    
    with torch.no_grad():
        # Standardize generation parameters for consistency
        output_ids = traffic_model.generate(pixel_values)
    
    caption = processor.decode(output_ids)[0]
    
    print(f"{'='*30}\nRESULT: {caption}\n{'='*30}")
    return caption

if __name__ == "__main__":
    import sys
    # Allow passing an image path via terminal: python src/predict.py my_image.jpg
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"
    
    try:
        predict(img_path)
    except Exception as e:
        print(f"Inference failed: {e}")
