import torch
from PIL import Image
from src.model import TrafficModel
from src.processor import TrafficProcessor

def predict(image_path, model_path=None):
    """
    Performs end-to-end inference on a single traffic image.
    
    Args:
        image_path (str): Path to the input image file.
        model_path (str, optional): Path to a fine-tuned model checkpoint. 
                                    Defaults to the base pretrained model.
    Returns:
        str: The generated textual description of the traffic scene.
    """
    # Detect hardware availability for optimized inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Component Initialization
    # Initialize the processor (handles normalization and tokenization)
    processor = TrafficProcessor()
    
    # Load either the baseline or a specialized fine-tuned model
    if model_path:
        traffic_model = TrafficModel(model_checkpoint=model_path)
    else:
        traffic_model = TrafficModel()
    
    # 2. Input Pipeline
    # Load and ensure RGB format to match model expectations
    raw_image = Image.open(image_path).convert("RGB")
    
    # Preprocess image into a standardized tensor [1, 3, 224, 224]
    pixel_values = processor.preprocess(raw_image).to(device)
    
    # 3. Model Generation
    print(f"Processing image: {image_path}...")
    
    # Disable gradient calculation to reduce memory consumption and latency
    with torch.no_grad():
        output_ids = traffic_model.generate(pixel_values)
    
    # 4. Post-processing
    # Transform numerical IDs back into human-readable text
    caption = processor.decode(output_ids)[0]
    
    print("-" * 30)
    print(f"Predicted Caption: {caption}")
    print("-" * 30)
    
    return caption

if __name__ == "__main__":
    """
    Entry point for CLI-based inference.
    Handles basic error reporting for missing files or runtime failures.
    """
    try:
        # Default test case for local verification
        predict("test_image.jpg", model_path=None)
    except FileNotFoundError:
        print("Error: 'test_image.jpg' not found. Please provide a valid image path.")
    except Exception as e:
        print(f"An error occurred during inference: {e}")