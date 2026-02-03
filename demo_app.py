import gradio as gr
import torch
from PIL import Image
from src.model import TrafficModel
from src.processor import TrafficProcessor

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TrafficModel(model_checkpoint="./models/traffic_model_v1_final")
model.model.to(device)
processor = TrafficProcessor(is_training=False)

def caption_image(image):
    """Generate caption for uploaded image"""
    if image is None:
        return "Please upload an image"
    
    try:
        # Preprocess
        pixel_values = processor.preprocess(image).to(device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(pixel_values)
        
        # Decode
        caption = processor.decode(output_ids)[0]
        return caption
    except Exception as e:
        return f"Error: {str(e)}"

# Create interface
demo = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil", label="Upload Scene"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Vision-Mentor: Scene Captioning",
    description="Upload a scene image to generate a description",
    examples=[
        ["test_image.jpg"]
    ]
)

if __name__ == "__main__":
    demo.launch()