from src.processor import TrafficProcessor
from src.model import TrafficModel

def main():
    image_path = "test_image.jpg"
    
    print("[INFO] Initializing Traffic Captioning System...")
    _processor = TrafficProcessor()
    _model = TrafficModel()

    print(f"[INFO] Processing image: {image_path}")
    # 1. Preprocess the image into pixels
    pixel_values = _processor.preprocess(image_path)
    
    print("[INFO] Generating caption (this may take a few seconds on CPU)...")
    # 2. Model generates token IDs
    IDs = _model.generate(pixel_values)
    
    print("[INFO] Decoding results...")
    # 3. Decode IDs back into human-readable text
    final_caption = _processor.decode(IDs)

    print("\n" + "="*30)
    print(f"FINAL PREDICTION: {final_caption[0]}")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()
