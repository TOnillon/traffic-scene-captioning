import pytest
import os
from src.predict import predict

def test_predict_flow():
    """
    Checks if the predict function runs without crashing and returns a string.
    """
    image_path = "test_image.jpg"
    
    # Skip if the user deleted the test image
    if not os.path.exists(image_path):
        pytest.skip("Test image not found at root.")

    # We test the base model inference
    try:
        # Note: update your predict() function in src/predict.py 
        # to 'return caption' at the end for this test to be fully useful
        caption = predict(image_path, model_path=None)
        
        assert isinstance(caption, str), "The predicted caption should be a string"
        assert len(caption) > 0, "The predicted caption should not be empty"
        
    except Exception as e:
        pytest.fail(f"Prediction failed with error: {e}")