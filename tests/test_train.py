import pytest
import os
import shutil
from src.train import train

def test_train_loop_functional():
    """
    Test the full training pipeline using mock data.
    Verifies that the model can complete 1 epoch and save artifacts.
    """
    # Define paths
    temp_model_dir = "./models/traffic_model_v1"
    final_model_dir = "./models/traffic_model_v1_final"

    # Ensure a clean state before testing
    for path in [temp_model_dir, final_model_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)

    try:
        # Run training for 1 epoch only
        # This validates: data loading, forward pass, loss calc, and metrics
        train(epochs=1, dataset_name="mock")

        # Assertions: Check if the trainer saved the expected files
        assert os.path.exists(final_model_dir), "Final model directory was not created"
        assert os.path.exists(os.path.join(final_model_dir, "config.json")), "config.json missing in final model"
        assert os.path.exists(os.path.join(final_model_dir, "generation_config.json")), "generation_config missing"
        
        # Check if at least one checkpoint was saved
        assert os.path.exists(temp_model_dir), "Training checkpoint directory missing"

    finally:
        # Cleanup: Remove generated models after the test to keep the workspace clean
        if os.path.exists("./models"):
            shutil.rmtree("./models")

if __name__ == "__main__":
    # Allows running this specific test file directly
    test_train_loop_functional()