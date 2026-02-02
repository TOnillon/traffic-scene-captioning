import torch
from src.data_loader import TrafficDataLoader

def test_data_loader_output_format():
    loader = TrafficDataLoader(dataset_name="mock")
    processed_dataset = loader.prepare_dataset()
    
    sample = processed_dataset["train"][0]
    
    assert "pixel_values" in sample
    assert "labels" in sample
    assert isinstance(sample["pixel_values"], torch.Tensor)
    assert sample["labels"].ndimension() == 1
    assert sample["pixel_values"].shape == (3, 224, 224)