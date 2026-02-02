from src.model import TrafficModel
import torch

def test_model():
    model=TrafficModel()

    fake_pixel_values=torch.randn((1,3,224,224))
    
    IDs=model.generate(fake_pixel_values)

    assert isinstance(IDs,torch.Tensor)
    assert IDs.dtype == torch.long
    assert IDs.shape[1]<=16
    