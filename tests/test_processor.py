from PIL import Image
from src.processor import TrafficProcessor
import torch
import os

def test_preprocess_output_shape():
    proc = TrafficProcessor()

    temp_path = "temp_test_image.jpg"
    fake_image = Image.new('RGB', (100, 100), color=(0, 0, 255))
    fake_image.save(temp_path)


    fake_ids = torch.tensor([[64, 1097, 319, 262, 2975]])
    fake_text = "a car on the road"


    pixel_values = proc.preprocess(temp_path)
    tokens_text = proc.tokenize_text(fake_text)
    decoded_sentences = proc.decode(fake_ids)



    assert isinstance(pixel_values, torch.Tensor)
    assert pixel_values.shape == (1, 3, 224, 224)

    assert "input_ids" in tokens_text
    assert tokens_text.input_ids.shape == (1, 32)

    assert isinstance(decoded_sentences, list)
    assert len(decoded_sentences) == 1
    assert isinstance(decoded_sentences[0], str)

    if os.path.exists(temp_path):
        os.remove(temp_path)