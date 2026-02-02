from datasets import load_dataset, Dataset, DatasetDict
from PIL import Image
import torch
from src.processor import TrafficProcessor

class TrafficDataLoader:
    """
    Handles loading and preprocessing of traffic image-captioning datasets.
    Supports both real datasets from Hugging Face Hub and mock data for testing.
    """
    def __init__(self, dataset_name="DamianBoborzi/car_images"):
        """
        Initializes the loader and fetches the raw dataset.
        
        Args:
            dataset_name (str): Name of the HF dataset or 'mock' for local testing.
        """
        self.processor = TrafficProcessor()
        
        if dataset_name == "mock":
            # Create a small red image to simulate car data for CI/CD pipelines
            fake_img = Image.new('RGB', (224, 224), color='red')
            self.dataset = DatasetDict({
                "train": Dataset.from_dict({
                    "image": [fake_img] * 8, 
                    "caption": ["a red car on the road"] * 8
                }),
                "test": Dataset.from_dict({
                    "image": [fake_img] * 2, 
                    "caption": ["a red car on the road"] * 2
                })
            })
        else:
            # Load a subset for rapid experimentation as required in R&D [cite: 122]
            self.dataset = load_dataset(dataset_name, split="train[:500]")

    def preprocess_callback(self, examples):
        """
        Callback function for the map operation. Processes images and text in batches.
        
        Args:
            examples (dict): Batch of raw data from the dataset.
        Returns:
            dict: Processed pixel values and labels (tokenized text).
        """
        img_column = "image"
        # Handle variations in dataset column naming (text vs caption)
        txt_column = "text" if "text" in examples else "caption"
        
        # Transform raw images into tensors
        pixel_values = self.processor.preprocess(examples[img_column])
        # Convert text into numerical tokens
        tokens = self.processor.tokenize_text(examples[txt_column])
        
        # Clone tokens to create labels for the cross-entropy loss
        labels = tokens["input_ids"].clone()
        # Replace padding token id with -100 so the loss function ignores them
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

    def prepare_dataset(self):
        """
        Splits, maps, and formats the dataset for PyTorch training.
        
        Returns:
            DatasetDict: Fully processed dataset ready for the Trainer API[cite: 115].
        """
        # Ensure we have both train and test splits
        if isinstance(self.dataset, DatasetDict):
            ds_dict = self.dataset
        else:
            # Standard 80/20 split for model evaluation [cite: 115]
            ds_dict = self.dataset.train_test_split(test_size=0.2, seed=42)

        # Apply preprocessing at scale using batched mapping
        processed_ds = ds_dict.map(
            self.preprocess_callback,
            batched=True,
            remove_columns=ds_dict["train"].column_names
        )
        
        # Set format to torch to ensure compatibility with GPUs
        processed_ds.set_format(type="torch")
        return processed_ds