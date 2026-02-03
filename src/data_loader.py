from datasets import load_dataset, Dataset, DatasetDict
from PIL import Image
import torch
from src.processor import TrafficProcessor

class TrafficDataLoader:
    def __init__(self, dataset_name="lmms-lab/COCO-Caption"):
        self.processor = TrafficProcessor(is_training=True)
        self.traffic_categories = ['car', 'truck', 'bus', 'motorcycle']
        
        if dataset_name == "mock":
            fake_img = Image.new('RGB', (224, 224), color='red')
            self.dataset = DatasetDict({
                "train": Dataset.from_dict({
                    "image": [fake_img] * 8, 
                    "answer": [["a red car on the road"]] * 8
                }),
                "validation": Dataset.from_dict({
                    "image": [fake_img] * 2, 
                    "answer": [["a red car on the road"]] * 2
                }),
                "test": Dataset.from_dict({
                    "image": [fake_img] * 2, 
                    "answer": [["a red car on the road"]] * 2
                })
            })
        else:
            raw_iterable = load_dataset(dataset_name, split="val", streaming=True)
            
            def gen():
                count = 0
                for example in raw_iterable:
                    if count >= 5000:
                        break
                    
                    text = " ".join(example['answer']).lower() if isinstance(example['answer'], list) else example['answer'].lower()
                    if any(cat in text for cat in self.traffic_categories):
                        yield example
                        count += 1

            self.dataset = Dataset.from_generator(gen)

    def preprocess_callback(self, examples):
        img_column = "image"
        captions = [a[0] if isinstance(a, list) else a for a in examples["answer"]]
        
        pixel_values = self.processor.preprocess(examples[img_column])
        tokens = self.processor.tokenize_text(captions)
        
        labels = tokens["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {"pixel_values": pixel_values, "labels": labels}

    def prepare_dataset(self):
        def filter_traffic(example):
            text = " ".join(example['answer']).lower() if isinstance(example['answer'], list) else example['answer'].lower()
            return any(category in text for category in self.traffic_categories)

        if isinstance(self.dataset, DatasetDict):
            self.dataset = self.dataset.filter(filter_traffic)
            ds_dict = self.dataset
        else:
            train_testval = self.dataset.train_test_split(test_size=0.3, seed=42)
            val_test = train_testval["test"].train_test_split(test_size=0.5, seed=42)
            
            ds_dict = DatasetDict({
                "train": train_testval["train"],
                "validation": val_test["train"],
                "test": val_test["test"]
            })

        processed_ds = ds_dict.map(
            self.preprocess_callback,
            batched=True,
            batch_size=16, 
            remove_columns=ds_dict["train"].column_names,
            keep_in_memory=False 
        )
        
        processed_ds.set_format(type="torch", columns=["pixel_values", "labels"])
        return processed_ds