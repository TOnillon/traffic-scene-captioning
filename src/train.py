import torch
import numpy as np
import evaluate
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, EarlyStoppingCallback
from src.model import TrafficModel
from src.data_loader import TrafficDataLoader
from src.processor import TrafficProcessor

def train(epochs=3, dataset_name="DamianBoborzi/car_images"):
    """
    Orchestrates the fine-tuning process of the image captioning model.
    Configures environment, metrics, and training hyperparameters.
    """
    # Set compute device and load base NLP metrics for evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize core components: Architecture, Processor and Data Pipeline
    traffic_model = TrafficModel()
    processor = TrafficProcessor()
    
    loader = TrafficDataLoader(dataset_name) 
    dataset_dict = loader.prepare_dataset()

    # Load linguistic metrics to assess the quality of generated descriptions
    rouge_metric = evaluate.load("rouge")
    bleu_metric = evaluate.load("bleu")
    meteor_metric = evaluate.load("meteor")

    def compute_metrics(eval_preds):
        """
        Calculates linguistic scores during the evaluation phase.
        Translates numerical tensors back to text for comparison.
        """
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Convert IDs back to strings, ignoring special tokens
        decoded_preds = processor.decode(preds)
        
        # Replace masked label indices (-100) with pad token for decoding
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        decoded_labels = processor.decode(labels)

        # Reformat references for BLEU metric (expects list of lists)
        references = [[l] for l in decoded_labels]

        # Compute standard NLP scores
        rouge_results = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        bleu_results = bleu_metric.compute(predictions=decoded_preds, references=references)
        meteor_results = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)

        return {
            "rougeL": round(rouge_results["rougeL"], 4),
            "bleu": round(bleu_results["bleu"], 4),
            "meteor": round(meteor_results["meteor"], 4)
        }

    # Define training hyperparameters optimized for local/HPC environments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./models/traffic_model_v1",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        predict_with_generate=True, # Required for generating text during eval
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True, # Saves the most semantically accurate version
        metric_for_best_model="rougeL",
        logging_steps=10,
        num_train_epochs=epochs,
        learning_rate=5e-5, # Standard LR for fine-tuning transformers
        weight_decay=0.01,  # Regularization to prevent overfitting
        fp16=torch.cuda.is_available(), # Mixed precision training for GPU speed
        push_to_hub=False,
        report_to="none"
    )

    # Initialize the high-level Trainer API
    trainer = Seq2SeqTrainer(
        model=traffic_model.model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"], 
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        # Stop training if the model stops improving to save resources
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Start the training execution
    trainer.train()
    
    # Export the final optimized weights for deployment
    trainer.save_model("./models/traffic_model_v1_final")

if __name__ == "__main__":
    train()