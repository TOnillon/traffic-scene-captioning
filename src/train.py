import torch
import numpy as np
import evaluate
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, EarlyStoppingCallback
from src.model import TrafficModel
from src.data_loader import TrafficDataLoader
from src.processor import TrafficProcessor

def train(epochs=3, dataset_name="lmms-lab/COCO-Caption"):
    """
    Orchestrates the fine-tuning process of the image captioning model.
    Handles environment setup, dataset preparation, and training execution.
    """
    # Initialize core components: Architecture, Processor, and Data Pipeline
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
        
        # Decode predictions and labels into human-readable strings
        decoded_preds = processor.decode(preds)
        
        # Masked label indices (-100) are replaced by pad tokens for proper decoding
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        decoded_labels = processor.decode(labels)

        # Format references for BLEU (expects a list of lists)
        references = [[l] for l in decoded_labels]

        # Compute standard NLP scores
        rouge_results = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        bleu_results = bleu_metric.compute(predictions=decoded_preds, references=references)
        meteor_results = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)

        # Extract scalar values from metric results
        return {
            "rougeL": round(rouge_results["rougeL"], 4),
            "bleu": round(bleu_results["bleu"], 4),
            "meteor": round(meteor_results["meteor"], 4)
        }

    # Training hyperparameters optimized for convergence and stability
    training_args = Seq2SeqTrainingArguments(
        output_dir="./models/traffic_model_v1",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        save_total_limit=1,
        logging_steps=10,
        num_train_epochs=epochs,      
        learning_rate=2e-5,     
        weight_decay=0.05,       
        fp16=torch.cuda.is_available(), # Mixed precision for faster training
        report_to="tensorboard",
        logging_dir="./models/traffic_model_v1/runs", # Updated path for TensorBoard
    )

    trainer = Seq2SeqTrainer(
        model=traffic_model.model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"], 
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Start training execution
    trainer.train()
    
    # Save the final optimized weights for inference/deployment
    trainer.save_model("./models/traffic_model_v1_final")
    
    return trainer

if __name__ == "__main__":
    train()