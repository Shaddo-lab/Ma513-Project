# Imports
import json
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report
import os
import torch

# Check if GPU is available and set device
if torch.cuda.is_available():
    print("GPU is available!")
    device = torch.device("cuda")
else:
    print("GPU not found. Using CPU.")
    device = torch.device("cpu")

# Suppress symlink warning for cache systems
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# Utility Functions
# ======================

# Load Dataset
def load_data(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


# Prepare Dataset
def prepare_dataset(data, tokenizer, label2id, is_test=False):
    tokenized_inputs = tokenizer([item["tokens"] for item in data], is_split_into_words=True, padding=True, truncation=True)
    labels = []
    for i, item in enumerate(data):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        
        # Handle cases where 'ner_tags' might be missing (for test data)
        if is_test:
            sample_labels = [-100 if word_id is None else 0 for word_id in word_ids]  # No labels in test set
        else:
            sample_labels = [
                -100 if word_id is None else label2id.get(item["ner_tags"][word_id], 0) 
                for word_id in word_ids
            ]
        
        labels.append(sample_labels)
    
    tokenized_inputs["labels"] = labels
    return Dataset.from_dict(tokenized_inputs)


# Map Labels
def create_label_mappings(data):
    all_tags = [tag for item in data for tag in item["ner_tags"]]
    unique_tags = sorted(set(all_tags))
    label2id = {tag: idx for idx, tag in enumerate(unique_tags)}
    id2label = {idx: tag for tag, idx in label2id.items()}
    return label2id, id2label


# Save Predictions
def save_predictions(data, predictions, id2label, tokenizer, output_file):
    with open(output_file, 'w') as f:
        for item, prediction in zip(data, predictions):
            # Map predictions from IDs back to their corresponding labels
            word_ids = tokenizer.convert_tokens_to_ids(item["tokens"])
            predicted_labels = []
            label_idx = 0

            for word_id in word_ids:
                # If it's a sub-token, assign the same label to it
                if word_id != -100:  # Only include tokens that are not padding
                    pred_label = id2label.get(prediction[label_idx], "O")
                    predicted_labels.append(pred_label)
                    label_idx += 1
                else:
                    # If word_id is None, it means it's padding, so we skip
                    predicted_labels.append("O")

            # Save to output file
            f.write(json.dumps({
                "unique_id": item["unique_id"],
                "tokens": item["tokens"],
                "ner_tags": predicted_labels
            }) + "\n")


# Visualize Tag Distribution
def visualize_tag_distribution(data):
    tag_counts = Counter(tag for item in data for tag in item["ner_tags"])
    plt.bar(tag_counts.keys(), tag_counts.values())
    plt.xticks(rotation=45)
    plt.title("Tag Distribution")
    plt.show()


# MAIN SCRIPT
# ====================== Workflow =========================

if __name__ == "__main__":
    # Paths
    train_path = "NER-TRAINING.jsonlines"
    val_path = "NER-VALIDATION.jsonlines"
    test_path = "NER-TESTING.jsonlines"
    predictions_path = "NER-TESTING-PREDICTIONS.jsonlines"

    # Load Data
    train_data = load_data(train_path)
    val_data = load_data(val_path)
    test_data = load_data(test_path)
    
    # Visualize Data
    visualize_tag_distribution(train_data)
    
    # Label Mappings
    label2id, id2label = create_label_mappings(train_data)
    num_classes = len(label2id)
    
    # Load Pretrained Model and Tokenizer
    model_name = "jackaduma/SecBERT" #markusbayer/CySecBERT #bert-base-casedbert-large-uncased
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=num_classes, id2label=id2label, label2id=label2id
    )

    # Move model to the GPU (if available)
    model.to(device)

    # Prepare Dataset
    train_dataset = prepare_dataset(train_data, tokenizer, label2id)
    val_dataset = prepare_dataset(val_data, tokenizer, label2id)
    dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

    # Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=1e-5, 
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16, 
        num_train_epochs=30, 
        weight_decay=1, 
        warmup_steps=500, 
    )
    
    # Trainer
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=dataset["train"], 
        eval_dataset=dataset["validation"], 
        tokenizer=tokenizer, 
        data_collator=data_collator, 
    )

    # Train Model
    print("Starting the training process...") 
    trainer.train() 
    
    # Generate Predictions for Validation Data 
    predictions, labels, _ = trainer.predict(dataset["validation"]) 
    predictions = np.argmax(predictions, axis=-1) 
    
    # Map true labels and predictions for evaluation 
    true_labels = [ 
        [id2label[label] for label in example if label != -100]  
        for example in labels 
    ] 
    pred_labels = [ 
        [id2label[pred] for pred, label in zip(example, labels[idx]) if label != -100] 
        for idx, example in enumerate(predictions) 
    ] 
    
    print("Evaluation Metrics:") 
    print(classification_report(true_labels, pred_labels)) 

    # Generate Predictions for Validation Data 
    val_predictions = trainer.predict(val_dataset) 
    save_predictions(val_data, np.argmax(val_predictions.predictions, axis=-1), id2label, tokenizer, predictions_path) 

    print(f"Predictions saved to: {predictions_path}")